"""Stage 4: audio extraction and speech-to-text."""

import re
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np

from .config import Config
from .models import PipelineError, Transcript, TranscriptSegment

_model = None
_model_lock = threading.Lock()


def extract_audio(video: Path, wav: Path, config: Config) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         str(wav)],
        capture_output=True, check=True, timeout=config.ffmpeg_timeout_s,
    )


def _read_wav(path: Path) -> np.ndarray:
    """Decode our 16 kHz mono PCM file, without faster-whisper's PyAV decoding."""
    with wave.open(str(path), "rb") as handle:
        pcm = handle.readframes(handle.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def get_model(config: Config):
    """Load Whisper once per process, falling back from GPU to CPU."""
    global _model
    with _model_lock:
        if _model is None:
            import torch
            from faster_whisper import WhisperModel

            if torch.cuda.is_available():
                try:
                    _model = WhisperModel(
                        config.whisper_model, device="cuda", compute_type="float16"
                    )
                except (RuntimeError, ValueError):
                    _model = None
            if _model is None:
                _model = WhisperModel(config.whisper_model, device="cpu", compute_type="int8")
    return _model


def _syllables(text: str) -> float:
    """Estimate syllables: one per Hangul, kana or CJK character, 2.5 letters otherwise."""
    dense = sum(
        "\uac00" <= c <= "\ud7a3" or "\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff"
        for c in text
    )
    other = sum(c.isalnum() for c in text) - dense
    return dense + other / 2.5


def split_impossible_speech(
    segments: list[TranscriptSegment], config: Config
) -> tuple[list[TranscriptSegment], list[TranscriptSegment]]:
    """Separate lines spoken impossibly fast, plus short fragments touching them."""
    removed = set()
    for index, segment in enumerate(segments):
        syllables = _syllables(segment.text)
        duration = max(segment.end - segment.start, 0.01)
        if syllables >= 4 and syllables / duration > config.max_syllables_per_s:
            removed.add(index)

    changed = bool(removed)
    while changed:
        changed = False
        for index, segment in enumerate(segments):
            if index in removed or segment.end - segment.start > config.burst_neighbour_max_s:
                continue
            touches_before = index - 1 in removed and (
                segment.start - segments[index - 1].end <= 0.25
            )
            touches_after = index + 1 in removed and (
                segments[index + 1].start - segment.end <= 0.25
            )
            if touches_before or touches_after:
                removed.add(index)
                changed = True
    return (
        [s for i, s in enumerate(segments) if i not in removed],
        [s for i, s in enumerate(segments) if i in removed],
    )


def transcription_flags(transcript: Transcript, config: Config) -> list[str]:
    flags = ["impossible_speech_removed"] if transcript.removed_segments else []
    if not transcript.segments:
        return flags
    if (transcript.language_probability or 0) < config.low_language_probability:
        flags.append("low_language_confidence")

    normalised = [re.sub(r"\W+", " ", s.text).strip().casefold() for s in transcript.segments]
    run = longest = 1
    for previous, current in zip(normalised, normalised[1:]):
        run = run + 1 if current and current == previous else 1
        longest = max(longest, run)
    if longest >= config.repeated_segments_flag:
        flags.append("possible_hallucination")
    return flags


def transcribe(wav: Path, config: Config) -> Transcript:
    model = get_model(config)
    samples = _read_wav(wav)
    budget = max(
        config.transcribe_timeout_s, len(samples) / 16000 * config.transcribe_s_per_audio_s
    )
    deadline = time.monotonic() + budget
    segments, info = model.transcribe(
        samples,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,  # Gives tighter segment boundaries over music and pauses.
        language=config.whisper_language,
    )
    audio_end = round(len(samples) / 16000, 2)
    output = []
    for segment in segments:  # Lazy: the decoding happens during this loop.
        if time.monotonic() > deadline:
            raise PipelineError("transcribe_timeout", "Transcription exceeded its time limit.")
        text = segment.text.strip()
        if text and segment.start < audio_end:
            output.append(
                TranscriptSegment(
                    start=round(segment.start, 2),
                    end=min(round(segment.end, 2), audio_end),
                    text=text,
                )
            )
    output, removed = split_impossible_speech(output, config)
    # Whisper always guesses a language, which means nothing without speech.
    return Transcript(
        language=info.language if output else None,
        language_probability=round(info.language_probability, 3) if output else None,
        model=f"faster-whisper {config.whisper_model}",
        segments=output,
        removed_segments=removed,
    )
