"""Stage 5: scene detection, keyframes, OCR and on-screen text merging."""

import bisect
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from rapidfuzz import fuzz

from .config import Config
from .models import Keyframe, OnScreenText, PipelineError

_readers: dict[tuple[str, ...], object] = {}
# EasyOCR can only combine these scripts with English, so each gets its own reader.
OCR_LANGUAGES = {"ko": ("ko", "en"), "ja": ("ja", "en"), "zh": ("ch_sim", "en")}
_reader_lock = threading.Lock()


@dataclass
class Sample:
    """A kept keyframe, and the last time its image was still on screen."""

    time: float
    last_seen: float
    path: Path
    phash: imagehash.ImageHash
    pixels: np.ndarray
    scene: int = 0


def scene_starts(video: Path, config: Config) -> list[float]:
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video), ContentDetector(threshold=config.scene_threshold))
    return [start.get_seconds() for start, _ in scenes] or [0.0]


def sample_times(duration: float, scenes: list[float], cap: int) -> list[float]:
    """One frame per scene start and per second, evenly thinned to the cap."""
    starts = sorted({round(t, 3) for t in scenes if 0 <= t < duration})
    ticks = [float(s) for s in range(int(np.ceil(duration)))]
    ticks = [t for t in ticks if all(abs(t - s) >= 0.2 for s in starts)]
    times = sorted(starts + ticks) or [0.0]
    if len(times) > cap:
        indexes = np.linspace(0, len(times) - 1, cap).round().astype(int)
        times = [times[i] for i in indexes]
    return times


def _extract_frame(video: Path, at: float, dest: Path, config: Config) -> Path | None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1",
         "-q:v", "2", "-update", "1", str(dest)],
        capture_output=True, timeout=config.ffmpeg_timeout_s,
    )
    return dest if result.returncode == 0 and dest.exists() else None


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L").resize((270, 480)), dtype=np.int16)


def extract_keyframes(video: Path, work_dir: Path, duration: float, config: Config) -> list[Sample]:
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    scenes = sorted(scene_starts(video, config))
    times = sample_times(duration, scenes, config.max_keyframes)
    paths = [frames_dir / f"{i:04d}.jpg" for i in range(len(times))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        extracted = list(
            pool.map(lambda item: _extract_frame(video, item[0], item[1], config),
                     zip(times, paths))
        )

    kept: list[Sample] = []
    for at, path in zip(times, extracted):
        if path is None:
            continue
        with Image.open(path) as image:
            phash = imagehash.phash(image)
            pixels = _gray(image)
        same_scene = kept and kept[-1].scene == bisect.bisect_right(scenes, at + 0.001)
        if same_scene and phash - kept[-1].phash <= config.phash_max_distance:
            changed = float(np.mean(np.abs(pixels - kept[-1].pixels) > 32))
            if changed <= config.max_changed_pixels:
                kept[-1].last_seen = at
                path.unlink()
                continue
        kept.append(Sample(at, at, path, phash, pixels, bisect.bisect_right(scenes, at + 0.001)))
    if not kept:
        raise PipelineError("no_frames", "No frame could be extracted.")
    split_missed_cuts(kept, config)
    return kept


def split_missed_cuts(samples: list[Sample], config: Config) -> None:
    """Renumber scenes, adding cuts visible between keyframes."""
    if len(samples) < 2:
        return
    stack = np.stack([s.pixels for s in samples])
    moving = stack.std(axis=0) >= config.static_pixel_std
    if moving.mean() < 0.05:  # Almost nothing moves, so use the whole frame.
        moving = np.ones_like(moving, dtype=bool)

    def histogram(pixels: np.ndarray) -> np.ndarray:
        counts = np.histogram(pixels[moving], bins=32, range=(0, 255))[0]
        return counts / max(counts.sum(), 1)

    previous = samples[0]
    previous_detected = previous.scene
    previous_histogram = histogram(previous.pixels)
    scene = 0
    previous.scene = 0
    for sample in samples[1:]:
        current_histogram = histogram(sample.pixels)
        histogram_change = float(np.abs(current_histogram - previous_histogram).sum() / 2)
        pixel_change = float(np.mean(np.abs(sample.pixels[moving] - previous.pixels[moving])))
        detected = sample.scene
        if (
            detected != previous_detected
            or histogram_change >= config.cut_histogram_change
            or pixel_change >= config.cut_pixel_change
        ):
            scene += 1
        sample.scene = scene
        previous, previous_detected, previous_histogram = sample, detected, current_histogram


def to_keyframes(samples: list[Sample], work_dir: Path) -> list[Keyframe]:
    return [
        Keyframe(time=s.time, path=s.path.relative_to(work_dir).as_posix(), phash=str(s.phash))
        for s in samples
    ]


def choose_ocr_languages(language: str | None, caption: str | None, config: Config) -> tuple:
    """Use the spoken language, or else the script of the post caption."""
    if language in OCR_LANGUAGES:
        return OCR_LANGUAGES[language]
    if language is None and caption:
        if any("\uac00" <= c <= "\ud7a3" for c in caption):
            return OCR_LANGUAGES["ko"]
        if any("\u3040" <= c <= "\u30ff" for c in caption):
            return OCR_LANGUAGES["ja"]
        if any("\u4e00" <= c <= "\u9fff" for c in caption):
            return OCR_LANGUAGES["zh"]
    return config.ocr_languages


def get_reader(config: Config):
    languages = tuple(config.ocr_languages)
    with _reader_lock:
        if languages not in _readers:
            import easyocr
            import torch

            _readers[languages] = easyocr.Reader(
                list(languages), gpu=torch.cuda.is_available(), verbose=False
            )
        return _readers[languages]


def split_blocks(boxes: list[dict]) -> list[list[dict]]:
    """Separate text areas, such as a fixed headline and a changing subtitle."""
    if not boxes:
        return []
    def top(box): return min(p[1] for p in box["bbox"])
    def bottom(box): return max(p[1] for p in box["bbox"])
    height = statistics.median(bottom(b) - top(b) for b in boxes)
    blocks: list[list[dict]] = []
    for box in sorted(boxes, key=top):
        if blocks and top(box) - max(bottom(b) for b in blocks[-1]) <= 0.8 * height:
            blocks[-1].append(box)
        else:
            blocks.append([box])
    return blocks


def join_boxes(boxes: list[dict]) -> str:
    """Join OCR boxes top-to-bottom, then left-to-right within a line."""
    if not boxes:
        return ""
    items = []
    for box in boxes:
        ys = [p[1] for p in box["bbox"]]
        xs = [p[0] for p in box["bbox"]]
        items.append(((min(ys) + max(ys)) / 2, min(xs), max(ys) - min(ys), box["text"]))
    tolerance = 0.6 * statistics.median(item[2] for item in items)

    lines: list[list[tuple]] = []
    for item in sorted(items, key=lambda i: i[0]):
        if lines and abs(item[0] - statistics.mean(i[0] for i in lines[-1])) <= tolerance:
            lines[-1].append(item)
        else:
            lines.append([item])
    ordered = [i[3].strip() for line in lines for i in sorted(line, key=lambda i: i[1])]
    return " ".join(t for t in ordered if t)


def ocr_frame(path: Path, config: Config) -> dict:
    results = get_reader(config).readtext(str(path))
    boxes = [
        {"text": text, "confidence": float(conf), "bbox": [[float(x), float(y)] for x, y in bbox]}
        for bbox, text, conf in results
        if conf >= config.ocr_min_confidence
    ]
    return {
        "blocks": [
            {
                "text": join_boxes(block),
                "confidence": round(statistics.mean(b["confidence"] for b in block), 3),
            }
            for block in split_blocks(boxes)
        ]
    }


def merge_ocr(frames: list[dict], config: Config) -> list[OnScreenText]:
    """Collapse each text repeated across nearby frames into one time range."""
    merged: list[OnScreenText] = []
    for frame in sorted(frames, key=lambda f: f["time"]):
        if not frame["text"]:
            continue
        end = frame.get("last_seen", frame["time"])
        for item in reversed(merged):
            if frame["time"] - item.end > config.ocr_merge_max_gap_s:
                continue
            if fuzz.ratio(item.text.casefold(), frame["text"].casefold()) >= config.ocr_merge_ratio:
                item.end = max(item.end, end)
                if (frame["confidence"] or 0) > (item.confidence or 0):
                    item.text, item.confidence = frame["text"], frame["confidence"]
                    item.frame = frame["path"]
                break
        else:
            merged.append(
                OnScreenText(
                    start=frame["time"], end=end, text=frame["text"],
                    confidence=frame["confidence"], frame=frame["path"],
                )
            )
    return merged


def run_ocr(samples: list[Sample], work_dir: Path, config: Config) -> list[OnScreenText]:
    budget = max(config.ocr_timeout_s, len(samples) * config.ocr_s_per_frame)
    deadline = time.monotonic() + budget
    frames = []
    for sample in samples:
        if time.monotonic() > deadline:
            raise PipelineError("ocr_timeout", "OCR exceeded its time limit.")
        for block in ocr_frame(sample.path, config)["blocks"]:
            frames.append(
                {
                    **block,
                    "time": sample.time,
                    "last_seen": sample.last_seen,
                    "path": sample.path.relative_to(work_dir).as_posix(),
                }
            )
    return merge_ocr(frames, config)
