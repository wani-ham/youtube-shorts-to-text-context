"""Run every stage for one input. CLI entry point."""

import argparse
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from . import __version__, audio, vision, visual
from .config import DEFAULT_CONFIG, Config
from .fetch import fetch
from .hashing import sha256_file
from .merge import build_llm_document, write_markdown, write_record
from .models import (
    STAGES, Media, PipelineError, PipelineInfo, RecordError, Source, TextRecord,
)
from .probe import normalise, probe


def _run_stage(record: TextRecord, name: str, action: Callable[[], None]) -> bool:
    """Run a stage; a failure is recorded and never stops the other branch."""
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - isolation is the point of this wrapper
        code = exc.code if isinstance(exc, PipelineError) else f"{name}_failed"
        record.pipeline.stages[name] = "failed"
        record.errors.append(RecordError(stage=name, code=code, message=str(exc)[:500]))
        return False
    record.pipeline.stages[name] = "ok"
    return True


def _audio_branch(record: TextRecord, video: Path, work: Path, config: Config) -> None:
    if not record.media.has_audio:
        record.pipeline.stages["audio"] = "skipped"
        record.pipeline.stages["transcribe"] = "skipped"
        return
    wav = work / "audio.wav"
    if not _run_stage(record, "audio", lambda: audio.extract_audio(video, wav, config)):
        record.pipeline.stages["transcribe"] = "skipped"
        return

    def transcribe() -> None:
        record.transcript = audio.transcribe(wav, config)
        record.flags.extend(audio.transcription_flags(record.transcript, config))

    _run_stage(record, "transcribe", transcribe)
    wav.unlink(missing_ok=True)


def _frames_branch(
    record: TextRecord, video: Path, work: Path, config: Config
) -> list[visual.Sample]:
    samples: list[visual.Sample] = []

    def frames() -> None:
        samples.extend(visual.extract_keyframes(video, work, record.media.duration_s, config))
        record.keyframes = visual.to_keyframes(samples, work)

    if not _run_stage(record, "frames", frames):
        record.pipeline.stages["ocr"] = "skipped"
        record.pipeline.stages["vision"] = "skipped"
    return samples


def _ocr_stage(
    record: TextRecord, samples: list[visual.Sample], work: Path, config: Config
) -> None:
    """Runs after transcription, because the spoken language selects the OCR model."""
    if not samples:
        return
    languages = visual.choose_ocr_languages(
        record.transcript.language, record.source.post_caption, config
    )

    def ocr() -> None:
        record.on_screen_text = visual.run_ocr(
            samples, work, replace(config, ocr_languages=languages)
        )
        record.on_screen_source = "easyocr"

    _run_stage(record, "ocr", ocr)


def _vision_stage(
    record: TextRecord, samples: list[visual.Sample], work: Path, config: Config
) -> None:
    if not samples:
        return
    if not vision.is_configured():
        record.pipeline.stages["vision"] = "skipped"
        record.flags.append("vision_not_configured")
        return

    def describe() -> None:
        result = vision.analyze_frames(samples, work, config)
        record.visual_context = result.context
        record.visual_descriptions = result.descriptions
        record.on_screen_text = result.on_screen
        record.on_screen_source = result.model or config.vision_model
        record.visual_model = result.model or config.vision_model

    _run_stage(record, "vision", describe)


def _load_complete(path: Path) -> TextRecord | None:
    """Return a stored record, unless it is partial and deserves a retry."""
    if not path.exists():
        return None
    try:
        record = TextRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if "vision_not_configured" in record.flags and vision.is_configured():
        return None  # A key was added since, so the scenes can now be described.
    return record if record.pipeline.status == "complete" else None


def process(
    url: str | None = None,
    file: str | Path | None = None,
    out: str | Path = "store",
    config: Config = DEFAULT_CONFIG,
    source_fields: dict | None = None,
    force: bool = False,
) -> TextRecord:
    if bool(url) == bool(file):
        raise PipelineError("bad_input", "Provide exactly one of a URL or a file.")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with tempfile.TemporaryDirectory(dir=out_dir, prefix=".tmp-") as temporary:
        temp_dir = Path(temporary)
        if url:
            original, fields = fetch(url, temp_dir, config)
        else:
            original = Path(file)
            if not original.is_file():
                raise PipelineError("file_not_found", f"No such file: {original}")
            fields = {"platform": "manual"}
        fields.update({k: v for k, v in (source_fields or {}).items() if v is not None})

        info = probe(original, config)
        media_id = sha256_file(original)  # Hash the source: re-encoding is not reproducible.
        store_dir = out_dir / media_id.removeprefix("sha256:")
        record_path = store_dir / "record.json"
        if not force and (existing := _load_complete(record_path)):
            if not (store_dir / "record.md").exists():
                write_markdown(existing, store_dir / "record.md")
            return existing

        build_dir = temp_dir / "build"
        build_dir.mkdir()
        video = build_dir / "video.mp4"
        if info.needs_reencode:
            try:
                normalise(original, video, config)
            except Exception as exc:
                raise PipelineError("normalise_failed", f"Could not re-encode: {exc}") from exc
        else:
            shutil.copy2(original, video)
        if (temp_dir / "info.json").exists():
            shutil.move(temp_dir / "info.json", build_dir / "info.json")

        stages = {name: "pending" for name in STAGES}
        stages["fetch"] = "ok" if url else "skipped"
        stages["probe"] = "ok"
        record = TextRecord(
            media_id=media_id,
            source=Source(fetched_at=fetched_at, **fields),
            media=Media(
                path=(store_dir / "video.mp4").as_posix(),
                duration_s=info.duration_s,
                width=info.width,
                height=info.height,
                has_audio=info.has_audio,
            ),
            pipeline=PipelineInfo(version=__version__, stages=stages),
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            audio_branch = pool.submit(_audio_branch, record, video, build_dir, config)
            frames_branch = pool.submit(_frames_branch, record, video, build_dir, config)
            audio_branch.result()
            samples = frames_branch.result()
        _vision_stage(record, samples, build_dir, config)
        if record.pipeline.stages["vision"] == "ok":
            record.pipeline.stages["ocr"] = "skipped"  # Claude already read the text.
        else:
            _ocr_stage(record, samples, build_dir, config)  # Local fallback.

        def merge() -> None:
            record.llm_document = build_llm_document(record)

        _run_stage(record, "merge", merge)
        record.errors.sort(key=lambda e: STAGES.index(e.stage))
        if any(status == "failed" for status in record.pipeline.stages.values()):
            record.pipeline.status = "partial"

        write_record(record, build_dir / "record.json")
        write_markdown(record, build_dir / "record.md")
        if store_dir.exists():
            shutil.rmtree(store_dir)
        shutil.move(str(build_dir), store_dir)
    return record


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env")  # Real environment variables take priority.
    parser = argparse.ArgumentParser(description="Turn a short video into a text record.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("link", nargs="?", help="public YouTube video or Short")
    group.add_argument("--url", help="same as giving the link directly")
    group.add_argument("--file", help="local media file")
    parser.add_argument("--out", default="store", help="output directory")
    parser.add_argument("--force", action="store_true", help="reprocess a known file")
    parser.add_argument("--ad-id")
    parser.add_argument("--sponsor")
    parser.add_argument("--date", dest="upload_date", help="publication date, YYYY-MM-DD")
    parser.add_argument("--caption", dest="post_caption")
    parser.add_argument("--platform", help="source platform of a manual file")
    args = parser.parse_args(argv)

    fields = {
        "ad_id": args.ad_id, "sponsor": args.sponsor, "upload_date": args.upload_date,
        "post_caption": args.post_caption, "platform": args.platform,
    }
    try:
        record = process(
            args.link or args.url, args.file, args.out, source_fields=fields, force=args.force
        )
    except PipelineError as exc:
        print(json.dumps({"error": exc.code, "message": exc.message}), file=sys.stderr)
        return 2
    print(Path(record.media.path).parent / "record.json")
    return 0 if record.pipeline.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
