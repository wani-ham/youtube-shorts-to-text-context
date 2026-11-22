"""Stage 2: inspect, validate and normalise a media file."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .models import PipelineError


@dataclass(frozen=True)
class ProbeResult:
    duration_s: float
    width: int
    height: int
    has_audio: bool
    needs_reencode: bool


def _rotation(stream: dict) -> int:
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            return abs(int(side_data["rotation"])) % 180
    return 0


def probe(path: Path, config: Config) -> ProbeResult:
    size = path.stat().st_size
    if size > config.max_filesize_bytes:
        raise PipelineError(
            "too_large",
            f"File is {size / 1024 / 1024:.0f} MB; "
            f"the limit is {config.max_filesize_bytes / 1024 / 1024:.0f} MB.",
        )
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json",
             "--", str(path)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        data = json.loads(result.stdout)
        streams = data["streams"]
        video = next(s for s in streams if s.get("codec_type") == "video")
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        duration = float(data["format"]["duration"])
        width, height = int(video["width"]), int(video["height"])
    except (subprocess.SubprocessError, KeyError, ValueError, StopIteration) as exc:
        raise PipelineError("probe_failed", f"Not a readable video: {path.name}") from exc

    if duration > config.max_duration_s:
        raise PipelineError(
            "too_long",
            f"Video is {duration:.0f}s; the limit is {config.max_duration_s:.0f}s.",
        )
    if _rotation(video) == 90:
        width, height = height, width

    is_mp4 = "mp4" in data["format"].get("format_name", "").split(",")
    compatible = (
        is_mp4
        and video.get("codec_name") == "h264"
        and video.get("pix_fmt") == "yuv420p"
        and (audio is None or audio.get("codec_name") == "aac")
    )
    return ProbeResult(round(duration, 3), width, height, audio is not None, not compatible)


def normalise(source: Path, dest: Path, config: Config) -> None:
    """Re-encode to H.264/AAC MP4 so that later stages get uniform input."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:a", "aac", "-movflags", "+faststart",
         str(dest)],
        capture_output=True, check=True, timeout=config.ffmpeg_timeout_s,
    )
