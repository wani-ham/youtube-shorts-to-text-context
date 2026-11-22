"""Stage 1: download a public YouTube video."""

from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

from .config import Config
from .models import PipelineError

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
# Prefer H.264/AAC so that Stage 2 rarely has to re-encode.
FORMAT = "bv*[vcodec^=avc1][height<=1080]+ba[acodec^=mp4a]/b[ext=mp4]/bv*+ba/b"


def normalise_url(url: str) -> str:
    """Accept a link as pasted, and return it as an https YouTube URL."""
    url = url.strip()
    if "://" not in url:
        url = f"https://{url}"
    parsed = urlparse(url)
    # Comparing the whole netloc also rejects user-info and custom ports.
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in YOUTUBE_HOSTS:
        raise PipelineError("unsupported_url", "Only YouTube links are supported.")
    return parsed._replace(scheme="https").geturl()


def fetch(url: str, out_dir: Path, config: Config) -> tuple[Path, dict]:
    url = normalise_url(url)
    opts = {
        "outtmpl": str(out_dir / "video.%(ext)s"),
        "format": FORMAT,
        "merge_output_format": "mp4",
        "writeinfojson": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "max_filesize": config.max_filesize_bytes,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            preview = ydl.extract_info(url, download=False)
            if preview.get("entries") is not None:
                raise PipelineError("unsupported_url", "Playlists are not supported.")
            duration = preview.get("duration")
            if duration and duration > config.max_duration_s:
                raise PipelineError(
                    "too_long",
                    f"Video is {duration:.0f}s; the limit is {config.max_duration_s:.0f}s.",
                )
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise PipelineError(
            "fetch_failed", f"Download failed; upload the file manually. {exc}"
        ) from exc

    video = out_dir / "video.mp4"
    if not video.exists():
        raise PipelineError(
            "too_large",
            "No file was downloaded; it probably exceeds the size limit.",
        )
    info_json = out_dir / "video.info.json"
    if info_json.exists():
        info_json.rename(out_dir / "info.json")

    return video, {
        "url": url,
        "platform": "youtube",
        "uploader": info.get("uploader"),
        "post_caption": info.get("description") or info.get("title"),
        "upload_date": info.get("upload_date"),
        "is_short": "/shorts/" in urlparse(url).path,
    }
