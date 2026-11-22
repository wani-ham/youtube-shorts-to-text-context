"""Limits, model names and thresholds."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    max_duration_s: float = 300.0
    max_filesize_bytes: int = 200 * 1024 * 1024

    whisper_model: str = "large-v3-turbo"
    whisper_language: str | None = None  # None = auto-detect
    low_language_probability: float = 0.6
    repeated_segments_flag: int = 3
    # Fast speech is about 8-9 syllables per second; beyond this it cannot be real.
    max_syllables_per_s: float = 14.0
    burst_neighbour_max_s: float = 0.6

    scene_threshold: float = 27.0
    max_keyframes: int = 90
    phash_max_distance: int = 4
    # A caption can change while the pHash barely moves, so a frame is only
    # dropped if this share of its pixels is also unchanged.
    max_changed_pixels: float = 0.002
    # Catch cuts the scene detector misses, ignoring fixed templates around the video.
    static_pixel_std: float = 6.0
    cut_histogram_change: float = 0.25
    cut_pixel_change: float = 45.0

    ocr_languages: tuple[str, ...] = ("fr", "en")
    ocr_min_confidence: float = 0.5
    ocr_merge_ratio: float = 90.0
    ocr_merge_max_gap_s: float = 2.0

    vision_model: str = "claude-haiku-5-5"
    vision_max_scenes: int = 60
    vision_max_image_side: int = 768
    vision_timeout_s: float = 180.0

    ffmpeg_timeout_s: int = 120
    # Minimum budgets, raised for longer inputs so that CPU runs can finish.
    transcribe_timeout_s: int = 120
    transcribe_s_per_audio_s: float = 5.0
    ocr_timeout_s: int = 120
    ocr_s_per_frame: float = 8.0


DEFAULT_CONFIG = Config()
