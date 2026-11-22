"""Pydantic models for the text record."""

from typing import Literal

from pydantic import BaseModel, Field

StageStatus = Literal["pending", "ok", "skipped", "failed"]
STAGES = ("fetch", "probe", "audio", "transcribe", "frames", "ocr", "vision", "merge")


class PipelineError(Exception):
    """A failure that stops the whole run, such as a rejected input."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class Source(BaseModel):
    url: str | None = None
    platform: str = "manual"
    uploader: str | None = None
    post_caption: str | None = None
    upload_date: str | None = None
    fetched_at: str
    ad_id: str | None = None
    sponsor: str | None = None
    is_short: bool = False


class Media(BaseModel):
    path: str
    duration_s: float
    width: int
    height: int
    has_audio: bool


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    language: str | None = None
    language_probability: float | None = None
    model: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    # Suspected invented speech, kept here for review instead of being deleted.
    removed_segments: list[TranscriptSegment] = Field(default_factory=list)


class OnScreenText(BaseModel):
    start: float
    end: float
    text: str
    confidence: float | None = None  # EasyOCR only; Claude readings have no score.
    frame: str
    kind: Literal["caption", "headline", "watermark", "logo", "other"] = "caption"


class VisualDescription(BaseModel):
    """A model's interpretation of a scene; never a quote from the video."""

    start: float
    end: float
    text: str
    frame: str


class TimeRange(BaseModel):
    start: float
    end: float


class VisualSetting(BaseModel):
    """A distinct place or look, with every time range where it appears."""

    description: str
    ranges: list[TimeRange]


class VisualContext(BaseModel):
    """A model's judgement of the whole video, from visual evidence only."""

    footage_type: str
    basis: str
    summary: str | None = None
    source_footage: str | None = None
    settings: list[VisualSetting] = Field(default_factory=list)
    editing_techniques: list[str] = Field(default_factory=list)
    editing_notes: str | None = None


class Keyframe(BaseModel):
    time: float
    path: str
    phash: str


class PipelineInfo(BaseModel):
    version: str
    status: Literal["complete", "partial"] = "complete"
    stages: dict[str, StageStatus]


class RecordError(BaseModel):
    stage: str
    code: str
    message: str


class TextRecord(BaseModel):
    media_id: str
    source: Source
    media: Media
    transcript: Transcript = Field(default_factory=Transcript)
    on_screen_text: list[OnScreenText] = Field(default_factory=list)
    visual_descriptions: list[VisualDescription] = Field(default_factory=list)
    visual_model: str | None = None
    visual_context: VisualContext | None = None
    on_screen_source: str | None = None
    keyframes: list[Keyframe] = Field(default_factory=list)
    llm_document: str = ""
    pipeline: PipelineInfo
    flags: list[str] = Field(default_factory=list)
    errors: list[RecordError] = Field(default_factory=list)
