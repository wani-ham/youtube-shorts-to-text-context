# Generate Text Contexts from Youtube Shorts

<!-- mcp-name: io.github.wani-ham/video-to-context -->

Turn a short-form Youtube video into structured, timestamped context in a text format. It includes both audio and visual context.

Most language models can't watch a video. This tool gives them a faithful text version of one: what was **said**, what was **written on screen**, and what was **shown**, with every item labelled by source and time.

```text
[ON-SCREEN:LOGO 00:00–00:16] EXAMPLE CHANNEL
[AUDIO 00:07–00:11] Today we're testing three ways to brew coffee at home.
[ON-SCREEN 00:08–00:10] 3 WAYS TO BREW COFFEE
[VISUAL 00:07–00:16] A person in an apron pours hot water into a glass coffee maker in a bright kitchen.
```

Use the output as the input to summarisation, search, fact-checking, content moderation, RAG, or your own agents.

## Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Choosing a vision model](#choosing-a-vision-model)
- [Usage](#usage)
- [Use it from AI tools (MCP)](#use-it-from-ai-tools-mcp)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Privacy and responsible use](#privacy-and-responsible-use)
- [Development](#development)
- [License](#license)

## How it works

```text
YouTube link or video file
        │
   download and check
        │
        ├── audio ──► speech-to-text (Whisper, runs locally)
        │
        └── frames ─► vision model: on-screen text, scenes, settings, editing
        │
   record.json + record.md
```

## Quick start

### 1. Install the requirements

You need [uv](https://docs.astral.sh/uv/), **ffmpeg**, and about 5 GB of free disk space. uv installs the right Python version for you.

| System | Install command |
| --- | --- |
| macOS | `brew install uv ffmpeg` |
| Ubuntu or Debian | `sudo apt install ffmpeg && curl -LsSf https://astral.sh/uv/install.sh \| sh` |

### 2. Install video-to-context

```bash
uv tool install --python 3.12 video-to-context
```

Add uv's tools to your PATH, then **open a new terminal**:

```bash
uv tool update-shell
```

You can also use `pip install video-to-context` inside a Python 3.12 environment.

### 3. Choose a vision model (optional)

Download the settings template:

```bash
mkdir -p ~/.video-to-context
curl -o ~/.video-to-context/.env https://raw.githubusercontent.com/wani-ham/youtube-shorts-to-text-context/main/.env.example
```

Open `~/.video-to-context/.env` and uncomment one option from the next section. Without one, the tool still produces the transcript and reads on-screen text locally, but it won't include visual analysis.

### 4. Run it

```bash
video-to-context "https://www.youtube.com/shorts/VIDEO_ID"
```

The first run is slower because it downloads the speech and OCR models. When it finishes, it prints the path to your result in a `store/` folder inside your current directory.

## Choosing a vision model

You don't need Claude. Any service with an OpenAI-compatible API and image support will work.

| Option | Cost | Your frames go to | Best for |
| --- | --- | --- | --- |
| **Claude Haiku 5.5** | About 1 cent per video (estimate) | Anthropic | Tested default |
| **Gemini** | Free tier available | Google | Trying it at no cost |
| **OpenAI** | Paid | OpenAI | Existing OpenAI users |
| **OpenRouter** | Free and paid models | The selected provider | Comparing models |
| **Ollama** | Free | **Nowhere, runs locally** | Private text extraction, no account |
| **None** | Free | Nowhere | Transcript and text only |

> **Status:** Claude has been tested on real videos. Ollama with Gemma 4 12B has been tested on one real video: it read on-screen text well, but its scene descriptions were unreliable. Gemini, OpenAI, and OpenRouter use the same connection as Ollama but haven't been tested with their real services.

### Claude

Create a key in the [Claude Console](https://platform.claude.com/). API usage is billed separately from Claude Pro and Max plans.

```bash
ANTHROPIC_API_KEY=your-key
```

### Gemini

Create a key in [Google AI Studio](https://aistudio.google.com/). Free-tier content may be used to improve Google's products, so avoid private videos on that tier.

```bash
VISION_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
VISION_API_KEY=your-key
VISION_MODEL=gemini-3.8-flash
```

### OpenAI

```bash
VISION_BASE_URL=https://api.openai.com/v1
VISION_API_KEY=your-key
VISION_MODEL=gpt-5.4-mini
```

### OpenRouter

Choose a model that accepts images from [openrouter.ai/models](https://openrouter.ai/models).

```bash
VISION_BASE_URL=https://openrouter.ai/api/v1
VISION_API_KEY=your-key
VISION_MODEL=provider/model-name
```

### Ollama (free and local)

Install [Ollama](https://ollama.com/), then download a vision model. Allow about 8 GB of disk space and at least 16 GB of memory, and close other heavy apps.

```bash
ollama pull gemma4:12b
```

Start Ollama with a larger context. Its default is too small for this prompt and the images:

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

If Ollama is already running in the background, quit it first. Then add this to `.env`:

```bash
VISION_BASE_URL=http://localhost:11434/v1
VISION_MODEL=gemma4:12b
VISION_MAX_FRAMES=12
VISION_REASONING_EFFORT=none
VISION_TIMEOUT_S=600
```

- `VISION_MAX_FRAMES` limits how many images are sent per video. Fewer frames use less memory but can miss short captions.
- `VISION_REASONING_EFFORT=none` turns off model thinking. On a 16 GB Apple Silicon Mac, one request took about 78 seconds this way and about 250 seconds with thinking, which gave somewhat better scene descriptions.
- `VISION_TIMEOUT_S` gives slower local models enough time to answer.

Model names change often. If one is rejected, check your provider's current model list.

## Usage

```bash
# A YouTube video or Short
video-to-context "https://www.youtube.com/shorts/VIDEO_ID"

# A video file on your computer
video-to-context --file path/to/video.mp4

# A file with details about where it came from
video-to-context --file ad.mp4 --sponsor "Brand" --date 2026-11-02 --caption "Post text"

# Process a video again, ignoring the saved result
video-to-context "https://youtu.be/VIDEO_ID" --force
```

Videos can be up to 5 minutes and 200 MB. To change these limits, edit [config.py](https://github.com/wani-ham/youtube-shorts-to-text-context/blob/main/video_to_context/config.py) in a source install.

The command exits with `0` for a complete record, `1` for a partial record, and `2` for a rejected input.

### Use it from Python

```python
from dotenv import load_dotenv
from video_to_context.run import process

load_dotenv()
record = process(url="https://www.youtube.com/shorts/VIDEO_ID", out="store")

print(record.llm_document)
print(record.visual_context.summary if record.visual_context else "No visual analysis")
```

## Use it from AI tools (MCP)

The included [MCP](https://modelcontextprotocol.io/) server lets AI tools analyse a video when you ask, for example: *"Summarise this Short and tell me what its on-screen text claims."*

Finish the [Quick start](#quick-start) first. The server reads your settings from `~/.video-to-context/.env` and stores results in `~/.video-to-context/store`.

### Claude Code

```bash
claude mcp add --scope user video-to-text -- "$(command -v video-to-context-mcp)"
```

### Cursor, Claude Desktop, and other clients

Find the full command path with `command -v video-to-context-mcp`, then add it to the client's MCP configuration. For Cursor, that's `~/.cursor/mcp.json`.

```json
{
  "mcpServers": {
    "video-to-text": {
      "command": "/absolute/path/to/video-to-context-mcp"
    }
  }
}
```

### Tools

| Tool | What it does |
| --- | --- |
| `start_video_analysis` | Starts a background job for a YouTube link or allowed local file |
| `get_analysis_status` | Reports whether the job is queued, running, or finished |
| `get_video_text` | Returns the finished result as Markdown or JSON |
| `list_analyzed_videos` | Lists videos that already have results |
| `check_setup` | Checks ffmpeg, the vision model, and file access |

Videos run one at a time in the background, because processing takes longer than most tools wait for a single call.

### Security

- **Local files are off by default.** To allow them, list specific folders in `VIDEO_TEXT_ALLOWED_DIRS` in `~/.video-to-context/.env`.
- **URLs are limited to YouTube.**
- **The server runs locally over stdio** and opens no network port.
- **Video text is untrusted.** A caption could try to manipulate your AI tool, so be careful when that tool can also run commands or edit files.

## Output

Each video gets its own folder, named after a hash of the file, so the same file is never processed twice:

```text
store/<sha256>/
├── record.md     Document for LLMs and people
├── record.json   Full structured data for code
├── video.mp4     The video
├── frames/       Keyframes referenced by the record
└── info.json     YouTube metadata (for YouTube links)
```

### `record.md`

Pass this file directly to a language model. It contains:

- **YAML metadata:** source, duration, language, footage type, status, and models used.
- **Quality notes:** anything downstream services should be careful about.
- **Visual analysis:** a summary, the distinct settings, editing techniques, and shots.
- **Timeline:** everything in time order, labelled by source.

| Label | Meaning | Safe to quote as the video's content? |
| --- | --- | --- |
| `AUDIO` | Transcribed speech | Yes, but it may have recognition errors |
| `ON-SCREEN` | Captions and text in the frame | Yes |
| `ON-SCREEN:HEADLINE`, `:WATERMARK`, `:LOGO`, `:OTHER` | Text or marks that stay on screen | Yes |
| `VISUAL` | A model's description of a scene | **No, it is an interpretation** |

Video text is untrusted input. It's kept inside code blocks so that your prompt can treat it as data, but a caption could still contain text written to manipulate a model. Tell your downstream model not to follow instructions found in it.

### `record.json`

The same information as structured fields, including `transcript`, `on_screen_text`, `visual_context`, `visual_descriptions`, `keyframes`, `flags`, and `errors`. Suspected invented speech is moved to `transcript.removed_segments` so you can review it.

If one stage fails, the others still finish and the record is marked `partial`.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `fetch_failed` | The video may be private, removed, or age-restricted, or YouTube may be blocking the download. Download limits are stricter on cloud servers. Use `--file` instead. |
| `vision_not_configured` in `flags` | No vision model is set. Check that `~/.video-to-context/.env` exists and an option is uncommented. |
| `vision_auth_failed` | The API key is wrong or expired. |
| `vision_invalid_output` | The model didn't return usable JSON. Try a more capable model. |
| Out of memory with Ollama | Lower `VISION_MAX_FRAMES` or use a smaller model. |
| `command not found: video-to-context` | Run `uv tool update-shell`, then open a new terminal. |
| MCP tools don't appear | Use the full path from `command -v video-to-context-mcp`, then restart the client. |
| `vision_timeout` | Raise `VISION_TIMEOUT_S`, or set `VISION_REASONING_EFFORT=none`. |
| `objc: Class AVF... is implemented in both` | A harmless macOS warning from two video libraries. |
| `ffmpeg` not found | Install ffmpeg (step 1), then open a new terminal. |
| It's slow | Speech-to-text runs on CPU unless you have an NVIDIA GPU. Expect a few minutes for a 2-minute video. |

## Privacy and responsible use

- Unless you use Ollama or no vision model, video frames are sent to the provider you choose. Check its data retention and training terms.
- Videos contain people's faces and voices. The vision prompt tells the model not to identify people, and you shouldn't add face or voice identification without legal review.
- YouTube's terms restrict downloading. Make sure you have the right to process a video, especially for commercial or public use.
- `.env`, `store/`, and media files are ignored by git. Never commit API keys.

## Development

```bash
git clone https://github.com/wani-ham/youtube-shorts-to-text-context.git
cd youtube-shorts-to-text-context
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
.venv/bin/python -m pytest
.venv/bin/python -m video_to_context.evaluate
```

In a source checkout, settings come from the project's `.env`, and the MCP server stores results in the project's `store/` folder.

To score accuracy on your own test set, put each video in `tests/samples/<name>.mp4` and its hand-written labels in `tests/expected/<name>.json`:

```json
{"transcript": "…", "on_screen": ["…"], "no_speech": false, "rejected": null}
```

## License

Licensed under the [Apache License 2.0](https://github.com/wani-ham/youtube-shorts-to-text-context/blob/main/LICENSE). See [NOTICE](https://github.com/wani-ham/youtube-shorts-to-text-context/blob/main/NOTICE) for copyright details.
