# Generate Text Contexts from Youtube Shorts

Turn a short-form Youtube video into structured, timestamped context in a text format. It includes both audio and visual context.

Most language models can't watch a video. This tool gives them a faithful text version of one: what was **said**, what was **written on screen**, and what was **shown**, with every item labelled by source and time.

```text
[ON-SCREEN:LOGO 00:00–00:16] EXAMPLE CHANNEL
[AUDIO 00:07–00:11] Today we're testing three ways to brew coffee at home.
[ON-SCREEN 00:08–00:10] 3 WAYS TO BREW COFFEE
[VISUAL 00:07–00:16] A person in an apron pours hot water into a glass coffee maker in a bright kitchen.
```

Use the output as the input to summarisation, search, fact-checking, content moderation, RAG, or your own agents.

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

You need **Python 3.12**, **ffmpeg**, and about 5 GB of free disk space.

| System | Install command |
| --- | --- |
| macOS | `brew install python@3.12 ffmpeg` |
| Ubuntu or Debian | `sudo apt install python3.12 python3.12-venv ffmpeg` |
| Windows | `winget install Python.Python.3.12 Gyan.FFmpeg` |

### 2. Download and set up the project

```bash
git clone https://github.com/wani-ham/youtube-shorts-to-text-context.git
cd youtube-shorts-to-text-context
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

On Windows, use `py -3.12 -m venv .venv`, then `.venv\Scripts\pip install -r requirements.txt` and `copy .env.example .env`. Replace `.venv/bin/python` with `.venv\Scripts\python` in the commands below.

### 3. Choose a vision model (optional)

Open `.env` and uncomment one option from the next section. Without one, the tool still produces the transcript and reads on-screen text locally, but it won't include visual analysis.

### 4. Run it

```bash
.venv/bin/python -m pipeline.run "https://www.youtube.com/shorts/VIDEO_ID"
```

The first run is slower because it downloads the speech and OCR models. When it finishes, it prints the path to your result.

## Choosing a vision model

You don't need Claude. Any service with an OpenAI-compatible API and image support will work.

| Option | Cost | Your frames go to | Best for |
| --- | --- | --- | --- |
| **Claude Haiku 5.5** | About 1 cent per video (estimate) | Anthropic | Tested default |
| **Gemini** | Free tier available | Google | Trying it at no cost |
| **OpenAI** | Paid | OpenAI | Existing OpenAI users |
| **OpenRouter** | Free and paid models | The selected provider | Comparing models |
| **Ollama** | Free | **Nowhere, runs locally** | Privacy, no account |
| **None** | Free | Nowhere | Transcript and text only |

> **Status:** the Claude path has been tested on real videos. The other options use the same OpenAI-compatible connection, which has only been tested against a mock server, so results may vary by model.

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

Install [Ollama](https://ollama.com/), then download a vision model. Allow about 8 GB of disk space and at least 16 GB of memory.

```bash
ollama pull gemma4:12b
```

```bash
VISION_BASE_URL=http://localhost:11434/v1
VISION_MODEL=gemma4:12b
VISION_MAX_FRAMES=12
```

`VISION_MAX_FRAMES` limits how many images are sent per video. Lower it if a local model runs out of memory or a free tier rejects the request.

Model names change often. If one is rejected, check your provider's current model list.

## Usage

```bash
# A YouTube video or Short
.venv/bin/python -m pipeline.run "https://www.youtube.com/shorts/VIDEO_ID"

# A video file on your computer
.venv/bin/python -m pipeline.run --file path/to/video.mp4

# A file with details about where it came from
.venv/bin/python -m pipeline.run --file ad.mp4 --sponsor "Brand" --date 2026-11-02 --caption "Post text"

# Process a video again, ignoring the saved result
.venv/bin/python -m pipeline.run "https://youtu.be/VIDEO_ID" --force
```

Videos can be up to 5 minutes and 200 MB. You can change these limits in [pipeline/config.py](pipeline/config.py).

The command exits with `0` for a complete record, `1` for a partial record, and `2` for a rejected input.

### Use it from Python

```python
from dotenv import load_dotenv
from pipeline.run import process

load_dotenv()
record = process(url="https://www.youtube.com/shorts/VIDEO_ID", out="store")

print(record.llm_document)
print(record.visual_context.summary if record.visual_context else "No visual analysis")
```

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
| `vision_not_configured` in `flags` | No vision model is set. Check that `.env` exists and an option is uncommented. |
| `vision_auth_failed` | The API key is wrong or expired. |
| `vision_invalid_output` | The model didn't return usable JSON. Try a more capable model. |
| Out of memory with Ollama | Lower `VISION_MAX_FRAMES` or use a smaller model. |
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
.venv/bin/python -m pytest pipeline/tests
.venv/bin/python -m pipeline.evaluate
```

To score accuracy on your own test set, put each video in `pipeline/tests/samples/<name>.mp4` and its hand-written labels in `pipeline/tests/expected/<name>.json`:

```json
{"transcript": "…", "on_screen": ["…"], "no_speech": false, "rejected": null}
```

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for copyright details.
