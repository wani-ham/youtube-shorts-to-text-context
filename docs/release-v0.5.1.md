# video-to-context v0.5.1

Give AI tools a text version of a short video: what was said, what was written on screen, and what was shown, with a source label and timestamp on every line.

## Install

You need [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
uv tool install --python 3.12 video-to-context
uv tool update-shell
```

Open a new terminal, then run:

```bash
video-to-context "https://www.youtube.com/shorts/VIDEO_ID"
```

## What's new

- **More careful visual analysis:** the vision prompt no longer infers places or events from weak cues, such as calling a blurred background "outdoors".
- **Cleaner shots:** near-identical shots in the same setting now merge into one line.
- **More accurate editing labels:** a logo or captions alone no longer count as a templated frame.
- **Easier setup:** the Quick start now includes the PATH step that uv needs.

## Known limitations

- Tested on macOS. Linux and Windows are untested.
- Claude Haiku is the tested vision model. Ollama with Gemma 4 reads on-screen text well, but its scene descriptions are unreliable. Gemini, OpenAI, and OpenRouter are untested with their real services.
- The footage type can vary between runs, and speech recognition can misspell names.
- Only process videos you have the right to use. YouTube's terms restrict downloading.
