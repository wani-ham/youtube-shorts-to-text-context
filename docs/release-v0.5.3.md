# video-to-context v0.5.3

This release fixes installation on Python 3.15.

## Fixed

- **Python 3.15:** the package no longer tries to install on Python 3.15, where its `onnxruntime` dependency is not available. Tools such as `uvx` now choose a supported Python version instead.
- **Builds from source:** the repository now pins Python 3.12 for `uv`, so hosted builds use the tested version.

## Install or upgrade

```bash
uv tool install --python 3.12 --upgrade video-to-context
```

## Known limitations

- Python 3.12 on macOS is the tested setup. Linux, Windows, and Python 3.13 and 3.14 have not had full video runs.
- Only process videos you have the right to use. YouTube's terms restrict downloading.
