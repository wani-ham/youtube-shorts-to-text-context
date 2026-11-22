"""Turn short videos into structured, timestamped text for AI services."""

import os
import sys
from pathlib import Path

__version__ = "0.5.3"

# Find the environment's ffmpeg even when the environment is not activated.
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
