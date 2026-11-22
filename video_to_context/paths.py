"""Where settings and results live, in a source checkout or an installed package."""

from pathlib import Path

from dotenv import load_dotenv

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
HOME_DIR = Path.home() / ".video-to-context"


def app_dir() -> Path:
    """The project folder in a checkout, otherwise a folder in the user's home."""
    return PACKAGE_PARENT if (PACKAGE_PARENT / "pyproject.toml").is_file() else HOME_DIR


def load_settings() -> None:
    """Load .env files. Real environment variables always take priority."""
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(app_dir() / ".env")
