"""Model payload paths, independent of runtime/vendor locations.

Electron passes an absolute LITESOUND_MODELS_DIR to every sidecar. Standalone
Python tools use the same environment variable and repository default.
"""
import os
from pathlib import Path


def models_dir() -> Path:
    configured = os.environ.get("LITESOUND_MODELS_DIR", "").strip()
    return Path(configured).resolve() if configured else Path(__file__).resolve().parents[2] / "bin" / "models"


def vibevoice_voices_dir() -> Path:
    override = os.environ.get("LITESOUND_VIBEVOICE_VOICES_DIR")
    return Path(override) if override else models_dir() / "vibevoice-voices"
