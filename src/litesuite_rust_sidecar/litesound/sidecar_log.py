"""Where a spawned sidecar's stdout/stderr go, so a sidecar that dies at startup leaves a trace.

The directory is LITESOUND_LOG_DIR, which the Electron main process sets to
`<dataDir>/logs` (Electron's own `logs` path derives from the same userData). The
fallback is the same folder relative to the repo, for a standalone engine run.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Tuple


def log_dir() -> Path:
    configured = os.environ.get("LITESOUND_LOG_DIR", "").strip()
    return Path(configured) if configured else Path(__file__).resolve().parents[2] / "data" / "logs"


def open_sidecar_log(name: str) -> Tuple[BinaryIO, Path]:
    """Open `<log dir>/<name>.log` for appending, stamped with the spawn time.

    The caller hands the file to Popen and closes its own handle; the child keeps its copy.
    """
    path = log_dir() / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "ab")
    f.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} spawn ===\n".encode())
    f.flush()
    return f, path
