"""`python -m litesuite_rust_sidecar.litesound`: LiteSound's audio engine on 127.0.0.1.

This is the `if __name__ == "__main__"` block of server.py, moved here unchanged in what it does.
"""
import uvicorn

from .server import PORT, app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
