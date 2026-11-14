"""The launch token (C4): nothing on the PC but the app that started an engine can drive it.

The parent app makes 32 random bytes per run, hex-encoded, and passes them to every engine it
starts in `LITESUITE_SIDECAR_TOKEN`. With that variable set, a request without the header
`X-LiteSuite-Sidecar-Token: <token>` (compared in constant time) is answered 401
`{"detail": "missing or wrong sidecar token"}`, except `/health`, which answers `{"ok": true}` and
nothing more. With the variable unset (an engine started by hand, or an app that does not pass
one), the engine behaves exactly as before.

This is security, not licensing. The token is read from the environment on each request, so it is
never written anywhere by the engine.
"""
from __future__ import annotations

import hmac
import json
import os

TOKEN_VAR = "LITESUITE_SIDECAR_TOKEN"
HEADER = b"x-litesuite-sidecar-token"
REFUSED = {"detail": "missing or wrong sidecar token"}


class SidecarToken:
    """ASGI middleware: wraps the whole app, so every route, present or future, is covered."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        token = os.environ.get(TOKEN_VAR, "")
        if scope["type"] != "http" or not token:
            await self.app(scope, receive, send)
            return
        given = b""
        for name, value in scope.get("headers", []):
            if name.lower() == HEADER:
                given = value
        if hmac.compare_digest(given, token.encode("utf-8")):
            await self.app(scope, receive, send)
            return
        if scope.get("path") == "/health":
            await _json(send, 200, {"ok": True})
        else:
            await _json(send, 401, REFUSED)


async def _json(send, status: int, body: dict) -> None:
    data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]})
    await send({"type": "http.response.body", "body": data})
