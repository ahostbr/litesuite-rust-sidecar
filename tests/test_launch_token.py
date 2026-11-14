"""C4, the launch token: no variable -> open as before; variable set and no header -> 401; wrong
header -> 401; right header -> 200; /health answers in both, with nothing but {"ok": true} to a
caller without the token."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient  # noqa: E402

from litesuite_rust_sidecar.auth import REFUSED, TOKEN_VAR  # noqa: E402
from litesuite_rust_sidecar.litesound import server  # noqa: E402

TOKEN = "ab" * 32
HEADER = "X-LiteSuite-Sidecar-Token"


class LaunchTokenTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def test_no_variable_is_open_as_before(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(TOKEN_VAR, None)
            self.assertEqual(self.client.get("/status").status_code, 200)
            health = self.client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertIn("backend", health.json())

    def test_no_header_is_refused(self):
        with mock.patch.dict(os.environ, {TOKEN_VAR: TOKEN}):
            r = self.client.get("/status")
            self.assertEqual((r.status_code, r.json()), (401, REFUSED))
            r = self.client.post("/backend/unload")
            self.assertEqual(r.status_code, 401)

    def test_wrong_header_is_refused(self):
        with mock.patch.dict(os.environ, {TOKEN_VAR: TOKEN}):
            for wrong in ["", TOKEN[:-1], TOKEN + "0", "cd" * 32, TOKEN.upper()]:
                r = self.client.get("/status", headers={HEADER: wrong})
                self.assertEqual((r.status_code, r.json()), (401, REFUSED), wrong)

    def test_right_header_is_served(self):
        with mock.patch.dict(os.environ, {TOKEN_VAR: TOKEN}):
            self.assertEqual(self.client.get("/status", headers={HEADER: TOKEN}).status_code, 200)
            self.assertEqual(self.client.get("/hub/backends", headers={HEADER: TOKEN}).status_code, 200)

    def test_health_answers_in_both_and_says_nothing_without_the_token(self):
        with mock.patch.dict(os.environ, {TOKEN_VAR: TOKEN}):
            r = self.client.get("/health")
            self.assertEqual((r.status_code, r.json()), (200, {"ok": True}))
            r = self.client.get("/health", headers={HEADER: TOKEN})
            self.assertEqual(r.status_code, 200)
            self.assertIn("backend", r.json())
            # Only the exact path is open: a near miss is refused like any other route.
            self.assertEqual(self.client.get("/health/x").status_code, 401)


if __name__ == "__main__":
    unittest.main()
