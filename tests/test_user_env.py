"""server._ensure_user_env: an engine launched without USERNAME must still resolve a user.

Without it, getpass.getuser() falls back to `import pwd` on Windows and torch._dynamo's
import dies there, so no diffusers backend can load (pattern 47ae9bb0-...-1790307365).
"""
import getpass
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parent.parent / "src"
AUDIO_ENGINE = SRC / "litesuite_rust_sidecar" / "litesound"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from litesuite_rust_sidecar.litesound import server  # noqa: E402

USER_VARS = ("LOGNAME", "USER", "LNAME", "USERNAME")


def _env_without_user():
    return {k: v for k, v in os.environ.items() if k not in USER_VARS}


class EnsureUserEnvTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "the pwd fallback only breaks on Windows")
    def test_stripped_env_breaks_getuser_then_is_repaired(self):
        with mock.patch.dict(os.environ, _env_without_user(), clear=True):
            with self.assertRaises((ImportError, OSError)):
                getpass.getuser()  # the defect: proves the test would fail without the fix
            server._ensure_user_env()
            self.assertEqual(getpass.getuser(), os.getlogin())

    def test_existing_user_is_left_alone(self):
        env = _env_without_user()
        env["USER"] = "someone-else"
        with mock.patch.dict(os.environ, env, clear=True):
            server._ensure_user_env()
            self.assertNotIn("USERNAME", os.environ)
            self.assertEqual(getpass.getuser(), "someone-else")


if __name__ == "__main__":
    unittest.main()
