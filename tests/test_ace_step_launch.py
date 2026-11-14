"""AceStepBackend.load() must launch the sidecar with `uv run --no-sync`.

A plain `uv run` syncs the runtime's .venv to its uv.lock before starting, and that
runtime holds a hand-installed flash-attn 2.8.2 wheel the lock does not list. Measured
2026-09-25 with `uv sync --dry-run` in bin/runtime/ACE-Step-1.5: "Would uninstall 1
package - flash-attn==2.8.2". Driven through the real load() with Popen captured, so
nothing is spawned and no model is loaded.
"""
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parent.parent / "src"
AUDIO_ENGINE = SRC / "litesuite_rust_sidecar" / "litesound"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from litesuite_rust_sidecar.litesound.backends import ace_step  # noqa: E402


class _Launched(Exception):
    pass


class AceStepLaunchTest(unittest.TestCase):
    def test_sidecar_is_launched_without_a_sync(self):
        captured = {}

        def fake_popen(args, **kwargs):
            captured["args"] = list(args)
            captured["cwd"] = kwargs.get("cwd")
            raise _Launched  # stop load() before its health-poll loop

        backend = ace_step.AceStepBackend()
        with mock.patch.dict(sys.modules, {"requests": types.ModuleType("requests")}), \
                mock.patch.object(ace_step.AceStepBackend, "_is_healthy", return_value=False), \
                mock.patch.object(ace_step.AceStepBackend, "_find_uv", return_value="uv"), \
                mock.patch.object(ace_step.subprocess, "Popen", side_effect=fake_popen):
            with self.assertRaises(_Launched):
                backend.load()

        args = captured["args"]
        self.assertEqual(args[:2], ["uv", "run"])
        self.assertIn("--no-sync", args[2:args.index("acestep-api")],
                      f"uv run must not sync the runtime venv: {args}")
        self.assertEqual(captured["cwd"], ace_step.ACESTEP_REPO)


if __name__ == "__main__":
    unittest.main()
