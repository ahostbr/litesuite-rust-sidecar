"""T0038: the ACE-Step sidecars must refuse a port a foreign process holds, and must log.

Measured 2026-09-25: a chrome-bridge bridge.py held 7430 and answered /health with HTTP
426. AceStepBackend.load() saw "not healthy", spawned `uv run --no-sync acestep-api`,
which could not bind, with stdout/stderr on DEVNULL, so it died unseen and load() ran
into its 5-minute timeout, naming neither the holder nor a log.

Real foreign processes on real free ports; Popen is replaced so no sidecar is launched
and no model is loaded. The CONTROLS matter as much as the refusals: a guard that
refuses every port passes the refusal tests too.
"""
import contextlib
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parent.parent / "src"
AUDIO_ENGINE = SRC / "litesuite_rust_sidecar" / "litesound"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from litesuite_rust_sidecar.litesound import port_reaper  # noqa: E402
from litesuite_rust_sidecar.litesound.backends import ace_step, ace_step_gguf  # noqa: E402
from tests.test_port_reaper import _free_port, _listen_src, _wait_listening  # noqa: E402


class _Launched(Exception):
    pass


_REAL_POPEN = subprocess.Popen


def _sidecar_popen(on_spawn):
    """Popen stand-in that fakes only the sidecar spawn.

    The reaper's own netstat/powershell calls go through subprocess.Popen too, so a blanket
    mock would blind the very check under test.
    """
    def popen(args, **kwargs):
        if str(args[0]).lower() in ("netstat", "powershell.exe"):
            return _REAL_POPEN(args, **kwargs)
        return on_spawn(args, **kwargs)
    return popen


class _FakeProc:
    """A sidecar that is spawned and then either keeps running or has already died."""

    def __init__(self, exit_code=None):
        self.pid = 4242
        self._exit_code = exit_code

    def poll(self):
        return self._exit_code


@unittest.skipUnless(sys.platform == "win32", "netstat reaper is Windows-only")
class ForeignPortHolderTest(unittest.TestCase):
    def setUp(self):
        self.procs = []
        self.tmp = tempfile.TemporaryDirectory(prefix="litesound-portbusy-")
        self.addCleanup(self.tmp.cleanup)

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait()

    def _foreign(self, port: int) -> subprocess.Popen:
        p = subprocess.Popen([sys.executable, "-c", _listen_src(port)])
        self.procs.append(p)
        _wait_listening(port)
        return p

    # --- the helper ---------------------------------------------------------------

    def test_helper_reports_foreign_ignores_owned_and_free(self):
        port = _free_port()
        self.assertIsNone(port_reaper.foreign_port_holder(port, ["x"]), "a free port reported a holder")

        foreign = self._foreign(port)
        holder = port_reaper.foreign_port_holder(port, [self.tmp.name])
        self.assertIsNotNone(holder)
        pid, cmd = holder
        self.assertEqual(pid, foreign.pid)
        self.assertIn("http.server", cmd)

        # CONTROL: once a marker in its command line claims it, it is not foreign.
        self.assertIsNone(port_reaper.foreign_port_holder(port, ["http.server"]),
                          "an owned holder was reported as foreign")

    # --- load() refuses -----------------------------------------------------------

    def _assert_load_refuses(self, module, port_attr, backend, extra_patches=()):
        port = _free_port()
        foreign = self._foreign(port)
        spawned = []

        def on_spawn(args, **kwargs):
            spawned.append(args)
            raise _Launched

        popen = _sidecar_popen(on_spawn)
        with mock.patch.object(module, port_attr, port), \
                mock.patch.object(type(backend), "_is_healthy", return_value=False), \
                mock.patch.dict(sys.modules, {"requests": types.ModuleType("requests")}), \
                mock.patch.object(module.subprocess, "Popen", popen):
            with contextlib.ExitStack() as stack:
                for p in extra_patches:
                    stack.enter_context(p)
                with self.assertRaises(RuntimeError) as ctx:
                    backend.load()
        msg = str(ctx.exception)
        self.assertIn(f":{port}", msg)
        self.assertIn(f"PID {foreign.pid}", msg)
        self.assertIn("http.server", msg)
        self.assertEqual(spawned, [], 'load() spawned a sidecar onto a port someone else holds')
        self.assertIsNone(foreign.poll(), "load() killed a process LiteSound does not own")
        _wait_listening(port, 1)

    def test_ace_step_load_refuses_a_foreign_holder(self):
        self._assert_load_refuses(
            ace_step, "ACESTEP_PORT", ace_step.AceStepBackend(),
            [mock.patch.object(ace_step.AceStepBackend, "_find_uv", return_value="uv")],
        )

    def test_ace_step_gguf_load_refuses_a_foreign_holder(self):
        self._assert_load_refuses(
            ace_step_gguf, "ACESTEP_GGUF_PORT", ace_step_gguf.AceStepGgufBackend(),
            [mock.patch.object(ace_step_gguf, "profile_installed", return_value=True)],
        )

    def test_control_a_free_port_reaches_the_spawn(self):
        port = _free_port()
        with mock.patch.object(ace_step, "ACESTEP_PORT", port), \
                mock.patch.object(ace_step.AceStepBackend, "_is_healthy", return_value=False), \
                mock.patch.object(ace_step.AceStepBackend, "_find_uv", return_value="uv"), \
                mock.patch.dict(sys.modules, {"requests": types.ModuleType("requests")}), \
                mock.patch.object(ace_step.subprocess, "Popen",
                                  _sidecar_popen(mock.Mock(side_effect=_Launched))):
            with self.assertRaises(_Launched):
                ace_step.AceStepBackend().load()


class SidecarLogTest(unittest.TestCase):
    """stdout/stderr go to a file under the log dir, and its path is in the failure."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="litesound-sidecarlog-")
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict("os.environ", {"LITESOUND_LOG_DIR": self.tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def _load(self, module, port_attr, backend, proc, timeout_attr, timeout, extra=()):
        captured = {}

        def on_spawn(args, **kwargs):
            captured.update(kwargs)
            return proc

        patches = [
            mock.patch.object(module, port_attr, _free_port()),
            mock.patch.object(module, timeout_attr, timeout),
            mock.patch.object(type(backend), "_is_healthy", return_value=False),
            mock.patch.object(module.subprocess, "Popen", _sidecar_popen(on_spawn)),
            mock.patch.dict(sys.modules, {"requests": types.ModuleType("requests")}),
            *extra,
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with self.assertRaises(RuntimeError) as ctx:
                backend.load()
        return captured, str(ctx.exception)

    def _gguf_extras(self):
        return [
            mock.patch.object(ace_step_gguf, "profile_installed", return_value=True),
            mock.patch.object(ace_step_gguf.AceStepGgufBackend, "_find_server_exe",
                              return_value=Path(self.tmp.name) / "ace-server.exe"),
            mock.patch.object(ace_step_gguf, "ACESTEP_GGUF_ADAPTERS_DIR", Path(self.tmp.name) / "adapters"),
            mock.patch.object(ace_step_gguf, "ACESTEP_GGUF_DIR", Path(self.tmp.name)),
        ]

    def _assert_logged(self, captured, msg, expected_name):
        self.assertIsNot(captured["stdout"], subprocess.DEVNULL, "stdout still goes to DEVNULL")
        self.assertIsNot(captured["stderr"], subprocess.DEVNULL, "stderr still goes to DEVNULL")
        log = Path(self.tmp.name) / expected_name
        self.assertTrue(log.is_file(), f"no sidecar log at {log}")
        self.assertIn(str(log), msg, "the failure does not say where the log is")

    def test_ace_step_timeout_names_the_log(self):
        captured, msg = self._load(
            ace_step, "ACESTEP_PORT", ace_step.AceStepBackend(), _FakeProc(),
            "STARTUP_TIMEOUT_S", 0,
            [mock.patch.object(ace_step.AceStepBackend, "_find_uv", return_value="uv")],
        )
        self._assert_logged(captured, msg, "ace-step-sidecar.log")

    def test_ace_step_sidecar_that_died_fails_fast_and_names_the_log(self):
        started = time.time()
        captured, msg = self._load(
            ace_step, "ACESTEP_PORT", ace_step.AceStepBackend(), _FakeProc(exit_code=1),
            "STARTUP_TIMEOUT_S", 60,
            [mock.patch.object(ace_step.AceStepBackend, "_find_uv", return_value="uv")],
        )
        self.assertLess(time.time() - started, 30, "load() waited out the timeout on a dead sidecar")
        self.assertIn("exited", msg)
        self.assertIn("code 1", msg)
        self._assert_logged(captured, msg, "ace-step-sidecar.log")

    def test_ace_step_gguf_timeout_names_the_log(self):
        captured, msg = self._load(
            ace_step_gguf, "ACESTEP_GGUF_PORT", ace_step_gguf.AceStepGgufBackend(), _FakeProc(),
            "STARTUP_TIMEOUT_S", 0, self._gguf_extras(),
        )
        self._assert_logged(captured, msg, "ace-step-gguf-sidecar.log")


if __name__ == "__main__":
    unittest.main()
