"""port_reaper and the two ACE-Step unload paths: a foreign python on the port SURVIVES.

Real processes on real free ports. Before this, AceStepBackend.unload() and
AceStepGgufBackend.unload() ran taskkill on ANY PID listening on their port. The
CONTROL proves the helper still kills a process it owns; a reaper that kills nothing
would pass the survival tests too.
"""
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
AUDIO_ENGINE = SRC / "litesuite_rust_sidecar" / "litesound"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from litesuite_rust_sidecar.litesound import port_reaper  # noqa: E402
from litesuite_rust_sidecar.litesound.backends import ace_step, ace_step_gguf  # noqa: E402


def _listen_src(port: int) -> str:
    return ("import http.server as h; "
            f"h.HTTPServer(('127.0.0.1', {port}), h.BaseHTTPRequestHandler).serve_forever()")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_listening(port: int, seconds: float = 10) -> None:
    end = time.time() + seconds
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"nothing listened on :{port}")


def _exited(proc: subprocess.Popen, seconds: float) -> bool:
    try:
        proc.wait(timeout=seconds)
        return True
    except subprocess.TimeoutExpired:
        return False


class IsOwnedCommandLineTest(unittest.TestCase):
    def test_rule(self):
        marker = r"C:\Apps\LiteSound\bin\runtime\acestep.cpp"
        self.assertTrue(port_reaper.is_owned_command_line(
            r'"c:\apps\litesound\bin\runtime\ACESTEP.CPP\build\Release\ace-server.exe" --port 7432', [marker]))
        self.assertFalse(port_reaper.is_owned_command_line(
            r"C:\Python311\python.exe D:\tools\other\serve.py serve", [marker]))
        self.assertFalse(port_reaper.is_owned_command_line("", [marker]))
        self.assertFalse(port_reaper.is_owned_command_line("anything", [""]))


@unittest.skipUnless(sys.platform == "win32", "netstat/taskkill reaper is Windows-only")
class ReaperAgainstRealProcessesTest(unittest.TestCase):
    def setUp(self):
        self.procs = []

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

    def _assert_survives_unload(self, module, port_attr: str, backend):
        port = _free_port()
        foreign = self._foreign(port)
        saved = getattr(module, port_attr)
        setattr(module, port_attr, port)
        try:
            backend.unload()  # _proc is None, so only the port reaper runs
        finally:
            setattr(module, port_attr, saved)
        self.assertFalse(_exited(foreign, 1.5), "unload() killed a python LiteSound does not own")
        _wait_listening(port, 1)

    def test_ace_step_unload_spares_a_foreign_python(self):
        self._assert_survives_unload(ace_step, "ACESTEP_PORT", ace_step.AceStepBackend())

    def test_ace_step_gguf_unload_spares_a_foreign_python(self):
        self._assert_survives_unload(ace_step_gguf, "ACESTEP_GGUF_PORT", ace_step_gguf.AceStepGgufBackend())

    def test_control_a_process_under_a_marker_is_killed(self):
        with tempfile.TemporaryDirectory(prefix="litesound-reaper-") as tmp:
            port = _free_port()
            script = Path(tmp) / "owned_server.py"
            script.write_text(_listen_src(port))
            owned = subprocess.Popen([sys.executable, str(script)])
            self.procs.append(owned)
            _wait_listening(port)
            port_reaper.kill_owned_port_holder(port, [tmp], "test")
            self.assertTrue(_exited(owned, 5), "the reaper left its own process running")


if __name__ == "__main__":
    unittest.main()
