"""Kill the process holding a port only if it is LiteSound's own.

Python twin of src/main/port-reaper.ts, with the same rule, for the backends that
run in the engine process. Ownership is decided by COMMAND LINE: a holder is ours
only if its command line names one of the caller's paths. Image names cannot decide
it, because a Windows venv python.exe is a redirector and the process that listens
is the base interpreter. An unreadable command line is not ours: refusing is safe,
killing is not. Measured 2026-09-25: an unrelated python (chrome-bridge bridge.py)
held 7429/7430 on this machine.
"""
import os
import subprocess
from typing import Iterable, List, Optional, Tuple

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def is_owned_command_line(command_line: str, markers: Iterable[str]) -> bool:
    cmd = command_line.lower()
    return any(m and m.lower() in cmd for m in markers)


def _listening_pids(port: int) -> List[int]:
    out = subprocess.check_output(["netstat", "-ano", "-p", "TCP"], creationflags=_NO_WINDOW, text=True)
    pids = []
    for line in out.splitlines():
        cols = line.split()
        # Proto  Local  Foreign  State  PID. Exact match, so :743 never matches :7430.
        if len(cols) == 5 and cols[1] == f"127.0.0.1:{port}" and cols[3] == "LISTENING":
            pid = int(cols[4])
            if pid > 0 and pid not in pids:
                pids.append(pid)
    return pids


def _command_line_of(pid: int) -> str:
    try:
        return subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
            creationflags=_NO_WINDOW, text=True, timeout=5,
        ).strip()
    except Exception:
        return ""


def foreign_port_holder(port: int, markers: Iterable[str]) -> Optional[Tuple[int, str]]:
    """The first process listening on `port` that is not LiteSound's, as (pid, command line).

    None when the port is free or every listener is ours. Never kills anything.
    """
    if os.name != "nt":
        return None
    markers = [str(m) for m in markers]
    try:
        pids = _listening_pids(port)
    except Exception as e:
        print(f"[port_reaper] foreign_port_holder({port}): {e}")
        return None
    for pid in pids:
        cmd = _command_line_of(pid)
        if not is_owned_command_line(cmd, markers):
            return pid, cmd
    return None


def port_busy_error(port: int, holder: Tuple[int, str], env_var: str) -> RuntimeError:
    pid, cmd = holder
    return RuntimeError(
        f"Port :{port} is already in use by PID {pid} ({cmd or 'command line unreadable'}), "
        f"which is not a LiteSound process. LiteSound will not kill it: stop it, "
        f"or set {env_var} to a free port."
    )


def kill_owned_port_holder(port: int, markers: Iterable[str], tag: str) -> None:
    if os.name != "nt":
        return
    markers = [str(m) for m in markers]
    try:
        pids = _listening_pids(port)
    except Exception as e:
        print(f"[{tag}] kill_owned_port_holder({port}): {e}")
        return
    for pid in pids:
        cmd = _command_line_of(pid)
        if not is_owned_command_line(cmd, markers):
            print(f"[{tag}] :{port} is held by PID {pid} ({cmd or 'command line unreadable'}), "
                  f"which is not a LiteSound process; refusing to kill it.")
            continue
        print(f"[{tag}] Killing orphan on :{port} (PID {pid})")
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=_NO_WINDOW)
