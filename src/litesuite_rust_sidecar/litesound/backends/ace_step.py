"""ACE-Step 1.5 proxy backend. Spawns the acestep-api sidecar on :7431 and
proxies generate calls over HTTP. MIT license — commercial safe."""
import json
import os
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple

from ..port_reaper import foreign_port_holder, kill_owned_port_holder, port_busy_error
from ..sidecar_log import open_sidecar_log


def _default_acestep_repo() -> str:
    return str(Path(__file__).resolve().parents[3] / "bin" / "runtime" / "ACE-Step-1.5")


ACESTEP_REPO = os.environ.get(
    "LITESOUND_ACESTEP_DIR",
    _default_acestep_repo(),
)
# Default must match engine-manager.ts's ACESTEP_PORT. 7431 is LiteSuite's PTY daemon
# — see the note there. The main process now passes LITESOUND_ACESTEP_PORT explicitly,
# so this literal is only the standalone-run fallback.
ACESTEP_PORT = int(os.environ.get("LITESOUND_ACESTEP_PORT", "7430"))
ACESTEP_URL = f"http://127.0.0.1:{ACESTEP_PORT}"
STARTUP_TIMEOUT_S = 300
POLL_INTERVAL = 2.0
WARMUP_BUFFER = 180
RENDER_MULTIPLIER = 4


def _transcode(wav_path: Path, fmt: str) -> Path:
    if fmt == "wav":
        return wav_path
    out = wav_path.with_suffix(f".{fmt}")
    args = ["ffmpeg", "-y", "-i", str(wav_path)]
    if fmt == "mp3":
        args += ["-q:a", "0"]
    args.append(str(out))
    subprocess.run(args, check=True, capture_output=True)
    wav_path.unlink(missing_ok=True)
    return out


class AceStepBackend:
    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None

    def load(self):
        import requests

        if self._is_healthy():
            print("[ACE-Step] Sidecar already running on :%d" % ACESTEP_PORT)
            return

        # Not healthy and not ours: a sidecar spawned now could not bind and would die unseen.
        holder = foreign_port_holder(ACESTEP_PORT, [ACESTEP_REPO])
        if holder:
            raise port_busy_error(ACESTEP_PORT, holder, "LITESOUND_ACESTEP_PORT")

        print("[ACE-Step] Spawning acestep-api sidecar...")
        uv = self._find_uv()
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        log_file, log_path = open_sidecar_log("ace-step-sidecar")
        try:
            self._proc = subprocess.Popen(
                # --no-sync: a plain `uv run` first syncs .venv to uv.lock, and the runtime holds
                # a hand-installed flash-attn 2.8.2 wheel that is not in the lock, so the first
                # load would uninstall it (`uv sync --dry-run`: "Would uninstall 1 package").
                [uv, "run", "--no-sync", "acestep-api", "--port", str(ACESTEP_PORT)],
                cwd=ACESTEP_REPO,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        finally:
            log_file.close()

        deadline = time.time() + STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if self._is_healthy():
                print("[ACE-Step] Sidecar ready on :%d (PID %d)" % (ACESTEP_PORT, self._proc.pid))
                return
            code = self._proc.poll()
            if code is not None:
                self._proc = None
                raise RuntimeError(f"ACE-Step sidecar exited with code {code} during startup; see {log_path}")
            time.sleep(2)

        raise RuntimeError(f"ACE-Step sidecar failed to start within {STARTUP_TIMEOUT_S}s; see {log_path}")

    def unload(self):
        if self._proc is not None:
            self._tree_kill(self._proc.pid)
            self._proc = None
        kill_owned_port_holder(ACESTEP_PORT, [ACESTEP_REPO], "ACE-Step")
        print("[ACE-Step] Sidecar stopped.")

    @staticmethod
    def _tree_kill(pid: int):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                os.killpg(os.getpgid(pid), 9)
        except Exception as e:
            print(f"[ACE-Step] tree_kill({pid}): {e}")

    def generate(
        self,
        prompt: str,
        mode: str,
        duration: int = 30,
        lyrics: Optional[str] = None,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        fmt: str = "mp3",
        job_id: str = "unknown",
        ace_step_options: Optional[dict] = None,
    ) -> Tuple[Path, int]:
        import requests

        opts = ace_step_options or {}
        full_prompt = prompt
        if mood:
            full_prompt = f"{full_prompt}, {mood}"
        if genre:
            full_prompt = f"{genre}, {full_prompt}"

        payload = {
            "prompt": full_prompt,
            "lyrics": lyrics or "",
            "audio_duration": float(duration),
            "inference_steps": opts.get("inference_steps", 8),
            "audio_format": "wav",
            "thinking": opts.get("thinking", True),
            "model": "acestep-v15-turbo",
            "batch_size": 1,
        }
        for key in ("bpm", "key_scale", "time_signature", "vocal_language",
                    "guidance_scale", "task_type", "infer_method"):
            if key in opts:
                payload[key] = opts[key]
        if "batch_size" in opts:
            payload["batch_size"] = opts["batch_size"]
        if "seed" in opts:
            payload["seed"] = opts["seed"]
            payload["use_random_seed"] = False
        if opts.get("use_random_seed") is False and "seed" not in opts:
            payload["use_random_seed"] = False

        budget = max(WARMUP_BUFFER, duration * RENDER_MULTIPLIER) + WARMUP_BUFFER
        print(f"[ACE-Step] Generating ({mode}, {duration}s, budget={budget}s): {full_prompt!r}")
        t0 = time.time()

        for attempt in range(3):
            try:
                resp = requests.post(f"{ACESTEP_URL}/release_task", json=payload, timeout=120)
                resp.raise_for_status()
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                if attempt == 2:
                    raise
                print("[ACE-Step] release_task timeout (model warmup), retrying...")
                time.sleep(5)
        task_id = resp.json()["data"]["task_id"]

        audio_url = self._poll_until_done(task_id, budget, requests)

        fetch_timeout = max(120, duration)
        audio_bytes = requests.get(f"{ACESTEP_URL}{audio_url}", timeout=fetch_timeout).content
        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        tmp_wav.write_bytes(audio_bytes)

        duration_ms = int((time.time() - t0) * 1000)
        return _transcode(tmp_wav, fmt), duration_ms

    def _poll_until_done(self, task_id: str, budget: float, requests) -> str:
        deadline = time.time() + budget
        while time.time() < deadline:
            try:
                resp = requests.post(
                    f"{ACESTEP_URL}/query_result",
                    json={"task_id_list": json.dumps([task_id])},
                    timeout=60,
                )
                resp.raise_for_status()
            except requests.exceptions.ReadTimeout:
                print("[ACE-Step] Poll read timeout (server busy with warmup/render), retrying...")
                time.sleep(POLL_INTERVAL)
                continue
            except requests.exceptions.ConnectionError:
                print("[ACE-Step] Poll connection error, retrying...")
                time.sleep(POLL_INTERVAL)
                continue

            items = resp.json()["data"]
            if not items:
                time.sleep(POLL_INTERVAL)
                continue

            item = items[0]
            status = item.get("status", 0)

            if status == 1:
                result = json.loads(item["result"])
                if result and result[0].get("file"):
                    return result[0]["file"]
                raise RuntimeError("ACE-Step succeeded but no audio file in result")

            if status == 2:
                raise RuntimeError(f"ACE-Step generation failed: {item}")

            time.sleep(POLL_INTERVAL)

        raise RuntimeError("ACE-Step generation timed out after %ds" % int(budget))

    def _is_healthy(self) -> bool:
        try:
            import requests
            r = requests.get(f"{ACESTEP_URL}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _find_uv() -> str:
        for candidate in ["uv", os.path.expanduser("~/.local/bin/uv"), os.path.expanduser("~/.local/bin/uv.exe")]:
            if subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            ).returncode == 0:
                return candidate
        raise RuntimeError("uv not found — install from https://docs.astral.sh/uv/")
