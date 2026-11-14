"""ACE-Step GGUF backend powered by ServeurpersoCom/acestep.cpp."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from email import message_from_bytes
from pathlib import Path
from typing import Optional, Tuple

from ..port_reaper import foreign_port_holder, kill_owned_port_holder, port_busy_error
from ..sidecar_log import open_sidecar_log

from ..ace_step_gguf_manifest import (
    DEFAULT_PROFILE_ID,
    get_profile,
    default_models_dir,
    default_runtime_dir,
    profile_installed,
)


ACESTEP_GGUF_DIR = Path(os.environ.get("LITESOUND_ACESTEP_GGUF_DIR", str(default_runtime_dir())))
ACESTEP_GGUF_MODELS_DIR = Path(
    os.environ.get("LITESOUND_ACESTEP_GGUF_MODELS_DIR", str(default_models_dir()))
)
ACESTEP_GGUF_ADAPTERS_DIR = Path(
    os.environ.get("LITESOUND_ACESTEP_GGUF_ADAPTERS_DIR", str(ACESTEP_GGUF_MODELS_DIR / "adapters"))
)
ACESTEP_GGUF_PORT = int(os.environ.get("LITESOUND_ACESTEP_GGUF_PORT", "7432"))
ACESTEP_GGUF_URL = f"http://127.0.0.1:{ACESTEP_GGUF_PORT}"
ACESTEP_GGUF_KEEP_LOADED = os.environ.get("LITESOUND_ACESTEP_GGUF_KEEP_LOADED", "1") not in {
    "0",
    "false",
    "False",
}
STARTUP_TIMEOUT_S = 120
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


class AceStepGgufBackend:
    def __init__(self, profile_id: Optional[str] = None):
        self._proc: Optional[subprocess.Popen] = None
        self.profile = get_profile(profile_id or os.environ.get(
            "LITESOUND_ACESTEP_GGUF_PROFILE", DEFAULT_PROFILE_ID))

    def set_profile(self, profile_id: Optional[str]) -> None:
        """Switch the active quant profile. ace-server loads models by filename
        per request, so the running server needs no restart — subsequent /lm and
        /synth calls just reference the new profile's files."""
        new = get_profile(profile_id)
        if new.id != self.profile.id:
            print(f"[ACE-Step GGUF] profile -> {new.id} ({new.label})")
        self.profile = new

    def load(self):
        if not profile_installed(ACESTEP_GGUF_MODELS_DIR, self.profile):
            raise RuntimeError(
                f"ACE-Step GGUF profile '{self.profile.label}' is incomplete. "
                f"Install its GGUF files to {ACESTEP_GGUF_MODELS_DIR}."
            )

        if self._is_healthy():
            print("[ACE-Step GGUF] ace-server already running on :%d" % ACESTEP_GGUF_PORT)
            return

        # Not healthy and not ours: an ace-server spawned now could not bind and would die unseen.
        holder = foreign_port_holder(ACESTEP_GGUF_PORT, [ACESTEP_GGUF_DIR])
        if holder:
            raise port_busy_error(ACESTEP_GGUF_PORT, holder, "LITESOUND_ACESTEP_GGUF_PORT")

        server = self._find_server_exe()
        ACESTEP_GGUF_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
        args = [
            str(server),
            "--models",
            str(ACESTEP_GGUF_MODELS_DIR),
            "--adapters",
            str(ACESTEP_GGUF_ADAPTERS_DIR),
            "--host",
            "127.0.0.1",
            "--port",
            str(ACESTEP_GGUF_PORT),
            "--max-batch",
            "1",
        ]
        if ACESTEP_GGUF_KEEP_LOADED:
            args.append("--keep-loaded")

        print("[ACE-Step GGUF] Spawning ace-server...")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        env = os.environ.copy()
        env["PATH"] = str(server.parent) + os.pathsep + env.get("PATH", "")
        log_file, log_path = open_sidecar_log("ace-step-gguf-sidecar")
        try:
            self._proc = subprocess.Popen(
                args,
                cwd=ACESTEP_GGUF_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                env=env,
            )
        finally:
            log_file.close()

        deadline = time.time() + STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if self._is_healthy():
                print("[ACE-Step GGUF] ace-server ready on :%d (PID %d)" % (ACESTEP_GGUF_PORT, self._proc.pid))
                return
            code = self._proc.poll()
            if code is not None:
                self._proc = None
                raise RuntimeError(f"ACE-Step GGUF ace-server exited with code {code} during startup; see {log_path}")
            time.sleep(2)

        raise RuntimeError(f"ACE-Step GGUF ace-server failed to start within {STARTUP_TIMEOUT_S}s; see {log_path}")

    def unload(self):
        if self._proc is not None:
            self._tree_kill(self._proc.pid)
            self._proc = None
        kill_owned_port_holder(ACESTEP_GGUF_PORT, [ACESTEP_GGUF_DIR], "ACE-Step GGUF")
        print("[ACE-Step GGUF] ace-server stopped.")

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

        request_json = self._build_request(full_prompt, lyrics, duration, opts)
        request_json["lm_model"] = self.profile.lm_model
        request_json["synth_model"] = self.profile.synth_model
        budget = max(WARMUP_BUFFER, duration * RENDER_MULTIPLIER) + WARMUP_BUFFER
        print(f"[ACE-Step GGUF] Generating ({mode}, {duration}s, budget={budget}s): {full_prompt!r}")
        t0 = time.time()

        lm_job = self._submit_job("/lm", request_json, requests)
        lm_requests = self._poll_json_result(lm_job, budget, requests)
        if not isinstance(lm_requests, list) or not lm_requests:
            raise RuntimeError("ACE-Step GGUF /lm returned no synth request")

        synth_payload = []
        for item in lm_requests:
            if isinstance(item, dict):
                item["output_format"] = "wav16"
                item["synth_model"] = self.profile.synth_model
                item["lm_model"] = self.profile.lm_model
                synth_payload.append(item)
        if not synth_payload:
            raise RuntimeError("ACE-Step GGUF /lm result did not contain request objects")

        synth_job = self._submit_job("/synth", synth_payload, requests)
        content, content_type = self._poll_binary_result(synth_job, budget, requests)
        audio_bytes = self._extract_audio(content, content_type)

        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        tmp_wav.write_bytes(audio_bytes)
        duration_ms = int((time.time() - t0) * 1000)
        return _transcode(tmp_wav, fmt), duration_ms

    @staticmethod
    def _build_request(prompt: str, lyrics: Optional[str], duration: int, opts: dict) -> dict:
        payload = {
            "caption": prompt,
            "lyrics": lyrics or "",
            "duration": float(duration),
            "seed": int(opts.get("seed", -1)) if not opts.get("use_random_seed", True) else -1,
            "lm_batch_size": 1,
            "synth_batch_size": int(opts.get("batch_size", 1)),
            "inference_steps": int(opts.get("inference_steps", 8)),
            "guidance_scale": float(opts.get("guidance_scale", 1.0)),
            "task_type": opts.get("task_type", "text2music"),
            "solver": AceStepGgufBackend._solver(opts.get("infer_method")),
            "output_format": "wav16",
        }
        if "thinking" in opts:
            payload["use_cot_caption"] = bool(opts["thinking"])
        mappings = {
            "bpm": "bpm",
            "key_scale": "keyscale",
            "time_signature": "timesignature",
            "vocal_language": "vocal_language",
        }
        for src, dest in mappings.items():
            if src in opts:
                payload[dest] = opts[src]
        return payload

    @staticmethod
    def _solver(value: object) -> str:
        if value == "sde":
            return "sde"
        if value == "stork4":
            return "stork4"
        return "euler"

    @staticmethod
    def _submit_job(endpoint: str, payload, requests) -> str:
        resp = requests.post(f"{ACESTEP_GGUF_URL}{endpoint}", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("id")
        if not job_id:
            raise RuntimeError(f"ACE-Step GGUF {endpoint} did not return a job id: {data}")
        return str(job_id)

    def _poll_json_result(self, job_id: str, budget: float, requests):
        content, _ = self._poll_result(job_id, budget, requests)
        return json.loads(content.decode("utf-8"))

    def _poll_binary_result(self, job_id: str, budget: float, requests) -> tuple[bytes, str]:
        return self._poll_result(job_id, budget, requests)

    @staticmethod
    def _poll_result(job_id: str, budget: float, requests) -> tuple[bytes, str]:
        deadline = time.time() + budget
        while time.time() < deadline:
            try:
                resp = requests.get(f"{ACESTEP_GGUF_URL}/job", params={"id": job_id}, timeout=60)
                resp.raise_for_status()
                status = resp.json().get("status")
            except requests.exceptions.RequestException:
                print("[ACE-Step GGUF] Poll error; retrying...")
                time.sleep(POLL_INTERVAL)
                continue

            if status == "done":
                result = requests.get(
                    f"{ACESTEP_GGUF_URL}/job",
                    params={"id": job_id, "result": "1"},
                    timeout=180,
                )
                result.raise_for_status()
                return result.content, result.headers.get("content-type", "")
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"ACE-Step GGUF job {job_id} ended with status {status}")

            time.sleep(POLL_INTERVAL)

        raise RuntimeError("ACE-Step GGUF generation timed out after %ds" % int(budget))

    @staticmethod
    def _extract_audio(content: bytes, content_type: str) -> bytes:
        if content_type.startswith("audio/"):
            return content
        if "multipart/" not in content_type:
            raise RuntimeError(f"ACE-Step GGUF returned non-audio content type: {content_type}")

        raw = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8") + content
        message = message_from_bytes(raw)
        for part in message.walk():
            if part.get_content_type().startswith("audio/"):
                payload = part.get_payload(decode=True)
                if payload:
                    return payload
        raise RuntimeError("ACE-Step GGUF multipart result did not contain an audio part")

    def _is_healthy(self) -> bool:
        try:
            import requests
            r = requests.get(f"{ACESTEP_GGUF_URL}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _find_server_exe() -> Path:
        candidates = [
            ACESTEP_GGUF_DIR / "build" / "Release" / "ace-server.exe",
            ACESTEP_GGUF_DIR / "build" / "ace-server.exe",
            ACESTEP_GGUF_DIR / "ace-server.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise RuntimeError(f"ace-server.exe not found under {ACESTEP_GGUF_DIR}; build acestep.cpp first")

    @staticmethod
    def _tree_kill(pid: int):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                os.killpg(os.getpgid(pid), 9)
        except Exception as e:
            print(f"[ACE-Step GGUF] tree_kill({pid}): {e}")

