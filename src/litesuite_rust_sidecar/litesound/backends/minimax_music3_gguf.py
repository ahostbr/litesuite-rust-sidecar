"""MiniMax-Music3 GGUF backend, powered by 0xShug0/audio.cpp.

Full songs from a style caption + lyrics: 32 kHz stereo, up to five minutes
in one generation. Pure C++ runtime (ggml) — no torch, no venv, so the
cu128-nightly torch that every diffusers backend depends on cannot be
touched by installing this one.

CLI-PER-GENERATION, deliberately. acestep.cpp gave LiteSound a resident
server; audiocpp has one too (audiocpp_server.exe), but a fresh process per
job keeps v1 simple and the load cost is minor against the workload: a
5-minute song at ~1.35x realtime is minutes of generation, so seconds of
model load do not change the experience. Keep-loaded is the follow-up if
short-duration iteration becomes the common case.

The invocation shape — family, session options for per-component quant
selection, request options — is verified against the BINARY's own
model_specs/minimax_music3.json, not against documentation: a flag is a
claim about the executable, and only the executable can confirm it.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

from ..minimax_music3_gguf_manifest import (
    DEFAULT_PROFILE_ID,
    FAMILY,
    get_profile,
    default_models_dir,
    default_runtime_dir,
    profile_installed,
    runtime_installed,
)

# Shares ffmpeg transcoding with the ACE GGUF backend rather than a third
# copy of the same ffmpeg argv. Private by convention, imported on purpose.
from .ace_step_gguf import _transcode

MM3_MODELS_DIR = Path(
    os.environ.get("LITESOUND_MM3_GGUF_MODELS_DIR", str(default_models_dir()))
)
MM3_RUNTIME_DIR = Path(
    os.environ.get("LITESOUND_MM3_GGUF_RUNTIME_DIR", str(default_runtime_dir()))
)

#: Ceiling the model card claims; the spec's duration_sec is a float cap.
MAX_DURATION_S = 300

#: Load + generation for a 5-minute song at ~0.74 RTF, with slack. A CLI that
#: is still running is visible in the job tracker; a shorter timeout would
#: kill real long-form generations that were simply working.
GENERATION_TIMEOUT_S = 1800


class MiniMaxMusic3GgufBackend:
    def __init__(self, profile_id: Optional[str] = None):
        self._profile = get_profile(profile_id or DEFAULT_PROFILE_ID)
        self._cli: Optional[Path] = None

    # -- lifecycle ----------------------------------------------------------
    def load(self):
        """Verify runtime + weights exist. The model itself loads per
        generation in the CLI process, so 'load' here is a preflight — it
        must FAIL LOUDLY now rather than let the first generation discover a
        missing 6GB file twenty minutes into a queue."""
        if not runtime_installed(MM3_RUNTIME_DIR):
            raise RuntimeError(
                f"audio.cpp runtime not found at {MM3_RUNTIME_DIR} "
                "(audiocpp_cli.exe missing). Install it from the Models panel."
            )
        if not profile_installed(MM3_MODELS_DIR, self._profile):
            raise RuntimeError(
                f"MiniMax-Music3 GGUF profile '{self._profile.id}' is not fully "
                f"installed under {MM3_MODELS_DIR}. Install it from the Models panel."
            )
        self._cli = MM3_RUNTIME_DIR / "audiocpp_cli.exe"
        print(f"[MiniMaxMusic3] ready — profile {self._profile.id}, "
              f"runtime {MM3_RUNTIME_DIR}")

    def unload(self):
        self._cli = None

    def set_profile(self, profile_id: str):
        self._profile = get_profile(profile_id)

    # -- generation ---------------------------------------------------------
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
        **_kwargs,
    ) -> Tuple[Path, int]:
        if self._cli is None:
            raise RuntimeError("backend not loaded")

        # The model takes a style CAPTION and LYRICS as separate inputs.
        # genre/mood fold into the caption; instrumental jobs get explicit
        # instrumental tags rather than an empty lyric — the spec marks
        # lyrics REQUIRED, and an empty string is a different statement than
        # "no vocals".
        caption = ", ".join(x for x in (prompt, genre, mood) if x)
        song_lyrics = lyrics if lyrics and lyrics.strip() else "[inst]"

        dur = max(1, min(int(duration or 30), MAX_DURATION_S))
        out_wav = Path(tempfile.mkdtemp(prefix="litesound_mm3_")) / f"{job_id}.wav"

        cmd = [
            str(self._cli),
            "--task", "gen",
            "--family", FAMILY,
            "--model", str(MM3_MODELS_DIR),
            "--backend", "cuda",
            "--text", caption,
            "--lyrics", song_lyrics,
            "--request-option", f"duration_sec={dur}",
            "--out", str(out_wav),
        ]
        for key, value in self._profile.session_options.items():
            cmd += ["--session-option", f"{key}={value}"]

        started = time.time()
        # cwd = runtime dir so the CUDA/ggml DLLs beside the exe resolve.
        proc = subprocess.run(
            cmd,
            cwd=str(MM3_RUNTIME_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GENERATION_TIMEOUT_S,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if proc.returncode != 0 or not out_wav.is_file() or out_wav.stat().st_size == 0:
            tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-2000:]
            raise RuntimeError(
                f"audiocpp_cli failed (exit {proc.returncode}) after "
                f"{time.time() - started:.0f}s:\n{tail}"
            )

        final = _transcode(out_wav, fmt)
        return final, int(time.time() - started)
