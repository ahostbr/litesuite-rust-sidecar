"""Chatterbox backend (Resemble AI). MIT license.

Zero-shot single-speaker speech, optionally cloned from a short reference clip.

⚠️ The `chatterbox-tts` package is NOT installed in python/.venv and must not be:
its dependencies are hard-pinned to torch==2.6.0, which has no sm_120 kernels and
would take the 5090 off CUDA for every other backend. See
`python/audio_engine/vendor/vibevoice/PROVENANCE.md` for the measurement. Imports
here stay lazy (as in every other backend) so that absence costs nothing until a
generation is actually requested.
"""
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple

_INSTALL_HINT = (
    "Chatterbox is not installed in this environment. Its dependencies pin "
    "torch==2.6.0, which would remove sm_120 (RTX 5090) CUDA support from every "
    "other backend, so it cannot share python/.venv. It needs an isolated "
    "environment — see python/audio_engine/vendor/vibevoice/PROVENANCE.md."
)


class ChatterboxBackend:
    def __init__(self):
        self._model = None
        self._device = "cuda"

    def load(self):
        print("[Chatterbox] Loading ChatterboxTTS...")
        import torch
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError as e:
            raise RuntimeError(f"{_INSTALL_HINT} (import failed: {e})") from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ChatterboxTTS.from_pretrained(device=self._device)
        print(f"[Chatterbox] Loaded on {self._device}.")

    def unload(self):
        import gc, torch
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Chatterbox] Unloaded.")

    def generate(
        self,
        prompt: str,
        mode: str,
        duration: int = 15,
        lyrics: Optional[str] = None,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        fmt: str = "mp3",
        job_id: str = "unknown",
        **kwargs,
    ) -> Tuple[Path, int]:
        if self._model is None:
            raise RuntimeError("Chatterbox not loaded")

        import numpy as np
        import scipy.io.wavfile as wav

        # `duration` is an input for music backends but an OUTPUT for speech —
        # the text decides how long the clip is. Deliberately unused.
        text = prompt or lyrics or ""
        if not text.strip():
            raise ValueError("Chatterbox requires non-empty text to speak")

        gen_kwargs = {}
        reference_path = kwargs.get("reference_path")
        if reference_path:
            if not Path(reference_path).is_file():
                raise FileNotFoundError(f"Reference clip not found: {reference_path}")
            gen_kwargs["audio_prompt_path"] = str(reference_path)
        for key in ("exaggeration", "cfg_weight"):
            if kwargs.get(key) is not None:
                gen_kwargs[key] = float(kwargs[key])

        print(f"[Chatterbox] Generating: {text[:80]!r}"
              + (f" (cloning {Path(reference_path).name})" if reference_path else ""))
        t0 = time.time()

        audio = self._model.generate(text, **gen_kwargs)
        # duration_ms is elapsed GENERATION time across this codebase (see bark.py,
        # musicgen.py), not the length of the audio. Kept consistent deliberately.
        duration_ms = int((time.time() - t0) * 1000)

        audio_np = audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        sample_rate = getattr(self._model, "sr", 24000)
        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        wav.write(str(tmp_wav), sample_rate, audio_np.astype(np.float32))

        if fmt == "wav":
            return tmp_wav, duration_ms

        out = tmp_wav.with_suffix(f".{fmt}")
        args = ["ffmpeg", "-y", "-i", str(tmp_wav)]
        if fmt == "mp3":
            args += ["-q:a", "0"]
        args.append(str(out))
        subprocess.run(args, check=True, capture_output=True)
        tmp_wav.unlink(missing_ok=True)
        return out, duration_ms
