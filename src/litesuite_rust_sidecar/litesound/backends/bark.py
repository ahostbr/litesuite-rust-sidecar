"""Bark backend (Suno AI). MIT license."""
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple


class BarkBackend:
    def __init__(self):
        self._loaded = False

    def load(self):
        print("[Bark] Preloading models...")
        from bark import preload_models
        preload_models()
        self._loaded = True
        print("[Bark] Loaded.")

    def unload(self):
        import gc, torch
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Bark] Unloaded.")

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
    ) -> Tuple[Path, int]:
        if not self._loaded:
            raise RuntimeError("Bark not loaded")

        import numpy as np
        import scipy.io.wavfile as wav
        from bark import generate_audio, SAMPLE_RATE

        text = prompt
        if lyrics:
            text = lyrics

        print(f"[Bark] Generating: {text[:80]!r}")
        t0 = time.time()

        audio_array = generate_audio(text)
        duration_ms = int((time.time() - t0) * 1000)

        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        wav.write(str(tmp_wav), SAMPLE_RATE, audio_array.astype(np.float32))

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
