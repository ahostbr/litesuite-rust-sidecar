"""MusicGen backend (Meta AudioCraft). Non-commercial CC-BY-NC 4.0."""
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple


class MusicGenBackend:
    def __init__(self):
        self._model = None
        self._processor = None
        self._model_id = "facebook/musicgen-medium"

    def load(self):
        print(f"[MusicGen] Loading {self._model_id}...")
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        import torch
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = MusicgenForConditionalGeneration.from_pretrained(self._model_id)
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        print("[MusicGen] Loaded.")

    def unload(self):
        import gc, torch
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[MusicGen] Unloaded.")

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
    ) -> Tuple[Path, int]:
        if self._model is None or self._processor is None:
            raise RuntimeError("MusicGen not loaded")

        import torch, scipy.io.wavfile as wav, numpy as np

        # Build full prompt
        full_prompt = prompt
        if genre:
            full_prompt = f"{genre} {full_prompt}"
        if mood:
            full_prompt = f"{full_prompt}, {mood}"

        print(f"[MusicGen] Generating: {full_prompt!r} ({duration}s)")
        t0 = time.time()

        inputs = self._processor(text=[full_prompt], padding=True, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        sampling_rate = self._model.config.audio_encoder.sampling_rate
        max_new_tokens = int(duration * sampling_rate / self._model.config.audio_encoder.hop_length)

        with torch.no_grad():
            audio_values = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        audio_np = audio_values[0, 0].cpu().numpy()
        duration_ms = int((time.time() - t0) * 1000)

        # Save to temp WAV first
        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        wav.write(str(tmp_wav), sampling_rate, audio_np.astype(np.float32))

        # Transcode if needed
        if fmt == "mp3":
            out_path = tmp_wav.with_suffix(".mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_wav), "-q:a", "0", str(out_path)],
                check=True, capture_output=True,
            )
            tmp_wav.unlink(missing_ok=True)
            return out_path, duration_ms
        elif fmt == "flac":
            out_path = tmp_wav.with_suffix(".flac")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_wav), str(out_path)],
                check=True, capture_output=True,
            )
            tmp_wav.unlink(missing_ok=True)
            return out_path, duration_ms
        else:
            return tmp_wav, duration_ms
