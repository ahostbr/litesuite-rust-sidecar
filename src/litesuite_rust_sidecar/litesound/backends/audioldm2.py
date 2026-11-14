"""AudioLDM2 backend via HuggingFace Diffusers."""
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple


class AudioLDM2Backend:
    def __init__(self):
        self._pipe = None
        self._model_id = "cvssp/audioldm2"
        self._sampling_rate = 16000

    def load(self):
        print(f"[AudioLDM2] Loading {self._model_id}...")
        import torch
        from diffusers import AudioLDM2Pipeline
        self._pipe = AudioLDM2Pipeline.from_pretrained(self._model_id, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            self._pipe = self._pipe.to("cuda")
        print("[AudioLDM2] Loaded.")

    def unload(self):
        import gc, torch
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[AudioLDM2] Unloaded.")

    def generate(
        self,
        prompt: str,
        mode: str,
        duration: int = 10,
        lyrics: Optional[str] = None,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        fmt: str = "mp3",
        job_id: str = "unknown",
    ) -> Tuple[Path, int]:
        if self._pipe is None:
            raise RuntimeError("AudioLDM2 not loaded")

        import torch, scipy.io.wavfile as wav, numpy as np

        audio_length = min(duration, 30)  # model supports ~10-30s
        full_prompt = prompt
        if mood:
            full_prompt = f"{full_prompt}, {mood}"

        print(f"[AudioLDM2] Generating: {full_prompt!r} ({audio_length}s)")
        t0 = time.time()

        with torch.no_grad():
            output = self._pipe(
                full_prompt,
                negative_prompt="low quality, average quality",
                num_inference_steps=200,
                audio_length_in_s=float(audio_length),
                num_waveforms_per_prompt=1,
            )

        audio = output.audios[0]
        duration_ms = int((time.time() - t0) * 1000)

        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        wav.write(str(tmp_wav), self._sampling_rate, audio.astype(np.float32))

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
