"""Stable Audio Open backend (Stability AI). Non-commercial community license."""
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple


class StableAudioBackend:
    def __init__(self, small: bool = False):
        self._pipe = None
        self._model_id = "stabilityai/stable-audio-open-small" if small else "stabilityai/stable-audio-open-1.0"
        self._max_duration = 11 if small else 47
        self._sampling_rate = 44100

    def load(self):
        print(f"[StableAudio] Loading {self._model_id}...")
        import torch
        from diffusers import StableAudioPipeline
        self._pipe = StableAudioPipeline.from_pretrained(self._model_id, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            self._pipe = self._pipe.to("cuda")
        print("[StableAudio] Loaded.")

    def unload(self):
        import gc, torch
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[StableAudio] Unloaded.")

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
        if self._pipe is None:
            raise RuntimeError("StableAudio not loaded")

        import torch
        import soundfile as sf

        # Cap at model max
        duration_s = min(duration, self._max_duration)
        full_prompt = prompt
        if mood:
            full_prompt = f"{full_prompt}, {mood}"

        print(f"[StableAudio] Generating: {full_prompt!r} ({duration_s}s)")
        t0 = time.time()

        with torch.no_grad():
            output = self._pipe(
                full_prompt,
                negative_prompt="low quality, noise, distortion",
                num_inference_steps=100,
                audio_end_in_s=float(duration_s),
                num_waveforms_per_prompt=1,
            )

        audio = output.audios[0]  # shape: (channels, samples)
        duration_ms = int((time.time() - t0) * 1000)

        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        # soundfile, not torchaudio.save: torchaudio 2.11 delegates writing to torchcodec
        # (not installed; same fix as stable_audio_small.py). (channels, samples) -> (frames, channels).
        sf.write(str(tmp_wav), audio.cpu().float().numpy().T, self._sampling_rate, subtype="PCM_16")

        return _transcode(tmp_wav, fmt), duration_ms


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
