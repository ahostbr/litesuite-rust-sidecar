"""Stable Audio Open Small (Stability AI). Non-commercial community license.

⚠️ WHY THIS IS A SEPARATE BACKEND FROM `stable_audio.py`.

`stabilityai/stable-audio-open-small` is NOT a diffusers repo. It is
stable-audio-tools format: 8 flat files, `model_config.json` + `.ckpt`, and no
`model_index.json` anywhere. `StableAudioPipeline.from_pretrained` therefore cannot
load it — the shared `StableAudioBackend(small=True)` path has been broken since the
day it was added, failing at load regardless of whether the weights were present.

It is also a distilled (adversarial post-trained) model with its own sampler, so it is
not just a smaller checkpoint behind the same API: it needs `steps=8`, `cfg_scale=1.0`
and `sampler_type="pingpong"`, per the model card. Folding that into the diffusers
backend would have meant two unrelated code paths in one class.

⚠️ INTERPRETER: runs in `python/.venv-stableaudio`, not the main venv.
`stable-audio-tools` pulls `torch==2.13.0`, which would replace the cu128 nightly and
drop sm_120 (RTX 5090) support for every other backend. Recipe in
audio_engine/vendor/vibevoice/PROVENANCE.md.

⚠️ GATED WEIGHTS: the repo is `gated=auto`. The user must accept the licence on the
model page AND have an HF token configured, or the download 403s.
"""
import random
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# Largest value stable-audio-tools' seed path can hold.
_MAX_SEED = 2 ** 32 - 2

_MODEL_ID = "stabilityai/stable-audio-open-small"

_INSTALL_HINT = (
    "stable-audio-tools is not importable. Stable Audio Small cannot run on the main "
    "venv — its dependencies pin torch==2.13.0, which would drop sm_120 (RTX 5090) "
    "support for every other backend. It needs python/.venv-stableaudio; see "
    "python/audio_engine/vendor/vibevoice/PROVENANCE.md for the setup recipe."
)


class StableAudioSmallBackend:
    def __init__(self):
        self._model = None
        self._config = None
        self._device = "cuda"
        # The model generates variable-length audio up to 11s at 44.1kHz stereo.
        self._max_duration = 11

    def load(self):
        print(f"[StableAudioSmall] Loading {_MODEL_ID}...")
        import torch
        try:
            from stable_audio_tools import get_pretrained_model
        except ImportError as e:
            raise RuntimeError(f"{_INSTALL_HINT} (import failed: {e})") from e

        try:
            self._model, self._config = get_pretrained_model(_MODEL_ID)
        except Exception as e:
            # The single most likely failure, and the one with a user action attached.
            if "gated" in str(e).lower() or "403" in str(e):
                raise RuntimeError(
                    f"{_MODEL_ID} is a gated repo. Accept the licence at "
                    f"https://huggingface.co/{_MODEL_ID} and set your HuggingFace token "
                    f"in Settings, then retry."
                ) from e
            raise

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        print(f"[StableAudioSmall] Loaded on {self._device}.")

    def unload(self):
        import gc, torch
        self._model = None
        self._config = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[StableAudioSmall] Unloaded.")

    def generate(
        self,
        prompt: str,
        mode: str,
        duration: int = 11,
        lyrics: Optional[str] = None,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        fmt: str = "mp3",
        job_id: str = "unknown",
        **kwargs,
    ) -> Tuple[Path, int]:
        if self._model is None or self._config is None:
            raise RuntimeError("StableAudioSmall not loaded")

        import torch
        import soundfile as sf
        from einops import rearrange
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        sample_rate = self._config["sample_rate"]
        sample_size = self._config["sample_size"]

        duration_s = min(duration, self._max_duration)
        full_prompt = prompt
        if genre:
            full_prompt = f"{genre} {full_prompt}"
        if mood:
            full_prompt = f"{full_prompt}, {mood}"

        # An explicit seed is REQUIRED on Windows, not just nice to have.
        # stable_audio_tools/inference/generation.py does, for seed == -1:
        #     seed = np.random.randint(0, 2**32 - 1)     # no dtype=
        # On Windows numpy's default integer is int32, so that raises
        # "ValueError: high is out of bounds for int32" and generation never starts.
        # (The sibling generate_diffusion_uncond passes dtype=np.uint32 and is fine —
        # this one was simply missed upstream.) Supplying a seed skips that branch
        # entirely, and makes reruns reproducible for a narration/SFX slate.
        seed = kwargs.get("seed")
        seed = int(seed) if seed is not None else random.randint(0, _MAX_SEED)

        print(f"[StableAudioSmall] Generating: {full_prompt!r} ({duration_s}s, seed={seed})")
        t0 = time.time()

        output = generate_diffusion_cond(
            self._model,
            # These three are not tunable defaults — this is a distilled model and the
            # model card specifies exactly this configuration. cfg_scale=1.0 means
            # guidance is effectively off, which is what post-training expects.
            steps=8,
            cfg_scale=1.0,
            sampler_type="pingpong",
            conditioning=[{"prompt": full_prompt, "seconds_total": duration_s}],
            sample_size=sample_size,
            seed=seed,
            device=self._device,
        )
        duration_ms = int((time.time() - t0) * 1000)

        # (batch, channels, samples) -> (channels, batch*samples)
        output = rearrange(output, "b d n -> d (b n)")
        # Peak-normalise, clip, and emit int16 PCM — matching the model card, and
        # incidentally the interchange-friendly format for game audio.
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        # soundfile, not torchaudio.save: torchaudio 2.11 delegates writing to
        # torchcodec ("TorchCodec is required for save_with_torchcodec"), which is not
        # installed and whose compatibility with this torch nightly is unverified.
        # soundfile is already a dependency here and takes (frames, channels), so the
        # (channels, samples) tensor is transposed. int16 in -> PCM_16 out.
        sf.write(str(tmp_wav), output.numpy().T, sample_rate, subtype="PCM_16")

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
