"""VibeVoice backend (Microsoft). MIT license.

Long-form expressive speech via the vendored first-party loader.

⚠️ CHECKPOINT: `microsoft/VibeVoice-Realtime-0.5B`, NOT VibeVoice-1.5B.
The vendored code is pinned to commit 94da20d9, where the only inference-capable
class is `VibeVoiceStreamingForConditionalGenerationInference` — and Realtime-0.5B's
config declares exactly that architecture (`model_type: vibevoice_streaming`).
VibeVoice-1.5B declares `VibeVoiceForConditionalGeneration`, whose inference subclass
Microsoft REMOVED from the repo at this commit; only the base class remains and it
cannot generate. Pointing this backend at 1.5B will not work.

⚠️ INTERPRETER: runs in `python/.venv-vibevoice` (transformers 4.51.3), not the main
venv. See vendor/vibevoice/PROVENANCE.md.

Batch semantics: the class is named "streaming", but no streamer is passed and the
complete result is consumed before returning. No chunks are yielded and no streaming
transport is introduced — the contract stays a file on disk, exactly as planned.

VOICES ARE NOT WAVS. Unlike Chatterbox, VibeVoice conditions on a pre-computed `.pt`
prefill tensor, fetched by the repo's demo/download_experimental_voices.sh. Handing
this backend a reference wav is a usage error and is reported as one.
"""
import io
import os
import sys
import time
import copy
import pickletools
import tempfile
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Set, Tuple

from ..model_paths import vibevoice_voices_dir

# Symbols a voice prefill is allowed to reference. Anything else and we refuse to load.
#
# WHY THIS EXISTS: the prefills are pickles, and torch's `weights_only=True` cannot read
# them — not because they are malicious, but because transformers' ModelOutput subclasses
# OrderedDict, and the safe unpickler's SETITEMS only accepts EXACT dict/OrderedDict/
# Counter. Rather than blindly flipping weights_only=False (arbitrary code execution on a
# downloaded file), we statically enumerate every GLOBAL the pickle imports and refuse
# unless all of them are inert data containers. That is strictly stronger than the flag.
#
# Audited 2026-08-07 across all 11 en1/en2 prefills: exactly these five, nothing else.
_PICKLE_ALLOWLIST: Set[Tuple[str, str]] = {
    ("collections", "OrderedDict"),
    ("torch", "BFloat16Storage"),
    ("torch", "HalfStorage"),
    ("torch", "FloatStorage"),
    ("torch", "LongStorage"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("transformers.cache_utils", "DynamicCache"),
    ("transformers.modeling_outputs", "BaseModelOutputWithPast"),
}


def audit_prefill_pickle(path: Path) -> Set[Tuple[str, str]]:
    """Enumerate every symbol a .pt prefill imports; raise if any is not allowlisted.

    Returns the set found, so callers can log exactly what was approved.
    """
    found: Set[Tuple[str, str]] = set()
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.endswith("data.pkl")]
        if not members:
            raise ValueError(f"{path.name} is not a torch archive (no data.pkl)")
        for member in members:
            for op, arg, _pos in pickletools.genops(io.BytesIO(z.read(member))):
                if op.name == "GLOBAL":
                    mod, _, name = str(arg).partition(" ")
                    found.add((mod, name))
                elif op.name == "STACK_GLOBAL":
                    # Module/name come off the stack, so they cannot be read statically.
                    # Refuse rather than guess — this is the opcode an attacker would use.
                    raise ValueError(
                        f"{path.name} uses STACK_GLOBAL, which cannot be audited "
                        f"statically. Refusing to load."
                    )
    bad = found - _PICKLE_ALLOWLIST
    if bad:
        raise ValueError(
            f"{path.name} references non-allowlisted symbols {sorted(bad)}. "
            f"Refusing to load — a voice prefill should contain only tensor data."
        )
    return found

# vendor/ holds the `vibevoice` package copied from the pinned commit, so putting
# vendor/ (not vendor/vibevoice/) on sys.path makes `import vibevoice` resolve to it.
_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"

_MODEL_ID = os.environ.get("LITESOUND_VIBEVOICE_MODEL", "microsoft/VibeVoice-Realtime-0.5B")


def _voices_dir() -> Path:
    return vibevoice_voices_dir()


_INSTALL_HINT = (
    "VibeVoice's loader is not importable. It is vendored at "
    "audio_engine/vendor/vibevoice (pinned commit 94da20d9) and needs the isolated "
    "python/.venv-vibevoice interpreter (transformers 4.51.3) — it will NOT run on "
    "the main venv. See vendor/vibevoice/PROVENANCE.md. Do NOT install the `vibevoice` "
    "package from PyPI: it is an unaffiliated third-party backup, not Microsoft's."
)


class VibeVoiceBackend:
    def __init__(self):
        self._model = None
        self._processor = None
        self._model_id = _MODEL_ID
        self._device = "cuda"

    def load(self):
        print(f"[VibeVoice] Loading {self._model_id}...")
        import torch

        if _VENDOR_DIR.is_dir() and str(_VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(_VENDOR_DIR))

        try:
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_streaming_processor import (
                VibeVoiceStreamingProcessor,
            )
        except ImportError as e:
            raise RuntimeError(f"{_INSTALL_HINT} (import failed: {e})") from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = VibeVoiceStreamingProcessor.from_pretrained(self._model_id)
        self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
            device_map=self._device,
        )
        self._model.eval()
        # Diffusion head steps. 5 is the first-party demo's setting; fewer steps
        # trade quality for speed and this is the balance Microsoft ships.
        self._model.set_ddpm_inference_steps(num_steps=5)
        print(f"[VibeVoice] Loaded on {self._device}.")

    def unload(self):
        import gc, torch
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[VibeVoice] Unloaded.")

    def _resolve_voice(self, kwargs) -> Optional[Path]:
        """Locate the .pt prefill for the requested voice, if any."""
        reference = kwargs.get("reference_path")
        if reference:
            path = Path(reference)
            if path.suffix.lower() != ".pt":
                raise ValueError(
                    f"VibeVoice conditions on a pre-computed .pt voice prefill, not an "
                    f"audio file (got {path.name}). Fetch prefills with the repo's "
                    f"demo/download_experimental_voices.sh and point at one of those, "
                    f"or use the chatterbox backend to clone directly from a wav."
                )
            if not path.is_file():
                raise FileNotFoundError(f"Voice prefill not found: {path}")
            return path

        voice = kwargs.get("voice")
        if voice:
            path = _voices_dir() / f"{voice}.pt"
            if not path.is_file():
                available = sorted(p.stem for p in _voices_dir().glob("*.pt")) if _voices_dir().is_dir() else []
                raise FileNotFoundError(
                    f"Voice {voice!r} not found in {_voices_dir()}. "
                    f"Available: {available or '(none installed)'}"
                )
            return path
        return None

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
        if self._model is None or self._processor is None:
            raise RuntimeError("VibeVoice not loaded")

        import torch

        # `duration` is an input for music backends but an OUTPUT for speech — the
        # text decides the clip length. Deliberately unused.
        text = prompt or lyrics or ""
        if not text.strip():
            raise ValueError("VibeVoice requires non-empty text to speak")

        voice_path = self._resolve_voice(kwargs)
        prefilled = None
        if voice_path is not None:
            # Static audit FIRST, then load. weights_only=True cannot read these files
            # (transformers' ModelOutput is an OrderedDict subclass and the safe
            # unpickler's SETITEMS rejects subclasses), so the protection is moved
            # earlier: every imported symbol is enumerated and allowlisted before any
            # unpickling happens. See audit_prefill_pickle above.
            audit_prefill_pickle(voice_path)
            prefilled = torch.load(
                str(voice_path), map_location=self._device, weights_only=False
            )

        print(f"[VibeVoice] Generating: {text[:80]!r}"
              + (f" (voice {voice_path.stem})" if voice_path else " (no voice prefill)"))
        t0 = time.time()

        inputs = self._processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=prefilled,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self._device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=float(kwargs.get("cfg_weight") or 1.3),
            tokenizer=self._processor.tokenizer,
            generation_config={"do_sample": False},
            all_prefilled_outputs=copy.deepcopy(prefilled) if prefilled is not None else None,
        )
        # duration_ms is elapsed GENERATION time across this codebase (see bark.py,
        # musicgen.py), not the length of the audio. Kept consistent deliberately.
        duration_ms = int((time.time() - t0) * 1000)

        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            raise RuntimeError("VibeVoice produced no audio for this input")

        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        self._processor.save_audio(outputs.speech_outputs[0], output_path=str(tmp_wav))

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
