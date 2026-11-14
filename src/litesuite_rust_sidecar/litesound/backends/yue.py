"""YuE2 (HKUST / M-A-P). Apache 2.0 — commercial safe. Full songs with vocals.

⚠️ THIS REPLACES A PHASE-1 SCAFFOLD THAT COULD NEVER HAVE RUN. The previous
version shelled out to a v1 `infer.py` inside `vendor/YuE`, a checkout that has
never existed in this repository — so the backend has never generated a note.
Worse, three different places disagreed about where that checkout would live:

    yue.py               <repo>/python/vendor/YuE
    engine-manager.ts    <repo>/vendor/YuE          (pythonDir/../vendor/YuE)
    the vendor dir that actually exists   <repo>/python/audio_engine/vendor/

so even a correct clone would have left one of the two gates saying
"unavailable". Keying availability on the INTERPRETER (below) deletes all three
paths rather than picking a winner among them.

⚠️ INTERPRETER: runs in `python/.venv-yue`, not the main venv. `yue2-infer`
pins `torch==2.10.0`; the main venv carries a cu128 nightly that other backends
depend on, and chatterbox pins `torch==2.6.0` which drops sm_120 (RTX 5090)
entirely. Same reason `.venv-stableaudio` and `.venv-vibevoice` exist.

⬜ WINDOWS, despite the model card saying Linux. MEASURED 2026-09-17 in
`python/.venv-yue`: python 3.12.9, `torch 2.10.0+cu128`, `torch.cuda.get_arch_list()`
includes `sm_120`, `YuE2Pipeline` imports. Every Windows-hostile import in the
package — `vllm`, `triton`, `fcntl` — lives in `yue2/fast.py`, which
`yue2/pipeline.py` imports INSIDE three functions, each guarded by
`self.backend == "vllm"`. The default is `backend="torch"` (pipeline.py:123), so
the supported path never touches them. Upstream's "Linux" is a support claim,
not a requirement — but nothing here has generated audio yet, so that remains
unverified past the import.

🔴 TWO REPOS, NOT ONE. `YuE2Pipeline.from_pretrained` takes
`model="m-a-p/YuE2-3B"` AND `vae="m-a-p/YuE2-Vae"` (pipeline.py:168). Pulling
only the model leaves the VAE to be fetched silently on first generate — or to
fail outright on a machine with no network. Both are registered in
`hub_downloader.py`; 7.30 GB + 0.53 GB.
"""
import re
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple

_MODEL_ID = "m-a-p/YuE2-3B"
_VAE_ID = "m-a-p/YuE2-Vae"

_INSTALL_HINT = (
    "yue2 is not importable. YuE2 cannot run on the main venv — it pins "
    "torch==2.10.0, which would replace the cu128 build every other backend "
    "depends on. It needs python/.venv-yue; see "
    "python/audio_engine/vendor/vibevoice/PROVENANCE.md for the recipe."
)

class YuEBackend:
    def __init__(self):
        self._pipeline = None

    def load(self):
        print(f"[YuE2] Loading {_MODEL_ID} (+ {_VAE_ID})...")
        try:
            from yue2.pipeline import YuE2Pipeline
        except ImportError as e:
            raise RuntimeError(f"{_INSTALL_HINT} (import failed: {e})") from e

        self._pipeline = YuE2Pipeline.from_pretrained(_MODEL_ID, device="cuda")
        print("[YuE2] Ready.")

    def unload(self):
        import gc
        import torch
        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[YuE2] Unloaded.")

    def generate(
        self,
        prompt: str,
        mode: str,
        duration: int = 120,
        lyrics: Optional[str] = None,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        fmt: str = "mp3",
        job_id: str = "unknown",
        yue_options: Optional[dict] = None,
    ) -> Tuple[Path, int]:
        if self._pipeline is None:
            raise RuntimeError("YuE2 not loaded")

        options = yue_options or {}

        # `style` is this pipeline's word for what LiteSound calls the prompt.
        # `genre` and `mood` are separate LiteSound fields with no argument of
        # their own here, so they join the style rather than being dropped.
        style = ", ".join(p for p in (prompt, genre, mood) if p)

        # ⬜ `cot` AND `abc` ARE NOT DEFAULTED HERE. `SongRequest` defaults
        # `cot="full"` and `abc=None` and validates both (`protocol.py:90-102`);
        # repeating either would be a second copy of its vocabulary, free to
        # drift. Only what the caller actually chose is passed.
        request = {}
        if options.get("cot") is not None:
            request["cot"] = options["cot"]
        if options.get("abc"):
            request["abc"] = options["abc"]

        # 🔴 `id` IS VALIDATED AGAINST A FILENAME-SAFE PATTERN and raises if it
        # is not (`protocol.py:97`). LiteSound job ids are ours to shape, but a
        # request that dies on the id after the weights are resident would cost
        # a full load to report a naming problem.
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", job_id):
            request["id"] = job_id

        print(f"[YuE2] Generating: cot={request.get('cot', 'full')} "
              f"abc={'provided' if request.get('abc') else 'generated'}")
        t0 = time.time()

        song = self._pipeline(style, lyrics or "", **request)
        duration_ms = int((time.time() - t0) * 1000)

        out_dir = Path(tempfile.mkdtemp())
        # ⚠️ `SongResult.save` ACCEPTS ONLY .wav/.flac and raises on anything
        # else (`pipeline.py:98`) — MP3 is a delivery conversion, exactly as it
        # is for every other backend here.
        wav_path = out_dir / "song.wav"
        song.save(wav_path)

        # The symbolic plan is the thing that makes this model editable; it
        # costs a line to say it exists rather than discarding it silently.
        if song.abc:
            print(f"[YuE2] ABC score: {len(song.abc)} chars")

        if fmt == "wav":
            return wav_path, duration_ms

        out = wav_path.with_suffix(f".{fmt}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), str(out)],
            check=True,
            capture_output=True,
        )
        wav_path.unlink(missing_ok=True)
        return out, duration_ms
