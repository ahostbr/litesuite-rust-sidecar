"""LiteSound Python audio engine."""
import inspect
import os
import sys
import asyncio
import tempfile
from pathlib import Path
from typing import Optional


def _ensure_user_env() -> None:
    """Give the process a user name in the environment when the launcher gave it none.

    getpass.getuser() reads LOGNAME/USER/LNAME/USERNAME and, when all four are empty,
    falls back to `import pwd`, which does not exist on Windows. torch's inductor cache
    (cache_dir_utils.default_cache_dir) calls it while `import torch._dynamo` runs, so an
    engine spawned from an environment without USERNAME (an agent's tool shell) could not
    load any diffusers backend. The retried import then reported the mask "Artifact of
    type=precompile already registered" instead. os.getlogin() asks Windows directly
    (GetUserNameW), not the environment. Every venv's engine runs this file, so this one
    call covers all of them and every subprocess they start.
    """
    if any(os.environ.get(k) for k in ("LOGNAME", "USER", "LNAME", "USERNAME")):
        return
    try:
        os.environ["USERNAME"] = os.getlogin()
    except OSError:
        pass  # no login session either; getuser() will still fail, loudly, at the import


_ensure_user_env()

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

PORT = int(os.environ.get("LITESOUND_PORT", 7428))

app = FastAPI(title="LiteSound Audio Engine", version="1.0.0")

# Global: one backend loaded at a time
_active_backend = None
_active_backend_id: Optional[str] = None
# The GGUF quant profile chosen in the Models panel — sticky, so every ace-step-gguf
# (re)load uses the user's selected quant rather than the default Q8.
_gguf_profile: Optional[str] = None


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LoadRequest(BaseModel):
    backend: str
    profile: Optional[str] = None


class AceStepOptions(BaseModel):
    bpm: Optional[int] = None
    keyScale: Optional[str] = None
    timeSignature: Optional[str] = None
    vocalLanguage: Optional[str] = None
    thinking: Optional[bool] = None
    inferenceSteps: Optional[int] = None
    guidanceScale: Optional[float] = None
    seed: Optional[int] = None
    useRandomSeed: Optional[bool] = None
    batchSize: Optional[int] = None
    taskType: Optional[str] = None
    inferMethod: Optional[str] = None

class YuEOptions(BaseModel):
    """YuE2's symbolic-planning controls.

    `cot` is the planning depth — "full", "melody" or "off" — and `abc` supplies
    a score instead of generating one. Both are validated by `SongRequest`
    itself (`yue2/protocol.py:90-102`), which is why neither is defaulted or
    re-checked here: a second copy of that vocabulary is a second thing to keep
    in step with the package.
    """
    cot: Optional[str] = None
    abc: Optional[str] = None


class SpeechOptions(BaseModel):
    reference_path: Optional[str] = None
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    voice: Optional[str] = None


class GenerateRequest(BaseModel):
    id: str
    mode: str  # music | song | sfx | ambient | speech
    prompt: str
    duration: Optional[int] = 30
    lyrics: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    backend: str = "musicgen"
    format: str = "mp3"
    aceStep: Optional[AceStepOptions] = None
    speech: Optional[SpeechOptions] = None
    yue: Optional[YuEOptions] = None


class DownloadRequest(BaseModel):
    backend_id: str
    hf_token: Optional[str] = None
    profile: Optional[str] = None


# ---------------------------------------------------------------------------
# Core routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"ok": True, "backend": _active_backend_id}


@app.get("/status")
async def status():
    return {
        "ok": True,
        "status": "ready" if _active_backend is not None else "idle",
        "active_backend": _active_backend_id,
    }


@app.post("/backend/load")
async def load_backend(req: LoadRequest):
    global _active_backend, _active_backend_id, _gguf_profile

    if req.backend in ("ace-step-gguf", "minimax-music3-gguf") and req.profile:
        _gguf_profile = req.profile  # remember the selected quant globally

    if _active_backend_id == req.backend:
        # Same backend already loaded. For GGUF this allows a hot quant-profile
        # switch: ace-server stays up and only the per-request model filenames
        # change, so no reload is needed.
        if req.profile and hasattr(_active_backend, "set_profile"):
            await asyncio.to_thread(_active_backend.set_profile, req.profile)
            return {"ok": True, "backend": req.backend, "profile": req.profile}
        return {"ok": True, "backend": req.backend, "message": "already loaded"}

    if _active_backend is not None:
        try:
            _active_backend.unload()
        except Exception:
            pass
        _active_backend = None
        _active_backend_id = None

    backend = _create_backend(req.backend)
    if backend is None:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {req.backend}")
    if req.profile and hasattr(backend, "set_profile"):
        backend.set_profile(req.profile)

    try:
        await asyncio.to_thread(backend.load)
        _active_backend = backend
        _active_backend_id = req.backend
        return {"ok": True, "backend": req.backend, "profile": req.profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backend/unload")
async def unload_backend():
    global _active_backend, _active_backend_id
    if _active_backend is not None:
        try:
            _active_backend.unload()
        except Exception:
            pass
        _active_backend = None
        _active_backend_id = None
    return {"ok": True}


def _accepts_generate_kwarg(backend, name: str) -> bool:
    """Can this backend's generate() actually receive `name`?

    🔴 REPLACES `hasattr(_active_backend, 'generate')`, which was TRUE FOR ALL TEN
    BACKENDS — a predicate that cannot discriminate is not a guard. The optional
    kwargs below must only be passed to backends that declare them: musicgen, bark,
    audioldm2, stable-audio and yue declare neither `ace_step_options` nor **kwargs,
    so an unconditional extra argument is a TypeError -> HTTP 500, not audio.

    It stayed invisible because it is BACKEND-DEPENDENT. stable-audio-small,
    chatterbox and vibevoice declare **kwargs and swallow the extra argument, so the
    SAME request succeeded or 500'd purely on which backend happened to be loaded —
    which reads as "that backend is flaky", and nobody files a dispatch bug for that.
    scripts/litesound-cli.mjs attaches `aceStep` to EVERY request regardless of
    backend, so `generate --backend musicgen` was a live 500 from shipped code.
    """
    try:
        sig = inspect.signature(backend.generate)
    except (AttributeError, TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True  # **kwargs absorbs anything
        if param.name == name and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _unsupported_kwargs(backend, names) -> list:
    """Which of `names` this backend's generate() CANNOT receive. Empty means all can.

    The speech branch's `if req.speech:` guards on whether the REQUEST carried options,
    not on whether the BACKEND can take them — so it never had a backend guard at all.
    That is a DIFFERENT defect from the ace_step_options guard beside it, which had a
    backend guard that could not say no. They sit adjacent and look alike, and the
    correct one lent the incorrect one credibility by proximity.

    Sending speech options to musicgen/bark/audioldm2/stable-audio/yue reached
    generate() with an argument it declares no parameter and no **kwargs for, raising
    TypeError, which the handler reported as HTTP 500 — the caller's mistake blamed on
    the server, with a stack-trace message.

    This does NOT silently drop the options. Discarding them would trade a loud failure
    for a silent one: the request would succeed and the user would get audio that
    ignored the reference clip they supplied. Rejected-with-a-reason and
    accepted-with-a-lie are not the same answer, and neither is 500-with-a-stack-trace.
    """
    return sorted(n for n in names if not _accepts_generate_kwarg(backend, n))


@app.post("/generate")
async def generate(req: GenerateRequest):
    global _active_backend, _active_backend_id

    if _active_backend_id != req.backend:
        backend = _create_backend(req.backend)
        if backend is None:
            raise HTTPException(status_code=400, detail=f"Unknown backend: {req.backend}")
        if _active_backend is not None:
            try:
                _active_backend.unload()
            except Exception:
                pass
        try:
            await asyncio.to_thread(backend.load)
            _active_backend = backend
            _active_backend_id = req.backend
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load backend: {e}")

    try:
        kwargs = dict(
            prompt=req.prompt,
            mode=req.mode,
            duration=req.duration or 30,
            lyrics=req.lyrics,
            genre=req.genre,
            mood=req.mood,
            fmt=req.format,
            job_id=req.id,
        )
        if req.aceStep and _accepts_generate_kwarg(_active_backend, "ace_step_options"):
            ace = req.aceStep
            kwargs["ace_step_options"] = {
                k: v for k, v in {
                    "bpm": ace.bpm,
                    "key_scale": ace.keyScale,
                    "time_signature": ace.timeSignature,
                    "vocal_language": ace.vocalLanguage,
                    "thinking": ace.thinking,
                    "inference_steps": ace.inferenceSteps,
                    "guidance_scale": ace.guidanceScale,
                    "seed": ace.seed,
                    "use_random_seed": ace.useRandomSeed,
                    "batch_size": ace.batchSize,
                    "task_type": ace.taskType,
                    "infer_method": ace.inferMethod,
                }.items() if v is not None
            }
        if req.yue and _accepts_generate_kwarg(_active_backend, "yue_options"):
            # Nested like aceStep rather than flattened like speech: `abc` is a
            # whole score and `cot` is a planning mode, and neither name means
            # anything to another backend.
            yue = req.yue
            yue_options = {
                k: v for k, v in {"cot": yue.cot, "abc": yue.abc}.items() if v is not None
            }
            if yue_options:
                kwargs["yue_options"] = yue_options
        if req.speech:
            # Flattened onto kwargs (not nested like aceStep) because the speech
            # backends read them via kwargs.get("reference_path").
            sp = req.speech
            speech_kwargs = {
                k: v for k, v in {
                    "reference_path": sp.reference_path,
                    "exaggeration": sp.exaggeration,
                    "cfg_weight": sp.cfg_weight,
                    "voice": sp.voice,
                }.items() if v is not None
            }
            # Derived from the dict just built, not from a second list of option names —
            # one source, so the check cannot drift from what is actually sent.
            unsupported = _unsupported_kwargs(_active_backend, speech_kwargs)
            if unsupported:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Backend '{req.backend}' cannot accept speech option(s): "
                        f"{', '.join(unsupported)}. Load a speech-capable backend, or "
                        f"send the request without them."
                    ),
                )
            kwargs.update(speech_kwargs)
        output_path, duration_ms = await asyncio.to_thread(
            _active_backend.generate, **kwargs,
        )
        return {"ok": True, "output_path": str(output_path), "duration_ms": duration_ms}
    except HTTPException:
        # 🔴 A deliberate 4xx MUST survive this block. HTTPException IS an Exception, so
        # without this clause the `except Exception` below catches it and re-raises it as
        # a 500 carrying the 400's own message as its detail — a precise, correct client
        # error downgraded into a generic server error. Every 4xx raised inside this try
        # was already affected; the speech guard is simply the first one to raise any.
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# HuggingFace Hub routes
# ---------------------------------------------------------------------------

@app.get("/hub/backends")
async def hub_backends():
    """List all downloadable backends with install status."""
    try:
        from .hub_downloader import list_available
        backends = await asyncio.to_thread(list_available)
        return {"ok": True, "backends": backends}
    except Exception as e:
        return {"ok": False, "error": str(e), "backends": []}


@app.post("/hub/download")
async def hub_download(req: DownloadRequest):
    """Start a background download for a backend."""
    from .hub_downloader import start_download
    try:
        dl_id = await asyncio.to_thread(start_download, req.backend_id, req.hf_token, req.profile)
        return {"ok": True, "download_id": dl_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hub/download/{dl_id}")
async def hub_download_status(dl_id: str):
    """Poll download progress."""
    from .hub_downloader import get_download
    dl = get_download(dl_id)
    if dl is None:
        raise HTTPException(status_code=404, detail="Download not found")
    return {"ok": True, **dl}


@app.get("/hub/downloads")
async def hub_downloads():
    """List all active/recent downloads."""
    from .hub_downloader import list_downloads
    return {"ok": True, "downloads": list_downloads()}


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _create_backend(backend_id: str):
    if backend_id == "musicgen":
        from .backends.musicgen import MusicGenBackend
        return MusicGenBackend()
    if backend_id == "stable-audio":
        from .backends.stable_audio import StableAudioBackend
        return StableAudioBackend(small=False)
    if backend_id == "stable-audio-small":
        # NOT StableAudioBackend(small=True): the small repo is stable-audio-tools
        # format with no model_index.json, so a diffusers pipeline cannot load it.
        # It also needs its own interpreter. See backends/stable_audio_small.py.
        from .backends.stable_audio_small import StableAudioSmallBackend
        return StableAudioSmallBackend()
    if backend_id == "bark":
        from .backends.bark import BarkBackend
        return BarkBackend()
    if backend_id == "yue":
        from .backends.yue import YuEBackend
        return YuEBackend()
    if backend_id == "audioldm2":
        from .backends.audioldm2 import AudioLDM2Backend
        return AudioLDM2Backend()
    if backend_id == "ace-step":
        from .backends.ace_step import AceStepBackend
        return AceStepBackend()
    if backend_id == "ace-step-gguf":
        from .backends.ace_step_gguf import AceStepGgufBackend
        return AceStepGgufBackend(_gguf_profile)
    if backend_id == "minimax-music3-gguf":
        from .backends.minimax_music3_gguf import MiniMaxMusic3GgufBackend
        return MiniMaxMusic3GgufBackend(_gguf_profile)
    if backend_id == "chatterbox":
        from .backends.chatterbox import ChatterboxBackend
        return ChatterboxBackend()
    if backend_id == "vibevoice":
        from .backends.vibevoice import VibeVoiceBackend
        return VibeVoiceBackend()
    return None


