"""HuggingFace Hub model downloader with progress tracking."""
import os
import time
import threading
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from .ace_step_gguf_manifest import (
    PROFILES as ACE_GGUF_PROFILES,
    DEFAULT_PROFILE_ID as ACE_GGUF_DEFAULT_PROFILE_ID,
    REPO_ID as ACE_GGUF_REPO_ID,
    TOTAL_BYTES as ACE_GGUF_TOTAL_BYTES,
    get_profile as ace_gguf_get_profile,
    default_models_dir as default_ace_gguf_models_dir,
    profile_installed as ace_gguf_profile_installed,
)
from . import minimax_music3_gguf_manifest as mm3_manifest

#: GGUF-profile backends resolve manifest functions through this table. The
#: download branch used to call ACE's functions by name — the exact shape by
#: which a second GGUF backend would have silently downloaded ACE's files
#: into ACE's directory while reporting success.
GGUF_MANIFESTS = {
    "ace-step-gguf": {
        "get_profile": ace_gguf_get_profile,
        "models_dir": default_ace_gguf_models_dir,
        "installed": ace_gguf_profile_installed,
        "profiles": ACE_GGUF_PROFILES,
        "default_id": ACE_GGUF_DEFAULT_PROFILE_ID,
    },
    "minimax-music3-gguf": {
        "get_profile": mm3_manifest.get_profile,
        "models_dir": mm3_manifest.default_models_dir,
        "installed": mm3_manifest.profile_installed,
        "profiles": mm3_manifest.PROFILES,
        "default_id": mm3_manifest.DEFAULT_PROFILE_ID,
    },
}


#: Characters that make an allow_pattern a GLOB rather than a literal filename.
#: Module-scope so tests assert against THIS value instead of keeping their own copy.
#: tests/test_backend_registry.py held a duplicate "*?[" — identical at the time, but a
#: duplicated constant is a drift that has not happened yet: add "{" here for brace
#: globs and the test's copy stays behind, so test_probe_is_always_a_concrete_filename
#: keeps passing while "{a,b}.json" sails through as a concrete filename and install
#: detection breaks the same silent way it broke before.
GLOB_CHARS = "*?["


def _default_acestep_repo() -> str:
    return str(Path(__file__).resolve().parents[2] / "bin" / "runtime" / "ACE-Step-1.5")


def probe_file_for(info: Dict[str, Any]) -> str:
    """Which cached file proves this backend is installed.

    ROOT-CAUSE NOTE. Install detection used to hardcode `config.json`. That silently
    broke for any repo that does not publish one (ResembleAI/chatterbox does not): the
    Models panel reported "Not installed" forever, so the user re-downloaded gigabytes
    on every click and nothing ever looked wrong from the outside.

    Hardcoding a per-backend override just moves the trap — the override can drift out
    of the set of files the install actually fetches. So the probe is DERIVED from
    `allow_patterns` when they exist, making them a single source of truth that cannot
    disagree with itself. `probe_file` stays available for repos that declare no
    patterns and do not publish config.json.

    The invariant "the probe is one of the downloaded files" is enforced offline by
    tests/test_backend_registry.py and against the live repos by
    scripts/audit_backend_registry.py.
    """
    explicit = info.get("probe_file")
    if explicit:
        return explicit
    for pattern in info.get("allow_patterns") or []:
        # Only a literal filename can be probed; skip globs.
        if not any(ch in pattern for ch in GLOB_CHARS):
            return pattern
    return "config.json"


# Maps backend IDs to HF repo IDs and local snapshot dirs
BACKEND_MODELS: Dict[str, Dict[str, Any]] = {
    # size_gb is what snapshot_download REALLY transfers, measured against the live repos
    # on 2026-08-07 by scripts/audit_backend_registry.py. Every figure here used to be a
    # guess and every one understated reality by 2.1x-4.1x — bark claimed 5GB and pulls
    # 20.68GB. These repos ship the same weights several times over (.ckpt + .safetensors
    # + .bin, fp16 + fp32), so "model size" and "download size" are very different numbers.
    # Run the audit after touching any entry here; it fails on drift.
    "musicgen": {
        "repo_id": "facebook/musicgen-medium",
        "repo_type": "model",
        "description": "MusicGen Medium (1.5B) — text-to-music",
        "size_gb": 11.14,
    },
    "musicgen-large": {
        "repo_id": "facebook/musicgen-large",
        "repo_type": "model",
        "description": "MusicGen Large (3.3B) — highest quality",
        "size_gb": 19.07,
    },
    "stable-audio": {
        "repo_id": "stabilityai/stable-audio-open-1.0",
        "repo_type": "model",
        # Diffusers repo: StableAudioPipeline.from_pretrained reads model_index.json and
        # the component subfolders. There is no config.json at the root, so the old
        # hardcoded probe reported "not installed" forever and the user re-downloaded
        # 14.6GB on every click.
        "probe_file": "model_index.json",
        # gated=auto on the Hub: the user must accept the licence on the model page AND
        # have an HF token in Settings, or snapshot_download 403s with no explanation.
        "gated": True,
        "description": "Stable Audio Open 1.0 — 47s stereo 44.1kHz",
        "size_gb": 14.60,
    },
    "stable-audio-small": {
        "repo_id": "stabilityai/stable-audio-open-small",
        "repo_type": "model",
        # ⚠️ This repo is stable-audio-tools format (model_config.json + .ckpt), NOT
        # diffusers — it has no model_index.json at all. See KNOWN_BROKEN below.
        "probe_file": "model_config.json",
        "gated": True,
        "description": "Stable Audio Open Small — 11s, <2GB VRAM",
        "size_gb": 4.69,
    },
    # 🔴 YuE2 IS TWO REPOS AND BOTH ARE REQUIRED. `YuE2Pipeline.from_pretrained`
    # defaults to `model="m-a-p/YuE2-3B"` AND `vae="m-a-p/YuE2-Vae"`
    # (`yue2/pipeline.py:168`). Registering only the model would report the
    # download complete and then fetch 0.53 GB more on first generate — or fail
    # outright on a machine that is offline by then. Sizes read from the HF API
    # on 2026-09-17, not estimated: 7,295,775,491 and 530,979,xxx bytes.
    "yue2": {
        "repo_id": "m-a-p/YuE2-3B",
        "repo_type": "model",
        "probe_file": "config.json",
        "description": "YuE2-3B — full songs with vocals, 48kHz stereo, Apache 2.0",
        "size_gb": 7.30,
    },
    "yue2-vae": {
        "repo_id": "m-a-p/YuE2-Vae",
        "repo_type": "model",
        "description": "YuE2 VAE — required by YuE2-3B, fetched separately",
        "size_gb": 0.53,
    },
    "bark": {
        "repo_id": "suno/bark",
        "repo_type": "model",
        "description": "Bark — speech, music, SFX, MIT license",
        "size_gb": 20.68,
    },
    "audioldm2": {
        "repo_id": "cvssp/audioldm2",
        "repo_type": "model",
        # Diffusers repo — same root-probe problem as stable-audio.
        "probe_file": "model_index.json",
        "description": "AudioLDM2 — text-to-audio via latent diffusion",
        "size_gb": 8.34,
    },
    "ace-step": {
        "repo_id": "ACE-Step/ACE-Step-v1-5-turbo",
        "repo_type": "model",
        "description": "ACE-Step 1.5 Turbo — full song+music generation. MIT license.",
        "size_gb": 9.0,
        "external": True,
    },
    "ace-step-gguf": {
        "repo_id": ACE_GGUF_REPO_ID,
        "repo_type": "model",
        "description": "ACE-Step GGUF Turbo 4B — native acestep.cpp, selectable quant. MIT license.",
        "size_gb": round(ACE_GGUF_TOTAL_BYTES / (1024 ** 3), 2),
        "gguf_profile": True,
    },
    "minimax-music3-gguf": {
        "repo_id": mm3_manifest.REPO_ID,
        "repo_type": "model",
        "description": (
            "MiniMax-Music3 GGUF — full 5-minute songs from lyrics + style caption, "
            "audio.cpp runtime, Q4 mix 9.8 GiB VRAM. MiniMax Community License."
        ),
        "size_gb": round(mm3_manifest.get_profile(None).total_bytes / (1024 ** 3), 2),
        "gguf_profile": True,
    },
    # Speech backends. Weights install normally from HF; their PYTHON PACKAGES cannot
    # share python/.venv (chatterbox pins torch==2.6.0 which drops sm_120/RTX 5090
    # CUDA; vibevoice downgrades transformers to 4.51.3). See
    # audio_engine/vendor/vibevoice/PROVENANCE.md for the measured resolution.
    "chatterbox": {
        "repo_id": "ResembleAI/chatterbox",
        "repo_type": "model",
        # The full repo is 22.02GB — it ships .pt AND .safetensors duplicates plus
        # multilingual variants. ChatterboxTTS.from_pretrained only fetches these five
        # (measured 2026-08-07: 2.97GB in the HF cache after a real load), so the
        # install is pattern-limited to match. Without allow_patterns, snapshot_download
        # would pull 19GB the runtime never opens.
        "allow_patterns": [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ],
        # No probe_file needed: this repo publishes no config.json, but the probe is
        # derived from allow_patterns above, so detection cannot drift from the install.
        "description": "Chatterbox — zero-shot single-speaker voice cloning. MIT license. Needs an isolated python env.",
        "size_gb": 2.97,
    },
    "vibevoice": {
        # 🔴 Realtime-0.5B, NOT VibeVoice-1.5B. The vendored loader is pinned to commit
        # 94da20d9, where the only inference-capable class is
        # VibeVoiceStreamingForConditionalGenerationInference — which is exactly what
        # this repo's config declares (model_type: vibevoice_streaming). 1.5B declares
        # VibeVoiceForConditionalGeneration, whose inference subclass Microsoft removed
        # at that commit, so 1.5B's 5.04GB would download and then be unloadable.
        "repo_id": "microsoft/VibeVoice-Realtime-0.5B",
        "repo_type": "model",
        # Measured 2026-08-07: 2,035,332,888 bytes model.safetensors, ~1.90 GB total.
        "description": "VibeVoice Realtime 0.5B — long-form speech. MIT license. Loader vendored, see vendor/vibevoice/PROVENANCE.md. Needs an isolated python env.",
        "size_gb": 1.90,
    },
}

# ✅ FIXED 2026-08-07. `stable-audio-small` had been broken since it was added:
# backends/stable_audio.py called StableAudioPipeline.from_pretrained() for BOTH
# variants, but `stabilityai/stable-audio-open-small` is stable-audio-tools format
# (8 flat files, model_config.json + .ckpt, NO model_index.json), so a diffusers
# pipeline could never load it. It now has its own backend
# (backends/stable_audio_small.py) on its own interpreter, python/.venv-stableaudio,
# because stable-audio-tools pins torch==2.13.0 and would clobber the cu128 nightly.
#
# Remaining user action: both stable-audio repos are gated (see "gated" below).

EXTERNAL_DIRS: Dict[str, str] = {
    "ace-step": os.environ.get("LITESOUND_ACESTEP_DIR", _default_acestep_repo()),
}

_downloads: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def _check_external_installed(backend_id: str) -> bool:
    repo_dir = EXTERNAL_DIRS.get(backend_id)
    if not repo_dir:
        return False
    return os.path.isdir(repo_dir) and os.path.isfile(os.path.join(repo_dir, "pyproject.toml"))


def list_available() -> list:
    """Return all downloadable backends with install status."""
    from huggingface_hub import try_to_load_from_cache
    results = []
    for backend_id, info in BACKEND_MODELS.items():
        gguf_profiles = None
        if info.get("external"):
            installed = _check_external_installed(backend_id)
        elif info.get("gguf_profile"):
            mani = GGUF_MANIFESTS[backend_id]
            md = mani["models_dir"]()
            gguf_profiles = [{
                "id": p.id,
                "label": p.label,
                "size_gb": round(p.total_bytes / (1024 ** 3), 2),
                "installed": mani["installed"](md, p),
                "default": p.id == mani["default_id"],
            } for p in mani["profiles"]]
            installed = any(pp["installed"] for pp in gguf_profiles)
        else:
            try:
                cached = try_to_load_from_cache(info["repo_id"], probe_file_for(info))
                installed = cached is not None and cached != "no_cached_file"
            except Exception:
                installed = False

        entry = {
            "id": backend_id,
            "repo_id": info["repo_id"],
            "description": info["description"],
            "size_gb": info["size_gb"],
            "installed": installed,
            # Surfaced so the UI can say WHY an install will fail before the user
            # clicks. A gated repo 403s with no actionable message otherwise.
            "gated": bool(info.get("gated")),
        }
        if gguf_profiles is not None:
            entry["profiles"] = gguf_profiles
        results.append(entry)
    return results


def _format_speed(downloaded: int, started_at: float) -> str:
    elapsed = max(time.time() - started_at, 0.001)
    speed_bps = downloaded / elapsed if downloaded else 0
    if speed_bps >= 1024 * 1024:
        return f"{speed_bps / (1024 * 1024):.1f} MB/s"
    if speed_bps >= 1024:
        return f"{speed_bps / 1024:.1f} KB/s"
    return f"{speed_bps:.0f} B/s"


def _profile_downloaded_bytes(models_dir: Path, files) -> int:
    downloaded = 0
    for f in files:
        target = models_dir / f.name
        if target.exists():
            downloaded += min(target.stat().st_size, f.size)
    return downloaded


def _update_download_progress(dl_id: str, downloaded: int, total: int, started_at: float) -> None:
    pct = int(downloaded / total * 100) if total > 0 else 0
    with _lock:
        _downloads[dl_id].update({
            "progress": min(pct, 100),
            "downloadedBytes": downloaded,
            "totalBytes": total,
            "speed": _format_speed(downloaded, started_at),
        })


def start_download(backend_id: str, hf_token: Optional[str] = None,
                   profile: Optional[str] = None) -> str:
    """Start async download. Returns download job ID."""
    if backend_id not in BACKEND_MODELS:
        raise ValueError(f"Unknown backend: {backend_id}")

    dl_id = str(uuid.uuid4())[:8]
    with _lock:
        _downloads[dl_id] = {
            "id": dl_id,
            "backendId": backend_id,
            "profile": profile,
            "status": "pending",
            "progress": 0,
            "downloadedBytes": 0,
            "totalBytes": 0,
            "speed": "",
            "startedAt": int(time.time() * 1000),
        }

    def _run():
        info = BACKEND_MODELS[backend_id]
        with _lock:
            _downloads[dl_id]["status"] = "downloading"

        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            from huggingface_hub.utils import HfHubHTTPError

            token = hf_token or os.environ.get("HF_TOKEN")
            started_at = time.time()
            if info.get("gguf_profile"):
                mani = GGUF_MANIFESTS[backend_id]
                prof = mani["get_profile"](profile)
                files = prof.files
                total = prof.total_bytes
                models_dir = mani["models_dir"]()
                models_dir.mkdir(parents=True, exist_ok=True)
                _update_download_progress(dl_id, _profile_downloaded_bytes(models_dir, files), total, started_at)

                for profile_file in files:
                    target = models_dir / profile_file.name
                    if target.exists() and target.stat().st_size != profile_file.size:
                        target.unlink()

                    if not target.exists():
                        hf_hub_download(
                            repo_id=info["repo_id"],
                            repo_type=info.get("repo_type", "model"),
                            filename=profile_file.name,
                            token=token,
                            local_dir=str(models_dir),
                            local_dir_use_symlinks=False,
                        )

                    actual_size = target.stat().st_size if target.exists() else 0
                    if actual_size != profile_file.size:
                        raise RuntimeError(
                            f"{profile_file.name} size mismatch: expected {profile_file.size}, got {actual_size}"
                        )
                    _update_download_progress(dl_id, _profile_downloaded_bytes(models_dir, files), total, started_at)

                if not mani["installed"](models_dir, prof):
                    raise RuntimeError(
                        f"{backend_id} profile did not pass final size verification")

                with _lock:
                    _downloads[dl_id].update({
                        "status": "complete",
                        "progress": 100,
                        "downloadedBytes": total,
                        "totalBytes": total,
                        "speed": "",
                        "completedAt": int(time.time() * 1000),
                    })
                return

            progress_state = {
                "downloaded": 0,
                "total": 0,
                "last_pct": -1,
            }

            def progress_callback(downloaded: int, total: int):
                pct = int(downloaded / total * 100) if total > 0 else 0
                if pct == progress_state["last_pct"] and downloaded != total:
                    return
                progress_state["last_pct"] = pct
                elapsed = max(time.time() - started_at, 0.001)
                speed_bps = downloaded / elapsed if downloaded else 0
                if speed_bps >= 1024 * 1024:
                    speed_str = f"{speed_bps / (1024 * 1024):.1f} MB/s"
                elif speed_bps >= 1024:
                    speed_str = f"{speed_bps / 1024:.1f} KB/s"
                else:
                    speed_str = f"{speed_bps:.0f} B/s"
                with _lock:
                    _downloads[dl_id].update({
                        "progress": pct,
                        "downloadedBytes": downloaded,
                        "totalBytes": total,
                        "speed": speed_str,
                    })

            class ProgressTqdm:
                def __init__(self, *args, total=None, initial=0, **kwargs):
                    self.total = int(total or 0)
                    self.n = int(initial or 0)
                    if self.total:
                        progress_state["total"] += self.total
                    progress_state["downloaded"] += self.n
                    progress_callback(progress_state["downloaded"], progress_state["total"])

                def update(self, n=1):
                    self.n += n
                    progress_state["downloaded"] += n
                    progress_callback(progress_state["downloaded"], progress_state["total"])

                def close(self):
                    progress_callback(progress_state["downloaded"], progress_state["total"])

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    self.close()
                    return False

            snapshot_download(
                repo_id=info["repo_id"],
                repo_type=info.get("repo_type", "model"),
                token=token,
                tqdm_class=ProgressTqdm,
                allow_patterns=info.get("allow_patterns"),
            )

            with _lock:
                _downloads[dl_id].update({
                    "status": "complete",
                    "progress": 100,
                    "downloadedBytes": progress_state["downloaded"],
                    "totalBytes": progress_state["total"],
                    "speed": "",
                    "completedAt": int(time.time() * 1000),
                })
        except Exception as e:
            msg = str(e)
            # A gated 403 is a USER ACTION, not a bug — say what the action is instead
            # of surfacing a raw HTTP error the user cannot act on.
            if info.get("gated") and ("401" in msg or "403" in msg or "gated" in msg.lower()):
                msg = (
                    f"{info['repo_id']} is a gated model. Accept the licence at "
                    f"https://huggingface.co/{info['repo_id']} and set your HuggingFace "
                    f"token in Settings, then retry."
                )
            with _lock:
                _downloads[dl_id].update({
                    "status": "error",
                    "error": msg,
                    "completedAt": int(time.time() * 1000),
                })

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return dl_id


def get_download(dl_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return dict(_downloads[dl_id]) if dl_id in _downloads else None


def list_downloads() -> list:
    with _lock:
        return [dict(d) for d in _downloads.values()]
