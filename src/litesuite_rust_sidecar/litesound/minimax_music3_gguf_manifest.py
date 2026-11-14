"""MiniMax-Music3 GGUF quant-profile manifest for LiteSound.

Runs on audio.cpp (0xShug0/audio.cpp, release-0.6.1+), the pure-C++ ggml
runtime — no Python venv, no torch, so it can never repeat the
stable-audio-small incident of replacing the cu128 nightly.

Weights: huggingface.co/audio-cpp/MiniMax-Music3-GGUF. Component design
mirrors ACE's: shared condition encoder + vocoder + configs + tokenizer,
per-quant language model / flow transformer / RVQ depth decoder. Every
profile coexists in ONE models dir — audiocpp selects components per session
via `--session-option language_model_gguf=<file>` etc., verified against the
binary's own model_specs/minimax_music3.json rather than a blog's rendering
of it. Sizes are exact bytes from the HF tree API; the installer verifies
them.

Benchmarks from the model card, measured on an RTX 5090: the
Q4 mix peaks at 9.8 GiB VRAM at 1.35x realtime — it coexists with a resident
qwen instead of evicting it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ID = "audio-cpp/MiniMax-Music3-GGUF"
BACKEND_ID = "minimax-music3-gguf"
FAMILY = "minimax_music3"

#: audio.cpp release this manifest was verified against.
RUNTIME_RELEASE = "release-0.6.1"
RUNTIME_ZIP = "audiocpp-windows-cuda-balance-26dcb5c4.zip"
RUNTIME_URL = (
    "https://github.com/0xShug0/audio.cpp/releases/download/"
    f"{RUNTIME_RELEASE}/{RUNTIME_ZIP}"
)


@dataclass(frozen=True)
class GgufFile:
    name: str
    size: int


#: Shared by every profile (constant across quants). Sizes: HF tree API.
SHARED_FILES: tuple[GgufFile, ...] = (
    GgufFile("condition_encoder.gguf", 100_677_184),
    GgufFile("vocoder.gguf", 216_704_192),
    GgufFile("config/condition_encoder.json", 292),
    GgufFile("config/language_model.json", 1_596),
    GgufFile("config/rvq_depth_decoder.json", 274),
    GgufFile("config/transformer.json", 294),
    GgufFile("config/vocoder.json", 251),
    GgufFile("tokenizer/tokenizer.json", 11_423_801),
    GgufFile("tokenizer/tokenizer_config.json", 377),
    GgufFile("config.json", 107),
)


@dataclass(frozen=True)
class GgufProfile:
    id: str
    label: str
    lm_file: str
    lm_size: int
    transformer_file: str
    transformer_size: int
    rvq_file: str
    rvq_size: int

    @property
    def files(self) -> tuple[GgufFile, ...]:
        return (
            GgufFile(self.lm_file, self.lm_size),
            GgufFile(self.transformer_file, self.transformer_size),
            GgufFile(self.rvq_file, self.rvq_size),
        ) + SHARED_FILES

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def session_options(self) -> dict:
        """The audiocpp --session-option values that select this profile."""
        return {
            "language_model_gguf": self.lm_file,
            "flow_transformer_gguf": self.transformer_file,
            "rvq_depth_decoder_gguf": self.rvq_file,
        }


# Ordered smallest -> max quality. The q4 mix is the model card's recommended
# default (lm+transformer Q4_0, rvq Q8_0): 9.8 GiB peak VRAM on a 5090.
PROFILES: tuple[GgufProfile, ...] = (
    GgufProfile("q4-default", "Q4 mix · Card default, 9.8 GiB VRAM",
                "language_model_q4_0.gguf", 6_006_866_496,
                "transformer_q4_0.gguf", 1_396_392_768,
                "rvq_depth_decoder_q8_0.gguf", 714_028_960),
    GgufProfile("q4k-balanced", "Q4_K mix · Balanced",
                "language_model_q4_k.gguf", 7_184_471_360,
                "transformer_q4_k.gguf", 1_396_397_600,
                "rvq_depth_decoder_bf16.gguf", 1_292_061_152),
    GgufProfile("q8-max", "Q8 mix · Max quality",
                "language_model_q8_0.gguf", 9_889_350_208,
                "transformer_q8_0.gguf", 2_606_842_688,
                "rvq_depth_decoder_bf16.gguf", 1_292_061_152),
)

DEFAULT_PROFILE_ID = "q4-default"


def get_profile(profile_id: Optional[str]) -> GgufProfile:
    for p in PROFILES:
        if p.id == (profile_id or DEFAULT_PROFILE_ID):
            return p
    return PROFILES[0]


def default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "minimax-music3-gguf"


def default_runtime_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "bin" / "runtime" / "audio.cpp"


def profile_installed(models_dir: Path | None = None,
                      profile: GgufProfile | str | None = None) -> bool:
    root = models_dir or default_models_dir()
    prof = profile if isinstance(profile, GgufProfile) else get_profile(profile)
    for file in prof.files:
        p = root / file.name
        if not p.is_file() or p.stat().st_size != file.size:
            return False
    return True


def runtime_installed(runtime_dir: Path | None = None) -> bool:
    root = runtime_dir or default_runtime_dir()
    return (root / "audiocpp_cli.exe").is_file()
