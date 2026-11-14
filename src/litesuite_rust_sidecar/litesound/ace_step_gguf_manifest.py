"""ACE-Step GGUF quant-profile manifest for LiteSound.

Every profile shares the embedding (Q8) + VAE (BF16) and differs only in the LM +
synth quant. ace-server loads models by filename per request, so all installed
profiles can coexist in one models dir and the active one is just a filename choice.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .model_paths import models_dir


REPO_ID = "Serveurperso/ACE-Step-1.5-GGUF"
BACKEND_ID = "ace-step-gguf"


@dataclass(frozen=True)
class GgufFile:
    name: str
    size: int


# Shared by every profile (constant across quants)
SHARED_FILES: tuple[GgufFile, ...] = (
    GgufFile("Qwen3-Embedding-0.6B-Q8_0.gguf", 784_144_960),
    GgufFile("vae-BF16.gguf", 337_420_928),
)


@dataclass(frozen=True)
class GgufProfile:
    id: str
    label: str
    lm_model: str
    synth_model: str
    lm_size: int
    synth_size: int

    @property
    def files(self) -> tuple[GgufFile, ...]:
        return (
            GgufFile(self.lm_model, self.lm_size),
            GgufFile(self.synth_model, self.synth_size),
        ) + SHARED_FILES

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


# Ordered max-quality -> smallest. Sizes are exact (the installer verifies them).
PROFILES: tuple[GgufProfile, ...] = (
    GgufProfile("q8-turbo-4b", "Q8 Turbo 4B · Max quality",
                "acestep-5Hz-lm-4B-Q8_0.gguf", "acestep-v15-turbo-Q8_0.gguf",
                4_457_323_648, 2_549_528_000),
    GgufProfile("q6-turbo-4b", "Q6_K Turbo 4B · Near-lossless",
                "acestep-5Hz-lm-4B-Q6_K.gguf", "acestep-v15-turbo-Q6_K.gguf",
                3_442_713_504, 1_970_472_064),
    GgufProfile("q5-turbo-4b", "Q5_K_M Turbo 4B · Balanced",
                "acestep-5Hz-lm-4B-Q5_K_M.gguf", "acestep-v15-turbo-Q5_K_M.gguf",
                3_025_965_984, 1_700_140_224),
    GgufProfile("q5q4-turbo-4b", "Q5/Q4 Turbo 4B · Smallest",
                "acestep-5Hz-lm-4B-Q5_K_M.gguf", "acestep-v15-turbo-Q4_K_M.gguf",
                3_025_965_984, 1_445_710_272),
)
# Smallest quant is the default: instrumental generation is ~indistinguishable
# from Q8 by ear, ~33% faster, ~31% smaller. Bump to Q8/Q6 via the dropdown for
# vocal/lyric work, where quantization hurts intelligibility more.
DEFAULT_PROFILE_ID = "q5q4-turbo-4b"


def get_profile(profile_id: Optional[str]) -> GgufProfile:
    target = profile_id or DEFAULT_PROFILE_ID
    for p in PROFILES:
        if p.id == target:
            return p
    return PROFILES[0]


def default_models_dir() -> Path:
    override = os.environ.get("LITESOUND_ACESTEP_GGUF_MODELS_DIR")
    return Path(override) if override else models_dir() / BACKEND_ID


def default_runtime_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "bin" / "runtime" / "acestep.cpp"


def profile_installed(models_dir: Path | None = None,
                      profile: GgufProfile | str | None = None) -> bool:
    root = models_dir or default_models_dir()
    prof = profile if isinstance(profile, GgufProfile) else get_profile(profile)
    for file in prof.files:
        path = root / file.name
        if not path.is_file() or path.stat().st_size != file.size:
            return False
    return True


def iter_profile_files(profile: GgufProfile | str | None = None) -> Iterable[GgufFile]:
    prof = profile if isinstance(profile, GgufProfile) else get_profile(profile)
    return prof.files


# ── Back-compat single-profile aliases (track the active default) ─────────────
_DEFAULT = get_profile(None)
PROFILE_ID = _DEFAULT.id
PROFILE_LABEL = _DEFAULT.label
TOTAL_BYTES = _DEFAULT.total_bytes
LM_MODEL = "acestep-5Hz-lm-4B-Q8_0"
SYNTH_MODEL = "acestep-v15-turbo-Q8_0"
LM_MODEL_FILE = _DEFAULT.lm_model
SYNTH_MODEL_FILE = _DEFAULT.synth_model
PROFILE_FILES = _DEFAULT.files
