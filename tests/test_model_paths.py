"""Path routing only: no model loading, downloads, or model file writes."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parent.parent / "src"
AUDIO_ENGINE = SRC / "litesuite_rust_sidecar" / "litesound"
sys.path.insert(0, str(SRC))

from litesuite_rust_sidecar.litesound.model_paths import models_dir, vibevoice_voices_dir
from litesuite_rust_sidecar.litesound.ace_step_gguf_manifest import default_models_dir, default_runtime_dir
from litesuite_rust_sidecar.litesound.hub_downloader import GGUF_MANIFESTS
from litesuite_rust_sidecar.litesound.backends.vibevoice import _voices_dir


class ModelPathsTests(unittest.TestCase):
    def test_existing_defaults_and_blank_fallback(self):
        with patch.dict(os.environ, {"LITESOUND_MODELS_DIR": "  ",
                                     "LITESOUND_ACESTEP_GGUF_MODELS_DIR": "",
                                     "LITESOUND_VIBEVOICE_VOICES_DIR": ""}):
            root = AUDIO_ENGINE.parents[1] / "bin" / "models"
            self.assertEqual(models_dir(), root)
            self.assertEqual(default_models_dir(), root / "ace-step-gguf")
            self.assertEqual(_voices_dir(), root / "vibevoice-voices")

    def test_relocated_root_and_consumers(self):
        root = Path("relocated-models").resolve()
        with patch.dict(os.environ, {"LITESOUND_MODELS_DIR": f"  {root}  ",
                                     "LITESOUND_ACESTEP_GGUF_MODELS_DIR": "",
                                     "LITESOUND_VIBEVOICE_VOICES_DIR": ""}):
            self.assertEqual(models_dir(), root)
            self.assertEqual(default_models_dir(), root / "ace-step-gguf")
            self.assertEqual(GGUF_MANIFESTS["ace-step-gguf"]["models_dir"](), default_models_dir())
            self.assertEqual(_voices_dir(), root / "vibevoice-voices")
            self.assertEqual(vibevoice_voices_dir(), _voices_dir())
            self.assertEqual(default_runtime_dir(), AUDIO_ENGINE.parents[1] / "bin" / "runtime" / "acestep.cpp")

    def test_dedicated_overrides_win_for_downloads_and_voices(self):
        with patch.dict(os.environ, {"LITESOUND_MODELS_DIR": "other-root",
                                     "LITESOUND_ACESTEP_GGUF_MODELS_DIR": "dedicated-ace",
                                     "LITESOUND_VIBEVOICE_VOICES_DIR": "dedicated-voices"}):
            self.assertEqual(default_models_dir(), Path("dedicated-ace"))
            self.assertEqual(GGUF_MANIFESTS["ace-step-gguf"]["models_dir"](), Path("dedicated-ace"))
            self.assertEqual(_voices_dir(), Path("dedicated-voices"))


if __name__ == "__main__":
    unittest.main()
