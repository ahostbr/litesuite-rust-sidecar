"""C3: `envsetup --dry-run <env>` prints the exact `uv pip` commands in order and runs none of them.

The expected steps of speech, vibevoice and stableaudio are those of LiteSound's
scripts/setup_speech_envs.py (SPECS, at LiteSound develop 1bf0a1a), written out here because that
script is not part of this package; the commit that added this test compared them with the script
itself.
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from litesuite_rust_sidecar import envsetup  # noqa: E402

NIGHTLY = "https://download.pytorch.org/whl/nightly/cu128"
SIDECAR = ["install", "fastapi", "uvicorn"]
EXPECTED = {
    "speech": [
        ["install", "--pre", "torch", "torchaudio", "--index-url", NIGHTLY],
        ["install", "--no-deps", "chatterbox-tts"],
        ["install", "numpy<2.0.0,>=1.24.0", "librosa==0.11.0", "s3tokenizer", "transformers==5.2.0", "diffusers==0.29.0", "resemble-perth>=1.0.0",
         "conformer==0.3.2", "safetensors==0.5.3", "spacy-pkuseg", "pykakasi==2.3.0", "pyloudnorm", "omegaconf", "scipy"],
        SIDECAR,
    ],
    "vibevoice": [
        ["install", "--pre", "torch", "torchaudio", "--index-url", NIGHTLY],
        ["install", "transformers==4.51.3", "diffusers", "librosa", "soundfile", "tqdm", "accelerate"],
        SIDECAR,
    ],
    "stableaudio": [
        ["install", "stable-audio-tools"],
        ["install", "--pre", "--force-reinstall", "--no-deps", "torch", "torchaudio", "--index-url", NIGHTLY],
        ["install", "--pre", "--force-reinstall", "--no-deps", "torchvision", "--index-url", NIGHTLY],
        ["install", "setuptools<81"],
        SIDECAR,
    ],
}


def as_uv(step, uv="UV", python="PY"):
    return [uv, "pip", step[0], "--python", python, *step[1:]]


class EnvSetupTest(unittest.TestCase):
    def test_the_speech_recipes_are_setup_speech_envs_steps(self):
        for env, steps in EXPECTED.items():
            self.assertEqual(envsetup.commands(env, "UV", "PY"), [as_uv(s) for s in steps], env)

    def test_dry_run_prints_the_commands_in_order_and_runs_nothing(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = envsetup.main(["stableaudio", "--uv", r"C:\no\such\uv.exe", "--dry-run"])
        self.assertEqual(code, 0)
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), len(EXPECTED["stableaudio"]))
        self.assertTrue(lines[0].startswith(r"C:\no\such\uv.exe pip install --python "), lines[0])
        self.assertIn("stable-audio-tools", lines[0])
        self.assertIn("setuptools<81", lines[3])

    def test_main_is_the_checkout_setup_torch_first_then_the_requirements(self):
        steps = envsetup.commands("main", "UV", "PY")
        self.assertEqual(steps[0], as_uv(["install", "--pre", "torch", "torchaudio", "--index-url", NIGHTLY]))
        self.assertIn("audiocraft>=1.3.0", steps[1])
        self.assertIn("uvicorn[standard]>=0.34.0", steps[1])

    def test_the_torch_index_is_data(self):
        for env, found in envsetup.recipes().items():
            if found.get("checkout_only"):
                continue
            for step in found["steps"]:
                self.assertNotIn(NIGHTLY, step, f"{env}: the index belongs in torch_index")

    def test_yue_and_unknown_environments_are_refused(self):
        with self.assertRaises(SystemExit) as yue:
            envsetup.commands("yue", "UV", "PY")
        self.assertIn("checkout only", str(yue.exception))
        with self.assertRaises(SystemExit):
            envsetup.commands("nonesuch", "UV", "PY")


if __name__ == "__main__":
    unittest.main()
