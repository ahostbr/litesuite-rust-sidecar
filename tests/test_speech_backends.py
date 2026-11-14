"""Contract tests for the speech backends. No GPU, no model load, no network.

Written against stdlib `unittest` rather than pytest: pytest is not installed in
python/.venv, and this venv is a CUDA-nightly build that should not gain dependencies
for a test file's sake. Run it with

    cd python && .venv/Scripts/python -m unittest discover -s audio_engine/tests -v

pytest collects unittest.TestCase classes natively, so this also works unchanged if
pytest is ever added.

These cover the STRUCTURAL criteria only. Whether the audio is any good is Ryan's ear,
by his explicit decision — no assertion here should ever be read as covering that.
"""
import inspect
import sys
import unittest
from pathlib import Path

# server.py and hub_downloader.py use flat imports (`from backends.x import Y`,
# `from ace_step_gguf_manifest import ...`), so audio_engine itself must be on sys.path.
SRC = Path(__file__).resolve().parent.parent / "src"
AUDIO_ENGINE = SRC / "litesuite_rust_sidecar" / "litesound"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from litesuite_rust_sidecar.litesound.server import _create_backend  # noqa: E402
from litesuite_rust_sidecar.litesound.hub_downloader import BACKEND_MODELS  # noqa: E402

SPEECH_BACKENDS = ("chatterbox", "vibevoice")

# Exactly what server.py's /generate handler always passes.
SERVER_KWARGS = {
    "prompt": "Welcome to the pit.",
    "mode": "speech",
    "duration": 15,
    "lyrics": None,
    "genre": None,
    "mood": None,
    "fmt": "wav",
    "job_id": "test",
}


class TestSpeechBackendDispatch(unittest.TestCase):
    def test_create_backend_returns_instance(self):
        for backend_id in SPEECH_BACKENDS:
            with self.subTest(backend=backend_id):
                backend = _create_backend(backend_id)
                self.assertIsNotNone(backend, f"{backend_id} did not dispatch")

    def test_exposes_backend_contract(self):
        for backend_id in SPEECH_BACKENDS:
            with self.subTest(backend=backend_id):
                backend = _create_backend(backend_id)
                for method in ("load", "unload", "generate"):
                    self.assertTrue(
                        callable(getattr(backend, method, None)),
                        f"{backend_id}.{method} is missing or not callable",
                    )

    def test_generate_accepts_full_server_kwarg_set(self):
        """Signature-level only — never calls generate(), so no model and no GPU."""
        for backend_id in SPEECH_BACKENDS:
            with self.subTest(backend=backend_id):
                sig = inspect.signature(_create_backend(backend_id).generate)
                sig.bind(**SERVER_KWARGS)

    def test_generate_accepts_speech_kwargs(self):
        """server.py flattens speech options onto kwargs when req.speech is set."""
        for backend_id in SPEECH_BACKENDS:
            with self.subTest(backend=backend_id):
                sig = inspect.signature(_create_backend(backend_id).generate)
                sig.bind(
                    **SERVER_KWARGS,
                    reference_path="C:/tmp/ref.wav",
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    voice="narrator",
                )

    def test_unknown_backend_returns_none(self):
        """Negative path: the dispatch must reject, not silently succeed.

        This is what makes /backend/load answer 400 instead of 500 — without it
        the other assertions here prove nothing about rejection.
        """
        self.assertIsNone(_create_backend("nope"))
        self.assertIsNone(_create_backend(""))


class TestSpeechBackendRegistry(unittest.TestCase):
    def test_registered_with_measured_size(self):
        for backend_id in SPEECH_BACKENDS:
            with self.subTest(backend=backend_id):
                self.assertIn(backend_id, BACKEND_MODELS)
                entry = BACKEND_MODELS[backend_id]
                self.assertTrue(entry["repo_id"])
                self.assertGreater(entry["size_gb"], 0)

    def test_repo_ids_are_the_verified_ones(self):
        """Guards against a plausible-looking edit to a mirror or a wrong org.

        `microsoft/VibeVoice-1.5B` in particular has a third-party PyPI impostor
        carrying Microsoft's own URLs — see vendor/vibevoice/PROVENANCE.md.
        """
        self.assertEqual(BACKEND_MODELS["chatterbox"]["repo_id"], "ResembleAI/chatterbox")
        # Realtime-0.5B, not 1.5B: the vendored loader at commit 94da20d9 only has the
        # streaming inference class, and only this repo declares that architecture.
        self.assertEqual(
            BACKEND_MODELS["vibevoice"]["repo_id"], "microsoft/VibeVoice-Realtime-0.5B"
        )


class TestExistingBackendsUnaffected(unittest.TestCase):
    """The speech kwargs are added conditionally precisely because these can't take them."""

    def test_music_backends_still_dispatch(self):
        for backend_id in ("musicgen", "bark", "audioldm2", "stable-audio"):
            with self.subTest(backend=backend_id):
                self.assertIsNotNone(_create_backend(backend_id))

    def test_bark_would_reject_speech_kwargs(self):
        """Proves the conditional in server.py is load-bearing, not decorative."""
        sig = inspect.signature(_create_backend("bark").generate)
        sig.bind(**SERVER_KWARGS)  # the base set binds fine
        with self.assertRaises(TypeError):
            sig.bind(**SERVER_KWARGS, reference_path="C:/tmp/ref.wav")


if __name__ == "__main__":
    unittest.main()
