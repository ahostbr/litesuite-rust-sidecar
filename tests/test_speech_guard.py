"""The speech-option guard on /generate, driven through the REAL route.

No GPU, no model, no network. The handler only loads a backend when the requested one
differs from the active one, so pinning server._active_backend / _active_backend_id to a
stub skips the load branch entirely and exercises the kwarg-assembly block for real —
including the actual HTTP status code, which a unit test on the helper cannot observe.

WHY THIS EXISTS. The speech branch was `if req.speech:` — a guard on whether the
REQUEST carried options, never on whether the BACKEND could take them. That is a
DIFFERENT defect from the ace_step_options guard four lines above it, which had a
backend guard that could not say no. The two sit adjacent and look alike, and the
correct one lent the incorrect one credibility by proximity.

Sending speech options to a backend that declares neither the parameter nor **kwargs
reached generate(), raised TypeError, and the handler reported it as HTTP 500 — the
caller's mistake blamed on the server, with a stack-trace message.

The fix does NOT discard unsupported options. Dropping them would trade a loud failure
for a silent one: the request would succeed and the user would receive audio that
ignored the reference clip they supplied. Rejected-with-a-reason and accepted-with-a-lie
are not the same answer, and neither is 500-with-a-stack-trace.
"""
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
AUDIO_ENGINE = SRC / "litesuite_rust_sidecar" / "litesound"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient  # noqa: E402

from litesuite_rust_sidecar.litesound import server  # noqa: E402


class StrictBackend:
    """Declares no speech parameters and no **kwargs — musicgen/bark/audioldm2/yue."""

    def load(self): ...
    def unload(self): ...

    def generate(self, prompt, mode, duration=30, lyrics=None, genre=None, mood=None,
                 fmt="mp3", job_id="unknown"):
        return Path("out.wav"), 1234


class SpeechBackend:
    """Declares **kwargs and reads the speech options from it — chatterbox/vibevoice."""

    def load(self): ...
    def unload(self): ...

    def generate(self, prompt, mode, duration=30, lyrics=None, genre=None, mood=None,
                 fmt="mp3", job_id="unknown", **kwargs):
        return Path("out.wav"), 4321


def body(**over):
    payload = {"id": "t1", "mode": "speech", "prompt": "welcome to the pit",
               "backend": "musicgen", "format": "wav"}
    payload.update(over)
    return payload


class SpeechGuardCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        self._saved = (server._active_backend, server._active_backend_id)

    def tearDown(self):
        server._active_backend, server._active_backend_id = self._saved

    def pin(self, backend, backend_id):
        """Make `backend` the active one so the route never loads a model."""
        server._active_backend = backend
        server._active_backend_id = backend_id


class TestGuardRefusesLoudly(SpeechGuardCase):
    def test_a_backend_that_cannot_take_speech_options_gets_400_not_500(self):
        self.pin(StrictBackend(), "musicgen")
        r = self.client.post("/generate", json=body(speech={"reference_path": "C:/ref.wav"}))
        self.assertEqual(
            r.status_code, 400,
            "500 would blame the server for the caller's mistake; the old code did exactly that",
        )

    def test_the_refusal_names_the_backend_AND_the_option(self):
        """A refusal that does not say what to change is only marginally better than a crash."""
        self.pin(StrictBackend(), "musicgen")
        r = self.client.post("/generate", json=body(speech={"reference_path": "C:/ref.wav"}))
        detail = r.json()["detail"]
        self.assertIn("musicgen", detail)
        self.assertIn("reference_path", detail)

    def test_every_unsupported_option_is_listed_not_just_the_first(self):
        self.pin(StrictBackend(), "musicgen")
        r = self.client.post("/generate", json=body(
            speech={"reference_path": "C:/ref.wav", "exaggeration": 0.5,
                    "cfg_weight": 0.5, "voice": "narrator"},
        ))
        detail = r.json()["detail"]
        for name in ("reference_path", "exaggeration", "cfg_weight", "voice"):
            self.assertIn(name, detail, f"{name} was sent and rejected but is not named")

    def test_the_options_are_REFUSED_not_silently_discarded(self):
        """The distinction the whole fix turns on.

        A 2xx here would mean the user got audio that ignored the reference clip they
        supplied — a silent wrong answer, which is worse than the 500 it replaced.
        """
        self.pin(StrictBackend(), "musicgen")
        r = self.client.post("/generate", json=body(speech={"reference_path": "C:/ref.wav"}))
        self.assertNotEqual(r.status_code, 200, "succeeding while dropping the options is the worst outcome")
        self.assertEqual(r.status_code, 400)


class TestGuardStillPermits(SpeechGuardCase):
    """CONTROLS. Every assertion above is satisfied by a route that refuses EVERYTHING,
    including one whose handler crashed before reaching the guard at all."""

    def test_a_speech_capable_backend_accepts_the_same_request(self):
        self.pin(SpeechBackend(), "chatterbox")
        r = self.client.post("/generate", json=body(
            backend="chatterbox", speech={"reference_path": "C:/ref.wav"},
        ))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["ok"])

    def test_a_strict_backend_with_NO_speech_options_still_works(self):
        """Proves the guard is scoped to the options and does not gate the route."""
        self.pin(StrictBackend(), "musicgen")
        r = self.client.post("/generate", json=body(mode="music"))
        self.assertEqual(r.status_code, 200, r.text)

    def test_an_empty_speech_object_is_not_a_rejection(self):
        """`speech: {}` requested nothing, so there is nothing the backend cannot take."""
        self.pin(StrictBackend(), "musicgen")
        r = self.client.post("/generate", json=body(speech={}))
        self.assertEqual(r.status_code, 200, r.text)


class TestThePredicateDiscriminates(unittest.TestCase):
    """The guard must be able to return BOTH answers. A predicate that cannot say no is
    not a guard, whatever it is named — the defect this branch's neighbour had."""

    def test_both_answers_occur_over_the_real_backend_set(self):
        answers = {
            bool(server._unsupported_kwargs(server._create_backend(b), ["reference_path"]))
            for b in ("musicgen", "bark", "audioldm2", "stable-audio", "yue",
                      "chatterbox", "vibevoice", "stable-audio-small")
        }
        self.assertEqual(answers, {True, False}, "the guard must both refuse and permit")

    def test_speech_backends_accept_every_speech_option(self):
        for backend_id in ("chatterbox", "vibevoice"):
            with self.subTest(backend=backend_id):
                self.assertEqual(
                    server._unsupported_kwargs(
                        server._create_backend(backend_id),
                        ["reference_path", "exaggeration", "cfg_weight", "voice"],
                    ),
                    [],
                )

    def test_strict_backends_accept_none_of_them(self):
        for backend_id in ("musicgen", "bark", "audioldm2", "stable-audio", "yue"):
            with self.subTest(backend=backend_id):
                self.assertEqual(
                    server._unsupported_kwargs(
                        server._create_backend(backend_id), ["reference_path", "voice"],
                    ),
                    ["reference_path", "voice"],
                )


class TestHttpExceptionSurvivesTheHandler(SpeechGuardCase):
    """`except Exception` in the handler catches HTTPException too — it IS an Exception.

    Without an `except HTTPException: raise` ahead of it, every deliberate 4xx raised
    inside that block is re-emitted as a 500 carrying the 4xx's own message as its
    detail. The speech guard is simply the first 4xx to be raised there, so this is
    asserted separately from the guard: they are two different fixes and either can
    regress alone.
    """

    def test_a_deliberate_400_is_not_downgraded_to_500(self):
        self.pin(StrictBackend(), "musicgen")
        r = self.client.post("/generate", json=body(speech={"reference_path": "C:/ref.wav"}))
        self.assertEqual(r.status_code, 400)
        self.assertNotIn(
            "400:", str(r.json().get("detail", "")),
            "a 500 wrapping a 400 shows up as the 400's message nested inside the 500's detail",
        )


if __name__ == "__main__":
    unittest.main()
