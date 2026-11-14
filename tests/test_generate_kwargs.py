"""The /generate kwarg contract for EVERY backend. No GPU, no model load, no network.

WHY THIS FILE EXISTS. test_speech_backends.py covers the speech branch of server.py's
kwarg assembly and proves its conditional is load-bearing. The OTHER branch —
`ace_step_options` — had no coverage at all, and neither did four of the ten backends:

    ace-step, ace-step-gguf, stable-audio-small, yue were never dispatch-tested.

ace-step-gguf is the DEFAULT backend in scripts/litesound-cli.mjs (sfxPayload hardcodes
it) and the one PitBorn's SFX generation actually calls, so the most-used generation
path in the repo had zero signature coverage while five less-used ones had it.

Signature-level only, exactly like the speech tests: inspect.signature(...).bind()
never calls generate(), so nothing loads a model.

SERVER_KWARGS is IMPORTED from test_speech_backends rather than copied. It already
mirrors server.py's handler by hand; a second copy would be a second thing to drift.
"""
import inspect
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
AUDIO_ENGINE = SRC / "litesuite_rust_sidecar" / "litesound"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from litesuite_rust_sidecar.litesound.server import _accepts_generate_kwarg, _create_backend  # noqa: E402
from test_speech_backends import SERVER_KWARGS  # noqa: E402

#: Every id the app can dispatch. server.py's factory is the single source of truth for
#: what /backend/load will accept, so an id missing here is an id that 400s in the UI.
ALL_BACKENDS = (
    "musicgen", "stable-audio", "stable-audio-small", "bark", "audioldm2",
    "yue", "ace-step", "ace-step-gguf", "chatterbox", "vibevoice",
)

#: The only two that declare `ace_step_options`. Measured from the real signatures.
ACE_FAMILY = ("ace-step", "ace-step-gguf")

#: Declare neither `ace_step_options` nor `**kwargs`, so an unconditional extra kwarg
#: is a TypeError for them — a 500 from /generate, not a 400.
ACE_REJECTORS = ("musicgen", "bark", "audioldm2", "stable-audio", "yue")

#: Cannot take `yue_options` either — no such parameter and no `**kwargs`. MEASURED,
#: not assumed: stable-audio-small, chatterbox and vibevoice DO declare `**kwargs`
#: and therefore absorb it, which is why this list is not "everything but yue".
YUE_OPTION_REJECTORS = (
    "musicgen", "stable-audio", "bark", "audioldm2", "ace-step", "ace-step-gguf",
)


class TestEveryBackendDispatches(unittest.TestCase):
    """Closes a real gap: four of ten ids were never dispatch-tested."""

    def test_all_ten_dispatch(self):
        for backend_id in ALL_BACKENDS:
            with self.subTest(backend=backend_id):
                self.assertIsNotNone(
                    _create_backend(backend_id),
                    f"{backend_id} is offered by the Models panel but does not dispatch",
                )

    def test_all_ten_expose_the_backend_contract(self):
        for backend_id in ALL_BACKENDS:
            with self.subTest(backend=backend_id):
                backend = _create_backend(backend_id)
                for method in ("load", "unload", "generate"):
                    self.assertTrue(
                        callable(getattr(backend, method, None)),
                        f"{backend_id}.{method} is missing or not callable",
                    )

    def test_all_ten_accept_the_base_server_kwargs(self):
        """The 8 keys server.py ALWAYS passes must bind for every backend."""
        for backend_id in ALL_BACKENDS:
            with self.subTest(backend=backend_id):
                inspect.signature(_create_backend(backend_id).generate).bind(**SERVER_KWARGS)

    # CONTROL for all three above. Each asserts that something IS present, and would
    # be satisfied by a factory that returned some permissive stub for any string.
    def test_the_factory_still_rejects_unknown_ids(self):
        for bogus in ("nope", "", "ace-step-ggml", "musicgen-large"):
            with self.subTest(backend=bogus):
                self.assertIsNone(
                    _create_backend(bogus),
                    f"{bogus!r} must not dispatch — /backend/load answers 400 on None",
                )


class TestAceStepOptionsContract(unittest.TestCase):
    """server.py:177 adds `ace_step_options` to kwargs whenever req.aceStep is set."""

    def test_the_ace_family_accepts_ace_step_options(self):
        for backend_id in ACE_FAMILY:
            with self.subTest(backend=backend_id):
                sig = inspect.signature(_create_backend(backend_id).generate)
                sig.bind(**SERVER_KWARGS, ace_step_options={"seed": 11001, "inference_steps": 8})

    def test_backends_outside_the_ace_family_would_REJECT_it(self):
        """Proves the conditional around that kwarg has to be genuine.

        🔴 AND IT CURRENTLY IS NOT. server.py:177 reads

            if req.aceStep and hasattr(_active_backend, 'generate'):

        and `hasattr(backend, 'generate')` is true for ALL TEN backends — the predicate
        is permanently true, so it constrains nothing. The speech branch below it is
        guarded correctly (`if req.speech:`), and test_speech_backends proves that one
        load-bearing. This one is decorative.

        That is reachable from shipped code, not theoretical: litesound-cli.mjs's
        sfxPayload() attaches `aceStep` to EVERY request regardless of backend, and
        api-server.ts forwards it. So `generate --backend musicgen` sends aceStep ->
        server adds ace_step_options -> MusicGenBackend.generate() raises TypeError ->
        HTTP 500 instead of audio.

        This test asserts the SIGNATURES, which are correct and should not change. It
        does not bless the guard. Fixing the guard (match on the parameter rather than
        on `hasattr(..., 'generate')`) leaves every assertion here green.
        """
        for backend_id in ACE_REJECTORS:
            with self.subTest(backend=backend_id):
                sig = inspect.signature(_create_backend(backend_id).generate)
                sig.bind(**SERVER_KWARGS)  # the base set binds fine
                with self.assertRaises(TypeError):
                    sig.bind(**SERVER_KWARGS, ace_step_options={"seed": 1})

    def test_kwargs_absorbing_backends_tolerate_it(self):
        """stable-audio-small, chatterbox and vibevoice declare **kwargs, so they
        silently swallow the extra argument instead of raising. Recorded because it
        means the 500 above is backend-dependent — the same request succeeds or fails
        based on which backend is loaded, which is why it stayed unnoticed."""
        for backend_id in ("stable-audio-small", "chatterbox", "vibevoice"):
            with self.subTest(backend=backend_id):
                sig = inspect.signature(_create_backend(backend_id).generate)
                sig.bind(**SERVER_KWARGS, ace_step_options={"seed": 1})


class TestTheGuardCanDiscriminate(unittest.TestCase):
    """The guard itself, not just the signatures it is supposed to respect.

    🔴 THESE ARE THE TESTS THAT WERE RED BEFORE THE FIX. server.py used to read

        if req.aceStep and hasattr(_active_backend, 'generate'):

    and `hasattr(backend, 'generate')` is true for all ten backends, so the predicate
    returned True for every one of them. test_the_guard_refuses_backends_that_cannot
    _take_it failed for all five rejectors — which is the whole defect, stated as an
    assertion instead of as a 500 in a user's log.

    A guard is only a guard if it can say NO. Asserting the signatures alone could
    never have caught this: the signatures were always correct, it was the decision
    about them that was broken.
    """

    def test_the_guard_admits_the_ace_family(self):
        for backend_id in ACE_FAMILY:
            with self.subTest(backend=backend_id):
                self.assertTrue(
                    _accepts_generate_kwarg(_create_backend(backend_id), "ace_step_options"),
                    f"{backend_id} declares ace_step_options and must still receive it",
                )

    def test_the_guard_refuses_backends_that_cannot_take_it(self):
        """RED under the old `hasattr` predicate, for all five of these."""
        for backend_id in ACE_REJECTORS:
            with self.subTest(backend=backend_id):
                self.assertFalse(
                    _accepts_generate_kwarg(_create_backend(backend_id), "ace_step_options"),
                    f"{backend_id} declares neither ace_step_options nor **kwargs — "
                    f"passing it means TypeError and HTTP 500 instead of audio",
                )

    def test_the_guard_admits_kwargs_absorbing_backends(self):
        """**kwargs genuinely can receive it, so refusing them would be over-correction."""
        for backend_id in ("stable-audio-small", "chatterbox", "vibevoice"):
            with self.subTest(backend=backend_id):
                self.assertTrue(
                    _accepts_generate_kwarg(_create_backend(backend_id), "ace_step_options"),
                )

    def test_the_predicate_discriminates_at_all(self):
        """The control that the OLD predicate could never have satisfied.

        `hasattr(backend, 'generate')` returns the same answer for every backend and
        every kwarg name. Requiring both answers to appear across the backend set is
        what makes this suite able to detect a non-discriminating predicate at all —
        each assertion above, taken alone, is satisfiable by a constant.
        """
        answers = {
            _accepts_generate_kwarg(_create_backend(b), "ace_step_options")
            for b in ALL_BACKENDS
        }
        self.assertEqual(answers, {True, False}, "the guard must return BOTH answers")

    def test_an_unknown_kwarg_is_refused_by_backends_without_kwargs(self):
        """Guards against a predicate that just returns True for any name."""
        self.assertFalse(
            _accepts_generate_kwarg(_create_backend("musicgen"), "totally_made_up_kwarg")
        )
        # ...but **kwargs still absorbs it, which is correct, not a miss.
        self.assertTrue(
            _accepts_generate_kwarg(_create_backend("chatterbox"), "totally_made_up_kwarg")
        )

    def test_a_backend_with_no_generate_is_refused_rather_than_raising(self):
        class NoGenerate:
            pass

        self.assertFalse(_accepts_generate_kwarg(NoGenerate(), "ace_step_options"))
        self.assertFalse(_accepts_generate_kwarg(None, "ace_step_options"))


class TestSpeechAndAceAreDisjoint(unittest.TestCase):
    """The two conditional branches must not be assumed to co-apply."""

    def test_the_ace_family_does_not_take_speech_kwargs(self):
        for backend_id in ACE_FAMILY:
            with self.subTest(backend=backend_id):
                sig = inspect.signature(_create_backend(backend_id).generate)
                with self.assertRaises(TypeError):
                    sig.bind(**SERVER_KWARGS, reference_path="C:/tmp/ref.wav")


if __name__ == "__main__":
    unittest.main()


class TestYuEOptionsContract(unittest.TestCase):
    """server.py adds `yue_options` to kwargs whenever req.yue is set (T777).

    The guard beside it is `_accepts_generate_kwarg(_active_backend, "yue_options")`,
    which is only load-bearing if some backend would REFUSE the kwarg. If every
    backend took it, the guard could be deleted and nothing would notice until a
    backend that cannot take it is added — the same shape as the ace_step_options
    guard above.
    """

    def test_yue_accepts_yue_options(self):
        sig = inspect.signature(_create_backend("yue").generate)
        sig.bind(**SERVER_KWARGS, yue_options={"cot": "melody"})
        self.assertTrue(_accepts_generate_kwarg(_create_backend("yue"), "yue_options"))

    def test_yue_NAMES_the_parameter_rather_than_absorbing_it(self):
        """Not via **kwargs — the name is what makes the option reviewable."""
        sig = inspect.signature(_create_backend("yue").generate)
        self.assertIn("yue_options", sig.parameters)
        self.assertFalse(
            any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()),
            "yue absorbs arbitrary kwargs, so accepting yue_options proves nothing",
        )

    def test_the_backends_without_kwargs_refuse_it(self):
        """CONTROL: the server guard has something to refuse.

        ⚠️ NOT "every backend but yue". THREE of them — stable-audio-small,
        chatterbox and vibevoice — declare `**kwargs`, and
        `_accepts_generate_kwarg` returns True for a VAR_KEYWORD by design
        (server.py:168-169), so they legitimately absorb anything. I wrote the
        stronger assertion first and it failed on all three; recording the real
        boundary is worth more than an arm that reads tidier.
        """
        for backend_id in YUE_OPTION_REJECTORS:
            with self.subTest(backend=backend_id):
                self.assertFalse(
                    _accepts_generate_kwarg(_create_backend(backend_id), "yue_options"),
                    f"{backend_id} claims yue_options — the server guard is then inert",
                )
