"""YuE2's request assembly, driven against a fake pipeline. No GPU, no weights.

🔴 THE REAL PIPELINE CANNOT APPEAR IN A SUITE. `YuE2Pipeline.from_pretrained`
pulls 7.83 GB across two repos and puts a 3B model on the GPU, so a test that
could only run for real would test NOTHING. What is actually at risk here is the
TRANSLATION — LiteSound's request fields into `SongRequest`'s — and every bit of
that is reachable by handing the backend an object with the same shape.

⬜ THE FAKE IS SHAPED LIKE THE REAL THING WHERE IT MATTERS. `SongResult.save`
raises on anything but .wav/.flac (`yue2/pipeline.py:98`), so this one does too:
a fake that accepts an .mp3 path would let a backend ship that silently writes
nothing and calls it success.
"""
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
AUDIO_ENGINE = SRC / "litesuite_rust_sidecar" / "litesound"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from litesuite_rust_sidecar.litesound.backends.yue import YuEBackend  # noqa: E402


class FakeSong:
    def __init__(self):
        self.saved = None
        self.abc = "X:1\nT:fake\n"

    def save(self, path):
        path = Path(path)
        if path.suffix.lower() not in {".flac", ".wav"}:
            raise ValueError("Use .flac or .wav; MP3 is an optional delivery conversion")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")
        self.saved = path
        return str(path)


class FakePipeline:
    """Records the call the backend makes, and nothing else."""

    def __init__(self):
        self.style = None
        self.lyrics = None
        self.kwargs = None
        self.song = FakeSong()

    def __call__(self, style=None, lyrics=None, **kwargs):
        self.style, self.lyrics, self.kwargs = style, lyrics, kwargs
        return self.song


def rig():
    backend = YuEBackend()
    pipe = FakePipeline()
    backend._pipeline = pipe
    return backend, pipe


BASE = dict(prompt="dream pop", mode="song", fmt="wav", job_id="job1")


class TestRequestAssembly(unittest.TestCase):
    def test_prompt_genre_and_mood_all_reach_the_style(self):
        """None of the three is dropped. `style` is the only field that can carry them."""
        backend, pipe = rig()
        backend.generate(**{**BASE, "genre": "shoegaze", "mood": "melancholy"})
        self.assertEqual(pipe.style, "dream pop, shoegaze, melancholy")

    def test_absent_genre_and_mood_leave_no_empty_separators(self):
        backend, pipe = rig()
        backend.generate(**BASE)
        self.assertEqual(pipe.style, "dream pop")

    def test_lyrics_reach_the_pipeline(self):
        backend, pipe = rig()
        backend.generate(**{**BASE, "lyrics": "[verse]\nhello"})
        self.assertEqual(pipe.lyrics, "[verse]\nhello")

    def test_absent_lyrics_become_empty_not_none(self):
        """`SongRequest` raises TypeError on a non-string (`protocol.py:93`)."""
        backend, pipe = rig()
        backend.generate(**BASE)
        self.assertEqual(pipe.lyrics, "")

    def test_cot_and_abc_are_passed_through_when_chosen(self):
        backend, pipe = rig()
        backend.generate(**BASE, yue_options={"cot": "melody", "abc": "X:1\n"})
        self.assertEqual(pipe.kwargs.get("cot"), "melody")
        self.assertEqual(pipe.kwargs.get("abc"), "X:1\n")

    def test_unchosen_options_are_ABSENT_not_defaulted(self):
        """🔴 The package owns these defaults and validates them.

        `SongRequest` defaults `cot="full"` and `abc=None` (`protocol.py:84-86`).
        Sending our own copy would be a second statement of its vocabulary, free
        to drift the first time upstream changes one — and it would make "the
        user did not choose" indistinguishable from "the user chose full".
        """
        backend, pipe = rig()
        backend.generate(**BASE)
        self.assertNotIn("cot", pipe.kwargs)
        self.assertNotIn("abc", pipe.kwargs)

    def test_an_empty_abc_is_not_sent(self):
        """`SongRequest` raises on a blank one (`protocol.py:99`)."""
        backend, pipe = rig()
        backend.generate(**BASE, yue_options={"abc": ""})
        self.assertNotIn("abc", pipe.kwargs)


class TestJobIdBecomesAValidRequestId(unittest.TestCase):
    """`id` is validated against `[A-Za-z0-9][A-Za-z0-9_.-]{0,179}` (protocol.py:97)."""

    def test_a_safe_job_id_is_used(self):
        backend, pipe = rig()
        backend.generate(**{**BASE, "job_id": "job-42_a.wav"})
        self.assertEqual(pipe.kwargs.get("id"), "job-42_a.wav")

    def test_an_unsafe_job_id_is_OMITTED_rather_than_sent(self):
        """⚠️ The alternative is a raise that names the ID, after the weights are
        resident — a full load spent reporting a naming problem. The package's
        own default id is a better answer than failing the generation."""
        backend, pipe = rig()
        for bad in ("../escape", "", "has space", "-leading"):
            with self.subTest(job_id=bad):
                backend.generate(**{**BASE, "job_id": bad})
                self.assertNotIn("id", pipe.kwargs)


class TestOutput(unittest.TestCase):
    def test_wav_is_returned_directly(self):
        backend, pipe = rig()
        path, duration_ms = backend.generate(**BASE)
        self.assertEqual(path.suffix, ".wav")
        self.assertTrue(path.exists())
        self.assertGreaterEqual(duration_ms, 0)

    def test_it_never_asks_the_pipeline_to_write_an_mp3(self):
        """CONTROL for the conversion step.

        The fake refuses a non-wav path exactly as `SongResult.save` does, so a
        backend that passed `fmt` straight through would raise here instead of
        converting. Without this the mp3 path would only be exercised by ffmpeg
        being installed, which is a different question.
        """
        backend, pipe = rig()
        try:
            backend.generate(**{**BASE, "fmt": "mp3"})
        except ValueError as e:  # the fake's own refusal — the defect this guards
            self.fail(f"the backend handed a non-wav path to save(): {e}")
        except Exception:
            # ffmpeg missing on this box is fine; the point is WHAT WAS SAVED.
            pass
        self.assertEqual(pipe.song.saved.suffix, ".wav")


class TestRefusals(unittest.TestCase):
    def test_generate_without_load_is_refused(self):
        with self.assertRaises(RuntimeError):
            YuEBackend().generate(**BASE)


if __name__ == "__main__":
    unittest.main()
