"""Structural invariants for BACKEND_MODELS. No network, no GPU, no downloads.

WHY THIS EXISTS. Two registry bugs shipped to users and stayed invisible for a long time:

  1. Install-detection hardcoded a `config.json` probe. Three backends
     (stable-audio, stable-audio-small, audioldm2) publish no such file at their repo
     root, so the Models panel reported "Not installed" FOREVER — the user re-downloaded
     many GB on every click and nothing ever surfaced an error.
  2. Every `size_gb` was a guess, understating the real download by 2.1x-4.1x.
     `bark` advertised 5GB and transfers 20.68GB.

Neither was a hard failure, so nothing caught them. These tests catch the parts that are
checkable offline; scripts/audit_backend_registry.py checks the rest against the live
repos (does the probe file actually exist, is size_gb still true).

Run with:  python/.venv/Scripts/python.exe python/audio_engine/tests/test_backend_registry.py
"""
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
AUDIO_ENGINE = SRC / "litesuite_rust_sidecar" / "litesound"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# GLOB_CHARS is IMPORTED, never re-declared. It used to be a local "*?[" here that
# happened to match the inline literal in probe_file_for — agreeing with a copy of the
# code instead of with the code. Zero drift at the time, but adding "{" upstream for
# brace globs would have left this behind and test_probe_is_always_a_concrete_filename
# would have kept passing while the bug it guards came back.
from litesuite_rust_sidecar.litesound.hub_downloader import BACKEND_MODELS, GLOB_CHARS, probe_file_for  # noqa: E402


class TestRegistryShape(unittest.TestCase):
    def test_every_entry_has_the_required_fields(self):
        for backend_id, info in BACKEND_MODELS.items():
            with self.subTest(backend=backend_id):
                for field in ("repo_id", "repo_type", "description", "size_gb"):
                    self.assertIn(field, info, f"{backend_id} is missing {field}")
                self.assertGreater(info["size_gb"], 0)

    def test_size_is_plausible(self):
        """A guessed-looking size is how the 2.1x-4.1x understatements survived.

        Real measured totals are never suspiciously round, so require at least some
        precision — this makes a placeholder impossible to leave in by accident.
        """
        for backend_id, info in BACKEND_MODELS.items():
            if info.get("gguf_profile"):
                continue  # computed from the manifest, not hand-written
            if info.get("external"):
                continue  # hand-placed local runtime; nothing on the Hub to measure
            with self.subTest(backend=backend_id):
                self.assertNotEqual(
                    float(info["size_gb"]), round(float(info["size_gb"])),
                    f"{backend_id}: size_gb={info['size_gb']} is a whole number — it "
                    f"looks guessed. Measure it with scripts/audit_backend_registry.py.",
                )


class TestProbeResolution(unittest.TestCase):
    """The probe decides whether a user ever stops re-downloading. It has to be right."""

    def test_probe_is_always_a_concrete_filename(self):
        for backend_id, info in BACKEND_MODELS.items():
            with self.subTest(backend=backend_id):
                probe = probe_file_for(info)
                self.assertTrue(probe)
                for ch in GLOB_CHARS:
                    self.assertNotIn(
                        ch, probe,
                        f"{backend_id}: probe {probe!r} is a glob; "
                        f"try_to_load_from_cache needs a literal filename",
                    )

    def test_probe_is_among_the_files_actually_downloaded(self):
        """THE invariant. A probe outside allow_patterns can never be found on disk,
        so detection would report 'not installed' forever — silently."""
        for backend_id, info in BACKEND_MODELS.items():
            patterns = info.get("allow_patterns")
            if not patterns:
                continue
            with self.subTest(backend=backend_id):
                probe = probe_file_for(info)
                self.assertIn(
                    probe, patterns,
                    f"{backend_id}: probe {probe!r} is not in allow_patterns {patterns} — "
                    f"the install will never place it, so the backend would read as "
                    f"permanently not-installed",
                )

    def test_probe_is_derived_when_patterns_exist(self):
        """Derivation is what keeps probe and patterns from drifting apart. An explicit
        probe_file alongside allow_patterns re-opens that gap, so require a reason."""
        for backend_id, info in BACKEND_MODELS.items():
            if info.get("allow_patterns") and info.get("probe_file"):
                with self.subTest(backend=backend_id):
                    self.assertIn(info["probe_file"], info["allow_patterns"])

    def test_diffusers_repos_do_not_use_the_config_json_default(self):
        """Regression lock on the exact bug that shipped.

        These three publish no root config.json. If someone drops their probe_file, the
        default silently returns and the re-download loop comes back.
        """
        for backend_id in ("stable-audio", "audioldm2", "stable-audio-small"):
            with self.subTest(backend=backend_id):
                self.assertNotEqual(
                    probe_file_for(BACKEND_MODELS[backend_id]), "config.json",
                    f"{backend_id} has no root config.json — probing for it reports "
                    f"'not installed' forever",
                )


class TestProbeDerivation(unittest.TestCase):
    """Unit-level checks on probe_file_for itself, independent of the registry."""

    def test_explicit_probe_wins(self):
        self.assertEqual(probe_file_for({"probe_file": "x.json"}), "x.json")

    def test_derives_first_literal_pattern(self):
        self.assertEqual(
            probe_file_for({"allow_patterns": ["a.safetensors", "b.json"]}), "a.safetensors"
        )

    def test_skips_globs_when_deriving(self):
        self.assertEqual(
            probe_file_for({"allow_patterns": ["unet/*.safetensors", "real.json"]}), "real.json"
        )

    def test_every_glob_char_is_skipped_when_deriving(self):
        """Cases are DERIVED from production's GLOB_CHARS, not from a list written here.

        This is what makes the import load-bearing rather than decorative: add "{" to
        GLOB_CHARS upstream for brace globs and this test starts covering it on the next
        run, with no edit here. A hand-written list of characters would silently keep
        testing the old set — which is exactly the drift the duplicate constant invited.
        """
        for ch in GLOB_CHARS:
            with self.subTest(char=ch):
                self.assertEqual(
                    probe_file_for({"allow_patterns": [f"unet{ch}x.safetensors", "real.json"]}),
                    "real.json",
                    f"a pattern containing {ch!r} is a glob and must never be the probe",
                )

    def test_falls_back_to_config_json(self):
        self.assertEqual(probe_file_for({}), "config.json")


if __name__ == "__main__":
    unittest.main()
