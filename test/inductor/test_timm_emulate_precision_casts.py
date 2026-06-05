# Owner(s): ["module: inductor"]
"""Regression test for benchmarks/dynamo/timm_models.py emulate_precision_casts.

Guards against two bugs:
  1. The flag leaking across iter_models iterations (a listed model flipped
     it on; later non-listed models silently ran with it on).
  2. The captured baseline being read before main() applied
     `--inductor-config emulate_precision_casts=...` overrides, which would
     clobber a user-specified True.
"""

import argparse
import pathlib
import sys
import unittest
from unittest import mock

import torch
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase


BENCHMARKS_DYNAMO = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "benchmarks" / "dynamo"
)


@unittest.skipUnless(BENCHMARKS_DYNAMO.is_dir(), "benchmarks/dynamo not present")
class TimmEmulatePrecisionCastsTest(TestCase):
    LISTED = "vit_base_patch14_dinov2.lvd142m"
    UNLISTED = "vit_base_patch16_siglip_256"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            import timm  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("timm not installed")
        sys.path.insert(0, str(BENCHMARKS_DYNAMO))
        from timm_models import TimmRunner

        cls.TimmRunner = TimmRunner

    def setUp(self):
        super().setUp()
        self._orig_flag = inductor_config.emulate_precision_casts
        import timm

        stub = lambda self, name: timm.create_model(  # noqa: E731
            "vit_tiny_patch16_224", pretrained=False, num_classes=10
        )
        self._patches = [
            mock.patch.object(self.TimmRunner, "_download_model", stub),
            mock.patch.object(self.TimmRunner, "validate_model", lambda *a, **k: None),
        ]
        for p in self._patches:
            p.start()
        ns = argparse.Namespace(
            training=False,
            use_eval_mode=True,
            channels_last=False,
            enable_activation_checkpointing=False,
            amp=False,
        )
        self.runner = self.TimmRunner()
        self.runner._args = self.runner.args = ns

    def tearDown(self):
        inductor_config.emulate_precision_casts = self._orig_flag
        for p in self._patches:
            p.stop()
        super().tearDown()

    def _load(self, name):
        # load_model may fail late (stub model vs. real input shape); the
        # flag assignment happens before any such failure.
        try:
            self.runner.load_model(torch.device("cpu"), name)
        except Exception:
            pass

    def test_flag_toggles_per_model_no_carryover(self):
        """Listed model flips flag True; subsequent unlisted model resets it."""
        inductor_config.emulate_precision_casts = False
        self._load(self.LISTED)
        self.assertTrue(inductor_config.emulate_precision_casts)
        self._load(self.UNLISTED)
        self.assertFalse(inductor_config.emulate_precision_casts)

    def test_user_override_preserved_in_production_order(self):
        """Mirrors timm_main(): TimmRunner() runs before main() applies
        --inductor-config overrides. The fix must capture baseline lazily
        (on first load_model), not eagerly in __init__."""
        # TimmRunner() already constructed in setUp.
        inductor_config.emulate_precision_casts = True  # CLI override applied
        self._load(self.UNLISTED)  # baseline captured here
        self.assertTrue(inductor_config.emulate_precision_casts)
        self._load(self.LISTED)
        self.assertTrue(inductor_config.emulate_precision_casts)


if __name__ == "__main__":
    run_tests()
