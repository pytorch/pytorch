# Owner(s): ["module: inductor"]
import logging
import math
import unittest

import torch

import torch._dynamo.config as dynamo_config
from torch._dynamo.test_case import run_tests, TestCase

from torch._inductor import config
from torch.testing._internal.inductor_utils import HAS_CPU


def dummy_fn(x):
    return torch.sigmoid(x + math.pi) / 10.0


class TestInductorConfig(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._saved_config = config.save_config()

    def tearDown(self):
        super().tearDown()
        config.load_config(self._saved_config)

    def test_set(self):
        config.max_fusion_size = 13337
        self.assertEqual(config.max_fusion_size, 13337)
        self.assertEqual(config.to_dict()["max_fusion_size"], 13337)
        config.to_dict()["max_fusion_size"] = 32
        self.assertEqual(config.max_fusion_size, 32)

        # a nested config
        prior = config.triton.cudagraphs
        config.triton.cudagraphs = not prior
        self.assertEqual(config.triton.cudagraphs, not prior)
        self.assertEqual(config.to_dict()["triton.cudagraphs"], not prior)

    def test_save_load(self):
        config.max_fusion_size = 123
        config.triton.cudagraphs = True
        saved1 = config.save_config()
        config.max_fusion_size = 321
        config.triton.cudagraphs = False
        saved2 = config.save_config()

        self.assertEqual(config.max_fusion_size, 321)
        self.assertEqual(config.triton.cudagraphs, False)
        config.load_config(saved1)
        self.assertEqual(config.max_fusion_size, 123)
        self.assertEqual(config.triton.cudagraphs, True)
        config.load_config(saved2)
        self.assertEqual(config.max_fusion_size, 321)
        self.assertEqual(config.triton.cudagraphs, False)

    def test_hasattr(self):
        self.assertTrue(hasattr(config, "max_fusion_size"))
        self.assertFalse(hasattr(config, "missing_name"))

    def test_invalid_names(self):
        self.assertRaises(AttributeError, lambda: config.does_not_exist)
        self.assertRaises(AttributeError, lambda: config.triton.does_not_exist)

        def store1():
            config.does_not_exist = True

        def store2():
            config.triton.does_not_exist = True

        self.assertRaises(AttributeError, store1)
        self.assertRaises(AttributeError, store2)

    def test_patch(self):
        with config.patch(max_fusion_size=456):
            self.assertEqual(config.max_fusion_size, 456)
            with config.patch(max_fusion_size=789):
                self.assertEqual(config.max_fusion_size, 789)
            self.assertEqual(config.max_fusion_size, 456)

        with config.patch({"cpp.threads": 9000, "max_fusion_size": 9001}):
            self.assertEqual(config.cpp.threads, 9000)
            self.assertEqual(config.max_fusion_size, 9001)
            with config.patch("cpp.threads", 8999):
                self.assertEqual(config.cpp.threads, 8999)
            self.assertEqual(config.cpp.threads, 9000)

    def test_log_level_property(self):
        old = dynamo_config.log_level
        try:
            dynamo_config.log_level = logging.CRITICAL
            self.assertEqual(logging.getLogger("torch._dynamo").level, logging.CRITICAL)
        finally:
            dynamo_config.log_level = old

    @unittest.skipIf(not HAS_CPU, "requires C++ compiler")
    def test_compile_api(self):
        # these are mostly checking config processing doesn't blow up with exceptions
        x = torch.randn(8)
        y = dummy_fn(x)
        checks = [
            {},
            {"mode": "default"},
            {"mode": "reduce-overhead"},
            {"mode": "max-autotune"},
            {
                "options": {
                    "max-fusion-size": 128,
                    "unroll_reductions_threshold": 32,
                    "triton.cudagraphs": False,
                }
            },
            {"dynamic": True},
            {"fullgraph": True, "backend": "inductor"},
            {"disable": True},
        ]

        for kwargs in checks:
            torch._dynamo.reset()
            opt_fn = torch.compile(dummy_fn, **kwargs)
            torch.testing.assert_allclose(
                opt_fn(x), y, msg=f"torch.compile(..., **{kwargs!r}) failed"
            )

    def test_compile_api_passes_config(self):
        # ensure configs are actually passed down to inductor
        self.assertRaises(
            torch._dynamo.exc.BackendCompilerFailed,
            lambda: torch.compile(dummy_fn, options={"_raise_error_for_testing": True})(
                torch.randn(10)
            ),
        )

    @torch._dynamo.config.patch(raise_on_backend_change=True)
    def test_inductor_config_changes_warning(self):
        import torch

        @torch.compile
        def a(x):
            return x + 1

        @torch.compile
        def b(x):
            return x + 2

        @torch.compile(mode="max-autotune")
        def c(x):
            return x + 3

        @torch.compile(mode="max-autotune")
        def d(x):
            return x + 4

        # no warning same config
        a(torch.randn(10))
        b(torch.randn(10))
        a(torch.randn(10))
        b(torch.randn(10))

        torch._dynamo.reset()
        # no warning after reset
        c(torch.randn(10))
        c(torch.randn(10))
        d(torch.randn(10))
        d(torch.randn(10))

        self.assertRaises(torch._dynamo.exc.ResetRequired, lambda: a(torch.randn(10)))

        with torch._dynamo.config.patch(
            raise_on_backend_change=False
        ), self.assertWarns(Warning):
            # normally it is just a warning
            a(torch.randn(10))

        # only warn once
        a(torch.randn(10))


if __name__ == "__main__":
    run_tests()
