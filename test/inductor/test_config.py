# Owner(s): ["module: inductor"]
import math
import unittest

import torch

from torch._inductor import config

from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU


def dummy_fn(x):
    return torch.sigmoid(x + math.pi) / 10.0


class DummyModule(torch.nn.Module):
    def forward(self, x):
        return dummy_fn(x)


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
        self.assertEqual(config.shallow_copy_dict()["max_fusion_size"], 13337)
        config.max_fusion_size = 32
        self.assertEqual(config.max_fusion_size, 32)

        # a nested config
        prior = config.triton.cudagraphs
        config.triton.cudagraphs = not prior
        self.assertEqual(config.triton.cudagraphs, not prior)
        self.assertEqual(config.shallow_copy_dict()["triton.cudagraphs"], not prior)

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

    def test_get_compiler_config(self):
        from torch._inductor import config as inductor_default_config

        default_cudagraphs = inductor_default_config._default["triton.cudagraphs"]

        # nn.Module: should update default config with a new value
        model = DummyModule()
        optimized_module = torch.compile(
            model, options={"triton.cudagraphs": not default_cudagraphs}
        )
        compiler_config = optimized_module.get_compiler_config()
        self.assertEqual(compiler_config["triton.cudagraphs"], not default_cudagraphs)

        # nn.Module: keep default config
        model = DummyModule()
        optimized_module = torch.compile(model)
        compiler_config = optimized_module.get_compiler_config()
        self.assertEqual(
            compiler_config["triton.cudagraphs"],
            default_cudagraphs,
        )

        # compile user func: should update default config with a new value
        optimized_module = torch.compile(
            dummy_fn, options={"triton.cudagraphs": not default_cudagraphs}
        )
        compiler_config = optimized_module.get_compiler_config()
        self.assertEqual(compiler_config["triton.cudagraphs"], not default_cudagraphs)

        # compile user func: keep default config
        optimized_module = torch.compile(dummy_fn)
        compiler_config = optimized_module.get_compiler_config()
        self.assertEqual(
            compiler_config["triton.cudagraphs"],
            default_cudagraphs,
        )

        # backend=eager: expect None
        optimized_module = torch.compile(dummy_fn, backend="eager")
        compiler_config = optimized_module.get_compiler_config()
        self.assertTrue(compiler_config is None)

    def test_compile_api_passes_config(self):
        # ensure configs are actually passed down to inductor
        self.assertRaises(
            torch._dynamo.exc.BackendCompilerFailed,
            lambda: torch.compile(dummy_fn, options={"_raise_error_for_testing": True})(
                torch.randn(10)
            ),
        )

    def test_api_options(self):
        reduce_overhead_opts = torch._inductor.list_mode_options("reduce-overhead")
        self.assertEqual(reduce_overhead_opts["triton.cudagraphs"], True)
        self.assertEqual(reduce_overhead_opts.get("max_autotune", False), False)

        max_autotune_opts = torch._inductor.list_mode_options("max-autotune")
        self.assertEqual(max_autotune_opts["max_autotune"], True)
        self.assertEqual(max_autotune_opts["triton.cudagraphs"], True)

        max_autotune_opts = torch._inductor.list_mode_options(
            "max-autotune", dynamic=True
        )
        self.assertEqual(max_autotune_opts["max_autotune"], True)
        self.assertEqual(max_autotune_opts["triton.cudagraphs"], True)

        max_autotune_no_cudagraphs_opts = torch._inductor.list_mode_options(
            "max-autotune-no-cudagraphs"
        )
        self.assertEqual(max_autotune_no_cudagraphs_opts["max_autotune"], True)
        self.assertEqual(
            max_autotune_no_cudagraphs_opts.get("triton.cudagraphs", False), False
        )

    def test_invalid_backend(self):
        self.assertRaises(
            torch._dynamo.exc.InvalidBackend,
            lambda: torch.compile(dummy_fn, backend="does_not_exist")(torch.randn(10)),
        )

    def test_non_inductor_backend(self):
        def assert_options(expected_mode=None, expected_options=None):
            def backend(gm, _, *, mode=None, options=None):
                nonlocal call_count
                self.assertEqual(mode, expected_mode)
                self.assertEqual(options, expected_options)
                call_count += 1
                return gm

            return backend

        inp = torch.randn(8)

        def fn(x):
            return x + 1

        for mode, options in [
            (None, None),
            ("fast-mode", None),
            (None, {"foo": "bar"}),
        ]:
            call_count = 0
            torch.compile(
                fn, backend=assert_options(mode, options), mode=mode, options=options
            )(inp)
            torch._dynamo.reset()
            self.assertEqual(call_count, 1)

        # TypeError: eager() got an unexpected keyword argument 'mode'
        self.assertRaises(
            torch._dynamo.exc.BackendCompilerFailed,
            lambda: torch.compile(fn, backend="eager", mode="nope")(inp),
        )


if __name__ == "__main__":
    run_tests()
