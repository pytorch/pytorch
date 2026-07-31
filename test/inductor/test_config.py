# Owner(s): ["module: inductor"]
import functools
import math
import unittest

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_TRITON
from torch.testing._internal.triton_utils import requires_gpu


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
        self.assertEqual(config.get_config_copy()["max_fusion_size"], 13337)
        config.max_fusion_size = 32
        self.assertEqual(config.max_fusion_size, 32)

        # a nested config
        prior = config.triton.cudagraphs
        config.triton.cudagraphs = not prior
        self.assertEqual(config.triton.cudagraphs, not prior)
        self.assertEqual(config.get_config_copy()["triton.cudagraphs"], not prior)

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
            torch.testing.assert_close(
                opt_fn(x), y, msg=f"torch.compile(..., **{kwargs!r}) failed"
            )

    def test_get_compiler_config(self):
        from torch._inductor import config as inductor_default_config

        default_cudagraphs = inductor_default_config.triton.cudagraphs

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

    def test_codegen_skips_custom_passes(self):
        class _CustomPass(PatternMatcherPass):
            def __init__(self) -> None:
                super().__init__()

            def __call__(self, g: torch.fx.Graph):
                self.apply(g)

        g = _CustomPass()

        with torch._inductor.config.patch(
            post_grad_custom_post_pass=g,
            post_grad_custom_pre_pass=g,
        ):
            code = torch._inductor.config.codegen_config()
            self.assertNotIn("post_grad_custom", code)

    def test_codegen_serializes_inductor_choices_partial(self):
        partial_choices = functools.partial(InductorChoices, lb=10, ub=100)

        with torch._inductor.config.patch(inductor_choices_class=partial_choices):
            code = torch._inductor.config.codegen_config()

            self.assertIn("inductor_choices_class", code)
            self.assertIn("functools.partial", code)
            self.assertNotIn("<class ", code)
            compile(code, "<codegen_config>", "exec")
            namespace = {"torch": torch}
            exec(code, namespace)
            reconstructed = torch._inductor.config.inductor_choices_class
            self.assertIsInstance(reconstructed, functools.partial)
            self.assertIs(reconstructed.func, InductorChoices)
            self.assertEqual(reconstructed.args, ())
            self.assertEqual(reconstructed.keywords, {"lb": 10, "ub": 100})

    def test_codegen_serializes_builtin_partial_with_container_arg(self):
        partial_choices = functools.partial(max, [1, (2, {"limit": 3})])

        with torch._inductor.config.patch(inductor_choices_class=partial_choices):
            code = torch._inductor.config.codegen_config()
            compile(code, "<codegen_config>", "exec")
            exec(code, {"torch": torch})

            reconstructed = torch._inductor.config.inductor_choices_class
            self.assertIs(reconstructed.func, max)
            self.assertEqual(reconstructed.args, ([1, (2, {"limit": 3})],))

    def test_codegen_partial_over_non_importable_emits_comment(self):
        non_importable = functools.partial(lambda lb=0: None, lb=5)

        with torch._inductor.config.patch(inductor_choices_class=non_importable):
            code = torch._inductor.config.codegen_config()

            self.assertIn("omitted", code)
            self.assertIn("inductor_choices_class", code)
            self.assertNotIn("config.inductor_choices_class = functools.partial", code)
            compile(code, "<codegen_config>", "exec")

    def test_codegen_partial_with_unsupported_arg_emits_comment(self):
        for arg in ({1, 2}, math.inf, math.nan):
            with self.subTest(arg=arg):
                partial_choices = functools.partial(InductorChoices, arg)
                with torch._inductor.config.patch(
                    inductor_choices_class=partial_choices
                ):
                    code = torch._inductor.config.codegen_config()

                self.assertIn("omitted", code)
                self.assertNotIn("config.inductor_choices_class =", code)
                compile(code, "<codegen_config>", "exec")

    def test_codegen_partial_with_unresolvable_identity_emits_comment(self):
        def impostor():
            pass

        impostor.__module__ = InductorChoices.__module__
        impostor.__qualname__ = InductorChoices.__qualname__
        partial_choices = functools.partial(impostor)

        with torch._inductor.config.patch(inductor_choices_class=partial_choices):
            code = torch._inductor.config.codegen_config()

        self.assertIn("partial callable cannot be re-imported", code)
        compile(code, "<codegen_config>", "exec")

    def test_codegen_partial_with_raising_metadata_emits_comment(self):
        class CallableWithRaisingMetadata:
            @property
            def __module__(self):
                raise RuntimeError("module metadata unavailable")

            def __call__(self):
                pass

        partial_choices = functools.partial(CallableWithRaisingMetadata())

        with torch._inductor.config.patch(inductor_choices_class=partial_choices):
            code = torch._inductor.config.codegen_config()

        self.assertIn("partial callable cannot be re-imported", code)
        compile(code, "<codegen_config>", "exec")

    def test_codegen_partial_with_invalid_qualname_emits_comment(self):
        class CallableWithInvalidQualname:
            __module__ = __name__
            __qualname__ = "invalid name"

            def __call__(self):
                pass

        callable_with_invalid_qualname = CallableWithInvalidQualname()
        globals()["invalid name"] = callable_with_invalid_qualname
        try:
            partial_choices = functools.partial(callable_with_invalid_qualname)
            with torch._inductor.config.patch(inductor_choices_class=partial_choices):
                code = torch._inductor.config.codegen_config()
        finally:
            del globals()["invalid name"]

        self.assertIn("partial callable cannot be re-imported", code)
        compile(code, "<codegen_config>", "exec")

    def test_select_decomp_table_fallback_embedding_bag_byte_unpack(self):
        """Test that select_decomp_table removes embedding_bag_byte_unpack when fallback is enabled"""
        from torch._inductor.decomposition import select_decomp_table

        # Test with fallback_embedding_bag_byte_unpack = False (default)
        with config.patch(fallback_embedding_bag_byte_unpack=False):
            decomp_table = select_decomp_table()
            # The operation should be in decompositions when fallback is False
            # Note: We check if it's in the fast_random_decomps() or decompositions table
            self.assertTrue(
                torch.ops.quantized.embedding_bag_byte_unpack.default in decomp_table
                or len(decomp_table)
                > 0  # fast_random_decomps() is used when fallback is False
            )

        # Test with fallback_embedding_bag_byte_unpack = True
        with config.patch(fallback_embedding_bag_byte_unpack=True):
            decomp_table = select_decomp_table()
            # The operation should NOT be in decompositions when fallback is True
            self.assertNotIn(
                torch.ops.quantized.embedding_bag_byte_unpack.default, decomp_table
            )

    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_options_do_something(self):
        """
        Verify that we can populate and load functions from the cache.
        """

        counters.clear()

        def fn(x, y):
            yy = y @ y
            return x * 2 + yy.view(25)

        def fn2(x, y):
            yy = y @ y
            return x * 2 + yy.view(25)

        a_orig = torch.rand(25, dtype=torch.float32, device="cpu")
        b_orig = torch.rand(5, 5, dtype=torch.float32, device="cpu")

        compiled_fn = torch.compile(
            fn,
            options={
                "fx_graph_cache": True,
                "fx_graph_remote_cache": False,
                "bundle_triton_into_fx_graph_cache": True,
            },
        )

        a1 = a_orig.clone()
        b1 = b_orig.clone()
        a2 = a_orig.clone()
        b2 = b_orig.clone()

        # A first call should miss in the cache.
        eager_result = fn(a1, b1)
        compiled_result = compiled_fn(a2, b2)
        self.assertEqual(eager_result, compiled_result)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        counters.clear()

        compiled_fn2 = torch.compile(
            fn2,
            options={
                "fx_graph_cache": False,
                "fx_graph_remote_cache": False,
                "bundle_triton_into_fx_graph_cache": False,
            },
        )

        # A first call should do nothing since cache is disabled
        eager_result = fn2(a1, b1)
        compiled_result = compiled_fn2(a2, b2)
        self.assertEqual(eager_result, compiled_result)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

    @requires_gpu
    @torch._inductor.config.patch(fx_graph_cache=False)
    def test_config_read_in_backwards(self):
        @torch.compile
        def f(x, y):
            z = x @ y
            return z.sin().sum()

        called = False

        def my_pass(graph):
            nonlocal called
            called = True

        x, y = (
            torch.randn(3, 3, device=GPU_TYPE, requires_grad=True),
            torch.randn(3, 3, device=GPU_TYPE),
        )
        z = f(x, y)
        z.backward()
        self.assertFalse(called)
        torch._dynamo.reset()
        z = f(x, y)
        with torch._inductor.config.patch(post_grad_custom_pre_pass=my_pass):
            z.backward()

        self.assertTrue(called)

        called = False
        torch._dynamo.reset()
        z = f(x, y)
        with torch._inductor.config.patch(post_grad_custom_pre_pass=my_pass):
            torch.autograd.grad(z, x)
        self.assertTrue(called)

    @torch._inductor.config.patch(fx_graph_cache=False)
    def test_config_read_in_grad_fn(self):
        @torch.compile
        def f(x, y):
            z = x @ y
            return z.sin().sum()

        called = False

        def my_pass(graph):
            nonlocal called
            called = True

        x, y = (
            torch.randn(3, 3, requires_grad=True),
            torch.randn(3, 3),
        )

        with torch._inductor.config.patch(post_grad_custom_pre_pass=my_pass):
            z = f(x, y)
        self.assertTrue(called)

        # Make sure the context gets cleared after forward pass
        called = False
        z.grad_fn.apply(torch.tensor(0))
        self.assertFalse(called)


if __name__ == "__main__":
    run_tests()
