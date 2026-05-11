# Owner(s): ["module: inductor"]
import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU_AND_TRITON,
    IS_BIG_GPU,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


@instantiate_parametrized_tests
class DeterministicTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(fresh_cache())

    def tearDown(self) -> None:
        self._exit_stack.close()
        super().tearDown()

    def test_use_deterministic_algorithsm(self):
        old_val = torch.are_deterministic_algorithms_enabled()
        try:
            for new_val in [True, False, True]:
                torch.use_deterministic_algorithms(new_val, warn_only=True)
                self.assertEqual(inductor_config.deterministic, new_val)
        finally:
            torch.use_deterministic_algorithms(old_val, warn_only=True)

    @parametrize("deterministic", [False, True])
    def test_mm_padding(self, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                return x @ y

            inps = [torch.rand([2049, 2049], device=GPU_TYPE) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] > 0)

    @parametrize("deterministic", [False, True])
    @inductor_config.patch(max_autotune=True)
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_max_autotune(self, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                return x @ y

            inps = [torch.rand([2048, 2048], device=GPU_TYPE) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] == 0)
            else:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] > 0)

    def test_pointwise_coordesc_tuning(self):
        @torch.compile(mode="max-autotune")
        def f(x):
            return x + 1

        x = torch.randn(2048, device=GPU_TYPE)
        self.assertEqual(f(x), x + 1)

        self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)

    @parametrize("deterministic", [False, True])
    def test_reduction_coordesc_tuning(self, deterministic):
        with inductor_config.patch(
            deterministic=deterministic, coordinate_descent_tuning=True
        ):

            @torch.compile()
            def foo(x):
                return x.sum(dim=-1)

            inp = torch.rand([2048, 2048], device=GPU_TYPE)

            out = foo(inp)
            self.assertEqual(out, inp.sum(dim=-1))

            if deterministic:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)

    @unittest.skipIf(not HAS_GPU_AND_TRITON, "requires GPU + Triton")
    @inductor_config.patch(batch_invariant=True)
    def test_persistent_reduction_batch_invariance(self):
        H = 768
        FULL = 1024

        def fn(x, w, b):
            return torch.nn.functional.layer_norm(x, (H,), weight=w, bias=b)

        torch.manual_seed(0)
        w = torch.randn(H, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(H, device=GPU_TYPE, dtype=torch.bfloat16)
        x_full = torch.randn(FULL, H, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled = torch.compile(fn)
        torch._dynamo.reset()
        out_full = compiled(x_full, w, b)
        self.assertEqual(out_full, fn(x_full, w, b))

        # Halving sweep, matching what the benchmark harness does.
        size = FULL // 2
        while size >= 1:
            torch._dynamo.reset()
            out = compiled(x_full[:size].contiguous(), w, b)
            ref = out_full[:size].contiguous()
            self.assertTrue(
                torch.equal(ref, out),
                f"persistent reduction diverged at size={size} (FULL={FULL})",
            )
            size //= 2

    @parametrize("batch_invariant", [False, True])
    @unittest.skipIf(not HAS_GPU_AND_TRITON, "requires GPU + Triton")
    def test_split_reduction_batch_invariance(self, batch_invariant):
        def fn(x):
            return x * x.mean(dim=[2, 3], keepdim=True)

        torch.manual_seed(0)
        x_full = torch.randn(16, 32, 256, 256, device=GPU_TYPE, dtype=torch.bfloat16)

        with inductor_config.patch(batch_invariant=batch_invariant):
            torch._dynamo.reset()
            torch._inductor.metrics.generated_kernel_count = 0
            compiled = torch.compile(fn)
            out_full = compiled(x_full)
            n_full = torch._inductor.metrics.generated_kernel_count

            torch._dynamo.reset()
            torch._inductor.metrics.generated_kernel_count = 0
            compiled = torch.compile(fn)
            out_half = compiled(x_full[:8].contiguous())
            n_half = torch._inductor.metrics.generated_kernel_count

        bitwise_equal = torch.equal(out_full[:8].contiguous(), out_half)

        if batch_invariant:
            self.assertTrue(
                bitwise_equal, "batch_invariant failed to produce bitwise-equal output"
            )
            self.assertEqual(
                n_full,
                n_half,
                f"batch_invariant should pin kernel count: {n_full} vs {n_half}",
            )
        else:
            # without the flag, the split-reduction decision flips
            self.assertFalse(
                bitwise_equal,
                "test shape no longer exercises split-reduction divergence",
            )
            self.assertNotEqual(n_full, n_half)

    @parametrize("inner_reduction", [True, False])
    @unittest.skipIf(not HAS_GPU_AND_TRITON, "requires GPU + Triton")
    def test_reduction_split_factor_batch_invariant(self, inner_reduction):
        # Under batch_invariant, K must depend only on reduction_numel_hint
        # so per-sample partial sums stay bitwise-stable across batch sizes.
        # Covers both inner and outer branches directly; the end-to-end test
        # above only exercises inner (its reduction dim has stride 1).
        from torch._inductor.choices import InductorChoices

        device = torch.device(GPU_TYPE, 0)
        R = 1 << 20  # 1M — large enough to split
        numels = (1, 8, 64, 512)

        def splits(bi):
            with inductor_config.patch(batch_invariant=bi):
                return [
                    InductorChoices.reduction_split_factor(
                        device, R, N, inner_reduction=inner_reduction
                    )
                    for N in numels
                ]

        bi = splits(True)
        self.assertEqual(len(set(bi)), 1, f"BI split varies with numel_hint: {bi}")
        self.assertGreater(bi[0], 1, "BI should still split a 1M reduction")

        # Guard against the test trivially passing if the non-BI path
        # were ever made invariant too.
        non_bi = splits(False)
        self.assertGreater(
            len(set(non_bi)), 1, f"non-BI no longer varies with numel_hint: {non_bi}"
        )

    def test_mark_batch_invariant_propagates_to_sizevars(self):
        # Stamp on input tensor → MetaTensorDesc → ShapeEnv.batch_invariant_symbols
        # → sizevars.is_batch_invariant_expr() answers correctly for expressions
        # involving marked vs. unmarked symbols.
        import sympy

        from torch._inductor.sizevars import SizeVarAllocator

        captured = {}

        def capture_backend(gm, example_inputs):
            fake_mode = torch._guards.detect_fake_mode(example_inputs)
            captured["shape_env"] = fake_mode.shape_env
            captured["sizevars"] = SizeVarAllocator(fake_mode.shape_env)
            return gm.forward

        x = torch.randn(8, 16, 32)
        y = torch.randn(8, 16, 32)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_batch_invariant(x, dim=0)

        torch._dynamo.reset()
        torch.compile(
            lambda a, b: a.sum(dim=-1) + b.sum(dim=-1),
            backend=capture_backend,
            dynamic=True,
        )(x, y)

        sv = captured["sizevars"]
        marked = captured["shape_env"].batch_invariant_symbols
        self.assertGreaterEqual(
            len(marked), 1, f"expected at least one marked symbol; got {marked}"
        )
        m = next(iter(marked))
        other = sympy.Symbol("s_other")

        # any expression containing the marked symbol is batch-variant
        self.assertFalse(sv.is_batch_invariant_expr(m))
        self.assertFalse(sv.is_batch_invariant_expr(m * 4))
        self.assertFalse(sv.is_batch_invariant_expr(m + other))
        # unmarked symbols and constants are invariant
        self.assertTrue(sv.is_batch_invariant_expr(8))
        self.assertTrue(sv.is_batch_invariant_expr(other))
        self.assertTrue(sv.is_batch_invariant_expr(other * 4))

    def test_mark_batch_invariant_input_validation(self):
        # Negative dim is normalized to positive (PyTorch convention).
        # Out-of-range dims raise. bool dim is rejected (was silently
        # accepted as int via Python's bool-is-int issue). Empty sequence
        # is rejected. Non-int / non-sequence raises.
        x = torch.randn(8, 16, 32)

        # Negative dim normalizes
        torch._dynamo.mark_batch_invariant(x, dim=-1)
        self.assertIn(2, x._dynamo_batch_invariant_dims)
        del x._dynamo_batch_invariant_dims

        # Out-of-range raises
        with self.assertRaises(IndexError):
            torch._dynamo.mark_batch_invariant(x, dim=5)
        with self.assertRaises(IndexError):
            torch._dynamo.mark_batch_invariant(x, dim=-5)

        # bool dim raises (bool is int subclass; would previously slip through)
        with self.assertRaises(TypeError):
            torch._dynamo.mark_batch_invariant(x, dim=True)

        # Empty sequence raises
        with self.assertRaises(ValueError):
            torch._dynamo.mark_batch_invariant(x, dim=())
        with self.assertRaises(ValueError):
            torch._dynamo.mark_batch_invariant(x, dim=[])

        # Non-int, non-sequence raises
        with self.assertRaises(TypeError):
            torch._dynamo.mark_batch_invariant(x, dim="0")  # type: ignore[arg-type]

        # Tuple/list of ints works
        torch._dynamo.mark_batch_invariant(x, dim=(0, 2))
        self.assertEqual(x._dynamo_batch_invariant_dims, {0, 2})

    def test_mark_batch_invariant_does_not_leak_when_unmarked(self):
        # Negative test: per-symbol API must not auto-apply when no input
        # is marked. Guards against an over-eager future change that turns
        # the global flag into "mark dim 0 of every input" without checking
        # whether dim 0 is actually batch (e.g. embedding tables, position
        # encodings have non-batch dim 0).
        from torch._inductor.sizevars import SizeVarAllocator

        captured = {}

        def cap(gm, ex):
            fake_mode = torch._guards.detect_fake_mode(ex)
            captured["se"] = fake_mode.shape_env
            captured["sv"] = SizeVarAllocator(fake_mode.shape_env)
            return gm.forward

        x = torch.randn(8, 16, 32)
        torch._dynamo.mark_dynamic(x, 0)
        # NO mark_batch_invariant call

        torch._dynamo.reset()
        torch.compile(lambda a: a.sum(-1), backend=cap, dynamic=True)(x)

        self.assertEqual(
            len(captured["se"].batch_invariant_symbols),
            0,
            "no input marked → no symbols should be batch-invariant",
        )

    def test_mark_batch_invariant_dim1_seq_first(self):
        # Spec §11: input shape (S, B, H) where batch dim is 1, not 0.
        # The harness's keep_batch_first only checks dim 0 — this test
        # exercises the per-symbol path that the harness can't.
        from torch._inductor.sizevars import SizeVarAllocator

        captured = {}

        def cap(gm, ex):
            fake_mode = torch._guards.detect_fake_mode(ex)
            captured["se"] = fake_mode.shape_env
            captured["sv"] = SizeVarAllocator(fake_mode.shape_env)
            return gm.forward

        x = torch.randn(64, 16, 768)  # (seq, batch, hidden)
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.mark_batch_invariant(x, dim=1)

        torch._dynamo.reset()
        torch.compile(lambda a: a.sum(-1), backend=cap, dynamic=True)(x)

        marked = captured["se"].batch_invariant_symbols
        self.assertGreaterEqual(
            len(marked),
            1,
            f"mark_batch_invariant(x, dim=1) must allocate a marked symbol; got {marked}",
        )
        # One of the marked symbols should appear in x's dim-1 size expression
        m = next(iter(marked))
        sv = captured["sv"]
        self.assertFalse(sv.is_batch_invariant_expr(m))

    def test_reduce_over_marked_dim_unconstrained(self):
        # Spec §2: outputs without a batch dim are unconstrained. Reducing
        # over the marked dim produces an output without that dim — the
        # function must compile and produce a correct (eager-matching)
        # result, but no batch-invariance obligation applies.
        x = torch.randn(8, 16, 32, device=GPU_TYPE, dtype=torch.bfloat16)
        torch._dynamo.mark_batch_invariant(x, dim=0)
        torch._dynamo.reset()
        compiled = torch.compile(lambda a: a.sum(dim=0))
        out = compiled(x)
        torch.testing.assert_close(out, x.sum(dim=0))

    def test_reduce_to_scalar_unconstrained(self):
        # Spec §2: scalar output, no batch dim, no obligation.
        x = torch.randn(8, 16, 32, device=GPU_TYPE, dtype=torch.bfloat16)
        torch._dynamo.mark_batch_invariant(x, dim=0)
        torch._dynamo.reset()
        compiled = torch.compile(lambda a: a.sum())
        out = compiled(x)
        torch.testing.assert_close(out, x.sum())

    def test_mark_batch_invariant_warns_on_static_dim(self):
        # A6/D6: marking a dim that ends up specialized to a literal int
        # (no mark_dynamic, no dynamic=True) is a silent no-op without this
        # warning. Verify the warning fires AND that the positive-control
        # path (with mark_dynamic) does NOT trigger it.
        import warnings

        x = torch.randn(8, 16, 32, device=GPU_TYPE, dtype=torch.bfloat16)
        torch._dynamo.mark_batch_invariant(x, dim=0)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            torch._dynamo.reset()
            torch.compile(lambda a: a.sum(-1))(x)
            self.assertTrue(
                any("mark_batch_invariant had no effect" in str(w.message) for w in ws),
                "expected silent-no-op warning for static dim",
            )

        x2 = torch.randn(8, 16, 32, device=GPU_TYPE, dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(x2, 0)
        torch._dynamo.mark_batch_invariant(x2, dim=0)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            torch._dynamo.reset()
            torch.compile(lambda a: a.sum(-1))(x2)
            self.assertFalse(
                any("mark_batch_invariant had no effect" in str(w.message) for w in ws),
                "warning must NOT fire when mark_dynamic provides a symbol",
            )

    def test_mark_batch_invariant_warns_on_parameter(self):
        # D7: marking an nn.Parameter is almost certainly a user error.
        import warnings

        linear = torch.nn.Linear(32, 32)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            torch._dynamo.mark_batch_invariant(linear.weight, dim=0)
            self.assertTrue(
                any("nn.Parameter" in str(w.message) for w in ws),
                "expected Parameter warning",
            )

        x = torch.randn(8, 16)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            torch._dynamo.mark_batch_invariant(x, dim=0)
            self.assertFalse(
                any("nn.Parameter" in str(w.message) for w in ws),
                "regular tensor must not trigger Parameter warning",
            )

    def test_mark_batch_invariant_warns_on_export(self):
        # D8: torch.export discards the mark — the exported graph carries
        # no batch-invariance guarantee. Warn at export entry.
        import warnings

        class M(torch.nn.Module):
            def forward(self, x):
                return x.sum(-1)

        x = torch.randn(8, 16, 32)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_batch_invariant(x, dim=0)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            torch.export.export(M(), (x,))
            self.assertTrue(
                any(
                    "mark_batch_invariant" in str(w.message)
                    and "torch.export" in str(w.message)
                    for w in ws
                ),
                "expected torch.export-discards warning",
            )

        x2 = torch.randn(8, 16, 32)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            torch.export.export(M(), (x2,))
            self.assertFalse(
                any("mark_batch_invariant" in str(w.message) for w in ws),
                "unmarked export must not trigger warning",
            )

    def test_mark_batch_invariant_warns_on_value_dependent_op(self):
        # F3: output value scales with the batch-marked dim (e.g. x * x.size(0)).
        # This is batch-variant by construction; warn so the user notices.
        import warnings

        def fn(x):
            return x * x.size(0)

        x = torch.randn(16, 32, device=GPU_TYPE, dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_batch_invariant(x, dim=0)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            torch._dynamo.reset()
            torch.compile(fn, dynamic=True)(x)
            self.assertTrue(
                any("output value depends on" in str(w.message) for w in ws),
                "expected value-dependent-op warning",
            )

        x2 = torch.randn(16, 32, device=GPU_TYPE, dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(x2, 0)
        torch._dynamo.mark_batch_invariant(x2, dim=0)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            torch._dynamo.reset()
            torch.compile(lambda a: a * 2.0, dynamic=True)(x2)
            self.assertFalse(
                any("output value depends on" in str(w.message) for w in ws),
                "constant scalar mul must not trigger value-dependent warning",
            )

    def test_mark_batch_invariant_propagates_across_graph_break(self):
        # A graph break creates a new ShapeEnv for the resume subgraph.
        # The mark on the user's input tensor must reach that subgraph too,
        # otherwise reductions in the resume body fall back to non-invariant
        # codegen. Verified by snapshotting each subgraph's ShapeEnv.
        from torch._inductor.sizevars import SizeVarAllocator

        shape_envs: list = []

        def cap(gm, ex):
            fake_mode = torch._guards.detect_fake_mode(ex)
            shape_envs.append(
                (
                    fake_mode.shape_env,
                    set(fake_mode.shape_env.batch_invariant_symbols),
                )
            )
            return gm.forward

        x = torch.randn(8, 16, 32, device=GPU_TYPE, dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_batch_invariant(x, dim=0)

        def fn(x):
            a = x + 1.0
            print("graph break here")  # force a graph break
            return a.sum(-1)

        torch._dynamo.reset()
        torch.compile(fn, backend=cap, dynamic=True)(x)

        self.assertGreaterEqual(
            len(shape_envs), 2, "expected at least two subgraphs from graph break"
        )
        for idx, (se, marked) in enumerate(shape_envs):
            self.assertGreater(
                len(marked),
                0,
                f"subgraph {idx} ShapeEnv has no batch-invariant symbols",
            )
            sv = SizeVarAllocator(se)
            m = next(iter(marked))
            self.assertFalse(sv.is_batch_invariant_expr(m))

    def test_reorder_for_locality_preserves_randint_order(self):
        with inductor_config.patch(fallback_random=True):

            def fn():
                torch.manual_seed(0)
                out = torch.randint(0, 100, (4, 1), dtype=torch.int64)
                _ = torch.randint(0, 100, (2, 1), dtype=torch.int64)
                return out

            compiled = torch.compile(fn, backend="inductor")

            torch.manual_seed(0)
            eager = fn()

            torch.manual_seed(0)
            compiled_out = compiled()

            torch.testing.assert_close(eager, compiled_out)

    @unittest.skipIf(IS_FBCODE, "Skipping run2run determinism test in fbcode")
    @parametrize("model_name", ["GoogleFnet", "BertForMaskedLM", "DistillGPT2"])
    @parametrize("training_or_inference", ["training", "inference"])
    @parametrize("precision", ["float32", "bfloat16", "float16", "amp"])
    def test_run2run_determinism(self, model_name, training_or_inference, precision):
        """
        Test run2run determinism for a few huggingface models.

        The test assumes benchmarks/dynamo/huggingface.py can be found from
        the current working directory.
        """

        def _setup_env(env):
            env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"  # disable autotune cache
            env["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] = "0"
            env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
            if enable_determinism:
                env["TORCHINDUCTOR_DETERMINISTIC"] = "1"

        # set to false if you want to check how the test fails without
        # the deterministic mode
        enable_determinism = True
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_pkl = os.path.join(tmpdir, "saved.pkl")
            cmd = (
                f"{sys.executable} {REPO_ROOT}/benchmarks/dynamo/huggingface.py --backend inductor"
                + f" --{precision} --accuracy --only {model_name} --{training_or_inference}"
                + f" --disable-cudagraphs --save-model-outputs-to={saved_pkl}"
            )
            print("Command", cmd)
            env = os.environ.copy()
            _setup_env(env)
            out = subprocess.run(cmd.split(), capture_output=True, env=env)

            # We don't check the accuracy against eager here because some
            # of the combination between model and precision can not
            # pass that accuracy test. But it's still valuable to make
            # sure we generate bitwise equivalent result from run to run.
            # self.assertTrue("pass" in out.stdout.decode())

            cmd = (
                f"{sys.executable} {REPO_ROOT}/benchmarks/dynamo/huggingface.py --backend inductor"
                + f" --{precision} --accuracy --only {model_name} --{training_or_inference}"
                + f" --disable-cudagraphs --compare-model-outputs-with={saved_pkl}"
            )
            print("Command", cmd)

            # distort benchmarking results
            env["TORCHINDUCTOR_DISTORT_BENCHMARKING_RESULT"] = "inverse"
            out = subprocess.run(cmd.split(), capture_output=True, env=env)
            self.assertTrue(
                "The result is bitwise equivalent to the previously saved result"
                in out.stdout.decode(),
                f"stdout: {out.stdout.decode()}, stderr: {out.stderr.decode()}",
            )


if __name__ == "__main__":
    if HAS_GPU_AND_TRITON:
        run_tests()
