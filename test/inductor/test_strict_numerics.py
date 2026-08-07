# Owner(s): ["module: inductor"]
"""Tests for strict inner-contiguous sum ordering."""

import os
import unittest
from unittest import mock

import torch
from torch._inductor import config, metrics
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._native.ops.sum.inner_tree_plan import compute_inner_tree_params, vec_size
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, skipIfNoCuteDSL
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.utils._triton import has_triton_reduction_ordering


def _singleton_input(device):
    return torch.as_strided(torch.full((1,), -0.0, device=device), (1, 1), (1, 0))


SUM_CASES = (
    ("persistent", (8192, 256), 1),
    ("looped", (32, 12000), 1),
    ("split", (8, 65536), 1),
)
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
INNER_TREE_CALL = "reduction_ordering=tl.constexpr(tl.ReductionOrdering.INNER_TREE)"

SUM_VARIANTS = (
    ("keepdim", (64, 300), 1, torch.float32, True, {}),
    ("autotune", (8192, 256), 1, torch.float32, False, {"max_autotune": True}),
)

DYNAMIC_CASES = (
    ("ragged", (65536, 65537), {}),
    ("plan_change", (512, 2048), {}),
    ("min_split", (65536,), {"min_num_split": 16}),
)

OUT_OF_SCOPE_CASES = (
    ("column", lambda z: torch.sum(z, 0)),
    ("multidim", lambda z: torch.sum(z, (0, 1))),
    ("mean", lambda z: torch.mean(z, 1)),
    ("dtype", lambda z: torch.sum(z, 1, dtype=torch.float64)),
    ("fp64", lambda z: torch.sum(z.to(torch.float64), 1)),
    ("integer", lambda z: torch.sum(z.to(torch.int32), 1)),
)

LAYOUT_CASES = (
    ("one_dim", lambda d: (torch.randn(300, device=d), 0), True),
    (
        "outer_strided",
        lambda d: (torch.randn(16384, 300, device=d)[::2], -1),
        True,
    ),
    (
        "non_last_inner",
        lambda d: (torch.randn(64, 300, 8, device=d).transpose(1, 2), 1),
        True,
    ),
    ("leading_one", lambda d: (torch.full((1, 4), -0.0, device=d), 0), True),
    ("singleton_dim1", lambda d: (_singleton_input(d), 1), True),
    (
        "noncollapsible",
        lambda d: (torch.randn(16, 8, 300, device=d)[::2], -1),
        False,
    ),
    (
        "size_one_outer_strided",
        lambda d: (
            torch.as_strided(torch.randn(16, device=d), (8, 1), (2, 1)),
            1,
        ),
        False,
    ),
)

SIGNED_ZERO_CASES = (
    ("multirow_1", 4, 1, True),
    ("multirow_2", 4, 2, True),
    ("multirow_4", 4, 4, True),
    ("persistent", 8192, 128, False),
)

FUSION_CASES = (
    "pointwise",
    "nested",
    "mix",
    "mix_append",
    "multi_kernel",
    "native_matmul",
    "sort",
    "multi_output",
)


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON
    and torch.version.hip is None
    and has_triton_reduction_ordering(),
    "requires CUDA, tl.ReductionOrdering, and the eager inner-tree implementation",
)
@skipIfNoCuteDSL
class StrictNumericsTest(TestCase):
    def setUp(self):
        super().setUp()
        env_patch = mock.patch.dict(os.environ, {"PYTORCH_SUM_INNER_TREE": "1"})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        torch.manual_seed(0)

    def _run(self, fn, *args, **cfg):
        with config.patch({"numerics": "strict", "force_disable_caches": True, **cfg}):
            torch._dynamo.reset()
            result, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        return result, "\n".join(codes)

    def _assert_bitwise_equal(self, eager, result):
        self.assertEqual(
            eager.contiguous().reshape(-1).view(torch.uint8),
            result.contiguous().reshape(-1).view(torch.uint8),
        )

    def _check_sum(self, device, shape, dim, dtype, keepdim=False, **cfg):
        x = torch.randn(*shape, device=device, dtype=dtype)

        def fn(z):
            return torch.sum(z, dim, keepdim=keepdim)

        eager = fn(x)
        result, code = self._run(fn, x, **cfg)
        self._assert_bitwise_equal(eager, result)
        self.assertIn(INNER_TREE_CALL, code)

    @dtypes(*DTYPES)
    @parametrize("case", SUM_CASES, name_fn=lambda c: c[0])
    def test_sum_bitwise(self, device, dtype, case):
        _, shape, dim = case
        self._check_sum(device, shape, dim, dtype)

    @parametrize("case", SUM_VARIANTS, name_fn=lambda c: c[0])
    def test_sum_variants(self, device, case):
        _, shape, dim, dtype, keepdim, cfg = case
        self._check_sum(device, shape, dim, dtype, keepdim, **cfg)

    def test_special_values_match_eager(self, device):
        values = [0.0, -0.0, torch.inf, -torch.inf, torch.nan, 1e20, -1e20, 1.0]
        x = torch.tensor(values, device=device).repeat(64, 38)[:, :300].contiguous()

        def fn(z):
            return z.sum(1)

        result, code = self._run(fn, x)
        self._assert_bitwise_equal(fn(x), result)
        self.assertIn(INNER_TREE_CALL, code)

    @parametrize("case", DYNAMIC_CASES, name_fn=lambda c: c[0])
    def test_dynamic_sum(self, device, case):
        _, sizes, cfg = case

        def fn(z):
            return torch.sum(z, 1)

        with config.patch({"numerics": "strict", "force_disable_caches": True, **cfg}):
            torch._dynamo.reset()
            compiled = torch.compile(fn, fullgraph=True, dynamic=True)
            for n in sizes:
                x = torch.randn(8, n, device=device)
                self._assert_bitwise_equal(fn(x), compiled(x))

    @parametrize("case", OUT_OF_SCOPE_CASES, name_fn=lambda c: c[0])
    def test_out_of_scope_uses_default_order(self, device, case):
        _, fn = case
        _, code = self._run(fn, torch.randn(64, 300, device=device))
        self.assertNotIn(INNER_TREE_CALL, code)

    @parametrize("case", LAYOUT_CASES, name_fn=lambda c: c[0])
    def test_layout_eligibility(self, device, case):
        _, make_input, eligible = case
        x, dim = make_input(device)

        def fn(z):
            return torch.sum(z, dim)

        result, code = self._run(fn, x)
        self.assertEqual(INNER_TREE_CALL in code, eligible)
        if eligible:
            self._assert_bitwise_equal(fn(x), result)

    def test_unbacked_reduction_size_uses_default_order(self, device):
        def fn(z):
            return z[z > 0].sum(0)

        x = torch.randn(1024, device=device)
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            result, code = self._run(fn, x)
        self.assertEqual(result, fn(x))
        self.assertNotIn(INNER_TREE_CALL, code)

    @dtypes(*DTYPES)
    @parametrize("case", SIGNED_ZERO_CASES, name_fn=lambda c: c[0])
    def test_signed_zero(self, device, dtype, case):
        _, rows, n, multirow = case
        x = torch.full((rows, n), -0.0, device=device, dtype=dtype)

        def fn(z):
            return torch.sum(z, 1)

        cfg = {"triton.persistent_reductions": False} if multirow else {}
        result, code = self._run(fn, x, **cfg)
        self._assert_bitwise_equal(fn(x), result)
        vector_size = vec_size(x.element_size())
        if multirow:
            num_loads = (n + vector_size - 1) // vector_size
            rblock = (1 << (num_loads - 1).bit_length()) * vector_size
        else:
            rblock = compute_inner_tree_params(n, 1, vector_size).batch_total_elements
        self.assertIn(f"R0_BLOCK: tl.constexpr = {rblock}", code)

    def _make_fusion_case(self, kind, device):
        cfg = {}
        expected_metrics = {}
        kernel_count = None
        result_index = None

        if kind == "pointwise":
            args = (
                torch.randn(8192, 300, device=device),
                torch.randn(8192, 1, device=device),
            )

            def fn(a, b):
                return torch.sum(a + b, -1)

        elif kind == "nested":
            batch, width, group = 64, 4096, 16
            args = (torch.randn(batch, width, device=device),)
            cfg = {"triton.nested_reduction": True}
            expected_metrics = {"codegen_nested_reduction": 0}

            def fn(x):
                outer = x.amax(-1, keepdim=True)
                y = torch.ops._inductor_test.realize(x + outer)
                return y.reshape(batch, width // group, group).sum(-1)

        elif kind in ("mix", "mix_append"):
            args = (torch.randn(32, 12000, device=device),)
            cfg = {
                "triton.mix_order_reduction": True,
                "triton.mix_order_reduction_non_strict_mode": True,
            }
            if kind == "mix":
                result_index = 0
                expected_metrics = {"codegen_mix_order_reduction": 0}

                def fn(x):
                    return x.sum(-1), x.prod(0)

            else:
                result_index = 2
                kernel_count = 2
                cfg.update(
                    max_fusion_buffer_group_pairwise_attempts=1,
                    split_reductions=False,
                )
                expected_metrics = {"codegen_mix_order_reduction": 1}

                def fn(x):
                    return x.prod(1), x.prod(0), x.sum(1)

        elif kind == "multi_kernel":
            args = (torch.randn(32, 12000, device=device),)
            cfg = {"triton.multi_kernel": True}

            def fn(x):
                return x.sum(1)

        elif kind == "native_matmul":
            args = (
                torch.randn(32, 4, device=device),
                torch.randn(4, 32, device=device),
            )
            cfg = {"triton.native_matmul": True}
            result_index = 1
            kernel_count = 2

            def fn(a, b):
                z = (a[:, None, :] + b.T[None, :, :]).contiguous()
                return a @ b, z.sum(-1)

        elif kind == "sort":
            args = (torch.randn(32, 513, device=device),)
            cfg = {"triton.decompose_sort_ops": True}
            result_index = 1
            kernel_count = 2

            def fn(x):
                return torch.sort(x, dim=1).values, x.sum(1)

        else:
            args = (torch.randn(32, 300, device=device),)
            cfg = {"online_softmax": True}
            result_index = 1
            kernel_count = 2

            def fn(x):
                return torch.softmax(x, -1), x.sum(-1)

        return fn, args, result_index, cfg, expected_metrics, kernel_count

    @parametrize("kind", FUSION_CASES)
    def test_fusion_preserves_strict_sum(self, device, kind):
        fn, args, index, cfg, expected_metrics, kernel_count = self._make_fusion_case(
            kind, device
        )
        eager = fn(*args)
        metrics.reset()
        result, code = self._run(fn, *args, **cfg)
        expected = eager if index is None else eager[index]
        actual = result if index is None else result[index]
        self._assert_bitwise_equal(expected, actual)
        self.assertEqual(code.count(INNER_TREE_CALL), 1)
        for metric, expected_value in expected_metrics.items():
            self.assertEqual(getattr(metrics, metric), expected_value)
        if kernel_count is not None:
            self.assertEqual(metrics.generated_kernel_count, kernel_count)
        if kind == "multi_kernel":
            self.assertNotIn("async_compile.multi_kernel(", code)
            self.assertIn("for r0_offset in", code)
        elif kind == "multi_output":
            self.assertEqual(eager[0], result[0])

    def test_combo_kernel_preserves_strict_sum_blocks(self, device):
        args = (
            torch.randn(32, 12000, device=device),
            torch.randn(32, 12000, device=device),
        )

        def fn(a, b):
            return a.sum(1), b.sum(1)

        eager = fn(*args)
        result, code = self._run(
            fn,
            *args,
            combo_kernels=True,
            combo_kernels_autotune=0,
            combo_kernel_peak_memory_pct_threshold=None,
        )
        for expected, actual in zip(eager, result, strict=True):
            self._assert_bitwise_equal(expected, actual)
        self.assertIn(INNER_TREE_CALL, code)
        self.assertNotIn("combo_grid_meta", code)

    @unittest.skipIf(not SM90OrLater, "requires TMA support")
    @parametrize("kind", ("multirow", "split"))
    def test_tma_preserves_strict_sum(self, device, kind):
        if kind == "multirow":
            x = torch.randn(128, 5, device=device)
        else:
            x = torch.zeros(1, 65536, device=device)
            params = compute_inner_tree_params(
                x.shape[1], x.shape[0], vec_size(x.element_size())
            )
            for batch, value in enumerate((1e20, 1, -1e20, 1)):
                x[0, batch * params.batch_total_elements] = value

        def fn(z):
            return torch.sum(z, 1)

        eager = fn(x)
        if kind == "split":
            self._assert_bitwise_equal(eager, torch.ones_like(eager))
        result, code = self._run(
            fn,
            x,
            assume_aligned_inputs=True,
            **{"triton.use_tensor_descriptor": True},
        )
        self._assert_bitwise_equal(eager, result)
        if kind == "split":
            self.assertEqual(code.count(INNER_TREE_CALL), 2)
        self.assertIn("tensor_descriptor" if kind == "split" else "tl.store", code)


instantiate_device_type_tests(StrictNumericsTest, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
