# Owner(s): ["module: inductor"]
"""Tests for strict inner-contiguous sum ordering."""

import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._triton import has_triton_reduction_ordering


# Strict sum consumes eager's planning API (torch._native.ops.sum.inner_tree_plan), which the
# eager INNER_TREE PR provides and this PR stacks on. Skip the suite if it isn't checked out.
try:
    from torch._native.ops.sum.inner_tree_plan import (  # noqa: F401
        compute_inner_tree_params,
    )

    _HAS_INNER_TREE_PLAN = True
except ImportError:
    _HAS_INNER_TREE_PLAN = False


# Persistent, looped, and split inner reductions, including ragged extents.
CASES = [
    ("row_persistent", (8192, 256), 1),
    ("row_nonpow2", (8192, 300), 1),
    ("row_tiny", (8192, 5), 1),
    ("row_looped", (4096, 8192), 1),
    ("row_split", (8, 65536), 1),
    ("row_split_ragged", (8, 65537), 1),
    ("row_split_wide", (64, 262144), 1),
    ("row_split_manyout", (200, 262144), 1),
    ("d1", (65536,), 0),
    ("d1_big", (1048576,), 0),
    ("d1_nonpow2", (5000,), 0),
]

DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

AUTOTUNE_CASES = [c for c in CASES if c[0] in ("row_persistent", "row_split")]


@unittest.skipUnless(
    HAS_GPU
    and GPU_TYPE == "cuda"
    and torch.version.hip is None
    and has_triton_reduction_ordering()
    and _HAS_INNER_TREE_PLAN,
    "requires CUDA, tl.ReductionOrdering, and the eager inner-tree planning API",
)
@config.patch({"force_disable_caches": True})
@instantiate_parametrized_tests
class StrictNumericsTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        torch._dynamo.reset()

    def _code(self, shape, dim, dtype, **cfg):
        x = torch.randn(*shape, device=GPU_TYPE, dtype=dtype)

        def fn(z):
            return torch.sum(z, dim)

        with config.patch({"numerics": "strict", **cfg}):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        return fn(x), result, code

    def _check(self, shape, dim, dtype, **cfg):
        eager, result, code = self._code(shape, dim, dtype, **cfg)
        self.assertEqual(eager, result, atol=0, rtol=0)
        self.assertIn(
            "reduction_ordering=tl.constexpr(tl.ReductionOrdering.INNER_TREE)", code
        )
        return code

    @parametrize("case", CASES, name_fn=lambda c: c[0])
    @parametrize("dtype", DTYPES)
    def test_sum_bitwise(self, case, dtype):
        _, shape, dim = case
        self._check(shape, dim, dtype)

    def test_sum_keepdim(self):
        x = torch.randn(64, 300, device=GPU_TYPE)

        def f(z):
            return torch.sum(z, 1, keepdim=True)

        eager = f(x)
        with config.patch({"numerics": "strict"}):
            torch._dynamo.reset()
            result, _ = run_and_get_code(torch.compile(f, fullgraph=True), x)
        self.assertEqual(eager, result, atol=0, rtol=0)

    @parametrize("case", AUTOTUNE_CASES, name_fn=lambda c: c[0])
    def test_sum_matches_eager_under_autotune(self, case):
        _, shape, dim = case
        self._check(shape, dim, torch.float32, max_autotune=True)

    def test_dynamic_matches_eager(self):
        # A divisibility hint must not enable split reduction for a dynamic extent.
        def fn(z):
            return torch.sum(z, 1)

        with config.patch({"numerics": "strict"}):
            torch._dynamo.reset()
            compiled = torch.compile(fn, fullgraph=True, dynamic=True)
            for n in (65536, 65537):
                x = torch.randn(8, n, device=GPU_TYPE)
                self.assertEqual(fn(x), compiled(x), atol=0, rtol=0)

    @parametrize(
        "case",
        [
            ("column", lambda z: torch.sum(z, 0)),
            ("multidim", lambda z: torch.sum(z, (0, 1))),
            ("non_sum", lambda z: torch.amax(z, 1)),
        ],
        name_fn=lambda c: c[0],
    )
    def test_out_of_scope_uses_default_order(self, case):
        _, fn = case
        x = torch.randn(64, 300, device=GPU_TYPE)
        with config.patch({"numerics": "strict"}):
            torch._dynamo.reset()
            _, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertNotIn("ReductionOrdering.INNER_TREE", code)


if __name__ == "__main__":
    run_tests()
