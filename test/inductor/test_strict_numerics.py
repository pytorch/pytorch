# Owner(s): ["module: inductor"]
"""Bitwise parity tests for ``torch._inductor.config.numerics == "strict"``.

Strict mode makes Inductor emit a fixed, layout-independent reduction order (Triton
``INNER_TREE``, tree-then-linear accumulation, shared R0_BLOCK/split + persistent/loop
thresholds) so that ``torch.compile`` matches EAGER (ATen) bit-for-bit. In this env ATen's
reduction is itself INNER_TREE (tree-then-linear), so ``numerics`` only affects the Inductor
codegen -- eager ``torch.sum`` is the reference and is unaffected by the flag.

Add a bitwise op by adding one entry to ``STRICT_OPS`` -- it inherits the whole matrix.
"""

import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._triton import has_triton_reduction_ordering


# op name -> fn(x, dim). Extend with mean/prod/amax/...
STRICT_OPS = {
    "sum": lambda z, d: torch.sum(z, d) if d is not None else torch.sum(z),
}

# (name, shape, dim): inner (contiguous) reduced dim only -- 2D row reductions (dim=-1) and 1D;
# sizes span persistent / looped / split reduction structures, pow2 + non-pow2. Column/strided,
# full-reduce, and multi-dim reductions are out of scope (eager runs classic ATen there).
CASES = [
    ("row_persistent", (8192, 256), 1),
    ("row_nonpow2", (8192, 300), 1),
    ("row_tiny", (8192, 5), 1),
    ("row_looped", (4096, 8192), 1),
    ("row_split", (8, 65536), 1),
    ("row_split_wide", (64, 262144), 1),
    ("row_split_manyout", (200, 262144), 1),
    ("d1", (65536,), 0),
    ("d1_big", (1048576,), 0),
    ("d1_nonpow2", (5000,), 0),
]

DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

AUTOTUNE_CASES = [c for c in CASES if c[0] in ("row_persistent", "row_split")]


@unittest.skipUnless(
    has_triton_reduction_ordering(),
    "requires a Triton build with tl.ReductionOrdering",
)
@config.patch({"force_disable_caches": True})
@instantiate_parametrized_tests
class StrictNumericsTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        torch._dynamo.reset()

    def _check(self, op, shape, dim, dtype, **cfg):
        fn = STRICT_OPS[op]
        x = torch.randn(*shape, device=GPU_TYPE, dtype=dtype)
        f = lambda z: fn(z, dim)
        eager = f(x)  # eager ATen (INNER_TREE here); numerics only affects Inductor codegen
        with config.patch({"numerics": "strict", **cfg}):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
        self.assertTrue(torch.equal(eager, result))  # bitwise-equal to eager ATen
        self.assertIn(  # strict emitted the fixed order
            "reduction_ordering=tl.constexpr(tl.ReductionOrdering.INNER_TREE)", code
        )
        return code

    @parametrize("case", CASES, name_fn=lambda c: c[0])
    @parametrize("dtype", DTYPES)
    def test_sum_bitwise(self, case, dtype):
        _, shape, dim = case
        self._check("sum", shape, dim, dtype)

    @parametrize(
        "case",
        [c for c in CASES if c[2] is not None],
        name_fn=lambda c: c[0],
    )
    def test_sum_keepdim(self, case):
        # keepdim=True must give the CORRECT shape AND bitwise-match eager ATen.
        _, shape, dim = case
        x = torch.randn(*shape, device=GPU_TYPE, dtype=torch.float32)
        f = lambda z: torch.sum(z, dim, keepdim=True)
        eager = f(x)
        with config.patch({"numerics": "strict"}):
            torch._dynamo.reset()
            result, _ = run_and_get_code(torch.compile(f, fullgraph=True), x)
        self.assertEqual(tuple(result.shape), tuple(eager.shape))
        self.assertTrue(torch.equal(eager, result))

    @parametrize("case", AUTOTUNE_CASES, name_fn=lambda c: c[0])
    @parametrize(
        "mode",
        [
            {"max_autotune": True},
            {"max_autotune": True, "coordinate_descent_tuning": True},
        ],
        name_fn=lambda m: "cdt" if "coordinate_descent_tuning" in m else "autotune",
    )
    def test_sum_matches_eager_under_autotune(self, case, mode):
        # a different autotuned num_warps/XBLOCK must still match eager; representative
        # subset only (config-invariance is geometry/dtype-independent, autotune is slow).
        _, shape, dim = case
        self._check("sum", shape, dim, torch.float32, **mode)

    def test_default_drifts_from_eager(self):
        # negative control: without strict, Inductor drifts from eager ATen (the drift strict fixes).
        x = torch.randn(1024, 1024, device=GPU_TYPE)
        fn = lambda z: torch.sum(z, 0)
        eager = fn(x)
        torch._dynamo.reset()
        with config.patch({"numerics": "default"}):
            default, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertNotIn("reduction_ordering", code)
        self.assertFalse(torch.equal(eager, default))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
