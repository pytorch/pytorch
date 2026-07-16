# Owner(s): ["module: inductor"]
"""Bitwise parity tests for ``torch._inductor.config.numerics == "strict"``.

Strict mode makes Inductor emit a fixed, layout-independent reduction order (Triton
``INNER_TREE``, plus a materialized-contiguous shuffle for strided reductions) so that
``torch.compile`` matches eager bitwise. ``eager`` here is the CuTeDSL strict reduction
override (the PR stacked below), which reads the same ``torch._inductor.config.numerics``
flag; importing it registers the eager ``torch.sum`` override.

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


# torch._native registers the eager CuTeDSL strict `sum` override at import (the PR stacked
# below, ops/reductions_strict), so torch.sum routes to it under numerics="strict" -- no manual
# import needed here. Skip the suite if that override / the CuTeDSL runtime isn't available.
try:
    _HAS_EAGER_STRICT = torch._native.ops.reductions_strict.is_available()
except Exception:
    _HAS_EAGER_STRICT = False

# op name -> fn(x, dim). Extend with mean/prod/amax/...
STRICT_OPS = {
    "sum": lambda z, d: torch.sum(z, d) if d is not None else torch.sum(z),
}

# (name, shape, dim): geometry coverage -- row (contiguous), column (strided), reduce-all,
# 3D mid/outer/multi-dim, 4D; sizes span persistent / looped / split reduction structures.
CASES = [
    ("row_persistent", (8192, 256), 1),
    ("row_nonpow2", (8192, 300), 1),
    ("row_tiny", (8192, 5), 1),
    ("row_looped", (4096, 8192), 1),
    ("row_split", (8, 65536), 1),
    ("row_split_wide", (64, 262144), 1),
    ("row_split_manyout", (200, 262144), 1),
    ("col", (65536, 8), 0),
    ("col_nonpow2", (777, 512), 0),
    ("reduce_all", (512, 512), None),
    ("d3_mid", (16, 512, 32), 1),
    ("d3_outer", (512, 16, 32), 0),
    ("d3_multidim", (16, 128, 64), (1, 2)),
    ("d4_mid", (8, 128, 16, 8), 1),
]

DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

AUTOTUNE_CASES = [c for c in CASES if c[0] in ("row_persistent", "row_split")]


@unittest.skipUnless(
    has_triton_reduction_ordering() and _HAS_EAGER_STRICT,
    "requires a Triton build with tl.ReductionOrdering and the eager CuTeDSL strict "
    "reduction (PR stacked below)",
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
        with config.patch({"numerics": "strict", **cfg}):
            eager = f(x)  # routes to the eager CuTeDSL strict override
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(f, fullgraph=True), x)
        self.assertTrue(torch.equal(eager, result))  # bitwise-equal to eager
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
        # keepdim=True must give the CORRECT (un-permuted) shape AND bitwise-match eager --
        # the strided make-contiguous permutes reduced dims, so the result is un-permuted back.
        _, shape, dim = case
        x = torch.randn(*shape, device=GPU_TYPE, dtype=torch.float32)
        f = lambda z: torch.sum(z, dim, keepdim=True)
        with config.patch({"numerics": "strict"}):
            eager = f(x)
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
        # negative control: without strict, Inductor drifts from eager (the drift strict fixes).
        x = torch.randn(1024, 1024, device=GPU_TYPE)
        fn = lambda z: torch.sum(z, 0)
        with config.patch({"numerics": "strict"}):
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
