# Owner(s): ["module: dsl-native-ops"]
#
# Correctness tests for the eager CuTeDSL strict-numerics `sum` override
# (torch._native.ops.reductions_strict), which is active under
# torch._inductor.config.numerics == "strict". The override must compute a CORRECT
# sum -- verified here against the aten reference (numerics == "default")
#
# Bitwise eager == Inductor parity is covered separately in
# test/inductor/test_strict_numerics.py; this file only checks the eager kernel vs aten.

import unittest

import torch

from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)

import torch._inductor.config as inductor_config


# (name, shape, dim): persistent / looped / split / column / 3D / multi-dim / reduce-all,
# including non-power-of-2 and tiny N (the strict order handles all of them).
CASES = [
    ("row", (256, 4096), 1),
    ("row_nonpow2", (256, 300), 1),
    ("row_tiny", (256, 5), 1),
    ("row_looped", (256, 65536), 1),
    ("split_fewout", (8, 262144), 1),
    ("col", (4096, 256), 0),
    ("reduce_all", (512, 512), None),
    ("d3_mid", (16, 128, 64), 1),
    ("d3_multidim", (16, 128, 64), (1, 2)),
    ("d4_mid", (8, 64, 16, 8), 1),
]
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def _sum(x, dim, keepdim, numerics):
    with inductor_config.patch({"numerics": numerics}):
        if dim is None:
            return torch.sum(x)
        return torch.sum(x, dim, keepdim)


@skipIfNoCuteDSL
@unittest.skipIf(not TEST_CUDA, "CUDA required")
class TestStrictSum(TestCase):
    """Eager CuTeDSL strict `sum` override vs the aten reference."""

    @parametrize("case", CASES, name_fn=lambda c: c[0])
    @parametrize("dtype", DTYPES, name_fn=lambda d: str(d).split(".")[-1])
    def test_sum_matches_aten(self, case, dtype):
        _, shape, dim = case
        torch.manual_seed(0)
        x = torch.randn(*shape, device="cuda", dtype=dtype)
        out = _sum(x, dim, False, "strict")   # eager CuTeDSL strict override
        ref = _sum(x, dim, False, "default")  # aten
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    @parametrize("case", [c for c in CASES if c[2] is not None], name_fn=lambda c: c[0])
    def test_sum_keepdim_matches_aten(self, case):
        _, shape, dim = case
        torch.manual_seed(0)
        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        out = _sum(x, dim, True, "strict")
        ref = _sum(x, dim, True, "default")
        self.assertEqual(out.shape, ref.shape)  # keepdim shape must match aten
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_strided_matches_aten(self):
        # A transposed (strided) input reduces over a strided axis; the override handles it
        # (strided-direct / materialize) and must still match aten.
        torch.manual_seed(0)
        x = torch.randn(512, 300, device="cuda", dtype=torch.float32).t()
        out = _sum(x, 1, False, "strict")
        ref = _sum(x, 1, False, "default")
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_routes_to_cutedsl(self):
        # Non-pow2 N: strict's fixed order differs bit-for-bit from aten, so equality would
        # mean the override silently fell back to aten (a routing bug).
        torch.manual_seed(0)
        x = torch.randn(256, 300, device="cuda", dtype=torch.float32)
        self.assertFalse(torch.equal(_sum(x, 1, False, "strict"), _sum(x, 1, False, "default")))

    def test_unsupported_falls_back_to_aten(self):
        # int inputs and the dtype= kwarg are not eligible -> must fall through to aten.
        with inductor_config.patch({"numerics": "strict"}):
            xi = torch.randint(0, 9, (256, 256), device="cuda", dtype=torch.int32)
            self.assertEqual(torch.sum(xi, 1), xi.sum(1))
            xf = torch.randn(256, 256, device="cuda", dtype=torch.float16)
            self.assertEqual(
                torch.sum(xf, 1, dtype=torch.float32), xf.sum(1, dtype=torch.float32)
            )

    def test_invalid_dim_raises_like_aten(self):
        # Out-of-range / duplicate dims must raise the same errors as aten (fall through).
        x = torch.randn(4, 6, device="cuda")
        with inductor_config.patch({"numerics": "strict"}):
            with self.assertRaises(IndexError):
                torch.sum(x, 2)
            with self.assertRaises(RuntimeError):
                torch.sum(x, (0, 0))


instantiate_parametrized_tests(TestStrictSum)


if __name__ == "__main__":
    run_tests()
