# Owner(s): ["module: dsl-native-ops"]
# Smoke-test K0 compilation and dispatch on a middle-dimension reduction, which forces
# the general path. Override OpInfo suites provide full numerical coverage.

import sys
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TEST_CUTEDSL, TestCase


# Guard before importing cutlass-dependent kernels to avoid collection errors.
if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

import cutlass

from torch._native.ops._cutedsl import traits as T
from torch._native.ops.reductions import kernel_general as kg


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestKernelGeneral(TestCase):
    def test_reduce_dim_general_path(self):
        x = torch.randn(8, 16, 32, device="cuda")
        out = kg.reduce_dim(
            T.SumOps(acc=cutlass.Float32), "smoke", x, [1], torch.float32
        )
        self.assertEqual(out, x.sum(dim=1), atol=1e-2, rtol=1e-2)

    def test_internal_invariants_raise(self):
        # Each invariant must raise explicitly because python -O strips asserts. Exercise every
        # check on its documented invalid input.
        trait = T.SumOps(acc=cutlass.Float32)
        # reduce-all needs a flat view, so a transposed input has to be refused, not reshaped.
        xt = torch.randn(64, 128, device="cuda").t()
        with self.assertRaisesRegex(AssertionError, "contiguous CUDA input"):
            kg.reduce_all(trait, "inv", xt, torch.float32)
        # The general path needs a CUDA input; a CPU tensor is refused, not silently run.
        with self.assertRaisesRegex(AssertionError, "need a CUDA input"):
            kg._reduce(trait, "inv", torch.randn(8, 8), [1], [torch.float32], 1)
        # The magic-division decode is exact only below 2^31.
        with self.assertRaisesRegex(AssertionError, "count and num_o"):
            kg.ReduceBlock(
                trait, count=2**31, num_o=1, red_pairs=((2**31, 1),), kept_pairs=()
            )

    def test_no_suppressed_asserts_survive(self):
        # Reject suppressed asserts directly because lint itself can be silenced.
        import pathlib

        from torch._native.ops import reductions

        root = pathlib.Path(reductions.__file__).parent
        offenders = [
            f"{path.name}:{i}"
            for path in sorted(root.glob("*.py"))
            for i, line in enumerate(path.read_text().splitlines(), 1)
            if "noqa: S101" in line
        ]
        # inner_tree_kernel.py is the reference implementation's, not this stack's.
        offenders = [o for o in offenders if not o.startswith("inner_tree_kernel.py")]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    run_tests()
