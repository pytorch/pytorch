# Owner(s): ["module: dsl-native-ops"]
#
# Minimal smoke test for the general (K0) reduction kernel + dispatcher. This only
# proves the kernel COMPILES and runs the general ReduceBlock path end to end on a
# tiny input; real numeric coverage arrives with the reduction OVERRIDES (via the
# numpy-referenced OpInfo suites) in a later commit. Reduces a MIDDLE dim so the
# dispatcher stays in the general path and does not pull in the row/col/xcta fast
# kernels (added in later commits).

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestKernelGeneral(TestCase):
    def test_reduce_dim_general_path(self):
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_general as kg

        x = torch.randn(8, 16, 32, device="cuda")
        out = kg.reduce_dim(
            T.SumOps(acc=cutlass.Float32), "smoke", x, [1], torch.float32
        )
        self.assertEqual(out, x.sum(dim=1), atol=1e-2, rtol=1e-2)

    def test_internal_invariants_raise(self):
        # These checks exist to hold under `python -O`, where a plain `assert` is stripped
        # entirely. A test cannot observe the -O build from here, so what it CAN pin is that each
        # check is a real raise reached on the documented input -- which is what a stripped assert
        # would stop doing. Every check these kernels carry is covered.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_general as kg

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
        # A `# noqa: S101` is a check that silently does nothing under -O, and lint's own rule can
        # be silenced. Assert the absence directly, so re-adding one fails here and not only in
        # lint.
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
