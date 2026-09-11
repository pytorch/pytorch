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

from torch._native.ops.reductions import (
    kernel_general as kg,
    kernel_xcta as xc,
    traits as T,
)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestKernelGeneral(TestCase):
    def test_reduce_dim_general_path(self):
        x = torch.randn(8, 16, 32, device="cuda")
        out = kg.reduce_dim(
            T.SumOps(acc=cutlass.Float32), "smoke", x, [1], torch.float32
        )
        self.assertEqual(out, x.sum(dim=1), atol=1e-2, rtol=1e-2)

    def test_two_stage_row_ragged_split(self):
        # A prime row requires ragged stage-1 chunks that stop at row boundaries.
        x = torch.randn(8, 65537, device="cuda")
        (out,) = kg._two_stage_row(
            T.SumOps(acc=cutlass.Float32), "smoke_rag", x, [torch.float32], 1
        )
        self.assertEqual(out, x.sum(dim=1), atol=1e-1, rtol=1e-3)

    def test_two_stage_row_index_is_global(self):
        # Chunk-local reductions must report global columns and preserve first-wins ties.
        x = torch.zeros(8, 65537, device="cuda")
        x[:, 40000] = 1.0
        x[:, 50000] = 1.0  # tie with the above -> the lower column must win
        (idx,) = kg._two_stage_row(
            T.ArgMaxOps(acc=cutlass.Float32), "smoke_ragidx", x, [torch.int32], 1
        )
        self.assertEqual(idx, torch.full((8,), 40000, device="cuda", dtype=torch.int32))

    def test_family_has_exactly_one_cute_kernel(self):
        # Assert every axis still shares one kernel. Glob files and match qualified or
        # annotated decorators so new drivers and spellings cannot evade the check.
        # Runtime introspection cannot distinguish @cute.kernel from @cute.jit wrappers.
        import pathlib
        import re

        from torch._native.ops import reductions

        root = pathlib.Path(reductions.__file__).parent
        deco = re.compile(r"@(?:\w+\.)*cute\.kernel\b")
        found = [
            f"{path.name}:{i}"
            # Exclude the reference implementation.
            for path in sorted(root.glob("*.py"))
            if path.name != "inner_tree_kernel.py"
            for i, line in enumerate(path.read_text().splitlines(), 1)
            if deco.match(line.strip())
        ]
        self.assertEqual(
            len(found), 1, f"expected one kernel in the family, got {found}"
        )
        self.assertTrue(found[0].startswith("tile.py"), f"the body moved: {found}")

    def test_internal_invariants_raise(self):
        # Each invariant must raise explicitly because python -O strips asserts. Exercise every
        # check on its documented invalid input.
        trait = T.SumOps(acc=cutlass.Float32)
        # reduce-all needs a flat view, so a transposed input has to be refused, not reshaped.
        xt = torch.randn(64, 128, device="cuda").t()
        with self.assertRaisesRegex(AssertionError, "contiguous CUDA input"):
            kg._reduce_all(trait, "inv", xt, [torch.float32], 1, 128, 4)
        # The general path needs a CUDA input; a CPU tensor is refused, not silently run.
        with self.assertRaisesRegex(AssertionError, "need a CUDA input"):
            kg._reduce(trait, "inv", torch.randn(8, 8), [1], [torch.float32], 1)
        # The magic-division decode is exact only below 2^31.
        with self.assertRaisesRegex(AssertionError, "count and num_o"):
            kg.ReduceBlock(
                trait, count=2**31, num_o=1, red_pairs=((2**31, 1),), kept_pairs=()
            )
        # A missing reduced run would index vals[-1]; empty kept runs remain valid.
        with self.assertRaisesRegex(AssertionError, "at least one reduced run"):
            kg.ReduceBlock(trait, count=1, num_o=1, red_pairs=(), kept_pairs=())
        # The reshaping cross-CTA path also requires contiguous CUDA input.
        with self.assertRaisesRegex(AssertionError, "CUDA"):
            xc.reduce_row_xcta(trait, "inv_xcta", xt, torch.float32)

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
