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

    def test_two_stage_row_ragged_split(self):
        # A PRIME row length: the chunk cannot divide it, so stage 1 has to clamp its
        # fold at the end of each row (ragged_chunk) instead of running into the next.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_general as kg

        x = torch.randn(8, 65537, device="cuda")
        (out,) = kg._two_stage_row(
            T.SumOps(acc=cutlass.Float32), "smoke_rag", x, [torch.float32], 1
        )
        self.assertEqual(out, x.sum(dim=1), atol=1e-1, rtol=1e-3)

    def test_two_stage_row_index_is_global(self):
        # gidx_from="chunk": the index a chunk reports must be the ABSOLUTE column, and
        # an exact tie must resolve first-wins as aten's argmax does.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_general as kg

        x = torch.zeros(8, 65537, device="cuda")
        x[:, 40000] = 1.0
        x[:, 50000] = 1.0  # tie with the above -> the lower column must win
        (idx,) = kg._two_stage_row(
            T.ArgMaxOps(acc=cutlass.Float32), "smoke_ragidx", x, [torch.int32], 1
        )
        self.assertEqual(idx, torch.full((8,), 40000, device="cuda", dtype=torch.int32))

    def test_family_has_exactly_one_cute_kernel(self):
        # The unification's claim, asserted rather than described: every axis compiles from the
        # SAME body. A second @cute.kernel anywhere in the family means an axis has been forked
        # back out, which is how a shared prologue, projection and store quietly stop being
        # shared.
        #
        # Source-level, but GLOBBED and matched on the decorator rather than one spelling. A
        # hardcoded file list misses a new kernel_foo.py, and `line == "@cute.kernel"` misses
        # `@cute.kernel  # note` and `@cutlass.cute.kernel` -- both the same fork with a
        # different source line, and both slipped past the earlier form. Runtime introspection
        # cannot replace this: @cute.kernel and @cute.jit both produce a plain function with the
        # same added attributes, so nothing distinguishes them after import.
        import pathlib
        import re

        from torch._native.ops import reductions

        root = pathlib.Path(reductions.__file__).parent
        deco = re.compile(r"@(?:\w+\.)*cute\.kernel\b")
        found = [
            f"{path.name}:{i}"
            # inner_tree_kernel.py holds the reference implementation's kernels, not this family's.
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
        # These checks exist to hold under `python -O`, where a plain `assert` is stripped
        # entirely. A test cannot observe the -O build from here, so what it CAN pin is that each
        # check is a real raise reached on the documented input -- which is what a stripped assert
        # would stop doing. Every check these kernels carry is covered.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_general as kg, kernel_xcta as xc

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
        # No reduced runs at all: the fold's decode would index vals[-1]. An empty KEPT list is
        # legal (a full reduction), which is why only the reduced side is refused here.
        with self.assertRaisesRegex(AssertionError, "at least one reduced run"):
            kg.ReduceBlock(trait, count=1, num_o=1, red_pairs=(), kept_pairs=())
        # The cross-CTA driver reshapes, so it needs a contiguous CUDA input too.
        with self.assertRaisesRegex(AssertionError, "CUDA"):
            xc.reduce_row_xcta(trait, "inv_xcta", xt, torch.float32)

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
