# Owner(s): ["module: dsl-native-ops"]
# Smoke-test K0 compilation and dispatch on a middle-dimension reduction, which forces
# the general path. Override OpInfo suites provide full numerical coverage.

import pathlib
import sys
import unittest

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TEST_CUTEDSL, TestCase


# Guard before importing cutlass-dependent kernels to avoid collection errors.
if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from torch._native.cutedsl import launch as _L
from torch._native.ops import reductions
from torch._native.ops.reductions import (
    kernel_general as kg,
    kernel_xcta as xc,
    tile,
    traits as T,
)


class _DecodeProbe:
    def __init__(self, wide):
        self.wide = wide

    @cute.jit
    def __call__(self, out, exts, strides, linear, stream):
        divs = (
            exts
            if cutlass.const_expr(self.wide)
            else [cute.FastDivmodDivisorV2(e) for e in exts]
        )
        self.kernel(out, divs, strides, linear).launch(
            grid=[1, 1, 1], block=[1, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, out, divs, strides, linear):
        out[0] = tile._decode_offset(
            linear,
            divs,
            strides,
            cutlass.const_expr(2),
            cutlass.const_expr(self.wide),
        )


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "Hopper+ required")
class TestKernelGeneral(TestCase):
    def test_reduce_dim_general_path(self):
        x = torch.randn(8, 16, 32, device="cuda")
        out = kg.reduce_dim(
            T.SumOps(acc=cutlass.Float32), "smoke", x, [1], torch.float32
        )
        torch.testing.assert_close(out, x.sum(dim=1), atol=1e-2, rtol=1e-2)

    def test_two_stage_row_ragged_split(self):
        # A prime row requires ragged stage-1 chunks that stop at row boundaries.
        x = torch.randn(8, 65537, device="cuda")
        (out,) = kg._two_stage_row(
            T.SumOps(acc=cutlass.Float32), "smoke_rag", x, [torch.float32], 1
        )
        torch.testing.assert_close(out, x.sum(dim=1), atol=1e-1, rtol=1e-3)

    def test_two_stage_row_index_is_global(self):
        # Chunk-local reductions must report global columns and preserve first-wins ties.
        x = torch.zeros(8, 65537, device="cuda")
        x[:, 40000] = 1.0
        x[:, 50000] = 1.0  # tie with the above -> the lower column must win
        (idx,) = kg._two_stage_row(
            T.ArgMaxOps(acc=cutlass.Float32), "smoke_ragidx", x, [torch.int32], 1
        )
        self.assertEqual(idx, torch.full((8,), 40000, device="cuda", dtype=torch.int32))

    def test_int64_mixed_radix_decode(self):
        """Decode logical indices beyond INT32_MAX without truncating strides or extents."""
        cases = (
            (False, (65537, 65539), (3, 5), (1 << 31) + 17),
            (True, ((1 << 31) + 17, 3), (7, 11), (1 << 31) + 23),
        )
        for wide, extents, strides, linear in cases:
            out = torch.empty(1, dtype=torch.int64, device="cuda")
            extent_type = Int64 if wide else Int32
            fn = _L.compile_kernel(
                _DecodeProbe(wide),
                _L.fake_compact(cutlass.Int64, (_L.sym(),)),
                [extent_type(e) for e in extents],
                [Int64(s) for s in strides],
                Int64(linear),
                _L.stream(),
            )
            fn(
                out,
                [extent_type(e) for e in extents],
                [Int64(s) for s in strides],
                Int64(linear),
                _L.stream(),
            )
            q, r = divmod(linear, extents[0])
            expected = torch.tensor([r * strides[0] + q * strides[1]], device="cuda")
            self.assertEqual(out, expected)

    def test_uniform_welford_tree(self):
        """Equal-count trees must retain Welford accuracy without serial counts."""
        rows, count = 17, 1024
        x = torch.randn(rows, count, device="cuda")
        out = torch.empty(rows, device="cuda")
        op = kg.ReduceBlock(
            T.WelfordOps(correction=1, acc=cutlass.Float32),
            count=count,
            num_o=rows,
            red_pairs=((count, 1),),
            kept_pairs=((rows, count),),
            tree_count=count,
        )
        kg._launch(
            op, ("test_uniform_welford_tree",) + op.cache_sig, [x.flatten()], [out]
        )
        self.assertEqual(out, x.var(dim=1), atol=1e-5, rtol=1e-5)

    def test_internal_invariants_raise(self):
        # Each invariant must raise explicitly because python -O strips asserts. Exercise every
        # check on its documented invalid input.
        trait = T.SumOps(acc=cutlass.Float32)
        # reduce-all has no flat view of a transposed input, so it ROUTES that through the general
        # path (which addresses via the TI decode) rather than refusing it.
        xt = torch.randn(64, 128, device="cuda").t()
        (flat,) = kg._reduce_all(trait, "inv", xt, [torch.float32], 1, 128, 4)
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(flat, xt.double().sum().float(), atol=1e-5, rtol=1e-5)
        # The general path needs a CUDA input; a CPU tensor is refused, not silently run.
        with self.assertRaisesRegex(AssertionError, "need a CUDA input"):
            kg._reduce(trait, "inv", torch.randn(8, 8), [1], [torch.float32], 1)
        # Ordinary geometry keeps the measured Int32 loop; large geometry selects Int64.
        narrow = kg.ReduceBlock(
            trait, count=2**31 - 1, num_o=1, red_pairs=((2**31 - 1, 1),), kept_pairs=()
        )
        wide = kg.ReduceBlock(
            trait, count=2**31, num_o=1, red_pairs=((2**31, 1),), kept_pairs=()
        )
        wide_output = kg.ReduceBlock(
            trait, count=1, num_o=2**31, red_pairs=(), kept_pairs=((2**31, 1),)
        )
        self.assertIs(type(kg._geom_args(narrow)[0]), Int32)
        self.assertIs(type(kg._geom_args(wide)[0]), Int64)
        self.assertIs(type(kg._geom_args(wide_output)[0]), Int32)
        # No reduced runs at all is LEGAL -- an extent-1 reduced axis coalesces away in TI. Pinned
        # here because the arm that allows it is one line in tile._decode_offset, and a crash.
        self.assertEqual(
            kg.ReduceBlock(
                trait, count=1, num_o=1, red_pairs=(), kept_pairs=()
            ).npairs_red,
            0,
        )
        # The reshaping cross-CTA path also requires contiguous CUDA input.
        with self.assertRaisesRegex(AssertionError, "CUDA"):
            xc.reduce_row_xcta(trait, "inv_xcta", xt, torch.float32)

    def test_no_suppressed_asserts_survive(self):
        # Reject suppressed asserts directly because lint itself can be silenced.
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
