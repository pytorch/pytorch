# Owner(s): ["module: dsl-native-ops"]
# Smoke-test K0 compilation and dispatch on a middle-dimension reduction, which forces
# the general path. Override OpInfo suites provide full numerical coverage.

import pathlib
import re
import sys
import unittest
from unittest import mock

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
    kernel_rowtile as rt,
    kernel_xcta as xc,
    tile,
    traits as T,
)


class _DecodeProbe:
    def __init__(self, native_div):
        self.native_div = native_div

    @cute.jit
    def __call__(self, out, exts, strides, linear, stream):
        divs = (
            exts
            if cutlass.const_expr(self.native_div)
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
            cutlass.const_expr(self.native_div),
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

    def test_extent_one_reduced_axis(self):
        # TensorIterator coalesces away the reduced run; the kernel must fold one value.
        x = torch.randn(4, 1, 5, device="cuda")
        out = torch.empty(4, 5, device="cuda")
        red, kept = kg._ti_pairs(x, kg._probe(x, {1}))
        op = kg.ReduceBlock(
            T.SumOps(acc=cutlass.Float32),
            count=1,
            num_o=out.numel(),
            red_pairs=red,
            kept_pairs=kept,
        )
        self.assertEqual(op.npairs_red, 0)
        kg._launch(op, ("unit_red", "unit_red"), [kg._flat(x)], [out.reshape(-1)])
        self.assertEqual(out, x.sum(dim=1))

    def test_wide_geometry_runs_kernel(self):
        trait = T.SumOps(acc=cutlass.Float32)

        # Exercise Int64 count arithmetic without allocating or reducing 2**31 elements.
        x = torch.tensor([3.0], device="cuda")
        out = torch.empty_like(x)
        wide_count = kg.ReduceBlock(
            trait,
            count=2**31,
            num_o=1,
            red_pairs=((65536, 1), (32768, 65536)),
            kept_pairs=(),
            flat_tail=True,
            limit=1,
        )
        kg._launch(wide_count, ("wide", "wide_count"), [x], [out])
        self.assertEqual(out, x)

        # Lower only the selection and grid limits to run the real wide-output split.
        x = torch.arange(6, dtype=torch.float32, device="cuda")
        out = torch.empty_like(x)
        with (
            mock.patch.object(kg, "_INT32_LIMIT", 4),
            mock.patch.object(kg, "_GRID_X_MAX", 3),
        ):
            wide_output = kg.ReduceBlock(
                trait,
                count=1,
                num_o=6,
                red_pairs=(),
                kept_pairs=((3, 1), (2, 3)),
            )
            kg._launch(wide_output, ("wide", "wide_output"), [x], [out])
        self.assertEqual(out, x)

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

    def test_general_split_layouts_and_welford(self):
        """Split strided, offset, multi-run, and two-output reductions correctly."""
        base = torch.randn(4097, 9, 17, device="cuda")
        x = base[1::2]
        with torch.backends.python_native.cutedsl.disabled():
            expected_var, expected_mean = torch.var_mean(x, dim=0)
            expected_total = x.sum(dim=(0, 2))
        with mock.patch.object(
            kg, "_two_stage_general", wraps=kg._two_stage_general
        ) as split:
            var, mean = kg.reduce_dim2(
                T.VarMeanOps(correction=1, acc=cutlass.Float32),
                "split_var_mean",
                x,
                [0],
                [torch.float32, torch.float32],
            )
            total = kg.reduce_dim(
                T.SumOps(acc=cutlass.Float32),
                "split_multi_run",
                x,
                [0, 2],
                torch.float32,
            )

        self.assertEqual(split.call_count, 2)
        self.assertEqual(var, expected_var, atol=1e-5, rtol=1e-4)
        self.assertEqual(mean, expected_mean, atol=1e-5, rtol=1e-4)
        self.assertEqual(total, expected_total, atol=1e-3, rtol=1e-4)

        red, kept = kg._ti_pairs(x, kg._probe(x, {0, 2}))
        declined = kg._two_stage_general(
            T.ArgMaxOps(acc=cutlass.Float32),
            "split_index_decline",
            x,
            red,
            kept,
            num_o=x.shape[1],
            count=x.shape[0] * x.shape[2],
            out_dtypes=[torch.int32],
            nouts=1,
        )
        self.assertIsNone(declined)

    def test_reduce_all2_route(self):
        """Route one-element two-output reductions through reduce_all2."""
        x = torch.randn(1, 65537, device="cuda")
        with mock.patch.object(kg, "reduce_all2", wraps=kg.reduce_all2) as reduce_all2:
            lo, hi = kg.reduce_dim2(
                T.AMinMaxOps(acc=cutlass.Float32),
                "reduce_all2",
                x,
                [1],
                [torch.float32, torch.float32],
            )
        self.assertEqual(reduce_all2.call_count, 1)
        with torch.backends.python_native.cutedsl.disabled():
            expected_lo, expected_hi = torch.aminmax(x, dim=1)
        self.assertEqual(lo, expected_lo)
        self.assertEqual(hi, expected_hi)

    def test_reduce_all_grid_stride_index_and_cache_key(self):
        """Carry global indices and separate fp32/fp64 stage-2 kernels."""
        n = (1 << 20) + 3
        winner = 700000
        for dtype, acc in (
            (torch.float32, cutlass.Float32),
            (torch.float64, cutlass.Float64),
        ):
            x = torch.zeros(n, device="cuda", dtype=dtype)
            x[winner] = 1
            x[winner + 100000] = 1
            out = kg.reduce_all(
                T.ArgMaxOps(acc=acc, idx=Int64),
                "index_cache_key",
                x,
                torch.int64,
            )
            with self.subTest(dtype=dtype):
                self.assertEqual(out, winner)

    def test_noncontiguous_reduce_all_index_is_logical(self):
        """Report row-major logical indices for non-contiguous inputs."""
        x = torch.zeros(64, 128, device="cuda").t()
        x[0, 1] = 1
        out = kg.reduce_all(
            T.ArgMaxOps(acc=cutlass.Float32, idx=Int64),
            "noncontiguous_index",
            x,
            torch.int64,
        )
        self.assertEqual(out, 1)

    def test_empty_inner_tree_reduce_all(self):
        """Use the reduction identity when an empty input has no tree order."""
        x = torch.empty(0, device="cuda")
        with mock.patch.object(rt, "inner_tree_order_enabled", return_value=True):
            out = kg.reduce_all(
                T.SumOps(acc=cutlass.Float32),
                "empty_inner_tree",
                x,
                torch.float32,
            )
        self.assertEqual(out, torch.zeros((), device="cuda"))

    def test_int64_mixed_radix_decode(self):
        """Decode logical indices beyond INT32_MAX without truncating strides or extents."""
        cases = (
            (False, Int32, (65537, 65539), (3, 5), (1 << 31) + 17),
            (True, Int32, (65537, 65539), (3, 5), (1 << 32) + 17),
            (True, Int64, ((1 << 31) + 17, 3), (7, 11), (1 << 31) + 23),
        )
        for native_div, extent_type, extents, strides, linear in cases:
            out = torch.empty(1, dtype=torch.int64, device="cuda")
            fn = _L.compile_kernel(
                _DecodeProbe(native_div),
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

    def test_family_has_exactly_one_cute_kernel(self):
        # Assert every axis still shares one kernel. Glob files and match qualified or
        # annotated decorators so new drivers and spellings cannot evade the check.
        # Runtime introspection cannot distinguish @cute.kernel from @cute.jit wrappers.
        root = pathlib.Path(reductions.__file__).parent
        deco = re.compile(r"@(?:\w+\.)*cute\.kernel\b")
        found = [
            f"{path.name}:{i}"
            for path in sorted(root.glob("*.py"))
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
        with self.assertRaisesRegex(ValueError, "indices representable in Int32"):
            kg.ReduceBlock(
                T.ArgMaxOps(acc=cutlass.Float32),
                count=2**31 + 1,
                num_o=1,
                red_pairs=((2**31 + 1, 1),),
                kept_pairs=(),
            )
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
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    run_tests()
