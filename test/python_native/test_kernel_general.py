# Owner(s): ["module: dsl-native-ops"]
# Smoke-test K0 compilation and dispatch on non-adjacent reduced axes, which have no
# column view and so force the general path. Override OpInfo suites provide full
# numerical coverage.

import collections
import itertools
import pathlib
import re
import sys
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


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


def _plan_tags(trait_key):
    return {key[0][1] for key in kg._PLAN if len(key[0]) > 2 and key[0][2] == trait_key}


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
    def test_plans_key_and_require_the_current_device(self):
        trait = T.SumOps(acc=cutlass.Float32)
        x = torch.randn(8, 16, 32, device="cuda")
        before = (set(kg._PLAN), set(kg._COMPILE_CACHE))
        # One GPU has no other device, so pretend another one is current.
        other = x.get_device() + 1
        with mock.patch.object(torch.cuda, "current_device", return_value=other):
            with self.assertRaisesRegex(AssertionError, "is current"):
                kg.reduce_dim(trait, "bydevice", x, [0, 2], torch.float32)
        self.assertEqual((set(kg._PLAN), set(kg._COMPILE_CACHE)), before)
        kg.reduce_dim(trait, "bydevice", x, [0, 2], torch.float32)
        # Neither a row nor a column layout, so every route uses _launch, which keys
        # plans with the device index first.
        keys = [k[0] for k in kg._PLAN if k[0][2] == "bydevice"]
        self.assertTrue(keys)
        self.assertEqual({k[0] for k in keys}, {x.get_device()})

    def test_reduce_dim_general_path(self):
        # Non-adjacent reduced axes have no column view, so the general kernel serves them.
        x = torch.randn(8, 16, 16, device="cuda")
        out = kg.reduce_dim(
            T.SumOps(acc=cutlass.Float32), "general_smoke", x, [0, 2], torch.float32
        )
        self.assertEqual(_plan_tags("general_smoke"), {"reduce"})
        torch.testing.assert_close(out, x.sum(dim=(0, 2)), atol=1e-2, rtol=1e-2)

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

    @unittest.skipIf(
        rt.inner_tree_order_enabled(), "requires unordered reduction order"
    )
    def test_general_split_layouts_and_welford(self):
        """Split strided, offset, multi-run, and two-output reductions correctly."""
        base = torch.randn(4097, 9, 17, device="cuda")
        x = base[1::2]
        with torch.backends.python_native.cutedsl.disabled():
            expected_var, expected_mean = torch.var_mean(x, dim=0)
            expected_total = x.sum(dim=(0, 2))
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

        self.assertEqual(_plan_tags("split_var_mean"), {"gensplit1", "gensplit2"})
        self.assertEqual(_plan_tags("split_multi_run"), {"gensplit1", "gensplit2"})
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

    @unittest.skipIf(
        rt.inner_tree_order_enabled(), "requires unordered reduction order"
    )
    def test_reduce_all2_route(self):
        """Route one-element two-output reductions through reduce_all2."""
        x = torch.randn(1, 65537, device="cuda")
        lo, hi = kg.reduce_dim2(
            T.AMinMaxOps(acc=cutlass.Float32),
            "reduce_all2",
            x,
            [1],
            [torch.float32, torch.float32],
        )
        self.assertEqual(_plan_tags("reduce_all2"), {"all1", "all2"})
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
                T.ArgMaxOps(acc=acc),
                "index_cache_key",
                x,
                torch.int32,
            )
            with self.subTest(dtype=dtype):
                self.assertEqual(out, winner)

    def test_noncontiguous_reduce_all_index_is_logical(self):
        """Report row-major logical indices for non-contiguous inputs."""
        x = torch.zeros(64, 128, device="cuda").t()
        x[0, 1] = 1
        out = kg.reduce_all(
            T.ArgMaxOps(acc=cutlass.Float32),
            "noncontiguous_index",
            x,
            torch.int32,
        )
        self.assertEqual(out, 1)

    def test_norm2_tree_leaves_match_across_trees(self):
        """Round norm2 tree leaves alike in the row and indexed trees, whose row bits
        test_inner_tree_order pins."""
        dtypes = (
            (torch.float32, cutlass.Float32, torch.int32),
            (torch.float64, cutlass.Float64, torch.int64),
        )
        # N = 2000 plans one looped stage and N = 100003 a two-stage split.
        shapes = ((64, 2000, ("",)), (8, 100003, ("1", "2")))
        for (dtype, acc, bits), (m, n, stages) in itertools.product(dtypes, shapes):
            with self.subTest(dtype=dtype, n=n):
                x = torch.randn(m, 2 * n, device="cuda", dtype=dtype)[:, ::2]
                trait, kw = T.NormOps(2, acc=acc), {"order": "inner_tree"}
                key = f"norm2_{dtype}_{n}"
                row_key, idx_key = f"{key}_rows", f"{key}_indexed"
                rows = kg.reduce_dim(trait, row_key, x.contiguous(), [1], dtype, **kw)
                indexed = kg.reduce_dim(trait, idx_key, x, [1], dtype, **kw)
                row_tags = {k[0] for k in rt._CACHE if k[1] == row_key}
                self.assertEqual(row_tags, {f"rowitree{s}" for s in stages})
                self.assertEqual(_plan_tags(idx_key), {f"genitree{s}" for s in stages})
                self.assertEqual(rows.view(bits), indexed.view(bits))
                self.assertEqual(rows, torch.linalg.vector_norm(x, dim=1))

    def test_all_folds_zero_flags_serially(self):
        """Fold all()'s flags through the serial reduce of the unordered kernels."""
        x = torch.ones(64, 4096, device="cuda")
        x[::2, 6] = 0
        trait, i32 = T.AllOps(acc=cutlass.Float32), torch.int32
        # Contiguous rows take the row kernel; strided rows take the general kernel.
        for key, view in (("all_rows", x), ("all_strided", x[:, ::2])):
            got = kg.reduce_dim(trait, key, view, [1], i32, order="unordered")
            self.assertEqual(got, (view != 0).all(1).to(i32))
        self.assertEqual({k[0] for k in rt._CACHE if k[1] == "all_rows"}, {"rowtile"})
        self.assertEqual(_plan_tags("all_strided"), {"gensplit1", "gensplit2"})

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

    def test_family_kernels_are_listed(self):
        # New kernels must be listed here deliberately: TileReduce is the shared body for every
        # axis, PartialColumns folds partition-major column partials, and _OrderedColReduce
        # preserves the row inner-tree order down columns. Glob files and match
        # qualified or annotated decorators so new drivers and spellings cannot evade the check.
        # Runtime introspection cannot distinguish @cute.kernel from @cute.jit wrappers.
        root = pathlib.Path(reductions.__file__).parent
        deco = re.compile(r"@(?:\w+\.)*cute\.kernel\b")
        found = collections.Counter(
            path.name
            for path in root.glob("*.py")
            for line in path.read_text().splitlines()
            if deco.match(line.strip())
        )
        self.assertEqual(found, {"tile.py": 1, "kernel_coltile.py": 2})

    # 4095 over 64 threads leaves leftover subtrees and a 63-value tail load with one masked
    # lane; count == block is a single load, and 200 over 128 one whole load and a tail.
    # Stride and the view offset are runtime geometry, so cache_sig omits them; shift scales
    # and offsets the input values.
    @parametrize(
        "count,block,dtype,stride,offset,shift,var_mean",
        [
            (1024, 128, torch.float32, 1, 0, False, False),
            (4095, 64, torch.float32, 1, 0, False, False),
            (4096, 64, torch.float32, 1, 0, False, False),
            (128, 128, torch.float32, 1, 0, False, False),
            (200, 128, torch.float32, 1, 0, False, False),
            (4095, 64, torch.float32, 2, 0, False, False),
            (4095, 64, torch.float32, 2, 1, False, False),
            (4095, 64, torch.float16, 1, 0, False, False),
            (4095, 64, torch.bfloat16, 2, 0, False, False),
            (4096, 64, torch.float32, 1, 0, True, False),
            (1024, 128, torch.float32, 1, 0, False, True),
            (4095, 64, torch.float32, 2, 1, True, True),
        ],
    )
    def test_uniform_welford_tree(
        self, count, block, dtype, stride, offset, shift, var_mean
    ):
        """Equal-count trees must retain Welford accuracy without serial counts."""
        rows = 17
        base = torch.randn(rows + offset, stride * count, device="cuda", dtype=dtype)
        if shift:
            base.mul_(8).add_(1024)
        x = base[offset:, ::stride]
        ops = T.VarMeanOps if var_mean else T.WelfordOps
        trait = ops(correction=1, acc=cutlass.Float32)
        outs = [torch.empty(rows, device="cuda") for _ in range(2 if var_mean else 1)]
        op = kg.ReduceBlock(
            trait,
            count=count,
            num_o=rows,
            red_pairs=((count, stride),),
            kept_pairs=((rows, stride * count),),
            in_base=x.storage_offset(),
            nouts=len(outs),
            block=block,
            uniform_tree=True,
        )
        plain = kg.ReduceBlock(
            trait,
            count=count,
            num_o=rows,
            red_pairs=((count, 1),),
            kept_pairs=((rows, count),),
            nouts=len(outs),
            block=block,
            uniform_tree=True,
        )
        self.assertEqual(op.cache_sig, plain.cache_sig)
        key = ("test_uniform_welford_tree", type(trait).__name__, dtype) + op.cache_sig
        # plain compiles the shared key for contiguous geometry, or reuses an earlier
        # case's kernel, so a strided or offset op runs a kernel built for other geometry.
        dense = [torch.empty_like(out) for out in outs]
        kg._launch(plain, key, [kg._flat(x.contiguous())], dense)
        kg._launch(op, key, [kg._flat(x)], outs)
        with torch.backends.python_native.cutedsl.disabled():
            var, mean = torch.var_mean(x.double(), dim=1)
        expected = [var.float(), mean.float()][: len(outs)]
        atol, rtol = (1e-3, 1e-4) if shift else (1e-5, 1e-5)
        self.assertEqual(dense, expected, atol=atol, rtol=rtol)
        self.assertEqual(outs, expected, atol=atol, rtol=rtol)

    def test_uniform_welford_tree_counts_share_kernels(self):
        # A tail kernel serves every count strictly between nloads * block and
        # (nloads + 1) * block, so 300 and 383 (2 * 128 < count < 3 * 128) share one;
        # 256 = 2 * 128 has the same whole-load count but takes a tail-free kernel, as does
        # 384 = 3 * 128.
        trait = T.WelfordOps(correction=1, acc=cutlass.Float32)
        for count in (256, 300, 383, 384):
            x = torch.randn(17, count, device="cuda")
            out = torch.empty(17, device="cuda")
            op = kg.ReduceBlock(
                trait,
                count=count,
                num_o=17,
                red_pairs=((count, 1),),
                kept_pairs=((17, count),),
                block=128,
                uniform_tree=True,
            )
            self.assertEqual(op.tile.uniform_tree_loads, count // 128)
            self.assertEqual(op.tile.uniform_tree_tail, bool(count % 128))
            kg._launch(
                op, ("uniform_share", "var") + op.cache_sig, [kg._flat(x)], [out]
            )
            with torch.backends.python_native.cutedsl.disabled():
                want = torch.var(x.double(), dim=1).float()
            self.assertEqual(out, want, atol=1e-5, rtol=1e-5)
        self.assertEqual(
            len([k for k in kg._COMPILE_CACHE if k[1] == "uniform_share"]), 3
        )

    def test_uniform_welford_tree_changes_bits(self):
        """uniform_tree=True must take the tree: its bits differ from the online chain's,
        and both stay within tolerance of fp64."""
        torch.manual_seed(0)
        rows, count = 17, 4096
        x = torch.randn(rows, count, device="cuda")
        trait = T.WelfordOps(correction=1, acc=cutlass.Float32)
        outs = []
        for uniform_tree in (True, False):
            op = kg.ReduceBlock(
                trait,
                count=count,
                num_o=rows,
                red_pairs=((count, 1),),
                kept_pairs=((rows, count),),
                block=64,
                uniform_tree=uniform_tree,
            )
            outs.append(torch.empty(rows, device="cuda"))
            key = ("uniform_bits", "var") + op.cache_sig
            kg._launch(op, key, [kg._flat(x)], outs[-1:])
        with torch.backends.python_native.cutedsl.disabled():
            want = torch.var(x.double(), dim=1).float()
        for out in outs:
            self.assertEqual(out, want, atol=1e-5, rtol=1e-5)
        tree, online = outs
        self.assertFalse(torch.equal(tree, online))

    def test_internal_invariants_raise(self):
        # Each invariant must raise explicitly because python -O strips asserts. Exercise every
        # check on its documented invalid input.
        trait = T.SumOps(acc=cutlass.Float32)
        # reduce-all has no flat view of a transposed input, so it ROUTES that through the general
        # path (which addresses via the TI decode) rather than refusing it.
        xt = torch.randn(64, 128, device="cuda").t()
        (flat,) = kg._reduce_all(trait, "inv", xt, [torch.float32], 1)
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
        # Storage bounds are absolute, while an index trait reports view-relative positions.
        offset = 2**31
        offset_op = kg.ReduceBlock(
            T.ArgMaxOps(acc=cutlass.Float32),
            count=1024,
            num_o=1,
            red_pairs=((1024, 1),),
            kept_pairs=(),
            in_base=offset,
            limit=offset + 1024,
            flat_tail=True,
            gidx_from="chunk",
        )
        self.assertEqual(offset_op.limit, offset + 1024)
        # No reduced runs at all is LEGAL -- an extent-1 reduced axis coalesces away in TI. Pinned
        # here because the arm that allows it is one line in tile._decode_offset, and a crash.
        self.assertEqual(
            kg.ReduceBlock(
                trait, count=1, num_o=1, red_pairs=(), kept_pairs=()
            ).npairs_red,
            0,
        )
        # The uniform Welford tree needs combine_equal, one unclamped Int32 reduced run, a linear
        # decoded (not partial) fold indexed by r, and 1 to UNIFORM_TREE_MAX_LOADS loads per thread.
        tree = dict(
            trait=T.WelfordOps(acc=cutlass.Float32),
            count=1024,
            num_o=1,
            red_pairs=((1024, 1),),
            kept_pairs=(),
            uniform_tree=True,
        )
        self.assertEqual(kg.ReduceBlock(**tree).tile.uniform_tree_loads, 1024 // 128)
        cap = 128 * tile.UNIFORM_TREE_MAX_LOADS
        cap64 = 64 * tile.UNIFORM_TREE_MAX_LOADS
        span = f"128 to {cap} values per output for 128 threads, got"
        fold, clamp = "a linear general-axis fold, got axis=", "an unclamped run, got"
        for bad, reason in (
            (dict(trait=trait), "a trait with has_combine_equal, got SumOps"),
            (dict(count=0, red_pairs=((0, 1),)), f"{span} 0"),
            (dict(count=64, red_pairs=((64, 1),)), f"{span} 64"),
            (dict(count=cap + 1, red_pairs=((cap + 1, 1),)), f"{span} {cap + 1}"),
            (
                dict(block=64, count=cap64 + 1, red_pairs=((cap64 + 1, 1),)),
                f"64 to {cap64} values per output for 64 threads, got {cap64 + 1}",
            ),
            (dict(red_pairs=((32, 1), (32, 64))), "one reduced run, got 2"),
            (dict(red_pairs=()), "one reduced run, got 0"),
            (dict(from_partials=True), "a fold of inputs, not partials"),
            (dict(gidx_from="chunk"), "indices from r, got gidx_from='chunk'"),
            (dict(order="inner_tree"), f"{fold}'general' order='inner_tree'"),
            (dict(flat_tail=True), f"{clamp} flat_tail=True ragged_chunk=False"),
            (dict(ragged_chunk=True), f"{clamp} flat_tail=False ragged_chunk=True"),
        ):
            with self.subTest(bad=bad):
                msg = re.escape(f"the uniform Welford tree needs {reason}") + "$"
                with self.assertRaisesRegex(ValueError, msg):
                    kg.ReduceBlock(**(tree | bad))
        # ReduceBlock always folds the general axis.
        with self.assertRaisesRegex(ValueError, re.escape(f"{fold}'row' ")):
            tile.TileReduce(tree["trait"], None, "row", 1024, uniform_tree_count=1024)
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


instantiate_parametrized_tests(TestKernelGeneral)

if __name__ == "__main__":
    run_tests()
