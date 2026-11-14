# Owner(s): ["module: dsl-native-ops"]

import os
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

import torch
import torch.backends.python_native as pn
from torch._native import flydsl_utils as fu
from torch._native.ops.scatter_add import flydsl_impl
from torch._native.ops.scatter_add.flydsl_geometry import lane_elems, MAX_STEPS_PER_LANE
from torch._native.ops.scatter_add.flydsl_impl import _SUPPORTED_ARCHES
from torch._vendor.packaging.version import Version
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoFlyDSL,
    TestCase,
)


_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
# The shipped work floor, before setUp lowers it for kernel tests.
_DEFAULT_FLOOR = dict(flydsl_impl._MIN_SRC_BYTES)
# N values covering one partial wave step, full steps and the 7168 MoE width.
_NS = (128, 1024, 1032, 7168)


def _int_valued(shape, dtype, *, low=-1, high=2, seed=0):
    """Small integers: every partial sum is exact in bf16/fp16/fp32, so results
    are independent of atomic ordering and can be compared exactly."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(low, high, shape, device="cuda", generator=g).to(dtype)


def _indices(kind: str, m_src: int, m_dst: int, dtype=torch.int64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "perm":
        if m_src > m_dst:
            raise ValueError("a permutation needs m_src <= m_dst")
        idx = torch.randperm(m_dst, device="cuda", generator=g)[:m_src]
    elif kind == "uniform":
        idx = torch.randint(0, m_dst, (m_src,), device="cuda", generator=g)
    elif kind == "zipf":
        # About half of the rows hit row 0, the rest are uniform.
        idx = torch.randint(0, m_dst, (m_src,), device="cuda", generator=g)
        idx[::2] = 0
    elif kind == "all_to_one":
        idx = torch.full((m_src,), m_dst - 1, device="cuda")
    else:
        raise AssertionError(kind)
    return idx.to(dtype)


def _row_index(idx_1d: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    shape = [1] * src.ndim
    shape[0] = idx_1d.numel()
    return idx_1d.view(shape).expand_as(src)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoFlyDSL
@unittest.skipIf(
    torch.version.hip is None
    or not fu._is_supported_arch(
        torch.cuda.current_device() if torch.cuda.is_available() else 0,
        _SUPPORTED_ARCHES,
    ),
    "FlyDSL scatter_add override requires gfx942/gfx950",
)
class TestFlyDSLRowScatterAdd(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        # Beyond the class decorators: the override registers only for
        # validated FlyDSL releases, and dispatches only when FlyDSL's
        # process-wide compile target is the current device's arch. Tests
        # here expect dispatch; excluded setups are covered as fallbacks.
        # Checked before super().setUp(): a skip raised after it would bypass
        # tearDown, leaving the base class's global state changes in place.
        if not flydsl_impl._flydsl_release_validated():
            self.skipTest("FlyDSL release not validated for this override")
        if not flydsl_impl._compile_target_matches(torch.cuda.current_device()):
            self.skipTest("FlyDSL compile target differs from the current device")
        super().setUp()
        # Most tests here check the kernel on small shapes, which the default
        # work floor (_MIN_SRC_BYTES) routes to ATen; test_dispatch_floor
        # checks the defaults themselves.
        floor = mock.patch.dict(flydsl_impl._MIN_SRC_BYTES, dict.fromkeys(_DTYPES, 0))
        floor.start()
        self.addCleanup(floor.stop)
        from torch._native.ops.scatter_add.flydsl_kernels import (
            clear_row_scatter_add_cache,
        )

        clear_row_scatter_add_cache()

    def _cache_info(self):
        from torch._native.ops.scatter_add.flydsl_kernels import (
            row_scatter_add_cache_info,
        )

        return row_scatter_add_cache_info()

    def _misses(self) -> int:
        return self._cache_info().misses

    def _hits(self) -> int:
        return self._cache_info().hits

    def _assert_dispatched(self) -> None:
        self.assertGreater(self._misses(), 0, "FlyDSL kernel was not used")

    def _assert_not_dispatched(self) -> None:
        self.assertEqual(self._misses(), 0, "FlyDSL kernel was unexpectedly used")

    # -- scatter_add -------------------------------------------------------

    @parametrize("dtype", _DTYPES)
    @parametrize("n", _NS)
    @parametrize("kind", ("perm", "uniform", "zipf", "all_to_one"))
    @parametrize("index_dtype", (torch.int64, torch.int32))
    def test_scatter_add_inplace_exact(self, dtype, n, kind, index_dtype):
        m_src, m_dst = 512, 1024
        src = _int_valued((m_src, n), dtype, seed=1)
        dst = _int_valued((m_dst, n), dtype, seed=2)
        index = _row_index(_indices(kind, m_src, m_dst, index_dtype), src)
        with pn.flydsl.disabled():
            ref = dst.clone().scatter_add_(0, index, src)
        got = dst.clone()
        got.scatter_add_(0, index, src)
        self._assert_dispatched()
        self.assertEqual(got, ref, atol=0, rtol=0)

    @parametrize("dtype", _DTYPES)
    def test_scatter_add_functional_and_out(self, dtype):
        n, m_src, m_dst = 256, 300, 200
        src = _int_valued((m_src, n), dtype, seed=3)
        dst = _int_valued((m_dst, n), dtype, seed=4)
        index = _row_index(_indices("uniform", m_src, m_dst), src)
        with pn.flydsl.disabled():
            ref = torch.scatter_add(dst, 0, index, src)
        self.assertEqual(torch.scatter_add(dst, 0, index, src), ref, atol=0, rtol=0)
        out = torch.empty_like(dst)
        torch.scatter_add(dst, 0, index, src, out=out)
        self.assertEqual(out, ref, atol=0, rtol=0)
        # out is self.
        alias = dst.clone()
        torch.scatter_add(alias, 0, index, src, out=alias)
        self.assertEqual(alias, ref, atol=0, rtol=0)
        self._assert_dispatched()

    def test_scatter_add_strided_views_and_offsets(self):
        # Destination and source are row slices of wider buffers with a
        # storage offset; index covers fewer rows than src.
        n = 512
        big_dst = _int_valued((300, 2 * n + 64), torch.bfloat16, seed=5)
        big_src = _int_valued((260, n + 64), torch.bfloat16, seed=6)
        dst = big_dst[10:, 64 : 64 + n]
        src = big_src[2:, 8 : 8 + n]
        self.assertEqual(src.data_ptr() % 16, 0)
        idx = _indices("uniform", 200, dst.shape[0])
        index = _row_index(idx, src[:200])
        with pn.flydsl.disabled():
            ref = big_dst.clone()
            ref[10:, 64 : 64 + n].scatter_add_(0, index, src)
        dst.scatter_add_(0, index, src)
        self._assert_dispatched()
        self.assertEqual(big_dst, ref, atol=0, rtol=0)

    def test_dst_only_4byte_aligned_reuses_compiled_kernel(self):
        # Compile with a 16-byte-aligned destination, then run the same
        # specialization on one that is only 4-byte aligned (the contract).
        n = 256
        src = _int_valued((128, n), torch.bfloat16, seed=27)
        idx = _indices("uniform", 128, 64)
        aligned = _int_valued((64, n), torch.bfloat16, seed=28)
        buf = _int_valued((64, n + 8), torch.bfloat16, seed=29)
        shifted = buf[:, 2 : 2 + n]
        self.assertEqual(shifted.data_ptr() % 16, 4)
        for dst in (aligned, shifted):
            with pn.flydsl.disabled():
                ref = dst.clone().index_add_(0, idx, src)
            hits = self._hits()
            dst.index_add_(0, idx, src)
            self.assertEqual(dst, ref, atol=0, rtol=0)
            if dst is shifted:
                # The shifted call itself reused the compiled kernel.
                self.assertEqual(self._hits(), hits + 1)
        self.assertEqual(self._misses(), 1)

    @parametrize("op", ("index_add_", "scatter_add_"))
    @parametrize("dtype", _DTYPES)
    def test_cold_and_warm_cache_give_identical_results(self, op, dtype):
        # flyc.compile launches once on its sample arguments; the first
        # (cold) call must still apply exactly one update.
        n = 256
        src = torch.ones((8, n), device="cuda", dtype=dtype)
        idx = torch.zeros(8, dtype=torch.int64, device="cuda")
        results = []
        for _ in range(2):  # cold, then warm
            dst = torch.zeros((16, n), device="cuda", dtype=dtype)
            if op == "index_add_":
                dst.index_add_(0, idx, src)
            else:
                dst.scatter_add_(0, _row_index(idx, src), src)
            results.append(dst)
        self.assertEqual(self._misses(), 1)
        self.assertGreaterEqual(self._hits(), 1)
        expected = torch.zeros((16, n), device="cuda", dtype=dtype)
        expected[0] = 8
        for got in results:
            self.assertEqual(got, expected, atol=0, rtol=0)

    def test_functional_noncontiguous_self_falls_back(self):
        # The functional result is contiguous; these layouts are only valid
        # for the original strides, so the override must decline, not raise.
        dst = torch.zeros((32, 128), device="cuda").T
        src = torch.ones((16, 128), device="cuda").T
        index = torch.arange(16, device="cuda")[None, :].expand(128, 16)
        with pn.flydsl.disabled():
            ref = torch.scatter_add(dst, 1, index, src)
        self.assertEqual(torch.scatter_add(dst, 1, index, src), ref)

        dst3 = torch.zeros((32, 8, 32), device="cuda").transpose(1, 2)
        src3 = torch.ones((16, 8, 32), device="cuda").transpose(1, 2)
        idx3 = torch.arange(16, device="cuda")
        with pn.flydsl.disabled():
            ref3 = torch.index_add(dst3, 0, idx3, src3)
        self.assertEqual(torch.index_add(dst3, 0, idx3, src3), ref3)
        self._assert_not_dispatched()

    def test_compile_target_mismatch_declines(self):
        src = _int_valued((64, 256), torch.float32, seed=30)
        dst = _int_valued((32, 256), torch.float32, seed=31)
        idx = _indices("uniform", 64, 32)
        with pn.flydsl.disabled():
            ref = dst.clone().index_add_(0, idx, src)
        with mock.patch(
            "torch._native.ops.scatter_add.flydsl_kernels.compile_target_arch",
            return_value="gfx000",
        ):
            dst.index_add_(0, idx, src)
        self._assert_not_dispatched()
        self.assertEqual(dst, ref)

    @parametrize("op", ("index_add", "index_add_", "scatter_add_", "scatter_out"))
    def test_impl_reuses_cond_layout(self, op):
        # The impl takes the plan its cond approved instead of re-running the
        # layout analysis: each dispatch analyzes only in the cond(s).
        src = _int_valued((64, 256), torch.float32, seed=40)
        dst = _int_valued((32, 256), torch.float32, seed=41)
        idx = _indices("uniform", 64, 32)
        index = _row_index(idx, src)

        def run(d):
            if op == "index_add":
                return torch.index_add(d, 0, idx, src)
            if op == "index_add_":
                return d.index_add_(0, idx, src)
            if op == "scatter_add_":
                return d.scatter_add_(0, index, src)
            return torch.scatter_add(d, 0, index, src, out=torch.empty_like(d))

        with pn.flydsl.disabled():
            ref = run(dst.clone())
        with (
            mock.patch.object(
                flydsl_impl, "_plan_rows", side_effect=flydsl_impl._plan_rows
            ) as plan_rows,
            mock.patch.object(
                flydsl_impl, "_geometry", side_effect=flydsl_impl._geometry
            ) as general,
        ):
            got = run(dst.clone())
        self._assert_dispatched()
        self.assertEqual(got, ref, atol=0, rtol=0)
        # .out analyzes self and out in its cond; the others analyze self.
        self.assertEqual(plan_rows.call_count, 2 if op == "scatter_out" else 1)
        # index_add reads contiguous rows off the strides, skipping the
        # general TensorIterator analysis that scatter_add needs.
        expected_general = 0 if "index" in op else plan_rows.call_count
        self.assertEqual(general.call_count, expected_general)

    def test_dispatch_floor(self):
        # The shipped floor: fp32 never dispatches; halves dispatch from
        # _MIN_SRC_BYTES of source. Checked on the predicate, then end to end.
        plan = flydsl_impl._RowPlan
        self.assertIsNone(_DEFAULT_FLOOR[torch.float32])
        for dtype in (torch.float16, torch.bfloat16):
            floor = _DEFAULT_FLOOR[dtype]
            self.assertGreater(floor, 0)
            n = 4096
            rows = floor // (n * dtype.itemsize)
            with mock.patch.dict(flydsl_impl._MIN_SRC_BYTES, _DEFAULT_FLOOR):
                worth = flydsl_impl._worth_dispatching
                self.assertTrue(worth(dtype, plan(rows, n, n, rows, n)))
                self.assertFalse(worth(dtype, plan(rows, n, n, rows - 1, n)))
                self.assertFalse(worth(torch.float32, plan(rows, n, n, 4 * rows, n)))
        # A small eligible call falls back to ATen under the shipped floor.
        src = _int_valued((64, 256), torch.bfloat16, seed=46)
        dst = _int_valued((32, 256), torch.bfloat16, seed=47)
        idx = _indices("uniform", 64, 32)
        with pn.flydsl.disabled():
            ref = dst.clone().index_add_(0, idx, src)
        with mock.patch.dict(flydsl_impl._MIN_SRC_BYTES, _DEFAULT_FLOOR):
            dst.index_add_(0, idx, src)
        self._assert_not_dispatched()
        self.assertEqual(dst, ref, atol=0, rtol=0)

    def test_index_add_non_row_layout_uses_general_analysis(self):
        # Inner dims transposed in both operands: not one run per row, so the
        # fast path declines and the general analysis decides (and coalesces).
        src = _int_valued((64, 32, 16), torch.float32, seed=44).transpose(1, 2)
        dst = _int_valued((32, 32, 16), torch.float32, seed=45).transpose(1, 2)
        idx = _indices("uniform", 64, 32)
        self.assertIsNone(flydsl_impl._row_layout(dst))
        with pn.flydsl.disabled():
            ref = dst.clone().index_add_(0, idx, src)
        with mock.patch.object(
            flydsl_impl, "_geometry", side_effect=flydsl_impl._geometry
        ) as general:
            dst.index_add_(0, idx, src)
        self.assertGreaterEqual(general.call_count, 1)
        self._assert_dispatched()
        self.assertEqual(dst, ref, atol=0, rtol=0)

    def test_stale_approval_is_not_used(self):
        # An approval is used only for the same tensors with the same layout.
        base = _int_valued((32, 512), torch.float32, seed=42)
        src = _int_valued((64, 256), torch.float32, seed=43)
        idx = _indices("uniform", 64, 32)
        dst = base[:, :256].contiguous()
        self.assertTrue(flydsl_impl._index_add_inplace_cond(dst, 0, idx, src))
        # Same object, new strides: rows of the 512-wide buffer.
        dst.set_(base.untyped_storage(), 0, (32, 256), (512, 1))
        with pn.flydsl.disabled():
            ref = base.clone()
            ref[:, :256].index_add_(0, idx, src)
        flydsl_impl._index_add_inplace_impl(dst, 0, idx, src)
        self.assertEqual(base, ref, atol=0, rtol=0)

    @parametrize(
        "var,value,expected",
        (
            ("ARCH", "gfx000", "gfx000"),
            ("FLYDSL_GPU_ARCH", "gfx000", "gfx000"),
            ("HSA_OVERRIDE_GFX_VERSION", "9.9.9", "gfx999"),
        ),
    )
    def test_compile_target_cache_follows_arch_override(self, var, value, expected):
        # Prime the cache, then change one override with the higher-priority
        # ones unset: the guards must see FlyDSL's new target, not the cached one.
        from torch._native.ops.scatter_add.flydsl_kernels import compile_target_arch

        overrides = ("ARCH", "FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION")
        # The process's target before any change; removing the higher-priority
        # overrides below can change it, so it is captured outside the patch.
        original = compile_target_arch()
        with mock.patch.dict(os.environ):
            for name in overrides[: overrides.index(var)]:
                os.environ.pop(name, None)
            primed = compile_target_arch()
            self.assertIsNotNone(primed)
            os.environ[var] = value
            self.assertEqual(compile_target_arch(), expected)
            self.assertFalse(
                flydsl_impl._compile_target_matches(torch.cuda.current_device())
            )
        self.assertEqual(compile_target_arch(), original)

    @parametrize("op", ("index_add_", "scatter_add_"))
    def test_python_dispatch_subclass_falls_back(self, op):
        # _is_cow_tensor raises for tensors with the Python dispatch key; the
        # conds must decline them instead of propagating that error.
        class PassthroughTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, value):
                return torch.Tensor._make_subclass(cls, value, False)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                with torch._C._DisableTorchDispatch():
                    return func(*args, **(kwargs or {}))

        src = _int_valued((64, 256), torch.float32, seed=36)
        idx = _indices("uniform", 64, 32)

        def run(dst):
            if op == "index_add_":
                dst.index_add_(0, idx, src)
            else:
                dst.scatter_add_(0, _row_index(idx, src), src)
            return dst.as_subclass(torch.Tensor).clone()

        base = _int_valued((32, 256), torch.float32, seed=37)
        with pn.flydsl.disabled():
            ref = run(PassthroughTensor(base.clone()))
        got = run(PassthroughTensor(base.clone()))
        self._assert_not_dispatched()
        self.assertEqual(got, ref, atol=0, rtol=0)

    @parametrize("op", ("index_add", "scatter_add"))
    def test_backward_through_overridden_op(self, op):
        g = torch.Generator(device="cuda").manual_seed(32)
        dst0 = torch.randn((64, 256), device="cuda", generator=g)
        src0 = torch.randn((128, 256), device="cuda", generator=g)
        idx = _indices("uniform", 128, 64)
        grad = torch.randn((64, 256), device="cuda", generator=g)

        def run():
            dst = dst0.clone().requires_grad_()
            src = src0.clone().requires_grad_()
            if op == "index_add":
                out = torch.index_add(dst, 0, idx, src)
            else:
                out = torch.scatter_add(dst, 0, _row_index(idx, src), src)
            out.backward(grad)
            return out.detach(), dst.grad, src.grad

        got = run()
        self._assert_dispatched()
        with pn.flydsl.disabled():
            ref = run()
        self.assertEqual(got[0], ref[0])
        self.assertEqual(got[1], ref[1], atol=0, rtol=0)
        self.assertEqual(got[2], ref[2], atol=0, rtol=0)

    @parametrize("dtype", _DTYPES)
    def test_row_width_cap_matches_kernel(self, dtype):
        # The cond and the kernel builder share flydsl_geometry: the widest
        # accepted row must compile and run, and the next 16-byte step must
        # fall back to ATen instead of reaching the builder's step limit.
        wave = torch.cuda.get_device_properties(0).warp_size
        widest = MAX_STEPS_PER_LANE * wave * lane_elems(dtype.itemsize)
        granule = 16 // dtype.itemsize
        from torch._native.ops.scatter_add.flydsl_kernels import (
            clear_row_scatter_add_cache,
        )

        for n, dispatches in ((widest, True), (widest + granule, False)):
            clear_row_scatter_add_cache()
            src = _int_valued((8, n), dtype, seed=38)
            dst = _int_valued((4, n), dtype, seed=39)
            idx = _indices("uniform", 8, 4)
            with pn.flydsl.disabled():
                ref = dst.clone().index_add_(0, idx, src)
            dst.index_add_(0, idx, src)
            if dispatches:
                self._assert_dispatched()
            else:
                self._assert_not_dispatched()
            self.assertEqual(dst, ref, atol=0, rtol=0)

    @parametrize("dtype", _DTYPES)
    def test_row_width_floor(self, dtype):
        # Rows below 256 bytes decline on every route, including the
        # index_add row fast path that skips the general analysis; 256-byte
        # rows dispatch. 16 and 240 bytes satisfy the 16-byte granule, so
        # only the minimum row width rejects them.
        from torch._native.ops.scatter_add.flydsl_kernels import (
            clear_row_scatter_add_cache,
        )

        m_src, m_dst = 64, 32
        idx = _indices("uniform", m_src, m_dst)
        for row_bytes in (16, 240, 256):
            n = row_bytes // dtype.itemsize
            accepted = row_bytes >= 256
            src = _int_valued((m_src, n), dtype, seed=48)
            dst = _int_valued((m_dst, n), dtype, seed=49)
            index = _row_index(idx, src)
            out = torch.empty_like(dst)
            conds = (
                flydsl_impl._index_add_cond(dst, 0, idx, src),
                flydsl_impl._index_add_inplace_cond(dst, 0, idx, src),
                flydsl_impl._index_add_out_cond(dst, 0, idx, src, out=out),
                flydsl_impl._scatter_add_cond(dst, 0, index, src),
                flydsl_impl._scatter_add_inplace_cond(dst, 0, index, src),
                flydsl_impl._scatter_add_out_cond(dst, 0, index, src, out=out),
            )
            self.assertEqual(conds, (accepted,) * 6, f"{row_bytes} B rows")
            runs = (
                lambda d: d.index_add_(0, idx, src),
                lambda d: d.scatter_add_(0, index, src),
            )
            for run in runs:
                clear_row_scatter_add_cache()
                with pn.flydsl.disabled():
                    ref = run(dst.clone())
                got = run(dst.clone())
                if accepted:
                    self._assert_dispatched()
                else:
                    self._assert_not_dispatched()
                self.assertEqual(got, ref, atol=0, rtol=0)

    def test_scatter_add_negative_dim_3d(self):
        src = _int_valued((64, 4, 64), torch.float32, seed=7)
        dst = _int_valued((32, 4, 64), torch.float32, seed=8)
        idx = _indices("uniform", 64, 32)
        index = idx.view(64, 1, 1).expand_as(src)
        with pn.flydsl.disabled():
            ref = dst.clone().scatter_add_(-3, index, src)
        dst.scatter_add_(-3, index, src)
        self._assert_dispatched()
        self.assertEqual(dst, ref, atol=0, rtol=0)

    @parametrize(
        "case",
        (
            "deterministic",
            "fp64",
            "small_slice",
            "odd_n",
            "misaligned_src",
            "not_row_broadcast",
            "empty",
        ),
    )
    def test_scatter_add_falls_through(self, case):
        dtype = torch.float64 if case == "fp64" else torch.float32
        n = {"small_slice": 32, "odd_n": 129}.get(case, 128)
        src = _int_valued((64, n + 4), dtype, seed=9)[:, :n]
        if case != "misaligned_src":
            src = src.contiguous()
        else:
            src = _int_valued((64, n + 1), dtype, seed=9)[:, 1:]
        dst = _int_valued((32, n), dtype, seed=10)
        idx = _indices("uniform", 64, 32)
        index = _row_index(idx, src)
        if case == "not_row_broadcast":
            index = torch.randint(0, 32, src.shape, device="cuda")
        if case == "empty":
            index = index[:0]
            src = src[:0]
        with pn.flydsl.disabled():
            ref = dst.clone().scatter_add_(0, index, src)
        if case == "deterministic":
            with DeterministicGuard(True):
                dst.scatter_add_(0, index, src)
        else:
            dst.scatter_add_(0, index, src)
        self._assert_not_dispatched()
        self.assertEqual(dst, ref)

    def test_overlapping_src_still_raises_aten_error(self):
        dst = _int_valued((32, 128), torch.float32, seed=24)
        src = dst[:16]
        index = _row_index(_indices("uniform", 16, 32), src)
        with self.assertRaisesRegex(RuntimeError, "unsupported operation"):
            dst.scatter_add_(0, index, src)
        with self.assertRaisesRegex(RuntimeError, "unsupported operation"):
            dst.index_add_(0, index[:, 0].contiguous(), src)
        self._assert_not_dispatched()

    # -- index_add ---------------------------------------------------------

    @parametrize("dtype", _DTYPES)
    @parametrize("kind", ("perm", "uniform", "zipf", "all_to_one"))
    @parametrize("index_dtype", (torch.int64, torch.int32))
    def test_index_add_variants_exact(self, dtype, kind, index_dtype):
        n, m_src, m_dst = 1024, 700, 1024
        src = _int_valued((m_src, n), dtype, seed=11)
        dst = _int_valued((m_dst, n), dtype, seed=12)
        idx = _indices(kind, m_src, m_dst, index_dtype)
        with pn.flydsl.disabled():
            ref = dst.index_add(0, idx, src)
        self.assertEqual(dst.index_add(0, idx, src), ref, atol=0, rtol=0)
        out = torch.empty_like(dst)
        torch.index_add(dst, 0, idx, src, out=out)
        self.assertEqual(out, ref, atol=0, rtol=0)
        inplace = dst.clone()
        inplace.index_add_(0, idx, src)
        self.assertEqual(inplace, ref, atol=0, rtol=0)
        self._assert_dispatched()

    def test_index_add_multidim_rows(self):
        src = _int_valued((96, 8, 32), torch.bfloat16, seed=13)
        dst = _int_valued((40, 8, 32), torch.bfloat16, seed=14)
        idx = _indices("uniform", 96, 40)
        with pn.flydsl.disabled():
            ref = dst.clone().index_add_(0, idx, src)
        dst.index_add_(0, idx, src)
        self._assert_dispatched()
        self.assertEqual(dst, ref, atol=0, rtol=0)

    @parametrize("case", ("alpha", "dim1", "bool_alpha", "index_2d_view"))
    def test_index_add_falls_through(self, case):
        src = _int_valued((64, 128), torch.float32, seed=15)
        dst = _int_valued((32, 128), torch.float32, seed=16)
        idx = _indices("uniform", 64, 32)
        kwargs = {}
        dim = 0
        if case == "alpha":
            kwargs["alpha"] = 2
        if case == "bool_alpha":
            kwargs["alpha"] = True
        if case == "dim1":
            dim = 1
            src = _int_valued((32, 64), torch.float32, seed=15)
            dst = _int_valued((32, 128), torch.float32, seed=16)
            idx = _indices("uniform", 64, 128)
        if case == "index_2d_view":
            idx = idx.repeat(2)[::2]  # non-contiguous 1D index
        with pn.flydsl.disabled():
            ref = dst.clone().index_add_(dim, idx, src, **kwargs)
        dst.index_add_(dim, idx, src, **kwargs)
        self._assert_not_dispatched()
        self.assertEqual(dst, ref)

    def test_declines_aten_sorted_index_add_shapes(self):
        # Mirrors index_add_sorted_rocm's acceptance test (Indexing.cu); meta
        # tensors keep the 1M-index check cheap.
        from torch._native.ops.scatter_add.flydsl_impl import (
            _aten_sorted_index_add_would_run,
        )

        self_ = torch.empty((10, 128), device="cuda")
        index = torch.empty(1_000_000, dtype=torch.long, device="meta")
        source = torch.empty((1_000_000, 128), device="meta")
        self.assertTrue(_aten_sorted_index_add_would_run(True, self_, index, source))
        self.assertFalse(_aten_sorted_index_add_would_run(False, self_, index, source))
        self.assertFalse(
            _aten_sorted_index_add_would_run(True, self_, index[:-1], source[:-1])
        )
        wide = torch.empty((10, 512), device="cuda")
        self.assertFalse(
            _aten_sorted_index_add_would_run(
                True, wide, index, torch.empty((1_000_000, 512), device="meta")
            )
        )

    def test_index_select_backward_uses_index_add(self):
        # index_select's backward is zeros(...).index_add_(0, index, grad).
        g = torch.Generator(device="cuda").manual_seed(25)
        weight = torch.randn((64, 256), device="cuda", generator=g)
        idx = torch.randint(0, 64, (500,), device="cuda", generator=g)
        grad = _int_valued((500, 256), torch.float32, seed=26)
        x = weight.clone().requires_grad_()
        torch.index_select(x, 0, idx).backward(grad)
        self._assert_dispatched()
        with pn.flydsl.disabled():
            x_ref = weight.clone().requires_grad_()
            torch.index_select(x_ref, 0, idx).backward(grad)
        self.assertEqual(x.grad, x_ref.grad, atol=0, rtol=0)

    # -- kernel-level variants ----------------------------------------------

    @parametrize("dtype", _DTYPES)
    @parametrize("mode", ("buffer", "global"))
    def test_atomic_modes(self, dtype, mode):
        from torch._native.ops.scatter_add.flydsl_kernels import (
            atomic_mode_supported,
            row_scatter_add_,
        )

        arch = fu._resolve_rocm_arch(torch.cuda.current_device())
        if not atomic_mode_supported(mode, dtype, arch):
            self.skipTest(f"{mode} atomics unsupported for {dtype} on {arch}")
        n, m_src, m_dst = 2048, 1000, 300
        src = _int_valued((m_src, n), dtype, seed=17)
        dst = _int_valued((m_dst, n), dtype, seed=18)
        idx = _indices("zipf", m_src, m_dst)
        with pn.flydsl.disabled():
            ref = dst.clone().index_add_(0, idx, src)
        row_scatter_add_(dst, idx, src, atomic_mode=mode)
        self.assertEqual(dst, ref, atol=0, rtol=0)

    def test_non_default_stream(self):
        src = _int_valued((256, 512), torch.bfloat16, seed=19)
        dst = _int_valued((128, 512), torch.bfloat16, seed=20)
        idx = _indices("uniform", 256, 128)
        with pn.flydsl.disabled():
            ref = dst.clone().index_add_(0, idx, src)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            dst.index_add_(0, idx, src)
        torch.cuda.current_stream().wait_stream(s)
        self._assert_dispatched()
        self.assertEqual(dst, ref, atol=0, rtol=0)

    @unittest.skipIf(not TEST_MULTIGPU, "requires two GPUs")
    def test_non_current_device(self):
        if not fu._is_supported_arch(1, _SUPPORTED_ARCHES):
            self.skipTest("second GPU arch unsupported")
        if not flydsl_impl._compile_target_matches(1):
            self.skipTest("FlyDSL compile target differs from the second GPU")
        src = _int_valued((256, 512), torch.float32, seed=21).to("cuda:1")
        dst = _int_valued((128, 512), torch.float32, seed=22).to("cuda:1")
        idx = _indices("uniform", 256, 128).to("cuda:1")
        with pn.flydsl.disabled():
            ref = dst.clone().index_add_(0, idx, src)
        with torch.cuda.device(0):
            dst.index_add_(0, idx, src)
        self._assert_dispatched()
        self.assertEqual(dst, ref, atol=0, rtol=0)

    def test_heavy_duplication_fp32_accuracy(self):
        # Non-integer data, all rows to one destination: compare against an
        # fp64 reference; the bound is loose fp32 accumulation error.
        g = torch.Generator(device="cuda").manual_seed(23)
        src = torch.randn((8192, 256), device="cuda", generator=g)
        dst = torch.zeros((4, 256), device="cuda")
        idx = torch.full((8192,), 2, device="cuda")
        dst.index_add_(0, idx, src)
        self._assert_dispatched()
        ref = src.double().sum(0)
        err = (dst[2].double() - ref).abs().max().item()
        bound = 8192 * torch.finfo(torch.float32).eps * src.abs().sum(0).max().item()
        self.assertLessEqual(err, bound)
        self.assertEqual(dst[[0, 1, 3]].abs().sum().item(), 0.0)

    @parametrize("bad", ("-1", "rows", "2**32"))
    def test_out_of_range_index_aborts(self, bad):
        # Runs in a subprocess: the kernel traps, which poisons the context.
        # A valid call first proves the shape dispatches to FlyDSL; the bad
        # call reuses that compiled kernel (conds never read index values).
        code = textwrap.dedent(
            f"""
            import torch
            from torch._native.ops.scatter_add import flydsl_impl
            from torch._native.ops.scatter_add.flydsl_kernels import (
                row_scatter_add_cache_info,
            )
            # Small enough to keep the trap cheap; below the shipped floor.
            flydsl_impl._MIN_SRC_BYTES[torch.float32] = 0
            rows = 16
            src = torch.ones((8, 256), device="cuda")
            dst = torch.zeros((rows, 256), device="cuda")
            idx = torch.zeros(8, dtype=torch.long, device="cuda")
            dst.index_add_(0, idx, src)
            torch.cuda.synchronize()
            if row_scatter_add_cache_info().misses == 1:
                print("DISPATCHED", flush=True)
            idx[5] = {bad}
            dst.index_add_(0, idx, src)
            torch.cuda.synchronize()
            print("NO_ABORT", flush=True)
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
        self.assertIn("DISPATCHED", proc.stdout, proc.stderr)
        self.assertNotIn("NO_ABORT", proc.stdout)
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class TestFlyDSLRowScatterAddRegistration(TestCase):
    """Host-only checks of the registration gate (no GPU or FlyDSL needed)."""

    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "version,expected",
        (
            ("0.3.4.1", True),
            ("0.3.4", True),
            ("0.3.2", False),
            ("0.3.3", False),
            ("0.4.1", False),
            ("0.3.4.2rc1", False),
        ),
    )
    def test_release_gate(self, version, expected):
        # check_native_version_skip() is cached during `import torch`, so
        # patch it rather than TORCH_NATIVE_SKIP_VERSION_CHECK.
        with (
            mock.patch.object(fu, "runtime_version", return_value=Version(version)),
            mock.patch.object(
                flydsl_impl, "check_native_version_skip", return_value=False
            ),
        ):
            self.assertEqual(flydsl_impl._flydsl_release_validated(), expected)

    def test_release_gate_honors_skip(self):
        with (
            mock.patch.object(fu, "runtime_version", return_value=Version("0.3.2")),
            mock.patch.object(
                flydsl_impl, "check_native_version_skip", return_value=True
            ),
        ):
            self.assertTrue(flydsl_impl._flydsl_release_validated())

    def test_unvalidated_release_registers_nothing(self):
        with (
            mock.patch.object(
                flydsl_impl, "_flydsl_release_validated", return_value=False
            ),
            mock.patch.object(fu, "register_op_override") as register,
        ):
            flydsl_impl.register_to_dispatch()
        register.assert_not_called()


instantiate_parametrized_tests(TestFlyDSLRowScatterAdd)
instantiate_parametrized_tests(TestFlyDSLRowScatterAddRegistration)


if __name__ == "__main__":
    run_tests()
