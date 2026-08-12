# Owner(s): ["module: dsl-native-ops"]
#
# WIRING tests for the CuTeDSL reduction overrides. NUMERIC correctness (value /
# shape / dtype vs a reference) is NOT tested here -- the overrides are transparent
# replacements for aten reduction kernels, so their numerics are covered by the
# existing numpy-referenced OpInfo suites (test_reductions.py, test_ops.py) running
# with the override active. Duplicating that here would (a) be weaker than the
# numpy reference and (b) risk self-reference (computing a tolerance via an
# overridden op recurses into the kernel under test).
#
# This file covers only invariants of the OVERRIDE WIRING that OpInfo cannot
# express: that a supported call actually routes through our kernel (vs silent
# fallback), that the capability `cond` declines unsupported inputs and lets aten
# serve them, and that the kernels capture into CUDA graphs.

import math
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


def _disabled():
    return torch.backends.python_native.cutedsl.disabled()


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestCuTeDSLReductionWiring(TestCase):
    def _fired_count(self, fn):
        # Count invocations of the dispatcher entry points our overrides funnel
        # through (reduce_dim / reduce_all single-output, reduce_dim2 two-output),
        # to prove a call routed to our kernel rather than aten.
        from torch._native.ops.reductions import kernel_general as kg

        names = ("reduce_dim", "reduce_dim2", "reduce_all")
        orig = {nm: getattr(kg, nm) for nm in names}
        n = [0]

        def wrap(f):
            def counting(*a, **k):
                n[0] += 1
                return f(*a, **k)

            return counting

        for nm in names:
            setattr(kg, nm, wrap(orig[nm]))
        try:
            fn()
        finally:
            for nm in names:
                setattr(kg, nm, orig[nm])
        return n[0]

    def test_supported_call_fires(self):
        # A supported call (CUDA, float, contiguous, valid dim) must route through
        # our kernel -- guards against a silently-all-fallback regression.
        x = torch.randn(128, 512, device="cuda")
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.mean(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.amax(x, dim=-1)), 1)
        # Group B: single-output index (argmax) and two-output (max.dim).
        self.assertEqual(self._fired_count(lambda: torch.argmax(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.max(x, dim=-1)), 1)
        # Group C: parameterized / non-float-output single reductions.
        self.assertEqual(self._fired_count(lambda: torch.var(x, dim=-1)), 1)
        self.assertEqual(
            self._fired_count(lambda: torch.linalg.vector_norm(x, dim=-1)), 1
        )
        self.assertEqual(self._fired_count(lambda: torch.count_nonzero(x, dim=-1)), 1)
        # Group D: two float-output reductions.
        self.assertEqual(self._fired_count(lambda: torch.var_mean(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.aminmax(x, dim=-1)), 1)

    def test_fp64_fires(self):
        # fp64 is a supported accumulator (not a fp32 cast), so a fp64 call must
        # route through our kernel rather than fall back to aten.
        x = torch.randn(128, 512, device="cuda", dtype=torch.float64)
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.amax(x, dim=-1)), 1)

    def test_unsupported_dtype_falls_back(self):
        # Integer input is outside the supported set -> must NOT hit our kernel.
        xi = torch.randint(0, 9, (64, 64), device="cuda")
        self.assertEqual(self._fired_count(lambda: torch.sum(xi, dim=-1)), 0)
        # ... and the result is still correct (served by aten).
        with _disabled():
            ref = torch.sum(xi, dim=-1)
        self.assertEqual(torch.sum(xi, dim=-1), ref)

    def test_noncontiguous_falls_back_correct(self):
        xt = torch.randn(64, 128, device="cuda").t()  # non-contiguous
        self.assertEqual(self._fired_count(lambda: torch.sum(xt, dim=-1)), 0)
        with _disabled():
            ref = torch.sum(xt, dim=-1)
        self.assertEqual(torch.sum(xt, dim=-1), ref)

    def test_scalar_falls_back(self):
        # 0-dim input must not crash the cond (regression: d % ndim with ndim==0).
        s = torch.tensor(3.5, device="cuda")
        self.assertEqual(self._fired_count(lambda: torch.sum(s)), 0)
        with _disabled():
            ref = torch.sum(s, dim=0, keepdim=True)
        self.assertEqual(torch.sum(s, dim=0, keepdim=True), ref)

    def test_invalid_dim_defers_to_aten(self):
        # dim args aten rejects (out-of-range / duplicate) must surface aten's
        # normal error -- the cond declines so aten validates, no wrapped result.
        x = torch.randn(4, 5, 6, device="cuda")
        with self.assertRaises(IndexError):
            torch.sum(x, dim=3)
        with self.assertRaises(RuntimeError):
            torch.sum(x, dim=(0, 0))

    def test_cow_input_served_and_preserved(self):
        # A copy-on-write input is SERVED by our kernel (it exports read-only via
        # ReadOnlyTensorWrapper -> from_dlpack reads through const_data_ptr()), and
        # must stay COW after -- reading it must not materialize it.
        base = torch.randn(128, 512, device="cuda")
        x = torch._lazy_clone(base)
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        self.assertTrue(torch._C._is_cow_tensor(x))

    def test_fast_geometry_routing(self):
        # The cond gates on whether a FAST kernel (row/col/xcta) can serve the
        # POST-TI-coalesce geometry; a K0-only geometry declines to aten (K0 is
        # ~5-8x slower than aten, so declining is a win, not a fallback).
        #   contiguous n-D whose reduced/kept axes coalesce -> served
        #   mid-dim / transposed / gapped (K0-only)            -> declined
        served = [
            ("2D last-dim", torch.randn(512, 512, device="cuda"), -1),
            ("2D dim0", torch.randn(512, 512, device="cuda"), 0),
            (
                "3D last-dim coalesces to row",
                torch.randn(64, 32, 512, device="cuda"),
                -1,
            ),
            (
                "3D dims (1,2) coalesce to row",
                torch.randn(128, 32, 32, device="cuda"),
                (1, 2),
            ),
        ]
        for name, x, dim in served:
            self.assertEqual(
                self._fired_count(lambda: torch.sum(x, dim=dim)),
                1,
                f"{name} should fire",
            )
        declined = [
            ("3D mid-dim (K0-only)", torch.randn(512, 512, 64, device="cuda"), 1),
            ("transpose (irregular)", torch.randn(512, 512, device="cuda").t(), -1),
        ]
        for name, x, dim in declined:
            self.assertEqual(
                self._fired_count(lambda: torch.sum(x, dim=dim)),
                0,
                f"{name} should decline",
            )
            with _disabled():
                ref = torch.sum(x, dim=dim)
            self.assertEqual(torch.sum(x, dim=dim), ref)  # aten serves it correctly

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs >= 2 GPUs")
    def test_other_device_defers(self):
        # A tensor not on the current device must fall back (kernel/stream caches
        # are current-device-bound). cuda:1 with cuda:0 current -> aten.
        x = torch.randn(128, 512, device="cuda:1")
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 0)

    def test_graph_capturable(self):
        # The override must capture into a CUDA graph and replay correctly (the
        # earlier _stream() bug made cute launches deadlock / produce empty graphs).
        x = torch.randn(8192, 1024, device="cuda")
        f = lambda: torch.sum(x, dim=-1)  # noqa: E731
        with _disabled():
            ref = f()
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                f()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = f()
        g.replay()
        torch.cuda.synchronize()
        self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

    def test_scalar_operand_is_served(self):
        # aten's unboxed parser turns `x * 2.0` into mul.Tensor(Tensor, Scalar) and
        # dispatches THAT overload -- the number never reaches aten::mul.Scalar. The
        # router now coerces the number to a 0-d tensor BEFORE the cond (it used to do
        # so only on the fallback path), and the cond treats a 0-d CPU operand as that
        # coerced number, so scalar calls are served instead of always declining.
        from torch._native.ops.pointwise import kernel as K

        def fired(fn):
            # Count real kernel launches -- plan-cache growth is not a liveness signal
            # once the cache is warm from an earlier identical call.
            orig, n = K.run, [0]

            def counting(*a, **k):
                n[0] += 1
                return orig(*a, **k)

            K.run = counting
            try:
                fn()
            finally:
                K.run = orig
            return n[0]

        x = torch.randn(1024, device="cuda")
        for fn in (
            lambda: x * 2.0,
            lambda: x + 2.0,
            lambda: x - 2.0,
            lambda: x / 2.0,
            lambda: torch.fmod(x, 2.0),
        ):
            got = fn()
            with _disabled():
                ref = fn()
            self.assertEqual(got, ref)
            self.assertEqual(fired(fn), 1, "scalar call should route to our kernel")
        # WEAK promotion must survive: a python number never widens the tensor dtype.
        b = torch.randn(64, device="cuda", dtype=torch.bfloat16)
        self.assertEqual((b * 2.0).dtype, torch.bfloat16)
        i = torch.randint(0, 9, (64,), device="cuda", dtype=torch.int32)
        self.assertEqual((i * 2).dtype, torch.int32)  # int stays int
        self.assertEqual((i + 0.5).dtype, torch.float32)  # float promotes the category
        # A dim>0 CPU tensor is a genuine cross-device call, not a scalar -> aten raises.
        with self.assertRaises(RuntimeError):
            x + torch.randn(1024)

    def test_pow_scalar_base_on_cuda(self):
        # REGRESSION: aten's pow_Scalar_out builds a wrapped_scalar_tensor on the
        # EXPONENT's device and redispatches to pow_out (Pow.cpp), so a CUDA
        # wrapped-number tensor reached our Python router; the boxed->Python
        # conversion asserts is_cpu() and this raised INTERNAL ASSERT. The assert
        # fires before any cond runs, so pow's .out overload is left unregistered.
        x = torch.tensor([1.0, 2.0], device="cuda")
        self.assertEqual((2.0**x).cpu(), torch.tensor([2.0, 4.0]))
        self.assertEqual(torch.pow(2.0, x).cpu(), torch.tensor([2.0, 4.0]))
        # float_power always promotes to double
        self.assertEqual(
            torch.float_power(2.0, x).cpu(),
            torch.tensor([2.0, 4.0], dtype=torch.float64),
        )
        xi = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
        self.assertEqual(torch.ldexp(xi, xi).cpu(), torch.tensor([2.0, 8.0]))
        # the tensor-tensor overload we DO serve still works
        self.assertEqual(torch.pow(x, x).cpu(), torch.tensor([1.0, 4.0]))

    def test_sub_warp_row_width_does_not_crash(self):
        # REGRESSION: the tpr ladder's small-N rungs return 8/16 threads per row,
        # and the cross-thread reduce divides by warps_per_row = tpr // WARP, which
        # floored to 0 -> ZeroDivisionError at trace time. That was a hard crash on
        # ordinary calls (x.sum(dim=1) for N=32/64/128), not a fallback. tpr is now
        # floored at one warp. Non-monotonic in N, so cover the whole small range.
        for n in (8, 16, 24, 32, 33, 48, 63, 64, 96, 128, 192):
            x = torch.rand(257, n, device="cuda")
            self.assertEqual(
                torch.sum(x, dim=1), x.double().sum(dim=1).float(), atol=1e-3, rtol=1e-3
            )
            torch.linalg.vector_norm(x, 2, dim=1)  # same reduce path, must not raise

    def test_strided_single_element_view_is_served(self):
        # REGRESSION: is_contiguous() is True for ANY single-element tensor whatever its
        # stride (with one element the stride is unobservable -- every stride addresses
        # the same element), so a.diagonal(offset=2) on (5,3) is a contiguous shape-(1,)
        # tensor that still declares stride (4,). The DSL compares the declared stride
        # against stride_order and rejected it ("The stride_order is not consistent with
        # the layout") -- a hard error on an ordinary sum, not a fallback. The wrap now
        # restrides such a tensor to the canonical form, so these are SERVED (declining
        # would give up coverage for a difference that cannot be observed).
        a = torch.randn(5, 3, device="cuda", dtype=torch.float64)
        d = a.diagonal(offset=2)
        self.assertEqual(d.shape, torch.Size([1]))
        self.assertNotEqual(d.stride(), (1,))  # the leftover stride is the whole point
        for fn in (lambda z: z.sum(), lambda z: z.mean()):
            with _disabled():
                ref = fn(d)
            self.assertEqual(fn(d), ref)

    def test_misaligned_and_lazy_metadata_inputs_decline(self):
        # Two cond gates that cannot be expressed as numerics:
        #   - A base pointer that is not 16-byte aligned: the row/col/xcta wraps claim an
        #     N-derived alignment that from_dlpack VALIDATES, and the compiled kernel
        #     BAKES its load width, so a plan built for an aligned call cannot serve a
        #     misaligned one (clamping the claim per call is not enough). Raised
        #     "Misaligned Tensor data" mid-call on an ordinary sum over a slice.
        #   - A NEG/CONJ bit: lazy metadata, so the exported buffer holds the UNNEGATED
        #     values. It must not be resolved in a cond either, since aten materializes
        #     such a view by CALLING copy_ -- any override of copy_ is then re-entered.
        base = torch.arange(4096, device="cuda", dtype=torch.float64) + 1
        for off in (0, 1, 2, 3):
            t = base[off : off + 512].view(256, 2)
            for fn in (
                lambda z: z.sum(dim=0),
                lambda z: z.sum(dim=1),
                lambda z: z.sum(),
            ):
                with _disabled():
                    ref = fn(t)
                self.assertEqual(fn(t), ref, f"off={off}")
        n = torch.randn(64, device="cuda", dtype=torch.float64)._neg_view()
        self.assertTrue(n.is_neg())
        for fn in (lambda z: z.sum(dim=0), lambda z: z.sum(), lambda z: z.mean()):
            with _disabled():
                ref = fn(n)
            self.assertEqual(fn(n), ref)

    def test_large_scalar_arg_compiles_and_is_not_baked(self):
        # REGRESSION: the DSL mangles every non-IR jit argument's VALUE into the
        # generated MLIR symbol name. Python's repr switches to exponent form at 1e16
        # ("1e+16") and the mangler does not strip "+", so the symbol was unparsable
        # and the compile died with an ICE ("expected '('"). That made any op taking a
        # large scalar fail -- including nan_to_num's DEFAULT posinf, which is the
        # dtype's finite max (3.4e38 for fp32, 1.8e308 for fp64). The kernel takes the
        # scalar as a runtime arg, so it now compiles against a placeholder value.
        from torch._native.ops.pointwise import kernel as K

        x = torch.randn(1024, device="cuda")
        for v in (1e16, 1e20, 3.4e38, -1e20, 1e-20, 2.5):
            got = torch.nn.functional.leaky_relu(x, negative_slope=v)
            with _disabled():
                ref = torch.nn.functional.leaky_relu(x, negative_slope=v)
            self.assertEqual(got, ref)
        # Scalars are runtime args, so many distinct VALUES must share one compile.
        n = len(K._KERNELS)
        for v in (3.0, 4.0, 5.0, 1e17, 1e18):
            torch.nn.functional.leaky_relu(x, negative_slope=v)
        self.assertEqual(len(K._KERNELS), n, "scalar value must not key the kernel")
        # nan_to_num's omitted bounds saturate at the OUTPUT dtype's max, not the fp32
        # compute dtype's (fp16 -> 65504); a wrong fill overflows back to inf.
        vals = [float("nan"), float("inf"), float("-inf"), 1.5]
        sp = torch.tensor(vals, device="cuda")
        for dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            t = sp.to(dt)
            with _disabled():
                ref = torch.nan_to_num(t)
            self.assertEqual(torch.nan_to_num(t), ref)

    def test_misaligned_view_shares_no_plan_with_aligned(self):
        # REGRESSION: the plan cache keyed on (shape, stride) but NOT on 16-byte
        # alignment, so base[0:16] and base[1:17].view(16, 2) -- same shape AND stride,
        # different alignment -- shared one plan. The aligned call picks the vec path,
        # which bakes assumed_align=16, so the misaligned call reusing that plan died in
        # from_dlpack with "Misaligned Tensor data on mIns[0]". Alignment is now part of
        # the key, and an out=/in-place target is alignment-gated in the cond too (the
        # kernel compiles against a fresh, always-aligned seed output).
        for dt in (torch.float64, torch.float32, torch.float16, torch.int32):
            base = torch.arange(4096, device="cuda").to(dt) + 1
            for off in (0, 1, 2, 4, 7, 8):
                for shape in ((512,), (256, 2), (64, 8)):
                    n = math.prod(shape)
                    t = base[off : off + n].view(shape)
                    for fn in (lambda z: z * z, lambda z: -z, torch.sigmoid):
                        with _disabled():
                            ref = fn(t)
                        self.assertEqual(fn(t), ref, f"{dt} off={off} shape={shape}")

    def test_empty_copy_destination_falls_back(self):
        # REGRESSION: copy_ BROADCASTS src up to self, so a 1-element source into an
        # EMPTY destination reached the kernel with a perfectly valid source (the
        # conversion gate only rejects an empty SOURCE). aten treats that as a no-op; for
        # us it is a zero-element grid -> "CUDA Error: cudaErrorInvalidConfiguration", or
        # an invalid cute layout when a shape extent is 0. Reached in practice via
        # linalg.matrix_power(torch.empty(0, 2, 2), 0), which fills an identity.
        dst = torch.empty(0, 2, device="cuda", dtype=torch.float64)
        dst.copy_(torch.ones(1, device="cuda", dtype=torch.float64))  # must not raise
        e = torch.empty(0, 2, 2, device="cuda", dtype=torch.float64)
        with _disabled():
            ref = torch.linalg.matrix_power(e, 0)
        self.assertEqual(torch.linalg.matrix_power(e, 0), ref)

    def test_scalar_in_first_operand_slot_is_served(self):
        # REGRESSION: the cond took its DEVICE reference from operand 0, but aten puts the
        # coerced scalar FIRST for the reflected overloads -- rsub.Scalar is
        # `at::sub(wrapped_scalar, self)`, and remainder.Scalar_Tensor / xlogy.Scalar_Self
        # are declared that way. Operand 0 was then a 0-d CPU tensor, failed the
        # is-this-CUDA test, and every such call declined (`1.0 - t` fired nothing). The
        # reference is now the first operand that is not a coerced scalar.
        from torch._native.ops.pointwise import kernel as K

        def served(fn):
            orig, n = K.run, [0]

            def counting(*a, **k):
                n[0] += 1
                return orig(*a, **k)

            K.run = counting
            try:
                out = fn()
            finally:
                K.run = orig
            return n[0], out

        t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        i = torch.tensor([2, 4, 6], dtype=torch.int32, device="cuda")
        for fn in (
            lambda: 1.0 - t,
            lambda: torch.remainder(2.0, t),
            lambda: torch.xlogy(2.0, t),
            lambda: torch.bitwise_and(3, i),
        ):
            n, got = served(fn)
            self.assertGreaterEqual(n, 1, "scalar-first call must be served")
            with _disabled():
                self.assertEqual(got, fn())
        # A genuine cross-device call must STILL decline and let aten raise, i.e. the
        # loosened reference must not have loosened the device check itself.
        cpu = torch.tensor([1.0, 2.0, 3.0])
        for fn in (lambda: t + cpu, lambda: cpu + t):
            with self.assertRaises(RuntimeError):
                fn()

    def test_int64_reduction_on_the_column_path(self):
        # REGRESSION: kernel_col._PART_TORCH lacked Int64 while kernel_general and
        # kernel_xcta had it, so K2 was the one path that KeyError'd when allocating an
        # int64 stage-1 partial buffer -- and integer reductions accumulate in int64.
        # dim=0 is the column path; dim=1 the row path, as a control.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_general as kg

        x = torch.arange(256 * 64, device="cuda", dtype=torch.int64).reshape(256, 64)
        for dim in (0, 1):
            out = kg.reduce_dim(
                T.SumOps(acc=cutlass.Int64), ("sum_i64",), x, {dim}, torch.int64
            )
            ref = x.sum(dim=dim)
            self.assertEqual(out.reshape(ref.shape), ref, f"dim={dim}")

    def test_vector_norm_ord0_and_nansum(self):
        # ord=0 is the NONZERO COUNT, not a |x|**p sum, so it needs CountNonzeroOps rather
        # than NormOps -- the cond used to decline it outright. And NanSumOps existed in
        # the trait library with zero call sites; wiring it also gets nanmean, which aten
        # decomposes into nansum / isnan.logical_not.sum.
        x = torch.tensor([[0.0, 1.0, 0.0, 2.0], [3.0, 0.0, 0.0, 0.0]], device="cuda")
        for ord_ in (0, 1, 2, 3, float("inf"), float("-inf")):
            got = torch.linalg.vector_norm(x, ord_, dim=1)
            with _disabled():
                self.assertEqual(got, torch.linalg.vector_norm(x, ord_, dim=1))
        nan = float("nan")
        y = torch.tensor(
            [[1.0, nan, 3.0], [nan, nan, nan], [4.0, 5.0, 6.0]], device="cuda"
        )
        for fn in (
            lambda: torch.nansum(y, dim=1),
            lambda: torch.nansum(y, dim=0),
            lambda: torch.nansum(y),
            lambda: torch.nansum(y, dim=1, dtype=torch.float64),
            lambda: torch.nanmean(y, dim=1),
        ):
            with _disabled():
                ref = fn()
            self.assertEqual(fn(), ref, equal_nan=True)

    def test_nullary_fill_serves_every_layout(self):
        # fill_ is the NULLARY (nin == 0) case: no input tensor at all, so the caller's
        # `self` is the only source of shape/device/layout AND the destination. Everything
        # in the kernel that normally reads inputs[0] has to come from the output instead.
        #
        # Coverage is the contract -- every layout is SERVED, not just the fast ones: a
        # contiguous aligned target takes the vec path, and unaligned / transposed /
        # strided targets fall to the strided route (which bakes the real layout, hence
        # the output layout appearing in both the plan and kernel keys).
        from torch._native.ops.pointwise import kernel as K

        def served(fn):
            orig, n = K.run, [0]

            def counting(*a, **k):
                n[0] += 1
                return orig(*a, **k)

            K.run = counting
            try:
                fn()
            finally:
                K.run = orig
            return n[0]

        base = torch.empty(4096, device="cuda")
        targets = (
            ("contiguous", torch.empty(1024, device="cuda")),
            ("transposed", torch.empty(32, 32, device="cuda").t()),
            ("strided", torch.empty(2048, device="cuda")[::2]),
            ("0-dim", torch.empty((), device="cuda")),
        )
        for name, t in targets:
            self.assertEqual(
                served(lambda t=t: t.fill_(2.5)), 1, f"{name} must be served"
            )
            self.assertTrue(bool((t == 2.5).all()), name)
        # A nonzero STORAGE OFFSET declines for now (cap.dlpack_offset_ok): CuteDSL's
        # tvm-ffi entry asserts the DLPack byte_offset is 0, and pytorch#182924 made that
        # field carry the offset. It is a layout we can otherwise serve -- the strided
        # route handles it -- so this expectation flips back to `served == 1` when that
        # gate is removed. Values must still be right, via aten.
        offset_target = base[1:1025]
        self.assertEqual(served(lambda t=offset_target: t.fill_(2.5)), 0)
        self.assertTrue(bool((offset_target == 2.5).all()))
        # Every constructor aten builds on empty()+fill_ rides the same override, with no
        # per-constructor row: full / zeros / ones / full_like, floats and ints alike.
        for fn in (
            lambda: torch.full((1024,), 3.5, device="cuda"),
            lambda: torch.zeros(1024, device="cuda"),
            lambda: torch.ones(1024, device="cuda"),
            lambda: torch.full((1024,), 9, device="cuda", dtype=torch.int64),
        ):
            self.assertEqual(served(fn), 1)
            with _disabled():
                ref = fn()
            self.assertEqual(fn(), ref)
        # An fp64 value must not narrow through fp32 (the const has to be boxed in the
        # compute dtype: 1e20 came back as its fp32 neighbour before that fix).
        d = torch.empty(8, device="cuda", dtype=torch.float64).fill_(1e20)
        self.assertEqual(d[0].item(), 1e20)

    def test_range_factories_use_the_flat_index(self):
        # arange/linspace are nullary AND index-consuming: each element's value comes from
        # its FLAT INDEX alone, which aten expresses with gpu_kernel_with_index and we
        # expose as the kernel's `with_index` flag (strided route only -- the vectorized
        # routes hand the fn a whole V-wide fragment, so there is no single index).
        from torch._native.ops.pointwise import kernel as K

        def served(fn):
            orig, n = K.run, [0]

            def counting(*a, **k):
                n[0] += 1
                return orig(*a, **k)

            K.run = counting
            try:
                out = fn()
            finally:
                K.run = orig
            return n[0], out

        # linspace: fp32/fp64 and the integer dtypes are BIT-EXACT with aten.
        for dt in (torch.float32, torch.float64, torch.int32, torch.int64):
            for a, b, steps in ((0, 1, 5), (-1, 1, 9), (5, -5, 64), (2.5, 3.5, 17)):
                fn = lambda: torch.linspace(  # noqa: E731
                    a, b, steps, device="cuda", dtype=dt
                )
                n, got = served(fn)
                self.assertEqual(n, 1, f"linspace {dt} must be served")
                with _disabled():
                    self.assertEqual(got, fn())
        # Halves compute in FP32 and narrow only on the store, where aten runs the whole
        # expression in scalar_t -- so we can differ by well under one ULP, on the MORE
        # accurate side. Endpoints stay exact regardless (that is what aten's halfway
        # split buys, and why the kernel reproduces it rather than stepping forward
        # throughout).
        for dt in (torch.float16, torch.bfloat16):
            for a, b, steps in ((0, 1, 5), (-1, 1, 64), (5, -5, 1001)):
                got = torch.linspace(a, b, steps, device="cuda", dtype=dt)
                with _disabled():
                    ref = torch.linspace(a, b, steps, device="cuda", dtype=dt)
                span = max(abs(float(a)), abs(float(b)))
                tol = torch.finfo(dt).eps * span
                self.assertLess((got.double() - ref.double()).abs().max().item(), tol)
                self.assertEqual(got[0].item(), torch.tensor(a, dtype=dt).item())
                self.assertEqual(got[-1].item(), torch.tensor(b, dtype=dt).item())
        # arange: we override arange.start_out, which the functional form reaches only
        # from C++ (at::arange_out), so drive the .out form the override actually serves.
        for dt in (torch.float32, torch.float64, torch.int32, torch.int64):
            for s, e, st in ((0, 10, 1), (0.0, 5.0, 0.5), (10, 0, -1), (-5, 5, 2)):
                with _disabled():
                    ref = torch.arange(s, e, st, device="cuda", dtype=dt)
                out = torch.empty_like(ref)
                n, got = served(lambda out=out: torch.arange(s, e, st, out=out))
                self.assertEqual(n, 1, f"arange {dt} must be served")
                self.assertEqual(got, ref)

    def test_optional_scalar_is_not_silently_dropped(self):
        # REGRESSION (wrong RESULTS, not a crash): logit's eps is `float?`, and the row
        # originally declared no scalars at all on the theory that an explicit eps would
        # decline. It did not -- the arg was silently IGNORED, so torch.logit(x, 1e-3)
        # ran the unclamped kernel and returned nan where aten clamps. An omitted eps
        # means "no clamping", which aten spells as a negative sentinel; optional_defaults
        # supplies it, so one row serves both overloads.
        x = torch.tensor(
            [0.0, 1e-8, 0.5, 0.9999, 1.5, -0.5, float("nan")], device="cuda"
        )
        for fn in (
            lambda t: torch.logit(t),
            lambda t: torch.logit(t, 1e-3),
            lambda t: torch.logit(t, 0.0),
            lambda t: torch.special.logit(t, eps=0.1),
            # eps > 0.5 CROSSES the bounds (lo=0.6 > hi=0.4). aten's nested ternary
            # returns lo and never re-clamps; a sequential clamp-low-then-high would pull
            # it back to hi and FLIP THE SIGN of the log (caught by test_out_logit).
            lambda t: torch.logit(t, 0.6),
        ):
            with _disabled():
                ref = fn(x)
            self.assertEqual(fn(x), ref, equal_nan=True)

    def test_pointwise_neg_and_conj_views_decline(self):
        # REGRESSION: a neg/conj bit is LAZY metadata -- the buffer holds the UNNEGATED
        # values -- and aten materializes such a view BY CALLING copy_, which THIS
        # commit's copy_ override intercepts. Resolving it inside a cond therefore
        # recursed until the stack blew: a plain torch.sin on a _neg_view() input raised
        # RecursionError (also relu, mul, clone, and every reduction, since they all
        # funnel through the same copy_). Declining lets aten resolve the bit.
        x = torch.randn(64, device="cuda", dtype=torch.float64)
        n = x._neg_view()
        self.assertTrue(n.is_neg())
        for fn in (
            torch.sin,
            torch.relu,
            torch.nan_to_num,
            lambda t: t * 2.0,
            lambda t: torch.clamp(t, -1.0, 1.0),
            lambda t: t.clone(),
            lambda t: t.float(),
        ):
            with _disabled():
                ref = fn(n)
            self.assertEqual(fn(n), ref)  # must not raise RecursionError
        # A conj view of a complex tensor is likewise declined rather than misread.
        c = torch.randn(64, device="cuda", dtype=torch.complex64).conj()
        with _disabled():
            ref = torch.real(c)
        self.assertEqual(torch.real(c), ref)


if __name__ == "__main__":
    run_tests()
