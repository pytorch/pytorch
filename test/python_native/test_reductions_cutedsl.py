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


if __name__ == "__main__":
    run_tests()
