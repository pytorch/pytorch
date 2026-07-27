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
        # Group C: parameterized / non-float-output single reductions.
        self.assertEqual(self._fired_count(lambda: torch.var(x, dim=-1)), 1)
        self.assertEqual(
            self._fired_count(lambda: torch.linalg.vector_norm(x, dim=-1)), 1
        )
        self.assertEqual(self._fired_count(lambda: torch.count_nonzero(x, dim=-1)), 1)
        # Group D: two float-output reductions.
        self.assertEqual(self._fired_count(lambda: torch.var_mean(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.aminmax(x, dim=-1)), 1)

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
                self._fired_count(lambda: torch.sum(x, dim=dim)), 1, f"{name} should fire"
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


if __name__ == "__main__":
    run_tests()
