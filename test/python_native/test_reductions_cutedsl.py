# Owner(s): ["module: dsl-native-ops"]
#
# Wiring tests for routing, fallback, and CUDA graph capture. OpInfo tests cover
# numerical behavior.

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
        from torch._native.ops.reductions import kernel_general as kg

        names = ("reduce_dim", "reduce_all")
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
        """Route every registered core operation instead of silently falling back."""
        x = torch.randn(128, 512, device="cuda")
        for op in (torch.sum, torch.mean, torch.amax, torch.amin, torch.prod):
            with self.subTest(op=op):
                self.assertEqual(self._fired_count(lambda: op(x, dim=-1)), 1)

    def test_unsupported_dtype_falls_back(self):
        """Leave unsupported dtypes to ATen instead of entering a mismatched kernel."""
        xi = torch.randint(0, 9, (64, 64), device="cuda")
        self.assertEqual(self._fired_count(lambda: torch.sum(xi, dim=-1)), 0)

    def test_noncontiguous_is_served(self):
        """Serve noncontiguous dimension and full reductions through indexed addressing."""
        xt = torch.randn(64, 128, device="cuda").t()
        for fn in (
            lambda t: torch.sum(t, dim=-1),
            lambda t: torch.sum(t, dim=0),
            lambda t: torch.sum(t),  # reduce-ALL of a non-contiguous input
        ):
            with self.subTest(fn=fn):
                self.assertEqual(self._fired_count(lambda: fn(xt)), 1)

    def test_scalar_falls_back(self):
        """Decline scalar input without dividing dimension indices by zero."""
        s = torch.tensor(3.5, device="cuda")
        self.assertEqual(self._fired_count(lambda: torch.sum(s)), 0)

    def test_invalid_dim_defers_to_aten(self):
        """Preserve ATen validation for out-of-range and duplicate dimensions."""
        x = torch.randn(4, 5, 6, device="cuda")
        with self.assertRaises(IndexError):
            torch.sum(x, dim=3)
        with self.assertRaises(RuntimeError):
            torch.sum(x, dim=(0, 0))

    def test_cow_input_served_and_preserved(self):
        """Read copy-on-write inputs without materializing or mutating their storage."""
        base = torch.randn(128, 512, device="cuda")
        x = torch._lazy_clone(base)
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        self.assertTrue(torch._C._is_cow_tensor(x))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs >= 2 GPUs")
    def test_other_device_defers(self):
        """Decline tensors whose device does not own the active stream and caches."""
        x = torch.randn(128, 512, device="cuda:1")
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 0)

    def test_graph_capturable(self):
        """Keep core reduction overrides capture-safe under graph replay."""
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
