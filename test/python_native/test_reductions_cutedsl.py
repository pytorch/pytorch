# Owner(s): ["module: dsl-native-ops"]
#
# Wiring tests for routing, fallback, and CUDA graph capture, plus numerics against ATen
# for the dtypes and dimension forms the overrides serve.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


def _disabled():
    return torch.backends.python_native.cutedsl.disabled()


# The overrides admit compute-capability majors 9 and 10 (overrides._ARCH_MAJORS).
_ARCH_OK = TEST_CUDA and torch.cuda.get_device_capability()[0] in (9, 10)


@unittest.skipUnless(_ARCH_OK, "requires an SM90 or SM100-family GPU")
@skipIfNoCuteDSL
class TestCuTeDSLReductionWiring(TestCase):
    hw_classification = HardwareClassification.CUDA

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

    @parametrize("op", ["sum", "mean", "amax", "amin", "prod"])
    def test_supported_call_fires(self, op):
        """Route every registered core operation instead of silently falling back."""
        x = torch.randn(128, 512, device="cuda")
        fn = getattr(torch, op)
        self.assertEqual(self._fired_count(lambda: fn(x, dim=-1)), 1)

    def test_unsupported_dtype_falls_back(self):
        """Leave unsupported dtypes to ATen instead of entering a mismatched kernel."""
        xi = torch.randint(0, 9, (64, 64), device="cuda")
        self.assertEqual(self._fired_count(lambda: torch.sum(xi, dim=-1)), 0)

    @parametrize(
        "dim", [-1, 0, None], name_fn=lambda d: "all" if d is None else f"dim{d}"
    )
    def test_noncontiguous_is_served(self, dim):
        """Serve noncontiguous dimension and full reductions through indexed addressing."""
        xt = torch.randn(64, 128, device="cuda").t()
        fn = (
            (lambda: torch.sum(xt)) if dim is None else (lambda: torch.sum(xt, dim=dim))
        )
        self.assertEqual(self._fired_count(fn), 1)

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
    def test_other_device_is_served(self):
        """Serve tensors on a non-current device under that device's guard."""
        x = torch.randn(128, 512, device="cuda:1")
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        with _disabled():
            ref = torch.sum(x, dim=-1)
        self.assertEqual(torch.sum(x, dim=-1), ref, atol=1e-4, rtol=1e-4)

    @parametrize("op", ["sum", "mean", "prod", "amax", "amin"])
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_matches_aten(self, op, dtype):
        """Match ATen across input dtypes, multi-dimension keepdim, and full reductions."""
        x = torch.rand(6, 33, 40, device="cuda", dtype=dtype) + 0.5
        fn = getattr(torch, op)
        calls = [lambda: fn(x, dim=1), lambda: fn(x, dim=1, keepdim=True)]
        if op != "prod":
            calls.append(lambda: fn(x, dim=(0, 2), keepdim=True))
        if op in ("sum", "mean"):
            calls.append(lambda: fn(x))
        tol = 1e-4 if dtype == torch.float32 else 1e-2
        for call in calls:
            self.assertGreater(self._fired_count(call), 0)
            with _disabled():
                ref = call()
            got = call()
            self.assertEqual(got.dtype, ref.dtype)
            self.assertEqual(got, ref, atol=tol, rtol=tol)

    @parametrize("op", ["sum", "mean", "prod"])
    def test_out_dtype_contract(self, op):
        """Serve widening dtype requests and leave input-casting requests to ATen."""
        fn = getattr(torch, op)
        x16 = torch.rand(64, 96, device="cuda", dtype=torch.float16) + 0.5
        widen = lambda: fn(x16, dim=1, dtype=torch.float32)  # noqa: E731
        self.assertEqual(self._fired_count(widen), 1)
        with _disabled():
            ref = widen()
        self.assertEqual(widen(), ref, atol=1e-3, rtol=1e-3)
        # ATen casts the input to a narrower requested dtype before reducing.
        x32 = torch.tensor([[7e4, -7e4]] * 32, device="cuda")
        narrow = lambda: fn(x32, dim=1, dtype=torch.float16)  # noqa: E731
        self.assertEqual(self._fired_count(narrow), 0)

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


instantiate_parametrized_tests(TestCuTeDSLReductionWiring)

if __name__ == "__main__":
    run_tests()
