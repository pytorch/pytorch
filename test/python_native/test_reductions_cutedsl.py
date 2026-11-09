# Owner(s): ["module: dsl-native-ops"]
#
# Wiring tests for routing, fallback, and CUDA graph capture, plus numerics against ATen
# for the dtypes, dimension forms, and layouts the overrides serve.

import functools
import os
import unittest
from unittest import mock

import torch
from torch._native import cutedsl_utils as cu
from torch._native.ops.reductions import cutedsl_impl, overrides as ov
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


def _disabled():
    return torch.backends.python_native.cutedsl.disabled()


_TOL = {
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
    torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
    torch.float32: {"atol": 1e-4, "rtol": 1e-4},
}


class TestCuTeDSLReductionWiring(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        # `cu.register_op_override` drops every registration when the runtime is
        # missing, disabled, or not known-good, so these tests would find no overrides.
        if (
            not cu.runtime_available()
            or cu.check_native_jit_disabled()
            or not cu._version_is_ok()
        ):
            self.skipTest(
                "CuTeDSL runtime unavailable, disabled, or at an unsupported version"
            )
        major = torch.cuda.get_device_capability()[0]
        if torch.version.hip or major not in ov._ARCH_MAJORS:
            self.skipTest("the overrides serve SM90 and SM100-family CUDA GPUs only")
        # The tests count served calls in replay entries, which sum and prod calls served
        # first by register_to_dispatch's inner-tree family do not add.
        if cutedsl_impl._INNER_TREE_ENABLED:
            self.skipTest("PYTORCH_SUM_INNER_TREE routes sum and prod elsewhere first")

    def _run(self, call):
        """Return call's result and how many override calls served it."""
        from torch._native.ops.reductions import kernel_general as kg

        # A served call adds one replay entry for its metadata, so start from none.
        kg._REPLAYS.clear()
        out = call()
        return out, len(kg._REPLAYS)

    def _check(self, call, fired=1, **tol):
        got, fires = self._run(call)
        self.assertEqual(fires, fired)
        with _disabled():
            ref = call()
        self.assertEqual(got, ref, **tol)

    @parametrize("op", ["sum", "mean", "amax", "amin", "prod"])
    def test_supported_call_fires(self, device, op):
        """Route every registered core operation instead of silently falling back."""
        x = torch.randn(128, 512, device=device)
        fn = getattr(torch, op)
        self.assertEqual(self._run(lambda: fn(x, dim=-1))[1], 1)

    def test_per_overload_opt_out(self, device):
        """Disable every override by the overload names the overrides document."""
        x = torch.randn(128, 512, device=device)
        ops = ("sum", "mean", "prod", "amax", "amin")
        calls = [functools.partial(getattr(torch, op), x, dim=-1) for op in ops]
        names = ("sum.dim_IntList", "mean.dim", "prod.dim_int", "amax", "amin")
        with torch.backends.python_native.operations_disabled(*names):
            self.assertEqual([self._run(call)[1] for call in calls], [0] * len(calls))
        self.assertEqual([self._run(call)[1] for call in calls], [1] * len(calls))

    def test_unsupported_dtype_falls_back(self, device):
        """Leave unsupported dtypes to ATen instead of entering a mismatched kernel."""
        xi = torch.randint(0, 9, (64, 64), device=device)
        self.assertEqual(self._run(lambda: torch.sum(xi, dim=-1))[1], 0)
        xd = torch.randn(64, 64, device=device, dtype=torch.float64)
        self.assertEqual(self._run(lambda: torch.sum(xd, dim=-1))[1], 0)

    def test_unregistered_overloads_fall_back(self, device):
        """Leave full prod and the out= overloads, which have no override, to ATen."""
        x = torch.rand(64, 96, device=device) + 0.5
        out = torch.empty(64, device=device)
        self.assertEqual(self._run(lambda: torch.prod(x))[1], 0)
        for op in ("sum", "mean", "prod", "amax", "amin"):
            call = functools.partial(getattr(torch, op), x, dim=1, out=out)
            self.assertEqual(self._run(call)[1], 0)

    @parametrize(
        "dim", [-1, 0, None], name_fn=lambda d: "all" if d is None else f"dim{d}"
    )
    def test_noncontiguous_is_served(self, device, dim):
        """Serve noncontiguous dimension and full reductions through indexed addressing."""
        xt = (torch.rand(64, 128, device=device) + 0.5).t()
        self._check(lambda: torch.sum(xt, dim=dim), **_TOL[torch.float32])

    def test_strided_layouts_are_served(self, device):
        """Serve sliced, permuted, and zero-stride inputs through the general decode."""
        base = torch.rand(8, 33, 64, device=device) + 0.5
        xp = base[:, 1:, ::2].permute(2, 0, 1)
        xe = (torch.rand(1, 512, device=device) + 0.5).expand(64, 512)
        calls = [
            lambda: torch.sum(xp, dim=1),
            lambda: torch.amax(xp, dim=(0, 2)),
            lambda: torch.mean(xp),
            lambda: torch.sum(xe, dim=0),
            lambda: torch.sum(xe, dim=-1),
            lambda: torch.amin(xe),
        ]
        for call in calls:
            self._check(call, **_TOL[torch.float32])

    def test_scalar_falls_back(self, device):
        """Decline scalar input without dividing dimension indices by zero."""
        s = torch.tensor(3.5, device=device)
        self.assertEqual(self._run(lambda: torch.sum(s))[1], 0)

    def test_invalid_dim_defers_to_aten(self, device):
        """Preserve ATen validation for out-of-range and duplicate dimensions."""
        x = torch.randn(4, 5, 6, device=device)
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            torch.sum(x, dim=3)
        with self.assertRaisesRegex(RuntimeError, "appears multiple times"):
            torch.sum(x, dim=(0, 0))

    def test_unsupported_layouts_decline(self, device):
        """Leave misaligned and empty inputs to ATen, keeping its empty-amax error."""
        flat = torch.rand(128 * 512 + 1, device=device, dtype=torch.float16)
        misaligned = flat[1:].view(128, 512)
        self.assertNotEqual(misaligned.data_ptr() % 16, 0)
        empty = torch.empty(0, 64, device=device)
        for call in (
            lambda: torch.sum(misaligned, dim=-1),
            lambda: torch.sum(empty, dim=0),
            lambda: torch.amax(empty, dim=-1),
        ):
            self._check(call, fired=0)
        with self.assertRaisesRegex(RuntimeError, "reduction dim to be specified"):
            torch.amax(empty)

    def test_neg_view_declines(self, device):
        """Decline negative views in the condition; dispatch resolves them before CUDA."""
        xn = torch._neg_view(torch.rand(128, 512, device=device))
        self.assertFalse(ov._cond(xn, -1))
        # The Negative fallback clones the view, so the CUDA kernel serves the clone.
        self._check(lambda: torch.sum(xn, dim=-1), **_TOL[torch.float32])

    @parametrize("kind", ["meta", "fake", "functional_fake"])
    @skipIfTorchDynamo("host-side capability predicates need no dynamo compilation")
    def test_traced_inputs_decline(self, device, kind):
        """Decline tensors without launchable storage."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        if kind == "meta":
            x = torch.empty(4, 128, device="meta")
        else:
            with FakeTensorMode():
                x = torch.empty(4, 128, device=device)
            if kind == "functional_fake":
                x = torch._to_functional_tensor(x)
        self.assertFalse(ov._cond(x, -1))
        self.assertFalse(ov._dtype_cond(x, -1))

    def test_cow_input_served_and_preserved(self, device):
        """Read copy-on-write inputs without materializing or mutating their storage."""
        base = torch.randn(128, 512, device=device)
        x = torch._lazy_clone(base)
        data_ptr = x.const_data_ptr()
        if not torch._C._is_cow_tensor(x):
            self.skipTest("allocator does not support COW sharing")
        self._check(lambda: torch.sum(x, dim=-1), **_TOL[torch.float32])
        self.assertTrue(torch._C._is_cow_tensor(x))
        self.assertEqual(x.const_data_ptr(), data_ptr)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs >= 2 GPUs")
    def test_other_device_is_served(self, device):
        """Serve tensors on a non-current device under that device's guard."""
        if torch.cuda.get_device_capability(1)[0] not in ov._ARCH_MAJORS:
            self.skipTest("the overrides do not serve cuda:1's architecture")
        x = torch.randn(128, 512, device="cuda:1")
        self._check(lambda: torch.sum(x, dim=-1), **_TOL[torch.float32])

    @parametrize("op", ["sum", "mean", "prod", "amax", "amin"])
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    def test_matches_aten(self, device, dtype, op):
        """Match ATen across input dtypes, multi-dimension keepdim, and full reductions;
        amax and amin exactly."""
        x = torch.rand(6, 33, 40, device=device, dtype=dtype) + 0.5
        exact = op in ("amax", "amin")
        tolerance = {"atol": 0, "rtol": 0} if exact else _TOL[dtype]
        fn = getattr(torch, op)
        calls = [lambda: fn(x, dim=1), lambda: fn(x, dim=1, keepdim=True)]
        if op != "prod":
            # amax and amin spell reduce-all as an empty dim list, not None.
            all_dims = () if op in ("amax", "amin") else None
            calls += [
                lambda: fn(x, dim=(0, 2), keepdim=True),
                lambda: fn(x),
                lambda: fn(x, dim=all_dims, keepdim=True),
            ]
        for call in calls:
            self._check(call, **tolerance)

    @parametrize("op", ["amax", "amin"])
    def test_nan_propagates(self, device, op):
        """Propagate NaN through dimension and full reductions like ATen."""
        x = torch.rand(128, 512, device=device)
        x[3, 7] = float("nan")
        x[100, 500] = float("nan")
        fn = getattr(torch, op)
        for call in (lambda: fn(x, dim=-1), lambda: fn(x, dim=0), lambda: fn(x)):
            self._check(call, atol=0, rtol=0)

    @parametrize("op", ["sum", "mean", "prod"])
    def test_out_dtype_contract(self, device, op):
        """Serve same-dtype and widening requests and leave input-casting requests to ATen."""
        fn = getattr(torch, op)
        x16 = torch.rand(64, 96, device=device, dtype=torch.float16) + 0.5
        xb = x16.bfloat16()
        x32 = x16.float()
        served = ((x16, torch.float32), (xb, torch.float32), (xb, torch.bfloat16))
        for x, out_dtype in served:
            call = functools.partial(fn, x, dim=1, dtype=out_dtype)
            self._check(call, **_TOL[out_dtype])
        # ATen casts the input to any other requested dtype before reducing.
        big = torch.tensor([[7e4, -7e4]] * 32, device=device)
        declined = (
            (x32, torch.float64),
            (big, torch.float16),
            (x16, torch.bfloat16),
            (xb, torch.float16),
            (x32, torch.bfloat16),
        )
        for x, out_dtype in declined:
            self._check(functools.partial(fn, x, dim=1, dtype=out_dtype), fired=0)

    def test_reduce_all_widening(self, device):
        """Serve reduce-all requests that widen fp16 and bf16 to fp32, with keepdim."""
        x16 = torch.rand(64, 96, device=device, dtype=torch.float16) + 0.5
        xb = x16.bfloat16()
        f32 = torch.float32
        calls = (
            lambda: torch.sum(x16, dtype=f32),
            lambda: torch.mean(xb, dim=None, dtype=f32),
            lambda: torch.sum(xb, dim=None, keepdim=True, dtype=f32),
            lambda: torch.mean(x16, dim=None, keepdim=True, dtype=f32),
        )
        for call in calls:
            self._check(call, **_TOL[f32])

    def test_inner_tree_env_orders_served_calls(self, device):
        """Serve calls under PYTORCH_NATIVE_INNER_TREE in the inner-tree order."""
        import cutlass

        from torch._native.ops.reductions import (
            kernel_general as kg,
            kernel_rowtile as rt,
            traits as T,
        )

        # Order-sensitive values (test_sum_cutedsl's), so the two orders' bits differ.
        values = torch.arange(128 * 512, device=device, dtype=torch.float64)
        x = (((values % 29) - 14) / 29 + ((values % 7) - 3) / 13).float().view(128, 512)
        with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}):
            got, fires = self._run(lambda: torch.sum(x, dim=-1))
        self.assertEqual(fires, 1)
        trait = T.SumOps(acc=cutlass.Float32)
        want = kg.reduce_dim(trait, "ref", x, [1], torch.float32, order="inner_tree")
        self.assertEqual(got, want, atol=0, rtol=0)
        with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "0"}):
            unordered, fires = self._run(lambda: torch.sum(x, dim=-1))
        self.assertEqual(fires, 1)
        self.assertFalse(torch.equal(unordered, got))

    def test_graph_capturable(self, device):
        """Keep core reduction overrides capture-safe and reading live input on replay."""
        x = torch.randn(8192, 1024, device=device)
        f = lambda: torch.sum(x, dim=-1)  # noqa: E731
        self.assertEqual(self._run(f)[1], 1)
        for _ in range(2):
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
        # Capture replays the launches the first call recorded.
        with torch.cuda.graph(g):
            out = f()
        for _ in range(2):
            with _disabled():
                ref = f()
            g.replay()
            self.assertEqual(out, ref, **_TOL[torch.float32])
            x.copy_(torch.randn_like(x))
        # A shape first seen inside capture records its launches there.
        from torch._native.ops.reductions import kernel_general as kg

        y = torch.randn(4096, 768, device=device)
        recorded = len(kg._REPLAYS)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = torch.sum(y, dim=-1)
        self.assertEqual(len(kg._REPLAYS), recorded + 1)
        y.copy_(torch.randn_like(y))
        g.replay()
        with _disabled():
            ref = torch.sum(y, dim=-1)
        self.assertEqual(out, ref, **_TOL[torch.float32])


instantiate_device_type_tests(TestCuTeDSLReductionWiring, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
