# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    from scipy import special as scipy_special

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestModifiedBesselFunctions(TestCase):
    def _skip_if_no_scipy(self):
        if not HAS_SCIPY:
            self.skipTest("scipy not available")

    def _tol(self, dtype):
        if dtype == torch.float32:
            return dict(rtol=1e-3, atol=1e-5)
        return dict(rtol=1e-5, atol=1e-8)

    def _scipy_tensor(self, scipy_fn, x, nu_val, device, dtype):
        return torch.as_tensor(
            scipy_fn(nu_val, x.cpu().numpy()),
            device=device,
            dtype=dtype,
        )

    def _assert_matches_scipy(
        self, torch_fn, scipy_fn, x, nu_val, device, dtype, finite_only=False
    ):
        nu = torch.full_like(x, nu_val)
        result = torch_fn(x, nu)
        expected = self._scipy_tensor(scipy_fn, x, nu_val, device, dtype)
        if finite_only:
            mask = torch.isfinite(expected)
            if not mask.any():
                return
            result = result[mask]
            expected = expected[mask]
        self.assertEqual(result, expected, **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_reference_values(self, device, dtype):
        self._skip_if_no_scipy()
        cases = [
            (
                torch.special.modified_bessel_i,
                scipy_special.iv,
                torch.linspace(0.1, 10, 50, device=device, dtype=dtype),
                list(range(11)) + [0.5, 1.5, 2.5, 3.5],
            ),
            (
                torch.special.modified_bessel_i,
                scipy_special.iv,
                torch.linspace(0.1, 15, 50, device=device, dtype=dtype),
                [2.73, 5.17, 12.73],
            ),
            (
                torch.special.modified_bessel_k,
                scipy_special.kv,
                torch.linspace(0.1, 10, 50, device=device, dtype=dtype),
                list(range(11)) + [0.5, 1.5, 2.5, 3.5],
            ),
            (
                torch.special.modified_bessel_k,
                scipy_special.kv,
                torch.linspace(0.1, 20, 100, device=device, dtype=dtype),
                [2.73, 5.17, 12.73],
            ),
        ]
        for torch_fn, scipy_fn, x, orders in cases:
            for nu_val in orders:
                self._assert_matches_scipy(
                    torch_fn, scipy_fn, x, float(nu_val), device, dtype
                )

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_edge_cases(self, device, dtype):
        x_zero = torch.tensor([0.0], device=device, dtype=dtype)
        self.assertEqual(
            torch.special.modified_bessel_i(
                x_zero, torch.tensor([0.0], device=device, dtype=dtype)
            ),
            torch.ones_like(x_zero),
        )
        for nu_val in [1.0, 2.5, 10.0]:
            nu = torch.tensor([nu_val], device=device, dtype=dtype)
            self.assertEqual(torch.special.modified_bessel_i(x_zero, nu), x_zero)

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_nan_inf(self, device, dtype):
        nu = torch.tensor([2.5], device=device, dtype=dtype)
        x_ok = torch.tensor([1.0], device=device, dtype=dtype)
        x_neg = torch.tensor([-1.0], device=device, dtype=dtype)
        x_zero = torch.tensor([0.0], device=device, dtype=dtype)
        self.assertTrue(
            torch.isnan(
                torch.special.modified_bessel_i(
                    torch.tensor([float("nan")], device=device, dtype=dtype), nu
                )
            ).all()
        )
        self.assertTrue(
            torch.isnan(
                torch.special.modified_bessel_i(
                    x_ok, torch.tensor([float("nan")], device=device, dtype=dtype)
                )
            ).all()
        )
        self.assertTrue(torch.isnan(torch.special.modified_bessel_i(x_neg, nu)).all())
        self.assertEqual(
            torch.special.modified_bessel_i(
                x_ok, torch.tensor([float("inf")], device=device, dtype=dtype)
            ),
            torch.zeros_like(x_ok),
        )
        eps = 1e-4 if dtype == torch.float32 else 1e-12
        for nu_val in [1.0 + eps, 2.0 - eps, -1.0 - eps]:
            nu_near_int = torch.tensor([nu_val], device=device, dtype=dtype)
            self.assertTrue(
                torch.isnan(torch.special.modified_bessel_i(x_neg, nu_near_int)).all()
            )
        self.assertEqual(
            torch.special.modified_bessel_i(
                x_zero, torch.tensor([1e-12], device=device, dtype=dtype)
            ),
            torch.zeros_like(x_zero),
        )
        for nu_val in [-1e-12, -1.0 - eps]:
            nu = torch.tensor([nu_val], device=device, dtype=dtype)
            self.assertTrue(
                torch.isinf(torch.special.modified_bessel_i(x_zero, nu)).all()
            )

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_negative_x_integer_order(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.tensor([-0.5, -1.0, -2.0, -5.0], device=device, dtype=dtype)
        for nu_val in [0, 1, 2, 3, 5, 10]:
            self._assert_matches_scipy(
                torch.special.modified_bessel_i,
                scipy_special.iv,
                x,
                float(nu_val),
                device,
                dtype,
            )

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_large_nu(self, device, dtype):
        self._skip_if_no_scipy()
        for nu_val, xs in [
            (201.0, [123.7, 250.0, 500.0]),
            (300.0, [123.7, 250.0, 500.0]),
            (767.0, [882.118]),
            (1360.0, [1024.56]),
            (2000.0, [1320.0, 1400.0, 1700.0]),
            (2001.0, [1320.66, 1500.0, 1800.0]),
            (2500.0, [1650.0, 2000.0]),
            (5000.0, [3300.0, 4000.0]),
        ]:
            self._assert_matches_scipy(
                torch.special.modified_bessel_k,
                scipy_special.kv,
                torch.tensor(xs, device=device, dtype=dtype),
                nu_val,
                device,
                dtype,
                finite_only=True,
            )

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_edge_cases(self, device, dtype):
        for nu_val in [0.0, 1.0, 2.5]:
            x = torch.tensor([0.0], device=device, dtype=dtype)
            nu = torch.tensor([nu_val], device=device, dtype=dtype)
            self.assertTrue(torch.isinf(torch.special.modified_bessel_k(x, nu)).all())

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_nan_inf(self, device, dtype):
        nu = torch.tensor([2.5], device=device, dtype=dtype)
        x_ok = torch.tensor([1.0], device=device, dtype=dtype)
        self.assertTrue(
            torch.isnan(
                torch.special.modified_bessel_k(
                    torch.tensor([float("nan")], device=device, dtype=dtype), nu
                )
            ).all()
        )
        self.assertTrue(
            torch.isnan(
                torch.special.modified_bessel_k(
                    x_ok, torch.tensor([float("nan")], device=device, dtype=dtype)
                )
            ).all()
        )
        x_neg = torch.tensor([-1.0], device=device, dtype=dtype)
        self.assertTrue(torch.isnan(torch.special.modified_bessel_k(x_neg, nu)).all())

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_symmetry(self, device, dtype):
        x = torch.linspace(0.1, 10, 50, device=device, dtype=dtype)
        nu_pos = torch.full_like(x, 2.73)
        nu_neg = torch.full_like(x, -2.73)
        tol = dict(rtol=1e-10, atol=1e-10)
        if dtype == torch.float32:
            tol = dict(rtol=1e-5, atol=1e-5)
        self.assertEqual(
            torch.special.modified_bessel_k(x, nu_pos),
            torch.special.modified_bessel_k(x, nu_neg),
            **tol,
        )

    @dtypes(torch.float64)
    def test_modified_bessel_near_integer_orders(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.tensor([1.0, 2.0, 5.0], device=device, dtype=dtype)
        near_integer_cases = [
            (base_nu, eps) for base_nu in [0.0, 1.0, 5.0] for eps in [1e-8, 1e-5, 1e-4]
        ]
        for torch_fn, scipy_fn in [
            (torch.special.modified_bessel_i, scipy_special.iv),
            (torch.special.modified_bessel_k, scipy_special.kv),
        ]:
            for base_nu, eps in near_integer_cases:
                for sign in [1, -1]:
                    nu_val = base_nu + sign * eps
                    nu = torch.full_like(x, nu_val)
                    expected = self._scipy_tensor(scipy_fn, x, nu_val, device, dtype)
                    self.assertEqual(
                        torch_fn(x, nu),
                        expected,
                        rtol=1e-5,
                        atol=1e-8,
                        msg=f"Failed for nu={nu_val}",
                    )

    @dtypes(torch.float64)
    def test_modified_bessel_continuity_at_integers(self, device, dtype):
        x = torch.tensor([2.0], device=device, dtype=dtype)
        for n in [0, 1, 5]:
            nu_exact = torch.tensor([float(n)], device=device, dtype=dtype)
            exact = [
                torch.special.modified_bessel_i(x, nu_exact),
                torch.special.modified_bessel_k(x, nu_exact),
            ]
            for eps in [1e-6, 1e-4]:
                for sign in [1, -1]:
                    nu_near = torch.tensor([n + sign * eps], device=device, dtype=dtype)
                    near = [
                        torch.special.modified_bessel_i(x, nu_near),
                        torch.special.modified_bessel_k(x, nu_near),
                    ]
                    for actual, expected in zip(near, exact):
                        rel_err = torch.abs(actual - expected) / torch.abs(expected)
                        self.assertTrue(rel_err.item() < 10 * eps)

    @dtypes(torch.float64)
    def test_modified_bessel_gradient(self, device, dtype):
        x = torch.tensor(
            [1.0, 2.0, 5.0], device=device, dtype=dtype, requires_grad=True
        )
        nu = torch.tensor([2.5, 2.5, 2.5], device=device, dtype=dtype)
        for torch_fn, scale in [
            (torch.special.modified_bessel_i, 0.5),
            (torch.special.modified_bessel_k, -0.5),
        ]:
            x.grad = None
            torch_fn(x, nu).sum().backward()
            with torch.no_grad():
                expected_grad = scale * (torch_fn(x, nu - 1) + torch_fn(x, nu + 1))
            self.assertEqual(x.grad, expected_grad, rtol=1e-4, atol=1e-6)

    @dtypes(torch.float64)
    def test_modified_bessel_gradcheck(self, device, dtype):
        from torch.autograd import gradcheck

        test_cases = [
            ([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0], 2.5),
            ([1.0, 3.0, 8.0], 0.5),
            ([0.5, 2.0, 7.0], 5.17),
            ([1.0, 5.0, 15.0], 0.01),
        ]
        for torch_fn in [
            torch.special.modified_bessel_i,
            torch.special.modified_bessel_k,
        ]:
            for x_vals, nu_val in test_cases:
                x = torch.tensor(x_vals, device=device, dtype=dtype, requires_grad=True)
                nu = torch.full_like(x, nu_val).detach()
                self.assertTrue(
                    gradcheck(lambda x: torch_fn(x, nu), (x,), eps=1e-6, atol=1e-4),
                    msg=f"gradcheck failed for nu={nu_val}",
                )

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_large_x(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.tensor([50.0, 100.0, 200.0], device=device, dtype=dtype)
        for torch_fn, scipy_fn in [
            (torch.special.modified_bessel_i, scipy_special.iv),
            (torch.special.modified_bessel_k, scipy_special.kv),
        ]:
            for nu_val in [0.5, 2.5, 5.0]:
                self._assert_matches_scipy(torch_fn, scipy_fn, x, nu_val, device, dtype)

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_broadcasting(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.tensor([[1.0], [2.0], [5.0]], device=device, dtype=dtype)
        nu = torch.tensor([[0.5, 1.5, 2.5, 3.5]], device=device, dtype=dtype)
        result = torch.special.modified_bessel_i(x, nu)
        self.assertEqual(result.shape, (3, 4))
        expected = torch.as_tensor(
            scipy_special.iv(nu.cpu().numpy(), x.cpu().numpy()),
            device=device,
            dtype=dtype,
        )
        self.assertEqual(result, expected, **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_out_parameter(self, device, dtype):
        x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        nu = torch.full_like(x, 2.5)
        out = torch.empty_like(x)
        ret = torch.special.modified_bessel_k(x, nu, out=out)
        self.assertTrue(ret.data_ptr() == out.data_ptr())
        self.assertEqual(
            out, torch.special.modified_bessel_k(x, nu), **self._tol(dtype)
        )

    def test_modified_bessel_int_to_float_promotion(self, device):
        x = torch.tensor([1, 2, 3], device=device, dtype=torch.int64)
        nu = torch.tensor([1, 2, 3], device=device, dtype=torch.int64)
        self.assertTrue(torch.special.modified_bessel_i(x, nu).is_floating_point())

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_cpu_cuda_parity(self, device, dtype):
        if device == "cpu" or not torch.cuda.is_available():
            self.skipTest("requires CUDA for cross-device comparison")
        x_cpu = torch.tensor(
            [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0, 1000.0],
            dtype=dtype,
        )
        x_cuda = x_cpu.to(device)

        for nu_val in [0.5, 2.5, 12.73, 50.0, 200.0, 2001.0, 5000.0]:
            nu_cpu = torch.full_like(x_cpu, nu_val)
            nu_cuda = nu_cpu.to(device)
            for fn_name in ("modified_bessel_i", "modified_bessel_k"):
                fn = getattr(torch.special, fn_name)
                out_cpu = fn(x_cpu, nu_cpu)
                out_cuda = fn(x_cuda, nu_cuda).cpu()
                mask = (
                    torch.isfinite(out_cpu)
                    & torch.isfinite(out_cuda)
                    & (out_cpu.abs() > 1e-300)
                )
                if not mask.any():
                    continue
                rel_err = (
                    ((out_cpu[mask] - out_cuda[mask]).abs() / out_cpu[mask].abs())
                    .max()
                    .item()
                )
                tol = 1e-5 if dtype == torch.float32 else 1e-12
                self.assertLess(
                    rel_err,
                    tol,
                    msg=f"{fn_name} nu={nu_val} dtype={dtype}: "
                    f"CPU/CUDA rel_err={rel_err:.2e} exceeds {tol:.0e}",
                )

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_float_boundaries(self, device, dtype):
        self._skip_if_no_scipy()
        if dtype == torch.float32:
            test_pairs = [(0.01, 30.0), (0.001, 25.0), (0.5, 100.0)]
        else:
            test_pairs = [(0.001, 200.0), (1e-5, 100.0), (0.5, 700.0), (1000.0, 2001.0)]

        for x_val, nu_val in test_pairs:
            x = torch.tensor([x_val], device=device, dtype=dtype)
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_i(x, nu).item()
            ref = float(scipy_special.iv(nu_val, x_val))
            if abs(ref) < 1e-300:
                self.assertLess(abs(result), 1e-250)
            else:
                self.assertLess(abs(result - ref) / abs(ref), self._tol(dtype)["rtol"])


instantiate_device_type_tests(
    TestModifiedBesselFunctions, globals(), only_for=("cpu", "cuda")
)


if __name__ == "__main__":
    run_tests()
