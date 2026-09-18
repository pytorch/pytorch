# Owner(s): ["module: autograd"]

import math

import torch
import torch.autograd.forward_ad as fwAD
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCPU,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


# Positive dQ/da at fixed x, from high-precision derivatives and tail quadrature.
_SHAPE_GRADIENT_REFERENCES = (
    (0.5, 0.25, 0.797947316783223),
    (1.0, 0.5, 0.48945757610237844),
    (2.0, 2.0, 0.29400469074623087),
    (4.0, 8.0, 0.04135992394465418),
    (1.0, 2.0, 0.2208254262118595),
    (2.0, 3.0, 0.19742541957920248),
    (20.0, 20.0, 0.08957920553912253),
    (100.0, 100.0, 0.03992749785778605),
    (10000.0, 10000.0, 0.003989456049453668),
    (1.0e6, 1.0e6, 0.0003989423136466252),
    (1.0e6, 963000.0, 6.19478434175564e-309),
    (5.0e-324, 1.0, 0.21938393439552029),
    (5e-324, 700.0, 1.406518766234033e-307),
    (2.2250738585072014e-308, 700.0, 1.406518766234033e-307),
    (1e-300, 700.0, 1.406518766234033e-307),
    (1e-100, 700.0, 1.406518766234033e-307),
    (1e-12, 700.0, 1.4065187662540891e-307),
    (100000.0, 88681.53608571063, 9.859676543780684e-305),
    (100000.0, 112241.75393242914, 9.859676543783857e-305),
    (100000.0, 112241.75393242916, 9.859676543768208e-305),
)


def _x_gradient(a, x):
    return math.exp((a - 1) * math.log(x) - x - math.lgamma(a))


class TestIGammaAutogradDevice(TestCase):
    @dtypes(torch.float64)
    @parametrize("op_name", ("igamma", "igammac"))
    def test_shape_gradient_references(self, device, dtype, op_name):
        op = getattr(torch, op_name)
        shape_sign = -1 if op_name == "igamma" else 1

        for a_value, x_value, dqda in _SHAPE_GRADIENT_REFERENCES:
            with self.subTest(a=a_value, x=x_value):
                a = torch.tensor(
                    a_value, device=device, dtype=dtype, requires_grad=True
                )
                x = torch.tensor(x_value, device=device, dtype=dtype)
                op(a, x).backward()
                expected_shape = shape_sign * dqda
                if abs(expected_shape) < math.exp(-700.0):
                    self.assertLessEqual(
                        abs(a.grad.item() - expected_shape),
                        4 * math.ulp(expected_shape),
                    )
                else:
                    self.assertEqual(a.grad, expected_shape, rtol=5e-13, atol=0)

    @dtypes(torch.float64)
    @parametrize(
        "function",
        (
            torch.igamma,
            torch.igammac,
            torch.special.gammainc,
            torch.special.gammaincc,
        ),
    )
    def test_function_and_special_alias_shape_gradient(self, device, dtype, function):
        a = torch.tensor(2.0, device=device, dtype=dtype, requires_grad=True)
        x = torch.tensor(3.0, device=device, dtype=dtype, requires_grad=True)
        function(a, x).backward()

        is_complement = function in (torch.igammac, torch.special.gammaincc)
        self.assertEqual(
            a.grad,
            (1 if is_complement else -1) * 0.19742541957920248,
            rtol=2e-12,
            atol=0,
        )
        self.assertEqual(
            x.grad,
            (-1 if is_complement else 1) * _x_gradient(2.0, 3.0),
            rtol=2e-12,
            atol=0,
        )

    @dtypes(torch.float64)
    @parametrize("method_name", ("igamma", "igammac"))
    def test_method_and_inplace_shape_gradient(self, device, dtype, method_name):
        shape_sign = -1 if method_name == "igamma" else 1
        x_sign = 1 if method_name == "igamma" else -1

        method_a = torch.tensor(1.0, device=device, dtype=dtype, requires_grad=True)
        method_x = torch.tensor(0.5, device=device, dtype=dtype, requires_grad=True)
        getattr(method_a, method_name)(method_x).backward()
        self.assertEqual(
            method_a.grad, shape_sign * 0.48945757610237844, rtol=2e-12, atol=0
        )
        self.assertEqual(
            method_x.grad, x_sign * _x_gradient(1.0, 0.5), rtol=2e-12, atol=0
        )

        base = torch.tensor(1.0, device=device, dtype=dtype, requires_grad=True)
        a = base + 0
        x = torch.tensor(0.5, device=device, dtype=dtype, requires_grad=True)
        result = getattr(a, method_name + "_")(x)
        result.backward()

        self.assertEqual(
            base.grad, shape_sign * 0.48945757610237844, rtol=2e-12, atol=0
        )
        self.assertEqual(x.grad, x_sign * _x_gradient(1.0, 0.5), rtol=2e-12, atol=0)

    @dtypes(torch.float64)
    @parametrize("op_name", ("igamma", "igammac"))
    def test_broadcasting_and_noncontiguous_shape_gradient(
        self, device, dtype, op_name
    ):
        op = getattr(torch, op_name)
        a_base = torch.tensor(
            [[1.0, 9.0], [2.0, 9.0]], device=device, dtype=dtype, requires_grad=True
        )
        x_base = torch.tensor(
            [[0.5, 2.0], [2.0, 3.0]], device=device, dtype=dtype, requires_grad=True
        )
        a = a_base[:, :1]
        x = x_base.transpose(0, 1)
        self.assertFalse(a.is_contiguous())
        self.assertFalse(x.is_contiguous())

        result = op(a, x)
        result.sum().backward()

        expanded_a = a.detach().expand_as(x).clone().requires_grad_()
        expanded_x = x.detach().clone().requires_grad_()
        op(expanded_a, expanded_x).sum().backward()
        self.assertEqual(a_base.grad[:, :1], expanded_a.grad.sum(dim=1, keepdim=True))
        self.assertEqual(a_base.grad[:, 1:], torch.zeros_like(a_base.grad[:, 1:]))
        expected_a = torch.tensor(
            [
                [0.48945757610237844 + 0.2208254262118595],
                [0.29400469074623087 + 0.19742541957920248],
            ],
            device=device,
            dtype=dtype,
        )
        self.assertEqual(
            a_base.grad[:, :1], (-1 if op_name == "igamma" else 1) * expected_a
        )
        self.assertEqual(x_base.grad.transpose(0, 1), expanded_x.grad)

    @dtypesIfCPU(torch.bfloat16, torch.float16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @parametrize("op_name", ("igamma", "igammac"))
    def test_supported_dtype_shape_gradient(self, device, dtype, op_name):
        op = getattr(torch, op_name)
        shape_sign = -1 if op_name == "igamma" else 1
        x_sign = 1 if op_name == "igamma" else -1
        a = torch.tensor(2.0, device=device, dtype=dtype, requires_grad=True)
        x = torch.tensor(2.0, device=device, dtype=dtype, requires_grad=True)
        op(a, x).backward()

        tolerance = {
            torch.float64: 2e-12,
            torch.float32: 1e-6,
            torch.float16: 1e-3,
            torch.bfloat16: 8e-3,
        }[dtype]
        self.assertEqual(
            a.grad, shape_sign * 0.29400469074623087, rtol=tolerance, atol=0
        )
        self.assertEqual(x.grad, x_sign * _x_gradient(2.0, 2.0), rtol=tolerance, atol=0)

    @dtypes(torch.float64)
    @parametrize("op_name", ("igamma", "igammac"))
    def test_scalar_empty_and_dtype_promotion(self, device, dtype, op_name):
        op = getattr(torch, op_name)
        a = torch.tensor(2.0, device=device, dtype=dtype, requires_grad=True)
        x = torch.tensor(3.0, device=device, dtype=dtype, requires_grad=True)
        self.assertEqual(op(a, x).shape, torch.Size([]))
        self.assertEqual(op(a, x).dtype, dtype)

        empty_a = torch.empty((0, 1), device=device, dtype=dtype, requires_grad=True)
        empty_x = torch.empty((1, 3), device=device, dtype=dtype, requires_grad=True)
        output = op(empty_a, empty_x)
        self.assertEqual(output.shape, (0, 3))
        output.sum().backward()
        self.assertEqual(empty_a.grad.shape, empty_a.shape)
        self.assertEqual(empty_x.grad.shape, empty_x.shape)
        self.assertEqual(empty_x.grad, torch.zeros_like(empty_x))

        mixed_a = torch.tensor(
            2.0, device=device, dtype=torch.float32, requires_grad=True
        )
        mixed_x = torch.tensor(3.0, device=device, dtype=dtype, requires_grad=True)
        promoted = op(mixed_a, mixed_x)
        self.assertEqual(promoted.dtype, torch.promote_types(torch.float32, dtype))
        promoted.backward()
        self.assertEqual(mixed_a.grad.dtype, torch.float32)
        self.assertEqual(mixed_x.grad.dtype, dtype)
        self.assertEqual(
            mixed_a.grad,
            (-1 if op_name == "igamma" else 1) * 0.19742541957920248,
            rtol=1e-6,
            atol=0,
        )
        self.assertEqual(
            mixed_x.grad,
            (1 if op_name == "igamma" else -1) * _x_gradient(2.0, 3.0),
            rtol=2e-12,
            atol=0,
        )

    @dtypes(torch.float64)
    @parametrize("op_name", ("igamma", "igammac"))
    @parametrize("inplace", (False, True))
    def test_shape_gradient_second_derivative_and_helper_are_not_implemented(
        self, device, dtype, op_name, inplace
    ):
        a = torch.tensor(2.0, device=device, dtype=dtype, requires_grad=True)
        x = torch.tensor(3.0, device=device, dtype=dtype, requires_grad=True)
        output = (
            getattr(a.clone(), op_name + "_")(x)
            if inplace
            else getattr(torch, op_name)(a, x)
        )
        da = torch.autograd.grad(output, a, create_graph=True)[0]
        with self.assertRaisesRegex(
            NotImplementedError, "derivative for .* is not implemented"
        ):
            torch.autograd.grad(da, a)

        for input_index in (0, 1):
            helper_a = a.detach().clone().requires_grad_()
            helper_x = x.detach().clone().requires_grad_()
            with self.subTest(input_index=input_index):
                with self.assertRaisesRegex(
                    NotImplementedError, "derivative for .* is not implemented"
                ):
                    torch.autograd.grad(
                        torch._igamma_grad_a(helper_a, helper_x),
                        (helper_a, helper_x)[input_index],
                    )

    @dtypes(torch.float64)
    @parametrize("op_name", ("igamma", "igammac"))
    def test_x_second_derivative_is_preserved(self, device, dtype, op_name):
        op = getattr(torch, op_name)
        x_sign = 1 if op_name == "igamma" else -1
        a = torch.tensor(2.0, device=device, dtype=dtype, requires_grad=True)
        x = torch.tensor(3.0, device=device, dtype=dtype, requires_grad=True)
        dx = torch.autograd.grad(op(a, x), x, create_graph=True)[0]
        dxx = torch.autograd.grad(dx, x)[0]
        expected = x_sign * _x_gradient(2.0, 3.0) * ((2.0 - 1) / 3.0 - 1)
        self.assertEqual(dxx, expected, rtol=2e-12, atol=0)

    @dtypes(torch.float64)
    @parametrize("op_name", ("igamma", "igammac"))
    def test_forward_ad_is_not_supported(self, device, dtype, op_name):
        a = torch.tensor(2.0, device=device, dtype=dtype)
        x = torch.tensor(3.0, device=device, dtype=dtype)
        with fwAD.dual_level():
            dual_a = fwAD.make_dual(a, torch.ones_like(a))
            dual_x = fwAD.make_dual(x, torch.ones_like(x))
            with self.assertRaisesRegex(
                NotImplementedError,
                r"Trying to use forward AD with .* that does not support it",
            ):
                getattr(torch, op_name)(dual_a, dual_x)

    @dtypes(torch.float64)
    @parametrize("op_name", ("igamma", "igammac"))
    def test_leaf_inplace_remains_an_error(self, device, dtype, op_name):
        a = torch.tensor(2.0, device=device, dtype=dtype, requires_grad=True)
        x = torch.tensor(3.0, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "leaf Variable.*in-place"):
            getattr(a, op_name + "_")(x)
        self.assertEqual(a, 2.0)


class TestIGammaBackendPreservation(TestCase):
    @dtypes(torch.float32)
    @parametrize("op_name", ("igamma", "igammac"))
    def test_other_only_backward(self, device, dtype, op_name):
        a = torch.tensor(2.0, device=device, dtype=dtype)
        x = torch.tensor(3.0, device=device, dtype=dtype, requires_grad=True)
        dx = torch.autograd.grad(getattr(torch, op_name)(a, x), x)[0]
        expected = (1 if op_name == "igamma" else -1) * _x_gradient(2.0, 3.0)
        self.assertEqual(dx, expected, rtol=1e-5, atol=0)

        if torch.device(device).type in ("mps", "xpu"):
            a.requires_grad_()
            with self.assertRaisesRegex(
                NotImplementedError, "derivative for .* is not implemented"
            ):
                torch.autograd.grad(getattr(torch, op_name)(a, x), a)

    @dtypes(torch.float32)
    @parametrize("op_name", ("igamma", "igammac"))
    def test_forward_zero_and_negative_domain(self, device, dtype, op_name):
        a = torch.tensor([0, 0, 1, -1, 1], device=device, dtype=dtype)
        x = torch.tensor([0, 1, 0, 1, -1], device=device, dtype=dtype)
        values = [math.nan, 1, 0, math.nan, math.nan]
        if op_name == "igammac":
            values = [math.nan, 0, 1, math.nan, math.nan]
        expected = torch.tensor(values, device=device, dtype=dtype)
        self.assertEqual(getattr(torch, op_name)(a, x), expected)


instantiate_device_type_tests(
    TestIGammaAutogradDevice, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(
    TestIGammaBackendPreservation,
    globals(),
    only_for=("cpu", "cuda", "mps", "xpu"),
    allow_mps=True,
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()
