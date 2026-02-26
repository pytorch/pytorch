# Owner(s): ["module: autograd"]

import torch
from torch.testing._internal.common_utils import gradcheck, run_tests, TestCase


class TestAutogradComplex(TestCase):
    def test_view_func_for_complex_views(self):
        # case 1: both parent and child have view_func
        x = torch.randn(2, 2, 2, dtype=torch.double, requires_grad=True)
        y = x.detach().requires_grad_(True)

        x0 = x.clone()
        x1 = torch.view_as_complex(x0)
        x2 = torch.view_as_real(x1)
        x2.mul_(2)
        x2.sum().abs().backward()

        y0 = y.clone()
        y0.mul_(2)
        y0.sum().abs().backward()

        self.assertEqual(x.grad, y.grad)

        # case 2: parent has view_func but child does not
        x = torch.randn(2, 2, 2, dtype=torch.double, requires_grad=True)
        y = x.detach().requires_grad_(True)

        def fn(a):
            b = a.clone()
            b1 = torch.view_as_complex(b)
            b2 = b1.reshape(b1.numel())
            return b2

        x0 = fn(x)
        x0.mul_(2)
        x0.sum().abs().backward()

        y0 = fn(y)
        y1 = y0.mul(2)
        y1.sum().abs().backward()

        self.assertEqual(x.grad, y.grad)

        # case 3: parent does not have a view_func but child does
        x = torch.randn(10, dtype=torch.cdouble, requires_grad=True)
        y = x.detach().requires_grad_(True)

        def fn(a, dim0_size=5):
            b = a.clone()
            b1 = b.reshape(dim0_size, 2)
            b2 = torch.view_as_real(b1)
            return b2

        x0 = fn(x)
        x0.mul_(2)
        x0.sum().abs().backward()

        y0 = fn(y)
        y1 = y0.mul(2)
        y1.sum().abs().backward()

        self.assertEqual(x.grad, y.grad)

    def test_view_with_multi_output(self):
        x = torch.randn(2, 2, 2, dtype=torch.double)

        x1 = torch.view_as_complex(x)
        # Taking an invalid view should always be allowed as long as it is not
        # modified inplace
        res = x1.unbind(0)

        with self.assertRaisesRegex(
            RuntimeError, "output of a function that returns multiple views"
        ):
            res[0] += torch.rand(2, requires_grad=True)

        x.requires_grad_(True)
        x1 = torch.view_as_complex(x)
        # Taking an invalid view should always be allowed as long as it is not
        # modified inplace
        res = x1.unbind(0)

        with self.assertRaisesRegex(
            RuntimeError, "output of a function that returns multiple views"
        ):
            res[0] += torch.rand(2, requires_grad=True)

    def as_identity(self):
        # view_as_real and view_as_complex behavior should be like an identity
        def func(z):
            z_ = torch.view_as_complex(z)
            z_select = torch.select(z_, z_.dim() - 1, 0)
            z_select_real = torch.view_as_real(z_select)
            return z_select_real.sum()

        z = torch.randn(10, 2, 2, dtype=torch.double, requires_grad=True)
        gradcheck(func, [z])
        func(z).backward()

        z1 = z.detach().clone().requires_grad_(True)
        torch.select(z1, z1.dim() - 2, 0).sum().backward()

        self.assertEqual(z.grad, z1.grad)


if __name__ == "__main__":
    run_tests()
