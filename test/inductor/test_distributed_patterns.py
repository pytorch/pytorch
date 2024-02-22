# Owner(s): ["oncall: pt2"]
import dataclasses
import functools

import torch
from torch._dynamo import compiled_autograd
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.inductor_utils import HAS_CPU


class DistributedPatternTests(TestCase):
    def test_intermediate_hook_with_closure(self):
        @dataclasses.dataclass
        class CustomObj:
            val: torch.Tensor

        def fn(x, obj):
            y = x.sin()
            closure_var = y + 1
            y.register_hook(lambda grad: grad + obj.val + closure_var)
            z = y.sin()
            return z

        opt = torch.compile(fn, fullgraph=True)

        obj1 = CustomObj(torch.tensor(88))
        obj2 = CustomObj(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd.enable(functools.partial(torch.compile, fullgraph=True)):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    @torch.no_grad()
    def test_storage_resize_zero(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            y = torch.sin(x)
            x.untyped_storage().resize_(0)
            return torch.cos(y)

        x = torch.randn(10)
        expected = torch.cos(torch.sin(x))
        y = fn(x)
        self.assertEqual(y, expected)
        self.assertEqual(x.untyped_storage().size(), 0)

    @torch.no_grad()
    def test_storage_resize_nonzero(self):
        @torch.compile(fullgraph=True)
        def fn(x, out):
            y = torch.sin(x)
            assert out.untyped_storage().size() == 0
            out.untyped_storage().resize_(x.untyped_storage().size())
            out.copy_(y.cos())

        x = torch.randn(10)
        out = torch.randn(10)
        expected = torch.cos(torch.sin(x))
        out.untyped_storage().resize_(0)
        fn(x, out)
        self.assertEqual(out.untyped_storage().size(), x.untyped_storage().size())
        self.assertEqual(out, expected)


if __name__ == "__main__":
    if HAS_CPU and not IS_MACOS:
        run_tests(needs="filelock")
