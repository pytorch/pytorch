# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._export import export, dynamic_dim
from torch._export.utils import unlift_exported_program_lifted_states
from torch.testing._internal.common_utils import run_tests, TestCase

from functorch.experimental.control_flow import cond, map

@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestExportUtils(TestCase):
    def test_export_unlift(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self, x):
                return x.cos() + self.buffer.sin()

        ep = export(Foo(), (torch.ones(6, 4),), _add_runtime_assertions=False)
        gm = unlift_exported_program_lifted_states(ep)
        self.assertEqual(gm(torch.ones(6, 4)), Foo()(torch.ones(6, 4)))

    def test_export_unlift_container(self):
        class FooContainerInputOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self, x):
                return x[0][0].cos() + x[0][1].sin() + self.buffer.sin()

        inp = ((torch.ones(6, 4), torch.ones(6, 4)),)
        ep = export(FooContainerInputOutput(), (inp,), _add_runtime_assertions=False)
        gm = unlift_exported_program_lifted_states(ep)
        self.assertEqual(gm(inp), FooContainerInputOutput()(inp))

    def test_export_unlift_container_input(self):
        class FooContainerInputOutputV2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self, x, y):
                return x[0].cos() + y[0].sin() + self.buffer.sin()

        inp = ((torch.ones(6, 4),), (torch.ones(6, 4),))
        ep = export(FooContainerInputOutputV2(), inp, _add_runtime_assertions=False)
        gm = unlift_exported_program_lifted_states(ep)
        self.assertEqual(gm(*inp), FooContainerInputOutputV2()(*inp))

    def test_export_cond(self):
        class A(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self):
                return self.buffer.cos()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = A()

            def forward(self, x):
                def true_fn(x):
                    return x.cos() + self.a().sum()

                def false_fn(x):
                    return x.sin()

                return cond(x.shape[0] > 4, true_fn, false_fn, [x])

        inp = torch.ones(6, 4)
        constraints = [dynamic_dim(inp, 0)]
        ep = export(Foo(), (inp,), constraints=constraints, _add_runtime_assertions=False)
        gm = unlift_exported_program_lifted_states(ep)
        self.assertEqual(gm(torch.ones(6, 4)), Foo()(torch.ones(6, 4)))

    def test_export_cond_map(self):
        class A(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self):
                return self.buffer.sum()

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = A()

            def inner(self, x, pred):
                def true_fn(x):
                    return x + x + self.a()

                def false_fn(x):
                    return x * x - self.a()

                return cond(pred, true_fn, false_fn, [x])

            def forward(self, pred, xs):
                def body(x, pred):
                    return self.inner(x, pred) + self.a()

                return map(body, xs, pred)

        mod = Module()
        inp = torch.randn(3, 2, 1)
        constraints = [dynamic_dim(inp, 0)]
        ep = export(Module(), (torch.tensor(True), inp), constraints=constraints, _add_runtime_assertions=False)
        gm = unlift_exported_program_lifted_states(ep)

        inp_test = torch.randn(3, 2, 1)
        self.assertEqual(gm(torch.tensor(True), inp_test), Module()(torch.tensor(True), inp_test))
        self.assertEqual(gm(torch.tensor(False), inp_test), Module()(torch.tensor(False), inp_test))

if __name__ == '__main__':
    run_tests()
