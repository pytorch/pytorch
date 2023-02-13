# Owner(s): ["module: dynamo"]
import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch.onnx.operators
from torch._dynamo.testing import (
    same,
)


class MyModule(torch.nn.Module):
    def __init__(self, z):
        super().__init__()
        self.z = z

    def forward(self, x, y):
        return x * x * x + y + self.z


class InteropTests(torch._dynamo.test_case.TestCase):
    def test_fx_inline(self):
        fx_mod = torch.fx.symbolic_trace(MyModule(15))

        @torch.fx.symbolic_trace
        def fx_fn(a, b):
            return a + b * 0.67

        def fn(x, y):
            return fx_mod(x, y) + fx_fn(x, y) + 1

        inputs = [torch.randn(10), torch.randn(10)]
        ref = fn(*inputs)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(*inputs)
        self.assertTrue(same(ref, res))

    def test_ts_inline(self):
        ts_mod = torch.jit.script(MyModule(15))

        @torch.jit.script
        def ts_fn(a, b):
            return a + b * 0.67

        def fn(x, y):
            return ts_mod(x, y) + ts_fn(x, y) + 1

        inputs = [torch.randn(10), torch.randn(10)]
        ref = fn(*inputs)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(*inputs)
        self.assertTrue(same(ref, res))

    def test_trace_inline(self):
        ex = [torch.zeros(10), torch.zeros(10)]
        ts_mod = torch.jit.trace(MyModule(15), ex)

        def ts_fn(a, b):
            return a + b * 0.67

        ts_fn = torch.jit.trace(ts_fn, ex)

        def fn(x, y):
            return ts_mod(x, y) + ts_fn(x, y) + 1

        inputs = [torch.randn(10), torch.randn(10)]
        ref = fn(*inputs)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(*inputs)
        self.assertTrue(same(ref, res))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
