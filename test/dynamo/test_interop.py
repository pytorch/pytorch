# Owner(s): ["module: dynamo"]
import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch.onnx.operators
from torch._dynamo.testing import same


class MyModule(torch.nn.Module):
    def __init__(self, z):
        super().__init__()
        self.z = z

    def forward(self, x, y):
        return x * x + y + self.z


mod = MyModule(15)

fx_mod = torch.fx.symbolic_trace(mod)
script_mod = torch.jit.script(mod)
trace_mod = torch.jit.trace(mod, [torch.zeros(10), torch.zeros(10)])


def fn(a, b):
    return a + b * 0.67


fx_fn = torch.fx.symbolic_trace(fn)
script_fn = torch.jit.script(fn)
trace_fn = torch.jit.trace(fn, [torch.zeros(10), torch.zeros(10)])


class InteropTests(torch._dynamo.test_case.TestCase):
    def _common(self, fn):
        inputs = [torch.randn(10), torch.randn(10)]
        ref = fn(*inputs)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(*inputs)
        self.assertTrue(same(ref, res))

    def test_fx_fn(self):
        self._common(lambda a, b: fx_fn(a, b) + 1)

    def test_script_fn(self):
        self._common(lambda a, b: script_fn(a, b) + 1)

    def test_trace_fn(self):
        self._common(lambda a, b: trace_fn(a, b) + 1)

    def test_fx_mod(self):
        self._common(lambda a, b: fx_mod(a, b) + 1)

    def test_script_mod(self):
        self._common(lambda a, b: script_mod(a, b) + 1)

    def test_trace_mod(self):
        self._common(lambda a, b: trace_mod(a, b) + 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
