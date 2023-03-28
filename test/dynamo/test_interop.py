# Owner(s): ["module: dynamo"]
import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch.onnx.operators
from torch._dynamo.testing import same


def fn(a, b):
    return a + b * 0.67


class InteropTests(torch._dynamo.test_case.TestCase):
    def _common(self, fn):
        inputs = [torch.randn(10), torch.randn(10)]
        ref = fn(*inputs)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(*inputs)
        self.assertTrue(same(ref, res))

    def test_fx_fn(self):
        fx_fn = torch.fx.symbolic_trace(fn)
        self._common(lambda a, b: fx_fn(a, b) + 1)

    def test_script_fn(self):
        script_fn = torch.jit.script(fn)
        self._common(lambda a, b: script_fn(a, b) + 1)

    def test_trace_fn(self):
        trace_fn = torch.jit.trace(fn, [torch.zeros(10), torch.zeros(10)])
        self._common(lambda a, b: trace_fn(a, b) + 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
