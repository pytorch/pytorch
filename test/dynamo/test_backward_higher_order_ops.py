# Owner(s): ["module: dynamo"]

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch import _inductor as inductor
from torch._dynamo.compiled_autograd import _invoke_in_backward
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch.fx.experimental.proxy_tensor import make_fx


def _multiply(x):
    return x * x


def _multiply_invoke(grad):
    return _invoke_in_backward(grad, fn=_multiply, reenter=True)


def compiler_fn(gm):
    """Same as torch.compile() but counts number of compiles"""

    def inner_compiler(gm_, example_inputs_):
        counters["compiled_autograd"]["compiles"] += 1
        return inductor.compile(gm_, example_inputs_)

    return torch.compile(gm, backend=inner_compiler, fullgraph=True, dynamic=True)


class BackwardHigherOrderOpTests(torch._dynamo.test_case.TestCase):
    def test_invoke_in_eager(self):
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        y = torch.tensor([0.5, 0.5], requires_grad=True)

        def fn(x, y):
            x.register_hook(_multiply_invoke)
            return x * y

        out = fn(x, y)
        out.backward(torch.tensor([2.0, 2.0]))
        self.assertEqual(x.grad, 2 * x)

    def test_invoke_in_pt2(self):
        for backend in ["eager", "aot_eager", "inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                x.register_hook(_multiply_invoke)
                return x * y

            fn = torch._dynamo.optimize(backend)(fn)
            out = fn(x, y)
            out.backward(torch.tensor([2.0, 2.0]))
            self.assertEqual(x.grad, 2 * x)

    def test_invoke_make_fx(self):
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        out = make_fx(_multiply_invoke)(x)
        self.assertEqual(out(x), torch.tensor([0.25, 0.25]))
        actual = normalize_gm(out.print_readable(False))

        expected = """\
class _multiply_invoke(torch.nn.Module):
    def forward(self, grad_1: f32[2]):
        invocation: f32[2] = functools_dynamo_interceding_fn_wrapper(grad_1);  grad_1 = None
        empty_strided: f32[2] = torch.ops.aten.empty_strided.default([2], [1], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
        detach: f32[2] = torch.ops.aten.detach.default(invocation);  invocation = None
        detach_1: f32[2] = torch.ops.aten.detach.default(detach);  detach = None
        detach_2: f32[2] = torch.ops.aten.detach.default(detach_1);  detach_1 = None
        detach_3: f32[2] = torch.ops.aten.detach.default(detach_2);  detach_2 = None
        return detach_3
"""
        self.assertExpectedInline(actual, expected)
