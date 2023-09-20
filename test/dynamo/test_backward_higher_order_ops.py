# Owner(s): ["module: dynamo"]
# flake8: noqa

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd
from torch._dynamo._trace_wrapped_higher_order_op import _trace_wrapped
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch.fx.experimental.proxy_tensor import make_fx


def _multiply(x):
    return x * x


def _multiply_invoke(grad):
    return _trace_wrapped(grad, fn=_multiply)


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

    def test_invoke_make_fx_forward_contrived(self):
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        out = make_fx(_multiply_invoke)(x)
        self.assertEqual(out(x), torch.tensor([0.25, 0.25]))
        actual = normalize_gm(out.print_readable(False))

        expected = """\
class _multiply_invoke(torch.nn.Module):
    def forward(self, grad_1: f32[2]):
        invocation: f32[2] = functools_self_invoke(grad_1);  grad_1 = None
        assert_1: f32[2] = torch._functional_assert_tensor_metadata(invocation, (2,), (1,), torch.float32);  invocation = None
        detach: f32[2] = torch.ops.aten.detach.default(assert_1);  assert_1 = None
        detach_1: f32[2] = torch.ops.aten.detach.default(detach);  detach = None
        detach_2: f32[2] = torch.ops.aten.detach.default(detach_1);  detach_1 = None
        detach_3: f32[2] = torch.ops.aten.detach.default(detach_2);  detach_2 = None
        return detach_3
"""
        self.assertExpectedInline(actual, expected)

    def test_invoke_make_bw(self):
        x = torch.tensor([0.5, 0.5], requires_grad=True)

        def fwd(x):
            z = x * x
            return z + z

        res = fwd(x)
        res.backward(torch.tensor([1.0, 1.0]))
        out = make_fx(_multiply_invoke)(x.grad)
        self.assertEqual(out(x.grad), torch.tensor([4.0, 4.0]))
        actual = normalize_gm(out.print_readable(False))

        expected = """\
class _multiply_invoke(torch.nn.Module):
    def forward(self, grad_1: f32[2]):
        invocation: f32[2] = functools_self_invoke(grad_1);  grad_1 = None
        assert_1: f32[2] = torch._functional_assert_tensor_metadata(invocation, (2,), (1,), torch.float32);  invocation = None
        return assert_1
"""
        self.assertExpectedInline(actual, expected)

    def test_invoke_in_pt2_compiled_autograd(self):
        graph = None

        def compiler_fn(gm):
            def inner_compiler(gm_, example_inputs_):
                nonlocal graph
                self.assertEqual(graph, None)
                graph = gm_
                return inductor.compile(gm_, example_inputs_)

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=True, dynamic=True
            )

        for backend in ["eager", "aot_eager", "inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                x.register_hook(_multiply_invoke)
                return x + y

            fn = torch._dynamo.optimize(backend)(fn)
            out = fn(x, y)
            with compiled_autograd.enable(compiler_fn):
                out.backward(torch.tensor([2.0, 2.0]))
            actual = normalize_gm(graph.print_readable(False))
            self.assertEqual(x.grad, torch.tensor([4.0, 4.0]))
            expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_0_ : torch.Tensor):
        getitem = L_inputs_0_

        new_empty_strided = torch.ops.aten.new_empty_strided.default(getitem, [2], [1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))

        copy_ = torch.ops.aten.copy_.default(new_empty_strided, getitem);  new_empty_strided = None

        call_hook = getitem * getitem;  getitem = None

        new_empty_strided_1 = torch.ops.aten.new_empty_strided.default(call_hook, [2], [1], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))

        copy__1 = torch.ops.aten.copy_.default(new_empty_strided_1, call_hook);  new_empty_strided_1 = call_hook = None
        return (copy_, copy__1)
"""
            self.assertExpectedInline(actual, expected)

            graph = None
