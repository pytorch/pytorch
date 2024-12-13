# Owner(s): ["module: dynamo"]
# flake8: noqa

import functools

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd
from torch._dynamo._trace_wrapped_higher_order_op import trace_wrapped
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch.fx.experimental.proxy_tensor import make_fx


def _multiply(x):
    return x * x


def _multiply_invoke(grad):
    return trace_wrapped(grad, fn=_multiply)


class BackwardHigherOrderOpTests(torch._dynamo.test_case.TestCase):
    def test_invoke_in_eager(self):
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        y = torch.tensor([0.5, 0.5], requires_grad=True)

        def fn(x, y):
            x.register_hook(_multiply_invoke)
            return x * y

        out = fn(x, y)
        grad_out = torch.tensor([2.0, 2.0])
        out.backward(grad_out)
        self.assertEqual(x.grad, y * grad_out)

    def test_invoke_in_pt2(self):
        for backend in ["eager", "aot_eager", "inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                x.register_hook(_multiply_invoke)
                return x * y

            fn = torch.compile(fn, backend=backend)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            out.backward(grad_out)
            self.assertEqual(x.grad, grad_out * y)

    def test_invoke_make_fx_forward_contrived(self):
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        out = make_fx(_multiply_invoke)(x)
        self.assertEqual(out(x), torch.tensor([0.25, 0.25]))
        actual = normalize_gm(out.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class _multiply_invoke(torch.nn.Module):
    def forward(self, grad_1: "f32[2]"):
        trace_wrapped: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op_self_invoke(grad_1);  grad_1 = None
        return trace_wrapped
""",
        )

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

        self.assertExpectedInline(
            actual,
            """\
class _multiply_invoke(torch.nn.Module):
    def forward(self, grad_1: "f32[2]"):
        trace_wrapped: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op_self_invoke(grad_1);  grad_1 = None
        return trace_wrapped
""",
        )

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

            fn = torch.compile(fn, backend=backend)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with compiled_autograd._enable(compiler_fn):
                out.backward(grad_out)
            actual = normalize_gm(graph.print_readable(False))
            self.assertEqual(x.grad, grad_out * grad_out)
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list):
        l_inputs_ = L_inputs_

        getitem: "f32[s0]" = l_inputs_[0];  l_inputs_ = None

        new_grad: "f32[s0]" = torch.clone(getitem)

        result: "f32[s0]" = getitem * getitem;  getitem = None

        new_grad_1: "f32[s0]" = torch.clone(result);  result = None
        return (new_grad, new_grad_1)
""",
            )

            graph = None

    def test_invoke_in_pt2_compiled_autograd_side_effect(self):
        def _side_effect_stateful_fn2(x, obj):
            obj.counter = obj.counter + 1
            return _multiply(x)

        def _side_effectful_invoke2(grad, fn):
            return trace_wrapped(grad, fn=fn)

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

            class MyObj:
                def __init__(self) -> None:
                    self.counter = 0

            obj = MyObj()
            inner_fn = functools.partial(_side_effect_stateful_fn2, obj=obj)
            hook_fn = functools.partial(_side_effectful_invoke2, fn=inner_fn)
            x.register_hook(hook_fn)

            def fn(x, y):
                return x + y

            fn = torch.compile(fn, backend=backend, fullgraph=True)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with compiled_autograd._enable(compiler_fn):
                out.backward(grad_out)
            actual = normalize_gm(graph.print_readable(False))
            self.assertEqual(obj.counter, 1)
            self.assertEqual(x.grad, grad_out + grad_out)
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list, L_hooks_0_keywords_fn_keywords_obj_counter: "Sym(s1)"):
        l_inputs_ = L_inputs_
        l_hooks_0_keywords_fn_keywords_obj_counter = L_hooks_0_keywords_fn_keywords_obj_counter

        getitem: "f32[s0]" = l_inputs_[0];  l_inputs_ = None

        new_grad: "f32[s0]" = torch.clone(getitem)

        add: "Sym(s1 + 1)" = l_hooks_0_keywords_fn_keywords_obj_counter + 1;  l_hooks_0_keywords_fn_keywords_obj_counter = None

        result: "f32[s0]" = getitem * getitem;  getitem = None

        new_grad_1: "f32[s0]" = torch.clone(result);  result = None
        return (new_grad, new_grad_1, add)
""",
            )

            out = fn(x, y)
            out.backward(grad_out)
            self.assertEqual(obj.counter, 2)

            out = fn(x, y)
            out.backward(grad_out)
            self.assertEqual(obj.counter, 3)
            graph = None

    def test_invoke_in_pt2_compiled_autograd_graph_breaks(self):
        def _graph_breaking_fn(x):
            print("Boo!")
            return _multiply(x)

        def _graph_break_invoke(grad):
            return trace_wrapped(grad, fn=_graph_breaking_fn)

        def compiler_fn(gm):
            return torch.compile(gm, backend="inductor", fullgraph=True, dynamic=True)

        for backend in ["eager", "aot_eager", "inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                x.register_hook(_graph_break_invoke)
                return x + y

            fn = torch.compile(fn, backend=backend, fullgraph=True)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "print",
            ):
                with compiled_autograd._enable(compiler_fn):
                    out.backward(grad_out)

            graph = None


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
