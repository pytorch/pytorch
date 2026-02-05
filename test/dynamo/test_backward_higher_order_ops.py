# Owner(s): ["module: dynamo"]
# flake8: noqa: B950

import functools
import itertools
from unittest import mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd
from torch._dynamo._trace_wrapped_higher_order_op import trace_wrapped
from torch._dynamo.testing import normalize_gm
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

    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    def test_invoke_in_pt2_compiled_autograd(self, _):
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
            if backend == "aot_eager":
                self.assertExpectedInline(
                    actual,
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list, s69: "Sym(s21)", L_sizes_0_: "f32[0, s21]"):
        l_inputs_ = L_inputs_
        l_sizes_0_ = L_sizes_0_

        getitem: "f32[s21]" = l_inputs_[0]
        getitem_1: "f32[s21]" = l_inputs_[1]
        getitem_2: "f32[s21]" = l_inputs_[2];  l_inputs_ = None

        size: "Sym(s21)" = l_sizes_0_.size(1);  l_sizes_0_ = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [size], False, 6)]);  getitem = size = None
        getitem_9: "f32[s21]" = validate_outputs[0];  validate_outputs = None

        call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((), [], [], getitem_9);  getitem_9 = None
        aot1_tangents_1: "f32[s21]" = call_aot_bwd_prologue[0];  call_aot_bwd_prologue = None

        accumulate_grad = torch__dynamo_compiled_autograd_ops_AccumulateGrad([aot1_tangents_1], getitem_1, None, False);  getitem_1 = None
        getitem_11: "f32[s21]" = accumulate_grad[0];  accumulate_grad = None

        result: "f32[s21]" = aot1_tangents_1 * aot1_tangents_1;  aot1_tangents_1 = None

        accumulate_grad_1 = torch__dynamo_compiled_autograd_ops_AccumulateGrad([result], getitem_2, None, False);  result = getitem_2 = None
        getitem_12: "f32[s21]" = accumulate_grad_1[0];  accumulate_grad_1 = None
        return (getitem_11, getitem_12)
""",
                )
            elif backend == "inductor":
                self.assertExpectedInline(
                    actual,
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list, s69: "Sym(s21)", L_sizes_0_: "f32[0, s21]"):
        l_inputs_ = L_inputs_
        l_sizes_0_ = L_sizes_0_

        getitem: "f32[s21]" = l_inputs_[0]
        getitem_1: "f32[s21]" = l_inputs_[1]
        getitem_2: "f32[s21]" = l_inputs_[2];  l_inputs_ = None

        size: "Sym(s21)" = l_sizes_0_.size(1);  l_sizes_0_ = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [size], False, 6)]);  getitem = size = None
        getitem_9: "f32[s21]" = validate_outputs[0];  validate_outputs = None

        call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((), [], [], getitem_9);  getitem_9 = None
        aot3_tangents_1: "f32[s21]" = call_aot_bwd_prologue[0];  call_aot_bwd_prologue = None

        accumulate_grad = torch__dynamo_compiled_autograd_ops_AccumulateGrad([aot3_tangents_1], getitem_1, None, False);  getitem_1 = None
        getitem_11: "f32[s21]" = accumulate_grad[0];  accumulate_grad = None

        result: "f32[s21]" = aot3_tangents_1 * aot3_tangents_1;  aot3_tangents_1 = None

        accumulate_grad_1 = torch__dynamo_compiled_autograd_ops_AccumulateGrad([result], getitem_2, None, False);  result = getitem_2 = None
        getitem_12: "f32[s21]" = accumulate_grad_1[0];  accumulate_grad_1 = None
        return (getitem_11, getitem_12)
""",
                )

            graph = None

    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    def test_invoke_in_pt2_compiled_autograd_side_effect(self, _):
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

        for backend in ["inductor"]:
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
            if backend in ["aot_eager", "inductor"]:
                self.assertExpectedInline(
                    actual,
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list, s69: "Sym(s21)", L_sizes_0_: "f32[0, s21]", L_hooks_1_keywords_fn_keywords_obj_counter: "Sym(s45)"):
        l_inputs_ = L_inputs_
        l_sizes_0_ = L_sizes_0_
        l_hooks_1_keywords_fn_keywords_obj_counter = L_hooks_1_keywords_fn_keywords_obj_counter

        getitem: "f32[s21]" = l_inputs_[0]
        getitem_1: "f32[s21]" = l_inputs_[1]
        getitem_2: "f32[s21]" = l_inputs_[2];  l_inputs_ = None

        size: "Sym(s21)" = l_sizes_0_.size(1);  l_sizes_0_ = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [size], False, 6)]);  getitem = size = None
        getitem_9: "f32[s21]" = validate_outputs[0];  validate_outputs = None

        call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((), [], [], getitem_9);  getitem_9 = None
        aot0_tangents_1: "f32[s21]" = call_aot_bwd_prologue[0];  call_aot_bwd_prologue = None

        accumulate_grad = torch__dynamo_compiled_autograd_ops_AccumulateGrad([aot0_tangents_1], getitem_1, None, False);  getitem_1 = None
        getitem_11: "f32[s21]" = accumulate_grad[0];  accumulate_grad = None

        add: "Sym(s45 + 1)" = l_hooks_1_keywords_fn_keywords_obj_counter + 1;  l_hooks_1_keywords_fn_keywords_obj_counter = None

        result: "f32[s21]" = aot0_tangents_1 * aot0_tangents_1;  aot0_tangents_1 = None

        accumulate_grad_1 = torch__dynamo_compiled_autograd_ops_AccumulateGrad([result], getitem_2, None, False);  result = getitem_2 = None
        getitem_12: "f32[s21]" = accumulate_grad_1[0];  accumulate_grad_1 = None
        return (getitem_11, getitem_12, add)
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
