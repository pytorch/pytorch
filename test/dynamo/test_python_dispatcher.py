# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter, EagerAndRecordGraphs, normalize_gm
from torch.testing._internal.common_cuda import TEST_CUDA


class PythonDispatcherTests(torch._dynamo.test_case.TestCase):
    def test_dispatch_key1(self):
        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            x = x + 1
            return torch._C._dispatch_keys(x)

        x = torch.randn(2, 3)
        self.assertTrue(fn(x).raw_repr() == torch._C._dispatch_keys(x + 1).raw_repr())

    def test_dispatch_key2(self):
        from torch.testing._internal.two_tensor import TwoTensor

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            x = x.sin()
            return torch._C._dispatch_keys(x)

        x = torch.randn(3)
        y = torch.randn(3)
        z = TwoTensor(x, y)
        self.assertTrue(fn(z).raw_repr() == torch._C._dispatch_keys(z.sin()).raw_repr())

    def test_dispatch_key3(self):
        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            key_set = torch._C._dispatch_tls_local_include_set()
            return torch.sin(x + 1), key_set

        x = torch.randn(2, 3)
        self.assertEqual(fn(x)[0], torch.sin(x + 1))
        self.assertTrue(
            fn(x)[1].raw_repr() == torch._C._dispatch_tls_local_include_set().raw_repr()
        )

    def test_dispatch_key4(self):
        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(x):
            key_set = torch._C._dispatch_tls_local_include_set()
            key_set = key_set | torch._C._dispatch_keys(x)
            key_set = key_set - torch._C._dispatch_tls_local_exclude_set()
            if key_set.highestPriorityTypeId() == torch.DispatchKey.PythonDispatcher:
                return torch.sin(x + 1)
            else:
                return torch.sin(x - 1)

        x = torch.randn(2, 3)
        self.assertEqual(fn(x), torch.sin(x - 1))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 3]"):
        l_x_ = L_x_

        sub: "f32[2, 3]" = l_x_ - 1;  l_x_ = None
        sin: "f32[2, 3]" = torch.sin(sub);  sub = None
        return (sin,)
""",  # NOQA: B950
        )

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_dispatch_key_set_guard(self):
        counter = CompileCounter()

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x, dks):
            if dks.has("CPU"):
                return torch.sin(x + 1)
            else:
                return torch.sin(x - 1)

        x1 = torch.randn(2, 3)
        dks1 = torch._C._dispatch_keys(x1)
        self.assertEqual(fn(x1, dks1), torch.sin(x1 + 1))
        self.assertEqual(counter.frame_count, 1)

        x2 = torch.randn(2, 3)
        dks2 = torch._C._dispatch_keys(x2)
        self.assertEqual(fn(x2, dks2), torch.sin(x2 + 1))
        # No recompile since the dispatch key set is the same though the tensor is different.
        self.assertEqual(counter.frame_count, 1)

        x3 = torch.randn(2, 3, device="cuda")
        dks3 = torch._C._dispatch_keys(x3)
        self.assertEqual(fn(x3, dks3), torch.sin(x3 - 1))
        # Re-compile since the dispatch key set is different.
        self.assertEqual(counter.frame_count, 2)

    def test_functorch_interpreter(self):
        counter = CompileCounter()

        def square_and_add(x, y):
            interpreter = (
                torch._functorch.pyfunctorch.retrieve_current_functorch_interpreter()
            )
            level = interpreter.level()
            if interpreter.key() == torch._C._functorch.TransformType.Vmap:
                return (x**2 + y) * level
            else:
                return x**2 * level

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x, y):
            return torch.vmap(square_and_add)(x, y)

        x = torch.tensor([1, 2, 3, 4])
        y = torch.tensor([10, 20, 30, 40])
        self.assertEqual(fn(x, y), torch.tensor([11, 24, 39, 56]))
        self.assertEqual(counter.frame_count, 1)

        x = torch.tensor([1, 2, 3, 1])
        y = torch.tensor([10, 20, 30, 10])
        self.assertEqual(fn(x, y), torch.tensor([11, 24, 39, 11]))
        # No recompile
        self.assertEqual(counter.frame_count, 1)

    def test_vmapped_autograd_function(self):
        eager = EagerAndRecordGraphs()

        class Foo(torch.autograd.Function):
            generate_vmap_rule = True

            @staticmethod
            def forward(x):
                return x * 2

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def backward(ctx, grad):
                return grad * 2

        @torch.compile(backend=eager, fullgraph=True)
        def fn(x):
            return torch.vmap(Foo.apply)(x)

        x = torch.randn(2, 3, requires_grad=True)
        self.assertEqual(fn(x), torch.vmap(Foo.apply)(x))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

        a: "f32[3]" = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        _are_functorch_transforms_active = torch._C._are_functorch_transforms_active();  _are_functorch_transforms_active = None

        _are_functorch_transforms_active_1 = torch._C._are_functorch_transforms_active();  _are_functorch_transforms_active_1 = None

        child: "f32[3]" = torch._C._functorch.unwrap_if_dead(a);  a = None

        _unwrap_batched = torch._C._functorch._unwrap_batched(child, 1);  child = None
        getitem: "f32[2, 3]" = _unwrap_batched[0];  _unwrap_batched = None

        pop_dynamic_layer_stack = torch._C._functorch.pop_dynamic_layer_stack()

        _are_functorch_transforms_active_2 = torch._C._are_functorch_transforms_active();  _are_functorch_transforms_active_2 = None

        function_ctx = torch.autograd.function.FunctionCtx();  function_ctx = None
        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0
        autograd_function_apply = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, getitem, args_tensor_mask = [True], non_differentiable_idx = []);  fwd_body_0 = bwd_body_0 = getitem = None
        outputs: "f32[2, 3]" = autograd_function_apply[0];  autograd_function_apply = None

        push_dynamic_layer_stack = torch._C._functorch.push_dynamic_layer_stack(pop_dynamic_layer_stack);  pop_dynamic_layer_stack = push_dynamic_layer_stack = None

        result: "f32[3]" = torch._C._functorch._add_batch_dim(outputs, 0, 1);  outputs = None

        _remove_batch_dim: "f32[2, 3]" = torch._C._functorch._remove_batch_dim(result, 1, 2, 0);  result = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)

    class fwd_body_0(torch.nn.Module):
        def forward(self, function_ctx : torch.autograd.function.Function, getitem: "f32[2, 3]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
            _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

            _add_batch_dim: "f32[3]" = torch._C._functorch._add_batch_dim(getitem, 0, 1)

            batched_outputs: "f32[3]" = _add_batch_dim * 2;  _add_batch_dim = None

            _unwrap_batched = torch._C._functorch._unwrap_batched(batched_outputs, 1);  batched_outputs = None
            outputs: "f32[2, 3]" = _unwrap_batched[0]
            getitem_2 = _unwrap_batched[1];  _unwrap_batched = getitem_2 = None

            _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
            _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting_1 = None

            inp: "f32[3]" = torch._C._functorch._add_batch_dim(getitem, 0, 1);  getitem = inp = None
            _add_batch_dim_2: "f32[3]" = torch._C._functorch._add_batch_dim(outputs, 0, 1);  _add_batch_dim_2 = None

            _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None
            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((outputs, 0), [])

    class bwd_body_0(torch.nn.Module):
        def forward(self, function_ctx : torch.autograd.function.Function, outputs: "f32[2, 3]", const_unused : int):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
            _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

            _add_batch_dim: "f32[3]" = torch._C._functorch._add_batch_dim(outputs, 0, 1);  outputs = None

            batched_outputs: "f32[3]" = _add_batch_dim * 2;  _add_batch_dim = None

            _unwrap_batched = torch._C._functorch._unwrap_batched(batched_outputs, 1);  batched_outputs = None
            grad_ins: "f32[2, 3]" = _unwrap_batched[0]
            getitem_1 = _unwrap_batched[1];  _unwrap_batched = getitem_1 = None

            _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

            lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

            _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting_1 = None

            _add_batch_dim_1: "f32[3]" = torch._C._functorch._add_batch_dim(grad_ins, 0, 1);  grad_ins = None

            batched_outputs_1: "f32[3]" = _add_batch_dim_1.sum_to_size((3,));  _add_batch_dim_1 = None

            _remove_batch_dim: "f32[2, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs_1, 1, 2, 0);  batched_outputs_1 = None

            _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None
            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (_remove_batch_dim,)
""",  # NOQA: B950
        )

    def test_vmapped_autograd_function_fwd_and_bwd(self):
        class LinearFunction(torch.autograd.Function):
            generate_vmap_rule = True

            @staticmethod
            def forward(input, weight, bias):
                output = input.mm(weight.t())
                if bias is not None:
                    output += bias.unsqueeze(0).expand_as(output)
                return output

            @staticmethod
            def setup_context(ctx, inputs, output):
                input, weight, bias = inputs
                ctx.save_for_backward(input, weight, bias)

            @staticmethod
            def backward(ctx, grad_output):
                input, weight, bias = ctx.saved_tensors
                grad_input = grad_weight = grad_bias = None
                if ctx.needs_input_grad[0]:
                    grad_input = grad_output.mm(weight)
                if ctx.needs_input_grad[1]:
                    grad_weight = grad_output.t().mm(input)
                if bias is not None and ctx.needs_input_grad[2]:
                    grad_bias = grad_output.sum(0)

                return grad_input, grad_weight, grad_bias

        def fn(input, weight, bias=None):
            return torch.vmap(LinearFunction.apply)(input, weight, bias)

        input1 = torch.randn(4, 2, 2, dtype=torch.double, requires_grad=True)
        input2 = input1.clone().detach().requires_grad_(True)
        weight1 = torch.randn(4, 3, 2, dtype=torch.double, requires_grad=True)
        weight2 = weight1.clone().detach().requires_grad_(True)
        bias1 = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
        bias2 = bias1.clone().detach().requires_grad_(True)

        compiled_fn = torch.compile(backend="aot_eager", fullgraph=True)(fn)

        output1 = fn(input1, weight1, bias1)
        output1.sum().backward()

        output2 = compiled_fn(input2, weight2, bias2)
        output2.sum().backward()

        self.assertEqual(output1, output2)
        self.assertEqual(input1.grad, input2.grad)
        self.assertEqual(weight1.grad, weight2.grad)
        self.assertEqual(bias1.grad, bias2.grad)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
