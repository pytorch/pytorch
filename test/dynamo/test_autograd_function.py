# Owner(s): ["module: dynamo"]
# flake8: noqa: B950
import copy
import math
from dataclasses import dataclass

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch.testing._internal.triton_utils import HAS_CUDA, requires_cuda


if HAS_CUDA:
    import triton

    from torch.testing._internal.triton_utils import add_kernel


class CustomFunc1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return foo + foo

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class CustomFunc3(torch.autograd.Function):
    # Test there is graph break in forward function
    @staticmethod
    def forward(ctx, foo):
        result = foo + foo
        torch._dynamo.graph_break()
        result = result + foo
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())


class Module1(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc1().apply(foo)


class Module2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fn = CustomFunc1.apply

    def forward(self, foo):
        return self.fn(foo)


class Module3(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc1().apply(foo)


class Module4(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fn = CustomFunc1.apply

    def forward(self, foo):
        return self.fn(foo)


class Module5(torch.nn.Module):
    def forward(self, foo):
        return CustomFunc3().apply(foo)


class Module6(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fn = CustomFunc3.apply

    def forward(self, foo):
        return self.fn(foo)


class LinearFunction(torch.autograd.Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
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


class ModuleLinear(torch.nn.Module):
    def forward(self, input, weight, bias=None):
        return LinearFunction.apply(input, weight, bias)


class MaterializingGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x.clone(), x.clone()

    @staticmethod
    def backward(ctx, grad_out1, grad_out2):
        return grad_out1, grad_out2


class MaterializingGradModule(torch.nn.Module):
    def forward(self, x):
        return MaterializingGradFunction.apply(x)


class CustomFuncBwdPrintGraphBreak(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return torch.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        print("graph break!")
        return grad_output


class CustomFuncBwdPrintModule(torch.nn.Module):
    def forward(self, x):
        return CustomFuncBwdPrintGraphBreak.apply(x)


class CustomFuncStrideBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        return torch.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * grad_output.stride()[-1]


class CustomFuncStrideModule(torch.nn.Module):
    def forward(self, x):
        return CustomFuncStrideBwd.apply(x)


class CustomFuncSaveForBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        result = foo + foo
        result = result + foo
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())


class SaveForBwdModule(torch.nn.Module):
    def forward(self, foo):
        return CustomFuncSaveForBwd().apply(foo)


class ContextSaveAndMark(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            ctx.save_for_backward(x)
            ctx.mark_non_differentiable(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ContextMarkAndSave(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            ctx.mark_non_differentiable(x)
            ctx.save_for_backward(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ModuleWithGradFunc(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.f = func.apply

    def forward(self, x):
        return self.f(x)


class AutogradFunctionTests(torch._dynamo.test_case.TestCase):
    # Sound behaviors, tested for working capture
    def test_autograd_function_equivalence(self):
        for grad in [True, False]:
            for i in range(1, 5):
                torch._dynamo.reset()
                model = globals()[f"Module{i}"]()
                opt_model = torch.compile(model, backend="eager")
                self.assertTrue(
                    torch.allclose(
                        opt_model(torch.ones(2, 3, requires_grad=grad)),
                        torch.tensor([2.0], requires_grad=grad),
                    )
                )

    def test_autograd_function_has_graph_break(self):
        for grad in [True, False]:
            x = torch.randn(10, requires_grad=grad)
            for model in [Module5(), Module6()]:
                torch._dynamo.reset()
                cnts = torch._dynamo.testing.CompileCounter()
                opt_model = torch.compile(model, backend=cnts)
                for _ in range(3):
                    ref = model(x)
                    res = opt_model(x)
                    self.assertTrue(torch.allclose(ref, res))
                self.assertEqual(cnts.frame_count, 2)

    def test_linear_setup_context(self):
        model = ModuleLinear()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        input = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        weight = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
        eager_result = model(input, weight)
        optim_result = opt_model(input, weight)
        self.assertEqual(optim_result, eager_result)

    def test_materialize_grad(self):
        model = MaterializingGradModule()
        opt_model = torch.compile(model, backend="eager")
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        optim_result = opt_model(x)
        eager_result = model(x)
        self.assertEqual(optim_result, eager_result)

    def test_print_in_bwd(self):
        model = CustomFuncBwdPrintModule()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Dynamo does not know how to trace builtin operator `print`",
        ):
            opt_model(x)

    def test_stride_in_bwd(self):
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()
        model = CustomFuncStrideModule()
        opt_model = torch.compile(backend=cnt, fullgraph=True)(model)
        x1 = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        x2 = copy.deepcopy(x1)
        ref = model(x1)
        ref.backward(x1.clone().detach())
        res = opt_model(x2)
        res.backward(x2.clone().detach())

        self.assertEqual(ref, res)
        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(cnt.frame_count, 1)

    def test_enum_arg(self):
        from enum import Enum

        class SomeEnum(Enum):
            A = 0
            B = 1

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, e):
                if e is SomeEnum.A:
                    return x.sin()
                else:
                    return x.cos()

            @staticmethod
            def backward(ctx, g):
                return g

        @torch.compile(backend="eager", fullgraph=True)
        def f(x, enum):
            output = Foo.apply(
                x,
                enum,
            )
            return output

        x = torch.tensor([[1.0, 2, 3], [4, 5, 6]], requires_grad=True)
        y = f(x, SomeEnum.A)
        self.assertEqual(y, x.sin())

    def test_save_for_bwd(self):
        model = SaveForBwdModule()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        opt_model(x)

    def test_allow_in_graph(self):
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.allow_in_graph
        class AllowInGraphFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                torch._dynamo.graph_break()
                ctx.x0 = x.size(0)
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * ctx.x0

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return AllowInGraphFunc.apply(x)

        x = torch.rand(2, 3, requires_grad=True)
        result = fn(x)

        self.assertEqual(result, AllowInGraphFunc.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_once_differentiable(self):
        from torch.autograd.function import once_differentiable

        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        class ScaleGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            @once_differentiable
            def backward(ctx, grad):
                return grad * 0.5

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return ScaleGradient.apply(x)

        x = torch.randn(3, requires_grad=True)
        result = fn(x)

        self.assertEqual(result, ScaleGradient.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_classmethod(self):
        class Shake(torch.autograd.Function):
            @classmethod
            def forward(cls, ctx, foo):
                return foo + foo

            @classmethod
            def backward(cls, ctx, grad_output):
                return grad_output

        def f(x):
            return Shake.apply(x)

        x = torch.randn(4, 4, 4, 4, requires_grad=True)
        opt_m = torch.compile(backend="eager")(f)
        opt_m(x)

    def test_function_context_save_and_mark(self):
        mod = ModuleWithGradFunc(ContextSaveAndMark)
        args, kwargs = ([torch.rand([1])], {})
        before = mod(*args, **kwargs)

        torch._dynamo.reset()
        compiled_model = torch.compile(mod, backend="eager")
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_function_context_mark_and_save(self):
        mod = ModuleWithGradFunc(ContextMarkAndSave)
        args, kwargs = ([torch.rand([1])], {})
        before = mod(*args, **kwargs)

        torch._dynamo.reset()
        compiled_model = torch.compile(mod, backend="eager")
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_multi_output(self):
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone(), x.clone()

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 + grad2

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return Foo.apply(x)

        x = torch.randn(3, requires_grad=True)
        result = f(x)

        self.assertEqual(result, Foo.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_data_in_bwd(self):
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                ctx.save_for_backward(input_tensor)
                return input_tensor * 3

            @staticmethod
            def backward(ctx, grad_output):
                (input_tensor,) = ctx.saved_tensors

                # Modify gradient using .data (Dangerous: Breaks autograd tracking!)
                modified_grad = grad_output.clone()
                modified_grad.data[input_tensor.data < 0] = (
                    0  # Zero-out gradients for negative inputs
                )

                return modified_grad * 3

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            return Foo.apply(x)

        x = torch.tensor([-2.0, 1.0, 3.0], requires_grad=True)
        res = fn(x)
        self.assertEqual(res, Foo.apply(x))
        res.sum().backward()
        self.assertEqual(x.grad, torch.tensor([0.0, 3.0, 3.0]))

    def test_requires_grad_in_bwd(self):
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return torch.sin(x + 1)

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                if grad_output.requires_grad:
                    return grad_output * torch.sin(
                        x + 1
                    )  # Wrong gradient, we should never get here.
                else:
                    return grad_output * torch.cos(x + 1)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            return Foo.apply(x)

        x = torch.tensor([1.0, 3.0], requires_grad=True)
        res = fn(x)
        self.assertEqual(res, Foo.apply(x))
        res.sum().backward()
        self.assertEqual(x.grad, torch.cos(x + 1))

    def test_amp_custom_fwd_bwd(self):
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        class MyMM(torch.autograd.Function):
            @staticmethod
            @torch.amp.custom_fwd(device_type="cuda")
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.amp.custom_bwd(device_type="cuda")
            def backward(ctx, grad):
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), a.t().mm(grad)

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            return MyMM.apply(a, b)

        a = torch.randn([64, 64], dtype=torch.float32, requires_grad=True)
        grad = a.clone()
        res = fn(a, a)
        res.backward(grad)

        self.assertEqual(res, MyMM.apply(a, a))
        self.assertEqual(cnt.frame_count, 1)

    def test_set_materialize_grads_no_graph_break(self):
        class MulY(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.set_materialize_grads(True)
                return x * 3

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * 3

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return MulY.apply(x)

        x = torch.tensor(2.0, requires_grad=True)
        result = f(x)
        result.sum().backward()
        self.assertEqual(result, MulY.apply(x))
        self.assertEqual(x.grad, 3.0)

    def test_user_defined_object_as_input(self):
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @dataclass
        class Weird:
            x: int
            b: torch.Tensor
            c: torch.Tensor

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x: torch.Tensor, weird: Weird, z: torch.Tensor):
                ctx.save_for_backward(weird.b, weird.c)
                return weird.b * weird.c * x.clone()

            @staticmethod
            def backward(ctx, grad):
                b, c = ctx.saved_tensors
                return grad * b * c, None, grad * 2

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, weird, z):
            return Foo.apply(x, weird, z)

        x = torch.tensor(2.0, requires_grad=True)
        weird = Weird(1.2, torch.tensor(2.5, requires_grad=True), torch.tensor(3.5))
        z = torch.tensor(3.0, requires_grad=True)

        result = f(x, weird, z)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, weird, z))
        self.assertEqual(x.grad, 2.5 * 3.5)
        self.assertEqual(z.grad, 2.0)
        self.assertEqual(weird.b.grad, None)

        # check Dynamo captured graph is correct!
        actual_graph = torch._dynamo.testing.normalize_gm(
            cnt.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[]", L_z_: "f32[]", L_weird_b: "f32[]", L_weird_c: "f32[]"):
        l_x_ = L_x_
        l_z_ = L_z_
        l_weird_b = L_weird_b
        l_weird_c = L_weird_c

        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0
        autograd_function_apply: "f32[]" = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_x_, l_z_, l_weird_b, l_weird_c, args_tensor_mask = [True, False, True], non_differentiable_idx = []);  fwd_body_0 = bwd_body_0 = l_x_ = l_z_ = l_weird_b = l_weird_c = None
        return (autograd_function_apply,)

    class fwd_body_0(torch.nn.Module):
        def forward(self, ctx : torch.autograd.function.Function, x: "f32[]", z: "f32[]", l_weird_b: "f32[]", l_weird_c: "f32[]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[]" = l_weird_b * l_weird_c
            clone: "f32[]" = x.clone();  x = None
            mul_1: "f32[]" = mul * clone;  mul = clone = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (mul_1, [l_weird_b, l_weird_c])

    class bwd_body_0(torch.nn.Module):
        def forward(self, ctx : torch.autograd.function.Function, grad: "f32[]", l_weird_b: "f32[]", l_weird_c: "f32[]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[]" = grad * l_weird_b;  l_weird_b = None
            mul_1: "f32[]" = mul * l_weird_c;  mul = l_weird_c = None
            mul_2: "f32[]" = grad * 2;  grad = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (mul_1, mul_2)
""",
        )

    def test_tensor_list_as_input(self):
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, tl):
                ctx.save_for_backward(tl[0], tl[1])
                return x.clone() * (tl[0] + tl[1])

            @staticmethod
            def backward(ctx, grad):
                tl0, tl1 = ctx.saved_tensors
                return grad * (tl0 + tl1), None

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, tl):
            return Foo.apply(x, tl)

        x = torch.tensor(2.0, requires_grad=True)
        tl = [
            torch.tensor(3.0, requires_grad=True),
            torch.tensor(4.0, requires_grad=True),
        ]

        result = f(x, tl)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, tl))
        self.assertEqual(x.grad, 7.0)
        self.assertEqual(tl[0].grad, None)
        self.assertEqual(tl[1].grad, None)

    def test_multiple_different_non_tensor_inputs(self):
        @dataclass
        class Weird:
            x: int
            b: torch.Tensor
            c: torch.Tensor

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weird, z, tl):
                ctx.save_for_backward(weird.b, weird.c, tl[0], tl[1])
                return x.clone() * weird.b * weird.c * tl[0]

            @staticmethod
            def backward(ctx, grad):
                b, c, tl0, _ = ctx.saved_tensors
                return grad * b * c * tl0, None, grad * 2, None

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, weird, z, tl):
            return Foo.apply(x, weird, z, tl)

        x = torch.tensor(2.0, requires_grad=True)
        weird = Weird(
            1.2,
            torch.tensor(2.5, requires_grad=True),
            torch.tensor(3.5, requires_grad=True),
        )
        z = torch.tensor(3.0, requires_grad=True)
        tl = [
            torch.tensor(0.5, requires_grad=True),
            torch.tensor(0.6, requires_grad=True),
        ]

        result = f(x, weird, z, tl)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, weird, z, tl))
        self.assertEqual(x.grad, 2.5 * 3.5 * 0.5)
        self.assertEqual(z.grad, 2.0)
        self.assertEqual(weird.b.grad, None)
        self.assertEqual(weird.c.grad, None)
        self.assertEqual(tl[0].grad, None)
        self.assertEqual(tl[1].grad, None)

    def test_backward_returns_none_for_tensor_input(self):
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(y)
                return x.clone() * y

            @staticmethod
            def backward(ctx, grad):
                (y,) = ctx.saved_tensors
                return grad * y, None

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, y):
            return Foo.apply(x, y)

        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)

        result = f(x, y)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, y))
        self.assertEqual(x.grad, 3.0)
        self.assertEqual(y.grad, None)

    def test_function_with_bound_free_variable(self):
        class LowerBound(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs, bound):
                ctx.save_for_backward(inputs, inputs.new_ones(1) * bound)
                return inputs.clamp(min=bound)

            @staticmethod
            def backward(ctx, grad_output):
                inputs, bound = ctx.saved_tensors
                return (inputs >= bound) * grad_output, None

        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gamma = torch.nn.Parameter(torch.rand([4, 128, 32, 32]))

            def forward(self, x):
                gamma = LowerBound.apply(self.gamma, 1)
                return x + gamma

        mod = MyMod()
        args, kwargs = ([torch.rand([4, 128, 32, 32])], {})
        before = mod(*args, **kwargs)

        compiled_model = torch.compile(mod, backend="eager")
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_forward_returns_constant(self):
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x, [1, 2, 3]  # Tensor and list of integers

            @staticmethod
            def backward(ctx, grad_output1, grad_output2):
                return grad_output1

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x):
            return Foo.apply(x)

        x = torch.tensor(2.0, requires_grad=True)
        result = f(x)
        result[0].sum().backward()

        self.assertEqual(result, Foo.apply(x))

    # I pulled all of these test cases from test_autograd.py
    # In the future, we should make the Dynamo test suite actually
    # run on test_autograd.py (it's disabled right now) and delete these.
    def test_smoke_from_test_autograd(self):
        def mult1(x):
            return x.prod(dim=-1).prod(dim=-1)

        class Mult(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = mult1(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x

        mult2 = Mult.apply

        class Double(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x**2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, _ = ctx.saved_tensors
                return grad_output * 2 * x

        # this is equivalent, but uses the output of .forward() in .backward()
        class Double2(Double):
            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * 2 * y / x

        double = Double.apply
        double2 = Double2.apply

        class Identity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                return a, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a + grad_b, grad_b

        class MyFunc2(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, gO):
                return torch.tensor(float("nan")).expand(10, 10)

        def run_fn(a):  # noqa: F841
            out = MyFunc2.apply(a)
            return out.sum()

        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                return grad

        class MyAdder(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a

            @staticmethod
            def backward(ctx, grad):
                return grad, grad

        class InplaceMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                result = x.mul_(2)
                ctx.mark_dirty(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                pass

            @staticmethod
            def jvp(ctx, x_t):
                if jvp_err:  # noqa: F821
                    return x_t
                else:
                    return x_t.mul_(2)

        class MyFn2(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x + y, x

            @staticmethod
            def vjp(ctx, gO1, gO2):
                return gO1 + gO2, gO1

            @staticmethod
            def jvp(ctx, x_t, y_t):
                return x_t + y_t, fn(x_t)  # noqa: F821

        class MyFn3(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp, inplace):
                view = inp.clone()[:3]
                if inplace:
                    view += 2
                return view

            @staticmethod
            def backward(ctx, grad):
                return grad, None

        def test():
            x = torch.ones(2, 4, 4).requires_grad_()
            mult2(x)

            x = torch.tensor(2).double().requires_grad_()
            double(x)
            double2(x)

            x = torch.randn(5, 5, requires_grad=True)
            y = torch.randn(5, 5, requires_grad=True)
            Identity.apply(x, y)

            a = torch.rand(1, 2)
            b = torch.rand(1, requires_grad=True)
            MyFn.apply(a)

            a = torch.ones(2, requires_grad=True)
            b = torch.ones(2, requires_grad=True)
            c = MyAdder.apply(a.clone(), b)
            c.sum().backward()

            z = torch.tensor(1.0, requires_grad=True)
            x = z.clone()
            y = InplaceMul.apply(x)

            a = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            b = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            c = torch.tensor(1.0, dtype=torch.double)
            d = torch.tensor(1.0, dtype=torch.double)
            MyFn2.apply(a, b)
            MyFn2.apply(c, d)

            base = torch.rand(10, requires_grad=True)
            MyFn3.apply(base, False)

        test()
        opt_test = torch.compile(test, backend="eager")
        opt_test()

    def test_tensor_subclass_intermediary_input(self):
        class FooTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data, config, scale):
                self = torch.Tensor._make_wrapper_subclass(
                    cls,
                    config[0],
                    strides=config[1],
                    storage_offset=config[2],
                    dtype=config[3],
                    layout=config[4],
                    requires_grad=config[5],
                    device=data.device,
                )
                self._data = data
                self._config = config
                self._scale = scale
                return self

            def __repr__(self):
                return "FooTensor"

            def __tensor_flatten__(self):
                return ("_data",), (
                    self._config,
                    self._scale,
                )

            @staticmethod
            def __tensor_unflatten__(tensors, metadatas, outer_size, outer_stride):
                return FooTensor(tensors["_data"], metadatas[0], metadatas[1])

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs=None):
                # handling clone and view is so dynamo fakefication passes, it's not
                # intended to be handling user code
                if func == torch.ops.aten.clone.default:
                    return FooTensor(
                        args[0]._data.clone(), args[0]._config, args[0]._scale
                    )
                elif func == torch.ops.aten.view.default:
                    new_data = args[0]._data.view(*args[1:])
                    return FooTensor(new_data, args[0]._config, args[0]._scale)

                raise NotImplementedError

        class foo_autograd_fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # access some data from `x`, where `x` is a tensor subclass
                x2 = x._data + 1.0
                # create and return a tensor subclass from within a torch.autograd.Function
                x3 = FooTensor(x2, x._config, x._scale)
                return x3._data

            @staticmethod
            def backward(ctx, g):
                return g

        x_ref = torch.randn(4, 4).requires_grad_(True)
        x = copy.deepcopy(x_ref)
        scale = torch.tensor(1.0)
        # Weird that this is needed, but not having this breaks a lot of things
        torch._dynamo.allow_in_graph(FooTensor)

        def foo(x, scale):
            config = (
                x.size(),
                x.stride(),
                x.storage_offset(),
                x.dtype,
                x.layout,
                x.requires_grad,
            )
            x = FooTensor(x, config, scale)
            x = foo_autograd_fn.apply(x)
            return x

        y_ref = foo(x_ref, scale)
        y_ref.sum().backward()

        foo_opt = torch.compile(foo, backend="eager")
        y = foo_opt(x, scale)
        y.sum().backward()

        self.assertEqual(y, y_ref)
        self.assertEqual(x.grad, x_ref.grad)

    def test_assert_is_contiguous_after_matmul(self):
        class LinearFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight):
                ctx.save_for_backward(x, weight)
                y = x.matmul(weight.t())
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, weight = ctx.saved_tensors
                grad_x = grad_output.matmul(weight)
                assert grad_x.is_contiguous()
                grad_weight = grad_output.transpose(0, 1).matmul(x)

                return grad_x, grad_weight

        def fn(x, weight):
            return LinearFunction.apply(x, weight)

        x1 = torch.randn(5, 3, requires_grad=True)
        x2 = copy.deepcopy(x1)
        W1 = torch.randn(4, 3, requires_grad=True)
        W2 = copy.deepcopy(W1)

        y1 = fn(x1, W1)
        y1.sum().backward()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        y2 = opt_fn(x2, W2)
        y2.sum().backward()

        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(W1.grad, W2.grad)
        self.assertEqual(cnts.frame_count, 1)

    def test_assert_is_contiguous_on_grad_output_directly(self):
        class LinearFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight):
                ctx.save_for_backward(x, weight)
                y = x.matmul(weight.t())
                return y

            @staticmethod
            def backward(ctx, grad_output):
                assert grad_output.is_contiguous()
                x, weight = ctx.saved_tensors
                grad_x = grad_output.matmul(weight)
                grad_weight = grad_output.transpose(0, 1).matmul(x)

                return grad_x, grad_weight

        def fn(x, weight):
            return LinearFunction.apply(x, weight)

        x1 = torch.randn(5, 3, requires_grad=True)
        x2 = copy.deepcopy(x1)
        W1 = torch.randn(4, 3, requires_grad=True)
        W2 = copy.deepcopy(W1)

        y1 = fn(x1, W1)
        y1.backward(y1.clone().detach().requires_grad_(True))

        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt)
        y2 = opt_fn(x2, W2)
        y2.backward(y2.clone().detach().requires_grad_(True))

        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(W1.grad, W2.grad)

        # Check the inserted .contiguous() call is there!
        actual_graph = torch._dynamo.testing.normalize_gm(
            cnt.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5, 3]", L_weight_: "f32[4, 3]"):
        l_x_ = L_x_
        l_weight_ = L_weight_

        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0
        autograd_function_apply: "f32[5, 4]" = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_x_, l_weight_, args_tensor_mask = [True, True], non_differentiable_idx = []);  fwd_body_0 = bwd_body_0 = l_x_ = l_weight_ = None
        return (autograd_function_apply,)

    class fwd_body_0(torch.nn.Module):
        def forward(self, ctx : torch.autograd.function.Function, x: "f32[5, 3]", weight: "f32[4, 3]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            t: "f32[3, 4]" = weight.t()
            y: "f32[5, 4]" = x.matmul(t);  t = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (y, [weight, x])

    class bwd_body_0(torch.nn.Module):
        def forward(self, function_ctx : torch.autograd.function.Function, y: "f32[5, 4]", weight: "f32[4, 3]", x: "f32[5, 3]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            contiguous: "f32[5, 4]" = y.contiguous();  y = None

            grad_x: "f32[5, 3]" = contiguous.matmul(weight);  weight = None

            transpose: "f32[4, 5]" = contiguous.transpose(0, 1);  contiguous = None
            grad_weight: "f32[4, 3]" = transpose.matmul(x);  transpose = x = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (grad_x, grad_weight)
""",
        )

    def test_smuggle_symint_issue_111031(self):
        from torch.autograd import Function

        class Foo(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.x0 = x.size(0)
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * ctx.x0

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            return Foo.apply(x)

        foo(torch.randn(2, requires_grad=True))
        self.assertEqual(cnts.frame_count, 1)

    def test_needs_input_grad(self):
        cnt = torch._dynamo.testing.CompileCounter()

        class NeedsInputGradFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, foo):
                result = foo + foo
                ctx.save_for_backward(result)
                return result

            @staticmethod
            @torch.compile(backend=cnt, fullgraph=True)
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                if ctx.needs_input_grad[0]:
                    return grad_output * result.sin()
                return None

        x = torch.randn(10, requires_grad=True)
        NeedsInputGradFunc.apply(x).sum().backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_repeated_save_for_backward_calls(self):
        from torch.autograd import Function

        class Foo(Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x)
                ctx.save_for_backward(x, y)
                return x * y

            @staticmethod
            def backward(ctx, grad_out):
                x, y = ctx.saved_tensors
                return grad_out * x, grad_out * y

        cnts = torch._dynamo.testing.CompileCounter()

        def foo(x, y):
            return Foo.apply(x, y)

        x_ref = torch.randn(2, requires_grad=True)
        y_ref = torch.randn(2, requires_grad=True)
        x_test = x_ref.detach().clone().requires_grad_()
        y_test = y_ref.detach().clone().requires_grad_()

        out_ref = foo(x_ref, y_ref)
        out_ref.sum().backward()

        out_test = torch.compile(foo, backend=cnts)(x_test, y_test)
        out_test.sum().backward()

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(x_ref.grad, x_test.grad)
        self.assertEqual(y_ref.grad, y_test.grad)

    def test_smuggle_tensor_and_complex_structures(self):
        from torch.autograd import Function

        class Foo(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.x0 = x
                ctx.x1 = [1, 2, 3]
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                x0mul = grad_out * ctx.x0
                for i in ctx.x1:
                    x0mul = (x0mul * i) + x0mul
                return x0mul

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            return Foo.apply(x)

        foo(torch.randn(2, requires_grad=True))
        self.assertEqual(cnts.frame_count, 1)

    def test_mark_non_differentiable(self):
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        from torch.autograd import Function

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x, y):
                out1 = x.sin()
                out2 = y * 2
                ctx.mark_non_differentiable(out2)
                return out1, out2

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1.cos(), grad2 * 0.0

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, y):
            return MyFunction.apply(x, y)

        x = torch.tensor(10.0, requires_grad=True)
        y = torch.tensor(20.0, requires_grad=True)
        ref1, ref2 = MyFunction.apply(x, y)
        res1, res2 = fn(x, y)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        # Ensure out1 requires gradients, out2 does not.
        self.assertTrue(ref1.requires_grad)
        self.assertTrue(res1.requires_grad)
        self.assertFalse(ref2.requires_grad)
        self.assertFalse(res2.requires_grad)
        res1.sum().backward()

        # check Dynamo captured graph is correct!
        actual_graph = torch._dynamo.testing.normalize_gm(
            cnt.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[]", L_y_: "f32[]"):
        l_x_ = L_x_
        l_y_ = L_y_

        fwd_body_0 = self.fwd_body_0
        bwd_body_0 = self.bwd_body_0
        autograd_function_apply = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_x_, l_y_, args_tensor_mask = [True, True], non_differentiable_idx = [1]);  fwd_body_0 = bwd_body_0 = l_x_ = l_y_ = None
        getitem: "f32[]" = autograd_function_apply[0]
        getitem_1: "f32[]" = autograd_function_apply[1];  autograd_function_apply = None
        return (getitem, getitem_1)

    class fwd_body_0(torch.nn.Module):
        def forward(self, ctx : torch.autograd.function.Function, x: "f32[]", y: "f32[]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            out1: "f32[]" = x.sin();  x = None

            out2: "f32[]" = y * 2;  y = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((out1, out2), [])

    class bwd_body_0(torch.nn.Module):
        def forward(self, ctx : torch.autograd.function.Function, grad1: "f32[]", grad2: "f32[]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            cos: "f32[]" = grad1.cos();  grad1 = None
            mul: "f32[]" = grad2 * 0.0;  grad2 = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (cos, mul)
""",
        )

    def test_mark_multi_output_non_differentiable(self):
        from torch.autograd import Function

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x, y, z):
                out1 = x.sin()
                out2 = y * 2
                out3 = z + 3
                ctx.mark_non_differentiable(out2, out3)
                return out1, out2, out3

            @staticmethod
            def backward(ctx, grad1, grad2, grad3):
                return grad1.cos(), grad2, grad3

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x, y, z):
            return MyFunction.apply(x, y, z)

        x = torch.tensor(10.0, requires_grad=True)
        y = torch.tensor(20.0, requires_grad=True)
        z = torch.tensor(30.0, requires_grad=True)
        ref1, ref2, ref3 = MyFunction.apply(x, y, z)
        res1, res2, res3 = fn(x, y, z)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        self.assertEqual(ref3, res3)
        # Ensure out1 requires gradients, out2 does not.
        self.assertTrue(ref1.requires_grad)
        self.assertTrue(res1.requires_grad)
        self.assertFalse(ref2.requires_grad)
        self.assertFalse(res2.requires_grad)
        self.assertFalse(ref3.requires_grad)
        self.assertFalse(res3.requires_grad)
        res1.sum().backward()

    def test_default_values(self):
        from torch.autograd import Function

        class Foo(Function):
            @staticmethod
            def forward(ctx, x, alpha=0.99):
                return x

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out

        @torch.compile
        def foo(x):
            return Foo.apply(x)

        # Make sure guards for default values do not crash
        foo(torch.randn(2))
        foo(torch.randn(2, requires_grad=True))

    def test_fwd_no_grad(self):
        # autograd.Function.forward should be traced and called under no_grad mode.
        # torch.exp with out=... arguments don't support automatic differentiation,
        # so can't be traced/called under grad mode (throwing RuntimeError),
        # therefore this unit test ensures fwd is under no_grad mode.
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                torch.exp(inputs, out=inputs)
                return inputs

            @staticmethod
            def backward(ctx, grad_output):
                return None

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return Foo.apply(x)

        x1 = torch.randn(2, 3, requires_grad=True)
        x2 = x1.clone()
        self.assertEqual(f(x1), Foo.apply(x2))

    # https://github.com/pytorch/pytorch/issues/129963
    def test_fwd_propogation_correctness(self):
        class MyCube(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                result = x**3
                dx = 3 * x**2
                ctx.save_for_backward(x, dx)
                return result, dx

            @staticmethod
            def backward(ctx, grad_output, grad_dx):
                x, dx = ctx.saved_tensors
                result = grad_output * dx + grad_dx * 6 * x
                # Intentionally return a wrong value to test if the backward is triggered twice.
                # Since if the first MyCube.apply returns values w/o requires_grad=True,
                # this backward would be only triggered once (the first MyCube.apply call),
                # as the second MyCube.apply is inlined by Dynamo and the corresponding backward
                # would be generated by autograd engine.
                return result * 0.5

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            x, _ = MyCube.apply(x)
            x, _ = MyCube.apply(x)
            return x

        inp = torch.ones(2, requires_grad=True)
        out = fn(inp)
        out.sum().backward()
        self.assertEqual(out, inp**3)
        self.assertEqual(inp.grad, torch.tensor([2.25, 2.25]))

    def test_tuple_arg(self):
        cnt = torch._dynamo.testing.CompileCounter()

        class TupleArgFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, shape):
                ctx.save_for_backward(torch.randn(shape))
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return result, None

        @torch.compile(backend=cnt, fullgraph=True)
        def fn():
            return TupleArgFunc.apply(x, shape)

        shape = (10, 10)
        x = torch.randn(shape, requires_grad=True)
        out = fn()
        out.sum().backward()
        self.assertEqual(out, x + 1)
        self.assertEqual(x.grad.shape, shape)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    @requires_cuda
    def test_triton_kernel_basic(self):
        class Add(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                output = torch.zeros_like(x)
                n_elements = output.numel()
                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                )
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return x * grad_output, y * grad_output

        @torch.compile(fullgraph=True, backend="inductor")
        def f(x, y):
            z = Add.apply(x, y)
            return z

        x = torch.randn(10, device="cuda", requires_grad=True)
        y = torch.randn(10, device="cuda", requires_grad=True)
        z = f(x, y)
        loss = z.sum()
        loss.backward()
        self.assertEqual(x + y, z)

    @requires_cuda
    def test_triton_kernel_multiple_out(self):
        class Add(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                ctx.t1 = x
                ctx.t2 = y
                output = torch.zeros_like(x)
                n_elements = output.numel()
                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                )
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
                return output, x

            @staticmethod
            def backward(ctx, grad_output, old_x):
                x, y = ctx.saved_tensors
                x1 = ctx.t1
                y1 = ctx.t2
                return old_x * x * x1 * grad_output, y * y1 * grad_output

        @torch.compile(fullgraph=True, backend="inductor")
        def f(x, y):
            z = Add.apply(x, y)
            return z

        x = torch.randn(10, device="cuda", requires_grad=True)
        y = torch.randn(10, device="cuda", requires_grad=True)
        z, _ = f(x, y)
        loss = z.sum()
        loss.backward()
        self.assertEqual(x + y, z)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
