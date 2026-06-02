# Owner(s): ["module: dynamo"]

import collections
import contextlib
import functools
import unittest

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from functorch.compile import nop
from torch._dynamo import compiled_autograd
from torch._functorch.aot_autograd import aot_module_simplified
from torch.utils.hooks import RemovableHandle


def compiler_fn(gm):
    return torch.compile(gm, backend="inductor", fullgraph=True, dynamic=True)


def global_hook_0(grad):
    return grad * 4


def global_hook_1(grad):
    return grad / 2


def global_hook_2(grad):
    return grad * 3


h0 = None


class ClassWithVal:
    def __init__(self, val):
        self.val = val


class HooksTests(torch._dynamo.test_case.TestCase):
    def test_tensor_only_register_hook_in_graph_lambda(self):
        def fn(x):
            x.register_hook(lambda grad: grad * 2)
            return x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 0)

    def test_tensor_register_hook_in_graph_lambda(self):
        def fn(x, y, z):
            x.register_hook(lambda grad: grad * 2)
            return x, y * y, z * z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_break_handle_lambda(self):
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            handle.remove()
            x.register_hook(lambda grad: grad * 3)
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([3.0, 6.0, 9.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_multi_handle_return(self):
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            h2 = handle
            z = z * z
            return x, y * y, z, handle, h2

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)
        self.assertNotEqual(h, None)
        self.assertNotEqual(h2, None)
        self.assertEqual(h2, h)

    def test_tensor_register_hook_repeated_handle_return(self):
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            h2 = handle  # noqa: F841
            z = z * z
            return x, y * y, z, handle, handle

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)
        self.assertIsInstance(h, RemovableHandle)
        self.assertIs(h2, h)

    def test_removed_handle_return(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            handle.remove()
            handle.remove()
            return x, y * y, z, handle, handle

        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsInstance(h, RemovableHandle)
        self.assertIs(h2, h)

    def test_tensor_register_hook_repeated_handle_not_local(self):
        def fn(x, y, z, mod):
            mod.handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts, fullgraph=True)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

        mod = torch.nn.Module()
        mod.handle = None

        v, y, z = fn(v, torch.randn([2, 2]), torch.randn([2, 2]), mod)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

        self.assertNotEqual(mod.handle, None)

    def test_tensor_only_register_hook_in_graph_local(self):
        def local_hook(grad):
            return grad * 2

        def fn(x):
            x.register_hook(local_hook)
            return x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 0)

    def test_tensor_only_register_hook_in_graph_local_inner(self):
        def fn(x):
            def local_hook(grad):
                return grad * 2

            z = x * x
            x.register_hook(local_hook)
            z.register_hook(local_hook)
            return x, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v[0].backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v[0].grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_local(self):
        def local_hook(grad):
            return grad * 2

        def fn(x, y, z):
            x.register_hook(local_hook)
            return x, y * y, z * z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_break_handle_local(self):
        def local_hook(grad):
            return grad * 2

        def local_hook2(grad):
            return grad * 3

        def fn(x, y, z):
            handle = x.register_hook(local_hook)
            z = z * z
            handle.remove()
            x.register_hook(local_hook2)
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(v.grad, torch.tensor([3.0, 6.0, 9.0]))

    def test_tensor_register_global_hook(self):
        def fn(x):
            x.register_hook(global_hook_0)
            return x, x * x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([4.0, 8.0, 12.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_multiple_hooks(self):
        def fn(x):
            x.register_hook(global_hook_0)  # * 4
            x.register_hook(global_hook_1)  # / 2
            x.register_hook(global_hook_2)  # * 3
            return x, x * x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([6.0, 12.0, 18.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_multiple_hooks_handles_in_list(self):
        def fn(x):
            h0 = x.register_hook(global_hook_0)  # * 4
            h1 = x.register_hook(global_hook_1)  # / 2
            h2 = x.register_hook(global_hook_2)  # * 3
            return x, x * x, h0, h1, h2

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, r, handle_0, handle_1, handle_2 = fn(v)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([6.0, 12.0, 18.0]))
        handle_0.remove()
        handle_1.remove()
        handle_2.remove()

        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # Handles gone, grad is just applied as is
        self.assertEqual(v.grad, torch.tensor([7.0, 14.0, 21.0]))

        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_global_hooks_handles_in_list(self):
        def fn(x):
            global h0
            h0 = x.register_hook(global_hook_0)  # * 4
            return x, x * x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, r = fn(v)

        self.assertIsNotNone(h0)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([4.0, 8.0, 12.0]))
        h0.remove()

        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # Handles gone, grad is just applied as is
        self.assertEqual(v.grad, torch.tensor([5.0, 10.0, 15.0]))

        # NYI!
        self.assertEqual(cnts.frame_count, 0)

    def test_hook_on_intermediate(self):
        def fn(x):
            y = x * 2
            y.register_hook(lambda grad: grad + 1)
            return y.sum()

        x_compiled = torch.randn(4, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        x_eager = x_compiled.detach().clone().requires_grad_(True)
        result_eager = fn(x_eager)
        result_eager.backward()

        self.assertEqual(x_compiled.grad, x_eager.grad)

    def test_hook_on_intermediate_with_container(self):
        glb_list = []
        glb_dict = {}

        def fn(x):
            y = x * 2
            glb_list.append(y)
            glb_dict["tensor"] = y
            a = glb_list[0] * 3  # Should use output of register_hook
            b = glb_dict["tensor"]
            y.register_hook(lambda grad: grad + 1)
            return (a + b).sum()

        glb_list.clear()
        glb_dict.clear()
        x_eager = torch.ones(4, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        glb_list.clear()
        glb_dict.clear()
        x_compiled = torch.ones(4, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_compiled.grad, x_eager.grad)
        # Without hook: dloss/dy = 4, dloss/dx = 8
        # With hook (+1): hooked = 5, dloss/dx = 10
        self.assertEqual(x_compiled.grad, torch.full_like(x_compiled, 10.0))

        glb_list.clear()
        glb_dict.clear()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(
            torch.ones(4, requires_grad=True)
        )
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    y = l_x_ * 2;  l_x_ = None
    a = y * 3
    hook_body_0 = self.hook_body_0
    register_hook = torch.ops.higher_order.register_hook(y, hook_body_0);  y = hook_body_0 = None
    add = a + register_hook;  a = None
    sum_1 = add.sum();  add = None
    return (sum_1, register_hook)""",
        )

    def test_hook_on_intermediate_used_before_and_after(self):
        def fn(x):
            y = x * 2
            z = y + 1  # Use y BEFORE hook
            y.register_hook(lambda g: g * 2)
            w = y * 3  # Use y AFTER hook
            return (z + w).sum()

        x_eager = torch.ones(2, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(2, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_on_intermediate_with_higher_order_op(self):
        def fn(x):
            y = x * 2
            y.register_hook(lambda g: g * 2)

            def true_fn(t):
                return t + 1

            def false_fn(t):
                return t - 1

            z = torch.cond(x.sum() > 0, true_fn, false_fn, (y,))
            return z.sum()

        x_eager = torch.ones(3, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(3, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_on_intermediate_returns_none(self):
        def fn(x):
            y = x * 2
            y.register_hook(lambda g: None)
            return y.sum()

        x_eager = torch.ones(4, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(4, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)
        self.assertEqual(x_compiled.grad, torch.full_like(x_compiled, 2.0))

    def test_hook_has_side_effect(self):
        def fn(x):
            y = x * 2
            z = y + 1  # Use y BEFORE hook
            y.register_hook(lambda g: g * 2)
            w = y * 3  # Use y AFTER hook
            return (z + w).sum()

        x_eager = torch.ones(2, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(2, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_bwd_inside_side_effects(self):
        global_list = []

        def fn(x):
            y = x * 2

            def _hook(grad):
                global_list.append(grad)
                return grad * 2

            y.register_hook(_hook)
            z = y + x
            return z.sum()

        x_eager = torch.ones(3, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        global_list.clear()

        x_compiled = torch.ones(3, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Unsafe side effect"
        ):
            _ = compiled_fn(x_compiled)

    def test_hook_on_intermediate_from_split(self):
        def fn(x):
            splits = x.split(2)
            result = torch.cat(splits)  # use splits before register_hook
            y = splits[0]
            y.register_hook(lambda g: g + 1)
            return result.sum() + y.sum()

        x_eager = torch.ones(6, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(6, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(
            torch.ones(6, requires_grad=True)
        )
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    split = l_x_.split(2);  l_x_ = None
    y = split[0]
    getitem_1 = split[1]
    getitem_2 = split[2];  split = None
    result = torch.cat((y, getitem_1, getitem_2));  getitem_1 = getitem_2 = None
    hook_body_0 = self.hook_body_0
    register_hook = torch.ops.higher_order.register_hook(y, hook_body_0);  y = hook_body_0 = None
    sum_1 = result.sum();  result = None
    sum_2 = register_hook.sum();  register_hook = None
    add = sum_1 + sum_2;  sum_1 = sum_2 = None
    return (add,)""",
        )

    def test_intermediary_hooks(self):
        def simple_hook(g):
            return g * 2

        def f(x):
            y = x + 1
            y.register_hook(simple_hook)
            z = y + 1
            return z

        out = torch.randn(1, requires_grad=True)
        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(f, backend=cnts, fullgraph=True)
        res = fn(out)
        res.backward()
        self.assertEqual(res, f(out))
        self.assertEqual(out.grad, torch.Tensor([2.0]))

    def test_intermediary_hooks_same_on_aot_eager(self):
        def my_hook(grad, *, k=0):
            return grad + k

        class MyMod(torch.nn.Module):
            def forward(self, x):
                y = x.mul(2)
                hook1 = functools.partial(my_hook, k=3)
                hook2 = functools.partial(my_hook, k=4)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(torch.ones(4))

        x1 = torch.ones(4, requires_grad=True)
        mod_compiled = aot_module_simplified(mod, (x1,), nop)
        aot_out = mod_compiled(x1)
        aot_out[0].backward(torch.ones(4))

        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="aot_eager", fullgraph=True)(x2)
            dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(dynamo_out, aot_out)
        self.assertEqual(dynamo_out, eager_out)

        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(x0.grad, x2.grad)

    def test_input_hooks_same(self):
        backends = ["eager", "aot_eager", "inductor"]
        for backend in backends:

            def my_hook(grad, *, k=0):
                return grad + k

            hook = functools.partial(my_hook, k=3)

            class MyMod(torch.nn.Module):
                def forward(self, x):
                    x.register_hook(hook)
                    y = x.mul(2)
                    z = y.mul(3)
                    return (z,)

            mod = MyMod()
            x0 = torch.ones(4, requires_grad=True)
            eager_out = mod(x0)
            eager_out[0].backward(torch.ones(4))

            x1 = torch.ones(4, requires_grad=True)
            mod_compiled = aot_module_simplified(mod, (x1,), nop)
            aot_out = mod_compiled(x1)
            aot_out[0].backward(torch.ones(4))

            x2 = torch.ones(4, requires_grad=True)
            dynamo_out = torch.compile(mod, backend=backend, fullgraph=True)(x2)
            with compiled_autograd._enable(compiler_fn):
                dynamo_out[0].backward(torch.ones(4))

            self.assertEqual(dynamo_out, aot_out)
            self.assertEqual(dynamo_out, eager_out)

            self.assertEqual(x0.grad, x1.grad)
            self.assertEqual(x0.grad, x2.grad)

    def test_intermediary_hooks_same_on_inductor(self):
        def my_hook(grad, *, k=0):
            return grad + k

        class MyMod(torch.nn.Module):
            def forward(self, x):
                y = x.mul(2)
                hook1 = functools.partial(my_hook, k=3)
                hook2 = functools.partial(my_hook, k=4)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(torch.ones(4))

        x1 = torch.ones(4, requires_grad=True)
        mod_compiled = aot_module_simplified(mod, (x1,), nop)
        aot_out = mod_compiled(x1)
        aot_out[0].backward(torch.ones(4))

        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="inductor", fullgraph=True)(x2)
            dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(dynamo_out, aot_out)
        self.assertEqual(dynamo_out, eager_out)

        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(x0.grad, x2.grad)

    def test_complex_state_mutation_in_intermediary_hooks_same_on_inductor(self):
        class SomePyClass:
            count = 0

            def do_stuff(self, grad):
                if self.count % 2 == 0:
                    r = grad * grad
                else:
                    r = grad + grad
                self.count += 1
                return r

        def complex_state_touching_hook(grad, *, obj):
            return obj.do_stuff(grad)

        class MyMod(torch.nn.Module):
            def forward(self, x, obj):
                y = x.mul(2)
                hook1 = functools.partial(complex_state_touching_hook, obj=obj)
                hook2 = functools.partial(complex_state_touching_hook, obj=obj)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj = SomePyClass()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0, obj)
        eager_out[0].backward(torch.ones(4))

        # Eager 2
        self.assertEqual(obj.count, 2)
        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="inductor", fullgraph=True)(x2, obj)
            dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(dynamo_out, eager_out)

        # Eager 2 + compiled 2
        self.assertEqual(obj.count, 4)
        self.assertEqual(x0.grad, x2.grad)

    def test_complex_state_mutation_in_intermediary_hooks_same_on_inductor_with_graph_break(
        self,
    ):
        class SomePyClass:
            grad_as_str = "None"
            count = 0

            def write_grad_as_str_and_do_stuff(self, grad):
                self.grad_as_str = str(grad)
                if self.count % 2 == 0:
                    r = grad * grad
                else:
                    r = grad + grad
                print("Break!")
                self.count += 1
                return r

        def complex_state_touching_hook(grad, *, obj):
            return obj.write_grad_as_str_and_do_stuff(grad)

        class MyMod(torch.nn.Module):
            def forward(self, x, obj):
                y = x.mul(2)
                hook1 = functools.partial(complex_state_touching_hook, obj=obj)
                hook2 = functools.partial(complex_state_touching_hook, obj=obj)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj = SomePyClass()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0, obj)
        eager_out[0].backward(torch.ones(4))

        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="inductor", fullgraph=True)(x2, obj)
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "Failed to trace builtin operator"
            ):
                dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(obj.count, 2)

    def test_register_hook_partial_guarding(
        self,
    ):
        def some_hook(grad, *, obj):
            return grad + obj.val

        class MyMod(torch.nn.Module):
            def forward(self, x, obj):
                y = x.mul(2)
                hook1 = functools.partial(some_hook, obj=obj)
                y.register_hook(hook1)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        obj3 = ClassWithVal(11)
        cnt = torch._dynamo.testing.CompileCounter()

        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)

        with compiled_autograd._enable(compiler_fn):
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj1)
            torch.compile(mod, backend=cnt, fullgraph=True)(x1, obj1)
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj2)
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj3)
            self.assertEqual(cnt.frame_count, 1)

    def test_hook_with_closure(self):
        def fn(x, obj):
            y = x.sin()
            x.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_fw = torch._dynamo.testing.CompileCounter()
        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend=cnt_fw, fullgraph=True)

        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_hook_with_nested_closure(self):
        def fn(x):
            def run():
                y = x.sin()
                x.register_hook(lambda grad: grad + y)
                z = y.sin()
                return z

            return run()

        cnt_fw = torch._dynamo.testing.CompileCounter()
        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend=cnt_fw, fullgraph=True)

        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        fn(x0).sum().backward()
        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x1).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x1.grad)

    def test_intermediate_hook_with_closure_eager(self):
        def fn(x, obj):
            y = x.sin()
            y.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_fw = torch._dynamo.testing.CompileCounter()
        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend=cnt_fw, fullgraph=True)

        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_intermediate_hook_with_closure_aot(self):
        def fn(x, obj):
            y = x.sin()
            y.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend="aot_eager", fullgraph=True)

        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_no_recompile_on_hook_identity_change(self):
        def my_hook(grad, k=0):
            return grad + k

        def my_hook2(grad):
            return grad * 2

        class MyMod(torch.nn.Module):
            def forward(self, x):
                y = x.mul(2)
                y.register_hook(my_hook)
                y.register_hook(my_hook)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(torch.ones(4))

        x1 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            comp_mod = torch.compile(mod, backend=cnts, fullgraph=True)
            comp_out = comp_mod(x1)
            comp_out[0].backward(torch.ones(4))

            self.assertEqual(cnts.frame_count, 1)
            my_hook = my_hook2
            self.assertEqual(x0.grad, x1.grad)

            eager_out = mod(x0)
            eager_out[0].backward(torch.ones(4))

            comp_out = comp_mod(x1)

            self.assertEqual(cnts.frame_count, 1)
            comp_out[0].backward(torch.ones(4))
            self.assertEqual(x0.grad, x1.grad)

    def test_functools_arg_vary(self):
        def pre_hook(grad, *, k):
            return grad * k

        hook = functools.partial(pre_hook, k=1)

        @torch.compile(backend="eager", fullgraph=True)
        def h(x):
            y = x.mul(2)
            y.register_hook(hook)
            return y.mul(3)

        with compiled_autograd._enable(torch.compile(backend="eager", fullgraph=True)):
            x = torch.randn(2, requires_grad=True)
            h(x).sum().backward()
            orig_grad = x.grad
            x.grad = None

            hook = functools.partial(pre_hook, k=2)
            h(x).sum().backward()
            self.assertEqual(orig_grad * 2, x.grad)

    def test_post_acc_grad_hook(self):
        def hook(input_t):
            input_t.mul_(input_t.grad)
            input_t.grad.mul_(5)

        def reg_and_mul(x, y):
            x.register_post_accumulate_grad_hook(hook)
            return x * y

        cnts = None

        def test_fn(fn):
            fn(x, y)
            b = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
            x.backward(b)
            if cnts:
                self.assertEqual(cnts.frame_count, 1)
            # These same exact assertions run on both eager and compiled
            # X goes to x*2 because of mul_
            self.assertEqual(x, torch.tensor([0.5, 0.5, 0.5]) * 2)
            # This test proves grad aliasing works -
            self.assertEqual(x.grad, b * 5)

        # Eager values
        x = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        test_fn(reg_and_mul)

        # Compiled
        for backend in ["eager", "aot_eager", "inductor"]:
            for compiled_bwd in [False, True]:
                torch._dynamo.reset()
                x = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
                y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

                cnts = torch._dynamo.testing.CompileCounterWithBackend(backend)
                compiled_fn = torch.compile(reg_and_mul, backend=cnts, fullgraph=True)

                compiled_bwd_ctx = (
                    compiled_autograd._enable(
                        torch.compile(backend=backend, fullgraph=True)
                    )
                    if compiled_bwd
                    else contextlib.nullcontext()
                )
                with compiled_bwd_ctx:
                    test_fn(compiled_fn)

    def test_recompile(self):
        def hook(param):
            param.grad *= 2

        x = torch.ones(10)
        x.requires_grad = True

        def run(input):
            return x * input

        x.register_post_accumulate_grad_hook(hook)
        with compiled_autograd._enable(compiler_fn):
            for i in range(5):
                with unittest.mock.patch(
                    "torch._dynamo.config.error_on_recompile", True
                ):
                    # Mimic optimizer.zero_grad() to clear the gradient
                    x.grad = None
                    run(i).sum().backward()

    def test_no_recompile_on_same_hook(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fw_hook(inp):
            return (inp[0] + 1,)

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList()
                for _ in range(10):
                    layer = torch.nn.Linear(16, 16)
                    layer.register_forward_pre_hook(lambda _, inp: fw_hook(inp))
                    layer = torch.compile(layer, backend=cnts)
                    self.layers.append(layer)

            def forward(self, x):
                for l in self.layers:
                    x = l(x)
                return x

        mod = Mod()
        x = torch.ones(16, 16, requires_grad=True)
        mod(x)

        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(skip_nnmodule_hook_guards=False)
    def test_nnmodule_hook_guards(self):
        # Compile a model and then apply a hook

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(x)

        cnts = torch._dynamo.testing.CompileCounter()

        mod = Mod()

        def fn(x):
            return mod(x)

        opt_fn = torch.compile(fn, backend=cnts)

        x = torch.ones(16, 16)
        opt_fn(x)

        # Register a hook
        def forward_hook(self, inputs, out):
            return out * 2

        mod.register_forward_hook(forward_hook)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_register_forward_hook_inside_compiled_region(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def forward_hook(module, args, kwargs, output):
                    return output + 1

                handle = self.linear.register_forward_hook(
                    forward_hook, prepend=True, with_kwargs=True, always_call=True
                )
                try:
                    return self.linear(x)
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            mod = Mod()
            x = torch.randn(4, 4)
            ref = mod(x)
            self.assertEqual(len(mod.linear._forward_hooks), 0)
            self.assertEqual(len(mod.linear._forward_hooks_with_kwargs), 0)
            self.assertEqual(len(mod.linear._forward_hooks_always_called), 0)
            next_id_before_compile = RemovableHandle.next_id

            cnts = torch._dynamo.testing.CompileCounter()
            opt_mod = torch.compile(mod, backend=cnts, fullgraph=True)

            for _ in range(3):
                res = opt_mod(x)
                self.assertEqual(ref, res)
                self.assertEqual(len(mod.linear._forward_hooks), 0)
                self.assertEqual(len(mod.linear._forward_hooks_with_kwargs), 0)
                self.assertEqual(len(mod.linear._forward_hooks_always_called), 0)
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(RemovableHandle.next_id, next_id_before_compile + 3)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_register_forward_hook_inside_compiled_region_with_existing_hook(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def forward_hook(module, args, kwargs, output):
                    return output + 1

                handle = self.linear.register_forward_hook(
                    forward_hook, with_kwargs=True, always_call=True
                )
                try:
                    return self.linear(x)
                finally:
                    handle.remove()

        def existing_hook(module, args, output):
            return output + 2

        old_next_id = RemovableHandle.next_id
        pre_handle = None
        try:
            mod = Mod()
            pre_handle = mod.linear.register_forward_hook(
                existing_hook, always_call=True
            )
            x = torch.randn(4, 4)
            ref = mod(x)
            next_id_before_compile = RemovableHandle.next_id

            cnts = torch._dynamo.testing.CompileCounter()
            opt_mod = torch.compile(mod, backend=cnts, fullgraph=True)

            for _ in range(3):
                res = opt_mod(x)
                self.assertEqual(ref, res)
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(RemovableHandle.next_id, next_id_before_compile + 3)
        finally:
            if pre_handle is not None:
                pre_handle.remove()
            RemovableHandle.next_id = old_next_id

    def test_register_forward_hook_inside_compiled_region_existing_hook_id_collision_graph_breaks(
        self,
    ):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output + 1

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x)
                finally:
                    handle.remove()

        def existing_hook(module, args, output):
            return output + 2

        old_next_id = RemovableHandle.next_id
        pre_handle = None
        try:
            RemovableHandle.next_id = 601
            mod = Mod()
            pre_handle = mod.linear.register_forward_hook(existing_hook)
            self.assertEqual(pre_handle.id, 601)
            RemovableHandle.next_id = 600

            opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "RemovableHandle.id"
            ):
                opt_mod(torch.randn(4, 4))
        finally:
            if pre_handle is not None:
                pre_handle.remove()
            RemovableHandle.next_id = old_next_id

    def test_register_forward_hook_inside_compiled_region_existing_hook_next_id_reset_invalidates_guard(
        self,
    ):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output + 1

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x)
                finally:
                    handle.remove()

        def existing_hook(module, args, output):
            return output + 2

        old_next_id = RemovableHandle.next_id
        pre_handle = None
        try:
            RemovableHandle.next_id = 0
            mod = Mod()
            pre_handle = mod.linear.register_forward_hook(existing_hook)
            x = torch.randn(4, 4)
            ref = mod(x)

            cnts = torch._dynamo.testing.CompileCounter()
            opt_mod = torch.compile(mod, backend=cnts, fullgraph=True)
            self.assertEqual(opt_mod(x), ref)
            self.assertEqual(cnts.frame_count, 1)

            RemovableHandle.next_id = pre_handle.id
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "RemovableHandle.id"
            ):
                opt_mod(x)
        finally:
            if pre_handle is not None:
                pre_handle.remove()
            RemovableHandle.next_id = old_next_id

    def test_register_forward_hook_inside_compiled_region_existing_hook_key_replacement_invalidates_guard(
        self,
    ):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output + 1

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x)
                finally:
                    handle.remove()

        def existing_hook(module, args, output):
            return output + 2

        old_next_id = RemovableHandle.next_id
        pre_handle = None
        replacement_handle = None
        try:
            RemovableHandle.next_id = 0
            mod = Mod()
            pre_handle = mod.linear.register_forward_hook(existing_hook)
            x = torch.randn(4, 4)
            ref = mod(x)

            cnts = torch._dynamo.testing.CompileCounter()
            opt_mod = torch.compile(mod, backend=cnts, fullgraph=True)
            self.assertEqual(opt_mod(x), ref)
            self.assertEqual(cnts.frame_count, 1)

            pre_handle.remove()
            RemovableHandle.next_id = 100
            replacement_handle = mod.linear.register_forward_hook(existing_hook)
            self.assertEqual(replacement_handle.id, 100)
            RemovableHandle.next_id = replacement_handle.id
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "RemovableHandle.id"
            ):
                opt_mod(x)
        finally:
            if pre_handle is not None:
                pre_handle.remove()
            if replacement_handle is not None:
                replacement_handle.remove()
            RemovableHandle.next_id = old_next_id

    def test_register_forward_hook_inside_compiled_region_existing_hook_key_replacement_preserves_runtime_key(
        self,
    ):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output + 1

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x)
                finally:
                    handle.remove()

        def existing_hook(module, args, output):
            return output + 2

        old_next_id = RemovableHandle.next_id
        pre_handle = None
        replacement_handle = None
        try:
            RemovableHandle.next_id = 0
            mod = Mod()
            pre_handle = mod.linear.register_forward_hook(existing_hook)
            x = torch.randn(4, 4)
            ref = mod(x)

            cnts = torch._dynamo.testing.CompileCounter()
            opt_mod = torch.compile(mod, backend=cnts, fullgraph=True)
            self.assertEqual(opt_mod(x), ref)
            self.assertEqual(cnts.frame_count, 1)

            pre_handle.remove()
            RemovableHandle.next_id = 100
            replacement_handle = mod.linear.register_forward_hook(existing_hook)
            self.assertEqual(replacement_handle.id, 100)
            self.assertEqual(opt_mod(x), ref)
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(list(mod.linear._forward_hooks.keys()), [100])

            replacement_handle.remove()
            self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            if pre_handle is not None:
                pre_handle.remove()
            if replacement_handle is not None:
                replacement_handle.remove()
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_public_hook_dict_keys_replacement_recompiles(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output + 1

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    out = self.linear(x)
                finally:
                    handle.remove()
                return out + sum(self.linear._forward_hooks.keys())

        def existing_hook(module, args, output):
            return output + 2

        old_next_id = RemovableHandle.next_id
        pre_handle = None
        replacement_handle = None
        try:
            RemovableHandle.next_id = 0
            mod = Mod()
            pre_handle = mod.linear.register_forward_hook(existing_hook)
            x = torch.randn(4, 4)

            cnts = torch._dynamo.testing.CompileCounter()
            opt_mod = torch.compile(mod, backend=cnts, fullgraph=True)
            self.assertEqual(opt_mod(x), torch.full_like(x, 3))
            self.assertEqual(cnts.frame_count, 1)

            pre_handle.remove()
            RemovableHandle.next_id = 100
            replacement_handle = mod.linear.register_forward_hook(existing_hook)
            self.assertEqual(replacement_handle.id, 100)
            self.assertEqual(opt_mod(x), torch.full_like(x, 103))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(list(mod.linear._forward_hooks.keys()), [100])
        finally:
            if pre_handle is not None:
                pre_handle.remove()
            if replacement_handle is not None:
                replacement_handle.remove()
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_public_hook_dict_direct_iteration_replacement_recompiles(
        self,
    ):
        class Mod(torch.nn.Module):
            def __init__(self, mode) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.mode = mode
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output + 1

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    out = self.linear(x)
                finally:
                    handle.remove()

                if self.mode == "sum_iter":
                    key_value = sum(self.linear._forward_hooks)
                else:
                    key_value = next(iter(self.linear._forward_hooks))
                return out + key_value

        def existing_hook(module, args, output):
            return output + 2

        old_next_id = RemovableHandle.next_id
        try:
            for mode in ("sum_iter", "list_index"):
                with self.subTest(mode=mode):
                    pre_handle = None
                    replacement_handle = None
                    try:
                        RemovableHandle.next_id = 0
                        mod = Mod(mode)
                        pre_handle = mod.linear.register_forward_hook(existing_hook)
                        x = torch.randn(4, 4)

                        cnts = torch._dynamo.testing.CompileCounter()
                        opt_mod = torch.compile(mod, backend=cnts, fullgraph=True)
                        self.assertEqual(opt_mod(x), torch.full_like(x, 3))
                        self.assertEqual(cnts.frame_count, 1)

                        pre_handle.remove()
                        RemovableHandle.next_id = 100
                        replacement_handle = mod.linear.register_forward_hook(
                            existing_hook
                        )
                        self.assertEqual(replacement_handle.id, 100)
                        self.assertEqual(opt_mod(x), torch.full_like(x, 103))
                        self.assertEqual(cnts.frame_count, 2)
                        self.assertEqual(list(mod.linear._forward_hooks.keys()), [100])
                    finally:
                        if pre_handle is not None:
                            pre_handle.remove()
                        if replacement_handle is not None:
                            replacement_handle.remove()
        finally:
            RemovableHandle.next_id = old_next_id

    def test_register_forward_pre_hook_inside_compiled_region_prepend(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def forward_pre_hook(module, args):
                    return (args[0] + 1,)

                handle = self.linear.register_forward_pre_hook(
                    forward_pre_hook, prepend=True
                )
                try:
                    return self.linear(x)
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            mod = Mod()
            x = torch.randn(4, 4)
            ref = mod(x)
            self.assertEqual(len(mod.linear._forward_pre_hooks), 0)
            self.assertEqual(len(mod.linear._forward_pre_hooks_with_kwargs), 0)

            cnts = torch._dynamo.testing.CompileCounter()
            opt_mod = torch.compile(mod, backend=cnts, fullgraph=True)

            res = opt_mod(x)
            self.assertEqual(ref, res)
            self.assertEqual(len(mod.linear._forward_pre_hooks), 0)
            self.assertEqual(len(mod.linear._forward_pre_hooks_with_kwargs), 0)
            self.assertEqual(cnts.frame_count, 1)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_plain_dict_matches_eager_type_error(self):
        def fn(x):
            RemovableHandle({})
            return x + 1

        opt_fn = torch.compile(fn, backend="eager")
        with self.assertRaisesRegex(
            TypeError, "cannot create weak reference to 'dict' object"
        ):
            opt_fn(torch.ones(()))

    def test_removable_handle_tuple_extra_dict_ignored(self):
        mod = torch.nn.Linear(4, 4)
        old_next_id = RemovableHandle.next_id

        def fn(x):
            handle = RemovableHandle(
                mod._forward_hooks,
                extra_dict=(mod._forward_hooks_with_kwargs,),
            )
            handle.remove()
            return x + 1

        try:
            x = torch.randn(4, 4)
            mod._forward_hooks_with_kwargs[old_next_id] = True
            cnts = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(len(mod._forward_hooks), 0)
            self.assertEqual(
                list(mod._forward_hooks_with_kwargs.items()),
                [(old_next_id, True)],
            )
            self.assertEqual(RemovableHandle.next_id, old_next_id + 1)
        finally:
            mod._forward_hooks.clear()
            mod._forward_hooks_with_kwargs.clear()
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_read_after_allocation_graph_breaks(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x) + handle.id
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            mod = Mod()
            opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "RemovableHandle.id"
            ):
                opt_mod(torch.ones(4, 4))
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_read_spoofed_nn_modules_filename_graph_breaks(self):
        fake_filename = "/tmp/user/torch/nn/modules/not_internal.py"
        namespace = {"torch": torch}
        exec(
            compile(
                """
class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        def forward_hook(module, args, output):
            return output

        handle = self.linear.register_forward_hook(forward_hook)
        try:
            return self.linear(x) + handle.id
        finally:
            handle.remove()
""",
                fake_filename,
                "exec",
            ),
            namespace,
        )

        old_next_id = RemovableHandle.next_id
        try:
            mod = namespace["Mod"]()
            opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "RemovableHandle.id"
            ):
                opt_mod(torch.ones(4, 4))
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_read_after_allocation_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x) + handle.id
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            RemovableHandle.next_id = 100
            mod = Mod()
            x = torch.randn(4, 4)
            opt_mod = torch.compile(mod, backend="eager")

            res0 = opt_mod(x)
            res1 = opt_mod(x)

            self.assertEqual(res0, torch.full_like(res0, 100))
            self.assertEqual(res1, torch.full_like(res1, 101))
            self.assertEqual(RemovableHandle.next_id, 102)
            self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_hash_after_allocation_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x) + hash(handle.id)
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            RemovableHandle.next_id = 600
            mod = Mod()
            x = torch.randn(4, 4)
            opt_mod = torch.compile(mod, backend="eager")

            res0 = opt_mod(x)
            res1 = opt_mod(x)

            self.assertEqual(res0, torch.full_like(res0, 600))
            self.assertEqual(res1, torch.full_like(res1, 601))
            self.assertEqual(RemovableHandle.next_id, 602)
            self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_dunder_hash_after_allocation_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x) + handle.id.__hash__()
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            RemovableHandle.next_id = 700
            mod = Mod()
            x = torch.randn(4, 4)
            opt_mod = torch.compile(mod, backend="eager")

            res0 = opt_mod(x)
            res1 = opt_mod(x)

            self.assertEqual(res0, torch.full_like(res0, 700))
            self.assertEqual(res1, torch.full_like(res1, 701))
            self.assertEqual(RemovableHandle.next_id, 702)
            self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_public_dict_key_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x) + {handle.id: 5}.get(600, 7)
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            RemovableHandle.next_id = 600
            mod = Mod()
            x = torch.randn(4, 4)
            opt_mod = torch.compile(mod, backend="eager")

            res0 = opt_mod(x)
            res1 = opt_mod(x)

            self.assertEqual(res0, torch.full_like(res0, 5))
            self.assertEqual(res1, torch.full_like(res1, 7))
            self.assertEqual(RemovableHandle.next_id, 602)
            self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_public_dict_key_miss_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x) + {handle.id: 5}.get(601, 7)
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            RemovableHandle.next_id = 600
            mod = Mod()
            x = torch.randn(4, 4)
            opt_mod = torch.compile(mod, backend="eager")

            res0 = opt_mod(x)
            res1 = opt_mod(x)

            self.assertEqual(res0, torch.full_like(res0, 7))
            self.assertEqual(res1, torch.full_like(res1, 5))
            self.assertEqual(RemovableHandle.next_id, 602)
            self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_public_dict_float_key_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x) + {handle.id: 5}.get(601.0, 7)
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            RemovableHandle.next_id = 600
            mod = Mod()
            x = torch.randn(4, 4)
            opt_mod = torch.compile(mod, backend="eager")

            res0 = opt_mod(x)
            res1 = opt_mod(x)

            self.assertEqual(res0, torch.full_like(res0, 7))
            self.assertEqual(res1, torch.full_like(res1, 5))
            self.assertEqual(RemovableHandle.next_id, 602)
            self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_public_set_key_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x) + (5 if 700 in {handle.id} else 7)
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            RemovableHandle.next_id = 700
            mod = Mod()
            x = torch.randn(4, 4)
            opt_mod = torch.compile(mod, backend="eager")

            res0 = opt_mod(x)
            res1 = opt_mod(x)

            self.assertEqual(res0, torch.full_like(res0, 5))
            self.assertEqual(res1, torch.full_like(res1, 7))
            self.assertEqual(RemovableHandle.next_id, 702)
            self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_public_set_key_miss_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x) + (5 if 701 in {handle.id} else 7)
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            RemovableHandle.next_id = 700
            mod = Mod()
            x = torch.randn(4, 4)
            opt_mod = torch.compile(mod, backend="eager")

            res0 = opt_mod(x)
            res1 = opt_mod(x)

            self.assertEqual(res0, torch.full_like(res0, 7))
            self.assertEqual(res1, torch.full_like(res1, 5))
            self.assertEqual(RemovableHandle.next_id, 702)
            self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_public_set_float_key_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x) + (5 if 601.0 in {handle.id} else 7)
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            RemovableHandle.next_id = 600
            mod = Mod()
            x = torch.randn(4, 4)
            opt_mod = torch.compile(mod, backend="eager")

            res0 = opt_mod(x)
            res1 = opt_mod(x)

            self.assertEqual(res0, torch.full_like(res0, 7))
            self.assertEqual(res1, torch.full_like(res1, 5))
            self.assertEqual(RemovableHandle.next_id, 602)
            self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_public_hook_dict_access_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self, mode) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.mode = mode
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    if self.mode == "contains":
                        value = 5 if 601 in self.linear._forward_hooks else 7
                    elif self.mode == "get":
                        value = 5 if self.linear._forward_hooks.get(601) else 7
                    else:
                        value = len(self.linear._forward_hooks | {601: forward_hook})
                    return self.linear(x) + value
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            for mode, expected0, expected1 in (
                ("contains", 7, 5),
                ("get", 7, 5),
                ("union", 2, 1),
            ):
                with self.subTest(mode=mode):
                    RemovableHandle.next_id = 600
                    mod = Mod(mode)
                    x = torch.randn(4, 4)
                    opt_mod = torch.compile(mod, backend="eager")

                    res0 = opt_mod(x)
                    res1 = opt_mod(x)

                    self.assertEqual(res0, torch.full_like(res0, expected0))
                    self.assertEqual(res1, torch.full_like(res1, expected1))
                    self.assertEqual(RemovableHandle.next_id, 602)
                    self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_public_key_respects_next_id_reset(self):
        class Mod(torch.nn.Module):
            def __init__(self, mode) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.mode = mode
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    if self.mode == "dict":
                        value = {handle.id: 5}.get(599, 7)
                    else:
                        value = 5 if 599 in {handle.id} else 7
                    return self.linear(x) + value
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            for mode in ("dict", "set"):
                with self.subTest(mode=mode):
                    RemovableHandle.next_id = 600
                    mod = Mod(mode)
                    x = torch.randn(4, 4)
                    opt_mod = torch.compile(mod, backend="eager")

                    res0 = opt_mod(x)
                    RemovableHandle.next_id = 599
                    res1 = opt_mod(x)

                    self.assertEqual(res0, torch.full_like(res0, 7))
                    self.assertEqual(res1, torch.full_like(res1, 5))
                    self.assertEqual(RemovableHandle.next_id, 600)
                    self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_public_set_algebra_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self, mode) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.mode = mode
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    if self.mode == "mixed_literal":
                        value = len({handle.id, 601})
                    elif self.mode == "reverse_mixed_literal":
                        value = len({601, handle.id})
                    elif self.mode == "union":
                        value = len({handle.id} | {601})
                    elif self.mode == "difference":
                        value = len({handle.id} - {601})
                    elif self.mode == "reverse_difference":
                        value = len({601} - {handle.id})
                    elif self.mode == "inplace_union":
                        values = {handle.id}
                        values |= {601}
                        value = len(values)
                    elif self.mode == "add":
                        values = {601}
                        values.add(handle.id)
                        value = len(values)
                    else:
                        values = {handle.id}
                        values -= {601}
                        value = len(values)
                    return self.linear(x) + value
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            for mode, expected0, expected1 in (
                ("mixed_literal", 2, 1),
                ("reverse_mixed_literal", 2, 1),
                ("union", 2, 1),
                ("difference", 1, 0),
                ("reverse_difference", 1, 0),
                ("inplace_union", 2, 1),
                ("add", 2, 1),
                ("inplace_difference", 1, 0),
            ):
                with self.subTest(mode=mode):
                    RemovableHandle.next_id = 600
                    mod = Mod(mode)
                    x = torch.randn(4, 4)
                    opt_mod = torch.compile(mod, backend="eager")

                    res0 = opt_mod(x)
                    res1 = opt_mod(x)

                    self.assertEqual(res0, torch.full_like(res0, expected0))
                    self.assertEqual(res1, torch.full_like(res1, expected1))
                    self.assertEqual(RemovableHandle.next_id, 602)
                    self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_id_public_dict_algebra_uses_runtime_id(self):
        class Mod(torch.nn.Module):
            def __init__(self, mode) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.mode = mode
                with torch.no_grad():
                    self.linear.weight.zero_()
                    self.linear.bias.zero_()

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    if self.mode == "mixed_literal":
                        value = len({handle.id: 1, 601: 2})
                    elif self.mode == "reverse_mixed_literal":
                        value = len({601: 2, handle.id: 1})
                    elif self.mode == "union":
                        value = len({handle.id: 1} | {601: 2})
                    elif self.mode == "reverse_union":
                        value = len({601: 2} | {handle.id: 1})
                    elif self.mode == "inplace_union":
                        values = {handle.id: 1}
                        values |= {601: 2}
                        value = len(values)
                    elif self.mode == "update":
                        values = {handle.id: 1}
                        values.update({601: 2})
                        value = len(values)
                    elif self.mode == "dunder_init_dict":
                        values = {handle.id: 1}
                        values.__init__({601: 2})
                        value = len(values)
                    elif self.mode == "dunder_init_items":
                        values = {handle.id: 1}
                        values.__init__([(601, 2)])
                        value = len(values)
                    elif self.mode == "insert":
                        values = {601: 7}
                        values[handle.id] = 5
                        value = len(values)
                    elif self.mode == "insert_get":
                        values = {601: 7}
                        values[handle.id] = 5
                        value = values.get(601)
                    elif self.mode == "setdefault":
                        values = {601: 7}
                        values.setdefault(handle.id, 5)
                        value = len(values)
                    elif self.mode == "ordered_insert":
                        values = collections.OrderedDict([(601, 7)])
                        values[handle.id] = 5
                        value = len(values)
                    elif self.mode == "keys_and":
                        values = {handle.id: 1}
                        value = len(values.keys() & {601})
                    else:
                        values = {handle.id: 1}
                        value = len(values.keys() ^ {601})
                    return self.linear(x) + value
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            for mode, expected0, expected1 in (
                ("mixed_literal", 2, 1),
                ("reverse_mixed_literal", 2, 1),
                ("union", 2, 1),
                ("reverse_union", 2, 1),
                ("inplace_union", 2, 1),
                ("update", 2, 1),
                ("dunder_init_dict", 2, 1),
                ("dunder_init_items", 2, 1),
                ("insert", 2, 1),
                ("insert_get", 7, 5),
                ("setdefault", 2, 1),
                ("ordered_insert", 2, 1),
                ("keys_and", 0, 1),
                ("keys_xor", 2, 0),
            ):
                with self.subTest(mode=mode):
                    RemovableHandle.next_id = 600
                    mod = Mod(mode)
                    x = torch.randn(4, 4)
                    opt_mod = torch.compile(mod, backend="eager")

                    res0 = opt_mod(x)
                    res1 = opt_mod(x)

                    self.assertEqual(res0, torch.full_like(res0, expected0))
                    self.assertEqual(res1, torch.full_like(res1, expected1))
                    self.assertEqual(RemovableHandle.next_id, 602)
                    self.assertEqual(len(mod.linear._forward_hooks), 0)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_invalid_extra_dict_preserves_next_id_increment(self):
        def fn(x):
            try:
                RemovableHandle(collections.OrderedDict(), extra_dict={})
            except TypeError:
                pass
            return x + 1

        old_next_id = RemovableHandle.next_id
        try:
            RemovableHandle.next_id = 200
            cnts = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

            x = torch.ones(())
            self.assertEqual(opt_fn(x), x + 1)
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(RemovableHandle.next_id, 201)
        finally:
            RemovableHandle.next_id = old_next_id

    def test_removable_handle_next_id_read_after_allocation_graph_breaks(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return x + RemovableHandle.next_id
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id

        try:
            mod = Mod()
            opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "RemovableHandle.next_id"
            ):
                opt_mod(torch.ones(4, 4))
        finally:
            RemovableHandle.next_id = old_next_id

    @torch._dynamo.config.patch(replay_side_effects=False)
    def test_removable_handle_next_id_respects_replay_side_effects_false(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def forward_hook(module, args, output):
                    return output + 1

                handle = self.linear.register_forward_hook(forward_hook)
                try:
                    return self.linear(x)
                finally:
                    handle.remove()

        old_next_id = RemovableHandle.next_id
        try:
            mod = Mod()
            x = torch.randn(4, 4)
            expected = mod.linear(x) + 1
            cnts = torch._dynamo.testing.CompileCounter()
            opt_mod = torch.compile(mod, backend=cnts, fullgraph=True)

            self.assertEqual(opt_mod(x), expected)
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(RemovableHandle.next_id, old_next_id)
        finally:
            RemovableHandle.next_id = old_next_id

    @torch._dynamo.config.patch(wrap_top_frame=True)
    def test_wrap_top_frame_with_hooks(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = torch.nn.Linear(18, 18, bias=False)

            def forward(self, x):
                return self.net1(x)

        mod = ToyModel()
        mod.register_forward_pre_hook(lambda mod, input: input[0] + 1)

        # Case 1: torch.compile(mod)
        cnts = torch._dynamo.testing.CompileCounter()
        compiled_mod = torch.compile(mod, backend=cnts)

        x = torch.rand(18, 18)
        ref = mod(x)
        res = compiled_mod(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        # Case 2: mod.compile()
        cnts = torch._dynamo.testing.CompileCounter()
        mod.compile(backend=cnts)
        res = mod(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

    def test_global_module_forward_pre_hook(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                return x - 1

        counter = 0

        def hook(mod, args):
            nonlocal counter
            counter += 1
            return args

        x = torch.rand(18, 18)
        mod = Mod()
        compiled_mod = torch.compile(mod, backend="eager")

        try:
            hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(hook)
            ref = mod(x)
            self.assertEqual(counter, 1)
            with self.assertWarnsRegex(
                UserWarning,
                r"Using `torch.compile\(module\)` when there are global hooks.*",
            ):
                res = compiled_mod(x)
            self.assertEqual(counter, 3)
            self.assertEqual(ref, res)
        finally:
            hook_handle.remove()

    def test_register_hook_on_intermediate_stride_dependent(self):
        def hook(grad):
            if grad.is_contiguous():
                return grad.sin()
            else:
                return grad.cos()

        def fn(x):
            y = x * 2
            y.register_hook(hook)
            return y.sum()

        # Hooks that branch on grad metadata (e.g. is_contiguous)
        # graph break because grad properties are unknown at trace time.
        x = torch.randn(4, requires_grad=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_register_hook_on_intermediate_autograd_cache(self):
        from torch._dynamo.utils import counters

        def fn(x):
            y = x * 2
            y.register_hook(lambda g: g * 0.5)
            return y.sum()

        torch._functorch.config.enable_autograd_cache = True
        try:
            # First compile
            torch._dynamo.reset()
            counters.clear()
            x = torch.randn(4, device="cuda", requires_grad=True)
            torch.compile(fn, fullgraph=True)(x).backward()

            # Second compile (force recompile to test cache)
            torch._dynamo.reset()
            x2 = torch.randn(4, device="cuda", requires_grad=True)
            torch.compile(fn, fullgraph=True)(x2).backward()

            aot_counters = counters["aot_autograd"]
            self.assertEqual(aot_counters.get("autograd_cache_bypass", 0), 0)
        finally:
            torch._functorch.config.enable_autograd_cache = False

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_register_hook_on_intermediate_autograd_cache_different_hooks(self):
        from torch._dynamo.utils import counters

        def fn_a(x):
            y = x * 2
            y.register_hook(lambda g: g * 0.5)
            return y.sum()

        def fn_b(x):
            y = x * 2
            y.register_hook(lambda g: g * 3.0)
            return y.sum()

        torch._functorch.config.enable_autograd_cache = True
        try:
            torch._dynamo.reset()
            counters.clear()

            # Compile fn_a
            x = torch.randn(4, device="cuda", requires_grad=True)
            torch.compile(fn_a, fullgraph=True)(x).backward()

            # Compile fn_b (different hook — must NOT cache hit from fn_a)
            x2 = torch.randn(4, device="cuda", requires_grad=True)
            torch.compile(fn_b, fullgraph=True)(x2).backward()

            # fn_b should give grad = 2 * 3.0 = 6.0, not 2 * 0.5 = 1.0
            self.assertEqual(x2.grad, torch.tensor([6.0] * 4, device="cuda"))
            self.assertEqual(
                counters["aot_autograd"].get("autograd_cache_bypass", 0), 0
            )
        finally:
            torch._functorch.config.enable_autograd_cache = False


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
