# Owner(s): ["module: dynamo"]

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
    return torch._dynamo.optimize("inductor", nopython=True, dynamic=True)(gm)


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
        fn = torch._dynamo.optimize(cnts)(fn)
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
        fn = torch._dynamo.optimize(cnts)(fn)
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
        fn = torch._dynamo.optimize(cnts)(fn)
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
        fn = torch._dynamo.optimize(cnts)(fn)
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
            h2 = handle
            z = z * z
            return x, y * y, z, handle, handle

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)
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
        fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
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
        fn = torch._dynamo.optimize(cnts)(fn)
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
        fn = torch._dynamo.optimize(cnts)(fn)
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
        fn = torch._dynamo.optimize(cnts)(fn)
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
        fn = torch._dynamo.optimize(cnts)(fn)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(v.grad, torch.tensor([3.0, 6.0, 9.0]))

    def test_tensor_register_global_hook(self):
        def fn(x):
            x.register_hook(global_hook_0)
            return x, x * x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)
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
        fn = torch._dynamo.optimize(cnts)(fn)
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
        fn = torch._dynamo.optimize(cnts)(fn)
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
        fn = torch._dynamo.optimize(cnts)(fn)
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

    def test_intermediary_hooks(self):
        # Graph breaks because compiled_autograd is not set
        def simple_hook(g):
            return g * 2

        def f(x):
            y = x + 1
            y.register_hook(simple_hook)
            z = y + 1
            return z

        out = torch.randn(1, requires_grad=True)
        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts, nopython=False)(f)
        res = fn(out)
        res.backward()
        self.assertEqual(res, f(out))
        self.assertEqual(cnts.frame_count, 2)
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
        with compiled_autograd.enable(compiler_fn):
            dynamo_out = torch._dynamo.optimize("aot_eager", nopython=True)(mod)(x2)
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
            dynamo_out = torch._dynamo.optimize(backend, nopython=True)(mod)(x2)
            with compiled_autograd.enable(compiler_fn):
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
        with compiled_autograd.enable(compiler_fn):
            dynamo_out = torch._dynamo.optimize("inductor", nopython=True)(mod)(x2)
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
        with compiled_autograd.enable(compiler_fn):
            dynamo_out = torch._dynamo.optimize("inductor", nopython=True)(mod)(x2, obj)
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
        with compiled_autograd.enable(compiler_fn):
            dynamo_out = torch._dynamo.optimize("inductor", nopython=True)(mod)(x2, obj)
            with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "builtin: str"):
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

        with compiled_autograd.enable(compiler_fn):
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

        with compiled_autograd.enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

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

        with compiled_autograd.enable(
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

        with compiled_autograd.enable(
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
        with compiled_autograd.enable(compiler_fn):
            cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            comp_mod = torch._dynamo.optimize(cnts, nopython=True)(mod)
            comp_out = comp_mod(x1)
            comp_out[0].backward(torch.ones(4))

            self.assertEqual(cnts.frame_count, 1)
            my_hook = my_hook2  # noqa: F811
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

        with compiled_autograd.enable(torch.compile(backend="eager", fullgraph=True)):
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
            # X goes to x*2 becaue of mul_
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
                compiled_fn = torch._dynamo.optimize(cnts, nopython=True)(reg_and_mul)

                compiled_bwd_ctx = (
                    compiled_autograd.enable(
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
        with compiled_autograd.enable(compiler_fn):
            for i in range(5):
                with unittest.mock.patch(
                    "torch._dynamo.config.error_on_recompile", True
                ):
                    # Mimic optimizer.zero_grad() to clear the gradient
                    x.grad = None
                    run(i).sum().backward()

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    def test_no_recompile_on_same_hook(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fw_hook(inp):
            return (inp[0] + 1,)

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList()
                for i in range(10):
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
