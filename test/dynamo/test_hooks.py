# Owner(s): ["module: dynamo"]

import functools

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from functorch.compile import nop
from torch._functorch.aot_autograd import aot_module_simplified


def global_hook_0(grad):
    return grad * 4


def global_hook_1(grad):
    return grad / 2


def global_hook_2(grad):
    return grad * 3


h0 = None


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
        self.assertEqual(cnts.frame_count, 2)

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
        self.assertNotEqual(h, None)
        self.assertNotEqual(h2, None)
        self.assertEqual(h2, h)

    def test_tensor_register_hook_repeated_handle_not_local(self):
        def fn(x, y, z, mod):
            mod.handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)
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
        # Will be 1 when we support hooks on intermediaries
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
        dynamo_out = torch._dynamo.optimize("aot_eager")(mod)(x2)
        dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(dynamo_out, aot_out)
        self.assertEqual(dynamo_out, eager_out)

        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(x0.grad, x2.grad)

    def test_intermediary_hooks_same_on_inductor(self):
        def my_hook(grad, k=0):
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
        dynamo_out = torch._dynamo.optimize("inductor")(mod)(x2)
        dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(dynamo_out, aot_out)
        self.assertEqual(dynamo_out, eager_out)

        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(x0.grad, x2.grad)

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
        cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        comp_mod = torch._dynamo.optimize(cnts)(mod)
        comp_out = comp_mod(x1)
        comp_out[0].backward(torch.ones(4))

        # Will be 1 when we support hooks on intermediaries
        self.assertEqual(cnts.frame_count, 2)
        my_hook = my_hook2  # noqa: F811
        self.assertEqual(x0.grad, x1.grad)

        eager_out = mod(x0)
        eager_out[0].backward(torch.ones(4))

        comp_out = comp_mod(x1)

        # Will be 2 when we support hooks on intermediaries
        self.assertEqual(cnts.frame_count, 4)
        comp_out[0].backward(torch.ones(4))
        self.assertEqual(x0.grad, x1.grad)
