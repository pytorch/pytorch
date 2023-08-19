import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing


class HooksTests(torch._dynamo.test_case.TestCase):
    def test_tensor_only_register_hook_in_graph_lambda(self):
        def fn(x):
            v.register_hook(lambda grad: grad * 2)  # double the gradient
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
            v.register_hook(lambda grad: grad * 2)  # double the gradient
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
            handle = v.register_hook(lambda grad: grad * 2)
            z = z * z
            handle.remove()
            v.register_hook(lambda grad: grad * 3)
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([3.0, 6.0, 9.0]))
        self.assertEqual(cnts.frame_count, 2)

    def test_tensor_only_register_hook_in_graph_local(self):
        def local_hook(grad):
            return grad * 2

        def fn(x):
            v.register_hook(local_hook)  # double the gradient
            return x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 0)

    def test_tensor_register_hook_in_graph_local(self):
        def local_hook(grad):
            return grad * 2

        def fn(x, y, z):
            v.register_hook(local_hook)  # double the gradient
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
            handle = v.register_hook(local_hook)
            z = z * z
            handle.remove()
            v.register_hook(local_hook2)
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch._dynamo.optimize(cnts)(fn)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(v.grad, torch.tensor([3.0, 6.0, 9.0]))
