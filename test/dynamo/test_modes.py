# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.utils._python_dispatch import TorchDispatchMode


class TorchDispatchModeTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_skip_torch_dispatch_modes(self):
        class RewriteAddToMul(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is torch.ops.aten.add.Tensor:
                    func = torch.ops.aten.mul.Tensor
                return func(*args, **kwargs)

        def fn(x):
            return x + x

        cnt = torch._dynamo.testing.CompileCounter()

        x = torch.tensor([3.0])
        with RewriteAddToMul():
            eager_res = fn(x)
            compiled_res = torch._dynamo.optimize(cnt)(fn)(x)

        self.assertEqual(eager_res, compiled_res)
        self.assertEqual(cnt.frame_count, 0)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
