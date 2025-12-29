# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo
from torch._dynamo.testing import same
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMatchClassBackends(TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def _test_backend(self, backend):
        pass

    def test_eager_backend(self):
        @torch.compile(backend="eager")
        def fn(x):
            match x:
                case torch.Tensor():
                    return x * 2
                case _:
                    return torch.zeros(1)

        x = torch.randn(4)
        self.assertTrue(same(fn(x), x * 2))

    def test_aot_eager_backend(self):
        @torch.compile(backend="aot_eager")
        def fn(x):
            match x:
                case torch.Tensor():
                    return x * 2
                case _:
                    return torch.zeros(1)

        x = torch.randn(4)
        self.assertTrue(same(fn(x), x * 2))

    def test_inductor_backend(self):
        @torch.compile(backend="inductor")
        def fn(x):
            match x:
                case torch.Tensor():
                    return x * 2
                case _:
                    return torch.zeros(1)

        x = torch.randn(4)
        self.assertTrue(same(fn(x), x * 2))

    def test_nn_module_match(self):
        class MyMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

        @torch.compile(backend="inductor")
        def fn(mod):
            match mod:
                case torch.nn.Module():
                    return "is_module"
                case _:
                    return "not_module"

        mod = MyMod()
        self.assertEqual(fn(mod), "is_module")


if __name__ == "__main__":
    run_tests()
