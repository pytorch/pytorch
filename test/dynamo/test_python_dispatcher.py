# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo.test_case
from torch._dispatch.python import enable_python_dispatcher


class PythonDispatcherTests(torch._dynamo.test_case.TestCase):
    def test_dispatch_key(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            key_set = torch._C._dispatch_tls_local_include_set()
            key_set = key_set | torch._C._dispatch_keys(x)
            key_set = key_set - torch._C._dispatch_tls_local_exclude_set()
            if key_set.highestPriorityTypeId() == torch.DispatchKey.PythonDispatcher:
                return torch.sin(x + 1)
            else:
                return torch.sin(x - 1)

        x = torch.randn(2, 3)
        with enable_python_dispatcher():
            self.assertEqual(fn(x), torch.sin(x + 1))

    def test_functorch_interpreter(self):
        def square_and_add(x, y):
            interpreter = (
                torch._functorch.pyfunctorch.retrieve_current_functorch_interpreter()
            )
            level = interpreter.level()
            if interpreter.key() == torch._C._functorch.TransformType.Vmap:
                return (x**2 + y) * level
            else:
                return x**2 * level

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, y):
            return torch.vmap(square_and_add)(x, y)

        x = torch.tensor([1, 2, 3, 4])
        y = torch.tensor([10, 20, 30, 40])
        self.assertEqual(fn(x, y), torch.tensor([11, 24, 39, 56]))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
