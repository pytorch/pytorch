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

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
