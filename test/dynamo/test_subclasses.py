# Owner(s): ["module: dynamo"]
import contextlib

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.utils.checkpoint


class MockSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


@contextlib.contextmanager
def preserve_subclass_config():
    old_subclass_set = set(torch._dynamo.config.traceable_tensor_subclasses)
    try:
        torch._dynamo.config.traceable_tensor_subclasses.add(MockSubclass)
        yield
    finally:
        torch._dynamo.config.traceable_tensor_subclasses.clear()
        torch._dynamo.config.traceable_tensor_subclasses.update(old_subclass_set)


class SubclassTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(preserve_subclass_config())

    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()

    def test_torch_function_state_graph_break(self):
        @torch.compile(backend="eager")
        def fn(x):
            with torch._C.DisableTorchFunctionSubclass():
                torch._dynamo.graph_break()
                return torch._C._is_torch_function_enabled(), torch.add(x, 1.0)

        input = torch.ones(2, 2)
        res, _ = fn(input)
        self.assertFalse(res)

    def test_torch_function_state_tracing(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            with torch._C.DisableTorchFunctionSubclass():
                torch.add(x, 1.0)

        input = torch.ones(2, 2)

        res = fn(input)

    def test_torch_function_state_guards(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            torch.add(x, 1.0)

        input = torch.ones(2, 2)

        with torch._C.DisableTorchFunctionSubclass():
            res = fn(input)

        res = fn(input)

        self.assertEqual(cnt.frame_count, 2)

    def test_return_subclass(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return MockSubclass(torch.add(x, 1.0))

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, MockSubclass)

    def test_return_local_subclass(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        torch._dynamo.config.traceable_tensor_subclasses.add(LocalSubclass)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return LocalSubclass(torch.add(x, 1.0))

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, LocalSubclass)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
