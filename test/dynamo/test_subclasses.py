# Owner(s): ["module: dynamo"]
import contextlib

import torch

import torch._dynamo.test_case
import torch._functorch.config
import torch.utils.checkpoint


class MockSubclass(torch.Tensor):
    def __repr__(self):
        return f"MockSubclass({self.elem})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
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

    def test_return_subclass(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return MockSubclass(torch.add(x, 1.0))

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, MockSubclass)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
