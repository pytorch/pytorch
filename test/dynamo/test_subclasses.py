# Owner(s): ["module: dynamo"]
import contextlib

import torch
import torch._C

import torch._dynamo.test_case
import torch._functorch.config
import torch.utils.checkpoint
from torch.utils._pytree import tree_map_only


class Add1Subclass(torch.Tensor):
    __slots__ = ["elem"]

    def __init__(self, elem: torch.Tensor):
        super().__init__()
        self.elem = elem

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        def add1(x):
            return torch.add(x.elem, 1.0)

        args = tree_map_only(cls, add1, args)
        return func(*args, **kwargs)


class Add2Subclass(torch.Tensor):
    pass


class MockSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


GLOBAL_TEST_SUBCLASSES = {Add1Subclass, Add2Subclass, MockSubclass}

compile_full_eager = torch.compile(backend="eager", fullgraph=True)


@contextlib.contextmanager
def preserve_subclass_config():
    old_subclass_set = set(torch._dynamo.config.traceable_tensor_subclasses)
    try:
        torch._dynamo.config.traceable_tensor_subclasses.update(GLOBAL_TEST_SUBCLASSES)
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
        @compile_full_eager
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

        @compile_full_eager
        def fn(x):
            return LocalSubclass(torch.add(x, 1.0))

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, LocalSubclass)

    def test_unique_multi_subclass_dispatch(self):
        pass

    def test_torch_function_trace(self):
        def fn(x):
            return torch.add(x, 1.0)

        fn_opt = compile_full_eager(fn)

        input = Add1Subclass(torch.ones(2, 2))
        res_exp = fn(input)
        res_act = fn_opt(input)

        self.assertEqual(res_exp, res_act)

    def test_torch_function_trace_other_arg_positions(self):
        def fn(x):
            return torch.div(1.0, x)

        fn_opt = compile_full_eager(fn)

        input = torch.ones(2, 2).as_subclass(Add1Subclass)
        res_exp = fn(input)
        res_act = fn_opt(input)

        self.assertEqual(res_exp, res_act)

    def test_unwrap_redispatch(self):
        pass

    # For example, calling + on tensor subclass
    # should trigger torch function tracing
    def test_builtin_torch_function_trigger(self):
        pass

    def test_disable_torch_function_context(self):
        pass


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
