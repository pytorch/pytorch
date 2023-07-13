# Owner(s): ["module: dynamo"]
import contextlib

import torch
import torch._C

import torch._dynamo.test_case
import torch._functorch.config
import torch.utils.checkpoint
from torch.utils._pytree import tree_map_only


class PassthroughLeftAddSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == torch.add:
            return args[0]

        return super().__torch_function__(func, types, args, kwargs)


class PassthroughRightAddSubclassLeft(PassthroughLeftAddSubclass):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == torch.add:
            return args[1]

        return super().__torch_function__(func, types, args, kwargs)


class PassthroughRightAddSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == torch.add:
            return args[1]

        return super().__torch_function__(func, types, args, kwargs)


class PassthroughMulSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == torch.mul:
            return args[0]

        return super().__torch_function__(func, types, args, kwargs)


class WrapperSubclass:
    def __init__(self, tensor):
        self.tensor = tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = tree_map_only(WrapperSubclass, lambda x: x.tensor, args)

        return func(*args, **kwargs)


class MockSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)


GLOBAL_TEST_SUBCLASSES = {
    PassthroughLeftAddSubclass,
    PassthroughRightAddSubclass,
    PassthroughRightAddSubclassLeft,
    PassthroughMulSubclass,
    WrapperSubclass,
    MockSubclass,
}
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

    def test_multi_subclass_dispatch_notimpl(self):
        @compile_full_eager
        def fn(x, y, z):
            return torch.sqrt(z), torch.div(x, y)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "returned NotImplemented"
        ):
            input0 = torch.ones(2, 2).as_subclass(PassthroughLeftAddSubclass)
            input1 = torch.ones(2, 2).as_subclass(PassthroughMulSubclass)
            fn(input0, input1, torch.ones(2, 2))

    def test_multi_subclass_dispatch_subclass_tiebreak(self):
        @compile_full_eager
        def fn(x, y, z):
            return torch.sqrt(z), torch.add(x, y)

        input0 = torch.ones(2, 2).as_subclass(PassthroughLeftAddSubclass)
        input1 = torch.zeros(2, 2).as_subclass(PassthroughRightAddSubclass)

        _, res = fn(input0, input1, torch.ones(2, 2))

        self.assertEqual(res, input0)

    def test_multi_subclass_dispatch_ordering_tiebreak(self):
        @compile_full_eager
        def fn(x, y, z):
            return torch.sqrt(z), torch.add(x, y)

        input0 = torch.ones(2, 2).as_subclass(PassthroughLeftAddSubclass)
        input1 = torch.zeros(2, 2).as_subclass(PassthroughRightAddSubclassLeft)

        _, res = fn(input0, input1, torch.ones(2, 2))

        self.assertEqual(res, input1)

    def test_multi_subclass_dispatch_first_notimpl(self):
        @compile_full_eager
        def fn(x, y, z):
            return torch.sqrt(z), torch.add(x, y)

        input0 = torch.ones(2, 2).as_subclass(PassthroughMulSubclass)
        input1 = torch.zeros(2, 2).as_subclass(PassthroughLeftAddSubclass)

        _, res = fn(input0, input1, torch.ones(2, 2))

        self.assertEqual(res, input0)

    def test_torch_function_trace(self):
        def fn(x, y):
            return torch.sqrt(y), torch.add(x, 10.0)

        fn_opt = compile_full_eager(fn)

        input = torch.ones(2, 2).as_subclass(PassthroughLeftAddSubclass)
        _, res_exp = fn(input, torch.ones(2, 2))
        _, res_act = fn_opt(input, torch.ones(2, 2))

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_act, torch.ones(2, 2))

    def test_torch_function_trace_other_arg_positions(self):
        def fn(x, y):
            return torch.sqrt(y), torch.add(torch.ones(3, 3), x)

        fn_opt = compile_full_eager(fn)

        input = torch.ones(2, 2).as_subclass(PassthroughLeftAddSubclass)
        _, res_exp = fn(input, torch.ones(2, 2))
        _, res_act = fn_opt(input, torch.ones(2, 2))

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_act, torch.ones(3, 3))

    def test_unwrap_redispatch(self):
        pass

    # For example, calling + on tensor subclass
    # should trigger torch function tracing
    def test_builtin_torch_function_trigger(self):
        pass

    def test_disable_torch_function_context(self):
        import logging

        torch._logging.set_logs(dynamo=logging.DEBUG)

        @compile_full_eager
        def fn(x, y, z):
            with torch._C.DisableTorchFunctionSubclass(), torch.no_grad():
                return torch.sqrt(z), torch.add(x, y)

        input0 = torch.ones(2, 2)
        input1 = torch.ones(2, 2).as_subclass(PassthroughLeftAddSubclass)

        _, res = fn(input0, input1, torch.ones(2, 2))

        with torch._C.DisableTorchFunctionSubclass():
            exp = torch.add(input0, input1)

        self.assertEqual(exp, res)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
