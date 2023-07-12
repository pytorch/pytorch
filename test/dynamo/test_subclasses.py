# Owner(s): ["module: dynamo"]
import contextlib

import torch
import torch._C

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.utils.checkpoint
from torch.utils._pytree import tree_map_only


class PassthroughAddSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == torch.add:
            return args[0]

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
    PassthroughAddSubclass,
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
        import logging

        torch._logging.set_logs(dynamo=logging.DEBUG)

        def fn(x, y):
            z = torch.add(x, y)
            return torch.div(x, z)

        fn_opt = compile_full_eager(fn)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "returned NotImplemented"
        ):
            input0 = torch.ones(2, 2).as_subclass(PassthroughAddSubclass)
            input1 = torch.ones(2, 2).as_subclass(PassthroughMulSubclass)
            fn_opt(input0, input1)

    def test_multi_subclass_dispatch_subclass_tiebreak(self):
        pass

    def test_multi_subclass_dispatch_ordering_tiebreak(self):
        pass

    def test_torch_function_trace(self):
        def fn(x):
            return torch.add(x, 10.0)

        fn_opt = compile_full_eager(fn)

        input = torch.ones(2, 2).as_subclass(PassthroughAddSubclass)
        res_exp = fn(input)
        res_act = fn_opt(input)

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_act, torch.ones(2, 2))

    def test_torch_function_trace_other_arg_positions(self):
        def fn(x):
            return torch.add(torch.ones(3, 3), x)

        fn_opt = compile_full_eager(fn)

        input = torch.ones(2, 2).as_subclass(PassthroughAddSubclass)
        res_exp = fn(input)
        res_act = fn_opt(input)

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_act, torch.ones(3, 3))

    def test_unwrap_redispatch(self):
        pass

    # For example, calling + on tensor subclass
    # should trigger torch function tracing
    def test_builtin_torch_function_trigger(self):
        pass

    def test_disable_torch_function_context(self):
        pass

    def test_disable_torch_function_context_recompile(self):
        pass

    def test_enable_torch_function_context_recompile(self):
        pass


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
