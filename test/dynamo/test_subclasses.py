# Owner(s): ["module: dynamo"]
import contextlib
import functools

import torch
import torch._C

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.testing import normalize_gm
from torch._functorch.aot_autograd import to_fun
from torch._higher_order_ops.wrap import wrap

from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
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


class EagerRecordGraphAndInputs:
    def __init__(self):
        self.graphs = []
        self.example_inputs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.graphs.append(gm)
        self.example_inputs.append(example_inputs)
        return gm


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

    def test_compile_with_fake_tensor_dynamic_dim(self):
        x = torch.randn([3, 4])

        def f(x):
            return torch.sin(x)

        def test_dynamic_dim(f, x, dim_dynamic, exp_frame_count, exp_op_count):
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounter()

            opt_f = torch.compile(f, backend=cnt, fullgraph=True)

            x1 = torch.rand_like(x)
            f(x)
            f(torch.randn([4, 3]))
            shape_env = ShapeEnv()
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                x_fake = fake_mode.from_tensor(
                    x, dynamic_dims=[dim_dynamic for i in range(x.dim())]
                )
                x1_fake = fake_mode.from_tensor(
                    x1, dynamic_dims=[dim_dynamic for i in range(x.dim())]
                )
                opt_f(x_fake)
                opt_f(x1_fake)

            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)

        test_dynamic_dim(f, x, DimDynamic.DYNAMIC, 1, 1)
        test_dynamic_dim(f, x, DimDynamic.DUCK, 1, 1)
        test_dynamic_dim(f, x, DimDynamic.STATIC, 1, 1)

    def test_compile_with_fake_tensor_automatic_dynamic(self):
        def f(x):
            return torch.sin(x)

        def test_automatic_dynamic(f, inps, dim_dynamic, exp_frame_count, exp_op_count):
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounter()
            opt_f = torch.compile(f, backend=cnt, fullgraph=True)

            shape_env = ShapeEnv()
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                for inp in inps:
                    fake_inp = fake_mode.from_tensor(
                        inp, dynamic_dims=[dim_dynamic for i in range(x.dim())]
                    )
                    opt_f(fake_inp)
            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)

        x = torch.randn([3, 4])
        y = torch.randn([4, 5])
        z = torch.randn([5, 6])
        a = torch.randn([3, 5])
        b = torch.randn([4, 4])
        for dim_dynamic in [DimDynamic.DYNAMIC, DimDynamic.DUCK, DimDynamic.STATIC]:
            # Recompile once, first with dim 0 and 1 become Dynamic
            test_automatic_dynamic(f, [x, y, z], dim_dynamic, 2, 2)
            # Recompile 2 times, first with dim 1 become Dynamic, second with dim 0 becomes Dynamic.
            test_automatic_dynamic(f, [x, a, z], dim_dynamic, 3, 3)
            # Recompile 2 times, first with dim 0 become Dynamic, second with dim 1 becomes Dynamic.
            test_automatic_dynamic(f, [x, b, z], dim_dynamic, 3, 3)

    def test_compile_with_functionalization(self):
        x = torch.randn([3, 4])
        x_clone = x.clone()
        x_clone2 = x.clone()
        backend = EagerRecordGraphAndInputs()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return x.add_(1.0) + torch.nn.functional.relu_(x)

        f_out = f(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.example_inputs), 1)

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        add_ = l_x_.add_(1.0)
        relu_ = torch.relu_(l_x_);  l_x_ = None
        add = add_ + relu_;  add_ = relu_ = None
        return (add,)
"""
        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        self.assertEqual(actual, expected)

        ff = torch.func.functionalize(f)
        ff_out = ff(x_clone)

        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 6)
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(len(backend.example_inputs), 2)
        actual = normalize_gm(backend.graphs[1].print_readable(print_output=False))
        self.assertEqual(actual, expected)
        self.assertTrue(torch._is_functional_tensor(backend.example_inputs[1][0]))

        def aot_f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(to_fun, args)
                    func_kwargs = pytree.tree_map(to_fun, kwargs)
                    return func(*func_args, **func_kwargs)
                finally:
                    torch._disable_functionalization()

            return wrapper

        aot_ff = aot_f_wrapper(f)
        aot_ff_out = aot_ff(x_clone2)

        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 9)
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(len(backend.example_inputs), 3)
        actual = normalize_gm(backend.graphs[2].print_readable(print_output=False))
        self.assertEqual(actual, expected)
        self.assertTrue(torch._is_functional_tensor(backend.example_inputs[1][0]))

        self.assertEqual(f_out, ff_out)
        self.assertEqual(f_out, aot_ff_out)

        try:
            torch._enable_functionalization(reapply_views=False)
            xf = pytree.tree_map(to_fun, x)
            x_view = xf.t()
            with self.assertRaisesRegex(RuntimeError, "Cannot safely fakify a view"):
                f(x_view)
        finally:
            torch._disable_functionalization()

    def test_compile_higher_order_with_functionalization(self):
        backend = EagerRecordGraphAndInputs()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return wrap(lambda x: x.add_(1.0), x)

        def check_count_and_graph(
            exp_frame_count, exp_op_count, exp_n_graph, exp_graph
        ):
            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)
            self.assertEqual(len(backend.graphs), exp_n_graph)
            actual = normalize_gm(
                backend.graphs[exp_n_graph - 1].print_readable(print_output=False)
            )
            self.assertExpectedInline(actual, exp_graph)

        t = torch.randn([3, 4])
        t_clone = t.clone()
        t_clone2 = t.clone()
        f(t)

        expected_graph = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        return (wrap,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            add_ = l_x_.add_(1.0);  l_x_ = None
            return add_
"""
        check_count_and_graph(1, 1, 1, expected_graph)

        ff = torch.func.functionalize(f)
        ff_out = ff(t_clone)
        # frame count and op count are incremented due to re-compilation
        check_count_and_graph(2, 2, 2, expected_graph)

        try:
            x = torch._to_functional_tensor(t_clone2, mirror_autograd_meta=True)
            torch._enable_functionalization(reapply_views=False)
            aot_f_out = f(x)
        finally:
            torch._disable_functionalization()

        # frame count and op count are incremented due to re-compilation
        check_count_and_graph(3, 3, 3, expected_graph)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
