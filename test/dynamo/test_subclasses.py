# Owner(s): ["module: dynamo"]
import functools
import itertools
import unittest

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.testing import normalize_gm
from torch._higher_order_ops.wrap import wrap

from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.nested._internal.nested_tensor import jagged_from_list, ViewBufferFromNested
from torch.testing._internal.inductor_utils import HAS_CUDA


requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")

compile_full_eager = torch.compile(backend="eager", fullgraph=True)


class MockSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


class DummyNDim(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == torch.Tensor.ndim.__get__:
            return 10

        return super().__torch_function__(func, types, args, kwargs)


class SigmoidToExpSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == torch.Tensor.sigmoid:
            return super().__torch_function__(torch.Tensor.exp, types, args, kwargs)

        return super().__torch_function__(func, types, args, kwargs)


class EagerRecordGraphAndInputs:
    def __init__(self):
        self.graphs = []
        self.example_inputs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.graphs.append(gm)
        self.example_inputs.append(example_inputs)
        return gm


GLOBAL_TEST_SUBCLASSES = {MockSubclass, DummyNDim, SigmoidToExpSubclass}


class SubclassTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch(
                "traceable_tensor_subclasses", GLOBAL_TEST_SUBCLASSES
            )
        )

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

    def test_torch_function_state_nested(self):
        @torch.compile(backend="eager")
        def fn(x):
            with torch._C.DisableTorchFunctionSubclass():
                with torch._C.DisableTorchFunctionSubclass():
                    x = x + 1
                # Should reset to the outer state (disabled) after exiting ctx manager
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

        with torch._dynamo.config.patch("traceable_tensor_subclasses", {LocalSubclass}):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                return LocalSubclass(torch.add(x, 1.0))

            input = torch.ones(2, 2)

            res = fn(input)
            self.assertIsInstance(res, LocalSubclass)

    def test_torch_function_call_on_method(self):
        x = torch.ones(2, 2)
        y = torch.ones(2, 2)
        z = torch.ones(2, 2)
        wrapped = x.as_subclass(SigmoidToExpSubclass)
        wrapped2 = y.as_subclass(SigmoidToExpSubclass)

        def fn(w):
            return w.sigmoid()

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped2)
        res_exp2 = z.exp()

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_exp, res_exp2)

    def test_user_overidden_method_unsupported(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

            def sigmoid(self):
                return None

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            x.sigmoid()

        msg = (
            "Accessing overridden method/attribute sigmoid on a tensor"
            " subclass with a __torch_function__ override is not supported"
        )
        with torch._dynamo.config.patch(
            "traceable_tensor_subclasses", {LocalSubclass}
        ), self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

    def test_user_overidden_attr_unsupported(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

            ndim = 10

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.ndim

        msg = (
            "Accessing overridden method/attribute ndim on a tensor"
            " subclass with a __torch_function__ override is not supported"
        )
        with torch._dynamo.config.patch(
            "traceable_tensor_subclasses", {LocalSubclass}
        ), self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

    def test_user_overidden_property_unsupported(self):
        class LocalSubclass(torch.Tensor):
            def __init__(self):
                self._ndim = 10

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

            @property
            def ndim(self):
                return self._ndim

            @ndim.setter
            def ndim(self, value):
                self._ndim = value

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.ndim

        msg = (
            "Accessing overridden method/attribute ndim on a tensor"
            " subclass with a __torch_function__ override is not supported"
        )
        with torch._dynamo.config.patch(
            "traceable_tensor_subclasses", {LocalSubclass}
        ), self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

    def test_overridden_method_guarding(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.sigmoid()

        with torch._dynamo.config.patch(
            error_on_recompile=True, traceable_tensor_subclasses={LocalSubclass}
        ):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)
            fn(x)
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

        with torch._dynamo.config.patch(
            traceable_tensor_subclasses={LocalSubclass}
        ), self.assertRaisesRegex(
            TypeError,
            "'bool' object is not callable",
        ):
            LocalSubclass.sigmoid = False
            fn(x)

    def test_torch_function_call_on_attr(self):
        x = torch.ones(2, 2)
        wrapped = x.as_subclass(DummyNDim)

        def fn(w):
            return w.ndim + torch.ones(2)

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_exp, torch.ones(2) + 10)

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
        # When inputs' DimDynamic is DYNAMIC or DUCK, the inputs
        # to opt_f will be tensors with SymInt sizes. Dynamo will treat input
        # as dynamic automatically and will only compile once
        for dim_dynamic in [DimDynamic.DYNAMIC, DimDynamic.DUCK]:
            test_automatic_dynamic(f, [x, y, z], dim_dynamic, 1, 1)
            test_automatic_dynamic(f, [x, a, z], dim_dynamic, 1, 1)
            test_automatic_dynamic(f, [x, b, z], dim_dynamic, 1, 1)

        for dim_dynamic in [DimDynamic.STATIC]:
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

        # Cannot re-use the version from AOTAutograd, since that uses python functional tensors.
        def to_fun(x):
            x_functional = torch._to_functional_tensor(x)
            torch._mirror_autograd_meta_to(x, x_functional)
            return x_functional

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
        getitem = wrap[0];  wrap = None
        return (getitem,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            add_ = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
"""
        check_count_and_graph(1, 2, 1, expected_graph)

        ff = torch.func.functionalize(f)
        ff_out = ff(t_clone)
        # frame count and op count are incremented due to re-compilation
        check_count_and_graph(2, 4, 2, expected_graph)

        try:
            x = torch._to_functional_tensor(t_clone2)
            torch._mirror_autograd_meta_to(t_clone2, x)
            torch._enable_functionalization(reapply_views=False)
            aot_f_out = f(x)
        finally:
            torch._disable_functionalization()

        # frame count and op count are incremented due to re-compilation
        check_count_and_graph(3, 6, 3, expected_graph)

    def test_has_torch_function(self):
        class MyTensor:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                if func is torch.max:
                    return torch.tensor(123)
                return func(*args, **kwargs)

        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        def fn(x):
            return torch.overrides.has_torch_function_unary(
                x
            ), torch.overrides.has_torch_function_variadic(x)

        for test_class in [MyTensor, LocalSubclass]:
            x = test_class()
            ref0 = fn(x)
            ref1 = fn(4)
            opt_fn = torch._dynamo.optimize("eager")(fn)
            res0 = opt_fn(x)
            res1 = opt_fn(4)
            self.assertEqual(ref0, res0)
            self.assertEqual(ref1, res1)

    def test_wrapper_subclass_guards_on_inner_tensor(self):
        # Holds an inner tensor, that has a distinct shape from the outer wrapper tensor.
        # Also adds additional guards on the inner tensor's sizes.
        # When the first input to an op has x.shape[0] > 5, we insert an extra add node.
        class DoubleSizeMaybeAddGeThreeTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, inner):
                # Double the outer-most dimension
                outer_shape = (inner.shape[0] * 2,) + inner.shape[1:]
                return torch.Tensor._make_wrapper_subclass(
                    # TODO: right now, _make_wrapper_subclass's dynamic shape interaction is not great.
                    # Calling the overload that has kwargs causes us to go down the first overload path,
                    # which will **always** specialize sizes.
                    # We should probably eventually fix this so that the first overload can just handle dynamic shapes.
                    cls,
                    outer_shape,
                    inner.stride(),
                    None,
                    None,
                    inner.dtype,
                    inner.layout,
                    inner.device,
                    False,
                    inner.requires_grad,
                )

            def __init__(self, inner):
                self.inner_elem = inner

            def __tensor_flatten__(self):
                return ["inner_elem"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, _):
                return DoubleSizeMaybeAddGeThreeTensor(inner_tensors["inner_elem"])

            def __repr__(self):
                return f"DoubleSizeMayberAddGeThreeTensor({repr(self.inner_elem)})"

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                args_inner = torch.utils._pytree.tree_map_only(
                    DoubleSizeMaybeAddGeThreeTensor, lambda x: x.inner_elem, args
                )
                out_inner = func(*args_inner, **kwargs)

                # Add guards on the  inner tensor's sizes
                if args_inner[0].shape[0] > 3:
                    out_inner += 2

                return DoubleSizeMaybeAddGeThreeTensor(out_inner)

        lower_bound_str = None
        upper_bound_str = None
        curr_var_to_val = None
        curr_var_to_sources = None

        def backend(gm, args):
            print(gm.code)
            context = torch._guards.TracingContext.get()
            val_to_guards = list(context.fake_mode.shape_env.var_to_guards.values())

            # Grab info on sources and guards from the shapeenv
            nonlocal lower_bound_str
            nonlocal upper_bound_str
            nonlocal curr_var_to_val
            nonlocal curr_var_to_sources

            lower_bound_str = str(val_to_guards[0][0].expr)
            upper_bound_str = str(val_to_guards[0][1].expr)
            curr_var_to_val = {
                str(k): v for k, v in context.fake_mode.shape_env.var_to_val.items()
            }
            curr_var_to_sources = {
                str(k): v[0].name()
                for k, v in context.fake_mode.shape_env.var_to_sources.items()
            }
            return gm

        @torch.compile(backend=backend)
        def fn(x):
            if x.shape[0] < 10:
                return torch.mul(x, x)
            else:
                return torch.div(x, x)

        inp = torch.ones(4, 4)

        x = DoubleSizeMaybeAddGeThreeTensor(inp)
        torch._dynamo.mark_dynamic(x, 0)
        res = fn(x)
        # During fakeifying, we end up allocating a separate symint
        # for the outer and inner tensor (in this test, s0 is unused).
        expected_var_to_val = {
            "s0": 8,
            "s1": 4,
        }
        expected_var_to_sources = {
            "s0": "L['x'].size()[0]",
            "s1": "L['x'].inner_elem.size()[0]",
        }
        # lower bound comes from code underneath torch_dispatch  (operating on the inner tensor size)
        expected_lower_bound = "s1 > 3"
        # upper bound comes from user code (operating on the wrapper size)
        expected_upper_bound = "2*s1 < 10"
        self.assertEqual(curr_var_to_val, expected_var_to_val)
        self.assertEqual(curr_var_to_sources, expected_var_to_sources)
        self.assertEqual(lower_bound_str, expected_lower_bound)
        self.assertEqual(upper_bound_str, expected_upper_bound)

    def test_recompile_with_symbool_inputs(self):
        def f(pred: bool):
            if pred:
                return torch.ones([3, 4])
            else:
                return torch.ones([4, 3])

        def test_recompilation(
            f, x, sizes, exp_graphs, exp_frame_count, exp_shape_env_guards
        ):
            torch._dynamo.reset()
            shape_env = ShapeEnv()
            backend = torch._dynamo.testing.EagerAndRecordGraphs()
            cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
            f_cond = torch.compile(f, backend=cnt, fullgraph=True)
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                fake_inp = fake_mode.from_tensor(
                    x, dynamic_dims=[DimDynamic.DYNAMIC for i in range(x.dim())]
                )
                for i, size in enumerate(sizes):
                    pred = fake_inp.size(0) == size
                    f_cond(pred)
                    actual = normalize_gm(
                        backend.graphs[exp_frame_count[i] - 1].print_readable(
                            print_output=False
                        )
                    )
                    actual_guard_str = [str(guard.expr) for guard in shape_env.guards]
                    self.assertExpectedInline(actual, exp_graphs[i])
                    self.assertEqual(cnt.frame_count, exp_frame_count[i])
                    self.assertEqual(actual_guard_str, exp_shape_env_guards[i])

        true_graph = """\
class GraphModule(torch.nn.Module):
    def forward(self):
        ones = torch.ones([3, 4])
        return (ones,)
"""
        false_graph = """\
class GraphModule(torch.nn.Module):
    def forward(self):
        ones = torch.ones([4, 3])
        return (ones,)
"""
        test_recompilation(
            f,
            torch.randn([3, 4]),
            [3, 3, 4, 5],
            exp_graphs=[true_graph, true_graph, false_graph, false_graph],
            exp_frame_count=[1, 1, 2, 2],
            exp_shape_env_guards=[
                [],
                # s0 is specialized and guarded in outter shape_env when dynamo checks the guards
                ["Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)"],
                [
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 4)), (0, True)), 1)",
                ],
                [
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 4)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                ],
            ],
        )

        test_recompilation(
            f,
            torch.randn([3, 4]),
            [4, 5, 3, 3],
            exp_graphs=[false_graph, false_graph, true_graph, true_graph],
            exp_frame_count=[1, 1, 2, 2],
            exp_shape_env_guards=[
                [],
                # s0 is specialized and guarded in outter shape_env when dynamo checks the guards
                ["Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)"],
                [
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                ],
                [
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                ],
            ],
        )


class TestNestedTensor(torch._dynamo.test_case.TestCase):
    def _get_jagged_tensor(self, nested_size, offsets, requires_grad=True):
        # Makes a jagged tensor with N constituent tensors with size
        # as specified ((S0, S1, S2), D)
        D = nested_size[1]
        out = []
        for s in nested_size[0]:
            out.append(
                torch.randn(s, D, requires_grad=requires_grad, dtype=torch.float64)
            )
        return jagged_from_list(out, offsets)

    def _check_recompiles(self, fn, inputs1, inputs2, recompiles):
        compile_count = [0]

        def counter(gm, example_inputs):
            compile_count[0] += 1
            return gm

        compiled_f = torch.compile(fn, fullgraph=True, backend=counter, dynamic=True)
        out = compiled_f(*inputs1)
        self.assertEqual(compile_count[0], 1)
        out = compiled_f(*inputs2)
        self.assertEqual(compile_count[0], 2 if recompiles else 1)

    def test_unary_does_not_recompile(self):
        nt1, _ = self._get_jagged_tensor(((2, 3, 4), 3), None)
        nt2, _ = self._get_jagged_tensor(((3, 4, 5, 6), 4), None)
        self._check_recompiles(lambda nt1: nt1.sin(), (nt1,), (nt2,), False)

    def test_binary_does_not_recompile(self):
        def binary(nt1, nt2):
            if nt1.shape == nt2.shape:
                return nt1 + nt2
            else:
                return nt1.sin()

        # Basic binary
        nt1, offsets = self._get_jagged_tensor(((2, 3, 4), 3), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 4), 3), offsets)
        nt3, offsets = self._get_jagged_tensor(((3, 4, 5), 4), None)
        nt4, _ = self._get_jagged_tensor(((3, 4, 5), 4), offsets)
        self._check_recompiles(binary, (nt1, nt2), (nt3, nt4), False)

    def test_binary_recompiles(self):
        def binary(nt1, nt2):
            if nt1.shape == nt2.shape:
                return nt1 + nt2
            else:
                return nt1.sin()

        # Binary recompiles because singleton ints no longer match
        nt1, offsets = self._get_jagged_tensor(((2, 3, 4), 3), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 4), 3), offsets)
        nt3, _ = self._get_jagged_tensor(((2, 3, 4), 3), None)
        self._check_recompiles(binary, (nt1, nt2), (nt1, nt3), True)

    def test_binary_recompiles_due_to_duck_sizing(self):
        # Even though the input is unused, we still guard due to duck sizing
        nt1, offsets = self._get_jagged_tensor(((2, 3, 4), 3), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 4), 3), offsets)
        nt3, _ = self._get_jagged_tensor(((2, 3, 4), 3), None)
        self._check_recompiles(lambda nt1, nt2: nt1.sin(), (nt1, nt2), (nt1, nt3), True)

    # TODO: cannot parametrize this test class with device for some reason
    def _test_autograd(self, backend):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        nt, offsets = jagged_from_list([a, b, c], None)
        nt2, _ = jagged_from_list([a, b, c], offsets)

        def fn1(nt1, nt2):
            return (nt1 + nt2).sin().cos()

        compiled_f = torch.compile(
            fn1, fullgraph=True, backend="aot_eager", dynamic=True
        )
        out = compiled_f(nt, nt2)
        out_buffer = ViewBufferFromNested.apply(out)
        ga, gb, gc = torch.autograd.grad(out_buffer.sum(), (a, b, c))

        out_ref = compiled_f(nt, nt2)
        out_buffer_ref = ViewBufferFromNested.apply(out_ref)
        ga_ref, gb_ref, gc_ref = torch.autograd.grad(out_buffer_ref.sum(), (a, b, c))

        self.assertTrue(torch.allclose(ga, ga_ref))
        self.assertTrue(torch.allclose(gb, gb_ref))
        self.assertTrue(torch.allclose(gc, gc_ref))

    def test_basic_autograd(self):
        self._test_autograd("aot_eager")

    @requires_cuda()
    def test_basic_autograd_inductor(self):
        self._test_autograd("inductor")

    def test_unbind(self):
        nt, _ = self._get_jagged_tensor(((2, 3, 4), 3), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 5), 2), None)
        nt3, _ = self._get_jagged_tensor(((2, 3, 4, 5), 3), None)

        def fn(x):
            return x.unbind()

        compiled_f = torch.compile(fn, fullgraph=True, backend="eager", dynamic=True)
        out = compiled_f(nt)

        out_ref = fn(nt)

        # correctness
        self.assertEqual(len(out), len(out_ref))
        for x, x_ref in zip(out, out_ref):
            self.assertTrue(torch.allclose(x, x_ref))

        # We specialize on the length of offsets, e.g. (1) we recompile if the
        # length of the offsets is different. (2) we don't recompile if the
        # length of the offsets is the same, even if the size of the constituent
        # tensors are different.
        self._check_recompiles(fn, (nt,), (nt2,), False)
        self._check_recompiles(fn, (nt,), (nt3,), True)

    def _get_views(self):
        # There are three cases to consider here based on the logic in
        # meta_utils.py
        #
        # (1) basic case:
        # view is not a leaf and has the same requires grad as its basic case
        x, _ = self._get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)
        self.assertEqual(x.is_leaf, False)
        yield x.unsqueeze(-1)

        # (2) leaf view case:
        # the view has to be a leaf (w/ requires_grad True or requires_grad False)
        # base w/ requires_grad True or requires_grad False
        for requires_grad_1, requires_grad_2 in itertools.product(
            [True, False], repeat=2
        ):
            x, _ = self._get_jagged_tensor(
                ((2, 3, 4), 3), None, requires_grad=requires_grad_1
            )
            with torch.no_grad():
                x_view = x.unsqueeze(-1)
                # The issue is this doesn't quite work
                x_view.requires_grad_(requires_grad_2)
            yield x_view

        # (3) obscure case:
        # view is not a leaf (implies requires_grad True)
        # base w/ requires_grad False)
        x, _ = self._get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=False)
        # intermediate leaf view
        with torch.no_grad():
            x_view = x.unsqueeze(-1)
        x_view.requires_grad_(True)
        x_view_view = x_view.unsqueeze(-1)
        yield x_view_view

    def test_inputs_to_compiled_fn_are_views(self):
        for nt_view in self._get_views():

            def fn(x):
                return x.sin()

            out_ref = fn(nt_view)
            torch._dynamo.reset()
            compile_fn = torch.compile(
                fn, fullgraph=True, backend="aot_eager", dynamic=True
            )
            out = compile_fn(nt_view)

            # Check metadata and values are correct
            self.assertTrue(out.size() == out_ref.size())
            self.assertTrue(out.stride() == out_ref.stride())
            self.assertTrue(torch.allclose(out.values(), out_ref.values()))

            # Check that no guards are incurred
            def backend(gm, args):
                context = torch._guards.TracingContext.get()
                val_to_guards = context.fake_mode.shape_env.var_to_guards.values()
                self.assertEqual(len(val_to_guards), 0)
                return gm

            torch._dynamo.reset()
            compile_fn = torch.compile(
                fn, fullgraph=True, backend=backend, dynamic=True
            )
            out = compile_fn(nt_view)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
