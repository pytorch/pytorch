# Owner(s): ["module: dynamo"]
import contextlib
import functools

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.testing import normalize_gm
from torch._functorch.aot_autograd import to_fun


class MockSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


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

    def test_compile_with_fake_tensor(self):
        x = torch.randn([3, 4])
        x2 = torch.randn([4, 3])
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return torch.sin(x)

        f(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

        f(x2)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)

        with torch._subclasses.fake_tensor.FakeTensorMode() as fake_mode:
            fake_tensor = fake_mode.from_tensor(x)
            f(fake_tensor)

        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 3)

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

                args_inner = pytree.tree_map_only(
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

            # Grab info on sources and guards from the shapenv
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
            "s0": 4,
        }
        expected_var_to_sources = {
            "s0": "L['x'].inner_elem.size()[0]",
        }
        # lower bound comes from code underneath torch_dispatch  (operating on the inner tensor size)
        expected_lower_bound = "s0 > 3"
        # upper bound comes from user code (operating on the wrapper size)
        expected_upper_bound = "2*s0 < 10"
        self.assertEqual(curr_var_to_val, expected_var_to_val)
        self.assertEqual(curr_var_to_sources, expected_var_to_sources)
        self.assertEqual(lower_bound_str, expected_lower_bound)
        self.assertEqual(upper_bound_str, expected_upper_bound)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
