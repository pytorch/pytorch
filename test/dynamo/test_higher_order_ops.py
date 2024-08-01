# Owner(s): ["module: dynamo"]
import enum
import functools
import pprint
import re
import unittest
import warnings

import functorch.experimental.control_flow as control_flow
import torch
import torch._dynamo.config as config
import torch._dynamo.test_case
import torch._functorch.config
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import (
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    normalize_gm,
)
from torch._dynamo.utils import counters, ifdynstaticdefault
from torch._higher_order_ops.wrap import wrap
from torch.testing._internal.common_utils import (
    munge_exc,
    TEST_WITH_TORCHDYNAMO,
    xfailIfTorchDynamo,
)
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")


def check_dynamic_shape_capture():
    # This also mirrors config from `test/dynamo/test_dynamic_shapes.py:make_dynamic_cls`
    return not config.assume_static_by_default


def count_ops(gm, args, freq, op):
    actual = [node.target for node in gm.graph.nodes].count(op)
    assert actual == freq, f"expected={freq}, actual={actual}"
    return gm


class Obj:
    pass


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.existing = torch.nn.Parameter(torch.ones([]))

    def forward(self, x):
        return self.existing * x


global_obj = Obj()
global_module = MyModule()
global_var = torch.randn(3)
global_num = 3.14
global_list = []


def find_first_node(gm, func):
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


def op_count(gm):
    result = 0
    for node in gm.graph.nodes:
        if "call" in node.op:
            result += 1
    return result


# Checks that a dict matches a dict with "regex keys". That is,
# the keys are regex expressions.
def assert_dict_matches_regex(self, dct, dct_with_regex_keys):
    regex_keys = dct_with_regex_keys.keys()
    regex_key_to_actual_key = {}
    for regex_key in regex_keys:
        for key in dct:
            if re.match(regex_key, key):
                if regex_key in regex_key_to_actual_key:
                    raise AssertionError(
                        f"Single key regex mapped to multiple keys. Please improve your "
                        f"regex. Got: regex='{regex_key}' "
                        f"keys='{regex_key_to_actual_key[regex_key]}',"
                        f"'{key}'"
                    )
                regex_key_to_actual_key[regex_key] = key
    new_dct = {}
    for regex_key in regex_keys:
        if regex_key not in regex_key_to_actual_key:
            raise AssertionError(
                f"Got regex '{regex_key}' but could not match any key in dict with "
                f"keys {dct.keys()}"
            )
        new_dct[regex_key_to_actual_key[regex_key]] = dct_with_regex_keys[regex_key]
    self.assertEqual(dct, new_dct)


def default_args_generator(seed_value):
    flat_args, args_spec = pytree.tree_flatten(seed_value)
    for i in range(3):
        new_flat_arg = []
        for val in flat_args:
            if isinstance(val, torch.Tensor):
                new_val = val + 0.1 * i
            elif isinstance(val, int):
                new_val = val + 1 * i
            elif isinstance(val, float):
                new_val = val + 0.1 * i
            elif isinstance(val, enum.Enum):
                new_val = val
            else:
                raise AssertionError("unexpected arg type")

            new_flat_arg.append(new_val)
        new_args = pytree.tree_unflatten(new_flat_arg, args_spec)
        yield new_args


class HigherOrderOpTests(torch._dynamo.test_case.TestCase):
    def _assert_wrap_fallback(self, func, args, setup=lambda: None):
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        setup()
        expected = func(*args)
        setup()
        result = torch.compile(func, backend=cnt, fullgraph=False)(*args)
        num_graph_breaks = len(counters["graph_break"].keys())
        self.assertGreater(num_graph_breaks, 0)

        for gm in backend.graphs:
            for node in gm.graph.nodes:
                self.assertFalse(node.target is wrap)

        self.assertEqual(result, expected)

    def _test_wrap_simple(
        self,
        func,
        args_generator,
        expected_num_wrap_args,
        expected_opcount=2,
        return_graph=False,
    ):
        # Given a `func` that has a single call to `wrap`,
        # we check that:
        # - there are no graph breaks
        # - eager vs torch.compile has the same result (correctness)
        # - other compilation metrics, e.g, # of ops in the dynamo captured graph,
        #   the wrap has the expected number of args, etc
        #
        # we have one or multiple runs through with each of the args from args_generator,
        # and we will check:
        # - correctness and no graph breaks for every run
        # - other compilation metrics only for the first run, since automatic_dynamic_shapes
        #   may compile another dynamic version graph for the later runs
        graph = None
        for i, args in enumerate(args_generator):
            backend = EagerAndRecordGraphs()
            cnt = CompileCounterWithBackend(backend)
            expected = func(*args)
            result = torch.compile(func, fullgraph=True, backend=cnt)(*args)
            # check correctness and no graph breaks
            self.assertEqual(result, expected)
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(len(backend.graphs), 1)
            # check other compilation metrics
            if i == 0:
                self.assertEqual(cnt.op_count, expected_opcount)
                graph = backend.graphs[0]
                wrap_node = find_first_node(graph, wrap)
                self.assertEqual(len(wrap_node.args), expected_num_wrap_args)
        # We always return/check the graph from the first run if return_graph = True
        if return_graph:
            return normalize_gm(graph.print_readable(print_output=False))

    def test_error_message_sane(self):
        foo = []

        def inner(x):
            foo.append(x)
            return x.clone()

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return wrap(inner, x)

        x = torch.randn(3)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"HigherOrderOperator: Mutating a variable not in the current scope \(SideEffects\)",
        ):
            f(x)

    def test_no_freevars(self):
        def f(x):
            return wrap(lambda x: torch.sin(x), x)

        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 2)

    def test_enum_arg(self):
        class SomeEnum(enum.Enum):
            A = 0
            B = 1

        def g(x, val):
            if val == SomeEnum.A:
                return torch.sin(x)
            return torch.cos(x)

        def f(x, val):
            return wrap(g, x, val)

        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x, SomeEnum.A)), 2)

    def test_return_captured_var(self):
        freevar = torch.randn(3)

        def test(x):
            return freevar

        def fn(x):
            return wrap(test, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.
        self._test_wrap_simple(fn, default_args_generator((x,)), 2)

    def test_return_captured_vars(self):
        freevar1 = torch.randn(3)
        freevar2 = torch.randn(3)

        def test(x):
            return freevar1, freevar2, freevar1

        def fn(x):
            return wrap(test, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.
        self._test_wrap_simple(fn, default_args_generator((x,)), 3, 4)

    def test_return_captured_var_used_multiple_times(self):
        freevar = torch.randn(3)

        def test(x):
            y = x + freevar
            return y, freevar

        def fn(x):
            return wrap(test, x)

        x = torch.randn(3)
        self._test_wrap_simple(fn, default_args_generator((x,)), 3, 3)

    def test_capture_untracked_global(self):
        def f(x):
            return wrap(lambda x: x + global_var, x)

        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 3)

    def test_symint_input(self):
        def f(x):
            i = x.size(0)
            return wrap(lambda x, i: x.view(i), x, i)

        x = torch.randn(3, 1)
        self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            ifdynstaticdefault(2, 3),
            expected_opcount=2,
        )

    def test_wrap_pytree_args_nested(self):
        def f(x, y, z):
            def fn(d):
                return d["x"].sin() + d["y"][0].cos() - d["y"][1][2].sin()

            return wrap(fn, d)

        x = torch.tensor(1.5)
        y = torch.tensor(2.0)
        z = torch.tensor(3.0)
        d = {"x": x, "y": (y, [x, y, z])}

        def my_args_generator(t):
            yield t
            yield t[0] + 0.1, t[1], t[2]
            yield t[0], t[1] + 0.1, t[2]

        actual_graph = self._test_wrap_simple(
            f,
            my_args_generator((x, y, z)),
            4,
            return_graph=True,
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_d_x_: "f32[]", L_d_y_0_: "f32[]", L_d_y_1_2_: "f32[]"):
        l_d_x_ = L_d_x_
        l_d_y_0_ = L_d_y_0_
        l_d_y_1_2_ = L_d_y_1_2_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_d_x_, l_d_y_0_, l_d_y_1_2_);  wrap_body_0 = l_d_x_ = l_d_y_0_ = l_d_y_1_2_ = None
        getitem: "f32[]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_d_x_: "f32[]", l_d_y_0_: "f32[]", l_d_y_1_2_: "f32[]"):
            sin: "f32[]" = l_d_x_.sin();  l_d_x_ = None
            cos: "f32[]" = l_d_y_0_.cos();  l_d_y_0_ = None
            add: "f32[]" = sin + cos;  sin = cos = None
            sin_1: "f32[]" = l_d_y_1_2_.sin();  l_d_y_1_2_ = None
            sub: "f32[]" = add - sin_1;  add = sin_1 = None
            return (sub,)
""",  # NOQA: B950
        )

    def test_wrap_pytree_args_with_symint_constant(self):
        def f(x, y):
            i = x.size(0)
            return wrap(lambda t: t[0].view(t[2]) + t[1], (x, y, i))

        x = torch.randn(3, 1)
        y = 0.5
        actual_graph = self._test_wrap_simple(
            f,
            default_args_generator((x, y)),
            ifdynstaticdefault(2, 3),
            expected_opcount=2,
            return_graph=True,
        )
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                actual_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 1]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 1]"):
            view: "f32[3]" = l_x_.view(3);  l_x_ = None
            add: "f32[3]" = view + 0.5;  view = None
            return (add,)
""",
            )
        else:
            self.assertExpectedInline(
                actual_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s0: "Sym(s0)", L_x_: "f32[s0, 1]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_, s0);  wrap_body_0 = l_x_ = s0 = None
        getitem: "f32[s0]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[s0, 1]", size: "Sym(s0)"):
            view: "f32[s0]" = l_x_.view(size);  l_x_ = size = None
            add: "f32[s0]" = view + 0.5;  view = None
            return (add,)
""",
            )

    def test_wrap_pytree_kwargs(self):
        def f(x, y, z):
            def fn(*, x, y, z):
                z1, z2 = z
                return (x * 2) + y + z1

            return wrap(fn, x=x, y=y, z=z)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        def my_args_generator(t):
            yield t
            x1 = t[0] + 0.1
            y1 = t[1] + 0.1
            yield (x1, y1, (x1, y1))
            x2 = t[0] + 0.2
            y2 = t[0] + 0.2
            yield (x2, y2, (x2, y2))

        self._test_wrap_simple(f, my_args_generator((x, y, (x, y))), 3)

    def test_wrap_pytree_args_not_const_symint_tensor(self):
        class MyClass:
            def __init__(self, x):
                self.val = x

        def f(x, y):
            return wrap(lambda z: z[0].sin() * z[1].val.cos(), (x, y))

        x = torch.tensor(1.2)
        y = MyClass(torch.tensor(3.4))
        self._test_wrap_simple(f, [(x, y)], 3)

    def test_capture_constants(self):
        x = torch.randn(3, 3)
        y = 4.0

        def fn(x, y, z):
            if z:
                return x + y
            return x * y

        def f(x, y, z):
            return wrap(fn, x, y, z)

        args = (x, 4.0, None)
        opt_f = torch.compile(f, fullgraph=True, backend=CompileCounter())
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)

        # Ensure that we recompile here
        args = (x, 5.0, None)
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)

    def test_capture_untracked_global_nested(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return wrap(lambda x: wrap(lambda x: x + global_var, x), x)

        x = torch.randn(3)
        result = f(x)

        self.assertEqual(result, x + global_var)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)

        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_capture_untracked_nonlocal(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            def g(x):
                return wrap(lambda x: x + y, x)

            self._test_wrap_simple(g, default_args_generator((x,)), 3)
            return g(x)

        f(x, y)

    def test_capture_tracked(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            return wrap(lambda x: x + y, x)

        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_capture_tracked_nested(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            return wrap(lambda x: wrap(lambda x: x + y, x), x)

        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_inlined_functions(self):
        def g(x, y):
            return x + y

        def f(x, y):
            return wrap(lambda x: g(x, y), x)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_same_freevar_twice(self):
        free = torch.randn(3)

        def g(x):
            y = free.sin()
            z = free.cos()
            return y, z

        def f(x):
            return wrap(g, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.
        self._test_wrap_simple(f, default_args_generator((x,)), 2, 3)

    def test_register_subclass(self):
        from torch._higher_order_ops.cond import cond_op
        from torch.testing._internal.two_tensor import TwoTensor

        a = torch.tensor([1.0, 0.0, 1.0])
        b = torch.randn(3)
        t = TwoTensor(a, b)
        with self.assertRaisesRegex(NotImplementedError, "no rule registered"):
            res = cond_op(a.sum() > 0, torch.sin, torch.cos, (t,))

        called = 0

        # Using cond.py_impl
        @cond_op.py_impl(TwoTensor)
        def _(pred, true_fn, false_fn, operands):
            nonlocal called
            called += 1
            assert len(operands) == 1
            a = cond_op(pred, true_fn, false_fn, (operands[0].a,))
            b = cond_op(pred, true_fn, false_fn, (operands[0].b,))
            return TwoTensor(a, b)

        res = cond_op(a.sum() > 0, torch.sin, torch.cos, (t,))
        self.assertEqual(res.a, torch.sin(a))
        self.assertEqual(res.b, torch.sin(b))
        self.assertEqual(called, 1)

    def test_register_mode(self):
        from torch._higher_order_ops.cond import cond_op

        torch_dispatch_called = 0

        class MyMode(torch.utils._python_dispatch.TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                nonlocal torch_dispatch_called
                torch_dispatch_called += 1
                return func(*args, **kwargs)

        a = torch.tensor([1.0, 0.1, 1.0])
        pred = a.sum() > 0
        with self.assertRaisesRegex(NotImplementedError, "no rule registered"):
            with MyMode():
                res = cond_op(pred, torch.sin, torch.cos, (a,))

        py_impl_called = 0

        # Using cond.py_impl
        @cond_op.py_impl(MyMode)
        def _(mode, pred, true_fn, false_fn, operands):
            nonlocal py_impl_called
            py_impl_called += 1
            return cond_op(pred, true_fn, false_fn, operands)

        a = torch.tensor([1.0, 0.1, 1.0])
        pred = a.sum() > 0
        with MyMode():
            res = cond_op(pred, torch.sin, torch.cos, (a,))
        self.assertEqual(res, a.sin())

    def test_capture_value_created_in_subgraph(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            return wrap(inner, x, y)

        result = f(x, y)

        self.assertEqual(result, x + y + x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(backend.graphs), 1)

        # No changes to args of outer wrap
        gm = backend.graphs[0]
        wrap_node = find_first_node(gm, wrap)
        self.assertTrue(len(wrap_node.args), 3)

        # z was lifted to arg of inner wrap
        body_function = getattr(gm, wrap_node.args[0].name)
        # addition + wrap + getitem
        self.assertEqual(op_count(body_function), 3)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

        # Innermost body function: z was also lifted to arg
        body_function = getattr(body_function, inner_wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_side_effect_set_new_attr_global_obj(self):
        def setup():
            global global_obj
            global_obj = Obj()

        def f(x):
            def h(x):
                def g(x):
                    global_obj.foo = x + 1
                    return x.clone()

                y = wrap(g, x)
                return y + global_obj.foo

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_set_existing_attr_global_obj(self):
        def setup():
            global global_obj
            global_obj = Obj()
            global_obj.foo = nn.Parameter(torch.tensor(4.0))

        def f(x):
            def h(x):
                def g(x):
                    global_obj.foo = x + 1
                    return x.clone()

                y = wrap(g, x)
                return y + global_obj.foo

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_del_existing_attr_global_obj(self):
        def setup():
            global global_obj
            global_obj = Obj()
            global_obj.foo = torch.tensor(4.0)

        def f(x):
            def h(x):
                def g(x):
                    del global_obj.foo
                    return x.clone()

                y = wrap(g, x)
                return y

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_set_new_attr_global_module(self):
        def setup():
            global global_module
            global_module = MyModule()

        def h(x):
            def g(x):
                global_module.foo = nn.Parameter(x + 1)
                return x.clone()

            y = wrap(g, x)
            return y + global_module.foo

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_set_existing_attr_global_module(self):
        def setup():
            global global_module
            global_module = MyModule()

        def h(x):
            def g(x):
                global_module.existing = nn.Parameter(torch.tensor(4.0))
                return global_module(x)

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_del_existing_attr_global_module(self):
        def setup():
            global global_module
            global_module = MyModule()

        def h(x):
            def g(x):
                del global_module.existing
                return x.clone()

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_mutate_global_num(self):
        def setup():
            global global_num
            global_num = 3.14

        def f(x):
            def g(x):
                global global_num
                global_num = global_num + 1
                return x + global_num

            y = wrap(g, x)
            return y + global_num

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_num_builtin(self):
        def setup():
            global global_num
            global_num = 3.14

        def f(x):
            def g(x):
                global global_num
                global_num += 1
                return x + global_num

            y = wrap(g, x)
            return y + global_num

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_tensor(self):
        def setup():
            global global_var
            global_var = torch.ones(3)

        def f(x):
            def g(x):
                global global_var
                global_var = global_var + 1
                return x + global_var

            y = wrap(g, x)
            return y + global_var

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_tensor_builtin(self):
        def setup():
            global global_var
            global_var = torch.ones(3)

        def f(x):
            def g(x):
                global global_var
                global_var += 1
                return x + global_var

            y = wrap(g, x)
            return y + global_var

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_list(self):
        def setup():
            global global_list
            global_list = []

        def f(x):
            def g(x):
                val = x + 1
                global_list.append(val)
                return global_list[-1]

            y = wrap(g, x)
            z = y + global_list[-1]
            return z

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_nonlocal_num(self):
        def f(x):
            def h(x):
                val = 1

                def g(x):
                    nonlocal val
                    val = val + 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_new_attr_nonlocal_obj(self):
        def f(x):
            def h(x):
                obj = Obj()

                def g(x):
                    obj.val = x.dim()
                    return x.clone()

                y = wrap(g, x)
                z = y + obj.val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_existing_attr_nonlocal_obj(self):
        def f(x):
            def h(x):
                obj = Obj()
                obj.val = 3

                def g(x):
                    obj.val = x.dim()
                    return x.clone()

                y = wrap(g, x)
                z = y + obj.val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_del_existing_attr_nonlocal_obj(self):
        def f(x):
            def h(x):
                obj = Obj()
                obj.val = 3

                def g(x):
                    del obj.val
                    return x.clone()

                y = wrap(g, x)
                return y

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_new_attr_nonlocal_module(self):
        def h(x):
            obj = MyModule()

            def g(x):
                obj.val = x.dim()
                return x.clone()

            y = wrap(g, x)
            z = y + obj.val
            return z

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_set_existing_attr_nonlocal_module(self):
        def h(x):
            obj = MyModule()

            def g(x):
                obj.existing = nn.Parameter(torch.tensor(3.14))
                return obj(x)

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_del_existing_attr_nonlocal_module(self):
        def h(x):
            obj = MyModule()

            def g(x):
                del obj.existing
                return x.clone()

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_mutate_nonlocal_tensor(self):
        def f(x):
            def h(x):
                val = torch.tensor(1.0)

                def g(x):
                    nonlocal val
                    val = val + 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_mutate_nonlocal_num_builtin(self):
        def f(x):
            def h(x):
                val = 1

                def g(x):
                    nonlocal val
                    val += 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_mutate_nonlocal_tensor_builtin(self):
        def f(x):
            def h(x):
                val = torch.tensor(1.0)

                def g(x):
                    nonlocal val
                    val += 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_nonlocal_list_append_graph_break(self):
        def g(x):
            y = []

            def f(k):
                m = k + 1
                y.append(m)
                return k

            wrap(f, x)
            return y[0]

        x = torch.randn(3, 3)
        self._assert_wrap_fallback(g, (x,))

    def test_side_effect_nested_nonlocal_list_append_graph_break(self):
        def g(x):
            def h(x):
                y = []

                def f(k):
                    m = k + 1
                    y.append(m)
                    return k

                wrap(f, x)
                return y[0]

            return h(x)

        x = torch.randn(3, 3)
        self._assert_wrap_fallback(g, (x,))

    def test_side_effect_local_list_append_no_graph_break(self):
        def g(x):
            def f(k):
                y = []
                y.append(k + 1)
                return y[0]

            return wrap(f, x)

        x = torch.randn(3, 3)
        self._test_wrap_simple(g, default_args_generator((x,)), 2)

    def test_wrap_kwarg(self):
        def f(x, y):
            return wrap(lambda x, y: x + y, x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_wrap_kwarg_int(self):
        def f(x, y):
            return wrap(lambda x, y: x + y, x, y=y)

        x = torch.randn(3)
        y = 8

        self._test_wrap_simple(
            f, default_args_generator((x, y)), ifdynstaticdefault(2, 3)
        )

    def test_wrap_all_kwarg(self):
        def f(y, x):
            return wrap(lambda x, y: (x * 2) + y, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_wrap_kwarg_only(self):
        def f(x, y):
            def fn(*, x, y):
                return (x * 2) + y

            return wrap(fn, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_wrap_kwarg_default(self):
        def f(x, y):
            def fn(*, x, y, z=8):
                return (x * 2) + y + z

            return wrap(fn, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_wrap_kwarg_default_if_branch(self):
        def f(x, y):
            def fn(*, x, y, z=None):
                if z is None:
                    return (x * 2) + y
                else:
                    return 2 * x

            return wrap(fn, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_wrap_kwarg_recompile(self):
        def f(x, y, z=None):
            def fn(*, x, y, z=None):
                if z is None:
                    return (x * 2) + y
                else:
                    return 2 * x

            return wrap(fn, x=x, y=y, z=z)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        counters.clear()
        opt = torch.compile(f, backend="eager", fullgraph=True)
        opt(x, y)
        self.assertEqual(counters["stats"]["calls_captured"], 2)

        # verify that we `don't` recompile
        opt(x, y)
        self.assertEqual(counters["stats"]["calls_captured"], 2)

        output = opt(x, y, 8)
        self.assertEqual(counters["stats"]["calls_captured"], 4)
        self.assertEqual(output, 2 * x)

    def test_wrap_kwarg_default_else_branch(self):
        def f(x, y, z):
            def fn(*, x, y, z=None):
                if z is None:
                    return (x * 2) + y
                else:
                    return 2 * x

            return wrap(fn, x=x, y=y, z=z)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        self._test_wrap_simple(f, default_args_generator((x, y, 8)), 2)

    def test_map_subgraph_name_is_valid(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        xs = torch.randn(2, 3, 3)
        y = torch.randn(3)

        def map_f(xs, y):
            def inner(x, y):
                def inner2(x, y):
                    return x + y

                return control_flow.map(inner2, x, y)

            return control_flow.map(inner, xs, y)

        graphs = self._check_map_graph_and_extract(map_f, (xs, y))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_xs_ : torch.Tensor, L_y_ : torch.Tensor):
    l_xs_ = L_xs_
    l_y_ = L_y_
    map_body_1 = self.map_body_1
    map_impl = torch.ops.higher_order.map_impl(map_body_1, [l_xs_], [l_y_]);  map_body_1 = l_xs_ = l_y_ = None
    getitem_1 = map_impl[0];  map_impl = None
    return (getitem_1,)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child, l_y_):
    child_1 = child[0];  child_1 = None
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [child], [l_y_]);  map_body_0 = child = l_y_ = None
    getitem_1 = map_impl[0];  map_impl = None
    return (getitem_1,)""",
            )

    def test_map_multi_return(self):
        cnt = CompileCounter()

        def f(x):
            return control_flow.map(lambda x: (x.sin(), x.sin()), x)

        x = torch.randn(3)
        graphs = self._check_map_graph_and_extract(f, (x,))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], []);  map_body_0 = l_x_ = None
    getitem_1 = map_impl[0]
    getitem_2 = map_impl[1];  map_impl = None
    return (getitem_1, getitem_2)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child):
    child_1 = child.sin()
    child_2 = child.sin();  child = None
    return (child_1, child_2)""",
            )

    def test_map_pytree_return(self):
        cnt = CompileCounter()

        def _construct_pytree(a):
            return (a, [[[a]]], a, (a, (a,), a), {"a": a})

        def f(x):
            def inner_f(xs):
                return _construct_pytree(xs)

            return control_flow.map(inner_f, x)

        x = torch.randn(3)
        graphs = self._check_map_graph_and_extract(f, (x,))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], []);  map_body_0 = l_x_ = None
    getitem_1 = map_impl[0]
    getitem_2 = map_impl[1]
    getitem_3 = map_impl[2]
    getitem_4 = map_impl[3]
    getitem_5 = map_impl[4]
    getitem_6 = map_impl[5]
    getitem_7 = map_impl[6];  map_impl = None
    return (getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child):
    return (child, child, child, child, child, child, child)""",
            )

    def test_map_kwargs(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            return control_flow.map(lambda x: x.sin(), x=x)

        x = torch.randn(3)
        self.assertRaises(TypeError, lambda: f(x))
        self.assertEqual(cnt.frame_count, 0)

    def test_map_symint_input(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        def fn(x, y):
            def inner(x, y):
                return torch.sin(x + y)

            return control_flow.map(inner, x, y.size(0))

        x = torch.randn(3, 1)
        y = torch.randn(3, 1)
        graphs = self._check_map_graph_and_extract(fn, (x, y))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], [3]);  map_body_0 = l_x_ = None
    getitem_1 = map_impl[0];  map_impl = None
    return (getitem_1,)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child, const_unused):
    add = child + 3;  child = None
    sin = torch.sin(add);  add = None
    return (sin,)""",
            )

    def test_map_lowers_to_graph(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        def fn(x, y):
            def inner(x, y):
                return torch.sin(x + y)

            return control_flow.map(inner, x, y.size(0))

        x = torch.randn(3, 1)
        y = torch.randn(3, 1)
        graphs = self._check_map_graph_and_extract(fn, (x, y))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], [3]);  map_body_0 = l_x_ = None
    getitem_1 = map_impl[0];  map_impl = None
    return (getitem_1,)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child, const_unused):
    add = child + 3;  child = None
    sin = torch.sin(add);  add = None
    return (sin,)""",
            )

    def test_map_example_value_metadata_consistent_with_eager(self):
        from torch._higher_order_ops.map import map_dense

        backend = EagerAndRecordGraphs()

        def inner(x):
            return x.sin(), x.cos().T, x.sin().view(-1)

        rand_44 = torch.randn(4, 4)
        inps = [
            torch.randn(3),
            torch.randn(3, 4),
            torch.randn(3, 4, 5, requires_grad=True),
            torch.randn(3, 4, 5, requires_grad=True).permute((2, 0, 1)),
            torch.randn(3, 4, 5, requires_grad=True).detach(),
            torch.randn(3, 4, 5, requires_grad=True).narrow(1, 1, 2),
            rand_44.T,
            rand_44[::2],
            rand_44[::2, ::2],
            rand_44[1::3, 1::3],
            rand_44[1::3, 1::2].T,
            rand_44.unsqueeze(1),
            rand_44.squeeze(0),
            rand_44.reshape(2, 8),
        ]
        for x in inps:
            compiled_ret = torch.compile(
                control_flow.map, backend=backend, fullgraph=True
            )(inner, x)
            eager_sin, eager_transpose, eager_view = map_dense(inner, (x,), ())

            map_node = next(
                node
                for node in backend.graphs[0].graph.nodes
                if node.op == "call_function" and "map" in node.name
            )

            fake_sin, fake_transpose, fake_view = map_node.meta["example_value"]

            def _check_size_stride_contiguous(x, y):
                self.assertEqual(y.size(), x.size())
                self.assertEqual(y.stride(), x.stride())
                self.assertEqual(y.requires_grad, x.requires_grad)
                self.assertEqual(x.is_contiguous(), True)
                self.assertEqual(y.is_contiguous(), True)

            _check_size_stride_contiguous(eager_sin, fake_sin)
            _check_size_stride_contiguous(eager_transpose, fake_transpose)
            _check_size_stride_contiguous(eager_view, fake_view)

            torch._dynamo.reset()
            backend.graphs.clear()

    def test_cond_subgraph_name_is_valid(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        pred = torch.tensor(True)
        pred2 = torch.tensor(False)
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def cond_f(pred, pred2, x, y):
            def true_fn(pred2, x, y):
                return x + y

            def false_fn(pred2, x, y):
                def true_fn2(x, y):
                    return x.sin() - y.cos()

                def false_fn2(x, y):
                    return x.cos() - y.sin()

                return control_flow.cond(pred2, true_fn2, false_fn2, [x, y])

            return control_flow.cond(pred, true_fn, false_fn, [pred2, x, y])

        result = cond_f(pred, pred2, xs, y)
        self.assertEqual(result, xs + y)

        cond_gm = backend.graphs[0]
        name_set = set()
        name_set.update(name for name, _ in cond_gm.named_modules())
        self.assertEqual(
            name_set,
            {
                "",
                "cond_true_1",
                "cond_false_1",
                "cond_false_1.cond_false_0",
                "cond_false_1.cond_true_0",
            },
        )

    @torch._dynamo.config.patch(
        assume_static_by_default=True,
        dynamic_shapes=True,
    )
    def test_cond_graph_break_in_one_branch(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, x):
                def true_fn(x):
                    self.buffer += 1
                    return self.buffer.sum() + x.sum()

                def false_fn(x):
                    return (x - 1).sum()

                return control_flow.cond(x.sum() > 4, true_fn, false_fn, [x])

        mod_for_compile = torch.compile(Foo(), backend=cnt, dynamic=True)
        mod_for_eager = Foo()

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Cond doesn't work unless it is captured completely with torch.compile",
        ):
            mod_for_eager(torch.ones(6, 4))

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Cond doesn't work unless it is captured completely with torch.compile",
        ):
            mod_for_compile(torch.ones(3, 4))

    def test_cond_free_variable_in_both_branches(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        z = torch.ones(4, 4)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, x, y):
                def true_fn(x):
                    return x.sum() + self.buffer.sum() + z.sum()

                def false_fn(x):
                    return x.sum() - z.sum() - self.buffer.sum()

                return control_flow.cond(y, true_fn, false_fn, [x])

        mod_for_compile = torch.compile(
            Foo(), backend=cnt, dynamic=True, fullgraph=True
        )
        mod_for_eager = Foo()

        self.assertEqual(
            mod_for_compile(torch.tensor(True), torch.tensor(5)),
            mod_for_eager(torch.tensor(True), torch.tensor(5)),
        )

        for node in backend.graphs[0].graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.cond
            ):
                _, _, _, operands = node.args
                # Each branch takes 3 inputs (buffer, x, z)
                self.assertEqual(len(operands), 3)
            if node.op == "get_attr":
                if str(node.target) in ("cond_true_0, cond_false_0"):
                    num_placeholders = len(
                        [
                            node
                            for node in getattr(
                                backend.graphs[0], str(node.target)
                            ).graph.nodes
                            if node.op == "placeholder"
                        ]
                    )
                    self.assertEqual(num_placeholders, 3)

    def _check_cond_graph_and_extract(self, fn, args):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        out = torch.compile(fn, backend=cnt, fullgraph=True)(*args)
        self.assertEqual(out, fn(*args))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(backend.graphs), 1)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        gm = backend.graphs[0]
        graph = gm.code.strip()
        true_graph = gm.cond_true_0.code.strip()
        false_graph = gm.cond_false_0.code.strip()
        return (graph, true_graph, false_graph)

    def _check_map_graph_and_extract(self, fn, args):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        out = torch.compile(fn, backend=cnt, fullgraph=True)(*args)
        self.assertEqual(out, fn(*args))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(backend.graphs), 1)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        gm = backend.graphs[0]
        graph = gm.code.strip()
        subgraphs = []
        for module_name in gm._modules.keys():
            subgraphs.append(getattr(gm, module_name).code.strip())
        return (graph, *subgraphs)

    def test_cond_branches_no_arguments(self):
        def fn(x):
            def true_fn():
                return torch.sin(x)

            def false_fn():
                return torch.cos(x)

            return control_flow.cond(x.sum() > 0, true_fn, false_fn, ())

        graphs = self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        if graphs is not None:
            graph, true_graph, false_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    sum_1 = l_x_.sum()
    gt = sum_1 > 0;  sum_1 = None
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, [l_x_]);  gt = cond_true_0 = cond_false_0 = l_x_ = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                true_graph,
                """\
def forward(self, l_x_):
    l_x__1 = l_x_
    sin = torch.sin(l_x__1);  l_x__1 = None
    return (sin,)""",
            )
            self.assertExpectedInline(
                false_graph,
                """\
def forward(self, l_x_):
    l_x__1 = l_x_
    cos = torch.cos(l_x__1);  l_x__1 = None
    return (cos,)""",
            )

    def test_cond_branches_no_arguments_no_closure(self):
        def fn(x):
            def true_fn():
                return torch.ones(3, 4)

            def false_fn():
                return torch.ones(3, 4).sin()

            return control_flow.cond(x.sum() > 0, true_fn, false_fn, ())

        self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        graphs = self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        if graphs is not None:
            graph, true_graph, false_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    sum_1 = l_x_.sum();  l_x_ = None
    gt = sum_1 > 0;  sum_1 = None
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, []);  gt = cond_true_0 = cond_false_0 = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                true_graph,
                """\
def forward(self):
    ones = torch.ones(3, 4)
    return (ones,)""",
            )
            self.assertExpectedInline(
                false_graph,
                """\
def forward(self):
    ones = torch.ones(3, 4)
    sin = ones.sin();  ones = None
    return (sin,)""",
            )

    def test_cond_side_effect_in_one_branches(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        z = [torch.ones(4, 4)]

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, y, x):
                def true_fn(x):
                    z.append(x)
                    z.append(x)
                    z.pop()
                    return x.sum() + z[-1].sum()

                def false_fn(x):
                    return x.sum() - z[0].sum()

                return control_flow.cond(y, true_fn, false_fn, [x])

        mod_for_eager = Foo()
        mod_for_compile = torch.compile(
            Foo(), backend=cnt, dynamic=True, fullgraph=False
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Cond doesn't work unless it is captured completely with torch.compile",
        ):
            mod_for_eager(torch.tensor(True), torch.tensor(5))

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Cond doesn't work unless it is captured completely with torch.compile",
        ):
            mod_for_compile(torch.tensor(True), torch.tensor(5))

    def test_cond_with_constant_pred(self):
        def test(pred, x):
            def true_fn(x):
                return x

            def false_fn(x):
                return -x

            return control_flow.cond(pred, true_fn, false_fn, [x])

        opt_test = torch.compile(test, backend="eager")
        inp = torch.ones(3, 3)
        self.assertTrue(torch.allclose(test(True, inp), opt_test(True, inp)))
        self.assertTrue(torch.allclose(test(False, inp), opt_test(False, inp)))

    def test_map_graph_break(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, xs):
                def body(x):
                    self.w += 1
                    return x

                return control_flow.map(body, xs)

        mod = Module()

        mod_for_compile = torch.compile(mod, backend=cnt, dynamic=True, fullgraph=False)
        mod_for_eager = Module()

        res = mod_for_compile(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))
        # There is graph break right when we enter body of map
        self.assertEqual(len(backend.graphs), 0)
        self.assertEqual(
            res, mod_for_eager(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))
        )

    def test_map_side_effect(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        z = [torch.ones(6, 4)]

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, xs):
                def body(x):
                    z.append(x)
                    z.append(x)
                    z.pop()
                    return x + z[-1].sum()

                return control_flow.map(body, xs)

        mod = Module()

        mod_for_compile = torch.compile(mod, backend=cnt, dynamic=True, fullgraph=False)
        mod_for_eager = Module()

        res = mod_for_compile(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))
        res = mod_for_compile(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))

        eager = mod_for_eager(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))
        eager = mod_for_eager(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))

        self.assertEqual(len(backend.graphs), 0)
        self.assertEqual(res, eager)

    def test_wrap_subgraph_name_is_valid(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            return wrap(inner, x, y)

        result = f(x, y)

        self.assertEqual(result, x + y + x)
        wrap_gm = backend.graphs[0]
        names = set()
        names.update(mod_name for mod_name, _ in wrap_gm.named_modules())
        self.assertEqual(
            names,
            {
                "",
                "wrap_body_2",
                "wrap_body_2.wrap_body_1",
                "wrap_body_2.wrap_body_1.wrap_body_0",
            },
        )

    def test_wrap_allow_local_assign_in_body_fn(self):
        def f(arg1, arg2):
            def inner_f(arg1, arg2):
                a = arg1
                b = arg2
                ret = []
                for x in a:
                    ret.append(x + 1)
                for x in b:
                    ret.append(x + 1)
                return ret

            return wrap(inner_f, arg1, arg2)

        x = torch.ones(3)

        def my_args_generator():
            yield [x], [x.sin()]
            yield (x,), (x.sin(),)

        actual_graph = self._test_wrap_simple(
            f,
            my_args_generator(),
            3,
            3,
            return_graph=True,
        )

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_arg1_0_: "f32[3]", L_arg2_0_: "f32[3]"):
        l_arg1_0_ = L_arg1_0_
        l_arg2_0_ = L_arg2_0_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_arg1_0_, l_arg2_0_);  wrap_body_0 = l_arg1_0_ = l_arg2_0_ = None
        getitem: "f32[3]" = wrap[0]
        getitem_1: "f32[3]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_arg1_0_: "f32[3]", l_arg2_0_: "f32[3]"):
            child: "f32[3]" = l_arg1_0_ + 1;  l_arg1_0_ = None

            child_1: "f32[3]" = l_arg2_0_ + 1;  l_arg2_0_ = None
            return (child, child_1)
""",
        )

    def test_capture_global_num(self):
        def f(x):
            return wrap(lambda x: x + global_num, x)

        x = torch.zeros([])
        # Numbers don't get lifted, so args is still 2.
        self._test_wrap_simple(f, default_args_generator((x,)), 2)

    def test_capture_global_num_adds_guard(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return wrap(lambda x: x + global_num, x)

        global global_num
        x = torch.zeros([])
        result = f(x)
        self.assertEqual(result, x + global_num)

        global_num = torch.randn([]).item()
        result = f(x)
        self.assertEqual(result, x + global_num)

    def test_capture_input_num(self):
        def f(x, y):
            return wrap(lambda x: x + y, x)

        x = torch.zeros([])
        y = 3.14
        # Numbers don't get lifted, so args is still 2.
        self._test_wrap_simple(f, default_args_generator((x, y)), 2)

    def test_side_effect_in_body(self):
        counters.clear()
        backend = EagerAndRecordGraphs()

        x = torch.randn([])
        y = torch.randn([])

        def inner(x):
            nonlocal y
            y = x
            return x.clone()

        @torch.compile(backend=backend)
        def f(x):
            return wrap(inner, x)

        f(x)
        self.assertEqual(y, x)
        assert_dict_matches_regex(
            self,
            dict(counters["graph_break"]),
            {
                r".*HigherOrderOperator: Mutating a variable not in the current scope \(SideEffects\)": 1
            },
        )

    def test_fallback_on_graph_break_simple(self):
        # In the future, there should be a per-HigherOrderOperator switch
        # on whether or not to fallback or raise a loud error.
        # For now we just fallback by default.
        cnt = CompileCounter()
        x = torch.randn([])

        def inner(x):
            y = x.sin()
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=cnt)
        def f(x):
            return wrap(inner, x)

        result = f(x)
        self.assertEqual(result, inner(x))
        self.assertEqual(cnt.frame_count, 0)

    def test_fallback_on_graph_break_complicated(self):
        cnt = CompileCounter()
        x = torch.randn([])

        def inner(x):
            y = x.sin()
            y = y * global_var
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=cnt)
        def f(x):
            x = x.clone()
            result = wrap(inner, x)
            return result.clone()

        result = f(x)
        self.assertEqual(result, inner(x))
        self.assertEqual(cnt.frame_count, 2)

    def test_modules(self):
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return wrap(lambda x: mod(x), x)

        result = f(x)

        self.assertEqual(result, mod(x))
        self.assertEqual(cnt.frame_count, 1)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        # 3 args - 1 for input, and other 2 for the weight and bias
        self.assertTrue(len(wrap_node.args), 3)

        # Check that the linear bias and weight are getattr in the outer graph
        if not torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertTrue(len(dict(backend.graphs[0].named_parameters())) == 2)

        # Check that the inner function has one op and its a linear op
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)
        linear_node = find_first_node(body_function, torch._C._nn.linear)
        self.assertTrue(linear_node is not None)

        # Check that the innermost graph does not have any params
        self.assertTrue(len(dict(body_function.named_parameters())) == 0)
        self.assertTrue(len(dict(body_function.named_children())) == 0)

    def test_flat_list_output(self):
        def f(x):
            return wrap(lambda x: [torch.sin(x), torch.cos(x)], x)

        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 2, expected_opcount=3)

    def test_fallback_on_python_primitives_output(self):
        counters.clear()
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            return wrap(lambda x: [1, torch.sin(x), 2.0], x)

        x = torch.randn(3)
        result = f(x)
        self.assertEqual(result, [1, torch.sin(x), 2.0])
        self.assertEqual(cnt.frame_count, 0)
        assert_dict_matches_regex(
            self,
            dict(counters["graph_break"]),
            {".*HigherOrderOperator body's output must consist of tensors only": 1},
        )

    def test_nested_tuple_output(self):
        def f(x):
            ((a, b),) = wrap(lambda x: ((x.sin(), x.cos()),), x)
            return a + b

        x = torch.randn(2, 3)

        counters.clear()
        graph = self._test_wrap_simple(
            f, default_args_generator((x,)), 2, 4, return_graph=True
        )
        self.assertEqual(len(counters["graph_break"]), 0)

        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 3]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        a: "f32[2, 3]" = wrap[0]
        b: "f32[2, 3]" = wrap[1];  wrap = None

        add: "f32[2, 3]" = a + b;  a = b = None
        return (add,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[2, 3]"):
            child: "f32[2, 3]" = l_x_.sin()
            child_1: "f32[2, 3]" = l_x_.cos();  l_x_ = None
            return (child, child_1)
""",
        )

    def test_output_with_dict(self):
        def f(x):
            return wrap(lambda x: [{"a": -x}], x)

        x = torch.randn(3)

        counters.clear()
        graph = self._test_wrap_simple(
            f, default_args_generator((x,)), 2, 2, return_graph=True
        )
        self.assertEqual(len(counters["graph_break"]), 0)

        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3]"):
            child: "f32[3]" = -l_x_;  l_x_ = None
            return (child,)
""",
        )

    def test_access_module_attr(self):
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            y = mod(x)
            return wrap(lambda y: y - mod.bias, y)

        result = f(x)
        self.assertEqual(result, mod(x) - mod.bias)
        self.assertEqual(cnt.frame_count, 1)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)

        # Check that the linear bias and weight are getattr in the outer graph
        if not torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertTrue(len(dict(backend.graphs[0].named_parameters())) == 2)

        # Check that the inner function has one op and its a linear op
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)

        # Check that the innermost graph does not have any params
        self.assertTrue(len(dict(body_function.named_parameters())) == 0)
        self.assertTrue(len(dict(body_function.named_children())) == 0)

    def test_make_closure(self):
        def f(x, y):
            def g(x):
                return x + y

            return g(x)

        def h(x, y):
            return wrap(f, x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(h, default_args_generator((x, y)), 3)

    def test_internal_nonlocal(self):
        def f(x, y):
            w = 1

            def g(x):
                nonlocal w
                w = x
                return x

            def h(x):
                nonlocal w
                w = w + 1
                return x

            g(x)
            h(x)
            return w + y

        def h(x, y):
            return wrap(f, x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(h, default_args_generator((x, y)), 3)

    def test_capture_numpy_number(self):
        import numpy as np

        y = np.float32(1.0)

        def f(x):
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        # np.number are lifted to graph inputs
        self._test_wrap_simple(f, default_args_generator((x,)), 3)

    def test_freevars_as_inputs_to_wrap(self):
        y = torch.randn(3)

        def f(x):
            return wrap(lambda x, y: x + y, x, y)

        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 3)

    def test_lift_tensor_constant(self):
        def f(x):
            y = torch.tensor(1.0)
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 3, expected_opcount=3)

    def test_nested_wrap(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        mod = MockModule()

        # Two levels of wrap ops
        def gn(x):
            return torch.cos(x) + wrap(mod, x)

        def fn(x):
            return wrap(gn, x)

        self._test_wrap_simple(fn, default_args_generator((torch.randn(10, 10),)), 4)

    def test_fn_with_kwargs_in_torch_ops(self):
        def fn(x):
            return wrap(lambda z: torch.cos(input=z), x)

        x = torch.randn(3)
        self._test_wrap_simple(fn, default_args_generator((x,)), 2)

    def test_hooks(self):
        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        forward_handles = {}
        activations = {}

        def save_activations(mod, inp, out):
            activations[name] = inp

        for name, module in model.named_children():
            forward_handles[name] = module.register_forward_hook(save_activations)

        @torch.compile(backend="eager")
        def fn(x):
            return wrap(lambda x: model(x), x)

        for i in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            activations.clear()
            x = torch.randn((10, 10))
            pred = fn(x)
            loss = pred.sum()
            loss.backward()

        self.assertTrue(activations.keys() == forward_handles.keys())

    def _get_source_fn_stack(self, gm, node_names):
        ret = {}
        for mod in gm.modules():
            for node in mod.graph.nodes:
                if node.name in node_names:
                    actual_stack = [
                        name for name, _ in node.meta.get("source_fn_stack", [])
                    ]
                    ret[node.name] = actual_stack
        return ret

    def test_wrap_source_fn_stack(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        mod = MockModule()

        def gn(x):
            return torch.cos(x) + wrap(mod, x)

        def fn(x):
            return wrap(gn, x)

        backend = EagerAndRecordGraphs()
        inp = torch.randn((4, 4))
        torch.compile(fn, backend=backend, fullgraph=True)(inp)

        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "linear"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """\
{'add': ['wrap', 'add'],
 'cos': ['wrap', 'cos'],
 'linear': ['wrap', 'wrap', 'linear']}""",
        )

    def test_cond_source_fn_stack(self):
        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def cond_f(pred, pred2, x, y):
            def true_fn(pred2, x, y):
                return x + y

            def false_fn(pred2, x, y):
                def true_fn2(x, y):
                    return x.sin() - y.cos()

                def false_fn2(x, y):
                    return x.cos() - y.sin()

                return control_flow.cond(pred2, true_fn2, false_fn2, [x, y])

            return control_flow.cond(pred, true_fn, false_fn, [pred2, x, y])

        pred = torch.tensor(True)
        pred2 = torch.tensor(False)
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3, 3)
        cond_f(pred, pred2, xs, y)

        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "sin", "sub"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """\
{'add': ['cond', 'add'],
 'cos': ['cond', 'cond', 'cos'],
 'sin': ['cond', 'cond', 'sin'],
 'sub': ['cond', 'cond', 'sub']}""",
        )

    def test_map_source_fn_stack(self):
        backend = EagerAndRecordGraphs()

        xs = torch.randn(2, 3, 3)
        y = torch.randn(3)

        @torch.compile(backend=backend, fullgraph=True)
        def map_f(xs, y):
            def inner(x, y):
                def inner2(x, y):
                    return x + y

                return control_flow.map(inner2, x, y) * y.cos()

            return control_flow.map(inner, xs, y).sin()

        result = map_f(xs, y)

        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "sin"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """{'add': ['map', 'map', 'add'], 'cos': ['map', 'cos'], 'sin': ['sin']}""",
        )

    def test_grad_source_fn_stack(self):
        backend = EagerAndRecordGraphs()

        def fn(x):
            return x.sin().sum()

        @torch.compile(backend=backend, fullgraph=False)
        def wrapper_fn(x):
            return torch.func.grad(torch.func.grad(fn))(x)

        x = torch.randn(())

        wrapper_fn(x)
        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"sum_1", "sin"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """{'sin': ['sin']}""",
        )

    def test_vmap_multiply_scalar(self):
        @torch.compile(backend="inductor", fullgraph=True)
        def g(x):
            return torch.vmap(torch.mul, in_dims=(0, None))(x, 3.14)

        x = torch.randn(3)
        y = g(x)
        self.assertEqual(y, x * 3.14)

        @torch.compile(backend="inductor", fullgraph=True)
        def f(x):
            return torch.vmap(torch.mul, in_dims=(0, None))(x, 314)

        x = torch.randn(3)
        y = f(x)
        self.assertEqual(y, x * 314)

    def test_vmap_source_fn_stack(self):
        backend = EagerAndRecordGraphs()

        def inner_fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            return torch.func.vmap(lambda x: inner_fn(x.cos()))(x)

        x = torch.randn(3, 3, 3, 3)
        fn(x)
        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(
            gm, {"sum_1", "sum_2", "batched_output"}
        )
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """{'sum_1': ['sum_1'], 'sum_2': ['sum_2']}""",
        )

    def test_cond_pytree_operands(self):
        def _construct_pytree():
            a = torch.randn(3, 3)
            b = torch.randn(3, 3)
            c = torch.randn(3, 3)
            d = torch.randn(3, 3)
            e = torch.randn(3, 3)
            f = torch.randn(3, 3)
            g = torch.randn(3, 3)
            return (a, [[[b]]], c, (d, (e,), f), {"g": g})

        pred = torch.tensor(True)
        inp = _construct_pytree()

        def _reduce_sum(flattened):
            init = 0
            for val in flattened:
                init += val
            return init

        def _reduce_max(flattened):
            init = flattened[0]
            for val in flattened:
                init = max(val, init)
            return init

        def true_fn(pytree_in):
            flattened, spec = pytree.tree_flatten(pytree_in)
            return _reduce_sum(flattened)

        def false_fn(pytree_in):
            flattened, spec = pytree.tree_flatten(pytree_in)
            return _reduce_max(flattened)

        def fn(pred, pytree_in):
            return torch.cond(pred, true_fn, false_fn, [pytree_in])

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        compiled_res = torch.compile(fn, backend=backend)(pred, inp)
        eager_res = fn(pred, inp)
        self.assertEqual(compiled_res, eager_res)
        graph = backend.graphs[0]

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            graph.code.strip(),
            """\
def forward(self, L_pred_ : torch.Tensor, L_pytree_in_0_ : torch.Tensor, L_pytree_in_1_0_0_0_ : torch.Tensor, L_pytree_in_2_ : torch.Tensor, L_pytree_in_3_0_ : torch.Tensor, L_pytree_in_3_1_0_ : torch.Tensor, L_pytree_in_3_2_ : torch.Tensor, L_pytree_in_4_g_ : torch.Tensor):
    l_pred_ = L_pred_
    l_pytree_in_0_ = L_pytree_in_0_
    l_pytree_in_1_0_0_0_ = L_pytree_in_1_0_0_0_
    l_pytree_in_2_ = L_pytree_in_2_
    l_pytree_in_3_0_ = L_pytree_in_3_0_
    l_pytree_in_3_1_0_ = L_pytree_in_3_1_0_
    l_pytree_in_3_2_ = L_pytree_in_3_2_
    l_pytree_in_4_g_ = L_pytree_in_4_g_
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(l_pred_, cond_true_0, cond_false_0, [l_pytree_in_0_, l_pytree_in_1_0_0_0_, l_pytree_in_2_, l_pytree_in_3_0_, l_pytree_in_3_1_0_, l_pytree_in_3_2_, l_pytree_in_4_g_]);  l_pred_ = cond_true_0 = cond_false_0 = l_pytree_in_0_ = l_pytree_in_1_0_0_0_ = l_pytree_in_2_ = l_pytree_in_3_0_ = l_pytree_in_3_1_0_ = l_pytree_in_3_2_ = l_pytree_in_4_g_ = None
    getitem = cond[0];  cond = None
    return (getitem,)""",  # noqa: B950
        )

    def test_cond_pytree_operands_with_non_tensor_leaves(self):
        def fn(pred, pytree_in):
            return torch.cond(
                pred, lambda x: x[0] + 1, lambda x: x[0] * 2, (pytree_in,)
            )

        pred = torch.tensor(True)
        for pytree_in in [(1,), ("string",), (1.0,)]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"Expect operands to be a tuple of possibly nested dict/list/tuple",
            ):
                fn(pred, pytree_in)

        for pytree_in in [(1,), ("string",), (1.0,)]:
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Cond doesn't work unless it is captured completely with torch.compile",
            ):
                torch.compile(fn, backend="eager")(pred, pytree_in)


class HigherOrderOpVmapGuardTests(LoggingTestCase):
    @make_logging_test(recompiles=True)
    def test_vmap_grad_guard_ok(self, records):
        vmap = torch.vmap
        grad = torch.func.grad

        def g(x):
            return vmap(grad(torch.sin))(x)

        @torch.compile(backend="eager")
        def fn(x):
            return vmap(g)(x)

        x = torch.randn(4, 5)
        y = fn(x)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(x.cos(), y)

        # Calling the same function again won't have any effect on guards
        fn(x)
        self.assertEqual(len(records), 0)

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_grad_guard_fail(self, records):
        grad = torch.func.grad

        @torch.compile(backend="eager")
        def fn(x):
            return grad(torch.sin)(x.sum())

        x = torch.randn([])
        fn(x)
        self.assertEqual(len(records), 0)

        # calling again should not invalidate the graph
        fn(x)
        self.assertEqual(len(records), 0)

        # call grad should retrigger compilation
        x = torch.randn(3)
        grad(fn)(x)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([])""",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(recompiles=True)
    def test_dual_level_guard(self, records):
        fwAD = torch.autograd.forward_ad

        @torch.compile(backend="eager", fullgraph=True)
        def fn(foo, tangent):
            with fwAD.dual_level():
                dual = fwAD.make_dual(foo, tangent[1:])
                return dual

        foo = torch.rand(2)
        tangent = torch.rand(3)
        fn(foo, tangent)
        self.assertEqual(len(records), 0)

        # calling again should not invalidate the graph
        fn(foo, tangent)
        self.assertEqual(len(records), 0)

        # assertRaises is only here because Nested forward mode AD is not supported
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError):
            with fwAD.dual_level():
                fn(foo, tangent)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "forward_ad")
        self.assertIn(
            """torch.autograd.forward_ad._current_level == -1""",
            munge_exc(record.getMessage()),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_jvp_guard_fail(self, records):
        jvp = torch.func.jvp
        vmap = torch.func.vmap

        @torch.compile(backend="eager")
        def fn(x):
            return jvp(torch.sin, (x,), (x,))

        x = torch.randn(3, 4)
        fn(x)
        self.assertEqual(len(records), 0)

        # calling again should not invalidate the graph
        fn(x)
        self.assertEqual(len(records), 0)

        # call jvp should retrigger compilation
        x = torch.randn(3, 4, 5)
        jvp(vmap(fn), (x,), (x,))

        self.assertGreater(len(records), 0)
        if self.hasRecord(records, "pyfunctorch"):
            record = self.getRecord(records, "pyfunctorch")
            self.assertIn(
                """torch._functorch.pyfunctorch.compare_functorch_state([])""",
                munge_exc(record.getMessage()),
            )
        elif self.hasRecord(records, "forward_ad"):
            record = self.getRecord(records, "forward_ad")
            self.assertIn(
                """torch.autograd.forward_ad._current_level == -1""",
                munge_exc(record.getMessage()),
            )

    @make_logging_test(recompiles=True)
    def test_vmap_guard_ok(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.randn(3, 3, 4, 5)
        y = fn(x)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(x.sin(), y)

        # Calling the same function again won't have any effect on guards
        z = fn(x)
        self.assertEqual(len(records), 0)
        self.assertEqual(x.sin(), z)

        # calling with a different object will also not affect guards
        w = fn(z)
        self.assertEqual(len(records), 0)
        self.assertEqual(z.sin(), w)

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_guard_fail_different_state(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 4)
        y = torch.vmap(fn, randomness="same")(x)
        self.assertEqual(x.sin(), y)
        self.assertEqual(len(records), 0)

        # call vmap(vmap(fn))(x) should retrigger compilation
        y = torch.vmap(fn, randomness="different")(x)
        self.assertEqual(x.sin(), y)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'same')])""",
            record.getMessage(),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_guard_fail(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        y = torch.vmap(fn)(x)
        self.assertEqual(x.sin(), y)
        self.assertEqual(len(records), 0)

        # call vmap(vmap(fn))(x) should retrigger compilation as
        # _functorch.current_level() is not the same
        x = torch.zeros(3, 3, 3, 4, 5)
        y = torch.vmap(torch.vmap(fn))(x)
        self.assertEqual(x.sin(), y)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            record.getMessage(),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_grad_vmap_guard_fail(self, records):
        vmap = torch.vmap
        grad = torch.func.grad

        def g(x):
            y = vmap(torch.sin, randomness="same")(x)
            return y.sum(0)

        @torch.compile(backend="eager")
        def fn(x):
            return grad(g)(x)

        x = torch.randn(3, 3)
        y = vmap(fn, randomness="error")(x)
        self.assertEqual(x.cos(), y)

        # previous FX graph should be invalidated
        x = torch.randn(3, 3, 4)
        y = vmap(vmap(fn, randomness="different"))(x)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            munge_exc(record.getMessage()),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_recompile_different_states(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        y = torch.vmap(fn, randomness="same")(x)
        self.assertEqual(len(records), 0)  # sanity check

        y = torch.vmap(fn, randomness="different")(x)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'same')])""",
            munge_exc(record.getMessage()),
        )

    @config.patch(capture_func_transforms=True)
    @make_logging_test(guards=True)
    def test_emit_functorch_guard_if_active(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.sin(x)

        x = torch.randn(3, 4)
        _ = fn(x)
        self.assertFalse(self.hasRecord(records, "pyfunctorch"))  # sanity check

        _ = torch.vmap(fn)(x)
        self.assertTrue(self.hasRecord(records, "pyfunctorch"))
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(recompiles=True)
    def test_linearize_recompiles(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            out, jvp_fn = torch.func.linearize(torch.sin, x)
            return out, jvp_fn(x)

        x = torch.randn(2, 3)
        fn(x)
        self.assertEqual(len(records), 0)

        z = torch.randn(2, 3)
        fn(z)
        self.assertEqual(len(records), 0)

        y = torch.randn(3, 4)
        fn(y)
        self.assertGreater(len(records), 0)


class FuncTorchHigherOrderOpTests(torch._dynamo.test_case.TestCase):
    def tearDown(self):
        # Ensure that in the case of a test failure, the next test won't fail
        # because of a previous call to _vmap_increment_nesting that wasn't undone
        # i.e. test_vmap_free_tensor fails when PYTORCH_TEST_WITH_DYNAMO=1
        # and the call to increment nesting is not undone
        if not TEST_WITH_TORCHDYNAMO:
            return

        warn = False
        while ci := torch._C._functorch.peek_interpreter_stack():
            if ci.key() == torch._C._functorch.TransformType.Vmap:
                warn = True
                torch._C._functorch._vmap_decrement_nesting()
            else:
                break

        if warn:
            msg = (
                "Interpreter stack is not empty. Test should have called "
                "'torch._C._functorch._vmap_decrement_nesting()'"
            )
            warnings.warn(msg)

    def _compile_check(self, fn, inputs, fullgraph=True, graph_idx=0):
        backend = EagerAndRecordGraphs()
        actual = fn(*inputs)
        expected = torch.compile(fn, backend=backend, fullgraph=fullgraph)(*inputs)

        self.assertEqual(actual, expected)

        wrapped_gm = backend.graphs[graph_idx]
        return wrapped_gm

    def test_hessian(self):
        counters.clear()

        def wrapper_fn(x):
            return torch.func.hessian(torch.sin)(x)

        x = torch.randn(4, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]"):
        l_x_ = L_x_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_x_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 4, 3]" = chunk.view(12, 4, 3);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        child_1 = torch._C._functorch._add_batch_dim(child, 0, 1);  child = None

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (child_1,));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        child_2 = torch._make_dual(l_x_, child_1, level = 0);  child_1 = None

        _wrap_for_grad = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_primals = torch._C._functorch._wrap_for_grad(child_2, 3);  child_2 = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        o = torch.sin(diff_primals)

        results = torch._C._functorch._unwrap_for_grad(o, 3)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        tensor_1 = torch.tensor((12,))
        cumsum_1 = tensor_1.cumsum(dim = 0);  tensor_1 = None
        getitem_1 = cumsum_1[slice(None, -1, None)];  cumsum_1 = None
        neg_1 = getitem_1.neg();  getitem_1 = None
        unbind_1 = neg_1.unbind();  neg_1 = unbind_1 = None

        chunk_1 = results.new_zeros(12, 12);  results = None

        diagonal_1 = chunk_1.diagonal(0)
        fill__1 = diagonal_1.fill_(1);  diagonal_1 = fill__1 = None

        basis = chunk_1.view(12, 4, 3);  chunk_1 = None

        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_1 = torch._C._functorch._add_batch_dim(basis, 0, 3);  basis = None

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare(o, _add_batch_dim_1);  _vjp_treespec_compare = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([o], [diff_primals], [_add_batch_dim_1], retain_graph = True, create_graph = True);  o = diff_primals = _add_batch_dim_1 = None
        batched_outputs = _autograd_grad[0];  _autograd_grad = None

        chunked_result = torch._C._functorch._remove_batch_dim(batched_outputs, 3, 12, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = chunked_result.split((12,), dim = 0);  chunked_result = None
        split_1 = split[0];  split = None

        output_input = split_1.view((4, 3, 4, 3));  split_1 = None

        _unpack_dual = torch._unpack_dual(output_input, level = 0);  output_input = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[4, 3, 4, 3]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = primals_out_unflatten = None

        tangents_out_unflatten = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        results_1: "f32[12, 4, 3, 4, 3]" = torch._C._functorch._remove_batch_dim(tangents_out_unflatten, 1, 12, 0);  tangents_out_unflatten = None

        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None

        movedim: "f32[4, 3, 4, 3, 12]" = results_1.movedim(0, -1);  results_1 = None
        split_2 = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[4, 3, 4, 3, 12]" = split_2[0];  split_2 = None

        unflatten: "f32[4, 3, 4, 3, 4, 3]" = jac_out_in.unflatten(-1, (4, 3));  jac_out_in = None
        return (unflatten,)
""",
        )

    def test_hessian_argnums(self):
        counters.clear()

        def fn(x, y):
            return x.sin()

        def wrapper_fn(x, y):
            return torch.func.hessian(fn, argnums=(1,))(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            "\n".join(actual.split("\n")[:-2]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_y_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 3, 4]" = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        child_1 = torch._C._functorch._add_batch_dim(child, 0, 1);  child = None

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_y_,), (child_1,));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        child_3 = torch._make_dual(l_y_, child_1, level = 0);  child_1 = None

        child_2 = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = None
        _wrap_for_grad_1 = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = _wrap_for_grad_1 = None

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad_2 = torch._C._functorch._wrap_for_grad(child_2, 3);  child_2 = None
        child_4 = torch._C._functorch._wrap_for_grad(child_3, 3);  child_3 = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(child_4);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        o = _wrap_for_grad_2.sin();  _wrap_for_grad_2 = None

        results = torch._C._functorch._unwrap_for_grad(o, 3)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        tensor_1 = torch.tensor((12,))
        cumsum_1 = tensor_1.cumsum(dim = 0);  tensor_1 = None
        getitem_1 = cumsum_1[slice(None, -1, None)];  cumsum_1 = None
        neg_1 = getitem_1.neg();  getitem_1 = None
        unbind_1 = neg_1.unbind();  neg_1 = unbind_1 = None

        chunk_1 = results.new_zeros(12, 12);  results = None

        diagonal_1 = chunk_1.diagonal(0)
        fill__1 = diagonal_1.fill_(1);  diagonal_1 = fill__1 = None

        basis = chunk_1.view(12, 4, 3);  chunk_1 = None

        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_1 = torch._C._functorch._add_batch_dim(basis, 0, 3);  basis = None

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare(o, _add_batch_dim_1);  _vjp_treespec_compare = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([o], [child_4], [_add_batch_dim_1], retain_graph = True, create_graph = True);  o = child_4 = _add_batch_dim_1 = None
        child_5 = _autograd_grad[0];  _autograd_grad = None

        child_6 = torch._C._functorch._remove_batch_dim(child_5, 3, 12, 0);  child_5 = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = child_6.split((12,), dim = 0);  child_6 = None
        split_1 = split[0];  split = None

        child_7 = split_1.view((4, 3, 3, 4));  split_1 = None

        _unpack_dual = torch._unpack_dual(child_7, level = 0);  child_7 = None
        primal = _unpack_dual[0];  _unpack_dual = None

        tangent = torch.zeros_like(primal)

        child_8: "f32[4, 3, 3, 4]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = child_8 = None

        child_9: "f32[4, 3, 3, 4]" = torch._C._functorch._unwrap_for_grad(tangent, 2);  tangent = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        child_10: "f32[12, 4, 3, 3, 4]" = torch._C._functorch._remove_batch_dim(child_9, 1, 12, 0);  child_9 = None

        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None

        movedim: "f32[4, 3, 3, 4, 12]" = child_10.movedim(0, -1);  child_10 = None
        split_2 = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[4, 3, 3, 4, 12]" = split_2[0];  split_2 = None

        unflatten: "f32[4, 3, 3, 4, 3, 4]" = jac_out_in.unflatten(-1, (3, 4));  jac_out_in = None""",
        )

        self.assertExpectedInline(
            actual.split("\n")[-2],
            """        return (unflatten,)""",
        )

    def test_hessian_disable_capture(self):
        counters.clear()

        with config.patch(capture_func_transforms=False):
            # We have verified above that this
            # function compiles
            def wrapper_fn(x):
                return torch.func.hessian(torch.sin)(x)

            x = torch.randn(3, 3, 3)
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            self.assertEqual(len(counters["graph_break"]), 2)
            self.assertEqual(
                {
                    "torch.func.vmap capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 2,
                    "torch.func.hessian capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 1,
                },
                dict(counters["graph_break"]),
            )
            self.assertEqual(actual, expected)

    def test_jacrev(self):
        counters.clear()

        def wrapper_fn(x):
            return torch.func.jacrev(torch.sin)(x)

        x = torch.randn(4, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_primals = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        o = torch.sin(diff_primals)

        results: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(o, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = results.new_zeros(12, 12);  results = None

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        basis: "f32[12, 4, 3]" = chunk.view(12, 4, 3);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim = torch._C._functorch._add_batch_dim(basis, 0, 1);  basis = None

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare(o, _add_batch_dim);  _vjp_treespec_compare = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([o], [diff_primals], [_add_batch_dim], retain_graph = True, create_graph = True);  o = diff_primals = _add_batch_dim = None
        batched_outputs = _autograd_grad[0];  _autograd_grad = None

        chunked_result: "f32[12, 4, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 12, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = chunked_result.split((12,), dim = 0);  chunked_result = None
        split_1: "f32[12, 4, 3]" = split[0];  split = None

        output_input: "f32[4, 3, 4, 3]" = split_1.view((4, 3, 4, 3));  split_1 = None
        return (output_input,)
""",
        )

    def test_jacrev_two_tensors_argnums(self):
        counters.clear()

        def fn(x, y):
            return y.sin()

        def wrapper_fn(x, y):
            return torch.func.jacrev(fn, argnums=1)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = _wrap_for_grad = None
        diff_primals = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        o = diff_primals.sin()

        results: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(o, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = results.new_zeros(12, 12);  results = None

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        basis: "f32[12, 3, 4]" = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim = torch._C._functorch._add_batch_dim(basis, 0, 1);  basis = None

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare(o, _add_batch_dim);  _vjp_treespec_compare = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([o], [diff_primals], [_add_batch_dim], retain_graph = True, create_graph = True);  o = diff_primals = _add_batch_dim = None
        batched_outputs = _autograd_grad[0];  _autograd_grad = None

        chunked_result: "f32[12, 3, 4]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 12, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = chunked_result.split((12,), dim = 0);  chunked_result = None
        split_1: "f32[12, 3, 4]" = split[0];  split = None

        output_input: "f32[3, 4, 3, 4]" = split_1.view((3, 4, 3, 4));  split_1 = None
        return (output_input,)
""",
        )

    def test_jacrev_has_aux(self):
        counters.clear()

        def fn(x, y):
            return y.sin(), x

        def wrapper_fn(x, y):
            return torch.func.jacrev(fn, argnums=1, has_aux=True)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        aux = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        diff_primals = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        o = diff_primals.sin()

        aux_1: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        results: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(o, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = results.new_zeros(12, 12);  results = None

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        basis: "f32[12, 3, 4]" = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim = torch._C._functorch._add_batch_dim(basis, 0, 1);  basis = None

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare(o, _add_batch_dim);  _vjp_treespec_compare = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([o], [diff_primals], [_add_batch_dim], retain_graph = True, create_graph = True);  o = diff_primals = _add_batch_dim = None
        batched_outputs = _autograd_grad[0];  _autograd_grad = None

        chunked_result: "f32[12, 3, 4]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 12, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = chunked_result.split((12,), dim = 0);  chunked_result = None
        split_1: "f32[12, 3, 4]" = split[0];  split = None

        output_input: "f32[3, 4, 3, 4]" = split_1.view((3, 4, 3, 4));  split_1 = None
        return (output_input, aux_1)
""",
        )

    def test_jacrev_disable_capture(self):
        counters.clear()

        with config.patch(capture_func_transforms=False):
            # We have verified above that this
            # function compiles
            def wrapper_fn(x):
                return torch.func.jacrev(torch.sin)(x)

            x = torch.randn(3, 3, 3)
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            self.assertEqual(len(counters["graph_break"]), 2)
            self.assertEqual(
                dict(counters["graph_break"]),
                {
                    "torch.func.vmap capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 2,
                    "torch.func.jacrev capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 1,
                },
            )
            self.assertEqual(actual, expected)

    def test_vjp(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x, v):
            (out, vjpfunc) = torch.func.vjp(fn, x)
            return out

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        child_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(child);  child_1 = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin = child.sin();  child = None
        o = sin.sum();  sin = None

        results: "f32[]" = torch._C._functorch._unwrap_for_grad(o, 1);  o = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (results,)
""",
        )

    def test_vjp_multiple_outputs(self):
        counters.clear()

        def wrapper_fn(x, v):
            fn = lambda x: (x.sin(), x.cos())  # noqa: E731
            (out, vjpfunc) = torch.func.vjp(fn, x)
            vjps = vjpfunc((v, v))
            return out, vjps

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]", L_v_: "f32[5]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        child_3 = torch._functorch.eager_transforms._set_tensor_requires_grad(child)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        child_1 = child.sin()
        child_2 = child.cos();  child = None

        _unwrap_for_grad: "f32[5]" = torch._C._functorch._unwrap_for_grad(child_1, 1)
        _unwrap_for_grad_1: "f32[5]" = torch._C._functorch._unwrap_for_grad(child_2, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare((child_1, child_2), (l_v_, l_v_));  _vjp_treespec_compare = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([child_1, child_2], [child_3], [l_v_, l_v_], retain_graph = True, create_graph = True);  child_1 = child_2 = child_3 = l_v_ = None
        getitem: "f32[5]" = _autograd_grad[0];  _autograd_grad = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, getitem)
""",
        )

    def test_vjp_multiple_outputs_python_struct(self):
        counters.clear()

        def wrapper_fn(x, v):
            fn = lambda x: {"first": x.sin(), "second": x.cos()}  # noqa: E731
            (out, vjpfunc) = torch.func.vjp(fn, x)
            vjps = vjpfunc({"first": v, "second": v.sin()})
            return out, vjps

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]", L_v_: "f32[5]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        child_3 = torch._functorch.eager_transforms._set_tensor_requires_grad(child)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        child_1 = child.sin()
        child_2 = child.cos();  child = None

        _unwrap_for_grad: "f32[5]" = torch._C._functorch._unwrap_for_grad(child_1, 1)
        _unwrap_for_grad_1: "f32[5]" = torch._C._functorch._unwrap_for_grad(child_2, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        child_4: "f32[5]" = l_v_.sin()

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare({'first': child_1, 'second': child_2}, {'first': l_v_, 'second': child_4});  _vjp_treespec_compare = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([child_1, child_2], [child_3], [l_v_, child_4], retain_graph = True, create_graph = True);  child_1 = child_2 = child_3 = l_v_ = child_4 = None
        getitem: "f32[5]" = _autograd_grad[0];  _autograd_grad = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, getitem)
""",
        )

    def test_vjp_has_aux(self):
        counters.clear()

        def fn(x):
            return x.sin().sum(), x

        def wrapper_fn(x, v):
            (out, vjpfunc, _) = torch.func.vjp(fn, x, has_aux=True)
            return out

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        child_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(child);  child_1 = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin = child.sin()
        o = sin.sum();  sin = None

        aux: "f32[5]" = torch._C._functorch._unwrap_for_grad(child, 1);  child = aux = None

        results: "f32[]" = torch._C._functorch._unwrap_for_grad(o, 1);  o = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (results,)
""",
        )

    def test_vjp_disable_capture(self):
        counters.clear()

        with config.patch(capture_func_transforms=False):
            # We have verified above that this
            # function compiles
            def wrapper_fn(x):
                (out, vjpfunc) = torch.func.vjp(torch.sin, x)
                return out

            x = torch.randn(3, 3, 3)
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            self.assertEqual(len(counters["graph_break"]), 1)
            self.assertEqual(
                dict(counters["graph_break"]),
                {
                    "torch.func.vjp capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 1
                },
            )
            self.assertEqual(actual, expected)

    @config.patch(inline_inbuilt_nn_modules=True)
    def test_functional_call(self):
        def wrapper_fn(model, params, inputs, targets):
            prediction = torch.func.functional_call(model, params, (inputs,))
            return torch.nn.functional.mse_loss(prediction, targets)

        model = torch.nn.Linear(3, 3)
        params = dict(model.named_parameters())
        inputs = torch.randn(64, 3)
        targets = torch.randn(64, 3)

        wrapped_gm = self._compile_check(wrapper_fn, (model, params, inputs, targets))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        if torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_model_parameters_weight_: "f32[3, 3]", L_model_parameters_bias_: "f32[3]", L_inputs_: "f32[64, 3]", L_targets_: "f32[64, 3]"):
        l_model_parameters_weight_ = L_model_parameters_weight_
        l_model_parameters_bias_ = L_model_parameters_bias_
        l_inputs_ = L_inputs_
        l_targets_ = L_targets_

        prediction: "f32[64, 3]" = torch._C._nn.linear(l_inputs_, l_model_parameters_weight_, l_model_parameters_bias_);  l_inputs_ = l_model_parameters_weight_ = l_model_parameters_bias_ = None

        mse_loss: "f32[]" = torch.nn.functional.mse_loss(prediction, l_targets_);  prediction = l_targets_ = None
        return (mse_loss,)
""",
            )
        else:
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_: "f32[64, 3]", L_targets_: "f32[64, 3]"):
        l_inputs_ = L_inputs_
        l_targets_ = L_targets_

        prediction: "f32[64, 3]" = self.model(l_inputs_);  l_inputs_ = None

        mse_loss: "f32[]" = torch.nn.functional.mse_loss(prediction, l_targets_);  prediction = l_targets_ = None
        return (mse_loss,)
""",
            )

    @config.patch(inline_inbuilt_nn_modules=True)
    def test_functional_call_sequential_params_and_buffers(self):
        # copied from test/test_stateless.py
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = torch.nn.Linear(1, 1)
                self.register_buffer("buffer", torch.ones(1))
                self.foo = 0.0

            def forward(self, x):
                return self.l1(x) + self.buffer

        def wrapper_fn(model, params, buffers, inputs):
            # two separate dictionaries
            return torch.func.functional_call(model, (params, buffers), inputs)

        model = MockModule()
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        inputs = torch.tensor([[1.5]])

        wrapped_gm = self._compile_check(
            wrapper_fn, (model, params, buffers, inputs), fullgraph=False
        )
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        if torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_params_l1_weight_: "f32[1, 1]", L_params_l1_bias_: "f32[1]", L_buffers_buffer_: "f32[1]", L_inputs_: "f32[1, 1]"):
        l_params_l1_weight_ = L_params_l1_weight_
        l_params_l1_bias_ = L_params_l1_bias_
        l_buffers_buffer_ = L_buffers_buffer_
        l_inputs_ = L_inputs_

        linear: "f32[1, 1]" = torch._C._nn.linear(l_inputs_, l_params_l1_weight_, l_params_l1_bias_);  l_inputs_ = l_params_l1_weight_ = l_params_l1_bias_ = None
        add: "f32[1, 1]" = linear + l_buffers_buffer_;  linear = l_buffers_buffer_ = None
        return (add,)
""",
            )
        else:
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[1, 1]"):
        l_x_ = L_x_

        l__self___l1: "f32[1, 1]" = self.L__self___l1(l_x_);  l_x_ = None
        l__self___buffer: "f32[1]" = self.L__self___buffer
        add: "f32[1, 1]" = l__self___l1 + l__self___buffer;  l__self___l1 = l__self___buffer = None
        return (add,)
""",
            )

    @config.patch(inline_inbuilt_nn_modules=True)
    def test_functional_call_disable_capture(self):
        counters.clear()

        with config.patch(capture_func_transforms=False):
            # We have verified above that this
            # function compiles
            def wrapper_fn(model, params, inputs, targets):
                prediction = torch.func.functional_call(model, params, (inputs,))
                return torch.nn.functional.mse_loss(prediction, targets)

            model = torch.nn.Linear(3, 3)
            params = dict(model.named_parameters())
            inputs = torch.randn(64, 3)
            targets = torch.randn(64, 3)

            actual = wrapper_fn(model, params, inputs, targets)
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                model, params, inputs, targets
            )
            self.assertEqual(len(counters["graph_break"]), 1)
            self.assertEqual(
                {
                    "torch.func.functional_call capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 1,
                },
                dict(counters["graph_break"]),
            )
            self.assertEqual(actual, expected)

    @config.patch(inline_inbuilt_nn_modules=False)
    def test_functional_call_disable_inline_nn_module(self):
        counters.clear()

        def wrapper_fn(model, params, inputs, targets):
            prediction = torch.func.functional_call(model, params, (inputs,))
            return torch.nn.functional.mse_loss(prediction, targets)

        model = torch.nn.Linear(3, 3)
        params = dict(model.named_parameters())
        inputs = torch.randn(64, 3)
        targets = torch.randn(64, 3)

        actual = wrapper_fn(model, params, inputs, targets)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
            model, params, inputs, targets
        )
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(
            {
                "torch.func.functional_call capture is disabled, it can be "
                "turned on by setting `torch._dynamo.config.inline_inbuilt_nn_modules=True`": 1,
            },
            dict(counters["graph_break"]),
        )
        self.assertEqual(actual, expected)

    def test_grad(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin = diff_args.sin()
        output = sin.sum();  sin = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1,)
""",
        )

    def test_grad_freevar_tensor(self):
        counters.clear()
        y = torch.randn(3, 3)

        def fn(x):
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        expected = wrapper_fn(x)
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(actual, expected)

    def test_grad_freevar_python_scalar(self):
        counters.clear()
        y = 3

        def fn(x):
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin = diff_args.sin()
        add = sin + 3;  sin = None
        output = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1,)
""",
        )

    def test_grad_capture_tensor(self):
        counters.clear()

        def wrapper_fn(x):
            y = torch.randn(3)

            def fn(x):
                return (x.sin() + y).sum()

            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)

        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        y: "f32[3]" = torch.randn(3)

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin = diff_args.sin()
        add = sin + y;  sin = None
        output = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (y, grad_input_1)
""",
        )

    def test_grad_closure_scalar(self):
        counters.clear()

        def wrapper_fn(x):
            y = 3.14

            def fn(x):
                return (x.sin() + y).sum()

            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)

        # Graph break because dynamo is unable to get source `fn` and
        # functools.wraps in `grad` leads to graph-break
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin = diff_args.sin()
        add = sin + 3.14;  sin = None
        output = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1,)
""",
        )

    def test_grad_has_aux(self):
        counters.clear()

        y = 3.14

        def fn(x):
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn(x):
            return torch.func.grad(fn, has_aux=True)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin = diff_args.sin()
        add = sin + 3.14;  sin = None
        output = add.sum();  add = None
        aux = diff_args.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1, aux_1)
""",
        )

    def test_grad_two_tensor_has_aux(self):
        counters.clear()

        def fn(x, y):
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn(x, y):
            return torch.func.grad(fn, has_aux=True)(x, y)

        y = torch.randn(3, 3, 3)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        _wrap_for_grad_1 = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin = diff_args.sin()
        add = sin + _wrap_for_grad_1;  sin = _wrap_for_grad_1 = None
        output = add.sum();  add = None
        aux = diff_args.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1, aux_1)
""",
        )

    def test_grad_two_tensor_all_grad_has_aux(self):
        counters.clear()

        nums = (0, 1)

        def fn(x, y):
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn_const_var(x, y):
            return torch.func.grad(fn, argnums=(0, 1), has_aux=True)(x, y)

        def wrapper_fn_tuple_var(x, y):
            return torch.func.grad(fn, argnums=nums, has_aux=True)(x, y)

        y = torch.randn(3, 3, 3)
        x = torch.randn(3, 3, 3)
        wrapped_gm_const_var = self._compile_check(wrapper_fn_const_var, (x, y))
        wrapped_gm_tuple_var = self._compile_check(wrapper_fn_tuple_var, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual_const_var = normalize_gm(
            wrapped_gm_const_var.print_readable(print_output=False)
        )
        actual_tuple_var = normalize_gm(
            wrapped_gm_tuple_var.print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_const_var,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        child_1 = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(child);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None
        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed_2 = None

        _set_tensor_requires_grad_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(child_1);  _set_tensor_requires_grad_1 = None

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_3 = None

        sin = child.sin()
        add = sin + child_1;  sin = None
        output = add.sum();  add = None
        aux = child.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [child, child_1], create_graph = True);  child = child_1 = None
        child_2 = _autograd_grad[0]
        child_3 = _autograd_grad[1];  _autograd_grad = None

        _unwrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(child_2, 1);  child_2 = None
        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(child_3, 1);  child_3 = None

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, aux_1)
""",
        )
        self.assertExpectedInline(
            actual_tuple_var,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        child = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        child_1 = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(child);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None
        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed_2 = None

        _set_tensor_requires_grad_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(child_1);  _set_tensor_requires_grad_1 = None

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_3 = None

        sin = child.sin()
        add = sin + child_1;  sin = None
        output = add.sum();  add = None
        aux = child.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [child, child_1], create_graph = True);  child = child_1 = None
        child_2 = _autograd_grad[0]
        child_3 = _autograd_grad[1];  _autograd_grad = None

        _unwrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(child_2, 1);  child_2 = None
        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(child_3, 1);  child_3 = None

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        aux_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, aux_1)
""",
        )

    def test_grad_over_grad(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x):
            return torch.func.grad(torch.func.grad(fn))(x)

        x = torch.randn(())
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False)

        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None
        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable_1 = None
        _grad_increment_nesting_1 = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting_1 = None

        diff_args_1 = torch._C._functorch._wrap_for_grad(diff_args, 2)

        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed_2 = None

        _set_tensor_requires_grad_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args_1);  _set_tensor_requires_grad_1 = None

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_3 = None

        sin = diff_args_1.sin()
        output = sin.sum();  sin = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args_1], create_graph = True);  diff_args_1 = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad_input_1 = torch._C._functorch._unwrap_for_grad(grad_input, 2);  grad_input = None

        output_1 = torch._C._functorch._unwrap_for_grad(output, 2);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable_2 = None

        _autograd_grad_1 = torch._functorch.eager_transforms._autograd_grad((grad_input_1,), [diff_args], create_graph = True);  diff_args = None
        grad_input_2 = _autograd_grad_1[0];  _autograd_grad_1 = None

        grad_input_3: "f32[]" = torch._C._functorch._unwrap_for_grad(grad_input_2, 1);  grad_input_2 = None

        output_2: "f32[]" = torch._C._functorch._unwrap_for_grad(grad_input_1, 1);  grad_input_1 = output_2 = None

        _grad_decrement_nesting_1 = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting_1 = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_3,)
""",
        )

    def test_grad_with_graph_break(self):
        counters.clear()

        def fn(x):
            torch._dynamo.graph_break()
            return x.sin().sum()

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(actual, expected)

    def test_grad_with_side_effect(self):
        counters.clear()

        foo = [1, 2]

        def fn(x):
            foo.append(3)
            return x.sin().sum()

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_grad_pytree(self):
        counters.clear()

        def fn(x):
            x1, x2 = x
            return x1.sin().sum() + x2

        def wrapper_fn(x):
            return torch.func.grad(fn)(x)

        x1 = torch.randn(3, 3, 3)
        x2 = torch.randn(())
        actual = wrapper_fn((x1, x2))
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
            (x1, x2)
        )
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_grad_non_tensor_input(self):
        counters.clear()

        def fn(x, y):
            return x.sin().sum() + y

        def wrapper_fn(x, y):
            return torch.func.grad(fn)(x, y)

        x = torch.randn(3, 3, 3)
        y = 3.0
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        diff_args = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin = diff_args.sin()
        sum_1 = sin.sum();  sin = None
        output = sum_1 + 3.0;  sum_1 = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad_input_1: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        output_1: "f32[]" = torch._C._functorch._unwrap_for_grad(output, 1);  output = output_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (grad_input_1,)
""",
        )

    def test_grad_disable_capture(self):
        counters.clear()

        with config.patch(capture_func_transforms=False):
            # We have verified above that this
            # function compiles
            def fn(x):
                return x.sin().sum()

            def wrapper_fn(x):
                return torch.func.grad(fn)(x)

            x = torch.randn(3, 3)
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            self.assertEqual(len(counters["graph_break"]), 1)
            self.assertEqual(
                dict(counters["graph_break"]),
                {
                    "torch.func.grad capture is disabled, it can be turned "
                    "on by setting `torch._dynamo.config.capture_func_transforms=True`": 2
                },
            )
            self.assertEqual(actual, expected)

    def test_grad_fn_with_kwargs(self):
        def fn(x, y):
            return (x + y).sum()

        def wrapper_fn(x, y):
            return torch.func.grad(fn)(x, y=y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_jacfwd(self):
        counters.clear()

        def wrapper_fn(x):
            return torch.func.jacfwd(torch.sin)(x)

        x = torch.randn(4, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]"):
        l_x_ = L_x_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_x_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 4, 3]" = chunk.view(12, 4, 3);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        child_1 = torch._C._functorch._add_batch_dim(child, 0, 1);  child = None

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (child_1,));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual = torch._make_dual(l_x_, child_1, level = 0);  child_1 = None

        _wrap_for_grad = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None

        result_duals = torch.sin(_make_dual);  _make_dual = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = primals_out_unflatten = None

        tangents_out_unflatten = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        results: "f32[12, 4, 3]" = torch._C._functorch._remove_batch_dim(tangents_out_unflatten, 1, 12, 0);  tangents_out_unflatten = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        movedim: "f32[4, 3, 12]" = results.movedim(0, -1);  results = None
        split = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[4, 3, 12]" = split[0];  split = None

        unflatten: "f32[4, 3, 4, 3]" = jac_out_in.unflatten(-1, (4, 3));  jac_out_in = None
        return (unflatten,)
""",
        )

    def test_jacfwd_two_tensors_argnums(self):
        counters.clear()

        def fn(x, y):
            return y.sin()

        def wrapper_fn(x, y):
            return torch.func.jacfwd(fn, argnums=1)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_y_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 3, 4]" = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        child_1 = torch._C._functorch._add_batch_dim(child, 0, 1);  child = None

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_y_,), (child_1,));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual = torch._make_dual(l_y_, child_1, level = 0);  child_1 = None

        _wrap_for_grad = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None
        _wrap_for_grad_1 = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = _wrap_for_grad_1 = None

        result_duals = _make_dual.sin();  _make_dual = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = primals_out_unflatten = None

        tangents_out_unflatten = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        results: "f32[12, 3, 4]" = torch._C._functorch._remove_batch_dim(tangents_out_unflatten, 1, 12, 0);  tangents_out_unflatten = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        movedim: "f32[3, 4, 12]" = results.movedim(0, -1);  results = None
        split = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[3, 4, 12]" = split[0];  split = None

        unflatten: "f32[3, 4, 3, 4]" = jac_out_in.unflatten(-1, (3, 4));  jac_out_in = None
        return (unflatten,)
""",
        )

    def test_jacfwd_has_aux(self):
        counters.clear()

        def fn(x, y):
            return y.sin(), x

        def wrapper_fn(x, y):
            return torch.func.jacfwd(fn, argnums=1, has_aux=True)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_y_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 3, 4]" = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        child_1 = torch._C._functorch._add_batch_dim(child, 0, 1);  child = None

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_y_,), (child_1,));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual = torch._make_dual(l_y_, child_1, level = 0);  child_1 = None

        aux = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = None
        _wrap_for_grad_1 = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = _wrap_for_grad_1 = None

        result_duals = _make_dual.sin();  _make_dual = None

        aux_1: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(aux, 2);  aux = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = primals_out_unflatten = None

        tangents_out_unflatten = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        results: "f32[12, 3, 4]" = torch._C._functorch._remove_batch_dim(tangents_out_unflatten, 1, 12, 0);  tangents_out_unflatten = None
        aux_2: "f32[12, 4, 3]" = torch._C._functorch._remove_batch_dim(aux_1, 1, 12, 0);  aux_1 = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        aux_3: "f32[4, 3]" = aux_2[0];  aux_2 = None

        movedim: "f32[3, 4, 12]" = results.movedim(0, -1);  results = None
        split = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[3, 4, 12]" = split[0];  split = None

        unflatten: "f32[3, 4, 3, 4]" = jac_out_in.unflatten(-1, (3, 4));  jac_out_in = None
        return (unflatten, aux_3)
""",
        )

    def test_jacfwd_randomness(self):
        counters.clear()

        def fn(x, y):
            return y.sin(), x

        def wrapper_fn(x, y):
            return torch.func.jacfwd(fn, randomness="same")(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        tensor: "i64[1]" = torch.tensor((12,))
        cumsum: "i64[1]" = tensor.cumsum(dim = 0);  tensor = None
        getitem: "i64[0]" = cumsum[slice(None, -1, None)];  cumsum = None
        neg: "i64[0]" = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = unbind = None

        chunk: "f32[12, 12]" = l_x_.new_zeros(12, 12)

        diagonal: "f32[12]" = chunk.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        child: "f32[12, 4, 3]" = chunk.view(12, 4, 3);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'same');  _vmap_increment_nesting = None

        child_1 = torch._C._functorch._add_batch_dim(child, 0, 1);  child = None

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (child_1,));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        child_3 = torch._make_dual(l_x_, child_1, level = 0);  child_1 = None

        _wrap_for_grad = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None
        _wrap_for_grad_1 = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = None

        child_2 = _wrap_for_grad_1.sin();  _wrap_for_grad_1 = None

        _unpack_dual = torch._unpack_dual(child_2, level = 0);  child_2 = None
        primal = _unpack_dual[0];  _unpack_dual = None

        tangent = torch.zeros_like(primal)

        _unpack_dual_1 = torch._unpack_dual(child_3, level = 0);  child_3 = None
        primal_1 = _unpack_dual_1[0]
        dual = _unpack_dual_1[1];  _unpack_dual_1 = None

        child_4: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = child_4 = None
        child_5: "f32[4, 3]" = torch._C._functorch._unwrap_for_grad(primal_1, 2);  primal_1 = child_5 = None

        child_6: "f32[3, 4]" = torch._C._functorch._unwrap_for_grad(tangent, 2);  tangent = None
        child_7 = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        child_8: "f32[12, 3, 4]" = torch._C._functorch._remove_batch_dim(child_6, 1, 12, 0);  child_6 = None
        child_9: "f32[12, 4, 3]" = torch._C._functorch._remove_batch_dim(child_7, 1, 12, 0);  child_7 = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        movedim: "f32[3, 4, 12]" = child_8.movedim(0, -1);  child_8 = None
        split = movedim.split((12,), dim = -1);  movedim = None
        jac_out_in: "f32[3, 4, 12]" = split[0];  split = None

        unflatten: "f32[3, 4, 4, 3]" = jac_out_in.unflatten(-1, (4, 3));  jac_out_in = None

        movedim_1: "f32[4, 3, 12]" = child_9.movedim(0, -1);  child_9 = None
        split_1 = movedim_1.split((12,), dim = -1);  movedim_1 = None
        jac_out_in_1: "f32[4, 3, 12]" = split_1[0];  split_1 = None

        unflatten_1: "f32[4, 3, 4, 3]" = jac_out_in_1.unflatten(-1, (4, 3));  jac_out_in_1 = None
        return (unflatten, unflatten_1)
""",
        )

    def test_jacfwd_disable_capture(self):
        counters.clear()

        with config.patch(capture_func_transforms=False):
            # We have verified above that this
            # function compiles
            def wrapper_fn(x):
                return torch.func.jacfwd(torch.sin)(x)

            x = torch.randn(3, 3, 3)
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            self.assertEqual(len(counters["graph_break"]), 2)
            self.assertEqual(
                dict(counters["graph_break"]),
                {
                    "torch.func.vmap capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 2,
                    "torch.func.jacfwd capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 1,
                },
            )
            self.assertEqual(actual, expected)

    def test_jvp_simple(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x, v):
            return torch.func.jvp(fn, (x,), (v,))

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (l_v_,));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual = torch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin = _make_dual.sin();  _make_dual = None
        result_duals = sin.sum();  sin = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None

        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        return (primals_out_unflatten, tangents_out_unflatten)
""",
        )

    def test_jvp_has_aux(self):
        counters.clear()

        def fn(x):
            return x.sin().sum(), x

        def wrapper_fn(x, v):
            return torch.func.jvp(fn, (x,), (v,), has_aux=True)

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (l_v_,));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        aux = torch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin = aux.sin()
        result_duals = sin.sum();  sin = None

        aux_1: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None

        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        return (primals_out_unflatten, tangents_out_unflatten, aux_1)
""",
        )

    def test_jvp_two_tensors_has_aux(self):
        counters.clear()

        def fn(x, y):
            return (x.sin().sum() + y.cos()), x

        def wrapper_fn(x, y, v):
            return torch.func.jvp(fn, (x, y), (v, v), has_aux=True)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_
        l_v_ = L_v_

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_, l_y_), (l_v_, l_v_));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        aux = torch._make_dual(l_x_, l_v_, level = 0);  l_x_ = None

        _maybe_load_decompositions_1 = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions_1 = None

        _make_dual_1 = torch._make_dual(l_y_, l_v_, level = 0);  l_y_ = l_v_ = None

        sin = aux.sin()
        sum_1 = sin.sum();  sin = None
        cos = _make_dual_1.cos();  _make_dual_1 = None
        result_duals = sum_1 + cos;  sum_1 = cos = None

        aux_1: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None

        tangents_out_unflatten: "f32[3, 3]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        return (primals_out_unflatten, tangents_out_unflatten, aux_1)
""",
        )

    def test_jvp_two_tensors_disable_grad(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x, v):
            with torch.autograd.forward_ad._set_fwd_grad_enabled(False):
                return torch.func.jvp(fn, (x,), (v,))

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled = None

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (l_v_,));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual = torch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin = _make_dual.sin();  _make_dual = None
        result_duals = sin.sum();  sin = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None

        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_2 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_2 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        _set_fwd_grad_enabled_3 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_3 = None
        return (primals_out_unflatten, tangents_out_unflatten)
""",
        )

    def test_jvp_two_tensors_disable_enable_disable_grad(self):
        counters.clear()

        def fn(x):
            return x.sin().sum()

        def wrapper_fn(x, v):
            with torch.autograd.forward_ad._set_fwd_grad_enabled(False):  # (1)
                with torch.autograd.forward_ad._set_fwd_grad_enabled(True):  # (2)
                    with torch.autograd.forward_ad._set_fwd_grad_enabled(False):  # (3)
                        return torch.func.jvp(fn, (x,), (v,))  # (4)

            # Start True
            # False      (1)
            #   True     (2)
            #     False  (3)
            #       True (4)
            #     True   (undo 3)
            #   False    (undo 2)
            # True       (undo 1)

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_v_: "f32[3, 3]"):
        l_x_ = L_x_
        l_v_ = L_v_

        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _set_fwd_grad_enabled_2 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_2 = None

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (l_v_,));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled_3 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_3 = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual = torch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin = _make_dual.sin();  _make_dual = None
        result_duals = sin.sum();  sin = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(primal, 1);  primal = None

        tangents_out_unflatten: "f32[]" = torch._C._functorch._unwrap_for_grad(dual, 1);  dual = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_4 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_4 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        _set_fwd_grad_enabled_5 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_5 = None
        _set_fwd_grad_enabled_6 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_6 = None
        _set_fwd_grad_enabled_7 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_7 = None
        return (primals_out_unflatten, tangents_out_unflatten)
""",
        )

    def test_jvp_freevar_tensor(self):
        counters.clear()
        y = torch.randn(3, 3)

        def fn(x):
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            return torch.func.jvp(fn, (x,), (x,))

        x = torch.randn(3, 3)
        expected = wrapper_fn(x)
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(actual, expected)

    def test_jvp_jvp(self):
        counters.clear()

        if check_dynamic_shape_capture():
            self.skipTest("test fails with dynamic shapes")

        def fn(x):
            return torch.func.jvp(torch.sin, (x,), (x,))

        def wrapper_fn(x):
            return torch.func.jvp(fn, (x,), (x,))

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _jvp_treespec_compare = torch._functorch.eager_transforms._jvp_treespec_compare((l_x_,), (l_x_,));  _jvp_treespec_compare = None

        _jvp_increment_nesting = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._C._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        child = torch._make_dual(l_x_, l_x_, level = 0);  l_x_ = None

        _jvp_treespec_compare_1 = torch._functorch.eager_transforms._jvp_treespec_compare((child,), (child,));  _jvp_treespec_compare_1 = None

        _jvp_increment_nesting_1 = torch._C._functorch._jvp_increment_nesting();  _jvp_increment_nesting_1 = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None

        _maybe_load_decompositions_1 = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions_1 = None

        _make_dual_1 = torch._make_dual(child, child, level = 0);  child = None

        result_duals = torch.sin(_make_dual_1);  _make_dual_1 = None

        _unpack_dual = torch._unpack_dual(result_duals, level = 0);  result_duals = None
        primal = _unpack_dual[0]
        dual = _unpack_dual[1];  _unpack_dual = None

        primals_out_unflatten = torch._C._functorch._unwrap_for_grad(primal, 2);  primal = None

        tangents_out_unflatten = torch._C._functorch._unwrap_for_grad(dual, 2);  dual = None

        _set_fwd_grad_enabled_2 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_2 = None
        _jvp_decrement_nesting = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        _unpack_dual_1 = torch._unpack_dual(primals_out_unflatten, level = 0);  primals_out_unflatten = None
        primal_1 = _unpack_dual_1[0]
        dual_1 = _unpack_dual_1[1];  _unpack_dual_1 = None
        _unpack_dual_2 = torch._unpack_dual(tangents_out_unflatten, level = 0);  tangents_out_unflatten = None
        primal_2 = _unpack_dual_2[0]
        dual_2 = _unpack_dual_2[1];  _unpack_dual_2 = None

        _unwrap_for_grad_2: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(primal_1, 1);  primal_1 = None
        _unwrap_for_grad_3: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(primal_2, 1);  primal_2 = None

        _unwrap_for_grad_4: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(dual_1, 1);  dual_1 = None
        _unwrap_for_grad_5: "f32[3, 3, 3]" = torch._C._functorch._unwrap_for_grad(dual_2, 1);  dual_2 = None

        _exit_dual_level = torch._C._exit_dual_level(0);  _exit_dual_level = None
        _set_fwd_grad_enabled_3 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_3 = None
        _jvp_decrement_nesting_1 = torch._C._functorch._jvp_decrement_nesting();  _jvp_decrement_nesting_1 = None
        return (_unwrap_for_grad_2, _unwrap_for_grad_3, _unwrap_for_grad_4, _unwrap_for_grad_5)
""",
        )

    def test_jvp_freevar_python_scalar(self):
        counters.clear()
        y = 3

        def fn(x):
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            return torch.func.jvp(fn, (x,), (x,))

        x = torch.randn(3, 3, 3)
        expected = wrapper_fn(x)
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(actual, expected)

    def test_jvp_disable_capture(self):
        counters.clear()

        with config.patch(capture_func_transforms=False):
            # We have verified above that this
            # function compiles
            def wrapper_fn(x):
                return torch.func.jvp(torch.sin, (x,), (x,))

            x = torch.randn(3, 3, 3)
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            self.assertEqual(len(counters["graph_break"]), 1)
            self.assertEqual(
                dict(counters["graph_break"]),
                {
                    "torch.func.jvp capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 1
                },
            )
        self.assertEqual(actual, expected)

    @config.patch(capture_func_transforms=True)
    def test_linearize_jvp_fn(self):
        counters.clear()

        def wrapper_fn(x):
            output, jvp_fn = torch.func.linearize(torch.sin, x)
            return output, jvp_fn(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False, graph_idx=0)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_buffers_tensor_constant0_: "f32[3, 3, 3]"):
        l_self_buffers_tensor_constant0_ = L_self_buffers_tensor_constant0_

        alias_default: "f32[3, 3, 3]" = torch.ops.aten.alias.default(l_self_buffers_tensor_constant0_);  l_self_buffers_tensor_constant0_ = None

        sin_default: "f32[3, 3, 3]" = torch.ops.aten.sin.default(alias_default)

        alias_default_1: "f32[3, 3, 3]" = torch.ops.aten.alias.default(alias_default)

        cos_default: "f32[3, 3, 3]" = torch.ops.aten.cos.default(alias_default_1);  alias_default_1 = None

        alias_default_2: "f32[3, 3, 3]" = torch.ops.aten.alias.default(sin_default);  alias_default_2 = None
        return (alias_default, cos_default, sin_default)
""",
        )

        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False, graph_idx=1)
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_0_: "f32[3, 3, 3]", L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_1_: "f32[3, 3, 3]", L_flat_tangents_1_: "f32[3, 3, 3]"):
        l_self_modules_fx_const_folded_attrs_parameters_0_ = L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_0_
        l_self_modules_fx_const_folded_attrs_parameters_1_ = L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_1_
        l_flat_tangents_1_ = L_flat_tangents_1_

        _new_zeros_with_same_feature_meta_default: "f32[3, 3, 3]" = torch.ops.aten._new_zeros_with_same_feature_meta.default(l_flat_tangents_1_, l_self_modules_fx_const_folded_attrs_parameters_0_);  l_self_modules_fx_const_folded_attrs_parameters_0_ = None

        copy__default: "f32[3, 3, 3]" = torch.ops.aten.copy_.default(_new_zeros_with_same_feature_meta_default, l_flat_tangents_1_);  _new_zeros_with_same_feature_meta_default = l_flat_tangents_1_ = None

        mul_tensor: "f32[3, 3, 3]" = torch.ops.aten.mul.Tensor(copy__default, l_self_modules_fx_const_folded_attrs_parameters_1_);  copy__default = l_self_modules_fx_const_folded_attrs_parameters_1_ = None
        return (mul_tensor,)
""",
        )

    def test_linearize_disable_capture(self):
        counters.clear()
        with config.patch(capture_func_transforms=False):
            # We have verified above that this
            # function compiles
            def wrapper_fn(x):
                out, _ = torch.func.linearize(torch.sin, x)
                return out

            x = torch.randn(2, 3)
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            self.assertEqual(len(counters["graph_break"]), 1)
            self.assertEqual(
                {
                    "torch.func.linearize capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 1,
                },
                dict(counters["graph_break"]),
            )
            self.assertEqual(actual, expected)

    @config.patch(capture_func_transforms=True)
    @config.patch(error_on_recompile=True)
    def test_vmap_recompile(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        y = torch.vmap(fn)(x)
        # should not recompile on second call. See Pytorch issue #118493
        y = torch.vmap(fn)(x)

    @xfailIfTorchDynamo
    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_different_config(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        y = torch.vmap(fn)(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            fn(x)

    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_same_config(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(torch.vmap(fn, randomness="same"), randomness="same")(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            torch.vmap(torch.vmap(fn, randomness="same"), randomness="error")(x)

    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_with_randomness(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(fn, randomness="same")(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            torch.vmap(fn, randomness="different")(x)

    @config.patch(error_on_recompile=True)
    def test_grad_recompile(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.func.grad(torch.sin)(x)

        x = torch.randn([])
        torch.func.grad(fn)(x)
        # should not recompile on second call
        torch.func.grad(fn)(x)

    def test_vmap_get_wrapped(self):
        counters.clear()

        def g(x):
            return x.sin()

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn():
            return torch.vmap(g)

        x = torch.randn(3, 4)
        expected = torch.vmap(g)(x)
        wrapper = fn()
        got = wrapper(x)
        self.assertEqual(expected, got)

    def test_vmap_with_conditional_graph_break(self):
        def g(x):
            if len(x.shape) < 2:
                torch._dynamo.graph_break()
                return x.sin()
            else:
                return x.cos()

        @torch.compile(backend="aot_eager")
        def fn(x):
            return torch.vmap(g)(x)

        counters.clear()
        x = torch.randn(2, 3)
        expected = x.sin()
        got = fn(x)
        self.assertEqual(expected, got)
        self.assertEqual(len(counters["graph_break"]), 1)

        counters.clear()
        y = torch.randn(2, 3, 4)
        expected = y.cos()
        got = fn(y)
        self.assertEqual(expected, got)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_vmap_with_graph_break(self):
        counters.clear()

        def g(x):
            y = x.cos()
            print("hi")
            return y.sin()

        def fn(x):
            return torch.vmap(g)(x)

        x = torch.randn(3, 4)
        opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
        expected = fn(x)
        got = opt(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(expected, got)

    def test_vmap_with_graph_break_2(self):
        counters.clear()

        def cos(x):
            print("cos")
            return x.cos()

        def sin(x):
            print("sin")
            return x.sin()

        def g(x):
            y = cos(x)
            return sin(y)

        def fn(x):
            return torch.vmap(g, randomness="same")(x)

        x = torch.randn(3, 4)
        opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
        expected = fn(x)
        got = opt(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(expected, got)

    def test_vmap_with_graph_break_lambda(self):
        counters.clear()

        def sin(x):
            print("sin")
            return x.sin()

        def fn(x):
            return torch.vmap(lambda x: sin(x))(x)

        x = torch.randn(3, 4)
        opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
        expected = fn(x)
        got = opt(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(expected, got)

    def test_vmap(self):
        def fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1 = _add_batch_dim.sum(0)
        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        batched_outputs = sum_1 + sum_2;  sum_1 = sum_2 = None

        _remove_batch_dim: "f32[3, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_free_const(self):
        y = 3

        def fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1) + y)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1 = _add_batch_dim.sum(0)
        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        add = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_outputs = add + 3;  add = None

        _remove_batch_dim: "f32[3, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_free_tensor(self):
        y = torch.randn(3, 3)

        def fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1) + y)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1 = _add_batch_dim.sum(0)
        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        add = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_outputs = add + l_y_;  add = l_y_ = None

        _remove_batch_dim: "f32[3, 3, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_two_inputs(self):
        def fn(x, y):
            return torch.func.vmap(
                lambda x, y: x.sum(0) + x.sum(1) + y, in_dims=(0, 1)
            )(x, y)

        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        _add_batch_dim_1 = torch._C._functorch._add_batch_dim(l_y_, 1, 1);  l_y_ = None

        sum_1 = _add_batch_dim.sum(0)
        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        add = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_outputs = add + _add_batch_dim_1;  add = _add_batch_dim_1 = None

        _remove_batch_dim: "f32[3, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_two_inputs_tuple_in_dims(self):
        in_dims = (0, 1)

        def fn(x, y):
            return torch.func.vmap(
                lambda x, y: x.sum(0) + x.sum(1) + y, in_dims=in_dims
            )(x, y)

        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        _add_batch_dim_1 = torch._C._functorch._add_batch_dim(l_y_, 1, 1);  l_y_ = None

        sum_1 = _add_batch_dim.sum(0)
        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        add = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_outputs = add + _add_batch_dim_1;  add = _add_batch_dim_1 = None

        _remove_batch_dim: "f32[3, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs, 1, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_over_vmap_two_inputs(self):
        def fn(x, y):
            return torch.func.vmap(torch.func.vmap(lambda x, y: x + y, in_dims=1))(x, y)

        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        child = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        child_1 = torch._C._functorch._add_batch_dim(l_y_, 0, 1);  l_y_ = None

        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_2 = torch._C._functorch._add_batch_dim(child, 1, 2);  child = None
        _add_batch_dim_3 = torch._C._functorch._add_batch_dim(child_1, 1, 2);  child_1 = None

        batched_outputs = _add_batch_dim_2 + _add_batch_dim_3;  _add_batch_dim_2 = _add_batch_dim_3 = None

        batched_outputs_1 = torch._C._functorch._remove_batch_dim(batched_outputs, 2, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        _remove_batch_dim_1: "f32[3, 3, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs_1, 1, 3, 0);  batched_outputs_1 = None

        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None
        return (_remove_batch_dim_1,)
""",
        )

    def test_vmap_over_vmap_captured(self):
        x = torch.ones(2, 3)
        y = torch.ones(5, 3)

        def fn(x):
            return torch.func.vmap(torch.func.vmap(lambda y: x * y))(y)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_: "f32[5, 3]", L_x_: "f32[2, 3]"):
        l_y_ = L_y_
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(5, 'error');  _vmap_increment_nesting = None

        child = torch._C._functorch._add_batch_dim(l_y_, 0, 1);  l_y_ = None

        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_1 = torch._C._functorch._add_batch_dim(child, 0, 2);  child = None

        batched_outputs = l_x_ * _add_batch_dim_1;  l_x_ = _add_batch_dim_1 = None

        batched_outputs_1 = torch._C._functorch._remove_batch_dim(batched_outputs, 2, 3, 0);  batched_outputs = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        _remove_batch_dim_1: "f32[5, 3, 2, 3]" = torch._C._functorch._remove_batch_dim(batched_outputs_1, 1, 5, 0);  batched_outputs_1 = None

        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None
        return (_remove_batch_dim_1,)
""",
        )

    def test_vmap_multiple_outputs(self):
        x = torch.ones(2, 4, 3)

        def fn(x):
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)))(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        child = _add_batch_dim.sum(0)
        child_1 = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim: "f32[2, 3]" = torch._C._functorch._remove_batch_dim(child, 1, 2, 0);  child = None
        _remove_batch_dim_1: "f32[2, 4]" = torch._C._functorch._remove_batch_dim(child_1, 1, 2, 0);  child_1 = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    def test_vmap_multiple_outputs_diff_dims(self):
        x = torch.ones(2, 4, 3)

        def fn(x):
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)), out_dims=(1, 0))(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        child = _add_batch_dim.sum(0)
        child_1 = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim: "f32[3, 2]" = torch._C._functorch._remove_batch_dim(child, 1, 2, 1);  child = None
        _remove_batch_dim_1: "f32[2, 4]" = torch._C._functorch._remove_batch_dim(child_1, 1, 2, 0);  child_1 = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    def test_vmap_multiple_outputs_out_dims_tuple(self):
        x = torch.ones(2, 4, 3)
        out_dims = (1, 0)

        def fn(x):
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)), out_dims=out_dims)(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

        _add_batch_dim = torch._C._functorch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        child = _add_batch_dim.sum(0)
        child_1 = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim: "f32[3, 2]" = torch._C._functorch._remove_batch_dim(child, 1, 2, 1);  child = None
        _remove_batch_dim_1: "f32[2, 4]" = torch._C._functorch._remove_batch_dim(child_1, 1, 2, 0);  child_1 = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    def test_vmap_kwargs(self):
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        def fn(x, y):
            return torch.func.vmap(lambda x, y: x + y)(x, y=y)

        actual = fn(x, y)
        expected = torch.compile(fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_pytree_inputs(self):
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        def vmap_fn(inps):
            x = inps["x"]
            y = inps["y"]
            return x + y

        def fn(x, y):
            return torch.func.vmap(vmap_fn)({"x": x, "y": y})

        actual = fn(x, y)
        expected = torch.compile(fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_side_effects(self):
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        some_list = []

        def f(x, y):
            some_list.append(1)
            return x + y

        def wrapper_fn(x, y):
            return torch.func.vmap(f)(x, y)

        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)
        self.assertEqual(some_list, [1, 1])

    @unittest.expectedFailure
    def test_vmap_side_effects_append_input(self):
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        some_list = []

        def f(x, y):
            some_list.append(x)
            return x + y

        def wrapper_fn(x, y):
            return torch.func.vmap(f)(x, y)

        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_previous_illegal_op_no_graph_break(self):
        counters.clear()

        # calling .stride() would previously graph break
        def bad_fn(x):
            y = x.view((4, 3))
            y.stride()
            return y

        def wrapper_fn(x):
            return torch.func.vmap(bad_fn)(x)

        x = torch.randn(2, 3, 4)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_disable_capture(self):
        counters.clear()

        with config.patch(capture_func_transforms=False):
            # We have verified above that this
            # function compiles
            def wrapper_fn(x):
                return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)

            x = torch.randn(3, 3, 3)
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
                x
            )
            self.assertEqual(len(counters["graph_break"]), 1)
            self.assertEqual(
                dict(counters["graph_break"]),
                {
                    "torch.func.vmap capture is disabled, it can be "
                    "turned on by setting `torch._dynamo.config.capture_func_transforms=True`": 2
                },
            )
            self.assertEqual(actual, expected)

    def test_vmap_multiple_invocation_in_dims(self):
        counters.clear()

        def wrapper_fn(x, in_dims):
            return torch.func.vmap(torch.sum, in_dims)(x)

        x = torch.randn(3, 3, 3, 3)
        cnt = CompileCounter()
        opt = torch.compile(wrapper_fn, backend=cnt, fullgraph=False, dynamic=True)
        expected = wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2)
        # Third invocation of `opt` makes `in_dims` as SymInt.
        actual = opt(x, 0), opt(x, 1), opt(x, 2)
        self.assertEqual(expected, actual)
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 21)

    def test_vmap_multiple_invocation_out_dims(self):
        counters.clear()

        def wrapper_fn(x, out_dims):
            return torch.func.vmap(lambda x: torch.sum(x, 0), out_dims=out_dims)(x)

        x = torch.randn(3, 3, 3, 3)
        cnt = CompileCounter()
        opt = torch.compile(wrapper_fn, backend=cnt, fullgraph=False, dynamic=True)
        expected = wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2)
        # Third invocation of `opt` makes `in_dims` as SymInt.
        actual = opt(x, 0), opt(x, 1), opt(x, 2)
        self.assertEqual(expected, actual)
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 21)

    def test_vmap_new_tensor_in_body(self):
        def fn(x):
            return x + torch.ones(3)

        def wrapper_fn(x):
            return torch.func.vmap(fn)(x)

        x = torch.randn(
            3,
        )
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)

    def test_vmap_new_tensor_unused_in_body(self):
        def fn(x):
            return torch.tensor(0.5)

        def wrapper_fn(x):
            return torch.func.vmap(fn)(x)

        x = torch.randn(3)
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)

    def test_vmap_new_tensor_implicit_via_op(self):
        def wrapper_fn(x):
            return torch.func.vmap(lambda t: torch.add(t, 0.5))(x)

        x = torch.randn(3)
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)


class ActivationCheckpointingTests(torch._dynamo.test_case.TestCase):
    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))

        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()

        opt_fn = torch.compile(fn, fullgraph=fullgraph, backend=backend)
        torch.manual_seed(0)
        result = opt_fn(*cloned_args)
        result.sum().backward()

        if not skip_check:
            self.assertEqual(result, expected)
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(arg.grad, cloned_arg.grad)

    @requires_cuda
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function_with_kwargs(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                torch.sin(x),
                y,
                use_reentrant=True,
                preserve_rng_state=False,
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout(self):
        def gn(x, y):
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.rngprims.philox_rand.default
        )
        # philox_rand is passed from fwd
        bw_compiler = functools.partial(
            count_ops, freq=0, op=torch.ops.rngprims.philox_rand.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(
            fn, backend, x, y, skip_check=True
        )  # dropout decomp is known to diverge with eager

    @requires_cuda
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout_inductor(self):
        def gn(x, y):
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        backend = "inductor"
        self._validate(
            fn, backend, x, y, skip_check=True
        )  # dropout decomp is known to diverge with eager

    @requires_cuda
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_fallback(self):
        def gn(x, y):
            torch._dynamo.graph_break()
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.cos(
                torch.utils.checkpoint.checkpoint(
                    gn, torch.sin(x), y, use_reentrant=True
                ),
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)

        self.assertEqual(result, expected)

        # One graph for torch.sin on the input, and other for torch.cos.
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(backend.graphs), 2)

    @requires_cuda
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        mod = MockModule()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )

        x = torch.randn(10, 10, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        # sigmoid passed from fwd
        bw_compiler = functools.partial(
            count_ops, freq=0, op=torch.ops.aten.sigmoid.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x)

    def test_override_fallthrough_dispatch_key(self):
        test_op = torch._ops.HigherOrderOperator("_fallthrough_test_only")
        default_keys = torch._ops._HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS
        self.assertTrue(
            not any(test_op.non_fallthrough_keys.has(key) for key in default_keys)
        )

        foos = [lambda x=i: x for i, k in enumerate(default_keys)]
        for foo, fallthrough_key in zip(foos, default_keys):
            test_op.py_impl(fallthrough_key)(foo)

        self.assertTrue(
            all(test_op.non_fallthrough_keys.has(key) for key in default_keys)
        )
        self.assertEqual(
            list(range(len(default_keys))),
            [test_op.py_kernels[key]() for key in default_keys],
        )

    def test_cond_with_kwargs(self):
        from torch._higher_order_ops.cond import cond_op

        def test(pred, x):
            def true_fn(x):
                return x

            def false_fn(x):
                return -x

            return cond_op(pred=pred, true_fn=true_fn, false_fn=false_fn, operands=[x])

        cnt = CompileCounter()
        opt_test = torch.compile(test, backend=cnt, fullgraph=True)
        inp = torch.ones(3, 3)
        true_pred = torch.Tensor([True])
        false_pred = torch.Tensor([False])
        self.assertTrue(torch.allclose(test(true_pred, inp), opt_test(true_pred, inp)))
        self.assertEqual(cnt.frame_count, 1)
        self.assertTrue(
            torch.allclose(test(false_pred, inp), opt_test(false_pred, inp))
        )
        self.assertEqual(cnt.frame_count, 1)

    def test_cond_with_invalid_kwargs(self):
        from torch._higher_order_ops.cond import cond_op

        def test(pred, mode, x):
            def true_fn(x):
                return x

            def false_fn(x):
                return -x

            if mode:
                return cond_op(
                    pred=pred,
                    true_fn=true_fn,
                    false_fn=false_fn,
                    operands=[x],
                    invalid=True,
                )
            else:
                return cond_op(
                    pred,
                    pred=pred,
                    true_fn=true_fn,
                    false_fn=false_fn,
                    operands=[x],
                )

        cnt = CompileCounter()
        opt_test = torch.compile(test, backend=cnt)
        inp = torch.ones(3, 3)
        with self.assertRaises(torch._dynamo.exc.UncapturedHigherOrderOpError):
            opt_test(True, True, inp)

        with self.assertRaises(AssertionError):
            opt_test(True, False, inp)

    def test_non_aliasing_util(self):
        from torch._dynamo.variables.higher_order_ops import _assert_tensors_nonaliasing

        a = [torch.tensor(1), {"a": torch.tensor(1)}]
        b = (torch.tensor(1),)
        _assert_tensors_nonaliasing(a, b)

        with self.assertRaisesRegex(
            AssertionError, "inputs to function body cannot alias outputs"
        ):
            _assert_tensors_nonaliasing(a, a)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
