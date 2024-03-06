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
    if not config.assume_static_by_default:
        return True
    return False


def count_ops(gm, args, freq, op):
    assert [node.target for node in gm.graph.nodes].count(op) == freq
    return gm


class Obj:
    pass


class MyModule(nn.Module):
    def __init__(self):
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
            expected_opcount=ifdynstaticdefault(2, 3),
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
    def forward(self, L_d_x_ : torch.Tensor, L_d_y_0_ : torch.Tensor, L_d_y_1_2_ : torch.Tensor):
        l_d_x_ = L_d_x_
        l_d_y_0_ = L_d_y_0_
        l_d_y_1_2_ = L_d_y_1_2_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_d_x_, l_d_y_0_, l_d_y_1_2_);  wrap_body_0 = l_d_x_ = l_d_y_0_ = l_d_y_1_2_ = None
        getitem = wrap[0];  wrap = None
        return (getitem,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_d_x_, l_d_y_0_, l_d_y_1_2_):
            sin = l_d_x_.sin();  l_d_x_ = None
            cos = l_d_y_0_.cos();  l_d_y_0_ = None
            add = sin + cos;  sin = cos = None
            sin_1 = l_d_y_1_2_.sin();  l_d_y_1_2_ = None
            sub = add - sin_1;  add = sin_1 = None
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
            expected_opcount=ifdynstaticdefault(2, 3),
            return_graph=True,
        )
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                actual_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem = wrap[0];  wrap = None
        return (getitem,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            view = l_x_.view(3);  l_x_ = None
            add = view + 0.5;  view = None
            return (add,)
""",
            )
        else:
            self.assertExpectedInline(
                actual_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s0 : torch.SymInt, L_x_ : torch.Tensor):
        l_x_ = L_x_

        size = l_x_.size(0)

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_, size);  wrap_body_0 = l_x_ = size = None
        getitem = wrap[0];  wrap = None
        return (getitem,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_, size):
            view = l_x_.view(size);  l_x_ = size = None
            add = view + 0.5;  view = None
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
def forward(self, getitem, l_y_):
    getitem_1 = getitem[0]
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [getitem], [l_y_]);  map_body_0 = getitem = l_y_ = None
    getitem_2 = map_impl[0];  map_impl = None
    return (getitem_2,)""",
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
def forward(self, getitem):
    sin = getitem.sin()
    sin_1 = getitem.sin();  getitem = None
    return (sin, sin_1)""",
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
def forward(self, getitem):
    return (getitem, getitem, getitem, getitem, getitem, getitem, getitem)""",
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
def forward(self, getitem, const):
    add = getitem + 3;  getitem = None
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
def forward(self, getitem, const):
    add = getitem + 3;  getitem = None
    sin = torch.sin(add);  add = None
    return (sin,)""",
            )

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
        for name, _ in cond_gm.named_modules():
            name_set.add(name)
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
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self, x):
                def true_fn(x):
                    self.buffer += 1
                    return self.buffer.sum() + x.sum()

                def false_fn(x):
                    return (x - 1).sum()

                return control_flow.cond(x.shape[0] > 4, true_fn, false_fn, [x])

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
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

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

            return control_flow.cond(x.sum() > 0, true_fn, false_fn, tuple())

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

            return control_flow.cond(x.sum() > 0, true_fn, false_fn, tuple())

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
            def __init__(self):
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
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.ones(6, 4))

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
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.ones(6, 4))

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
        for mod_name, _ in wrap_gm.named_modules():
            names.add(mod_name)
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
    def forward(self, L_arg1_0_ : torch.Tensor, L_arg2_0_ : torch.Tensor):
        l_arg1_0_ = L_arg1_0_
        l_arg2_0_ = L_arg2_0_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_arg1_0_, l_arg2_0_);  wrap_body_0 = l_arg1_0_ = l_arg2_0_ = None
        getitem = wrap[0]
        getitem_1 = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class GraphModule(torch.nn.Module):
        def forward(self, l_arg1_0_, l_arg2_0_):
            add = l_arg1_0_ + 1;  l_arg1_0_ = None

            add_1 = l_arg2_0_ + 1;  l_arg2_0_ = None
            return (add, add_1)
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
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        a = wrap[0]
        b = wrap[1];  wrap = None

        add = a + b;  a = b = None
        return (add,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            sin = l_x_.sin()
            cos = l_x_.cos();  l_x_ = None
            return (sin, cos)
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
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem = wrap[0];  wrap = None
        return (getitem,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            neg = -l_x_;  l_x_ = None
            return (neg,)
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
            def __init__(self):
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
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        forward_handles = {}
        activations = dict()

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
            def __init__(self):
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

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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
            """\
{'batched_output': ['add'], 'sum_1': ['sum_1'], 'sum_2': ['sum_2']}""",
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
    @config.patch(capture_func_transforms=True)
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
    @config.patch(capture_func_transforms=True)
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
            """\
    triggered by the following guard failure(s):
    - torch._functorch.pyfunctorch.compare_functorch_state([])""",
            munge_exc(record.getMessage()),
        )

    @config.patch(capture_func_transforms=True)
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
    @config.patch(capture_func_transforms=True)
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
            """\
    triggered by the following guard failure(s):
    - torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'same')])""",
            record.getMessage(),
        )

    @xfailIfTorchDynamo
    @config.patch(capture_func_transforms=True)
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
            """\
    triggered by the following guard failure(s):
    - torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            record.getMessage(),
        )

    @xfailIfTorchDynamo
    @config.patch(capture_func_transforms=True)
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
            """\
    triggered by the following guard failure(s):
    - torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            munge_exc(record.getMessage()),
        )

    @xfailIfTorchDynamo
    @config.patch(capture_func_transforms=True)
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
            """\
    triggered by the following guard failure(s):
    - torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'same')])""",
            munge_exc(record.getMessage()),
        )


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

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        child_3 = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        diff_primals = torch._C._functorch._wrap_for_grad(child_3, 1);  child_3 = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        primal_out = torch.sin(diff_primals)

        out_1 = torch._C._functorch._unwrap_for_grad(primal_out, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        tensor = torch.tensor((12,))
        cumsum = tensor.cumsum(dim = 0);  tensor = None
        getitem = cumsum[slice(None, -1, None)];  cumsum = None
        neg = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = None

        chunk = out_1.new_zeros(12, 12);  out_1 = None

        diagonal = chunk.diagonal(0)
        fill_ = diagonal.fill_(1);  diagonal = None

        arg_4 = chunk.view(12, 4, 3);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(arg_4, 0, 1);  arg_4 = None

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare(primal_out, _add_batch_dim)

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([primal_out], [diff_primals], [_add_batch_dim], retain_graph = True, create_graph = True);  _add_batch_dim = None
        batched_output = _autograd_grad[0];  _autograd_grad = None

        result = torch._C._functorch._remove_batch_dim(batched_output, 1, 12, 0);  batched_output = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable_1 = torch._C._autograd._saved_tensors_hooks_enable()

        split = result.split((12,), dim = 0);  result = None
        split_1 = split[0];  split = None

        output_input = split_1.view((4, 3, 4, 3));  split_1 = None
        return (output_input, diff_primals, primal_out)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        child_2 = L_x_
        child_5 = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        _wrap_for_grad = torch._C._functorch._wrap_for_grad(child_2, 1);  child_2 = None
        diff_primals = torch._C._functorch._wrap_for_grad(child_5, 1);  child_5 = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        primal_out = diff_primals.sin()

        out_1 = torch._C._functorch._unwrap_for_grad(primal_out, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        tensor = torch.tensor((12,))
        cumsum = tensor.cumsum(dim = 0);  tensor = None
        getitem = cumsum[slice(None, -1, None)];  cumsum = None
        neg = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = None

        chunk = out_1.new_zeros(12, 12);  out_1 = None

        diagonal = chunk.diagonal(0)
        fill_ = diagonal.fill_(1);  diagonal = None

        arg_5 = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(arg_5, 0, 1);  arg_5 = None

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare(primal_out, _add_batch_dim)

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([primal_out], [diff_primals], [_add_batch_dim], retain_graph = True, create_graph = True);  _add_batch_dim = None
        batched_output = _autograd_grad[0];  _autograd_grad = None

        result = torch._C._functorch._remove_batch_dim(batched_output, 1, 12, 0);  batched_output = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable_1 = torch._C._autograd._saved_tensors_hooks_enable()

        split = result.split((12,), dim = 0);  result = None
        split_1 = split[0];  split = None

        output_input = split_1.view((3, 4, 3, 4));  split_1 = None
        return (output_input, diff_primals, primal_out)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        child_2 = L_x_
        child_5 = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        aux = torch._C._functorch._wrap_for_grad(child_2, 1);  child_2 = None
        diff_primals = torch._C._functorch._wrap_for_grad(child_5, 1);  child_5 = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_primals)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        primal_out = diff_primals.sin()

        aux_2 = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        out_1 = torch._C._functorch._unwrap_for_grad(primal_out, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        tensor = torch.tensor((12,))
        cumsum = tensor.cumsum(dim = 0);  tensor = None
        getitem = cumsum[slice(None, -1, None)];  cumsum = None
        neg = getitem.neg();  getitem = None
        unbind = neg.unbind();  neg = None

        chunk = out_1.new_zeros(12, 12);  out_1 = None

        diagonal = chunk.diagonal(0)
        fill_ = diagonal.fill_(1);  diagonal = None

        arg_5 = chunk.view(12, 3, 4);  chunk = None

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(12, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(arg_5, 0, 1);  arg_5 = None

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare(primal_out, _add_batch_dim)

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([primal_out], [diff_primals], [_add_batch_dim], retain_graph = True, create_graph = True);  _add_batch_dim = None
        batched_output = _autograd_grad[0];  _autograd_grad = None

        result = torch._C._functorch._remove_batch_dim(batched_output, 1, 12, 0);  batched_output = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable_1 = torch._C._autograd._saved_tensors_hooks_enable()

        split = result.split((12,), dim = 0);  result = None
        split_1 = split[0];  split = None

        output_input = split_1.view((3, 4, 3, 4));  split_1 = None
        return (output_input, aux_2, diff_primals, primal_out)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        child = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        child_1 = torch._C._functorch._wrap_for_grad(child, 1);  child = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        child_2 = torch._functorch.eager_transforms._set_tensor_requires_grad(child_1)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = child_1.sin();  child_1 = None
        primal_out = sin.sum();  sin = None

        out = torch._C._functorch._unwrap_for_grad(primal_out, 1);  primal_out = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (out,)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor, L_v_ : torch.Tensor):
        child = L_x_
        child_8 = L_v_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        child_1 = torch._C._functorch._wrap_for_grad(child, 1);  child = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        child_4 = torch._functorch.eager_transforms._set_tensor_requires_grad(child_1)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        primal_out = child_1.sin()
        primal_out_1 = child_1.cos();  child_1 = None

        _unwrap_for_grad = torch._C._functorch._unwrap_for_grad(primal_out, 1)
        _unwrap_for_grad_1 = torch._C._functorch._unwrap_for_grad(primal_out_1, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare((primal_out, primal_out_1), (child_8, child_8))

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([primal_out, primal_out_1], [child_4], [child_8, child_8], retain_graph = True, create_graph = True);  primal_out = primal_out_1 = child_4 = child_8 = None
        getitem = _autograd_grad[0];  _autograd_grad = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, getitem)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor, L_v_ : torch.Tensor):
        child = L_x_
        child_7 = L_v_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        child_1 = torch._C._functorch._wrap_for_grad(child, 1);  child = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        child_4 = torch._functorch.eager_transforms._set_tensor_requires_grad(child_1)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        primal_out = child_1.sin()
        primal_out_1 = child_1.cos();  child_1 = None

        _unwrap_for_grad = torch._C._functorch._unwrap_for_grad(primal_out, 1)
        _unwrap_for_grad_1 = torch._C._functorch._unwrap_for_grad(primal_out_1, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()

        child_8 = child_7.sin()

        _vjp_treespec_compare = torch._functorch.eager_transforms._vjp_treespec_compare({'first': primal_out, 'second': primal_out_1}, {'first': child_7, 'second': child_8})

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([primal_out, primal_out_1], [child_4], [child_7, child_8], retain_graph = True, create_graph = True);  primal_out = primal_out_1 = child_4 = child_7 = child_8 = None
        getitem = _autograd_grad[0];  _autograd_grad = None
        return (_unwrap_for_grad, _unwrap_for_grad_1, getitem)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        child = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        aux = torch._C._functorch._wrap_for_grad(child, 1);  child = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        child_2 = torch._functorch.eager_transforms._set_tensor_requires_grad(aux)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = aux.sin()
        primal_out = sin.sum();  sin = None

        _ = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        out = torch._C._functorch._unwrap_for_grad(primal_out, 1);  primal_out = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (out,)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        child = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        diff_args = torch._C._functorch._wrap_for_grad(child, 1);  child = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = diff_args.sin()
        output = sin.sum();  sin = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        _ = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (grad,)
""",
        )

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        child = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        diff_args = torch._C._functorch._wrap_for_grad(child, 1);  child = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = diff_args.sin()
        add = sin + 3;  sin = None
        output = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        _ = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (grad,)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        child = L_x_

        y = torch.randn(3)

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        diff_args = torch._C._functorch._wrap_for_grad(child, 1);  child = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = diff_args.sin()
        add = sin + y;  sin = None
        output = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        _ = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (grad, y)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        child = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        diff_args = torch._C._functorch._wrap_for_grad(child, 1);  child = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = diff_args.sin()
        add = sin + 3.14;  sin = None
        output = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        _ = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (grad,)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        child = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        diff_args = torch._C._functorch._wrap_for_grad(child, 1);  child = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = diff_args.sin()
        add = sin + 3.14;  sin = None
        output = add.sum();  add = None
        aux = diff_args.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        _ = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        aux_2 = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (grad, aux_2)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        child = L_x_
        child_1 = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        diff_args = torch._C._functorch._wrap_for_grad(child, 1);  child = None
        _wrap_for_grad_1 = torch._C._functorch._wrap_for_grad(child_1, 1);  child_1 = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = diff_args.sin()
        add = sin + _wrap_for_grad_1;  sin = _wrap_for_grad_1 = None
        output = add.sum();  add = None
        aux = diff_args.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        _ = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        aux_2 = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (grad, aux_2)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        child = L_x_
        child_1 = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        child_4 = torch._C._functorch._wrap_for_grad(child, 1);  child = None
        child_5 = torch._C._functorch._wrap_for_grad(child_1, 1);  child_1 = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(child_4)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)
        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(child_5)

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = child_4.sin()
        add = sin + child_5;  sin = None
        output = add.sum();  add = None
        aux = child_4.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [child_4, child_5], create_graph = True);  child_4 = child_5 = None
        child_6 = _autograd_grad[0]
        child_7 = _autograd_grad[1];  _autograd_grad = None

        _unwrap_for_grad = torch._C._functorch._unwrap_for_grad(child_6, 1);  child_6 = None
        _unwrap_for_grad_1 = torch._C._functorch._unwrap_for_grad(child_7, 1);  child_7 = None

        _ = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        aux_2 = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_unwrap_for_grad, _unwrap_for_grad_1, aux_2)
""",
        )
        self.assertExpectedInline(
            actual_tuple_var,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        child = L_x_
        child_1 = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        child_4 = torch._C._functorch._wrap_for_grad(child, 1);  child = None
        child_5 = torch._C._functorch._wrap_for_grad(child_1, 1);  child_1 = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(child_4)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)
        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(child_5)

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = child_4.sin()
        add = sin + child_5;  sin = None
        output = add.sum();  add = None
        aux = child_4.cos()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [child_4, child_5], create_graph = True);  child_4 = child_5 = None
        child_6 = _autograd_grad[0]
        child_7 = _autograd_grad[1];  _autograd_grad = None

        _unwrap_for_grad = torch._C._functorch._unwrap_for_grad(child_6, 1);  child_6 = None
        _unwrap_for_grad_1 = torch._C._functorch._unwrap_for_grad(child_7, 1);  child_7 = None

        _ = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        aux_2 = torch._C._functorch._unwrap_for_grad(aux, 1);  aux = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_unwrap_for_grad, _unwrap_for_grad_1, aux_2)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        child = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        child_1 = torch._C._functorch._wrap_for_grad(child, 1);  child = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(child_1)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)
        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting_1 = torch._C._functorch._grad_increment_nesting()

        diff_args_1 = torch._C._functorch._wrap_for_grad(child_1, 2)

        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad_1 = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args_1)

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = diff_args_1.sin()
        output = sin.sum();  sin = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args_1], create_graph = True);  diff_args_1 = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        output_2 = torch._C._functorch._unwrap_for_grad(grad_input, 2);  grad_input = None

        _ = torch._C._functorch._unwrap_for_grad(output, 2);  output = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        _autograd_grad_1 = torch._functorch.eager_transforms._autograd_grad((output_2,), [child_1], create_graph = True);  child_1 = None
        grad_input_2 = _autograd_grad_1[0];  _autograd_grad_1 = None

        grad_1 = torch._C._functorch._unwrap_for_grad(grad_input_2, 1);  grad_input_2 = None

        __1 = torch._C._functorch._unwrap_for_grad(output_2, 1);  output_2 = None

        _grad_decrement_nesting_1 = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (grad_1,)
""",
        )

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        child = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting()

        diff_args = torch._C._functorch._wrap_for_grad(child, 1);  child = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True)

        _set_tensor_requires_grad = torch._functorch.eager_transforms._set_tensor_requires_grad(diff_args)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False)

        sin = diff_args.sin()
        sum_1 = sin.sum();  sin = None
        output = sum_1 + 3.0;  sum_1 = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((output,), [diff_args], create_graph = True);  diff_args = None
        grad_input = _autograd_grad[0];  _autograd_grad = None

        grad = torch._C._functorch._unwrap_for_grad(grad_input, 1);  grad_input = None

        _ = torch._C._functorch._unwrap_for_grad(output, 1);  output = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (grad,)
""",
        )

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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
    @config.patch(capture_func_transforms=True)
    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_different_config(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        y = torch.vmap(fn)(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            fn(x)

    @config.patch(capture_func_transforms=True)
    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_same_config(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(torch.vmap(fn, randomness="same"), randomness="same")(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            torch.vmap(torch.vmap(fn, randomness="same"), randomness="error")(x)

    @config.patch(capture_func_transforms=True)
    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_with_randomness(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(fn, randomness="same")(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            torch.vmap(fn, randomness="different")(x)

    @config.patch(capture_func_transforms=True)
    @config.patch(error_on_recompile=True)
    def test_grad_recompile(self):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.func.grad(torch.sin)(x)

        x = torch.randn([])
        torch.func.grad(fn)(x)
        # should not recompile on second call
        torch.func.grad(fn)(x)

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        arg = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(arg, 0, 1);  arg = None

        sum_1 = _add_batch_dim.sum(0)
        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        batched_output = sum_1 + sum_2;  sum_1 = sum_2 = None

        _remove_batch_dim = torch._C._functorch._remove_batch_dim(batched_output, 1, 3, 0);  batched_output = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_remove_batch_dim,)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        arg = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(arg, 0, 1);  arg = None

        sum_1 = _add_batch_dim.sum(0)
        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        add = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_output = add + 3;  add = None

        _remove_batch_dim = torch._C._functorch._remove_batch_dim(batched_output, 1, 3, 0);  batched_output = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_remove_batch_dim,)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        arg = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(arg, 0, 1);  arg = None

        sum_1 = _add_batch_dim.sum(0)
        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        add = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_output = add + l_y_;  add = l_y_ = None

        _remove_batch_dim = torch._C._functorch._remove_batch_dim(batched_output, 1, 3, 0);  batched_output = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_remove_batch_dim,)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        arg = L_x_
        arg_3 = L_y_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(arg, 0, 1);  arg = None
        _add_batch_dim_1 = torch._C._functorch._add_batch_dim(arg_3, 1, 1);  arg_3 = None

        sum_1 = _add_batch_dim.sum(0)
        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        add = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_output = add + _add_batch_dim_1;  add = _add_batch_dim_1 = None

        _remove_batch_dim = torch._C._functorch._remove_batch_dim(batched_output, 1, 3, 0);  batched_output = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_remove_batch_dim,)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        arg = L_x_
        arg_3 = L_y_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(arg, 0, 1);  arg = None
        _add_batch_dim_1 = torch._C._functorch._add_batch_dim(arg_3, 1, 1);  arg_3 = None

        sum_1 = _add_batch_dim.sum(0)
        sum_2 = _add_batch_dim.sum(1);  _add_batch_dim = None
        add = sum_1 + sum_2;  sum_1 = sum_2 = None
        batched_output = add + _add_batch_dim_1;  add = _add_batch_dim_1 = None

        _remove_batch_dim = torch._C._functorch._remove_batch_dim(batched_output, 1, 3, 0);  batched_output = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_remove_batch_dim,)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        arg = L_x_
        arg_3 = L_y_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(3, 'error')

        arg_8 = torch._C._functorch._add_batch_dim(arg, 0, 1);  arg = None
        arg_9 = torch._C._functorch._add_batch_dim(arg_3, 0, 1);  arg_3 = None

        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(3, 'error')

        _add_batch_dim_2 = torch._C._functorch._add_batch_dim(arg_8, 1, 2);  arg_8 = None
        _add_batch_dim_3 = torch._C._functorch._add_batch_dim(arg_9, 1, 2);  arg_9 = None

        batched_output = _add_batch_dim_2 + _add_batch_dim_3;  _add_batch_dim_2 = _add_batch_dim_3 = None

        batched_output_1 = torch._C._functorch._remove_batch_dim(batched_output, 2, 3, 0);  batched_output = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        _remove_batch_dim_1 = torch._C._functorch._remove_batch_dim(batched_output_1, 1, 3, 0);  batched_output_1 = None

        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_remove_batch_dim_1,)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_y_ : torch.Tensor, L_x_ : torch.Tensor):
        arg = L_y_
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(5, 'error')

        arg_3 = torch._C._functorch._add_batch_dim(arg, 0, 1);  arg = None

        lazy_load_decompositions_1 = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting_1 = torch._C._functorch._vmap_increment_nesting(3, 'error')

        _add_batch_dim_1 = torch._C._functorch._add_batch_dim(arg_3, 0, 2);  arg_3 = None

        batched_output = l_x_ * _add_batch_dim_1;  l_x_ = _add_batch_dim_1 = None

        batched_output_1 = torch._C._functorch._remove_batch_dim(batched_output, 2, 3, 0);  batched_output = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")

        _remove_batch_dim_1 = torch._C._functorch._remove_batch_dim(batched_output_1, 1, 5, 0);  batched_output_1 = None

        _vmap_decrement_nesting_1 = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_remove_batch_dim_1,)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        arg = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(arg, 0, 1);  arg = None

        batched_output = _add_batch_dim.sum(0)
        batched_output_1 = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim = torch._C._functorch._remove_batch_dim(batched_output, 1, 2, 0);  batched_output = None
        _remove_batch_dim_1 = torch._C._functorch._remove_batch_dim(batched_output_1, 1, 2, 0);  batched_output_1 = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        arg = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(arg, 0, 1);  arg = None

        batched_output = _add_batch_dim.sum(0)
        batched_output_1 = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim = torch._C._functorch._remove_batch_dim(batched_output, 1, 2, 1);  batched_output = None
        _remove_batch_dim_1 = torch._C._functorch._remove_batch_dim(batched_output_1, 1, 2, 0);  batched_output_1 = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    @config.patch(capture_func_transforms=True)
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
    def forward(self, L_x_ : torch.Tensor):
        arg = L_x_

        lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case.")
        _vmap_increment_nesting = torch._C._functorch._vmap_increment_nesting(2, 'error')

        _add_batch_dim = torch._C._functorch._add_batch_dim(arg, 0, 1);  arg = None

        batched_output = _add_batch_dim.sum(0)
        batched_output_1 = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim = torch._C._functorch._remove_batch_dim(batched_output, 1, 2, 1);  batched_output = None
        _remove_batch_dim_1 = torch._C._functorch._remove_batch_dim(batched_output_1, 1, 2, 0);  batched_output_1 = None

        _vmap_decrement_nesting = torch._C._functorch._vmap_decrement_nesting()
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable()
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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
    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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
        self.assertEqual(cnt.op_count, 33)

    @config.patch(capture_func_transforms=True)
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
        self.assertEqual(cnt.op_count, 30)

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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

    @config.patch(capture_func_transforms=True)
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
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
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
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
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
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.rngprims.philox_rand.default
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
            def __init__(self):
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
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
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
        opt_test = torch.compile(test, backend=cnt)
        inp = torch.ones(3, 3)
        self.assertTrue(torch.allclose(test(True, inp), opt_test(True, inp)))
        self.assertEqual(cnt.frame_count, 1)
        self.assertTrue(torch.allclose(test(False, inp), opt_test(False, inp)))
        self.assertEqual(cnt.frame_count, 2)

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
