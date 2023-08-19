# Owner(s): ["module: dynamo"]
import contextlib
import functools
import unittest

import functorch.experimental.control_flow as control_flow

import torch
import torch._dynamo.config as config

import torch._dynamo.test_case
import torch._functorch.config
import torch.nn as nn
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import (
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    normalize_gm,
)
from torch._dynamo.utils import counters
from torch._higher_order_ops.wrap import wrap
from torch.testing._internal.inductor_utils import HAS_CUDA


requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")


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

    def _test_wrap_simple(self, func, args, expected_num_wrap_args, expected_opcount=1):
        # Given a `func` that has a single call to `wrap`,
        # we check that:
        # - there are no graph breaks
        # - eager vs torch.compile has the same result
        # - after dynamo capture, the wrap has the expected number of args
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        expected = func(*args)
        result = torch.compile(func, fullgraph=True, backend=cnt)(*args)

        self.assertEqual(result, expected)

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, expected_opcount)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertEqual(len(wrap_node.args), expected_num_wrap_args)

    def test_no_freevars(self):
        def f(x):
            return wrap(lambda x: torch.sin(x), x)

        x = torch.randn(3)
        self._test_wrap_simple(f, (x,), 2)

    def test_return_captured_var(self):
        freevar = torch.randn(3)

        def test(x):
            return freevar

        def fn(x):
            return wrap(test, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.
        self._test_wrap_simple(fn, (x,), 2)

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
        self._test_wrap_simple(fn, (x,), 3, 4)

    def test_return_captured_var_used_multiple_times(self):
        freevar = torch.randn(3)

        def test(x):
            y = x + freevar
            return y, freevar

        def fn(x):
            return wrap(test, x)

        x = torch.randn(3)
        self._test_wrap_simple(fn, (x,), 3, 3)

    def test_capture_untracked_global(self):
        def f(x):
            return wrap(lambda x: x + global_var, x)

        x = torch.randn(3)
        self._test_wrap_simple(f, (x,), 3)

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
        self.assertEqual(cnt.op_count, 1)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)

        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_capture_untracked_nonlocal(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            def g(x):
                return wrap(lambda x: x + y, x)

            self._test_wrap_simple(g, (x,), 3)
            return g(x)

        f(x, y)

    def test_capture_tracked(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            return wrap(lambda x: x + y, x)

        self._test_wrap_simple(f, (x, y), 3)

    def test_capture_tracked_nested(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            return wrap(lambda x: wrap(lambda x: x + y, x), x)

        self._test_wrap_simple(f, (x, y), 3)

    def test_inlined_functions(self):
        def g(x, y):
            return x + y

        def f(x, y):
            return wrap(lambda x: g(x, y), x)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, (x, y), 3)

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
        self._test_wrap_simple(f, (x,), 2, 3)

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
        self.assertEqual(cnt.op_count, 1)
        self.assertEqual(len(backend.graphs), 1)

        # No changes to args of outer wrap
        gm = backend.graphs[0]
        wrap_node = find_first_node(gm, wrap)
        self.assertTrue(len(wrap_node.args), 3)

        # z was lifted to arg of inner wrap
        body_function = getattr(gm, wrap_node.args[0].name)
        # addition + wrap
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

        # Innermost body function: z was also lifted to arg
        body_function = getattr(body_function, inner_wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)
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
        self._test_wrap_simple(g, (x,), 2)

    def test_wrap_kwarg(self):
        def f(x, y):
            return wrap(lambda x, y: x + y, x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, (x, y), 3)

    def test_wrap_kwarg_int(self):
        def f(x, y):
            return wrap(lambda x, y: x + y, x, y=y)

        x = torch.randn(3)
        y = 8

        # When running with dynamic shapes, `y` is captured as SymNodeVariable,
        # which is not supported currently.
        err_msg = "HigherOrderOperator with body that accepts non-Tensors as input"
        err_ctx = (
            self.assertRaisesRegex(torch._dynamo.exc.Unsupported, err_msg)
            if check_dynamic_shape_capture()
            else contextlib.nullcontext()
        )

        with err_ctx:
            # int are not passed as argument and directly
            # baked into the graph.
            self._test_wrap_simple(f, (x, y), 2)

    def test_wrap_all_kwarg(self):
        def f(y, x):
            return wrap(lambda x, y: (x * 2) + y, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        self._test_wrap_simple(f, (x, y), 3)

    def test_wrap_kwarg_only(self):
        def f(x, y):
            def fn(*, x, y):
                return (x * 2) + y

            return wrap(fn, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        self._test_wrap_simple(f, (x, y), 3)

    def test_wrap_kwarg_default(self):
        def f(x, y):
            def fn(*, x, y, z=8):
                return (x * 2) + y + z

            return wrap(fn, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        self._test_wrap_simple(f, (x, y), 3)

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

        self._test_wrap_simple(f, (x, y), 3)

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
        self.assertEqual(counters["stats"]["calls_captured"], 1)

        # verify that we `don't` recompile
        opt(x, y)
        self.assertEqual(counters["stats"]["calls_captured"], 1)

        # When running with dynamic shapes, `z` is captured as SymNodeVariable,
        # which is not supported currently.
        err_msg = "HigherOrderOperator with body that accepts non-Tensors as input"
        err_ctx = (
            self.assertRaisesRegex(torch._dynamo.exc.Unsupported, err_msg)
            if check_dynamic_shape_capture()
            else contextlib.nullcontext()
        )

        with err_ctx:
            # verify that we `do` recompile
            output = opt(x, y, 8)
            self.assertEqual(counters["stats"]["calls_captured"], 2)
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

        # When running with dynamic shapes, `z` is captured as SymNodeVariable,
        # which is not supported currently.
        err_msg = "HigherOrderOperator with body that accepts non-Tensors as input"
        err_ctx = (
            self.assertRaisesRegex(torch._dynamo.exc.Unsupported, err_msg)
            if check_dynamic_shape_capture()
            else contextlib.nullcontext()
        )

        with err_ctx:
            # expected_num_wrap_args = 2 because in this case,
            # we take the `else` branch and `y` is not lifted.
            self._test_wrap_simple(f, (x, y, 8), 2)

    def test_wrap_unsupported_kwarg(self):
        def f(x, y, z):
            def fn(*, x, y, z):
                z1, z2 = z
                return (x * 2) + y + z1

            return wrap(fn, x=x, y=y, z=z)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        self._assert_wrap_fallback(f, (x, y, (x, y)))

    def test_map_subgraph_name_is_valid(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        xs = torch.randn(2, 3, 3)
        y = torch.randn(3)

        @torch.compile(backend=cnt, fullgraph=True)
        def map_f(xs, y):
            def inner(x, y):
                def inner2(x, y):
                    return x + y

                return control_flow.map(inner2, x, y)

            return control_flow.map(inner, xs, y)

        result = map_f(xs, y)
        self.assertEqual(result, xs + y)

        map_gm = backend.graphs[0]
        name_set = set()
        for name, _ in map_gm.named_modules():
            name_set.add(name)
        self.assertEqual(name_set, {"", "map_body_1.map_body_0", "map_body_1"})

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

        actual = mod_for_compile(torch.ones(6, 4))
        ref = mod_for_eager(torch.ones(6, 4))
        self.assertEqual(actual, ref)

        actual = mod_for_compile(torch.ones(3, 4))
        ref = mod_for_eager(torch.ones(3, 4))
        self.assertEqual(actual, ref)

        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, s0 : torch.SymInt, s1 : torch.SymInt, L_x_ : torch.Tensor):
    l_x_ = L_x_
    size = l_x_.size();  l_x_ = None
    getitem = size[0];  size = None
    gt = getitem > 4;  getitem = None
    return (gt,)""",
        )

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
            if node.op == "call_function" and node.target == control_flow.cond:
                _, _, _, operands = node.args
                # Each branch takes 5 inputs (x, true_buffer, true_z, false_buffer, false_z)
                self.assertEqual(len(operands), 5)
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
                    self.assertEqual(num_placeholders, 5)

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

        mod_for_compile = torch.compile(
            Foo(), backend=cnt, dynamic=True, fullgraph=False
        )
        mod_for_eager = Foo()

        res = mod_for_compile(torch.tensor(True), torch.tensor(5))
        res = mod_for_compile(torch.tensor(True), torch.tensor(5))

        self.assertEqual(len(backend.graphs), 0)
        self.assertEqual(res, mod_for_eager(torch.tensor(True), torch.tensor(5)))

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

    def test_capture_global_num(self):
        def f(x):
            return wrap(lambda x: x + global_num, x)

        x = torch.zeros([])
        # Numbers don't get lifted, so args is still 2.
        self._test_wrap_simple(f, (x,), 2)

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
        self._test_wrap_simple(f, (x, y), 2)

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
        self.assertEqual(
            dict(counters["graph_break"]),
            {
                "HigherOrderOperator: Mutating a variable not in the current scope (SideEffects)": 1
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
        self._test_wrap_simple(f, (x,), 2, expected_opcount=3)

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
        self.assertEqual(
            dict(counters["graph_break"]),
            {"HigherOrderOperator body's output must consist of tensors only": 1},
        )

    def test_fallback_on_nested_tuple_output(self):
        counters.clear()

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt)
        def f(x):
            ((a, b),) = wrap(lambda x: ((x.sin(), x.cos()),), x)
            return a + b

        x = torch.randn(2, 3)
        result = f(x)

        self.assertEqual(result, x.sin() + x.cos())
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 1)
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)

    def test_fallback_on_output_with_dict(self):
        # We can likely support this in the future, I just don't want to deal
        # with it right now
        counters.clear()
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            return wrap(lambda x: [{"a": -x}], x)

        x = torch.randn(3)
        result = f(x)
        self.assertEqual(result, [{"a": -x}])
        self.assertEqual(cnt.frame_count, 0)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"HigherOrderOperator body's output must consist of tensors only": 1},
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
        self._test_wrap_simple(h, (x, y), 3)

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
        self._test_wrap_simple(h, (x, y), 3)

    def test_capture_numpy_number(self):
        import numpy as np

        y = np.float32(1.0)

        def f(x):
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        # np.number are lifted to graph inputs
        self._test_wrap_simple(f, (x,), 3)

    def test_freevars_as_inputs_to_wrap(self):
        y = torch.randn(3)

        def f(x):
            return wrap(lambda x, y: x + y, x, y)

        x = torch.randn(3)
        self._test_wrap_simple(f, (x,), 3)

    def test_lift_tensor_constant(self):
        def f(x):
            y = torch.tensor(1.0)
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        self._test_wrap_simple(f, (x,), 3, expected_opcount=2)

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

        self._test_wrap_simple(fn, (torch.randn(10, 10),), 4, expected_opcount=1)

    def test_fn_with_kwargs_in_torch_ops(self):
        def fn(x):
            return wrap(lambda z: torch.cos(input=z), x)

        x = torch.randn(3)
        self._test_wrap_simple(fn, (x,), 2, expected_opcount=1)

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


class FuncTorchHigherOrderOpTests(torch._dynamo.test_case.TestCase):
    def run(self, result=None):
        # capture_func_transform will be set to False (for 2.1) till we
        # support all transforms, so manually patch it to `True`` for
        # testing on release branch.
        with config.patch(capture_func_transforms=True):
            super().run(result)

    def _compile_check(self, fn, inputs, fullgraph=True, graph_idx=0):
        backend = EagerAndRecordGraphs()
        actual = fn(*inputs)
        expected = torch.compile(fn, backend=backend, fullgraph=fullgraph)(*inputs)

        self.assertEqual(actual, expected)

        wrapped_gm = backend.graphs[graph_idx]
        return wrapped_gm

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

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        grad_body_0 = self.grad_body_0
        grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None
        call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None
        contiguous = call.contiguous();  call = None
        return (contiguous,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            _set_grad_enabled = torch._C._set_grad_enabled(True)

            sin = l_x_.sin();  l_x_ = None
            sum_1 = sin.sum();  sin = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
            return sum_1
"""

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

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

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        grad_body_0 = self.grad_body_0
        grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None
        call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None
        contiguous = call.contiguous();  call = None
        return (contiguous,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            _set_grad_enabled = torch._C._set_grad_enabled(True)

            sin = l_x_.sin();  l_x_ = None
            add = sin + 3;  sin = None
            sum_1 = add.sum();  add = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
            return sum_1
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

    def test_grad_capture_tensor(self):
        counters.clear()

        def wrapper_fn(x):
            y = torch.randn(3)

            def fn(x):
                return (x.sin() + y).sum()

            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)

        # Graph break because dynamo is unable to get source `fn` and
        # functools.wraps in `grad` leads to graph-break
        # There are two graphs, first for generating `y` and
        # second for application of `grad`.
        # We are interested in the second graph.
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False, graph_idx=1)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        l_x_ = L_x_
        l_y_ = L_y_

        grad_body_0 = self.grad_body_0
        grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None
        call = grad_proxy.__call__(l_x_, l_y_);  grad_proxy = l_x_ = l_y_ = None
        contiguous = call.contiguous();  call = None
        return (contiguous,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_, l_y_):
            _set_grad_enabled = torch._C._set_grad_enabled(True)

            sin = l_x_.sin();  l_x_ = None
            add = sin + l_y_;  sin = l_y_ = None
            sum_1 = add.sum();  add = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
            return sum_1
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

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        grad_body_0 = self.grad_body_0
        grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None
        call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None
        contiguous = call.contiguous();  call = None
        return (contiguous,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            _set_grad_enabled = torch._C._set_grad_enabled(True)

            sin = l_x_.sin();  l_x_ = None
            add = sin + 3.14;  sin = None
            sum_1 = add.sum();  add = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
            return sum_1
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

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

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        grad_body_0 = self.grad_body_0
        grad_proxy = torch.func.grad(grad_body_0, 0, True);  grad_body_0 = None
        call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None
        getitem = call[0]
        getitem_1 = call[1];  call = None
        contiguous = getitem.contiguous();  getitem = None
        return (contiguous, getitem_1)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            _set_grad_enabled = torch._C._set_grad_enabled(True)

            sin = l_x_.sin()
            add = sin + 3.14;  sin = None
            sum_1 = add.sum();  add = None
            cos = l_x_.cos();  l_x_ = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
            return (sum_1, cos)
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

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

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        l_x_ = L_x_
        l_y_ = L_y_

        grad_body_0 = self.grad_body_0
        grad_proxy = torch.func.grad(grad_body_0, 0, True);  grad_body_0 = None
        call = grad_proxy.__call__(l_x_, l_y_);  grad_proxy = l_x_ = l_y_ = None
        getitem = call[0]
        getitem_1 = call[1];  call = None
        contiguous = getitem.contiguous();  getitem = None
        return (contiguous, getitem_1)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_, l_y_):
            _set_grad_enabled = torch._C._set_grad_enabled(True)

            sin = l_x_.sin()
            add = sin + l_y_;  sin = l_y_ = None
            sum_1 = add.sum();  add = None
            cos = l_x_.cos();  l_x_ = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
            return (sum_1, cos)
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

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

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        l_x_ = L_x_
        l_y_ = L_y_

        grad_body_0 = self.grad_body_0
        grad_proxy = torch.func.grad(grad_body_0, (0, 1), True);  grad_body_0 = None
        call = grad_proxy.__call__(l_x_, l_y_);  grad_proxy = l_x_ = l_y_ = None
        getitem = call[0]
        getitem_1 = getitem[0]
        getitem_2 = getitem[1];  getitem = None
        getitem_3 = call[1];  call = None
        contiguous = getitem_1.contiguous();  getitem_1 = None
        contiguous_1 = getitem_2.contiguous();  getitem_2 = None
        return (contiguous, contiguous_1, getitem_3)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_, l_y_):
            _set_grad_enabled = torch._C._set_grad_enabled(True)

            sin = l_x_.sin()
            add = sin + l_y_;  sin = l_y_ = None
            sum_1 = add.sum();  add = None
            cos = l_x_.cos();  l_x_ = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
            return (sum_1, cos)
"""
        actual_const_var = normalize_gm(
            wrapped_gm_const_var.print_readable(print_output=False)
        )
        actual_tuple_var = normalize_gm(
            wrapped_gm_tuple_var.print_readable(print_output=False)
        )
        self.assertExpectedInline(actual_const_var, expected)
        self.assertExpectedInline(actual_tuple_var, expected)

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

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        grad_body_1 = self.grad_body_1
        grad_proxy = torch.func.grad(grad_body_1, 0, False);  grad_body_1 = None
        call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None
        contiguous = call.contiguous();  call = None
        return (contiguous,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_):
            _set_grad_enabled = torch._C._set_grad_enabled(True)

            grad_body_0 = self.grad_body_0
            grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None
            call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None
            contiguous = call.contiguous();  call = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
            return contiguous

        class GraphModule(torch.nn.Module):
            def forward(self, l_x_):
                _set_grad_enabled = torch._C._set_grad_enabled(True)

                sin = l_x_.sin();  l_x_ = None
                sum_1 = sin.sum();  sin = None

                _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
                return sum_1
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

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
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {
                "HigherOrderOperator: Mutating a variable not in the current scope (replace_all)": 2
            },
        )
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
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"HigherOrderOperator with body that accepts non-Tensors as input": 2},
        )
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

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        grad_body_0 = self.grad_body_0
        grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None
        call = grad_proxy.__call__(l_x_, 3.0);  grad_proxy = l_x_ = None
        contiguous = call.contiguous();  call = None
        return (contiguous,)

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_, const):
            _set_grad_enabled = torch._C._set_grad_enabled(True)

            sin = l_x_.sin();  l_x_ = None
            sum_1 = sin.sum();  sin = None
            add = sum_1 + 3.0;  sum_1 = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
            return add
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

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
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"torch.func.grad: kwargs arguments are currently unsupported.": 2},
        )
        self.assertEqual(actual, expected)

    def test_vmap(self):
        def fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')

        select = l_x_.select(0, 0)
        vmap_body_0 = self.vmap_body_0
        vmap_proxy = torch.func.vmap(vmap_body_0, (0,), 0, 'error');  vmap_body_0 = None
        call = vmap_proxy.__call__(l_x_);  vmap_proxy = l_x_ = None
        return (call,)

    class GraphModule(torch.nn.Module):
        def forward(self, select):
            sum_1 = select.sum(0)
            sum_2 = select.sum(1);  select = None
            add = sum_1 + sum_2;  sum_1 = sum_2 = None
            return add
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

    def test_vmap_free_const(self):
        y = 3

        def fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1) + y)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')

        select = l_x_.select(0, 0)
        vmap_body_0 = self.vmap_body_0
        vmap_proxy = torch.func.vmap(vmap_body_0, (0,), 0, 'error');  vmap_body_0 = None
        call = vmap_proxy.__call__(l_x_);  vmap_proxy = l_x_ = None
        return (call,)

    class GraphModule(torch.nn.Module):
        def forward(self, select):
            sum_1 = select.sum(0)
            sum_2 = select.sum(1);  select = None
            add = sum_1 + sum_2;  sum_1 = sum_2 = None
            add_1 = add + 3;  add = None
            return add_1
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

    def test_vmap_free_tensor(self):
        y = torch.randn(3, 3)

        def fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1) + y)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        l_x_ = L_x_
        l_y_ = L_y_

        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')

        select = l_x_.select(0, 0)
        vmap_body_0 = self.vmap_body_0
        vmap_proxy = torch.func.vmap(vmap_body_0, (0, None), 0, 'error');  vmap_body_0 = None
        call = vmap_proxy.__call__(l_x_, l_y_);  vmap_proxy = l_x_ = l_y_ = None
        return (call,)

    class GraphModule(torch.nn.Module):
        def forward(self, select, l_y_):
            sum_1 = select.sum(0)
            sum_2 = select.sum(1);  select = None
            add = sum_1 + sum_2;  sum_1 = sum_2 = None
            add_1 = add + l_y_;  add = l_y_ = None
            return add_1
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

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

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        l_x_ = L_x_
        l_y_ = L_y_

        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')

        select = l_x_.select(0, 0)
        select_1 = l_y_.select(1, 0)
        vmap_body_0 = self.vmap_body_0
        vmap_proxy = torch.func.vmap(vmap_body_0, (0, 1), 0, 'error');  vmap_body_0 = None
        call = vmap_proxy.__call__(l_x_, l_y_);  vmap_proxy = l_x_ = l_y_ = None
        return (call,)

    class GraphModule(torch.nn.Module):
        def forward(self, select, select_1):
            sum_1 = select.sum(0)
            sum_2 = select.sum(1);  select = None
            add = sum_1 + sum_2;  sum_1 = sum_2 = None
            add_1 = add + select_1;  add = select_1 = None
            return add_1
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

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

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        l_x_ = L_x_
        l_y_ = L_y_

        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')

        select = l_x_.select(0, 0)
        select_1 = l_y_.select(1, 0)
        vmap_body_0 = self.vmap_body_0
        vmap_proxy = torch.func.vmap(vmap_body_0, (0, 1), 0, 'error');  vmap_body_0 = None
        call = vmap_proxy.__call__(l_x_, l_y_);  vmap_proxy = l_x_ = l_y_ = None
        return (call,)

    class GraphModule(torch.nn.Module):
        def forward(self, select, select_1):
            sum_1 = select.sum(0)
            sum_2 = select.sum(1);  select = None
            add = sum_1 + sum_2;  sum_1 = sum_2 = None
            add_1 = add + select_1;  add = select_1 = None
            return add_1
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

    def test_vmap_over_vmap_two_inputs(self):
        def fn(x, y):
            return torch.func.vmap(torch.func.vmap(lambda x, y: x + y, in_dims=1))(x, y)

        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        l_x_ = L_x_
        l_y_ = L_y_

        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')
        _check_randomness_arg_1 = torch._functorch.vmap._check_randomness_arg('error')

        select = l_x_.select(0, 0)
        select_1 = l_y_.select(0, 0)
        vmap_body_1 = self.vmap_body_1
        vmap_proxy = torch.func.vmap(vmap_body_1, (0, 0), 0, 'error');  vmap_body_1 = None
        call = vmap_proxy.__call__(l_x_, l_y_);  vmap_proxy = l_x_ = l_y_ = None
        return (call,)

    class GraphModule(torch.nn.Module):
        def forward(self, select, select_1):
            select_2 = select.select(1, 0)
            select_3 = select_1.select(1, 0)
            vmap_body_0 = self.vmap_body_0
            vmap_proxy = torch.func.vmap(vmap_body_0, (1, 1), 0, 'error');  vmap_body_0 = None
            call = vmap_proxy.__call__(select, select_1);  vmap_proxy = select = select_1 = None
            return call

        class GraphModule(torch.nn.Module):
            def forward(self, select_2, select_3):
                add = select_2 + select_3;  select_2 = select_3 = None
                return add
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

    def test_vmap_over_vmap_captured(self):
        x = torch.ones(2, 3)
        y = torch.ones(5, 3)

        def fn(x):
            return torch.func.vmap(torch.func.vmap(lambda y: x * y))(y)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
        l_x_ = L_x_
        l_y_ = L_y_

        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')
        _check_randomness_arg_1 = torch._functorch.vmap._check_randomness_arg('error')

        select = l_y_.select(0, 0)
        vmap_body_1 = self.vmap_body_1
        vmap_proxy = torch.func.vmap(vmap_body_1, (0, None), 0, 'error');  vmap_body_1 = None
        call = vmap_proxy.__call__(l_y_, l_x_);  vmap_proxy = l_y_ = l_x_ = None
        return (call,)

    class GraphModule(torch.nn.Module):
        def forward(self, select, l_x_):
            select_1 = select.select(0, 0)
            vmap_body_0 = self.vmap_body_0
            vmap_proxy = torch.func.vmap(vmap_body_0, (0, None), 0, 'error');  vmap_body_0 = None
            call = vmap_proxy.__call__(select, l_x_);  vmap_proxy = select = l_x_ = None
            return call

        class GraphModule(torch.nn.Module):
            def forward(self, select_1, l_x_):
                mul = l_x_ * select_1;  l_x_ = select_1 = None
                return mul
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

    def test_vmap_multiple_outputs(self):
        x = torch.ones(2, 4, 3)

        def fn(x):
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)))(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')

        select = l_x_.select(0, 0)
        vmap_body_0 = self.vmap_body_0
        vmap_proxy = torch.func.vmap(vmap_body_0, (0,), 0, 'error');  vmap_body_0 = None
        call = vmap_proxy.__call__(l_x_);  vmap_proxy = l_x_ = None
        getitem = call[0]
        getitem_1 = call[1];  call = None
        return (getitem, getitem_1)

    class GraphModule(torch.nn.Module):
        def forward(self, select):
            sum_1 = select.sum(0)
            sum_2 = select.sum(1);  select = None
            return (sum_1, sum_2)
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

    def test_vmap_multiple_outputs_diff_dims(self):
        x = torch.ones(2, 4, 3)

        def fn(x):
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)), out_dims=(1, 0))(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')

        select = l_x_.select(0, 0)
        vmap_body_0 = self.vmap_body_0
        vmap_proxy = torch.func.vmap(vmap_body_0, (0,), (1, 0), 'error');  vmap_body_0 = None
        call = vmap_proxy.__call__(l_x_);  vmap_proxy = l_x_ = None
        getitem = call[0]
        getitem_1 = call[1];  call = None
        return (getitem, getitem_1)

    class GraphModule(torch.nn.Module):
        def forward(self, select):
            sum_1 = select.sum(0)
            sum_2 = select.sum(1);  select = None
            return (sum_1, sum_2)
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

    def test_vmap_multiple_outputs_out_dims_tuple(self):
        x = torch.ones(2, 4, 3)
        out_dims = (1, 0)

        def fn(x):
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)), out_dims=out_dims)(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_

        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')

        select = l_x_.select(0, 0)
        vmap_body_0 = self.vmap_body_0
        vmap_proxy = torch.func.vmap(vmap_body_0, (0,), (1, 0), 'error');  vmap_body_0 = None
        call = vmap_proxy.__call__(l_x_);  vmap_proxy = l_x_ = None
        getitem = call[0]
        getitem_1 = call[1];  call = None
        return (getitem, getitem_1)

    class GraphModule(torch.nn.Module):
        def forward(self, select):
            sum_1 = select.sum(0)
            sum_2 = select.sum(1);  select = None
            return (sum_1, sum_2)
"""
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, expected)

    def test_vmap_kwargs(self):
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        def fn(x, y):
            return torch.func.vmap(lambda x, y: x + y)(x, y=y)

        actual = fn(x, y)
        expected = torch.compile(fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"NYI - torch.func.vmap: kwargs arguments are currently unsupported.": 2},
        )
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
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"HigherOrderOperator with body that accepts non-Tensors as input": 2},
        )
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
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {
                "HigherOrderOperator: Mutating a variable not in the current scope (replace_all)": 2
            },
        )
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

    def test_vmap_illegal_op_graph_break(self):
        counters.clear()

        def bad_fn(x):
            x.stride()
            return x

        def wrapper_fn(x):
            return torch.func.vmap(bad_fn)(x)

        x = torch.randn(3, 3, 3)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"Illegal getattr invocation stride in strict mode": 2},
        )
        self.assertEqual(actual, expected)

    def test_vmap_multiple_invocation_in_dims(self):
        counters.clear()

        def wrapper_fn(x, in_dims):
            return torch.func.vmap(torch.sum, in_dims)(x)

        x = torch.randn(3, 3, 3, 3)
        opt = torch.compile(wrapper_fn, backend="eager", fullgraph=False, dynamic=True)
        expected = wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2)
        # Third invocation of `opt` makes `in_dims` as SymInt.
        actual = opt(x, 0), opt(x, 1), opt(x, 2)
        self.assertEqual(expected, actual)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"torch.func.vmap: in_dims is not an int or tuple variable.": 2},
        )

    def test_vmap_multiple_invocation_out_dims(self):
        counters.clear()

        def wrapper_fn(x, out_dims):
            return torch.func.vmap(lambda x: torch.sum(x, 0), out_dims=out_dims)(x)

        x = torch.randn(3, 3, 3, 3)
        opt = torch.compile(wrapper_fn, backend="eager", fullgraph=False, dynamic=True)
        expected = wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2)
        # Third invocation of `opt` makes `in_dims` as SymInt.
        actual = opt(x, 0), opt(x, 1), opt(x, 2)
        self.assertEqual(expected, actual)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"torch.func.vmap: out_dims is not an int or tuple variable.": 2},
        )

    def test_vmap_new_tensor_in_body(self):
        def fn(x):
            return x + torch.ones(3)

        def wrapper_fn(x):
            return torch.func.vmap(fn)(x)

        x = torch.randn(
            3,
        )
        opt = torch.compile(wrapper_fn, backend="eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)

    def test_vmap_new_tensor_unused_in_body(self):
        def fn(x):
            return torch.tensor(0.5)

        def wrapper_fn(x):
            return torch.func.vmap(fn)(x)

        x = torch.randn(3)
        opt = torch.compile(wrapper_fn, backend="eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)

    def test_vmap_new_tensor_implicit_via_op(self):
        def wrapper_fn(x):
            return torch.func.vmap(lambda t: torch.add(t, 0.5))(x)

        x = torch.randn(3)
        opt = torch.compile(wrapper_fn, backend="eager", fullgraph=True)
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

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function_with_kwargs(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True, preserve_rng_state=False
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout(self):
        def gn(x, y):
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)

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

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout_inductor(self):
        def gn(x, y):
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        backend = "inductor"
        self._validate(
            fn, backend, x, y, skip_check=True
        )  # dropout decomp is known to diverge with eager

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_fallback(self):
        def gn(x, y):
            torch._dynamo.graph_break()
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.cos(torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y))

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

    @requires_cuda()
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
            return torch.utils.checkpoint.checkpoint(mod, torch.sin(x))

        x = torch.randn(10, 10, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
