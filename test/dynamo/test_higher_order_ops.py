# Owner(s): ["module: dynamo"]
import unittest

import functorch.experimental.control_flow as control_flow

import torch

import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend
from torch._dynamo.utils import counters, ifdyn, ifdynstaticdefault
from torch._higher_order_ops.wrap import wrap


# Equivalent to backend="eager", but also records graphs that
# we can assert on
class EagerAndRecordGraphs:
    def __init__(self):
        self.graphs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.graphs.append(gm)
        return gm


global_var = torch.randn(3)
global_num = 3.14


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

    def test_capture_untracked_global(self):
        def f(x):
            return wrap(lambda x: x + global_var, x)

        x = torch.randn(3)
        self._test_wrap_simple(f, (x,), 3)

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

    def test_inlined_functions(self):
        def g(x, y):
            return x + y

        def f(x, y):
            return wrap(lambda x: g(x, y), x)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, (x, y), 3)

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
        self.assertEqual(name_set, {"", "map_body_0.map_body_0", "map_body_0"})

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
                "cond_true_0",
                "cond_false_0",
                "cond_false_0.cond_false_0",
                "cond_false_0.cond_true_0",
            },
        )

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

    # TODO: Ideally we would error out if there are any new live side
    # effects (for example, if the body function mutates a global variable).
    # I don't know how to detect this in a robust way, because it conflicts with
    # benign side effects like storing and loading cells that is necessary for
    # capturing variables.
    @unittest.expectedFailure
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
            {"side effects in HigherOrderOperator body": 1},
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
        # It's unclear if this is correct: dynamo graph breaks on wrap but
        # then interposes on wrap.__call__, which invokes fn(*args),
        # leading to two graphs being compiled
        self.assertEqual(cnt.frame_count, 2)

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
        # It's unclear if this is correct: dynamo graph breaks on wrap but
        # then interposes on wrap.__call__, which invokes fn(*args),
        # leading to four graphs being compiled: clone, sin, sin, clone
        self.assertEqual(cnt.frame_count, 4)

    def test_fallback_on_modules(self):
        # We can likely support this in the future, I just don't want to deal
        # with it right now
        counters.clear()
        cnt = CompileCounter()
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt)
        def f(x):
            return wrap(lambda x: mod(x), x)

        result = f(x)

        self.assertEqual(result, mod(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"Invoking an nn.Module inside HigherOrderOperator": 1},
        )

    def test_fallback_on_non_single_tensor_output(self):
        # We can likely support this in the future, I just don't want to deal
        # with it right now
        counters.clear()
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            return wrap(lambda x: (x.sin(), x.cos()), x)

        x = torch.randn(2, 3)
        result = f(x)

        self.assertEqual(result, (x.sin(), x.cos()))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"HigherOrderOperator with body with non single Tensor output": 1},
        )

    def test_fallback_on_non_tensor_inputs(self):
        # We can likely support this in the future, I just don't want to deal
        # with it right now
        counters.clear()
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            return wrap(lambda x, y: x + y, x, 193)

        x = torch.randn(2, 3)
        result = f(x)

        self.assertEqual(result, x + 193)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"HigherOrderOperator with body that accepts non-Tensors as input": 1},
        )

    def test_access_module_attr(self):
        # We can likely support this in the future, I just don't want to deal
        # with it right now
        counters.clear()
        cnt = CompileCounter()
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt)
        def f(x):
            y = mod(x)
            return wrap(lambda y: y - mod.bias, y)

        result = f(x)
        self.assertEqual(result, mod(x) - mod.bias)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"accessing attribute of nn.Module inside HigherOrderOperator": 1},
        )

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

    def test_capture_numpy_number(self):
        import numpy as np

        y = np.float32(1.0)

        def f(x):
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        # np.number are lifted to graph inputs
        self._test_wrap_simple(f, (x,), 3)

    @torch._dynamo.config.patch(specialize_int=False, dynamic_shapes=True)
    def test_capture_uncommon_int(self):
        y = 328

        def f(x):
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        # Under this specific config, uncommon ints are lifted to graph inputs.
        self._test_wrap_simple(f, (x,), ifdyn(ifdynstaticdefault(2, 3), 3))

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
