# Owner(s): ["module: dynamo"]
import re

import torch

import torch._dynamo.test_case
from torch._ops import wrap


class MockBackend:
    def __init__(self):
        self.graphs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.graphs.append(gm)
        return gm.forward


global_var = torch.randn(3)
global_num = 3.14


class TestHigherOrderOps(torch._dynamo.test_case.TestCase):
    def test_no_freevars(self):
        mock = MockBackend()

        def f(x):
            return wrap(lambda x: torch.sin(x), x)

        x = torch.randn(3)
        expected = f(x)
        result = torch.compile(f, backend=mock)(x)

        self.assertEqual(result, expected)
        self.assertEqual(len(mock.graphs), 1)
        self.assertIsNotNone(re.search(r"wrap\(\w+, \w+\);", mock.graphs[0].code))

    def test_capture_untracked_global(self):
        mock = MockBackend()

        def f(x):
            return wrap(lambda x: x + global_var, x)

        x = torch.randn(3)
        expected = f(x)
        result = torch.compile(f, backend=mock)(x)

        self.assertEqual(result, expected)
        self.assertEqual(len(mock.graphs), 1)
        # wrap(fn, x, global_var)
        self.assertIsNotNone(re.search(r"wrap\(\w+, \w+, \w+\);", mock.graphs[0].code))

    def test_capture_untracked_global_nested(self):
        mock = MockBackend()

        @torch.compile(backend=mock)
        def f(x):
            return wrap(lambda x: wrap(lambda x: x + global_var, x), x)

        x = torch.randn(3)
        result = f(x)

        self.assertEqual(result, x + global_var)
        self.assertEqual(len(mock.graphs), 1)
        gm = mock.graphs[0]
        self.assertIsNotNone(re.search(r"wrap\(\w+, \w+, \w+\);", gm.code))
        self.assertIsNotNone(re.search(r"wrap\(\w+, \w+, \w+\);", gm.cond_body_1.code))

    def test_capture_untracked_nonlocal(self):
        mock = MockBackend()

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            @torch.compile(backend=mock)
            def g(x):
                return wrap(lambda x: x + y, x)

            return g(x)

        result = f(x, y)
        expected = x + y

        self.assertEqual(result, expected)
        self.assertEqual(len(mock.graphs), 1)
        # wrap(fn, x, y)
        self.assertIsNotNone(re.search(r"wrap\(\w+, \w+, \w+\);", mock.graphs[0].code))

    def test_capture_tracked(self):
        mock = MockBackend()

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        @torch.compile(backend=mock)
        def f(x, y):
            return wrap(lambda x: x + y, x)

        result = f(x, y)

        self.assertEqual(result, x + y)
        self.assertEqual(len(mock.graphs), 1)
        self.assertIsNotNone(re.search(r"wrap\(\w+, \w+, \w+\);", mock.graphs[0].code))

    def test_inlined_functions(self):
        mock = MockBackend()

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def g(x, y):
            return x + y

        @torch.compile(backend=mock)
        def f(x, y):
            return wrap(lambda x: g(x, y), x)

        result = f(x, y)

        self.assertEqual(result, x + y)
        self.assertEqual(len(mock.graphs), 1)
        self.assertIsNotNone(re.search(r"wrap\(\w+, \w+, \w+\);", mock.graphs[0].code))

    def test_capture_value_created_in_subgraph(self):
        mock = MockBackend()

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=mock)
        def f(x, y):
            return wrap(inner, x, y)

        result = f(x, y)

        self.assertEqual(result, x + y + x)
        self.assertEqual(len(mock.graphs), 1)
        gm = mock.graphs[0]
        # Two inputs: no lifting
        self.assertIsNotNone(re.search(r"wrap\(\w+, \w+, \w+\);", gm.code))
        # z should have been lifted to input
        self.assertIsNotNone(re.search(r"wrap\(\w+, \w+, \w+\);", gm.cond_body_1.code))

    def test_capture_global_num(self):
        mock = MockBackend()
        x = torch.zeros([])

        @torch.compile(backend=mock)
        def f(x):
            return wrap(lambda x: x + global_num, x)

        global global_num
        result = f(x)
        self.assertEqual(result, x + global_num)
        self.assertEqual(len(mock.graphs), 1)
        gm = mock.graphs[0]
        # Numbers don't get lifted
        self.assertIsNotNone(re.search(r"wrap\(\w+, \w+\);", gm.code))

        # Check that we still guard on the number
        global_num = torch.randn([]).item()
        result = f(x)
        self.assertEqual(result, x + global_num)

    def test_capture_input_num(self):
        mock = MockBackend()
        x = torch.zeros([])
        y = 3.14

        @torch.compile(backend=mock)
        def f(x, y):
            return wrap(lambda x: x + y, x)

        result = f(x, y)
        self.assertEqual(result, x + y)
        self.assertEqual(len(mock.graphs), 1)
        gm = mock.graphs[0]
        # Numbers don't get lifted
        self.assertIsNotNone(re.search(r"wrap\(\w+, \w+\);", gm.code))

    def test_side_effect_in_body(self):
        from torch._dynamo.utils import counters

        counters.clear()

        mock = MockBackend()
        x = torch.randn([])
        y = torch.randn([])

        def inner(x):
            nonlocal y
            y = x
            return x.clone()

        @torch.compile(backend=mock)
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
        mock = MockBackend()
        x = torch.randn([])

        def inner(x):
            y = x.sin()
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=mock)
        def f(x):
            return wrap(inner, x)

        result = f(x)
        self.assertEqual(result, inner(x))
        # It's unclear if this is correct: dynamo graph breaks on wrap but
        # then interposes on wrap.__call__, which invokes fn(*args),
        # leading to two graphs being compiled
        self.assertEqual(len(mock.graphs), 2)

    def test_fallback_on_graph_break_complicated(self):
        mock = MockBackend()
        x = torch.randn([])

        def inner(x):
            y = x.sin()
            y = y * global_var
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=mock)
        def f(x):
            x = x.clone()
            result = wrap(inner, x)
            return result.clone()

        result = f(x)
        self.assertEqual(result, inner(x))
        # It's unclear if this is correct: dynamo graph breaks on wrap but
        # then interposes on wrap.__call__, which invokes fn(*args),
        # leading to four graphs being compiled: clone, sin, sin, clone
        self.assertEqual(len(mock.graphs), 4)

    def test_fallback_on_modules(self):
        # We can likely support this in the future, I just don't want to deal
        # with it right now
        from torch._dynamo.utils import counters

        counters.clear()
        mock = MockBackend()
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=mock)
        def f(x):
            return wrap(lambda x: mod(x), x)

        result = f(x)

        self.assertEqual(result, mod(x))
        self.assertEqual(len(mock.graphs), 1)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"Invoking an nn.Module inside HigherOrderOperator": 1},
        )

    def test_access_module_attr(self):
        # We can likely support this in the future, I just don't want to deal
        # with it right now
        from torch._dynamo.utils import counters

        counters.clear()

        mock = MockBackend()
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=mock)
        def f(x):
            y = mod(x)
            return wrap(lambda y: y - mod.bias, y)

        result = f(x)
        self.assertEqual(result, mod(x) - mod.bias)
        self.assertEqual(len(mock.graphs), 2)
        self.assertEqual(
            dict(counters["graph_break"]),
            {"accessing attribute of nn.Module inside HigherOrderOperator": 1},
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
