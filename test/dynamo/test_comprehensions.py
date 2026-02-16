# Owner(s): ["module: dynamo"]

"""
Tests for comprehension graph break handling

In Python 3.12+, comprehensions are inlined into the parent function's bytecode
rather than being compiled as separate code objects. These tests verify
that handling is the same as in previous Python versions, i.e.,
when a graph break occurs inside a comprehension, only the comprehension is
skipped (not the entire frame), resulting in multiple graphs being created.
"""

import operator

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import skipIfNotPy312


def count_op(graph, op):
    """Count occurrences of a specific operation in the graph."""
    return sum(1 for node in graph.graph.nodes if node.target == op)


@skipIfNotPy312
class ComprehensionTests(torch._dynamo.test_case.TestCase):
    def test_list_comprehension_graph_break(self):
        """Test that list comprehension with graph break creates 2 graphs."""

        def fn(x):
            y = x + 1
            result = [torch._dynamo.graph_break() or i for i in range(3)]
            z = x + 2
            return y, result, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_dict_comprehension_graph_break(self):
        """Test that dict comprehension with graph break creates 2 graphs."""

        def fn(x):
            y = x + 1
            result = {i: torch._dynamo.graph_break() or i**2 for i in range(3)}
            z = x + 2
            return y, result, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_list_comprehension_with_graph_break_function(self):
        """Test list comprehension calling a function that causes graph break."""

        def inner(i):
            if i == 3:
                torch._dynamo.graph_break()
            return i

        def fn(x):
            y = x * 2
            lst = [inner(i) for i in range(5)]
            z = x + sum(lst)
            return z, y

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.mul), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_simple_list_comprehension_no_break(self):
        """Test that simple list comprehension without graph break creates 1 graph."""

        def fn(x):
            y = x + torch.tensor([1, 2, 3, 4])
            result = [i * 2 for i in range(3)]
            return x + y + sum(result)

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 3)

    def test_multiple_comprehensions_one_break(self):
        """Test function with multiple comprehensions where only one has graph break."""

        def fn(x):
            a = x + 1
            list1 = [i for i in range(2)]  # noqa: C416
            list2 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 2
            list3 = [i * 2 for i in range(2)]
            return a, list1, list2, b, list3

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_comprehension_inner_break(self):
        """Test nested comprehension where inner comprehension causes graph break."""

        def fn(x):
            a = x + 1
            result = [
                [torch._dynamo.graph_break() or i * j for j in range(2)]
                for i in range(2)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_multi_iterator_comprehension_break(self):
        """Test comprehension with multiple iterators where graph break occurs."""

        def fn(x):
            a = x + 1
            result = [
                (torch._dynamo.graph_break() or i, j)
                for i in range(2)
                for j in range(2)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_discarded_comprehension_graph_break(self):
        """Test that discarded comprehension (result not assigned) with graph break works."""

        def fn(x):
            a = x + 1
            [torch._dynamo.graph_break() or i for i in range(3)]
            b = x + 2
            return a, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_in_expression_graph_break(self):
        """Test comprehension used in expression (e.g., sum([...])) with graph break."""

        def fn(x):
            a = x + 1
            total = sum([torch._dynamo.graph_break() or i for i in range(3)])
            b = x + 2
            return a, total, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_return_directly(self):
        """Test comprehension returned directly with graph break."""

        def fn(x):
            a = x + 1  # noqa: F841
            return [torch._dynamo.graph_break() or i for i in range(3)]

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)

    def test_walrus_operator_in_comprehension(self):
        """Test walrus operator (:=) inside comprehension with graph break."""

        def fn(x):
            a = x + 1
            result = [(y := (torch._dynamo.graph_break() or i * 2)) for i in range(3)]
            b = x + 2
            return a, result, y, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_walrus_operator_in_if_in_comprehension(self):
        """Test walrus operator (:=) in if clause of comprehension with graph break."""

        def fn(x):
            a = x + 1
            result = [
                (torch._dynamo.graph_break() or y) for i in range(5) if (y := i * 2) > 2
            ]
            b = x + 2
            return a, result, y, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_walrus_operator_in_comprehension_with_tensor(self):
        """Test walrus operator with tensor operation in comprehension with graph break."""

        def fn(x):
            a = x + 1
            result = [
                (torch._dynamo.graph_break() or y + x.numel())
                for i in range(5)
                if (y := i * 2) > 2
            ]
            b = x + 2
            return a, result, y, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_multiple_walrus_operators_in_comprehension(self):
        """Test multiple walrus operators (:=) inside comprehension with graph break."""

        def fn(x):
            a = x + 1
            result = [
                ((y := (torch._dynamo.graph_break() or i * 2)), (z := i * 3))
                for i in range(3)
            ]
            b = x + 2
            return a, result, y, z, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_comprehension_with_walrus_operators(self):
        """Test nested comprehension with walrus operators in both inner and outer."""

        def fn(x):
            a = x + 1
            result = [
                (
                    outer := i * 10,
                    [
                        (inner := (torch._dynamo.graph_break() or j * 2))
                        for j in range(2)
                    ],
                )
                for i in range(3)
            ]
            b = x + 2
            return a, result, outer, inner, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_comprehension_with_captured_outer_variable(self):
        """Test nested comprehension with captured outer variable in inner loop."""

        def fn(x):
            outer_val = 100
            a = x + 1
            result = [
                [outer_val + j for j in range(2) if torch._dynamo.graph_break() or True]
                for i in range(2)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_triple_nested_comprehension_with_walrus(self):
        """Test triple-nested comprehension with walrus at middle level."""

        def fn(x):
            a = x + 1
            result = [
                [
                    [
                        (w := i + j + k)
                        for k in range(2)
                        if torch._dynamo.graph_break() or True
                    ]
                    for j in range(2)
                ]
                for i in range(2)
            ]
            b = x + 2
            return a, result, w, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_dict_comprehension_with_walrus_and_captured(self):
        """Test nested dict comprehension with walrus in inner and captured variable."""

        def fn(x):
            multiplier = 10
            a = x + 1
            result = {
                i: {
                    j: (inner_val := j * multiplier)
                    for j in range(2)
                    if torch._dynamo.graph_break() or True
                }
                for i in range(3)
            }
            b = x + 2
            return a, result, inner_val, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_outer_variable_read(self):
        """Test accessing outer variable inside comprehension with graph break."""

        def fn(x):
            outer_val = 10
            a = x + 1
            result = [
                i + outer_val for i in range(3) if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_outer_list_mutation(self):
        """Test mutating outer list inside comprehension with graph break."""

        def fn(x):
            outer_list = []
            a = x + 1
            result = [
                outer_list.append(i) or i * 2
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, outer_list, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_outer_dict_mutation(self):
        """Test mutating outer dict inside comprehension with graph break."""

        def fn(x):
            outer_dict = {}
            a = x + 1
            result = [
                outer_dict.update({i: i * 10}) or i * 2
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, outer_dict, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_outer_list_extend(self):
        """Test extending outer list inside comprehension with graph break."""

        def fn(x):
            outer_list = [100]
            a = x + 1
            result = [
                outer_list.extend([i, i * 10]) or i
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, outer_list, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_outer_list_pop(self):
        """Test popping from outer list inside comprehension with graph break."""

        def fn(x):
            outer_list = [10, 20, 30, 40, 50]
            popped_values = []
            a = x + 1
            result = [
                popped_values.append(outer_list.pop()) or i
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, outer_list, popped_values, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_global_variable(self):
        """Test global variable access inside comprehension with graph break."""

        global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = 100

        def fn(x):
            a = x + 1
            result = [
                i + _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_closure_variable(self):
        """Test closure variable access inside comprehension with graph break."""

        def make_fn():
            closure_val = 50

            def fn(x):
                a = x + 1
                result = [
                    i + closure_val
                    for i in range(3)
                    if torch._dynamo.graph_break() or True
                ]
                b = x + 2
                return a, result, b

            return fn

        fn = make_fn()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_closure_list_mutation(self):
        """Test closure list mutation inside comprehension with graph break."""

        def make_fn():
            closure_list = []

            def fn(x):
                a = x + 1
                result = [
                    closure_list.append(i) or i * 2
                    for i in range(3)
                    if torch._dynamo.graph_break() or True
                ]
                b = x + 2
                return a, result, closure_list.copy(), b

            return fn

        fn = make_fn()
        fn_ref = make_fn()

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn_ref(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_multi_for_comprehension_graph_break(self):
        """Test nested comprehension with multiple for loops and graph break."""

        def fn(x):
            a = x + 1
            result = [
                [(torch._dynamo.graph_break() or i + j + k) for k in range(2)]
                for i in range(2)
                for j in range(2)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_multiple_comprehension_graph_breaks(self):
        """Test multiple comprehensions with graph breaks producing 3 graphs."""

        def fn(x):
            a = x + 1
            list1 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 2
            list2 = [torch._dynamo.graph_break() or i * 2 for i in range(2)]
            c = x + 3
            return a, list1, b, list2, c

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 1)

    def test_comprehension_modifying_global_variable(self):
        """Test modifying a global variable inside comprehension with graph break."""
        global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = 100
        original_value = _GLOBAL_VALUE_FOR_COMPREHENSION_TEST

        def fn(x):
            global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
            a = x + 1
            _GLOBAL_VALUE_FOR_COMPREHENSION_TEST += 1
            result = [
                torch._dynamo.graph_break() or _GLOBAL_VALUE_FOR_COMPREHENSION_TEST + i
                for i in range(3)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value
        compiled_result = compiled(x)
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value
        expected_result = fn(x)
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value

        self.assertEqual(compiled_result, expected_result)
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_modifying_closure_variable(self):
        """Test modifying a closure variable inside comprehension with graph break."""

        def make_fn():
            closure_val = [0]

            def fn(x):
                a = x + 1
                closure_val[0] += 1
                result = [
                    torch._dynamo.graph_break() or closure_val[0] + i for i in range(3)
                ]
                b = x + 2
                return a, result, closure_val[0], b

            return fn

        fn = make_fn()
        fn_ref = make_fn()

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn_ref(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_list_and_dict_comprehension_graph_breaks(self):
        """Test list and dict comprehensions with graph breaks together."""

        def fn(x):
            a = x + 1
            list1 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 2
            dict1 = {i: torch._dynamo.graph_break() or i * 10 for i in range(2)}
            c = x + 3
            return a, list1, b, dict1, c

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 1)

    def test_nested_dict_in_list_comprehension_graph_break(self):
        """Test dict comprehension nested in list comprehension with graph break."""

        def fn(x):
            a = x + 1
            result = [
                {j: torch._dynamo.graph_break() or i * j for j in range(2)}
                for i in range(2)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_nested_function_calls_with_comprehension_graph_break(self):
        """Test nested function calls where inner function has comprehension with graph break."""

        def h(x):
            x = x + 3
            [torch._dynamo.graph_break() or i for i in range(3)]
            x = x + 4
            return x

        def g(x):
            x = x + 2
            x = h(x)
            x = x + 5
            return x

        def f(x):
            x = x + 1
            x = g(x)
            x = x + 6
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f(x))
        self.assertEqual(len(backend.graphs), 6)
        for i in range(6):
            self.assertEqual(count_op(backend.graphs[i], operator.add), 1)


@skipIfNotPy312
class NestedGraphBreakTests(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_function_calls_with_comprehension_graph_break_nested_graph_breaks_true(
        self,
    ):
        """Test nested function calls where inner function has comprehension with graph break."""

        def h(x):
            x = x + 3
            [torch._dynamo.graph_break() or i for i in range(3)]
            x = x + 4
            return x

        def g(x):
            x = x + 2
            x = h(x)
            x = x + 5
            return x

        def f(x):
            x = x + 1
            x = g(x)
            x = x + 6
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 3)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 3)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_result_var_name_collision(self):
        """Comprehension result_var in leaf frame shares a name with a root frame local."""

        def inner(t):
            x = [torch._dynamo.graph_break() or i for i in range(3)]
            return sum(x) + t

        def root(t):
            x = t + 1
            result = inner(t) + x
            self.assertEqual(x, t + 1)
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(root, backend=backend)
        t = torch.randn(4)

        self.assertEqual(compiled(t), root(t))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 3)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_function_comprehension_with_walrus(self):
        """Nested function with comprehension containing walrus operator."""

        def inner(x):
            x = x + 2
            result = [(y := i * 2) for i in [torch._dynamo.graph_break() or 1, 2, 3]]
            x = x + 3
            return x + sum(result) + y

        def outer(x):
            x = x + 1
            x = inner(x)
            x = x + 4
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 4)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_function_comprehension_with_captured_var(self):
        """Nested function with comprehension capturing outer variable passed as parameter."""

        def inner(x, multiplier):
            x = x + 2
            result = [multiplier * i for i in [torch._dynamo.graph_break() or 1, 2, 3]]
            x = x + 3 + sum(result)
            return x

        def outer(x):
            multiplier = 5
            x = x + 1
            x = inner(x, multiplier)
            x = x + 4
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 3)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_triple_nested_function_comprehension(self):
        """Triple nested function calls (f->g->h->deepest) with comprehension in deepest."""

        def deepest(x):
            x = x + 4
            [torch._dynamo.graph_break() or i for i in range(3)]
            x = x + 5
            return x

        def h(x):
            x = x + 3
            x = deepest(x)
            x = x + 6
            return x

        def g(x):
            x = x + 2
            x = h(x)
            x = x + 7
            return x

        def f(x):
            x = x + 1
            x = g(x)
            x = x + 8
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 4)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 4)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_multiple_nested_comprehensions_in_different_functions(self):
        """Two sequential nested function calls each with comprehensions."""

        def inner1(x):
            x = x + 1
            [torch._dynamo.graph_break() or i for i in range(2)]
            x = x + 2
            return x

        def inner2(x):
            x = x + 3
            [torch._dynamo.graph_break() or i for i in range(2)]
            x = x + 4
            return x

        def outer(x):
            x = inner1(x)
            x = inner2(x)
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        # Graph 1: x+1, Graph 2: x+2, x+3, Graph 3: x+4
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 1)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_function_comprehension_with_outer_list_mutation(self):
        """Nested function with comprehension mutating outer list."""

        def inner(x, results):
            x = x + 2
            [results.append(torch._dynamo.graph_break() or i) for i in range(3)]
            x = x + 3
            return x

        def outer(x):
            results = []
            x = x + 1
            x = inner(x, results)
            x = x + 4
            return x, results

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_list_comprehension_graph_break(self):
        """Nested version of test_list_comprehension_graph_break."""

        def inner(x):
            y = x + 2
            result = [torch._dynamo.graph_break() or i for i in range(3)]
            z = x + 3
            return y, result, z

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_dict_comprehension_graph_break(self):
        """Nested version of test_dict_comprehension_graph_break."""

        def inner(x):
            y = x + 2
            result = {i: torch._dynamo.graph_break() or i**2 for i in range(3)}
            z = x + 3
            return y, result, z

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_list_comprehension_with_graph_break_function(self):
        """Nested version of test_list_comprehension_with_graph_break_function."""

        def graph_break_fn(i):
            if i == 3:
                torch._dynamo.graph_break()
            return i

        def inner(x):
            y = x * 2
            lst = [graph_break_fn(i) for i in range(5)]
            z = x + 3 + sum(lst)
            return z, y

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 5
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[0], operator.mul), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 3)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_multiple_comprehensions_one_break(self):
        """Nested version of test_multiple_comprehensions_one_break."""

        def inner(x):
            a = x + 2
            list1 = [i for i in range(2)]  # noqa: C416
            list2 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 3
            list3 = [i * 2 for i in range(2)]
            return a, list1, list2, b, list3

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_nested_comprehension_inner_break(self):
        """Nested version of test_nested_comprehension_inner_break."""

        def inner(x):
            a = x + 2
            result = [
                [torch._dynamo.graph_break() or i * j for j in range(2)]
                for i in range(2)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_multi_iterator_comprehension_break(self):
        """Nested version of test_multi_iterator_comprehension_break."""

        def inner(x):
            a = x + 2
            result = [
                (torch._dynamo.graph_break() or i, j)
                for i in range(2)
                for j in range(2)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_discarded_comprehension_graph_break(self):
        """Nested version of test_discarded_comprehension_graph_break."""

        def inner(x):
            a = x + 2
            [torch._dynamo.graph_break() or i for i in range(3)]
            b = x + 3
            return a, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_in_expression_graph_break(self):
        """Nested version of test_comprehension_in_expression_graph_break."""

        def inner(x):
            a = x + 2
            total = sum([torch._dynamo.graph_break() or i for i in range(3)])
            b = x + 3
            return a, total, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_return_directly(self):
        """Nested version of test_comprehension_return_directly."""

        def inner(x):
            a = x + 2  # noqa: F841
            return [torch._dynamo.graph_break() or i for i in range(3)]

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 3
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_walrus_operator_in_comprehension(self):
        """Nested version of test_walrus_operator_in_comprehension."""

        def inner(x):
            a = x + 2
            result = [(y := (torch._dynamo.graph_break() or i * 2)) for i in range(3)]
            b = x + 3
            return a, result, y, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_walrus_operator_in_if_in_comprehension(self):
        """Nested version of test_walrus_operator_in_if_in_comprehension."""

        def inner(x):
            a = x + 2
            result = [
                (torch._dynamo.graph_break() or y) for i in range(5) if (y := i * 2) > 2
            ]
            b = x + 3
            return a, result, y, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_walrus_operator_in_comprehension_with_tensor(self):
        """Nested version of test_walrus_operator_in_comprehension_with_tensor."""

        def inner(x):
            a = x + 2
            result = [
                (torch._dynamo.graph_break() or y + x.numel())
                for i in range(5)
                if (y := i * 2) > 2
            ]
            b = x + 3
            return a, result, y, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_multiple_walrus_operators_in_comprehension(self):
        """Nested version of test_multiple_walrus_operators_in_comprehension."""

        def inner(x):
            a = x + 2
            result = [
                ((y := (torch._dynamo.graph_break() or i * 2)), (z := i * 3))
                for i in range(3)
            ]
            b = x + 3
            return a, result, y, z, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_nested_comprehension_with_walrus_operators(self):
        """Nested version of test_nested_comprehension_with_walrus_operators."""

        def inner(x):
            a = x + 2
            result = [
                (
                    outer_val := i * 10,
                    [
                        (inner_val := (torch._dynamo.graph_break() or j * 2))
                        for j in range(2)
                    ],
                )
                for i in range(3)
            ]
            b = x + 3
            return a, result, outer_val, inner_val, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_nested_comprehension_with_captured_outer_variable(self):
        """Nested version of test_nested_comprehension_with_captured_outer_variable."""

        def inner(x):
            outer_val = 100
            a = x + 2
            result = [
                [outer_val + j for j in range(2) if torch._dynamo.graph_break() or True]
                for i in range(2)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_triple_nested_comprehension_with_walrus(self):
        """Nested version of test_triple_nested_comprehension_with_walrus."""

        def inner(x):
            a = x + 2
            result = [
                [
                    [
                        (w := i + j + k)
                        for k in range(2)
                        if torch._dynamo.graph_break() or True
                    ]
                    for j in range(2)
                ]
                for i in range(2)
            ]
            b = x + 3
            return a, result, w, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_nested_dict_comprehension_with_walrus_and_captured(self):
        """Nested version of test_nested_dict_comprehension_with_walrus_and_captured."""

        def inner(x):
            multiplier = 10
            a = x + 2
            result = {
                i: {
                    j: (inner_val := j * multiplier)
                    for j in range(2)
                    if torch._dynamo.graph_break() or True
                }
                for i in range(3)
            }
            b = x + 3
            return a, result, inner_val, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_with_outer_variable_read(self):
        """Nested version of test_comprehension_with_outer_variable_read."""

        def inner(x):
            outer_val = 10
            a = x + 2
            result = [
                i + outer_val for i in range(3) if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_with_outer_list_mutation(self):
        """Nested version of test_comprehension_with_outer_list_mutation."""

        def inner(x):
            outer_list = []
            a = x + 2
            result = [
                outer_list.append(i) or i * 2
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, outer_list, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_with_outer_dict_mutation(self):
        """Nested version of test_comprehension_with_outer_dict_mutation."""

        def inner(x):
            outer_dict = {}
            a = x + 2
            result = [
                outer_dict.update({i: i * 10}) or i * 2
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, outer_dict, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_with_outer_list_extend(self):
        """Nested version of test_comprehension_with_outer_list_extend."""

        def inner(x):
            outer_list = [100]
            a = x + 2
            result = [
                outer_list.extend([i, i * 10]) or i
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, outer_list, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_with_outer_list_pop(self):
        """Nested version of test_comprehension_with_outer_list_pop."""

        def inner(x):
            outer_list = [10, 20, 30, 40, 50]
            popped_values = []
            a = x + 2
            result = [
                popped_values.append(outer_list.pop()) or i
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, outer_list, popped_values, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_with_global_variable(self):
        """Nested version of test_comprehension_with_global_variable."""
        global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = 100

        def inner(x):
            a = x + 2
            result = [
                i + _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_with_closure_variable(self):
        """Nested version of test_comprehension_with_closure_variable."""

        def make_outer():
            closure_val = 50

            def inner(x):
                a = x + 2
                result = [
                    i + closure_val
                    for i in range(3)
                    if torch._dynamo.graph_break() or True
                ]
                b = x + 3
                return a, result, b

            def outer(x):
                x = x + 1
                result = inner(x)
                x = x + 4
                return result

            return outer

        outer = make_outer()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_with_closure_list_mutation(self):
        """Nested version of test_comprehension_with_closure_list_mutation."""

        def make_outer():
            closure_list = []

            def inner(x):
                a = x + 2
                result = [
                    closure_list.append(i) or i * 2
                    for i in range(3)
                    if torch._dynamo.graph_break() or True
                ]
                b = x + 3
                return a, result, closure_list.copy(), b

            def outer(x):
                x = x + 1
                result = inner(x)
                x = x + 4
                return result

            return outer

        outer = make_outer()
        outer_ref = make_outer()

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer_ref(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_nested_multi_for_comprehension_graph_break(self):
        """Nested version of test_nested_multi_for_comprehension_graph_break."""

        def inner(x):
            a = x + 2
            result = [
                [(torch._dynamo.graph_break() or i + j + k) for k in range(2)]
                for i in range(2)
                for j in range(2)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_multiple_comprehension_graph_breaks(self):
        """Nested version of test_multiple_comprehension_graph_breaks."""

        def inner(x):
            a = x + 2
            list1 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 3
            list2 = [torch._dynamo.graph_break() or i * 2 for i in range(2)]
            c = x + 4
            return a, list1, b, list2, c

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 5
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_modifying_global_variable(self):
        """Nested version of test_comprehension_modifying_global_variable."""
        global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = 100
        original_value = _GLOBAL_VALUE_FOR_COMPREHENSION_TEST

        def inner(x):
            global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
            a = x + 2
            _GLOBAL_VALUE_FOR_COMPREHENSION_TEST += 1
            result = [
                torch._dynamo.graph_break() or _GLOBAL_VALUE_FOR_COMPREHENSION_TEST + i
                for i in range(3)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value
        compiled_result = compiled(x)
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value
        expected_result = outer(x)
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value

        self.assertEqual(compiled_result, expected_result)
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_modifying_closure_variable(self):
        """Nested version of test_comprehension_modifying_closure_variable."""

        def make_outer():
            closure_val = [0]

            def inner(x):
                a = x + 2
                closure_val[0] += 1
                result = [
                    torch._dynamo.graph_break() or closure_val[0] + i for i in range(3)
                ]
                b = x + 3
                return a, result, closure_val[0], b

            def outer(x):
                x = x + 1
                result = inner(x)
                x = x + 4
                return result

            return outer

        outer = make_outer()
        outer_ref = make_outer()

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer_ref(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_list_and_dict_comprehension_graph_breaks(self):
        """Nested version of test_list_and_dict_comprehension_graph_breaks."""

        def inner(x):
            a = x + 2
            list1 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 3
            dict1 = {i: torch._dynamo.graph_break() or i * 10 for i in range(2)}
            c = x + 4
            return a, list1, b, dict1, c

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 5
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_nested_dict_in_list_comprehension_graph_break(self):
        """Nested version of test_nested_dict_in_list_comprehension_graph_break."""

        def inner(x):
            a = x + 2
            result = [
                {j: torch._dynamo.graph_break() or i * j for j in range(2)}
                for i in range(2)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_nonlocal_int_mutation(self):
        """Nonlocal int mutation around a comprehension graph break."""

        def f1(x):
            cell = 1

            def f2(x):
                b = x + 1  # noqa: F841
                nonlocal cell
                cell = cell + 1
                [torch._dynamo.graph_break() or i for i in range(2)]
                cell += 1
                return x + cell

            result = f2(x)
            self.assertEqual(cell, 3)
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f1, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f1(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_nonlocal_tensor_mutation(self):
        """Nonlocal tensor mutation around a comprehension graph break."""

        def f1(x):
            cell = torch.ones(4)

            def f2(x):
                b = x + 1  # noqa: F841
                nonlocal cell
                cell += 1
                [torch._dynamo.graph_break() or i for i in range(2)]
                cell = cell + 1
                return x + cell

            result = f2(x)
            self.assertEqual(cell, torch.full((4,), 3, dtype=torch.float32))
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f1, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f1(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_nonlocal_different_varnames(self):
        """Inner function uses different variable names than root function."""

        def f1(x):
            cell = 1

            def f2(y):
                a = x + 1
                b = y + 2
                nonlocal cell
                cell = cell + (b - y).sum()
                [torch._dynamo.graph_break() or y.sum() * i for i in range(2)]
                cell = cell + 1 + y.sum()
                return x + a + y

            result = f2(x)
            self.assertEqual(cell, 14)
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f1, backend=backend)
        x = torch.ones(4)

        self.assertEqual(compiled(x), f1(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 3)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 4)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_nonlocal_tensor_different_varnames(self):
        """Inner function uses different variable names than root function (tensor variant)."""

        def f1(x):
            cell = torch.ones(4)

            def f2(y):
                a = y + 1  # noqa: F841
                b = x + 2
                nonlocal cell
                cell += 1
                [torch._dynamo.graph_break() or i for i in range(2)]
                cell = cell + 1
                return b + cell

            result = f2(x)
            self.assertEqual(cell, torch.full((4,), 3, dtype=torch.float32))
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f1, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f1(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_captured_tensor(self):
        """Comprehension in inlined function captures a tensor from the inlined function."""

        def inner(x):
            t = x + 2
            result = [torch._dynamo.graph_break() or t.item() for i in range(2)]
            return x + 3, result

        def outer(x):
            x = x + 1
            return inner(x)

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.tensor(1.0)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_comprehension_name_shadowing(self):
        """Root and leaf frames have variables with the same name."""

        def inner(x):
            a = x + 2
            result = [torch._dynamo.graph_break() or a.item() for i in range(2)]
            self.assertEqual(a, torch.full((), 4, dtype=torch.float32))
            b = x + 3
            return b, result

        def outer(x):
            a = x + 1
            r = inner(a)
            self.assertEqual(a, torch.full((), 2, dtype=torch.float32))
            return a + 4, r

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.tensor(1.0)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 4)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 0)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[3], operator.add), 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
