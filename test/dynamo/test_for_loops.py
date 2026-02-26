# Owner(s): ["module: dynamo"]

"""
Tests for for loop graph break handling.

When a graph break occurs inside a for loop body, the loop is extracted
into a synthetic function that runs eagerly, enabling partial graph
compilation before and after the loop.
"""

import operator

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import skipIfNotPy312


_global_counter = 0
_modify_global = 0


def count_op(graph, op):
    """Count occurrences of a specific operation in the graph."""
    return sum(1 for node in graph.graph.nodes if node.target == op)


class ForLoopTests(torch._dynamo.test_case.TestCase):
    def test_for_loop_graph_break(self):
        def fn(x):
            y = x + 1
            for i in range(3):
                torch._dynamo.graph_break()
            z = x + 2
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_for_loop_modifies_local(self):
        def fn(x):
            y = x + 1
            a = 0
            for i in range(3):
                torch._dynamo.graph_break()
                a += i
            z = x + a
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    def test_simple_for_loop_no_break_no_graph_break(self):
        def fn(x):
            y = x + 1
            total = 0
            for i in range(3):
                total += i
            z = x + 2 + total
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 1)

    # Test 4: for loop calling a function that causes graph break
    def test_for_loop_with_graph_break_function(self):
        def inner(v):
            if v == 3:
                torch._dynamo.graph_break()
            return v

        def fn(x):
            y = x * 2
            total = 0
            for i in range(5):
                total += inner(i)
            z = x + total
            return z, y

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 5: two for loops, only first has graph break
    def test_multiple_for_loops_one_graph_break(self):
        def fn(x):
            y = x + 1
            for i in range(3):
                torch._dynamo.graph_break()
            total = 0
            for i in range(3):
                total += i
            z = x + 2 + total
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 6: two for loops, both have graph breaks
    def test_multiple_for_loops_multiple_graph_breaks(self):
        def fn(x):
            y = x + 1
            for i in range(3):
                torch._dynamo.graph_break()
            for j in range(3):
                torch._dynamo.graph_break()
            z = x + 2
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        # First for loop produces graph 1 (pre-loop), then runs eagerly.
        # Resume traces the second for loop which also graph-breaks and runs
        # eagerly, producing graph 2 (post-loop2). Total: 2.
        self.assertEqual(len(backend.graphs), 2)

    # Test 7: for loop with break statement
    def test_for_loop_break(self):
        def fn(x):
            y = x + 1
            result = 0
            for i in range(5):
                torch._dynamo.graph_break()
                if i == 3:
                    break
                result += i
            z = x + result
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)
        # result should be 0+1+2 = 3
        self.assertEqual(result[1], x + 3)
        self.assertEqual(len(backend.graphs), 2)

    # Test 8: for loop with continue statement
    def test_for_loop_continue(self):
        def fn(x):
            y = x + 1
            result = 0
            for i in range(5):
                torch._dynamo.graph_break()
                if i % 2 == 0:
                    continue
                result += i
            z = x + result
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 9: for/else (else executes when no break)
    def test_for_loop_else(self):
        def fn(x):
            y = x + 1
            found = False
            for i in range(3):
                torch._dynamo.graph_break()
            else:
                found = True
            z = x + 2
            return y, found, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)
        self.assertTrue(result[1])
        self.assertEqual(len(backend.graphs), 2)

    # Test 10: break skips else clause
    def test_for_loop_break_skips_else(self):
        def fn(x):
            y = x + 1
            found = False
            for i in range(3):
                torch._dynamo.graph_break()
                if i == 1:
                    break
            else:
                found = True
            z = x + 2
            return y, found, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        result = compiled(x)
        expected = fn(x)
        self.assertEqual(result, expected)
        self.assertFalse(result[1])
        self.assertEqual(len(backend.graphs), 2)

    # Test 11: conditional break
    def test_for_loop_break_conditional(self):
        def fn(x):
            y = x + 1
            last = -1
            for i in range(10):
                torch._dynamo.graph_break()
                last = i
                if i >= 4:
                    break
            z = x + last
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 12: multiple graph breaks in a single loop body
    def test_for_loop_multiple_graph_breaks_in_body(self):
        def fn(x):
            y = x + 1
            total = 0
            for i in range(3):
                torch._dynamo.graph_break()
                total += i
                torch._dynamo.graph_break()
            z = x + total
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 13: modifies multiple locals
    def test_for_loop_modifies_multiple_locals(self):
        def fn(x):
            y = x + 1
            a = 0
            b = 1
            for i in range(3):
                torch._dynamo.graph_break()
                a += i
                b *= i + 1
            z = x + a + b
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 14: reads outer variable
    def test_for_loop_with_outer_variable_read(self):
        def fn(x):
            y = x + 1
            factor = 3
            total = 0
            for i in range(4):
                torch._dynamo.graph_break()
                total += i * factor
            z = x + total
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 15: mutates outer list
    def test_for_loop_with_outer_list_mutation(self):
        def fn(x):
            y = x + 1
            items = []
            for i in range(3):
                torch._dynamo.graph_break()
                items.append(i)
            z = x + sum(items)
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 16: mutates outer dict
    def test_for_loop_with_outer_dict_mutation(self):
        def fn(x):
            y = x + 1
            d = {}
            for i in range(3):
                torch._dynamo.graph_break()
                d[i] = i * 2
            z = x + sum(d.values())
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 17: extends outer list
    def test_for_loop_with_outer_list_extend(self):
        def fn(x):
            y = x + 1
            items = [10]
            for i in range(3):
                torch._dynamo.graph_break()
                items.extend([i, i * 2])
            z = x + sum(items)
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 18: pops from outer list
    def test_for_loop_with_outer_list_pop(self):
        def fn(x):
            y = x + 1
            items = [10, 20, 30]
            popped = []
            for i in range(2):
                torch._dynamo.graph_break()
                popped.append(items.pop())
            z = x + sum(popped)
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 19: global variable
    def test_for_loop_with_global_variable(self):
        def fn(x):
            global _global_counter
            y = x + 1
            _global_counter = 0
            for i in range(3):
                torch._dynamo.graph_break()
                _global_counter += i
            z = x + _global_counter
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 20: closure variable (read-only)
    def test_for_loop_with_closure_variable(self):
        def fn(x):
            y = x + 1
            captured = 5
            total = 0
            for i in range(3):
                torch._dynamo.graph_break()
                total += captured
            z = x + total
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 21: closure list mutation
    def test_for_loop_with_closure_list_mutation(self):
        def fn(x):
            y = x + 1
            items = [100]
            for i in range(3):
                torch._dynamo.graph_break()
                items.append(i)
            z = x + len(items)
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 22: modifying global variable
    def test_for_loop_modifying_global_variable(self):
        def fn(x):
            global _modify_global
            y = x + 1
            _modify_global = 10
            for i in range(3):
                torch._dynamo.graph_break()
                _modify_global += 1
            z = x + _modify_global
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 23: modifying closure variable (local counter pattern)
    def test_for_loop_modifying_closure_variable(self):
        def fn(x):
            y = x + 1
            counter = 0
            for i in range(3):
                torch._dynamo.graph_break()
                counter += 1
            z = x + counter
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 24: nested for loops, only outer has graph break
    def test_nested_for_loops(self):
        def fn(x):
            y = x + 1
            total = 0
            for i in range(3):
                torch._dynamo.graph_break()
                for j in range(2):
                    total += i * j
            z = x + total
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 25: triple nested for loops
    def test_triple_nested_for_loops(self):
        def fn(x):
            y = x + 1
            total = 0
            for i in range(2):
                torch._dynamo.graph_break()
                for j in range(2):
                    for k in range(2):
                        total += i + j + k
            z = x + total
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 26: for loop containing a list comprehension
    def test_for_loop_containing_comprehension(self):
        def fn(x):
            y = x + 1
            all_items = []
            for i in range(3):
                torch._dynamo.graph_break()
                items = [j * i for j in range(3)]
                all_items.extend(items)
            z = x + sum(all_items)
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 27: comprehension calling a function with a for loop graph break
    @skipIfNotPy312
    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_comprehension_containing_for_loop_call(self):
        def inner(x, n):
            total = 0
            for i in range(n):
                torch._dynamo.graph_break()
                total += i
            return total + x

        def fn(x):
            y = x + 1
            results = [inner(0, i) for i in range(4)]
            z = x + sum(results)
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertGreaterEqual(len(backend.graphs), 2)

    # Test 28: function calls within for loop with graph break
    def test_nested_function_calls_with_for_loop_graph_break(self):
        def fn(x):
            def helper(v):
                return v * 2

            y = x + 1
            total = 0
            for i in range(3):
                torch._dynamo.graph_break()
                total += helper(i)
            z = x + total
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 29: tensor operations inside for loop with graph break
    def test_for_loop_with_tensor_operations(self):
        def fn(x):
            y = x + 1
            for i in range(3):
                torch._dynamo.graph_break()
                y = y + 1
            z = y + 2
            return z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    # Test 30: while loop still skips frame (no for-loop optimization)
    def test_while_loop_still_skips_frame(self):
        def fn(x):
            y = x + 1
            i = 0
            while i < 3:
                torch._dynamo.graph_break()
                i += 1
            z = x + 2
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 0)

    def test_for_loop_tuple_unpacking(self):
        def fn(x):
            y = x + 1
            total = 0
            for a, b in [(1, 2), (3, 4), (5, 6)]:
                torch._dynamo.graph_break()
                total += a + b
            z = x + total
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)

    def test_for_loop_empty_iterator(self):
        def fn(x):
            y = x + 1
            total = 0
            for i in range(0):
                torch._dynamo.graph_break()
                total += i
            z = x + 2 + total
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        # Empty iterator: loop body never executes, no graph break, 1 graph
        self.assertEqual(len(backend.graphs), 1)

    def test_nested_for_loops_both_graph_break(self):
        def fn(x):
            y = x + 1
            total = 0
            for i in range(3):
                torch._dynamo.graph_break()
                for j in range(2):
                    torch._dynamo.graph_break()
                    total += i * j
            z = x + total
            return y, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)


class NestedGraphBreakForLoopTests(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_graph_break(self):
        def inner(x):
            x = x + 2
            for i in range(3):
                torch._dynamo.graph_break()
            x = x + 3
            return x

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
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_with_graph_break_function(self):
        def leaf(v):
            if v == 2:
                torch._dynamo.graph_break()
            return v

        def inner(x):
            x = x + 2
            total = 0
            for i in range(4):
                total += leaf(i)
            x = x + 3 + total
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_multiple_for_loops_one_graph_break(self):
        def inner(x):
            x = x + 2
            for i in range(3):
                torch._dynamo.graph_break()
            total = 0
            for i in range(3):
                total += i
            x = x + 3 + total
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_break(self):
        def inner(x):
            x = x + 2
            result = 0
            for i in range(5):
                torch._dynamo.graph_break()
                if i == 3:
                    break
                result += i
            x = x + result
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_else(self):
        def inner(x):
            x = x + 2
            found = False
            for i in range(3):
                torch._dynamo.graph_break()
            else:
                found = True
            x = x + 3
            return x, found

        def outer(x):
            x = x + 1
            x, found = inner(x)
            x = x + 4
            return x, found

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        result = compiled(x)
        expected = outer(x)
        self.assertEqual(result, expected)
        self.assertTrue(result[1])
        self.assertEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_break_skips_else(self):
        def inner(x):
            x = x + 2
            found = False
            for i in range(3):
                torch._dynamo.graph_break()
                if i == 1:
                    break
            else:
                found = True
            x = x + 3
            return x, found

        def outer(x):
            x = x + 1
            x, found = inner(x)
            x = x + 4
            return x, found

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        result = compiled(x)
        expected = outer(x)
        self.assertEqual(result, expected)
        self.assertFalse(result[1])
        self.assertEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_modifies_local(self):
        def inner(x):
            x = x + 2
            a = 0
            for i in range(3):
                torch._dynamo.graph_break()
                a += i
            x = x + a
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_with_outer_variable_read(self):
        def inner(x):
            x = x + 2
            factor = 3
            total = 0
            for i in range(4):
                torch._dynamo.graph_break()
                total += i * factor
            x = x + total
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_with_outer_list_mutation(self):
        def inner(x):
            x = x + 2
            items = []
            for i in range(3):
                torch._dynamo.graph_break()
                items.append(i)
            x = x + sum(items)
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_with_outer_dict_mutation(self):
        def inner(x):
            x = x + 2
            d = {}
            for i in range(3):
                torch._dynamo.graph_break()
                d[i] = i * 2
            x = x + sum(d.values())
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_with_outer_list_extend(self):
        def inner(x):
            x = x + 2
            items = [10]
            for i in range(3):
                torch._dynamo.graph_break()
                items.extend([i, i * 2])
            x = x + sum(items)
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_with_outer_list_pop(self):
        def inner(x):
            x = x + 2
            items = [10, 20, 30]
            popped = []
            for i in range(2):
                torch._dynamo.graph_break()
                popped.append(items.pop())
            x = x + sum(popped)
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_with_global_variable(self):
        def inner(x):
            global _global_counter
            x = x + 2
            _global_counter = 0
            for i in range(3):
                torch._dynamo.graph_break()
                _global_counter += i
            x = x + _global_counter
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_with_closure_variable(self):
        def inner(x):
            x = x + 2
            captured = 5
            total = 0
            for i in range(3):
                torch._dynamo.graph_break()
                total += captured
            x = x + total
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_with_closure_list_mutation(self):
        def inner(x):
            x = x + 2
            items = [100]
            for i in range(3):
                torch._dynamo.graph_break()
                items.append(i)
            x = x + len(items)
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_modifying_global_variable(self):
        def inner(x):
            global _modify_global
            x = x + 2
            _modify_global = 10
            for i in range(3):
                torch._dynamo.graph_break()
                _modify_global += 1
            x = x + _modify_global
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_modifying_closure_variable(self):
        def inner(x):
            x = x + 2
            counter = 0
            for i in range(3):
                torch._dynamo.graph_break()
                counter += 1
            x = x + counter
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_nested_for_loops(self):
        def inner(x):
            x = x + 2
            total = 0
            for i in range(3):
                torch._dynamo.graph_break()
                for j in range(2):
                    total += i * j
            x = x + total
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_nonlocal_int_mutation(self):
        def outer(x):
            x = x + 1
            counter = 0

            def inner():
                nonlocal counter
                for i in range(3):
                    torch._dynamo.graph_break()
                    counter += 1

            inner()
            x = x + counter
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_nonlocal_tensor_mutation(self):
        def outer(x):
            y = x + 1
            t = torch.zeros(4)

            def inner():
                nonlocal t
                for i in range(3):
                    torch._dynamo.graph_break()
                    t = t + 1

            inner()
            z = y + t
            return z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_nonlocal_different_varnames(self):
        def outer(x):
            x = x + 1
            alpha = 0
            beta = 0

            def inner():
                nonlocal alpha, beta
                for i in range(3):
                    torch._dynamo.graph_break()
                    alpha += i
                    beta += i * 2

            inner()
            x = x + alpha + beta
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_nonlocal_tensor_different_varnames(self):
        def outer(x):
            y = x + 1
            a = torch.zeros(4)
            b = torch.ones(4)

            def inner():
                nonlocal a, b
                for i in range(3):
                    torch._dynamo.graph_break()
                    a = a + 1
                    b = b + 2

            inner()
            z = y + a + b
            return z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_captured_tensor(self):
        def outer(x):
            y = x + 1
            t = torch.ones(4) * 5

            def inner():
                total = torch.zeros(4)
                for i in range(3):
                    torch._dynamo.graph_break()
                    total = total + t
                return total

            result = inner()
            z = y + result
            return z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_name_shadowing(self):
        def inner(t):
            x = t + 2
            total = 0
            for i in range(3):
                torch._dynamo.graph_break()
                total += i
            x = x + total
            return x

        def outer(t):
            x = t + 1
            result = inner(t) + x
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_result_var_name_collision(self):
        def inner(t):
            x = [0]
            for i in range(3):
                torch._dynamo.graph_break()
                x[0] += i
            return x[0] + t

        def root(t):
            x = t + 1
            result = inner(t) + x
            self.assertEqual(x, t + 1)
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(root, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), root(x))
        self.assertEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_triple_nested_for_loops(self):
        def inner(x):
            x = x + 2
            total = 0
            for i in range(2):
                torch._dynamo.graph_break()
                for j in range(2):
                    for k in range(2):
                        total += i + j + k
            x = x + total
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_multiple_for_loops_in_different_functions(self):
        def inner1(x):
            x = x + 2
            for i in range(3):
                torch._dynamo.graph_break()
            x = x + 3
            return x

        def inner2(x):
            x = x + 5
            for i in range(3):
                torch._dynamo.graph_break()
            x = x + 6
            return x

        def outer(x):
            x = x + 1
            x = inner1(x)
            x = inner2(x)
            x = x + 4
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), outer(x))
        self.assertGreaterEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_function_for_loop_with_outer_list_mutation(self):
        def outer(x):
            y = x + 1
            items = []

            def inner():
                for i in range(3):
                    torch._dynamo.graph_break()
                    items.append(i)

            inner()
            z = y + sum(items)
            return z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_function_calls_with_for_loop_graph_break(self):
        def h(x):
            x = x + 3
            for i in range(3):
                torch._dynamo.graph_break()
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
    def test_nested_multiple_for_loop_graph_breaks(self):
        def inner(x):
            x = x + 2
            for i in range(3):
                torch._dynamo.graph_break()
            for j in range(3):
                torch._dynamo.graph_break()
            x = x + 3
            return x

        def outer(x):
            x = x + 1
            x = inner(x)
            x = x + 4
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), outer(x))
        self.assertGreaterEqual(len(backend.graphs), 2)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_tuple_unpacking(self):
        def inner(x):
            x = x + 2
            total = 0
            for a, b in [(1, 2), (3, 4), (5, 6)]:
                torch._dynamo.graph_break()
                total += a + b
            x = x + total
            return x

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

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_for_loop_empty_iterator(self):
        def inner(x):
            x = x + 2
            total = 0
            for i in range(0):
                torch._dynamo.graph_break()
                total += i
            x = x + 3 + total
            return x

        def outer(x):
            x = x + 1
            x = inner(x)
            x = x + 4
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)
        self.assertEqual(compiled(x), outer(x))
        # Empty iterator: loop body never runs, no graph break, 1 graph
        self.assertEqual(len(backend.graphs), 1)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_both_loops_graph_break(self):
        def inner(x):
            x = x + 2
            total = 0
            for i in range(3):
                torch._dynamo.graph_break()
                for j in range(2):
                    torch._dynamo.graph_break()
                    total += i * j
            x = x + total
            return x

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
