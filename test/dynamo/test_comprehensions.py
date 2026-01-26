# Owner(s): ["module: dynamo"]

"""
Tests for comprehension graph break handling in Python 3.12+.

In Python 3.12+, comprehensions are inlined into the parent function's bytecode
rather than being compiled as separate code objects. These tests verify that
when a graph break occurs inside a comprehension, only the comprehension is
skipped (not the entire frame), resulting in multiple graphs being created.
"""

import torch
import torch._dynamo.testing
import torch._inductor.test_case
from torch._dynamo.testing import CompileCounter, skipIfNotPy312


# Global variable for testing global variable access in comprehensions
_GLOBAL_VALUE_FOR_COMPREHENSION_TEST = 100


class ComprehensionTests(torch._inductor.test_case.TestCase):
    @skipIfNotPy312
    def test_list_comprehension_graph_break(self):
        """Test that list comprehension with graph break creates 2 graphs."""

        def fn(x):
            y = x + 1
            result = [print(f"Item: {i}") or i for i in range(3)]
            z = x + 2
            return y, result, z

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        y, result, z = compiled(x)

        self.assertEqual(y, torch.tensor([2.0]))
        self.assertEqual(result, [0, 1, 2])
        self.assertEqual(z, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_dict_comprehension_graph_break(self):
        """Test that dict comprehension with graph break creates 2 graphs."""

        def fn(x):
            y = x + 1
            result = {i: print(f"Key: {i}") or i**2 for i in range(3)}
            z = x + 2
            return y, result, z

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        y, result, z = compiled(x)

        self.assertEqual(y, torch.tensor([2.0]))
        self.assertEqual(result, {0: 0, 1: 1, 2: 4})
        self.assertEqual(z, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
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
            return z

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        inp = torch.tensor([3.0])
        result = compiled(inp)

        self.assertEqual(result, torch.tensor([13.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_simple_list_comprehension_no_break(self):
        """Test that simple list comprehension without graph break creates 1 graph."""

        def fn(x):
            result = [i * 2 for i in range(3)]
            return x + sum(result)

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        result = compiled(x)

        self.assertEqual(result, torch.tensor([7.0]))
        self.assertEqual(cnt.frame_count, 1)

    @skipIfNotPy312
    def test_multiple_comprehensions_one_break(self):
        """Test function with multiple comprehensions where only one has graph break."""

        def fn(x):
            a = x + 1
            list1 = [i for i in range(2)]
            list2 = [print(f"Mid: {i}") or i for i in range(2)]
            b = x + 2
            list3 = [i * 2 for i in range(2)]
            return a, list1, list2, b, list3

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, list1, list2, b, list3 = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(list1, [0, 1])
        self.assertEqual(list2, [0, 1])
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(list3, [0, 2])
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_nested_comprehension_inner_break(self):
        """Test nested comprehension where inner comprehension causes graph break."""

        def fn(x):
            a = x + 1
            result = [[print(f"{i},{j}") or i * j for j in range(2)] for i in range(2)]
            b = x + 2
            return a, result, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [[0, 0], [0, 1]])
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_multi_iterator_comprehension_break(self):
        """Test comprehension with multiple iterators where graph break occurs."""

        def fn(x):
            a = x + 1
            result = [(print(f"{i},{j}") or i, j) for i in range(2) for j in range(2)]
            b = x + 2
            return a, result, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_discarded_comprehension_graph_break(self):
        """Test that discarded comprehension (result not assigned) with graph break works."""

        def fn(x):
            a = x + 1
            [print(f"item: {i}") for i in range(3)]  # Result discarded
            b = x + 2
            return a, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_comprehension_in_expression_graph_break(self):
        """Test comprehension used in expression (e.g., sum([...])) with graph break."""

        def fn(x):
            a = x + 1
            total = sum([print(f"item: {i}") or i for i in range(3)])
            b = x + 2
            return a, total, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, total, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(total, 3)  # 0 + 1 + 2 = 3
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_comprehension_return_directly(self):
        """Test comprehension returned directly with graph break."""

        def fn(x):
            a = x + 1  # noqa: F841
            return [print(f"item: {i}") or i for i in range(3)]

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        result = compiled(x)

        self.assertEqual(result, [0, 1, 2])
        self.assertGreaterEqual(cnt.frame_count, 1)

    @skipIfNotPy312
    def test_walrus_operator_in_comprehension(self):
        """Test walrus operator (:=) inside comprehension with graph break."""

        def fn(x):
            a = x + 1
            result = [y := (print(f"item: {i}") or i * 2) for i in range(3)]
            b = x + 2
            return a, result, y, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, y, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [0, 2, 4])
        self.assertEqual(y, 4)  # Last value: 2 * 2 = 4
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_walrus_operator_in_if_in_comprehension(self):
        """Test walrus operator (:=) in if clause of comprehension with graph break."""

        def fn(x):
            a = x + 1
            result = [
                (print(f"item: {i}") or y) for i in range(5) if (y := i * 2) > 2
            ]
            b = x + 2
            return a, result, y, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, y, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [4, 6, 8])
        self.assertEqual(y, 8)
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_walrus_operator_in_comprehension_with_tensor(self):
        """Test walrus operator with tensor operation in comprehension with graph break."""

        def fn(x):
            a = x + 1
            result = [
                (print(f"item: {i}") or y + x.numel())
                for i in range(5)
                if (y := i * 2) > 2
            ]
            b = x + 2
            return a, result, y, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, y, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [5, 7, 9])
        self.assertEqual(y, 8)
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_multiple_walrus_operators_in_comprehension(self):
        """Test multiple walrus operators (:=) inside comprehension with graph break."""

        def fn(x):
            a = x + 1
            result = [
                (y := (print(f"item: {i}") or i * 2), z := i * 3) for i in range(3)
            ]
            b = x + 2
            return a, result, y, z, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, y, z, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [(0, 0), (2, 3), (4, 6)])
        self.assertEqual(y, 4)  # Last value: 2 * 2 = 4
        self.assertEqual(z, 6)  # Last value: 2 * 3 = 6
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_comprehension_with_outer_variable_read(self):
        """Test accessing outer variable inside comprehension with graph break."""

        def fn(x):
            outer_val = 10
            a = x + 1
            result = [
                i + outer_val for i in range(3) if print(f"item: {i}") or True
            ]
            b = x + 2
            return a, result, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [10, 11, 12])  # 0+10, 1+10, 2+10
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_comprehension_with_outer_list_mutation(self):
        """Test mutating outer list inside comprehension with graph break."""

        def fn(x):
            outer_list = []
            a = x + 1
            result = [
                outer_list.append(i) or i * 2
                for i in range(3)
                if print(f"item: {i}") or True
            ]
            b = x + 2
            return a, result, outer_list, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, outer_list, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [0, 2, 4])  # 0*2, 1*2, 2*2
        self.assertEqual(outer_list, [0, 1, 2])  # Side effect: appended values
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_comprehension_with_outer_dict_mutation(self):
        """Test mutating outer dict inside comprehension with graph break."""

        def fn(x):
            outer_dict = {}
            a = x + 1
            result = [
                outer_dict.update({i: i * 10}) or i * 2
                for i in range(3)
                if print(f"item: {i}") or True
            ]
            b = x + 2
            return a, result, outer_dict, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, outer_dict, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [0, 2, 4])
        self.assertEqual(outer_dict, {0: 0, 1: 10, 2: 20})
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_comprehension_with_outer_list_extend(self):
        """Test extending outer list inside comprehension with graph break."""

        def fn(x):
            outer_list = [100]
            a = x + 1
            result = [
                outer_list.extend([i, i * 10]) or i
                for i in range(3)
                if print(f"item: {i}") or True
            ]
            b = x + 2
            return a, result, outer_list, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, outer_list, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [0, 1, 2])
        self.assertEqual(outer_list, [100, 0, 0, 1, 10, 2, 20])
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_comprehension_with_outer_list_pop(self):
        """Test popping from outer list inside comprehension with graph break."""

        def fn(x):
            outer_list = [10, 20, 30, 40, 50]
            popped_values = []
            a = x + 1
            result = [
                popped_values.append(outer_list.pop()) or i
                for i in range(3)
                if print(f"item: {i}") or True
            ]
            b = x + 2
            return a, result, outer_list, popped_values, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, outer_list, popped_values, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [0, 1, 2])
        self.assertEqual(outer_list, [10, 20])  # 3 items popped
        self.assertEqual(popped_values, [50, 40, 30])  # Popped in reverse order
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_comprehension_with_global_variable(self):
        """Test global variable access inside comprehension with graph break."""

        def fn(x):
            a = x + 1
            result = [
                i + _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
                for i in range(3)
                if print(f"item: {i}") or True
            ]
            b = x + 2
            return a, result, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [100, 101, 102])  # 0+100, 1+100, 2+100
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_comprehension_with_closure_variable(self):
        """Test closure variable access inside comprehension with graph break."""

        def make_fn():
            closure_val = 50

            def fn(x):
                a = x + 1
                result = [
                    i + closure_val for i in range(3) if print(f"item: {i}") or True
                ]
                b = x + 2
                return a, result, b

            return fn

        fn = make_fn()
        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [50, 51, 52])  # 0+50, 1+50, 2+50
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_comprehension_with_closure_list_mutation(self):
        """Test closure list mutation inside comprehension with graph break."""

        def make_fn():
            closure_list = []

            def fn(x):
                a = x + 1
                result = [
                    closure_list.append(i) or i * 2
                    for i in range(3)
                    if print(f"item: {i}") or True
                ]
                b = x + 2
                return a, result, closure_list, b

            return fn

        fn = make_fn()
        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, closure_list, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        self.assertEqual(result, [0, 2, 4])
        self.assertEqual(closure_list, [0, 1, 2])
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfNotPy312
    def test_nested_multi_for_comprehension_graph_break(self):
        """Test nested comprehension with multiple for loops and graph break."""

        def fn(x):
            a = x + 1
            result = [
                [(print(f"{i},{j},{k}") or i + j + k) for k in range(2)]
                for i in range(2)
                for j in range(2)
            ]
            b = x + 2
            return a, result, b

        cnt = CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        x = torch.tensor([1.0])
        a, result, b = compiled(x)

        self.assertEqual(a, torch.tensor([2.0]))
        expected = [[0, 1], [1, 2], [1, 2], [2, 3]]
        self.assertEqual(result, expected)
        self.assertEqual(b, torch.tensor([3.0]))
        self.assertEqual(cnt.frame_count, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
