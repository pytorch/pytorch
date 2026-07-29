# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_named_expressions.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import unittest

GLOBAL_VAR = None

class NamedExpressionInvalidTest(CPythonTestCase):

    def test_named_expression_invalid_01(self):
        code = """x := 0"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_02(self):
        code = """x = y := 0"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_03(self):
        code = """y := f(x)"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_04(self):
        code = """y0 = y1 := f(x)"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_06(self):
        code = """((a, b) := (1, 2))"""

        with self.assertRaisesRegex(SyntaxError, "cannot use assignment expressions with tuple"):
            exec(code, {}, {})

    def test_named_expression_invalid_07(self):
        code = """def spam(a = b := 42): pass"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_08(self):
        code = """def spam(a: b := 42 = 5): pass"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_09(self):
        code = """spam(a=b := 'c')"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_10(self):
        code = """spam(x = y := f(x))"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_11(self):
        code = """spam(a=1, b := 2)"""

        with self.assertRaisesRegex(SyntaxError,
            "positional argument follows keyword argument"):
            exec(code, {}, {})

    def test_named_expression_invalid_12(self):
        code = """spam(a=1, (b := 2))"""

        with self.assertRaisesRegex(SyntaxError,
            "positional argument follows keyword argument"):
            exec(code, {}, {})

    def test_named_expression_invalid_13(self):
        code = """spam(a=1, (b := 2))"""

        with self.assertRaisesRegex(SyntaxError,
            "positional argument follows keyword argument"):
            exec(code, {}, {})

    def test_named_expression_invalid_14(self):
        code = """(x := lambda: y := 1)"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_15(self):
        code = """(lambda: x := 1)"""

        with self.assertRaisesRegex(SyntaxError,
            "cannot use assignment expressions with lambda"):
            exec(code, {}, {})

    def test_named_expression_invalid_16(self):
        code = "[i + 1 for i in i := [1,2]]"

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_17(self):
        code = "[i := 0, j := 1 for i, j in [(1, 2), (3, 4)]]"

        with self.assertRaisesRegex(SyntaxError,
                "did you forget parentheses around the comprehension target?"):
            exec(code, {}, {})

    def test_named_expression_invalid_in_class_body(self):
        code = """class Foo():
            [(42, 1 + ((( j := i )))) for i in range(5)]
        """

        with self.assertRaisesRegex(SyntaxError,
            "assignment expression within a comprehension cannot be used in a class body"):
            exec(code, {}, {})

    def test_named_expression_valid_rebinding_iteration_variable(self):
        # This test covers that we can reassign variables
        # that are not directly assigned in the
        # iterable part of a comprehension.
        cases = [
            # Regression tests from https://github.com/python/cpython/issues/87447
            ("Complex expression: c",
                "{0}(c := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: d",
                "{0}(d := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: e",
                "{0}(e := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: f",
                "{0}(f := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: g",
                "{0}(g := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: h",
                "{0}(h := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: i",
                "{0}(i := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: j",
                "{0}(j := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
        ]
        for test_case, code in cases:
            for lpar, rpar in [('(', ')'), ('[', ']'), ('{', '}')]:
                code = code.format(lpar, rpar)
                with self.subTest(case=test_case, lpar=lpar, rpar=rpar):
                    # Names used in snippets are not defined,
                    # but we are fine with it: just must not be a SyntaxError.
                    # Names used in snippets are not defined,
                    # but we are fine with it: just must not be a SyntaxError.
                    with self.assertRaises(NameError):
                        exec(code, {}) # Module scope
                    with self.assertRaises(NameError):
                        exec(code, {}, {}) # Class scope
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_iteration_variable(self):
        # This test covers that we cannot reassign variables
        # that are directly assigned in the iterable part of a comprehension.
        cases = [
            # Regression tests from https://github.com/python/cpython/issues/87447
            ("Complex expression: a", "a",
                "{0}(a := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: b", "b",
                "{0}(b := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
        ]
        for test_case, target, code in cases:
            msg = f"assignment expression cannot rebind comprehension iteration variable '{target}'"
            for lpar, rpar in [('(', ')'), ('[', ']'), ('{', '}')]:
                code = code.format(lpar, rpar)
                with self.subTest(case=test_case, lpar=lpar, rpar=rpar):
                    # Names used in snippets are not defined,
                    # but we are fine with it: just must not be a SyntaxError.
                    # Names used in snippets are not defined,
                    # but we are fine with it: just must not be a SyntaxError.
                    with self.assertRaisesRegex(SyntaxError, msg):
                        exec(code, {}) # Module scope
                    with self.assertRaisesRegex(SyntaxError, msg):
                        exec(code, {}, {}) # Class scope
                    with self.assertRaisesRegex(SyntaxError, msg):
                        exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_list_comprehension_iteration_variable(self):
        cases = [
            ("Local reuse", 'i', "[i := 0 for i in range(5)]"),
            ("Nested reuse", 'j', "[[(j := 0) for i in range(5)] for j in range(5)]"),
            ("Reuse inner loop target", 'j', "[(j := 0) for i in range(5) for j in range(5)]"),
            ("Unpacking reuse", 'i', "[i := 0 for i, j in [(0, 1)]]"),
            ("Reuse in loop condition", 'i', "[i+1 for i in range(5) if (i := 0)]"),
            ("Unreachable reuse", 'i', "[False or (i:=0) for i in range(5)]"),
            ("Unreachable nested reuse", 'i',
                "[(i, j) for i in range(5) for j in range(5) if True or (i:=10)]"),
        ]
        for case, target, code in cases:
            msg = f"assignment expression cannot rebind comprehension iteration variable '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_list_comprehension_inner_loop(self):
        cases = [
            ("Inner reuse", 'j', "[i for i in range(5) if (j := 0) for j in range(5)]"),
            ("Inner unpacking reuse", 'j', "[i for i in range(5) if (j := 0) for j, k in [(0, 1)]]"),
        ]
        for case, target, code in cases:
            msg = f"comprehension inner loop cannot rebind assignment expression target '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_list_comprehension_iterable_expression(self):
        cases = [
            ("Top level", "[i for i in (i := range(5))]"),
            ("Inside tuple", "[i for i in (2, 3, i := range(5))]"),
            ("Inside list", "[i for i in [2, 3, i := range(5)]]"),
            ("Different name", "[i for i in (j := range(5))]"),
            ("Lambda expression", "[i for i in (lambda:(j := range(5)))()]"),
            ("Inner loop", "[i for i in range(5) for j in (i := range(5))]"),
            ("Nested comprehension", "[i for i in [j for j in (k := range(5))]]"),
            ("Nested comprehension condition", "[i for i in [j for j in range(5) if (j := True)]]"),
            ("Nested comprehension body", "[i for i in [(j := True) for j in range(5)]]"),
        ]
        msg = "assignment expression cannot be used in a comprehension iterable expression"
        for case, code in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_set_comprehension_iteration_variable(self):
        cases = [
            ("Local reuse", 'i', "{i := 0 for i in range(5)}"),
            ("Nested reuse", 'j', "{{(j := 0) for i in range(5)} for j in range(5)}"),
            ("Reuse inner loop target", 'j', "{(j := 0) for i in range(5) for j in range(5)}"),
            ("Unpacking reuse", 'i', "{i := 0 for i, j in {(0, 1)}}"),
            ("Reuse in loop condition", 'i', "{i+1 for i in range(5) if (i := 0)}"),
            ("Unreachable reuse", 'i', "{False or (i:=0) for i in range(5)}"),
            ("Unreachable nested reuse", 'i',
                "{(i, j) for i in range(5) for j in range(5) if True or (i:=10)}"),
            # Regression tests from https://github.com/python/cpython/issues/87447
            ("Complex expression: a", "a",
                "{(a := 1) for a, (*b, c[d+e::f(g)], h.i) in j}"),
            ("Complex expression: b", "b",
                "{(b := 1) for a, (*b, c[d+e::f(g)], h.i) in j}"),
        ]
        for case, target, code in cases:
            msg = f"assignment expression cannot rebind comprehension iteration variable '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_set_comprehension_inner_loop(self):
        cases = [
            ("Inner reuse", 'j', "{i for i in range(5) if (j := 0) for j in range(5)}"),
            ("Inner unpacking reuse", 'j', "{i for i in range(5) if (j := 0) for j, k in {(0, 1)}}"),
        ]
        for case, target, code in cases:
            msg = f"comprehension inner loop cannot rebind assignment expression target '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_set_comprehension_iterable_expression(self):
        cases = [
            ("Top level", "{i for i in (i := range(5))}"),
            ("Inside tuple", "{i for i in (2, 3, i := range(5))}"),
            ("Inside list", "{i for i in {2, 3, i := range(5)}}"),
            ("Different name", "{i for i in (j := range(5))}"),
            ("Lambda expression", "{i for i in (lambda:(j := range(5)))()}"),
            ("Inner loop", "{i for i in range(5) for j in (i := range(5))}"),
            ("Nested comprehension", "{i for i in {j for j in (k := range(5))}}"),
            ("Nested comprehension condition", "{i for i in {j for j in range(5) if (j := True)}}"),
            ("Nested comprehension body", "{i for i in {(j := True) for j in range(5)}}"),
        ]
        msg = "assignment expression cannot be used in a comprehension iterable expression"
        for case, code in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_dict_comprehension_iteration_variable(self):
        cases = [
            ("Key reuse", 'i', "{(i := 0): 1 for i in range(5)}"),
            ("Value reuse", 'i', "{1: (i := 0) for i in range(5)}"),
            ("Both reuse", 'i', "{(i := 0): (i := 0) for i in range(5)}"),
            ("Nested reuse", 'j', "{{(j := 0): 1 for i in range(5)} for j in range(5)}"),
            ("Reuse inner loop target", 'j', "{(j := 0): 1 for i in range(5) for j in range(5)}"),
            ("Unpacking key reuse", 'i', "{(i := 0): 1 for i, j in {(0, 1)}}"),
            ("Unpacking value reuse", 'i', "{1: (i := 0) for i, j in {(0, 1)}}"),
            ("Reuse in loop condition", 'i', "{i+1: 1 for i in range(5) if (i := 0)}"),
            ("Unreachable reuse", 'i', "{(False or (i:=0)): 1 for i in range(5)}"),
            ("Unreachable nested reuse", 'i',
                "{i: j for i in range(5) for j in range(5) if True or (i:=10)}"),
            # Regression tests from https://github.com/python/cpython/issues/87447
            ("Complex expression: a", "a",
                "{(a := 1): 1 for a, (*b, c[d+e::f(g)], h.i) in j}"),
            ("Complex expression: b", "b",
                "{(b := 1): 1 for a, (*b, c[d+e::f(g)], h.i) in j}"),
        ]
        for case, target, code in cases:
            msg = f"assignment expression cannot rebind comprehension iteration variable '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_dict_comprehension_inner_loop(self):
        cases = [
            ("Inner reuse", 'j', "{i: 1 for i in range(5) if (j := 0) for j in range(5)}"),
            ("Inner unpacking reuse", 'j', "{i: 1 for i in range(5) if (j := 0) for j, k in {(0, 1)}}"),
        ]
        for case, target, code in cases:
            msg = f"comprehension inner loop cannot rebind assignment expression target '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_dict_comprehension_iterable_expression(self):
        cases = [
            ("Top level", "{i: 1 for i in (i := range(5))}"),
            ("Inside tuple", "{i: 1 for i in (2, 3, i := range(5))}"),
            ("Inside list", "{i: 1 for i in [2, 3, i := range(5)]}"),
            ("Different name", "{i: 1 for i in (j := range(5))}"),
            ("Lambda expression", "{i: 1 for i in (lambda:(j := range(5)))()}"),
            ("Inner loop", "{i: 1 for i in range(5) for j in (i := range(5))}"),
            ("Nested comprehension", "{i: 1 for i in {j: 2 for j in (k := range(5))}}"),
            ("Nested comprehension condition", "{i: 1 for i in {j: 2 for j in range(5) if (j := True)}}"),
            ("Nested comprehension body", "{i: 1 for i in {(j := True) for j in range(5)}}"),
        ]
        msg = "assignment expression cannot be used in a comprehension iterable expression"
        for case, code in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_mangled_class_variables(self):
        code = """class Foo:
            def bar(self):
                [[(__x:=2) for _ in range(2)] for __x in range(2)]
        """

        with self.assertRaisesRegex(SyntaxError,
            "assignment expression cannot rebind comprehension iteration variable '__x'"):
            exec(code, {}, {})


class NamedExpressionAssignmentTest(CPythonTestCase):

    def test_named_expression_assignment_01(self):
        (a := 10)

        self.assertEqual(a, 10)

    def test_named_expression_assignment_02(self):
        a = 20
        (a := a)

        self.assertEqual(a, 20)

    def test_named_expression_assignment_03(self):
        (total := 1 + 2)

        self.assertEqual(total, 3)

    def test_named_expression_assignment_04(self):
        (info := (1, 2, 3))

        self.assertEqual(info, (1, 2, 3))

    def test_named_expression_assignment_05(self):
        (x := 1, 2)

        self.assertEqual(x, 1)

    def test_named_expression_assignment_06(self):
        (z := (y := (x := 0)))

        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_named_expression_assignment_07(self):
        (loc := (1, 2))

        self.assertEqual(loc, (1, 2))

    def test_named_expression_assignment_08(self):
        if spam := "eggs":
            self.assertEqual(spam, "eggs")
        else: self.fail("variable was not assigned using named expression")

    def test_named_expression_assignment_09(self):
        if True and (spam := True):
            self.assertTrue(spam)
        else: self.fail("variable was not assigned using named expression")

    def test_named_expression_assignment_10(self):
        if (match := 10) == 10:
            self.assertEqual(match, 10)
        else: self.fail("variable was not assigned using named expression")

    def test_named_expression_assignment_11(self):
        def spam(a):
            return a
        input_data = [1, 2, 3]
        res = [(x, y, x/y) for x in input_data if (y := spam(x)) > 0]

        self.assertEqual(res, [(1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)])

    def test_named_expression_assignment_12(self):
        def spam(a):
            return a
        res = [[y := spam(x), x/y] for x in range(1, 5)]

        self.assertEqual(res, [[1, 1.0], [2, 1.0], [3, 1.0], [4, 1.0]])

    def test_named_expression_assignment_13(self):
        length = len(lines := [1, 2])

        self.assertEqual(length, 2)
        self.assertEqual(lines, [1,2])

    def test_named_expression_assignment_14(self):
        """
        Where all variables are positive integers, and a is at least as large
        as the n'th root of x, this algorithm returns the floor of the n'th
        root of x (and roughly doubling the number of accurate bits per
        iteration):
        """
        a = 9
        n = 2
        x = 3

        while a > (d := x // a**(n-1)):
            a = ((n-1)*a + d) // n

        self.assertEqual(a, 1)

    def test_named_expression_assignment_15(self):
        while a := False:
            self.fail("While body executed")  # This will not run

        self.assertEqual(a, False)

    def test_named_expression_assignment_16(self):
        a, b = 1, 2
        fib = {(c := a): (a := b) + (b := a + c) - b for __ in range(6)}
        self.assertEqual(fib, {1: 2, 2: 3, 3: 5, 5: 8, 8: 13, 13: 21})

    def test_named_expression_assignment_17(self):
        a = [1]
        element = a[b:=0]
        self.assertEqual(b, 0)
        self.assertEqual(element, a[0])

    def test_named_expression_assignment_18(self):
        class TwoDimensionalList:
            def __init__(self, two_dimensional_list):
                self.two_dimensional_list = two_dimensional_list

            def __getitem__(self, index):
                return self.two_dimensional_list[index[0]][index[1]]

        a = TwoDimensionalList([[1], [2]])
        element = a[b:=0, c:=0]
        self.assertEqual(b, 0)
        self.assertEqual(c, 0)
        self.assertEqual(element, a.two_dimensional_list[b][c])



class NamedExpressionScopeTest(CPythonTestCase):

    def test_named_expression_scope_01(self):
        code = """def spam():
    (a := 5)
print(a)"""

        with self.assertRaisesRegex(NameError, "name 'a' is not defined"):
            exec(code, {}, {})

    def test_named_expression_scope_02(self):
        total = 0
        partial_sums = [total := total + v for v in range(5)]

        self.assertEqual(partial_sums, [0, 1, 3, 6, 10])
        self.assertEqual(total, 10)

    def test_named_expression_scope_03(self):
        containsOne = any((lastNum := num) == 1 for num in [1, 2, 3])

        self.assertTrue(containsOne)
        self.assertEqual(lastNum, 1)

    def test_named_expression_scope_04(self):
        def spam(a):
            return a
        res = [[y := spam(x), x/y] for x in range(1, 5)]

        self.assertEqual(y, 4)

    def test_named_expression_scope_05(self):
        def spam(a):
            return a
        input_data = [1, 2, 3]
        res = [(x, y, x/y) for x in input_data if (y := spam(x)) > 0]

        self.assertEqual(res, [(1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)])
        self.assertEqual(y, 3)

    def test_named_expression_scope_06(self):
        res = [[spam := i for i in range(3)] for j in range(2)]

        self.assertEqual(res, [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(spam, 2)

    def test_named_expression_scope_07(self):
        len(lines := [1, 2])

        self.assertEqual(lines, [1, 2])

    def test_named_expression_scope_08(self):
        def spam(a):
            return a

        def eggs(b):
            return b * 2

        res = [spam(a := eggs(b := h)) for h in range(2)]

        self.assertEqual(res, [0, 2])
        self.assertEqual(a, 2)
        self.assertEqual(b, 1)

    def test_named_expression_scope_09(self):
        def spam(a):
            return a

        def eggs(b):
            return b * 2

        res = [spam(a := eggs(a := h)) for h in range(2)]

        self.assertEqual(res, [0, 2])
        self.assertEqual(a, 2)

    def test_named_expression_scope_10(self):
        res = [b := [a := 1 for i in range(2)] for j in range(2)]

        self.assertEqual(res, [[1, 1], [1, 1]])
        self.assertEqual(a, 1)
        self.assertEqual(b, [1, 1])

    def test_named_expression_scope_11(self):
        res = [j := i for i in range(5)]

        self.assertEqual(res, [0, 1, 2, 3, 4])
        self.assertEqual(j, 4)

    def test_named_expression_scope_17(self):
        b = 0
        res = [b := i + b for i in range(5)]

        self.assertEqual(res, [0, 1, 3, 6, 10])
        self.assertEqual(b, 10)

    def test_named_expression_scope_18(self):
        def spam(a):
            return a

        res = spam(b := 2)

        self.assertEqual(res, 2)
        self.assertEqual(b, 2)

    def test_named_expression_scope_19(self):
        def spam(a):
            return a

        res = spam((b := 2))

        self.assertEqual(res, 2)
        self.assertEqual(b, 2)

    def test_named_expression_scope_20(self):
        def spam(a):
            return a

        res = spam(a=(b := 2))

        self.assertEqual(res, 2)
        self.assertEqual(b, 2)

    def test_named_expression_scope_21(self):
        def spam(a, b):
            return a + b

        res = spam(c := 2, b=1)

        self.assertEqual(res, 3)
        self.assertEqual(c, 2)

    def test_named_expression_scope_22(self):
        def spam(a, b):
            return a + b

        res = spam((c := 2), b=1)

        self.assertEqual(res, 3)
        self.assertEqual(c, 2)

    def test_named_expression_scope_23(self):
        def spam(a, b):
            return a + b

        res = spam(b=(c := 2), a=1)

        self.assertEqual(res, 3)
        self.assertEqual(c, 2)

    def test_named_expression_scope_24(self):
        a = 10
        def spam():
            nonlocal a
            (a := 20)
        spam()

        self.assertEqual(a, 20)

    def test_named_expression_scope_25(self):
        ns = {}
        code = """a = 10
def spam():
    global a
    (a := 20)
spam()"""

        exec(code, ns, {})

        self.assertEqual(ns["a"], 20)

    def test_named_expression_variable_reuse_in_comprehensions(self):
        # The compiler is expected to raise syntax error for comprehension
        # iteration variables, but should be fine with rebinding of other
        # names (e.g. globals, nonlocals, other assignment expressions)

        # The cases are all defined to produce the same expected result
        # Each comprehension is checked at both function scope and module scope
        rebinding = "[x := i for i in range(3) if (x := i) or not x]"
        filter_ref = "[x := i for i in range(3) if x or not x]"
        body_ref = "[x for i in range(3) if (x := i) or not x]"
        nested_ref = "[j for i in range(3) if x or not x for j in range(3) if (x := i)][:-3]"
        cases = [
            ("Rebind global", f"x = 1; result = {rebinding}"),
            ("Rebind nonlocal", f"result, x = (lambda x=1: ({rebinding}, x))()"),
            ("Filter global", f"x = 1; result = {filter_ref}"),
            ("Filter nonlocal", f"result, x = (lambda x=1: ({filter_ref}, x))()"),
            ("Body global", f"x = 1; result = {body_ref}"),
            ("Body nonlocal", f"result, x = (lambda x=1: ({body_ref}, x))()"),
            ("Nested global", f"x = 1; result = {nested_ref}"),
            ("Nested nonlocal", f"result, x = (lambda x=1: ({nested_ref}, x))()"),
        ]
        for case, code in cases:
            with self.subTest(case=case):
                ns = {}
                exec(code, ns)
                self.assertEqual(ns["x"], 2)
                self.assertEqual(ns["result"], [0, 1, 2])

    def test_named_expression_global_scope(self):
        sentinel = object()
        global GLOBAL_VAR
        def f():
            global GLOBAL_VAR
            [GLOBAL_VAR := sentinel for _ in range(1)]
            self.assertEqual(GLOBAL_VAR, sentinel)
        try:
            f()
            self.assertEqual(GLOBAL_VAR, sentinel)
        finally:
            GLOBAL_VAR = None

    def test_named_expression_global_scope_no_global_keyword(self):
        sentinel = object()
        def f():
            GLOBAL_VAR = None
            [GLOBAL_VAR := sentinel for _ in range(1)]
            self.assertEqual(GLOBAL_VAR, sentinel)
        f()
        self.assertEqual(GLOBAL_VAR, None)

    def test_named_expression_nonlocal_scope(self):
        sentinel = object()
        def f():
            nonlocal_var = None
            def g():
                nonlocal nonlocal_var
                [nonlocal_var := sentinel for _ in range(1)]
            g()
            self.assertEqual(nonlocal_var, sentinel)
        f()

    def test_named_expression_nonlocal_scope_no_nonlocal_keyword(self):
        sentinel = object()
        def f():
            nonlocal_var = None
            def g():
                [nonlocal_var := sentinel for _ in range(1)]
            g()
            self.assertEqual(nonlocal_var, None)
        f()

    def test_named_expression_scope_in_genexp(self):
        a = 1
        b = [1, 2, 3, 4]
        genexp = (c := i + a for i in b)

        self.assertNotIn("c", locals())
        for idx, elem in enumerate(genexp):
            self.assertEqual(elem, b[idx] + a)

    def test_named_expression_scope_mangled_names(self):
        class Foo:
            def f(self_):
                global __x1
                __x1 = 0
                [_Foo__x1 := 1 for a in [2]]
                self.assertEqual(__x1, 1)
                [__x1 := 2 for a in [3]]
                self.assertEqual(__x1, 2)

        Foo().f()
        self.assertEqual(_Foo__x1, 2)

if __name__ == "__main__":
    run_tests()
