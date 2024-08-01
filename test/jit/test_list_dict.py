# Owner(s): ["oncall: jit"]

import inspect
import os
import sys
import types
import unittest
from collections import defaultdict, OrderedDict
from textwrap import dedent
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn

from torch import Tensor
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import skipIfTorchDynamo, TEST_CUDA
from torch.testing._internal.jit_utils import JitTestCase, make_global

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestList(JitTestCase):
    def test_list_bool_conversion(self):
        def if_predicate(l: List[int]):
            if l:
                s = 0
                for n in l:
                    s += n

                return s
            else:
                return -1

        self.checkScript(if_predicate, ([1, 2, 3],))
        self.checkScript(if_predicate, ([],))

        def while_predicate(l: List[int]):
            s = 0

            while l:
                s += l.pop()

        self.checkScript(while_predicate, ([1, 2, 3],))
        self.checkScript(while_predicate, ([],))

        def ternary_predicate(l: List[int]):
            return "non-empty" if l else "empty"

        self.checkScript(ternary_predicate, ([1, 2, 3],))
        self.checkScript(ternary_predicate, ([],))

    def test_in_check(self):
        def int_in(x: List[int]) -> bool:
            return 2 in x

        self.checkScript(int_in, ([1, 2, 3],))
        self.checkScript(int_in, ([1, 3, 3],))

        def float_in(x: List[float]) -> bool:
            return 2.0 in x

        self.checkScript(float_in, ([1.0, 2.0, 3.0],))
        self.checkScript(float_in, ([1.0, 3.0, 3.0],))

        def str_in(x: List[str]) -> bool:
            return "hi" in x

        self.checkScript(str_in, (["not", "here"],))
        self.checkScript(str_in, (["hi", "bye"],))
        self.checkScript(str_in, ([],))

    def test_list_literal(self):
        def reassign():
            x = [1]
            if 1 == 1:
                x = [2, 3]
            return

        self.checkScript(reassign, (), optimize=False)

        def reassign_arity_change():
            x = [1]
            if 1 == 1:
                x = [1, 2, 3]
            return

        self.checkScript(reassign_arity_change, (), optimize=False)

        def reassign_from_empty_literal():
            x = []
            if 1 == 1:
                x = [1, 2, 3]
            return

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"previously had type List\[Tensor\]", "x"
        ):
            self.checkScript(reassign_from_empty_literal, (), optimize=False)

        def reassign_from_empty_builtin():
            x = torch.jit.annotate(List[int], [])
            if 1 == 1:
                x = [1, 2, 3]
            y = torch.jit.annotate(List[float], [])
            if 1 == 1:
                y = [1.0, 2.0, 3.0]
            z = []
            if 1 == 1:
                z = [torch.randn([1])]
            return

        self.checkScript(reassign_from_empty_builtin, (), optimize=False)

        def reassign_bad_type():
            x = [1]
            if 1 == 1:
                x = [1.0]
            return

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "previously had type", "x"
        ):
            self.checkScript(reassign_bad_type, (), optimize=False)

        def reassign_nested():
            x = torch.jit.annotate(List[int], [])
            if 1 == 1:
                x = [1, 2, 3]
                if 1 == 1:
                    x = [1.0]
            return

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "previously had type", "x"
        ):
            self.checkScript(reassign_nested, (), optimize=False)

    def test_list_variance(self):
        """
        `List[T1]` is not a subtype of `List[T2]`, even if `T1` is a
        subtype of `T2`. However, if we have a temporary list object
        (that is, a list comprehension or a list literal) on the rhs of
        an assignment statement, we want to ignore the inferred type of
        the rhs if we can prove that: 1) both the lhs and the rhs are
        lists, and 2) the inner type of the lhs list is a subtype of the
        inner type of the rhs list.

        # This should pass
        x: List[Optional[int]] = [None, None, None]

        # This should fail
        y: List[None] = [None, None, None]
        x: List[Optional[int]] = y
        """

        def test_listliteral_is_typed_from_annotation():
            x: List[Optional[int]] = [None, None, None]
            return x

        self.checkScript(test_listliteral_is_typed_from_annotation, ())

        def test_listcomprehension_is_typed_from_annotation():
            x: List[Optional[int]] = [None for _ in range(3)]
            return x

        self.checkScript(test_listcomprehension_is_typed_from_annotation, ())

        def test_lists_with_different_internal_types_are_invariant(self):
            x: List[int] = [1, 2, 3]
            y: List[Optional[int]] = x
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'y' is "
            "annotated with type "
            r"List\[Optional\[int\]\] but is "
            "being assigned to a value of type "
            r"List\[int\]",
        ):
            torch.jit.script(test_lists_with_different_internal_types_are_invariant)

        def test_lists_with_different_internal_types_are_invariant_recursive(self):
            x: List[List[int]] = [[1, 2], [3]]
            y: List[List[Optional[int]]] = x
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'y' is "
            "annotated with type "
            r"List\[List\[Optional\[int\]\]\] "
            "but is being assigned to a value "
            r"of type List\[List\[int\]\]",
        ):
            torch.jit.script(
                test_lists_with_different_internal_types_are_invariant_recursive
            )

    def test_del(self):
        def inputs():
            return [1, 2, 3, 4]

        def fn(x: List[int]) -> List[int]:
            del x[1]
            return x

        python_out = fn(inputs())
        # checkScript reuses the same object, but here it's being mutated so do
        # it manually
        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(fn)))
        self.assertEqual(cu.fn(inputs()), python_out)
        self.assertEqual(torch.jit.script(fn)(inputs()), python_out)

        @torch.jit.script
        def fn2(x: List[int]) -> List[int]:
            del x[100]
            return x

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "out of range", "x[100]"
        ):
            fn2([])

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "deletion at a single index", "x[1:3]"
        ):

            @torch.jit.script
            def fn(x: List[int]) -> List[int]:
                del x[1:3]
                return x

    def test_list_keyword(self):
        def foo():
            return (
                list([1, 2, 3]),  # noqa: C410
                list(("a", "b")),  # noqa: C410
                list(range(5)),
                list("abcdefg"),
            )

        self.checkScript(foo, ())

        def foo2():
            x: List[int] = list()  # noqa: C408
            x.append(1)
            return (x,)

        self.checkScript(foo2, ())

        def foo3():
            return list(list("abc"))  # noqa: C414

        self.checkScript(foo3, ())
        FileCheck().check_count("aten::list", 2, exactly=True).run(
            torch.jit.script(foo3).graph
        )

    def test_dict_keyword_with_kwargs(self):
        def fn():
            return dict(foo=1, bar=2, baz=3)

        self.checkScript(fn, ())

    def test_dict_keyword_with_kwargs_using_container_values(self):
        def fn():
            return dict(foo=[1, 2, 3], bar=[4, 5, 6], baz=[7, 8, 9])

        self.checkScript(fn, ())

    def test_dict_keyword_with_iterable(self):
        def fn():
            return dict([("foo", 1), ("bar", 2), ("baz", 3)])  # noqa: C406

        self.checkScript(fn, ())

    def test_dict_keyword_with_empty_iterable(self):
        def fn():
            return dict([])  # noqa: C406

        self.checkScript(fn, ())

    def test_dict_keyword_with_internal_aggregate_function(self):
        def fn():
            return dict(zip(["foo", "baz", "bar"], [1, 2, 3]))

        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping(self):
        def fn():
            return {"foo": 1, "bar": 2, "baz": 3}

        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping_and_kwargs(self):
        def fn():
            return dict({"foo": 1, "bar": 2}, baz=3)

        self.checkScript(fn, ())

    def test_dict_keyword_with_dict_comprehension(self):
        def fn():
            return {i: chr(i + 65) for i in range(4)}

        self.checkScript(fn, ())

    def test_dict_keyword_with_dict_comprehension_and_kwargs(self):
        def fn():
            return dict({chr(65 + i): i for i in range(4)}, foo=2)

        self.checkScript(fn, ())

    def test_dict_keyword_with_empty_dict_comprehension(self):
        def fn():
            return {}

        self.checkScript(fn, ())

    def test_dict_keyword_is_correctly_typed(self):
        def fn():
            x: Dict[str, int] = dict()  # noqa: C408
            x["foo"] = 1
            return x

        self.checkScript(fn, ())

    def test_dict_keyword_with_mismatched_annotations(self):
        err_msg = (
            r"Dict type annotation `Dict\[int, str\]` did not "
            "match the type of an actual key type `str`"
        )
        with self.assertRaisesRegex(RuntimeError, err_msg):

            @torch.jit.script
            def fn():
                x: Dict[int, str] = dict(  # noqa: C406
                    [("foo", 1), ("bar", 2), ("baz", 3)]
                )
                return x

    def test_dict_keyword_with_nested_call(self):
        def fn():
            return dict(dict(foo=1, bar=2, baz=3))

        self.checkScript(fn, ())

    def test_dict_keyword_with_previously_declared_variable(self):
        def fn():
            d = {"foo": 1, "bar": 2}
            return dict(d)

        self.checkScript(fn, ())

    def test_dict_keyword_with_previously_declared_variable_and_kwargs(self):
        def fn():
            d = {"foo": 1, "bar": 2}
            return dict(d, baz=3)

        self.checkScript(fn, ())

    def test_min_bool_list(self):
        def jit_min_list(a: List[bool], b: List[bool]) -> List[bool]:
            return min(a, b)

        self.checkScript(jit_min_list, ([True, False], [False, True]))

    def test_min_max_list(self):
        def jit_min_list(a: List[int], b: List[int]) -> List[int]:
            return min(a, b)

        def jit_min_list_float(a: List[float], b: List[float]) -> List[float]:
            return min(a, b)

        def jit_min_list_bool(a: List[bool], b: List[bool]) -> List[bool]:
            return min(a, b)

        def run_tests(func, a, b):
            for t in zip(a, b):
                self.checkScript(func, t)

        args_left_int = [[1, 8, 8], [2, 1, 1], [], [2], [1], [1, 2, 3]]
        args_right_int = [[2, 1, 1], [1, 8, 8], [], [1], [], [1, 2]]
        run_tests(jit_min_list, args_left_int, args_right_int)

        args_left_float = [
            [1.0, 8.0, 8.0],
            [2.0, 1.0, 1.0],
            [],
            [2.0],
            [1.0],
            [1.0, 2.0, 3.0],
        ]
        args_right_float = [[2.0, 1.0, 1.0], [1.0, 8.0, 8.0], [], [1.0], [], [1.0, 2.0]]
        run_tests(jit_min_list_float, args_left_float, args_right_float)

        args_left_bool = [
            [],
            [],
            [],
            [False],
            [True],
            [False, True],
            [True, True],
            [False, False, False],
            [False, False, True],
        ]
        args_right_bool = [
            [],
            [False],
            [True],
            [True],
            [False],
            [True, True],
            [False, True],
            [False, False, True],
            [False, False, False],
        ]
        run_tests(jit_min_list_bool, args_left_bool, args_right_bool)

        def jit_max_list(a: List[int], b: List[int]) -> List[int]:
            return max(a, b)

        def jit_max_list_float(a: List[float], b: List[float]) -> List[float]:
            return max(a, b)

        def jit_max_list_bool(a: List[bool], b: List[bool]) -> List[bool]:
            return max(a, b)

        args_left_int = [[1, 8, 8], [8, 1, 1], [], [1], [], [1, 2]]
        args_right_int = [[8, 1, 1], [1, 8, 8], [], [2], [1], [1, 2, 3]]
        run_tests(jit_max_list, args_left_int, args_right_int)

        args_left_float = [[1.0, 8.0, 8.0], [8.0, 1.0, 1.0], [], [1.0], [], [1.0, 2.0]]
        args_right_float = [
            [8.0, 1.0, 1.0],
            [1.0, 8.0, 8.0],
            [],
            [2.0],
            [1.0],
            [1.0, 2.0, 3.0],
        ]
        run_tests(jit_max_list_float, args_left_float, args_right_float)

        run_tests(jit_max_list_bool, args_left_bool, args_right_bool)

    def test_list_gather(self):
        def index():
            a = [1, 2, 3]
            return a[1]

        self.checkScript(index, ())

        def negative_index():
            a = [1, 2, 3]
            return a[-1]

        self.checkScript(negative_index, ())

        def bad_index():
            a = [1, 2, 3]
            return a[4]

        self.checkScriptRaisesRegex(bad_index, (), Exception, "list index out of range")

        def bad_negative_index():
            a = [1, 2, 3]
            return a[-5]

        self.checkScriptRaisesRegex(
            bad_negative_index, (), Exception, "list index out of range"
        )

    def test_list_len(self):
        def func():
            a = [1, 2, 3]
            return len(a) == 3

        self.checkScript(func, ())

        def func2():
            a = []
            return len(a) == 0

        self.checkScript(func2, ())

    @skipIfTorchDynamo(
        "TorchDynamo fails to raise on this checkScriptRaisesRegex, because we trace it properly now"
    )
    def test_list_ops(self):
        def test_equality():
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a == b

        self.checkScript(test_equality, (), optimize=True)

        def test_equality_str():
            a = ["foo", "bar"]
            b = ["foo", "bar"]
            return a == b

        self.checkScript(test_equality_str, (), optimize=True)

        def test_inequality():
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a != b

        self.checkScript(test_inequality, (), optimize=True)

        def test_inequality_str():
            a = ["foo", "bar"]
            b = ["foo", "bar", "food"]
            return a != b

        self.checkScript(test_inequality_str, (), optimize=True)

        def test_non_equality():
            a = [1, 2, 3]
            b = [3]
            return a == b

        self.checkScript(test_non_equality, (), optimize=True)

        def test_non_inequality():
            a = [1, 2, 3]
            b = [3]
            return a != b

        self.checkScript(test_non_equality, (), optimize=True)

        def test_list_equality_as_cond():
            a = [1, 2, 3]
            b = [3]
            if a == b:
                c = 1
            else:
                c = 2
            return c

        self.checkScript(test_list_equality_as_cond, (), optimize=True)

        def test_list_add():
            a = [1, 2, 3]
            b = [2]
            c = a + b
            return c == [1, 2, 3, 2]

        self.checkScript(test_list_add, (), optimize=True)

        def test_list_add_empty():
            a = [1, 2, 3]
            b = torch.jit.annotate(List[int], [])
            c = a + b
            return c == [1, 2, 3]

        self.checkScript(test_list_add_empty, (), optimize=True)

        def test_tensor_list_equality():
            t1 = torch.ones([1, 1])
            t2 = torch.ones([1, 1])
            x = [t1, t2]
            y = [t2, t1]
            return x == y

        self.checkScript(test_tensor_list_equality, (), optimize=True)

        def test_invalid_list_equality():
            t1 = torch.ones([2, 2])
            t2 = torch.ones([2, 2])
            x = [t1, t2]
            y = [t2, t1]
            # will throw since the tensors have more than one element
            return x == y

        self.checkScriptRaisesRegex(
            test_invalid_list_equality, (), RuntimeError, "Boolean value of Tensor"
        )

    def test_list_sort(self):
        template = dedent(
            """
        def func():
            li_1 = {list_create}
            li_2 = {list_create}
            li_3 = {list_create}
            li_1.sort()
            li_2.sort(reverse=True)
            li_4 = sorted(li_3)
            return li_1, li_2, li_3, li_4
        """
        )

        lists = [
            "[]",
            "[1, 3, 2]",
            "[True, False, True]",
            "[1.2, .2, 3.2]",
            "[torch.tensor(1.0), torch.tensor(0.2), torch.tensor(0.5)]",
            "[torch.tensor(5), torch.tensor(-2), torch.tensor(4)]",
        ]
        for li in lists:
            code = template.format(list_create=li)
            scope = {}
            exec(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            t1 = cu.func()
            t2 = scope["func"]()
            self.assertEqual(t1, t2)

        def test_fail(x: List[Tensor]) -> List[Tensor]:
            x.sort()
            return x

        self.checkScriptRaisesRegex(
            test_fail,
            (([torch.zeros([2]), torch.zeros([2])],)),
            Exception,
            "Boolean value of Tensor with more than one value",
        )

        @torch.jit.script
        def test_mutation():
            a = [1, 2, 3]
            a.sort()
            return a

        test_mutation()
        FileCheck().check("aten::sort").run(test_mutation.graph_for())

        def test_sorted_copy():
            a = [torch.tensor(2), torch.tensor(0), torch.tensor(1)]
            b = sorted(a)
            a[0] = torch.tensor(10)
            return a, b

        self.checkScript(test_sorted_copy, ())

    def test_list_slice(self):
        def test_regular_slice():
            a = [0, 1, 2, 3, 4]
            return a[2:3] == [2]

        self.checkScript(test_regular_slice, ())

        def test_open_ended_slice():
            a = [0, 1, 2, 3, 4]
            return a[2:] == [2, 3, 4]

        self.checkScript(test_open_ended_slice, ())

        def test_open_ended_slice2():
            a = [0, 1, 2, 3, 4]
            return a[:2] == [0, 1]

        self.checkScript(test_open_ended_slice2, ())

        def test_negative_slice():
            a = [0, 1, 2, 3, 4]
            return a[:-1] == [0, 1, 2, 3]

        self.checkScript(test_negative_slice, ())

        def test_negative_slice2():
            a = [0, 1, 2, 3, 4]
            return a[-3:-1] == [2, 3]

        self.checkScript(test_negative_slice2, ())

        def test_backward_slice():
            a = [0, 1, 2, 3, 4]
            return a[3:2] == torch.jit.annotate(List[int], [])

        self.checkScript(test_backward_slice, ())

        def test_over_slice():
            a = [0, 1, 2, 3, 4]
            return a[3:10] == [3, 4]

        self.checkScript(test_backward_slice, ())

    def test_slice_index(self):
        a = torch.tensor(
            [
                [[1, 11], [2, 22]],
                [[3, 33], [4, 44]],
                [[5, 55], [6, 66]],
            ]
        )

        def test_index_slice1(x):
            x = x[:, :, [0, 1]]
            return x

        self.checkScript(test_index_slice1, (a,))

        def test_index_slice2(x):
            x = x[[2, 1, 0], :, :]
            return x

        self.checkScript(test_index_slice2, (a,))

        def test_index_slice3(x):
            x = x[[0, 1], :, [1]]
            return x

        self.checkScript(test_index_slice3, (a,))

        def test_index_slice_empty_list(x):
            empty_list: List[int] = []
            x = x[empty_list, :, :]
            return x

        self.checkScript(test_index_slice_empty_list, (a,))

        def test_index_slice_out_of_bounds_index(x):
            x = x[[4], :, :]
            return x

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "index 4 is out of bounds for dimension 0 with size 3",
            "x[[4], :, :]",
        ):
            self.checkScript(test_index_slice_out_of_bounds_index, (a,))

    def test_mutable_list_append(self):
        def test_append():
            a = [0, 1]
            a.append(2)
            a.append(3)
            return a == [0, 1, 2, 3]

        self.checkScript(test_append, ())

    def test_comprehensions_basic(self):
        def comp(l: List[int]) -> List[int]:
            n = [x * 3 for x in l]
            return n

        comp([1, 2, 3])
        self.checkScript(comp, ([1, 2, 3],))

    def test_comprehensions_basic_float(self):
        def comp(l: List[float]) -> List[float]:
            n = [x * 3 for x in l]
            return n

        self.checkScript(comp, ([1.0, 2.0, 3.0],))

    def test_comprehensions_two_comps(self):
        @torch.jit.script
        def comp(l1: List[int], l2: List[int]) -> List[int]:
            n = [x * 3 for x in l1]
            n2 = [x + 2 for x in l2]
            return n + n2

        self.assertEqual(comp([1, 2, 3], [4, 5]), [3, 6, 9, 6, 7])

    def test_comprehension_out_type_not_in_type(self):
        def list_cast() -> int:
            li = [int(i) for i in [torch.tensor(0), torch.tensor(1), torch.tensor(2)]]
            return li[0] + li[1] + li[2]

        self.checkScript(list_cast, ())

    def test_comprehension_iterable(self):
        def test_func(fn, inputs):
            self.assertEqual(fn(*inputs), torch.jit.script(fn)(*inputs))

        def foo(names: List[int], results: List[int]) -> List[Tuple[int, int]]:
            return [(k + 5, v - 2) for k, v in zip(names, results)]

        test_func(foo, ([1, 2, 4], [4, 7, 9]))
        test_func(foo, ([5], [4, 7, 9]))

        def fn(x: int) -> List[int]:
            return [i for i in range(x)]  # noqa: C416

        test_func(fn, (9,))
        test_func(fn, (0,))
        test_func(fn, (-1,))

        def changes_type():
            a = [float(i) for i in range(5)]
            b = [float(i) for i in [1, 2, 3, 4]]
            c = [(float(i), j) for i, j in enumerate([1, 2, 3, 8])]
            return a, b, c

        test_func(changes_type, ())

        def test_zero_iter():
            return [str(i) for i, j in zip("", "")]

        test_func(test_zero_iter, ())

    def test_mutable_list_append_2(self):
        def test_append_2():
            a = [0, 1]
            a.append(2)
            a = [1]
            a.append(4)
            return a == [1, 4]

        self.checkScript(test_append_2, ())

    def test_mutable_list_append_if(self):
        def test_append_if():
            a = [1]
            if 1 == 1:
                a.append(4)
            return a == [1, 4]

        self.checkScript(test_append_if, ())

    def test_mutable_list_append_if_else(self):
        def test_append_if_else():
            a = [1]
            if 1 == 2:
                a.append(4)
            else:
                a.append(10)
            return a == [1, 10]

        self.checkScript(test_append_if_else, ())

    def test_mutable_list_append_loop(self):
        def test_append_loop():
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                a.append(i)

            return a == [0, 1, 2, 3, 4]

        self.checkScript(test_append_loop, ())

    def test_mutable_list_append_loop_if(self):
        def test_append_loop_if():
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                if i > 3:
                    a.append(i)
                else:
                    a.append(0)

            return a == [0, 0, 0, 0, 4]

        self.checkScript(test_append_loop_if, ())

    def test_mutable_list_nested_loop(self):
        def test_nested_loop():
            a = torch.jit.annotate(List[int], [])
            for i in range(2):
                for j in range(2):
                    a.append(i + j)

            return a == [0, 1, 1, 2]

        self.checkScript(test_nested_loop, ())

    def test_mutable_list_function_inline(self):
        @torch.jit.script
        def bar(y: List[int]) -> None:
            y.append(4)

        @torch.jit.script
        def foo():
            x = [1, 2, 3]
            bar(x)
            return x

        self.assertEqual(foo(), [1, 2, 3, 4])

    def test_mutable_list_reverse_empty(self):
        def test_reverse_empty():
            a = []
            a.reverse()

            return a == []

        self.checkScript(test_reverse_empty, ())

    def test_mutable_list_reverse(self):
        def test_reverse():
            a = [1, 2, 3, 4]
            a.reverse()

            return a == [4, 3, 2, 1]

        self.checkScript(test_reverse, ())

    def test_mutable_tensor_list_reverse(self):
        def test_tensor_reverse():
            a = [torch.tensor(1), torch.tensor(2)]
            a.reverse()

            return a == [torch.tensor(2), torch.tensor(1)]

        self.checkScript(test_tensor_reverse, ())

    def test_mutable_list_pop_empty(self):
        @torch.jit.script
        def test_pop_empty():
            a = torch.jit.annotate(List[int], [])
            return a.pop()

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "pop from empty list", "a.pop"
        ):
            test_pop_empty()

    def test_mutable_list_pop(self):
        def test_pop():
            a = [1, 2, 3, 4]
            b = a.pop()

            return b == 4

        self.checkScript(test_pop, ())

    def test_mutable_list_pop2(self):
        def test_pop2():
            a = [1, 2, 3, 4]
            b = a.pop()

            return len(a) == 3

        self.checkScript(test_pop2, ())

    def test_mutable_list_pop_at(self):
        def test_pop_at():
            a = [1, 2, 3, 4]
            b = a.pop(1)

            return b == 2

        self.checkScript(test_pop_at, ())

    def test_mutable_list_pop_at2(self):
        def test_pop_at2():
            a = [1, 2, 3, 4]
            b = a.pop(1)

            return len(a) == 3

        self.checkScript(test_pop_at2, ())

    def test_mutable_list_pop_at_negative(self):
        def test_pop_at_negative():
            a = [1, 2, 3, 4]
            b = a.pop(-2)

            return b == 3

        self.checkScript(test_pop_at_negative, ())

    def test_mutable_list_pop_at_negative2(self):
        def test_pop_at_negative2():
            a = [1, 2, 3, 4]
            b = a.pop(-2)

            return len(a) == 3

        self.checkScript(test_pop_at_negative2, ())

    def test_mutable_list_pop_slice(self):
        def test_pop_slice():
            a = [1, 2, 3, 4]
            b = [1, 2, 3, 4]

            a.pop()
            b = b[:-1]

            return a == b

        self.checkScript(test_pop_slice, ())

    def test_mutable_list_clear_empty(self):
        def test_clear_empty():
            a = torch.jit.annotate(List[int], [])
            a.clear()

            return len(a) == 0

        self.checkScript(test_clear_empty, ())

    def test_mutable_list_clear(self):
        def test_clear():
            a = [1, 2, 3, 4]
            a.clear()

            return len(a) == 0

        self.checkScript(test_clear, ())

    def test_mutable_list_insert(self):
        def test_list_insert():
            a = [1, 2, 3, 4]
            a.insert(2, 5)

            return a == [1, 2, 5, 3, 4]

        self.checkScript(test_list_insert, ())

    def test_mutable_list_insert_negative(self):
        def test_list_insert_negative():
            a = [1, 2, 3, 4]
            a.insert(-1, 5)

            return a == [1, 2, 3, 5, 4]

        self.checkScript(test_list_insert_negative, ())

    def test_mutable_list_insert_neg_out_of_bounds(self):
        def test_list_insert_neg_out_of_bounds():
            a = [1, 2, 3, 4]
            a.insert(-10, 5)

            return a == [5, 1, 2, 3, 4]

        self.checkScript(test_list_insert_neg_out_of_bounds, ())

    def test_mutable_list_insert_out_of_bounds(self):
        def test_list_insert_out_of_bounds():
            a = [1, 2, 3, 4]
            a.insert(10, 5)

            return a == [1, 2, 3, 4, 5]

        self.checkScript(test_list_insert_out_of_bounds, ())

    def test_mutable_list_remove_not_existing(self):
        @torch.jit.script
        def test_list_remove_not_existing():
            a = [1, 2, 3, 4]
            a.remove(5)

            return a

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "x not in list", "a.remove"
        ):
            test_list_remove_not_existing()

    def test_mutable_list_remove(self):
        def test_list_remove():
            a = [1, 2, 3, 4]
            a.remove(3)

            return a == [1, 2, 4]

        self.checkScript(test_list_remove, ())

        def test_str_list_remove():
            a = ["foo", "bar"]
            a.remove("foo")

            return a == ["bar"]

        self.checkScript(test_str_list_remove, ())

    def test_list_index_not_existing(self):
        @torch.jit.script
        def list_index_not_existing():
            a = [4, 1, 3, 2]
            i = a.index(5)

            return i

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "'5' is not in list", "a.index"
        ):
            list_index_not_existing()

    def test_list_index(self):
        def list_index():
            a = [4, 1, 3, 2]
            i = a.index(3)

            return i == 2

        self.checkScript(list_index, ())

        def list_str_index():
            a = ["foo", "bar"]
            i = a.index("bar")

            return i == 1

        self.checkScript(list_str_index, ())

    def test_tensor_list_index(self):
        def tensor_list_index():
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(3), torch.tensor(2)]
            i = a.index(torch.tensor(3))

            return i == 2

        self.checkScript(tensor_list_index, ())

    def test_tensor_list_index_not_existing(self):
        @torch.jit.script
        def tensor_list_index_not_existing():
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(3), torch.tensor(2)]
            i = a.index(torch.tensor(5))

            return i

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "is not in list", "a.index"
        ):
            tensor_list_index_not_existing()

    def test_list_count(self):
        def list_count():
            a = [4, 1, 4, 2, 4]
            i = a.count(4)

            return i == 3

        self.checkScript(list_count, ())

        def list_str_count():
            a = ["foo", "bar", "foo"]
            i = a.count("foo")

            return i == 2

        self.checkScript(list_str_count, ())

    def test_list_count_not_existing(self):
        def list_count_not_existing():
            a = [4, 1, 4, 2, 4]
            i = a.count(5)

            return i == 0

        self.checkScript(list_count_not_existing, ())

    def test_tensor_list_count(self):
        def tensor_list_count():
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(4), torch.tensor(4)]
            i = a.count(torch.tensor(4))

            return i == 3

        self.checkScript(tensor_list_count, ())

    def test_tensor_list_count_not_existing(self):
        def tensor_list_count_not_existing():
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(4), torch.tensor(4)]
            i = a.count(torch.tensor(5))

            return i == 0

        self.checkScript(tensor_list_count_not_existing, ())

    def test_mutable_list_remove_tensor(self):
        def test_list_remove_tensor():
            a = [torch.ones(1), torch.zeros(1), torch.ones(2)]
            a.remove(torch.zeros(1))

            return len(a) == 2

        self.checkScript(test_list_remove_tensor, ())

    def test_mutable_list_remove2(self):
        def test_list_remove2():
            a = [1]
            a.remove(1)

            return len(a) == 0

        self.checkScript(test_list_remove2, ())

    def test_extend_list_mutable(self):
        @torch.jit.script
        def extend_list(a: List[Tensor], b: List[Tensor]) -> List[Tensor]:
            a.extend(b)
            return a

        for l in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
            for r in [
                [],
                [torch.rand(2)],
                [torch.rand(2), torch.rand(2), torch.rand(2)],
            ]:
                self.assertEqual(extend_list(l, r), l + r)

    def test_extend_list_immutable(self):
        @torch.jit.script
        def extend_list(a: List[int], b: List[int]) -> List[int]:
            a.extend(b)
            return a

        for l in [[], [1], [1, 2, 3]]:
            for r in [[], [1], [1, 2, 3]]:
                self.assertEqual(extend_list(l, r), l + r)

    def test_copy_list_mutable(self):
        @torch.jit.script
        def copy_list(a: List[Tensor]) -> List[Tensor]:
            return a.copy()

        for l in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
            self.assertEqual(copy_list(l), l)

    def test_copy_list_immutable(self):
        @torch.jit.script
        def copy_list(a: List[int]) -> List[int]:
            return a.copy()

        for l in [[], [1], [1, 2, 3]]:
            self.assertEqual(copy_list(l), l)

    def test_min_max_single_list(self):
        def min_intlist(li: List[int]) -> int:
            return min(li)

        def max_intlist(li: List[int]) -> int:
            return max(li)

        def min_boollist(li: List[bool]) -> bool:
            return min(li)

        def max_boollist(li: List[bool]) -> bool:
            return max(li)

        def min_floatlist(li: List[float]) -> float:
            return min(li)

        def max_floatlist(li: List[float]) -> float:
            return max(li)

        int_lists = [1], [2, 1, 2], [-3, 4, 2], [-2, -7, 1, 4], [2, 1, 0, 4], []

        def check_list(fn, li):
            if len(li) == 0:
                self.checkScriptRaisesRegex(fn, (li,), Exception, "empty")
            else:
                self.checkScript(fn, (li,))

        for int_list in int_lists:
            check_list(min_intlist, int_list)
            check_list(max_intlist, int_list)

            bool_li = [bool(x) for x in int_list]
            check_list(min_boollist, bool_li)
            check_list(max_boollist, bool_li)

            float_li = [float(x) for x in int_list]
            check_list(min_floatlist, float_li)
            check_list(max_floatlist, float_li)

    def test_to_list(self):
        """Unit tests for Tensor.tolist() function."""

        """
        Boolean dtype unit tests.
        """

        def to_list_bool_0D(x: torch.Tensor) -> bool:
            li = torch.jit.annotate(bool, x.tolist())
            return li

        def to_list_bool_1D(x: torch.Tensor) -> List[bool]:
            li = torch.jit.annotate(List[bool], x.tolist())
            return li

        def to_list_bool_2D(x: torch.Tensor) -> List[List[bool]]:
            li = torch.jit.annotate(List[List[bool]], x.tolist())
            return li

        def to_list_bool_3D(x: torch.Tensor) -> List[List[List[bool]]]:
            li = torch.jit.annotate(List[List[List[bool]]], x.tolist())
            return li

        self.checkScript(to_list_bool_0D, (torch.tensor(False, dtype=torch.bool),))
        bool_input_1D = torch.tensor([True, False, True, False], dtype=torch.bool)
        self.checkScript(to_list_bool_1D, (bool_input_1D,))
        bool_input_2D = torch.tensor(
            [[True, True, False], [False, True, False]], dtype=torch.bool
        )
        self.checkScript(to_list_bool_2D, (bool_input_2D,))
        bool_input_3D = torch.tensor(
            [[[True, False], [False, True]], [[True, False], [False, False]]],
            dtype=torch.bool,
        )
        self.checkScript(to_list_bool_3D, (bool_input_3D,))
        bool_input_noncontiguous = torch.tensor(
            [[[True, False], [False, True]], [[True, False], [False, False]]],
            dtype=torch.bool,
        ).transpose(0, 1)
        self.checkScript(to_list_bool_3D, (bool_input_noncontiguous,))

        """
        Int dtype unit tests.
        """

        def to_list_int_0D(x: torch.Tensor) -> int:
            li = torch.jit.annotate(int, x.tolist())
            return li

        def to_list_int_1D(x: torch.Tensor) -> List[int]:
            li = torch.jit.annotate(List[int], x.tolist())
            return li

        def to_list_int_2D(x: torch.Tensor) -> List[List[int]]:
            li = torch.jit.annotate(List[List[int]], x.tolist())
            return li

        def to_list_int_3D(x: torch.Tensor) -> List[List[List[int]]]:
            li = torch.jit.annotate(List[List[List[int]]], x.tolist())
            return li

        self.checkScript(to_list_int_0D, (torch.tensor(1, dtype=torch.long),))
        int_input_1D = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        self.checkScript(to_list_int_1D, (int_input_1D,))
        int_input_2D = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.long)
        self.checkScript(to_list_int_2D, (int_input_2D,))
        int_input_3D = torch.tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.long
        )
        self.checkScript(to_list_int_3D, (int_input_3D,))
        int_input_noncontiguous = torch.tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.long
        ).transpose(0, 1)
        self.checkScript(to_list_int_3D, (int_input_noncontiguous,))

        """
        Float dtype unit tests.
        """

        def to_list_float_0D(x: torch.Tensor) -> float:
            li = torch.jit.annotate(float, x.tolist())
            return li

        def to_list_float_1D(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        def to_list_float_2D(x: torch.Tensor) -> List[List[float]]:
            li = torch.jit.annotate(List[List[float]], x.tolist())
            return li

        def to_list_float_3D(x: torch.Tensor) -> List[List[List[float]]]:
            li = torch.jit.annotate(List[List[List[float]]], x.tolist())
            return li

        # Test with torch.float dtype Tensors to check that they are converted to double automatically.
        self.checkScript(to_list_float_0D, (torch.randn(5, dtype=torch.float)[0],))
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.float),))
        self.checkScript(to_list_float_2D, (torch.randn(5, 6, dtype=torch.float),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.float),))
        self.checkScript(
            to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.float).transpose(0, 1),)
        )

        self.checkScript(to_list_float_0D, (torch.randn(5, dtype=torch.double)[0],))
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.double),))
        self.checkScript(to_list_float_2D, (torch.randn(5, 6, dtype=torch.double),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.double),))
        self.checkScript(
            to_list_float_3D,
            (torch.randn(5, 6, 7, dtype=torch.double).transpose(0, 1),),
        )

        """
        Complex dtype unit tests.
        """

        def to_list_complex_0D(x: torch.Tensor) -> complex:
            li = torch.jit.annotate(complex, x.tolist())
            return li

        def to_list_complex_1D(x: torch.Tensor) -> List[complex]:
            li = torch.jit.annotate(List[complex], x.tolist())
            return li

        def to_list_complex_2D(x: torch.Tensor) -> List[List[complex]]:
            li = torch.jit.annotate(List[List[complex]], x.tolist())
            return li

        def to_list_complex_3D(x: torch.Tensor) -> List[List[List[complex]]]:
            li = torch.jit.annotate(List[List[List[complex]]], x.tolist())
            return li

        # Test with torch.complex dtype Tensors to check that they are converted to double automatically.
        self.checkScript(to_list_complex_0D, (torch.randn(5, dtype=torch.cfloat)[0],))
        self.checkScript(to_list_complex_1D, (torch.randn(5, dtype=torch.cfloat),))
        self.checkScript(to_list_complex_2D, (torch.randn(5, 6, dtype=torch.cfloat),))
        self.checkScript(
            to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cfloat),)
        )
        self.checkScript(
            to_list_complex_3D,
            (torch.randn(5, 6, 7, dtype=torch.cfloat).transpose(0, 1),),
        )

        self.checkScript(to_list_complex_0D, (torch.randn(5, dtype=torch.cdouble)[0],))
        self.checkScript(to_list_complex_1D, (torch.randn(5, dtype=torch.cdouble),))
        self.checkScript(to_list_complex_2D, (torch.randn(5, 6, dtype=torch.cdouble),))
        self.checkScript(
            to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cdouble),)
        )
        self.checkScript(
            to_list_complex_3D,
            (torch.randn(5, 6, 7, dtype=torch.cdouble).transpose(0, 1),),
        )

        """
        Non-happy path tests:
            - missing type annotation
            - mismatch between type annotation and input
            - type annotation with unsupported type
            - type annotation with the wrong dimension
            - type annotation with scalar type that doesn't match the input scalar type
        """

        def to_list_missing_type_annotation(x: torch.Tensor) -> List[float]:
            li = x.tolist()
            return li

        def to_list_incorrect_type_annotation(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(float, x.tolist())
            return li

        def to_list_unsupported_type_annotation(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(List[str], x.tolist())
            return li

        def to_list_type_annotation_wrong_dim(x: torch.Tensor) -> List[List[float]]:
            li = torch.jit.annotate(List[List[float]], x.tolist())
            return li

        def to_list_type_annotation_incorrect_scalar_type(
            x: torch.Tensor,
        ) -> List[float]:
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Expected type hint for result of tolist()", "x.tolist("
        ):
            self.checkScript(to_list_missing_type_annotation, (torch.randn(5),))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"Return value was annotated as having type List\[float\] but is actually of type float",
            "return li",
        ):
            self.checkScript(to_list_incorrect_type_annotation, (torch.randn(5),))

        with self.assertRaisesRegex(
            RuntimeError, r"str is not one of the supported element types for tolist"
        ):
            self.checkScript(to_list_unsupported_type_annotation, (torch.randn(5),))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Output annotation list dimension and runtime tensor dimension must match",
        ):
            self.checkScript(
                to_list_type_annotation_wrong_dim, (torch.randn(5, dtype=torch.double),)
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"Output annotation element type and runtime tensor element type must match",
        ):
            self.checkScript(
                to_list_type_annotation_incorrect_scalar_type,
                (torch.ones(5, dtype=torch.long),),
            )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_to_list_gpu(self):
        """GPU tests for Tensor.tolist() function."""

        def to_list_bool_1D(x: torch.Tensor) -> List[bool]:
            li = torch.jit.annotate(List[bool], x.tolist())
            return li

        def to_list_int_1D(x: torch.Tensor) -> List[int]:
            li = torch.jit.annotate(List[int], x.tolist())
            return li

        def to_list_float_1D(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        self.checkScript(
            to_list_bool_1D,
            (torch.tensor([True, False, True, False], dtype=torch.bool).cuda(),),
        )
        self.checkScript(
            to_list_int_1D, (torch.tensor([1, 2, 3, 4], dtype=torch.long).cuda(),)
        )
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.double).cuda(),))

    def test_no_element_type_annotation(self):
        def fn_with_comment(x: torch.Tensor) -> List:
            a: List = x.tolist()
            return a

        def annotated_fn(x: torch.Tensor) -> List:
            a: List = x.tolist()
            return a

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            torch.jit.script(annotated_fn)

    def test_list_none(self):
        with self.assertRaisesRegex(
            RuntimeError, "Can not create ListType with None type"
        ):
            x = torch._C.ListType(None)

    def test_list_unification_hint(self):
        with self.assertRaisesRegex(
            RuntimeError, "Expected an annotation of type List"
        ):

            @torch.jit.script
            def x():
                b: int = [2, 3]
                return b


class TestDict(JitTestCase):
    def dict(self):
        return {"a": torch.ones(1), "b": torch.ones(1) + 1, "c": torch.ones(1) + 2}

    def dict2(self):
        return {
            "x": torch.ones(1) + 100,
            "y": torch.ones(1) + 101,
            "z": torch.ones(1) + 102,
        }

    def dict_bool(self):
        return {True: 1}

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_dict_bool_conversion(self):
        def if_predicate(d: Dict[int, int]):
            if d:
                s, t = 0, 0
                for k, v in d.items():
                    s += k
                    t += v

                return s, t
            else:
                return -1, -1

        self.checkScript(if_predicate, ({1: 2, 3: 5},))
        self.checkScript(if_predicate, ({},))

        def while_predicate(d: Dict[int, int]):
            while d:
                d.clear()

        self.checkScript(while_predicate, ({1: 2, 3: 5},))
        self.checkScript(while_predicate, ({},))

        def ternary_predicate(d: Dict[int, int]):
            return "non-empty" if d else "empty"

        self.checkScript(ternary_predicate, ({1: 2, 3: 5},))
        self.checkScript(ternary_predicate, ({},))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_del(self):
        def inputs():
            return {"hi": 2, "bye": 3}

        def fn(x: Dict[str, int]) -> Dict[str, int]:
            del x["hi"]
            return x

        python_out = fn(inputs())
        # checkScript reuses the same object, but here it's being mutated so do
        # it manually
        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(fn)))
        self.assertEqual(cu.fn(inputs()), python_out)
        self.assertEqual(torch.jit.script(fn)(inputs()), python_out)
        with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", 'x["hi"]'):
            self.checkScript(fn, [{}])

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_dict_variance(self):
        """
        `Dict[T1, _]` is not a subtype of `Dict[T2, _]`, even if `T1` is
        a subtype of `T2`; similarly `Dict[_, T1]` would not be a
        subtype of `Dict[_, T2]`.

        However, if we have a temporary dict object (that is, a dict
        comprehension or a dict literal) on the rhs of an assignment
        statement, we want to ignore the inferred type of the rhs if we
        can prove that: 1) both the lhs and the rhs are dicts with the
        same key types (TorchScript has a restricted set of allowed key
        types, so we don't need to worry about subtyping relationships
        here), and 2) the value type of the dict is a subtype of the
        value type of the rhs dict.
        """

        def test_dictliteral_is_typed_from_annotation():
            x: Dict[str, Optional[int]] = {"foo": None, "bar": None, "baz": None}
            return x

        self.checkScript(test_dictliteral_is_typed_from_annotation, ())

        def test_dictcomprehension_is_typed_from_annotation():
            metasyntactics = ["foo", "bar", "baz"]
            x: Dict[str, Optional[int]] = {  # noqa: C420, RUF025
                word: None for word in metasyntactics
            }
            return x

        self.checkScript(test_dictcomprehension_is_typed_from_annotation, ())

        def test_dicts_with_different_value_types_are_invariant(self):
            x: Dict[str, int] = {"foo": 1, "bar": 2, "baz": 3}
            y: Dict[str, Optional[int]] = x
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'y' is "
            "annotated with type "
            r"Dict\[str, Optional\[int\]\] but "
            "is being assigned to a value of "
            r"type Dict\[str, int\]",
        ):
            torch.jit.script(test_dicts_with_different_value_types_are_invariant)

        def test_dicts_with_different_value_types_are_invariant_recursive(self):
            x: Dict[str, int] = {"foo": 1, "bar": 2, "baz": 3}
            y: Dict[str, Dict[str, int]] = {"foo": x, "bar": x, "baz": x}
            z: Dict[str, Dict[str, Optional[int]]] = y
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'z' is "
            "annotated with type "
            r"Dict\[str, Dict\[str, Optional"
            r"\[int\]\]\] but is being assigned"
            r" to a value of type Dict\[str, "
            r"Dict\[str, int\]\]",
        ):
            torch.jit.script(
                test_dicts_with_different_value_types_are_invariant_recursive
            )

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_keys(self):
        @torch.jit.script
        def keys(x: Dict[str, Tensor]) -> List[str]:
            return list(x.keys())

        self.assertEqual(set(keys(self.dict())), set(self.dict().keys()))

        @torch.jit.script
        def specialized_list():
            li = {1: 1, 2: 2}.keys()
            li.append(3)
            return li

        self.assertTrue(set(specialized_list()) == {1, 2, 3})

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_values(self):
        @torch.jit.script
        def values(x: Dict[str, Tensor]) -> List[Tensor]:
            return list(x.values())

        the_dict = self.dict()
        self.assertEqual(set(values(the_dict)), set(the_dict.values()))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_len(self):
        def length(x: Dict[str, Tensor]) -> int:
            return len(x)

        self.checkScript(length, (self.dict(),))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_copy(self):
        def func(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
            return x.copy()

        self.checkScript(func, (self.dict(),))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_items(self):
        def func(x: Dict[str, Tensor]) -> List[Tuple[str, Tensor]]:
            return x.items()

        # The value returned by Python is in arbitrary order, so we can't use
        # checkScript
        scripted_func = torch.jit.script(func)

        eager_out = func(self.dict())
        script_out = scripted_func(self.dict())

        self.assertEqual(len(eager_out), len(script_out))
        for item in eager_out:
            self.assertTrue(item in script_out)

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_pop(self):
        def pop(x: Dict[str, Tensor], key: str) -> Tuple[Tensor, Dict[str, Tensor]]:
            return x.pop(key), x

        # checkScript doesn't copy the inputs, so we can't use it since this mutates
        # the dict
        def tester(fn, *args):
            eager_out = fn(self.dict(), *args)
            script_out = torch.jit.script(fn)(self.dict(), *args)
            self.assertEqual(eager_out, script_out)

        tester(pop, "a")

        with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", "x.pop"):
            torch.jit.script(pop)(self.dict(), "x")

        def default_pop(
            x: Dict[str, Tensor], key: str, default: Tensor
        ) -> Tuple[Tensor, Dict[str, Tensor]]:
            return x.pop(key, default), x

        tester(default_pop, "a", torch.randn(2, 2))
        tester(default_pop, "x", torch.randn(2, 2))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_setdefault(self):
        def setdefault(
            x: Dict[str, Tensor], key: str, default: Tensor
        ) -> Dict[str, Tensor]:
            x.setdefault(key, default)
            return x

        self.checkScript(setdefault, (self.dict(), "a", torch.randn(2, 2)))
        self.checkScript(setdefault, (self.dict(), "nonexistant", torch.randn(2, 2)))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_update(self):
        def update(
            a: Dict[str, Tensor], b: Dict[str, Tensor]
        ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
            a.update(b)
            return a, b

        self.checkScript(update, (self.dict(), self.dict()))
        self.checkScript(update, (self.dict(), self.dict2()))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_update_existing_key(self):
        def foo() -> Dict[str, int]:
            a: Dict[str, int] = {}
            for i in range(3):
                a.update({"a": i})
            return a

        self.checkScript(foo, ())

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_aug_assign(self):
        def aug_assign_dict_tensor(a: Dict[str, Tensor]) -> Dict[str, Tensor]:
            a["a"] += 1
            a["b"] -= 12
            a["c"] *= 122
            a["c"] /= 2
            a["c"] %= 2
            return a

        def aug_assign_dict_prim(a: Dict[str, float]) -> Dict[str, float]:
            a["a"] += 3.4
            a["b"] -= 2.4
            a["c"] *= 3.0
            a["c"] /= 2.0
            a["c"] %= 2.0
            return a

        self.checkScript(aug_assign_dict_tensor, (self.dict(),))
        self.checkScript(aug_assign_dict_prim, ({"a": 3.0, "b": 2.0, "c": 4.0},))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_popitem(self):
        @torch.jit.script
        def popitem(
            x: Dict[str, Tensor]
        ) -> Tuple[Tuple[str, Tensor], Dict[str, Tensor]]:
            item = x.popitem()
            return item, x

        # The value returned by Python is arbitrary, so we can't use checkScript
        eager_in = self.dict()
        eager_out = (eager_in.popitem(), eager_in)

        script_out = popitem(self.dict())

        # Check that an item was removed
        self.assertEqual(len(eager_out[1]), len(script_out[1]))

        # Check that the item is the correct types
        self.assertTrue(isinstance(script_out[0][0], str))
        self.assertTrue(isinstance(script_out[0][1], torch.Tensor))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_clear(self):
        def clear(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
            x.clear()
            return x

        self.checkScript(clear, (self.dict(),))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_get(self):
        def get(x: Dict[str, Tensor], key: str) -> Optional[Tensor]:
            return x.get(key)

        self.checkScript(get, (self.dict(), "a"))
        self.checkScript(get, (self.dict(), "doesn't exist"))

        def get_default(x: Dict[str, Tensor], key: str) -> Optional[Tensor]:
            return x.get(key, torch.randn(2, 2))

        self.checkScript(get, (self.dict(), "a"))
        self.checkScript(get, (self.dict(), "doesn't exist"))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_get_boolkey(self):
        def get(x: Dict[bool, int], key: bool) -> Optional[int]:
            return x.get(key)

        self.checkScript(get, (self.dict_bool(), True))
        self.checkScript(get, (self.dict_bool(), False))

        def get_default(x: Dict[bool, int], key: bool) -> int:
            return x.get(key, 42)

        self.checkScript(get_default, (self.dict_bool(), True))
        self.checkScript(get_default, (self.dict_bool(), False))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_basic(self):
        def simple(x: Dict[str, int]) -> Dict[str, int]:
            return x

        self.checkScript(simple, ({"item": 20, "other_item": 120},))

        def index(x: Dict[str, int]) -> int:
            return x["item"]

        self.checkScript(index, ({"item": 20, "other_item": 120},))

        def type_default() -> Dict[str, Tensor]:
            return {}

        self.checkScript(type_default, ())

        @torch.jit.script
        def missing_index(x: Dict[str, int]) -> int:
            return x["dne"]

        with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", 'x["dne"'):
            missing_index({"item": 20, "other_item": 120})

        code = dedent(
            """
            def literal1():
                return torch.jit.annotate(Dict[int, float], {})
            def literal2():
                return torch.jit.annotate(Dict[int, float], {10: 1.2})
        """
        )
        cu = torch.jit.CompilationUnit(code)
        self.assertEqual({}, cu.literal1())
        self.assertEqual({10: 1.2}, cu.literal2())

        cu = torch.jit.CompilationUnit(
            dedent(
                """
            def literal3():
                return torch.jit.annotate(Dict[int, float], {10: 1.2, 11: 1.3})
        """
            )
        )
        self.assertEqual({10: 1.2, 11: 1.3}, cu.literal3())

        def list_of_dicts() -> List[Dict[str, Tensor]]:
            return [{"word": torch.ones(2) + 3}, {"other word": torch.ones(1) + 2}]

        self.checkScript(list_of_dicts, ())

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_mutability(self):
        @torch.jit.script
        def fn() -> Dict[str, int]:
            a = torch.jit.annotate(Dict[str, int], {})
            a["ok"] = 10
            return a

        self.assertEqual(fn(), {"ok": 10})

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_key_type(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "but instead found type", "a[None]"
        ):

            @torch.jit.script
            def fn(a: Dict[str, int]) -> int:
                return a[None]

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_loop(self):
        @torch.jit.script
        def fn(x: int) -> Dict[str, int]:
            a = torch.jit.annotate(Dict[str, int], {})
            for i in range(x):
                a["ok"] = i
            return a

        self.assertEqual(fn(10), {"ok": 9})

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_view(self):
        def fn(x, y):
            l = {"a": x}
            x_view = l["a"]
            a = x + x
            x_view.add_(y)
            b = x + x
            return a == b

        self.checkScript(fn, (torch.rand(2, 3), torch.rand(2, 3)))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_membership(self):
        def fn(x: Dict[int, int], y: int) -> int:
            return x.get(y, 3)

        d = {1: 2, 3: 4}
        self.checkScript(fn, (d, 3))
        self.checkScript(fn, (d, 2))

        def optional(x: Dict[int, int], y: int) -> bool:
            res = x.get(y)
            return res is None

        self.checkScript(fn, (d, 3))
        self.checkScript(fn, (d, 2))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "is actually of type Optional", "return x.get(y"
        ):

            @torch.jit.script
            def bad_types(x: Dict[int, int], y: int) -> int:
                return x.get(y)  # noqa: T484

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_dict_to_python(self):
        @torch.jit.ignore
        def python_lookup(my_dict: Dict[str, int], keys: List[str]) -> List[int]:
            return [my_dict[k] for k in keys]

        def fn(my_dict: Dict[str, int], keys: List[str]) -> List[int]:
            return python_lookup(my_dict, keys)

        a_dict = {"a": torch.ones(1), "b": torch.ones(1) + 1, "c": torch.ones(1) + 2}
        self.checkScript(fn, (a_dict, ("a", "c")))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_ordered_dict(self):
        def test_func(fn, inputs):
            self.assertEqual(fn(*inputs), torch.jit.script(fn)(*inputs))

        def repeated_key():
            return OrderedDict([(1, 2), (2, 3), (1, 4)])

        test_func(repeated_key, ())

        def no_args():
            a = OrderedDict()
            a["one"] = torch.tensor(1)
            a["two"] = torch.tensor(2)

        test_func(no_args, ())

        def test_dict_constructor():
            a = dict()  # noqa: C408
            a["one"] = torch.tensor(1)
            return a, dict([(1, 2), (2, 3), (1, 4)])  # noqa: C406

        test_func(test_dict_constructor, ())

        def test_dict_initializer_list():
            a = {"1": torch.tensor(1), "2": torch.tensor(2)}
            output_order = []
            for key in a:
                output_order.append(a[key])
            return output_order

        test_func(test_dict_initializer_list, ())

        def test_dict_error():
            a = dict()  # noqa: C408
            a[1] = 2
            return a

        with self.assertRaisesRegexWithHighlight(
            Exception, "Arguments for call are not", "a[1] = 2"
        ):
            torch.jit.script(test_dict_error)

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_type_annotation_missing_contained_type(self):
        """
        Test that the use of a Dict type annotation without contained
        key and value types produces an error.
        """

        # This function uses a type comment.
        def fn_with_comment(input: Dict) -> Any:
            return input

        # This function uses Python3 style type annotations.
        def annotated_fn(input: Dict) -> Any:
            return input

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            m = torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            m = torch.jit.script(annotated_fn)

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_dict_preserves_order(self):
        def dict_ordering():
            a: Dict[int, int] = {}
            for i in range(1000):
                a[i] = i + 1
            return a

        self.checkScript(dict_ordering, ())
        di = torch.jit.script(dict_ordering)()
        res = list(di.items())
        for i in range(1000):
            key, value = res[i]
            self.assertTrue(key == i and value == i + 1)

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_optional_dict_construct(self):
        class M(torch.nn.Module):
            def use(self, buffer: Dict[str, Optional[torch.Tensor]]):
                return buffer["prev_key"]

            def forward(self, x):
                prev_key = torch.rand(2, 3)
                next_key = torch.rand(2, 3)
                saved_state: Dict[str, Optional[torch.Tensor]] = {
                    "prev_key": prev_key,
                    "next_key": next_key,
                }

                return self.use(saved_state)

        self.checkModule(M(), (torch.rand(2, 2),))


class TestNamedTuple(JitTestCase):
    def test_namedtuple(self):
        class FeatureVector(NamedTuple):
            float_features: float
            sequence_features: List[float]
            time_since_first: float

        @torch.jit.script
        def foo(x) -> float:
            fv = FeatureVector(3.0, [3.0], 3.0)
            rv = fv.float_features
            for val in fv.sequence_features:
                rv += val
            rv *= fv.time_since_first
            return rv

        self.assertEqual(foo(torch.rand(3, 4)), 18.0)

    def test_namedtuple_constant(self):
        class Tup(NamedTuple):
            a: int
            b: int

        @torch.jit.script
        def foo():
            return Tup(1, 2)

        self.assertEqual(foo(), Tup(1, 2))

    def test_return_named_tuple(self):
        class FeatureVector(NamedTuple):
            float_features: float
            sequence_features: List[float]
            time_since_first: float

        @torch.jit.script
        def foo(x):
            fv = FeatureVector(3.0, [3.0], 3.0)
            return fv

        out = foo(torch.rand(3, 4))
        out = foo(torch.rand(3, 4))
        self.assertEqual(out.float_features, 3.0)
        self.assertEqual(out.sequence_features, [3.0])
        self.assertEqual(out.time_since_first, 3.0)

    def test_namedtuple_as_attr(self):
        class Config(NamedTuple):
            size: int

        class MyMod(nn.Module):
            configs: Dict[int, Config]

            def __init__(self, configs):
                super().__init__()
                self.configs = configs

            def forward(self, x):
                for config in self.configs.values():
                    x += config.size
                return x

        s = torch.jit.script(MyMod({0: Config(size=16)}))

    def test_namedtuple_resolution(self):
        class TheType(NamedTuple):
            t: int

        class MyModule(types.ModuleType):
            def __init__(self) -> None:
                super().__init__("MyModule")

            def __getattr__(self, attr):
                return TheType

        some_module = MyModule()

        def fn() -> some_module.Type:
            return some_module.Type(1)

        self.checkScript(fn, [])

    def test_namedtuple_slice_unpack(self):
        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo(a: int, b: float, c: List[int]):
            tup = MyCoolNamedTuple(a, b, c)
            my_a, my_b, my_c = tup
            return tup[:1], my_a, my_c

        self.assertEqual(foo(3, 3.5, [6]), ((3,), 3, [6]))

    def test_namedtuple_lower(self):
        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo(a: int):
            tup = MyCoolNamedTuple(a, 3.14, [9])
            return tup

        FileCheck().check("TupleConstruct").run(foo.graph)
        torch._C._jit_pass_lower_all_tuples(foo.graph)
        FileCheck().check_not("TupleConstruct").run(foo.graph)

    def test_namedtuple_type_annotation(self):
        global MyCoolNamedTuple  # see [local resolution in python]

        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo(x: MyCoolNamedTuple) -> MyCoolNamedTuple:
            return x

        mnt = MyCoolNamedTuple(42, 420.0, [666])
        self.assertEqual(foo(mnt), mnt)

    def test_namedtuple_wrong_types(self):
        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a value of type 'int' for argument 'a'"
            " but instead found type 'str'",
        ):

            @torch.jit.script
            def foo():
                tup = MyCoolNamedTuple("foo", "bar", "baz")
                return tup

    def test_namedtuple_kwarg_construct(self):
        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo():
            tup = MyCoolNamedTuple(c=[1, 2, 3], b=3.5, a=9)
            return tup

        tup = foo()
        self.assertEqual(tup.a, 9)
        self.assertEqual(tup.b, 3.5)
        self.assertEqual(tup.c, [1, 2, 3])

    @unittest.skipIf(True, "broken while these tests were not in CI")
    def test_namedtuple_serialization(self):
        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        class MyMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self):
                return MyCoolNamedTuple(3, 3.5, [3, 4, 5])

        mm = MyMod()
        mm.save("foo.zip")
        torch.testing._internal.jit_utils.clear_class_registry()
        loaded = torch.jit.load("foo.zip")

        out = mm()
        out_loaded = loaded()

        for name in ["a", "b", "c"]:
            self.assertEqual(getattr(out_loaded, name), getattr(out, name))

    def test_namedtuple_inside_forwardref(self):
        class FeatureVector(NamedTuple):
            float_features: "float"
            sequence_features: "List[float]"
            time_since_first: "float"

        @torch.jit.script
        def foo(x) -> float:
            fv = FeatureVector(3.0, [3.0], 3.0)
            rv = fv.float_features
            for val in fv.sequence_features:
                rv += val
            rv *= fv.time_since_first
            return rv

        self.assertEqual(foo(torch.rand(3, 4)), 18.0)

    def test_namedtuple_input_forwardref(self):
        class MyNamedTuple(NamedTuple):
            a: "int"
            b: "float"
            c: "torch.Tensor"

        make_global(MyNamedTuple)

        nt = MyNamedTuple(4, 2.5, torch.rand((2, 2)))

        def fn(obj: MyNamedTuple):
            return ((obj.c + obj.b) ** obj.a).sin()

        expected = fn(nt)
        fn_s = torch.jit.script(fn)
        actual = fn_s(nt)
        self.assertEqual(expected, actual)

    # see #95858
    @unittest.expectedFailure
    def test_namedtuple_resolution_forwardref(self):
        class TheType(NamedTuple):
            t: "int"

        class MyModule(types.ModuleType):
            def __init__(self) -> None:
                super().__init__("MyModule")

            def __getattr__(self, attr):
                return TheType

        some_module = MyModule()

        def fn() -> some_module.Type:
            return some_module.Type(1)

        self.checkScript(fn, [])


class TestScriptDict(JitTestCase):
    """
    This class contains a suite of tests for torch.jit.script, a
    function that returns a dictionary-like object that has reference
    semantics across the Python/TorchScript boundary. That is,
    it can be passed to a TorchScript function that mutates it
    and those modifications are visible in the scope of the Python
    caller of said TorchScript function.

    The vast majority of tests are for making sure that objects returned
    by torch.jit.script behave like dictionaries do so that they are fungible
    in almost all cirumstances with regular dictionaries.
    """

    def _script_dict_add(self, d: torch._C.ScriptDict, k: int, v: int):
        """
        This is a helper function that inserts the pair (k, v) into the
        dictionary d in TorchScript. It is used for testing reference
        semantics.
        """

        @torch.jit.script
        def dict_add(d: Dict[int, int], k: int, v: int):
            d[k] = v

        dict_add(d, k, v)

    def _compare_eager_and_script(self, fn, input_dict, script_input_dict=None):
        """
        This is a helper function that facilitates comparing behaviour between
        Python dictionaries and "scripted" dictionaries.

        Args:
            fn: The function to test and compare the behaviour of.
            input_dict: The input dictionary to use for the test (passed to fn).
            script_input_dict: The scripted input dictionary to use for the tests.
                                If None, input_dict is scripted with torch.jit.script
                                and used instead.
        """
        # Create ScriptDict version of input_dict if needed.
        script_input_dict = script_input_dict or torch.jit.script(input_dict)

        # Run fn with both input_dict and scripted_dict.
        eager_raised, script_raised = False, False

        try:
            eager_out = fn(input_dict)
        except Exception as e:
            eager_exception = e
            eager_raised = True

        try:
            script_out = fn(script_input_dict)
        except Exception as e:
            script_exception = e
            script_raised = True

        # Check that both calls raised or none of them raised.
        self.assertEqual(eager_raised, script_raised)

        if eager_raised:
            # If fn raised an exception, it should be the same between
            # regular and scripted dictionaries.
            self.assertEqual(type(eager_exception), type(script_exception))
        else:
            # Otherwise, make sure the outputs match and the dictionaries
            # match (the latter may not be the same as the output).
            self.assertEqual(eager_out, script_out)
            self.assertEqual(input_dict, script_input_dict)

    def test_repr(self):
        """
        Test the __repr__ method.
        """
        self._compare_eager_and_script(lambda d: repr(d), {1: 2})

    def test_bool(self):
        """
        Test the __bool__ method. This should return True
        if the dictionary is non-empty and False otherwise.
        """
        self._compare_eager_and_script(lambda d: bool(d), {1: 2})
        self._compare_eager_and_script(lambda d: bool(d), {})

    def test_iter(self):
        """
        Test iteration over a dictionary's keys.
        """

        def sum_keys(input_dict):
            s = 0
            for k in input_dict:
                s += k

            return s

        self._compare_eager_and_script(sum_keys, {1: 2, 3: 4})

    def test_items(self):
        """
        Test .items().
        """

        def sum_pair_product(input_dict):
            s = 0
            for k, v in input_dict.items():
                s += k * v

            return s

        self._compare_eager_and_script(sum_pair_product, {1: 2, 3: 4})

    def test_getitem(self):
        """
        Test accessing dictionary values using the [] operator.
        """
        data = {1: 2, 3: 4}
        self._compare_eager_and_script(lambda d: d[1], data)
        self._compare_eager_and_script(lambda d: d[4], data)
        self._compare_eager_and_script(lambda d: d[2], data)
        self._compare_eager_and_script(lambda d: d["key"], data)

    def test_setitem(self):
        """
        Test setting dictionary values using the [] operator.
        """
        data = {1: 2, 3: 4}

        def fn(input_dict):
            input_dict[1] = 10
            input_dict[3] = 11

        self._compare_eager_and_script(fn, data)

        # Check that using improperly typed keys and values
        # throws TypeError.
        # _compare_eager_and_script cannot be used here since
        # the following uses of __setitem__ are valid in
        # Python.
        script_data = torch.jit.script(data)

        with self.assertRaises(TypeError):
            script_data["str"] = 3

        with self.assertRaises(TypeError):
            script_data[3] = "str"

    def test_contains(self):
        """
        Test membership checks (x in y, x not in y).
        """
        data = {1: 2, 3: 4}

        def fn(input_dict):
            return (
                1 in input_dict,
                2 not in input_dict,
                3 in input_dict,
                4 not in input_dict,
            )

        self._compare_eager_and_script(fn, data)

        # Check that using an improperly typed key
        # throws KeyError.
        script_data = torch.jit.script(data)

        with self.assertRaises(KeyError):
            a = "str" in script_data

    def test_delitem(self):
        """
        Test deletion.
        """
        data = {1: 2, 3: 4}

        def del_fn(input_dict):
            del input_dict[1]

        def del_fn_raises(input_dict):
            del input_dict[10]

        self._compare_eager_and_script(del_fn, data)
        self._compare_eager_and_script(del_fn_raises, data)

        # Check that using an improperly typed key
        # throws TypeError.
        script_data = torch.jit.script(data)

        with self.assertRaises(TypeError):
            del script_data["str"]

    def test_len(self):
        """
        Test len() builtin function.
        """
        self._compare_eager_and_script(lambda d: len(d), {1: 2})
        self._compare_eager_and_script(lambda d: len(d), {})

    @unittest.skip(
        "Cannot pass until all dicts returned from TorchScript are ScriptDicts"
    )
    def test_nested(self):
        """
        Test that reference semantics are honoured when the ScriptDict that is
        mutated using TorchScript is inside another.
        """
        nested = torch.jit.script(
            {1: {1: 2}, 2: {3: 4}}, type_hint=Dict[int, Dict[int, int]]
        )

        one = nested[1]
        two = nested[2]

        self._script_dict_add(one, 9, 10)
        self._script_dict_add(two, 11, 12)

        # The mutation should be visible in the original dictionary, nested.
        self.assertEqual(len(one), 2)
        self.assertEqual(len(two), 2)
        self.assertEqual(len(nested[1]), 2)
        self.assertEqual(len(nested[2]), 2)

    def test_reference_semantics(self):
        """
        Test that reference semantics are honoured; that modifications made
        to a ScriptDict in TorchScript are visible in Python.
        """
        data = torch.jit.script({1: 2})
        self._script_dict_add(data, 3, 4)

        # The mutation should be visible in the original dictionary.
        self.assertEqual(len(data), 2)
        self.assertTrue(3 in data)
        self.assertEqual(data[3], 4)


class TestScriptList(JitTestCase):
    """
    This class contains a suite of tests for torch._C.ScriptList, a
    function that returns a list-like object that has reference
    semantics across the Python/TorchScript boundary. That is,
    it can be passed to a TorchScript function that mutates it
    and those modifications are visible in the scope of the Python
    caller of said TorchScript function.

    The vast majority of tests are for making sure that instances of
    torch._C.ScriptList behave like lists do so that they are fungible
    in almost all cirumstances with regular list.
    """

    def _script_list_add(self, l: torch._C.ScriptList, e: int):
        """
        This is a helper function that inserts the element e into the
        list l in TorchScript. It is used for testing reference
        semantics.
        """

        @torch.jit.script
        def list_add(l: List[int], e: int):
            l.append(e)

        list_add(l, e)

    def _compare_eager_and_script(self, fn, input_list, script_input_list=None):
        """
        This is a helper function that facilitates comparing behaviour between
        Python lists and "scripted" lists.
        Args:
            fn: The function to test and compare the behaviour of.
            input_list: The input list to use for the test (passed to fn).
            script_input_list: The scripted input list to use for the tests.
                                If None, input_list is scripted with torch.jit.script
                                and used instead.
        """
        # Create ScriptDict version of input_list if needed.
        script_input_list = script_input_list or torch.jit.script(input_list)

        # Run fn with both input_list and scripted_dict.
        eager_raised, script_raised = False, False

        try:
            eager_out = fn(input_list)
        except Exception as e:
            eager_exception = e
            eager_raised = True

        try:
            script_out = fn(script_input_list)
        except Exception as e:
            script_exception = e
            script_raised = True

        # Check that both calls raised or none of them raised.
        self.assertEqual(eager_raised, script_raised)

        if eager_raised:
            # If fn raised an exception, it should be the same between
            # regular and scripted lists.
            self.assertEqual(type(eager_exception), type(script_exception))
        else:
            # Otherwise, make sure the outputs match and the lists
            # match (the latter may not be the same as the output).
            self.assertEqual(eager_out, script_out)
            self.assertEqual(input_list, script_input_list)

    def test_repr(self):
        """
        Test the __repr__ method.
        """
        self._compare_eager_and_script(lambda l: repr(l), [1])

    def test_bool(self):
        """
        Test the __bool__ method. This should return True
        if the list is non-empty and False otherwise.
        """
        self._compare_eager_and_script(lambda l: bool(l), [1])
        self._compare_eager_and_script(lambda l: bool(l), [])

    def test_iter(self):
        """
        Test iteration over a list's elements.
        """

        def sum_elements(input_list):
            s = 0
            for k in input_list:
                s += k

            return s

        self._compare_eager_and_script(sum_elements, [1, 2, 3, 4])

    def test_getitem(self):
        """
        Test accessing list elements using the [] operator.
        """
        data = [1, 2, 3, 4]

        # Test regular indexing.
        self._compare_eager_and_script(lambda l: l[1], data)
        self._compare_eager_and_script(lambda l: l[3], data)
        self._compare_eager_and_script(lambda l: l[-1], data)

        # Test slicing.
        self._compare_eager_and_script(lambda l: l[1:3], data)
        self._compare_eager_and_script(lambda l: l[:], data)
        self._compare_eager_and_script(lambda l: l[1:], data)
        self._compare_eager_and_script(lambda l: l[:2], data)
        self._compare_eager_and_script(lambda l: l[-1], data)
        self._compare_eager_and_script(lambda l: l[-1::-1], data)

        # Test errors.
        self._compare_eager_and_script(lambda l: l[5], data)
        self._compare_eager_and_script(lambda l: l[-7], data)
        self._compare_eager_and_script(lambda l: l["key"], data)

    def test_setitem(self):
        """
        Test setting list elements using the [] operator.
        """
        data = [1, 2, 3, 4]

        # Test regular assignment.
        def setitem(input_list):
            input_list[1] = 10
            input_list[3] = 11
            input_list[-1] = 12

        self._compare_eager_and_script(setitem, data.copy())

        # Test slice assignment.
        # TODO: Something like input_list[:1] = [1, 2, 3, 4, 5]
        # is allowed in Python, but pybind11/stl_bind.h does not
        # allow it. Should we?
        def setitem_slice(input_list):
            input_list[:4:2] = [10, 11]
            input_list[-2:] = [15, 16]

        self._compare_eager_and_script(setitem_slice, data)

        # Test errors.
        def out_of_range(input_list):
            input_list[11] = 3

        def out_of_range_negative(input_list):
            input_list[-11] = 3

        def wrong_index_type(input_list):
            input_list["str"] = 3

        self._compare_eager_and_script(out_of_range, data)
        self._compare_eager_and_script(out_of_range_negative, data)
        self._compare_eager_and_script(wrong_index_type, data)

        # Check that using value of an incorrect type throws TypeError.
        # _compare_eager_and_script cannot be used here since
        # the following use of __setitem__ is valid in
        # Python.
        script_data = torch.jit.script(data)

        with self.assertRaises(TypeError):
            script_data[0] = "str"

    def test_contains(self):
        """
        Test membership checks (x in y, x not in y).
        """
        data = [1, 2, 3, 4]

        def fn(input_list):
            return (
                1 in input_list,
                2 not in input_list,
                3 in input_list,
                4 not in input_list,
            )

        self._compare_eager_and_script(fn, data)

        # Check that using a value of an incorrect type throws a TypeError.
        script_data = torch.jit.script(data)

        with self.assertRaises(TypeError):
            a = "str" in script_data

    def test_delitem(self):
        """
        Test deletion.
        """
        data = [1, 2, 3, 4]

        def del_fn(input_list):
            del input_list[1]

        def del_fn_out_of_range(input_list):
            del input_list[10]

        def del_fn_wrong_type(input_list):
            del input_list["str"]

        self._compare_eager_and_script(del_fn, data.copy())
        self._compare_eager_and_script(del_fn_out_of_range, data)
        self._compare_eager_and_script(del_fn_wrong_type, data)

    def test_len(self):
        """
        Test len() builtin function.
        """
        self._compare_eager_and_script(lambda l: len(l), [1, 2, 3, 4])
        self._compare_eager_and_script(lambda l: len(l), [])

    def test_count(self):
        """
        Test count method.
        """
        self._compare_eager_and_script(lambda l: l.count(3), [1, 2, 3, 3])

        # Check that using a value of an incorrect type throws TypeError.
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            script_data.count("str")

    def test_remove(self):
        """
        Test remove method.
        """
        self._compare_eager_and_script(lambda l: l.remove(1), [1, 2, 3])
        self._compare_eager_and_script(lambda l: l.remove(10), [1, 2, 3])

        # Check that using a value of an incorrect type throws TypeError.
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            script_data.remove("str")

    def test_append(self):
        """
        Test append method.
        """
        self._compare_eager_and_script(lambda l: l.append(1), [4, 3, 2])

        # Check that using a value of an incorrect type throws TypeError.
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            script_data.append("str")

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1991")
    def test_clear(self):
        """
        Test clear.
        """
        self._compare_eager_and_script(lambda l: l.clear(), [4, 3, 2])

    def test_extend(self):
        """
        Test extend.
        """

        class Iterable:
            def __init__(self, limit: int):
                self.limit = limit
                self.value = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.value == limit:  # noqa: F821
                    raise StopIteration

                ret = self.value
                self.value += 1
                return ret

        data = [1, 2, 3]

        def extend_list(input_list):
            input_list.extend([4, 5, 6])

        def extend_dict(input_list):
            input_list.extend({4: 10, 5: 11, 6: 12})

        def extend_iterable(input_list):
            input_list.extend(Iterable(3))

        self._compare_eager_and_script(extend_list, data.copy())
        self._compare_eager_and_script(extend_dict, data.copy())
        self._compare_eager_and_script(extend_iterable, data)

        # Check that using a value of an incorrect type throws TypeError.
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            script_data.extend(["a"])

        with self.assertRaises(TypeError):
            script_data.extend({"a": 1})

    def test_insert(self):
        """
        Test insert.
        """
        data = [1, 2, 4]

        self._compare_eager_and_script(lambda l: l.insert(3, 3), data.copy())
        self._compare_eager_and_script(lambda l: l.insert(0, 3), data.copy())
        self._compare_eager_and_script(lambda l: l.insert(-2, 3), data)

        # Check that using a value of an incorrect type throws TypeError.
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            script_data.insert((0, "str"))

    def test_pop(self):
        """
        Test pop.
        """
        data = [1, 2, 3, 4, 5]

        # Test normal cases.
        self._compare_eager_and_script(lambda l: l.pop(), data.copy())
        self._compare_eager_and_script(lambda l: l.pop(2), data.copy())
        self._compare_eager_and_script(lambda l: l.pop(-3), data.copy())

        # Test error cases.
        self._compare_eager_and_script(lambda l: l.pop(10), data)

    @unittest.skip(
        "Cannot pass until all list returned from TorchScript are ScriptLists"
    )
    def test_nested(self):
        """
        Test that reference semantics are honoured when the ScriptList that is
        mutated using TorchScript is inside another.
        """
        nested = torch.jit.script([[1], [2]], List[List[int]])

        one = nested[0]
        two = nested[1]

        self._script_list_add(one, 3)
        self._script_list_add(two, 4)

        # The mutation should be visible in the original list, nested.
        self.assertEqual(len(one), 2)
        self.assertEqual(len(two), 2)
        self.assertEqual(one[len(one) - 1], 3)
        self.assertEqual(two[len(one) - 1], 4)
        self.assertEqual(len(nested[0]), 2)
        self.assertEqual(len(nested[1]), 2)

    def test_reference_semantics(self):
        """
        Test that reference semantics are honoured; that modifications made
        to a ScriptList in TorchScript are visible in Python.
        """
        l = torch.jit.script([1, 2])
        self._script_list_add(l, 3)

        self.assertEqual(len(l), 3)
        self.assertTrue(3 in l)
        self.assertEqual(l[2], 3)

    def test_defaultdict(self):
        def get_dict():
            test_dict = defaultdict(list)
            return test_dict

        class Test(torch.nn.Module):
            segments_groupby_col: Dict[str, List[str]]

            def __init__(self) -> None:
                super().__init__()
                self.segments_groupby_col = get_dict()
                self.col1 = "a"
                self.col2 = "b"

            def forward(self):
                if self.col1 in self.segments_groupby_col.keys():
                    return 1
                else:
                    return 2

        test = Test()
        test_script = torch.jit.script(test)
        test_script.segments_groupby_col

        # Smoketest for flakiness. Takes around 2s.
        for i in range(300):
            test = Test()
            test_script = torch.jit.script(test)
