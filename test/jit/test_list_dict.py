import os
import sys
import inspect
import unittest
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from textwrap import dedent
from collections import OrderedDict

from torch import Tensor
import torch
import torch.nn as nn
import types
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

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
            return 2. in x

        self.checkScript(float_in, ([1., 2., 3.],))
        self.checkScript(float_in, ([1., 3., 3.],))

        def str_in(x: List[str]) -> bool:
            return 'hi' in x

        self.checkScript(str_in, (['not', 'here'],))
        self.checkScript(str_in, (['hi', 'bye'],))
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
        with self.assertRaisesRegexWithHighlight(RuntimeError, r"previously has type List\[Tensor\]", "x"):
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
        with self.assertRaisesRegexWithHighlight(RuntimeError, "previously has type", "x"):
            self.checkScript(reassign_bad_type, (), optimize=False)

        def reassign_nested():
            x = torch.jit.annotate(List[int], [])
            if 1 == 1:
                x = [1, 2, 3]
                if 1 == 1:
                    x = [1.0]
            return
        with self.assertRaisesRegexWithHighlight(RuntimeError, "previously has type", "x"):
            self.checkScript(reassign_nested, (), optimize=False)

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

        with self.assertRaisesRegexWithHighlight(RuntimeError, "out of range", "x[100]"):
            fn2([])

        with self.assertRaisesRegexWithHighlight(RuntimeError, "deletion at a single index", "x[1:3]"):
            @torch.jit.script
            def fn(x: List[int]) -> List[int]:
                del x[1:3]
                return x

    def test_list_keyword(self):
        def foo():
            return list([1, 2, 3]), list(("a", "b")), list(range(5)), list("abcdefg")  # noqa: C410

        self.checkScript(foo, ())

        def foo2():
            x: List[int] = list()
            x.append(1)
            return x,

        self.checkScript(foo2, ())

        def foo3():
            return list(list("abc"))

        self.checkScript(foo3, ())
        FileCheck().check_count("aten::list", 2, exactly=True).run(torch.jit.script(foo3).graph)

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
            return dict([("foo", 1), ("bar", 2), ("baz", 3)])    # noqa: C406

        self.checkScript(fn, ())

    def test_dict_keyword_with_empty_iterable(self):
        def fn():
            return dict([])    # noqa: C406

        self.checkScript(fn, ())

    def test_dict_keyword_with_internal_aggregate_function(self):
        def fn():
            return dict(zip(["foo", "baz", "bar"], [1, 2, 3]))

        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping(self):
        def fn():
            return dict({"foo" : 1, "bar" : 2, "baz" : 3})

        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping_and_kwargs(self):
        def fn():
            return dict({"foo" : 1, "bar" : 2}, baz=3)

        self.checkScript(fn, ())

    def test_dict_keyword_with_dict_comprehension(self):
        def fn():
            return dict({i: chr(i + 65) for i in range(4)})

        self.checkScript(fn, ())

    def test_dict_keyword_with_dict_comprehension_and_kwargs(self):
        def fn():
            return dict({chr(65 + i) : i for i in range(4)}, foo=2)

        self.checkScript(fn, ())

    def test_dict_keyword_with_empty_dict_comprehension(self):
        def fn():
            return dict({})

        self.checkScript(fn, ())

    def test_dict_keyword_is_correctly_typed(self):
        def fn():
            x: Dict[str, int] = dict()
            x["foo"] = 1
            return x

        self.checkScript(fn, ())

    def test_dict_keyword_with_mismatched_annotations(self):
        # TODO: This fails during function schema matching, so the error
        # message is not very informative to the user. Change logic so
        # that the error is thrown at a different time?
        err_msg = "Arguments for call are not valid"
        highlight_msg = "dict([(\"foo\", 1), (\"bar\", 2), (\"baz\", 3"
        with self.assertRaisesRegexWithHighlight(RuntimeError, err_msg, highlight_msg):
            @torch.jit.script
            def fn():
                x: Dict[int, str] = dict([("foo", 1), ("bar", 2), ("baz", 3)])    # noqa: C406
                return x

    def test_dict_keyword_with_nested_call(self):
        def fn():
            return dict(dict(foo=1, bar=2, baz=3))

        self.checkScript(fn, ())

    def test_dict_keyword_with_previously_declared_variable(self):
        def fn():
            d = {"foo" : 1, "bar" : 2}
            return dict(d)

        self.checkScript(fn, ())

    def test_dict_keyword_with_previously_declared_variable_and_kwargs(self):
        def fn():
            d = {"foo" : 1, "bar" : 2}
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

        args_left_float = [[1., 8., 8.], [2., 1., 1.], [], [2.], [1.], [1., 2., 3.]]
        args_right_float = [[2., 1., 1.], [1., 8., 8.], [], [1.], [], [1., 2.]]
        run_tests(jit_min_list_float, args_left_float, args_right_float)

        args_left_bool = [[], [], [], [False], [True], [False, True], [True, True],
                          [False, False, False], [False, False, True]]
        args_right_bool = [[], [False], [True], [True], [False], [True, True],
                           [False, True], [False, False, True], [False, False, False]]
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

        args_left_float = [[1., 8., 8.], [8., 1., 1.], [], [1.], [], [1., 2.]]
        args_right_float = [[8., 1., 1.], [1., 8., 8.], [], [2.], [1.], [1., 2., 3.]]
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

        self.checkScriptRaisesRegex(bad_index, (), Exception,
                                    "list index out of range")

        def bad_negative_index():
            a = [1, 2, 3]
            return a[-5]

        self.checkScriptRaisesRegex(bad_negative_index, (), Exception,
                                    "list index out of range")

    def test_list_len(self):
        def func():
            a = [1, 2, 3]
            return len(a) == 3

        self.checkScript(func, ())

        def func2():
            a = []
            return len(a) == 0

        self.checkScript(func2, ())

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
            test_invalid_list_equality,
            (),
            RuntimeError,
            "Boolean value of Tensor")

    def test_list_sort(self):
        template = dedent('''
        def func():
            li_1 = {list_create}
            li_2 = {list_create}
            li_3 = {list_create}
            li_1.sort()
            li_2.sort(reverse=True)
            li_4 = sorted(li_3)
            return li_1, li_2, li_3, li_4
        ''')

        lists = ["[]", "[1, 3, 2]", "[True, False, True]", "[1.2, .2, 3.2]",
                 "[torch.tensor(1.0), torch.tensor(0.2), torch.tensor(0.5)]",
                 "[torch.tensor(5), torch.tensor(-2), torch.tensor(4)]"]
        for li in lists:
            code = template.format(list_create=li)
            scope = {}
            exec(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            t1 = cu.func()
            t2 = scope['func']()
            self.assertEqual(t1, t2)

        def test_fail(x: List[Tensor]) -> List[Tensor]:
            x.sort()
            return x

        self.checkScriptRaisesRegex(test_fail, (([torch.zeros([2]), torch.zeros([2])],)), Exception,
                                    "Boolean value of Tensor with more than one value")

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
        with self.assertRaisesRegexWithHighlight(RuntimeError, "index 4 is out of bounds for dimension 0 with size 3",
                                                               "x[[4], :, :]"):
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

        with self.assertRaisesRegexWithHighlight(RuntimeError, "pop from empty list", "a.pop"):
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

        with self.assertRaisesRegexWithHighlight(RuntimeError, "x not in list", "a.remove"):
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

        with self.assertRaisesRegexWithHighlight(RuntimeError, "'5' is not in list", "a.index"):
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

        with self.assertRaisesRegexWithHighlight(RuntimeError, "is not in list", "a.index"):
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
            for r in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
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
                self.checkScriptRaisesRegex(fn, (li,), Exception, "arg is an empty sequence")
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
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.float).transpose(0, 1),))

        self.checkScript(to_list_float_0D, (torch.randn(5, dtype=torch.double)[0],))
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.double),))
        self.checkScript(to_list_float_2D, (torch.randn(5, 6, dtype=torch.double),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.double),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.double).transpose(0, 1),))

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
        self.checkScript(to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cfloat),))
        self.checkScript(to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cfloat).transpose(0, 1),))

        self.checkScript(to_list_complex_0D, (torch.randn(5, dtype=torch.cdouble)[0],))
        self.checkScript(to_list_complex_1D, (torch.randn(5, dtype=torch.cdouble),))
        self.checkScript(to_list_complex_2D, (torch.randn(5, 6, dtype=torch.cdouble),))
        self.checkScript(to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cdouble),))
        self.checkScript(to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cdouble).transpose(0, 1),))

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

        def to_list_type_annotation_incorrect_scalar_type(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"Expected type hint for result of tolist()",
            "x.tolist("
        ):
            self.checkScript(to_list_missing_type_annotation, (torch.randn(5),))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"Return value was annotated as having type List\[float\] but is actually of type float",
            "return li"
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
            self.checkScript(to_list_type_annotation_wrong_dim, (torch.randn(5, dtype=torch.double),))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Output annotation element type and runtime tensor element type must match",
        ):
            self.checkScript(
                to_list_type_annotation_incorrect_scalar_type,
                (torch.ones(5, dtype=torch.long),),
            )


    def test_to_list_gpu(self):
        """GPU tests for Tensor.tolist() function."""
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            self.skipTest("CUDA is not available")

        def to_list_bool_1D(x: torch.Tensor) -> List[bool]:
            li = torch.jit.annotate(List[bool], x.tolist())
            return li

        def to_list_int_1D(x: torch.Tensor) -> List[int]:
            li = torch.jit.annotate(List[int], x.tolist())
            return li

        def to_list_float_1D(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        self.checkScript(to_list_bool_1D, (torch.tensor(
            [True, False, True, False], dtype=torch.bool).cuda(),))
        self.checkScript(to_list_int_1D, (torch.tensor(
            [1, 2, 3, 4], dtype=torch.long).cuda(),))
        self.checkScript(to_list_float_1D, (torch.randn(
            5, dtype=torch.double).cuda(),))

    def test_no_element_type_annotation(self):
        def fn_with_comment(x: torch.Tensor) -> List:
            a: List = x.tolist()
            return a

        def annotated_fn(x: torch.Tensor) -> List:
            a: List = x.tolist()
            return a

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use List without a contained type"):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use List without a contained type"):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use List without a contained type"):
            torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use List without a contained type"):
            torch.jit.script(annotated_fn)

    def test_list_none(self):
        with self.assertRaisesRegex(RuntimeError, "Can not create ListType with None type"):
            x = torch._C.ListType(None)

    def test_list_unification_hint(self):
        with self.assertRaisesRegex(RuntimeError, "Expected a List type hint"):
            @torch.jit.script
            def x():
                b : int = [2, 3]
                return b


class TestDict(JitTestCase):
    def dict(self):
        return {u'a': torch.ones(1), u'b': torch.ones(1) + 1, u'c': torch.ones(1) + 2}

    def dict2(self):
        return {'x': torch.ones(1) + 100, 'y': torch.ones(1) + 101, 'z': torch.ones(1) + 102}

    def dict_bool(self):
        return {True: 1}

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

    def test_del(self):
        def inputs():
            return {'hi': 2, 'bye': 3}

        def fn(x: Dict[str, int]) -> Dict[str, int]:
            del x['hi']
            return x

        python_out = fn(inputs())
        # checkScript reuses the same object, but here it's being mutated so do
        # it manually
        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(fn)))
        self.assertEqual(cu.fn(inputs()), python_out)
        self.assertEqual(torch.jit.script(fn)(inputs()), python_out)
        with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", "x['hi']"):
            self.checkScript(fn, [{}])

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

        self.assertTrue(set(specialized_list()) == set([1, 2, 3]))

    def test_values(self):
        @torch.jit.script
        def values(x: Dict[str, Tensor]) -> List[Tensor]:
            return list(x.values())

        the_dict = self.dict()
        self.assertEqual(set(values(the_dict)), set(the_dict.values()))

    def test_len(self):
        def length(x: Dict[str, Tensor]) -> int:
            return len(x)

        self.checkScript(length, (self.dict(),))

    def test_copy(self):
        def func(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
            return x.copy()

        self.checkScript(func, (self.dict(),))

    def test_items(self):
        def func(x: Dict[str, Tensor]) -> List[Tuple[str, Tensor]]:
            return x.items()

        # The value returned by Python is in arbitrary order, so we can't use
        # checkScript
        scripted_func = torch.jit.script(func)

        eager_out = (func(self.dict()))
        script_out = (scripted_func(self.dict()))

        self.assertEqual(len(eager_out), len(script_out))
        for item in eager_out:
            self.assertTrue(item in script_out)

    def test_pop(self):
        def pop(x: Dict[str, Tensor], key: str) -> Tuple[Tensor, Dict[str, Tensor]]:
            return x.pop(key), x

        # checkScript doesn't copy the inputs, so we can't use it since this mutates
        # the dict
        def tester(fn, *args):
            eager_out = fn(self.dict(), *args)
            script_out = torch.jit.script(fn)(self.dict(), *args)
            self.assertEqual(eager_out, script_out)

        tester(pop, 'a')

        with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", "x.pop"):
            torch.jit.script(pop)(self.dict(), 'x')


        def default_pop(x: Dict[str, Tensor], key: str, default: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
            return x.pop(key, default), x

        tester(default_pop, 'a', torch.randn(2, 2))
        tester(default_pop, 'x', torch.randn(2, 2))

    def test_setdefault(self):
        def setdefault(x: Dict[str, Tensor], key: str, default: Tensor) -> Dict[str, Tensor]:
            x.setdefault(key, default)
            return x

        self.checkScript(setdefault, (self.dict(), 'a', torch.randn(2, 2)))
        self.checkScript(setdefault, (self.dict(), 'nonexistant', torch.randn(2, 2)))

    def test_update(self):
        def update(a: Dict[str, Tensor], b: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
            a.update(b)
            return a, b

        self.checkScript(update, (self.dict(), self.dict()))
        self.checkScript(update, (self.dict(), self.dict2()))

    def test_update_existing_key(self):
        def foo() -> Dict[str, int]:
            a: Dict[str, int] = {}
            for i in range(3):
                a.update({'a': i})
            return a

        self.checkScript(foo, ())

    def test_aug_assign(self):
        def aug_assign_dict_tensor(a: Dict[str, Tensor]) -> Dict[str, Tensor]:
            a['a'] += 1
            a['b'] -= 12
            a['c'] *= 122
            a['c'] /= 2
            a['c'] %= 2
            return a

        def aug_assign_dict_prim(a: Dict[str, float]) -> Dict[str, float]:
            a['a'] += 3.4
            a['b'] -= 2.4
            a['c'] *= 3.0
            a['c'] /= 2.0
            a['c'] %= 2.0
            return a

        self.checkScript(aug_assign_dict_tensor, (self.dict(),))
        self.checkScript(aug_assign_dict_prim, ({'a': 3.0, 'b': 2.0, 'c': 4.0},))

    def test_popitem(self):
        @torch.jit.script
        def popitem(x: Dict[str, Tensor]) -> Tuple[Tuple[str, Tensor], Dict[str, Tensor]]:
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

    def test_clear(self):
        def clear(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
            x.clear()
            return x

        self.checkScript(clear, (self.dict(),))

    def test_get(self):
        def get(x: Dict[str, Tensor], key: str) -> Optional[Tensor]:
            return x.get(key)

        self.checkScript(get, (self.dict(), 'a'))
        self.checkScript(get, (self.dict(), "doesn't exist"))

        def get_default(x: Dict[str, Tensor], key: str) -> Optional[Tensor]:
            return x.get(key, torch.randn(2, 2))

        self.checkScript(get, (self.dict(), 'a'))
        self.checkScript(get, (self.dict(), "doesn't exist"))

    def test_get_boolkey(self):
        def get(x: Dict[bool, int], key: bool) -> Optional[int]:
            return x.get(key)

        self.checkScript(get, (self.dict_bool(), True))
        self.checkScript(get, (self.dict_bool(), False))

        def get_default(x: Dict[bool, int], key: bool) -> int:
            return x.get(key, 42)

        self.checkScript(get_default, (self.dict_bool(), True))
        self.checkScript(get_default, (self.dict_bool(), False))

    def test_basic(self):
        def simple(x: Dict[str, int]) -> Dict[str, int]:
            return x

        self.checkScript(simple, ({'item': 20, 'other_item': 120},))

        def index(x: Dict[str, int]) -> int:
            return x['item']

        self.checkScript(index, ({'item': 20, 'other_item': 120},))

        def type_default() -> Dict[str, Tensor]:
            return {}

        self.checkScript(type_default, ())

        @torch.jit.script
        def missing_index(x: Dict[str, int]) -> int:
            return x['dne']

        with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", "x['d"):
            missing_index({'item': 20, 'other_item': 120})

        code = dedent('''
            def literal1():
                return torch.jit.annotate(Dict[int, float], {})
            def literal2():
                return torch.jit.annotate(Dict[int, float], {10: 1.2})
        ''')
        cu = torch.jit.CompilationUnit(code)
        self.assertEqual({}, cu.literal1())
        self.assertEqual({10: 1.2}, cu.literal2())

        cu = torch.jit.CompilationUnit(dedent('''
            def literal3():
                return torch.jit.annotate(Dict[int, float], {10: 1.2, 11: 1.3})
        '''))
        self.assertEqual({10: 1.2, 11: 1.3}, cu.literal3())

        def list_of_dicts() -> List[Dict[str, Tensor]]:
            return [{'word': torch.ones(2) + 3}, {'other word': torch.ones(1) + 2}]

        self.checkScript(list_of_dicts, ())

    def test_mutability(self):
        @torch.jit.script
        def fn() -> Dict[str, int]:
            a = torch.jit.annotate(Dict[str, int], {})
            a['ok'] = 10
            return a

        self.assertEqual(fn(), {'ok': 10})

    def test_key_type(self):
        with self.assertRaisesRegexWithHighlight(RuntimeError, "but instead found type", "a[None]"):
            @torch.jit.script
            def fn(a: Dict[str, int]) -> int:
                return a[None]

    def test_loop(self):
        @torch.jit.script
        def fn(x: int) -> Dict[str, int]:
            a = torch.jit.annotate(Dict[str, int], {})
            for i in range(x):
                a['ok'] = i
            return a

        self.assertEqual(fn(10), {'ok': 9})

    def test_view(self):
        def fn(x, y):
            l = {"a": x}
            x_view = l["a"]
            a = x + x
            x_view.add_(y)
            b = x + x
            return a == b
        self.checkScript(fn, (torch.rand(2, 3), torch.rand(2, 3)))

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

        with self.assertRaisesRegexWithHighlight(RuntimeError, "is actually of type Optional", "return x.get(y"):
            @torch.jit.script
            def bad_types(x: Dict[int, int], y: int) -> int:
                return x.get(y)  # noqa: T484

    def test_dict_to_python(self):
        @torch.jit.ignore
        def python_lookup(my_dict: Dict[str, int], keys: List[str]) -> List[int]:
            return [my_dict[k] for k in keys]

        def fn(my_dict: Dict[str, int], keys: List[str]) -> List[int]:
            return python_lookup(my_dict, keys)

        a_dict = {'a': torch.ones(1), 'b': torch.ones(1) + 1, 'c': torch.ones(1) + 2}
        self.checkScript(fn, (a_dict, ('a', 'c')))

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
            a = dict()
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
            a = dict()
            a[1] = 2
            return a

        with self.assertRaisesRegexWithHighlight(Exception, "Arguments for call are not", "a[1] = 2"):
            torch.jit.script(test_dict_error)

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

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Dict without contained types"):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Dict without contained types"):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Dict without contained types"):
            m = torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Dict without contained types"):
            m = torch.jit.script(annotated_fn)

    def test_dict_preserves_order(self):
        def dict_ordering():
            a : Dict[int, int] = {}
            for i in range(1000):
                a[i] = i + 1
            return a

        self.checkScript(dict_ordering, ())
        di = torch.jit.script(dict_ordering)()
        res = list(di.items())
        for i in range(1000):
            key, value = res[i]
            self.assertTrue(key == i and value == i + 1)

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
                for _id, config in self.configs.items():
                    x += config.size
                return x

        s = torch.jit.script(MyMod({0: Config(size=16)}))

    def test_namedtuple_resolution(self):
        class TheType(NamedTuple):
            t: int

        class MyModule(types.ModuleType):
            def __init__(self):
                super(MyModule, self).__init__('MyModule')

            def __getattr__(self, attr):
                return TheType

        some_module = MyModule()

        def fn() -> some_module.Type:
            return some_module.Type(1)

        self.checkScript(fn, [])

    def test_namedtuple_slice_unpack(self):
        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int]

        @torch.jit.script
        def foo(a : int, b : float, c : List[int]):
            tup = MyCoolNamedTuple(a, b, c)
            my_a, my_b, my_c = tup
            return tup[:1], my_a, my_c

        self.assertEqual(foo(3, 3.5, [6]), ((3,), 3, [6]))

    def test_namedtuple_lower(self):
        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int]

        @torch.jit.script
        def foo(a : int):
            tup = MyCoolNamedTuple(a, 3.14, [9])
            return tup

        FileCheck().check('TupleConstruct').run(foo.graph)
        torch._C._jit_pass_lower_all_tuples(foo.graph)
        FileCheck().check_not('TupleConstruct').run(foo.graph)

    def test_namedtuple_type_annotation(self):
        global MyCoolNamedTuple  # see [local resolution in python]

        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int]

        @torch.jit.script
        def foo(x : MyCoolNamedTuple) -> MyCoolNamedTuple:
            return x

        mnt = MyCoolNamedTuple(42, 420.0, [666])
        self.assertEqual(foo(mnt), mnt)

    def test_namedtuple_wrong_types(self):
        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int]

        with self.assertRaisesRegex(RuntimeError, "Expected a value of type 'int' for argument 'a'"
                                                  " but instead found type 'str'"):
            @torch.jit.script
            def foo():
                tup = MyCoolNamedTuple('foo', 'bar', 'baz')
                return tup

    def test_namedtuple_kwarg_construct(self):
        class MyCoolNamedTuple(NamedTuple):
            a : int
            b : float
            c : List[int]

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
            a : int
            b : float
            c : List[int]

        class MyMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self):
                return MyCoolNamedTuple(3, 3.5, [3, 4, 5])

        mm = MyMod()
        mm.save('foo.zip')
        torch.testing._internal.jit_utils.clear_class_registry()
        loaded = torch.jit.load('foo.zip')

        out = mm()
        out_loaded = loaded()

        for name in ['a', 'b', 'c']:
            self.assertEqual(getattr(out_loaded, name), getattr(out, name))
