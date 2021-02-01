import os
import re
import sys

import torch
from enum import Enum
from typing import Dict, List, Union

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestUnion(JitTestCase):

    def _generate_formatTypeMismatchMsg_error_message(self,
                                                      schema_str: str,
                                                      arg_str: str,
                                                      arg_pos: int,
                                                      actual_type: str,
                                                      actual_value: str) -> str:
        """
        Generate the verbose error message that we expect to be thrown at
        FunctionSchema::formatTypeMismatchMsg

        `schema_str` - a textual representation of the FunctionSchema
                       (this is what you'd get if you called
                        `operator<<` on a FunctionSchema object)
        `arg_str` -
        """
        actual_value = actual_value.replace("\"", "'")
        if actual_type == "str":
            actual_value = "'" + actual_value + "'"
        fn_name, _ = schema_str.split("(", 1)
        arg_list = self._get_all_args(arg_str)
        for i, arg in enumerate(arg_list):
            arg = (arg[0].replace("(", "["), arg[1])
            arg_list[i] = (arg[0].replace(")", "]"), arg[1])
        expected_type = arg_list[arg_pos][0]
        arg_name = arg_list[arg_pos][1]
        res = "\n".join([f"{fn_name}() Expected a value of type "
                         f"'{expected_type}' for argument '{arg_name}' but instead "
                         f"found type '{actual_type}'.", f"Position: {arg_pos}",
                         f"Value: {actual_value}", f"Declaration: {schema_str}", "Cast "
                         f"error details: Expected a member of {expected_type} but "
                         f"instead found type {actual_type}"])
        return re.escape(res)

    # Return a list of (TYPE-NAME, ARG-NAME) tuples
    def _get_all_args(self, arg_str: str) -> List[str]:
        res = []
        s, brackets = 0, 0
        for i, c in enumerate(arg_str):
            if c == "[":
                brackets += 1
            elif c == "]":
                brackets -= 1
            elif c == "," and brackets == 0:
                arg = arg_str[s:i]
                s = i + 1
                t, n = arg.rsplit(" ", 1)
                res.append((t, n))
        if not res:
            t, n = arg_str.rsplit(" ", 1)
            res.append((t, n))
        return res

    # Return a string representing the graph input
    def _input_str(self, fn: 'torch.jit.ScriptFunction') -> str:
        return fn.graph.str().partition(':\n')[0]

    def test_union_with_scalar_values(self):
        def fn(x: Union[int, float]) -> str:
            return "foo"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (1.0,))

        scripted_fn = torch.jit.script(fn)

        msg = self._generate_formatTypeMismatchMsg_error_message(
            "fn(Union(int, float) x) -> (str)", "Union[int, float] x", 0,
            "str", "1")
        with self.assertRaisesRegex(RuntimeError, msg):
            scripted_fn("1")

    def test_union_with_collections(self):
        def fn(x: Union[Dict[str, int], List[int]]) -> str:
            return "foo"

        self.checkScript(fn, ({"foo": 1, "bar": 2, "baz": 3},))
        self.checkScript(fn, ([1, 2, 3],))

        scripted_fn = torch.jit.script(fn)

        msg = self._generate_formatTypeMismatchMsg_error_message(
            "fn(Union(Dict(str, int), int[]) x) -> (str)",
            "Union[Dict[str, int], List[int]] x", 0,
            "Dict[str, str]", "{\"foo\": \"bar\", \"baz\": \"qux\"}")
        with self.assertRaisesRegex(RuntimeError, msg):
            scripted_fn({"foo": "bar", "baz": "qux"})

        msg = self._generate_formatTypeMismatchMsg_error_message(
            "fn(Union(Dict(str, int), int[]) x) -> (str)",
            "Union[Dict[str, int], List[int]] x", 0,
            "List[str]", "[\"foo\", \"bar\", \"baz\"]")
        with self.assertRaisesRegex(RuntimeError, msg):
            scripted_fn(["foo", "bar", "baz"])

        msg = self._generate_formatTypeMismatchMsg_error_message(
            "fn(Union(Dict(str, int), int[]) x) -> (str)",
            "Union[Dict[str, int], List[int]] x", 0, "str", "1")
        with self.assertRaisesRegex(RuntimeError, msg):
            scripted_fn("1")

    def test_union_with_enum(self):

        global Color

        class Color(Enum):
            RED = 1
            GREEN = 2

        def fn(x: Union[str, Color]) -> str:
            return "foo"

        self.checkScript(fn, (Color.RED,))
        self.checkScript(fn, ("red",))

        scripted_fn = torch.jit.script(fn)

        msg = self._generate_formatTypeMismatchMsg_error_message(
            "fn(Union(str, Enum<__torch__.jit.test_union.Color>) x) -> (str)",
            "Union[str, __torch__.jit.test_union.Color] x", 0, "int", "1")
        with self.assertRaisesRegex(RuntimeError, msg):
            scripted_fn(1)

    def test_union_in_class_constructor(self):

        @torch.jit.script
        class A(object):    # noqa B903
            def __init__(self, x: Union[int, str]) -> None:
                self.x = x

        def fn(x: Union[str, int]) -> A:
            return A(x)

        self.assertEqual(fn("foo").x, "foo")
        self.assertEqual(fn(1).x, 1)

        scripted_fn = torch.jit.script(fn)

        msg = self._generate_formatTypeMismatchMsg_error_message(
            "fn(Union(str, int) x) -> (__torch__.jit.test_union.A)",
            "Union[str, int] x", 0,
            "List[str]", "[\"foo\", \"bar\", \"baz\"]")
        with self.assertRaisesRegex(RuntimeError, msg):
            scripted_fn(["foo", "bar", "baz"])

    def test_union_return_type(self):
        def fn(x: int) -> Union[int, str]:
            return "foo"

        self.checkScript(fn, (1,))

    def test_union_return_type_with_same_type_branching(self):
        def fn(x: int) -> Union[int, str]:
            if x % 2:
                return "foo"
            else:
                return "bar"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (8,))

    def test_union_return_type_with_different_type_branching(self):
        def fn(x: int) -> Union[int, str]:
            if x % 2:
                return "foo"
            else:
                return 1

        self.checkScript(fn, (1,))
        self.checkScript(fn, (8,))

    def test_union_as_annotation(self):
        def fn() -> Union[int, str]:
            x: Union[int, str] = "foo"
            return x

        self.checkScript(fn, ())

    def test_union_as_annotation_in_typed_container(self):
        def fn() -> None:
            l: List[Union[int, str]] = []
            u1: Union[int, str] = "foo"
            u2: Union[int, str] = 1
            l.append(u1)
            l.append(u2)

        self.checkScript(fn, ())

    def test_union_variable_can_be_reassigned(self):
        def fn() -> Union[int, str]:
            x: Union[int, str] = "foo"
            i: int = 1
            s: str = "bar"
            x = i
            x = s
            return x

        self.checkScript(fn, ())

    def test_union_does_not_replace_existing_annotated_type(self):
        def fn():
            x: List[int] = [1, 2, 3]
            x.append("foo")
            return x

        with self.assertRaisesRegex(RuntimeError, "Could not match type str"):
            scripted_fn = torch.jit.script(fn)
            scripted_fn()

    def test_union_does_not_replace_existing_union_annotated_type(self):
        def fn():
            x: List[Union[int, str]] = [1, "foo", 3]
            x.append(2.0)
            return x

        with self.assertRaisesRegex(RuntimeError, "Could not match type float"):
            scripted_fn = torch.jit.script(fn)
            scripted_fn()

    def test_union_does_not_replace_existing_annotated_type_with_empty_container(self):
        def fn():
            x: List[int] = []
            x.append("foo")
            return x

        with self.assertRaisesRegex(RuntimeError, "Could not match type str"):
            scripted_fn = torch.jit.script(fn)
            scripted_fn()

    def test_unions_of_unions_are_flattened(self):
        @torch.jit.script
        def fn_with_union(x: Union[int, str, float]) -> str:
            return "foo"

        @torch.jit.script
        def fn_with_nested_union(x: Union[Union[int, str], float]) -> str:
            return "foo"

        self.assertEqual(self._input_str(fn_with_union),
                         self._input_str(fn_with_nested_union))

    def test_unions_of_a_single_argument_vanish(self):
        @torch.jit.script
        def fn_with_int(x: int) -> str:
            return "foo"

        @torch.jit.script
        def fn_with_union_of_int(x: Union[int]) -> str:
            return "foo"

        self.assertEqual(self._input_str(fn_with_int),
                         self._input_str(fn_with_union_of_int))

    def test_union_redundant_arguments_are_skipped(self):
        @torch.jit.script
        def fn(x: Union[int, str]) -> str:
            return "foo"

        @torch.jit.script
        def redundant_fn(x: Union[int, str, int]) -> str:
            return "foo"

        self.assertEqual(self._input_str(fn), self._input_str(redundant_fn))

    def test_union_argument_order_is_ignored(self):
        @torch.jit.script
        def fn1(x: Union[int, str]) -> str:
            return "foo"

        @torch.jit.script
        def fn2(x: Union[str, int]) -> str:
            return "foo"

        self.assertEqual(self._input_str(fn1), self._input_str(fn2))

    def test_union_T_None_is_equivalent_to_optional_T(self):
        def fn(x: Union[int, None]) -> str:
            return "foo"

        self.checkScript(fn, (1,))

    def test_union_subclasses_larger_union(self):
        def fn() -> Union[int, str, torch.Tensor]:
            x: Union[int, str] = "foo"
            return x

        self.checkScript(fn, ())

    def test_union_with_type_coalescing(self):
        def fn(x: Union[List[int], int]) -> int:
            lst = [1, 2, 3]
            if isinstance(x, int):
                x = lst
            return lst[0]

        self.checkScript(fn, (1,))
        self.checkScript(fn, ([4, 5, 6],))

    # TODO: The following tests cannot be implemented because they rely
    # on constructs that are not yet implemented in TorchScript:
    #   - Union accepts child classes in place of parent classes
    #     (relies on polymorphism)
    #   - Union accepts aliased types (relies on aliased types)
    #   - IN-DEPTH Union subtyping (TorchScript has a fairly flat type
    #     hierarchy so far)
