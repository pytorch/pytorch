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

    # Generate the verbose error message that we expect to be thrown at
    # FunctionSchema::formatTypeMismatchMsg
    def _generate_formatTypeMismatchMsg_error_message(self, declaration: str,
        arg_pos: int, actual_type: str, actual_value: str) -> str:
        actual_value = actual_value.replace("\"", "'")
        if actual_type == "str":
            actual_value = "'" + actual_value + "'"
        fn_name, _ = declaration.split("(", 1)
        arg_str, _, _ = declaration.rsplit(")", 2)
        arg_str = arg_str[len(fn_name)+1:]
        arg_list = self._get_all_args(arg_str)
        expected_type = arg_list[arg_pos][0]
        arg_name = arg_list[arg_pos][1]
        res = "\n".join([f"{fn_name}() Expected a value of type "
                f"'{expected_type}' for argument '{arg_name}' but instead "
                f"found type '{actual_type}'.", f"Position: {arg_pos}",
                f"Value: {actual_value}", f"Declaration: {declaration}", "Cast "
                f"error details: Expected a member of {expected_type} but "
                f"instead found type {actual_type}"])
        return re.escape(res)

    # Return a list of (TYPE-NAME, ARG-NAME) tuples
    def _get_all_args(self, arg_str: str) -> List[str]:
        res = []
        s, bracket = 0, 0
        for i, c in enumerate(arg_str):
            if c == "[":
                bracket += 1
            elif c == "]":
                bracket -= 1
            elif c == "," and bracket == 0:
                arg = arg_str[s:i]
                s = i + 1
                t, n = arg.rsplit(" ", 1)
                res.append((t, n))
        if not res:
            t, n = arg_str.rsplit(" ", 1)
            res.append((t, n))
        return res

    def test_union_with_scalar_values(self):
        def fn(x: Union[int, float]) -> str:
            return "foo"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (1.0,))

        scripted_fn = torch.jit.script(fn)

        msg = self._generate_formatTypeMismatchMsg_error_message(
            "fn(Union[int, float] x) -> (str)", 0, "str", "1")
        with self.assertRaisesRegex(RuntimeError, msg):
            scripted_fn("1")

    def test_union_with_collections(self):
        def fn(x: Union[Dict[str, int], List[int]]) -> str:
            return "foo"

        self.checkScript(fn, ({"foo":1, "bar":2, "baz":3},))
        self.checkScript(fn, ([1, 2, 3],))

        scripted_fn = torch.jit.script(fn)

        msg = self._generate_formatTypeMismatchMsg_error_message(
            "fn(Union[Dict[str, int], List[int]] x) -> (str)", 0, "Dict[str, str]", "{\"foo\": \"bar\", \"baz\": \"qux\"}")
        with self.assertRaisesRegex(RuntimeError, msg):
            scripted_fn({"foo":"bar", "baz":"qux"})

        msg = self._generate_formatTypeMismatchMsg_error_message(
            "fn(Union[Dict[str, int], List[int]] x) -> (str)", 0, "List[str]", "[\"foo\", \"bar\", \"baz\"]")
        with self.assertRaisesRegex(RuntimeError, msg):
            scripted_fn(["foo", "bar", "baz"])

        msg = self._generate_formatTypeMismatchMsg_error_message(
            "fn(Union[Dict[str, int], List[int]] x) -> (str)", 0, "str", "1")
        with self.assertRaisesRegex(RuntimeError, msg):
            scripted_fn("1")

    # TODO: problems with enum (must be global?)
    #def test_union_with_enum(self):

    #    global Color

    #    class Color(Enum):
    #        RED = 1
    #        GREEN = 2

    #    def fn(x: Union[Color, str]) -> str:
    #        return "foo"

    #    self.checkScript(fn, (Color.RED,))
    #    self.checkScript(fn, ("red",))

    #    scripted_fn = torch.jit.script(fn)

    #    with self.assertRaisesRegexWithHighlight(RuntimeError, "has no "
    #                                            "attribute 'PURPLE'",
    #                                            "Color.PURPLE"):
    #        scripted_fn(Color.PURPLE)

    #    msg = self._generate_formatTypeMismatchMsg_error_message(
    #        "fn(Union[Color, str] x) -> (str)", 0, "int", "1")
    #    with self.assertRaisesRegex(RuntimeError, msg):
    #        scripted_fn(1)

    def test_union_in_class_constructor(self):

        @torch.jit.script
        class A(object):
            def __init__(self, x: Union[int, str]) -> None:
                self.x = x

        def fn(x: Union[str, int]) -> A:
            return A(x)

        self.assertEqual(fn("foo").x, "foo")
        self.assertEqual(fn(1).x, 1)

        scripted_fn = torch.jit.script(fn)

        msg = self._generate_formatTypeMismatchMsg_error_message(
            "fn(Union[str, int] x) -> (__torch__.jit.test_union.A)", 0, "List[str]", "[\"foo\", \"bar\", \"baz\"]")
        with self.assertRaisesRegex(RuntimeError, msg):
            scripted_fn(["foo", "bar", "baz"])

    #def test_union_return_type(self):
    #    def fn(x: int) -> Union[int, str]:
    #        if x % 2:
    #            return 1
    #        else:
    #            return "foo"

    #    self.checkScript(fn, (1,))
    #    self.checkScript(fn, (8,))

    #def test_unions_of_unions_are_flattened(self):
    #    @torch.jit.script
    #    def fn_with_union(x: Union[int, str, float]) -> str:
    #        return "foo"

    #    @torch.jit.script
    #    def fn_with_nested_union(x: Union[Union[int, str], float]) -> str:
    #        return "foo"

    #    self.assertEqual(fn_with_union.graph, fn_with_nested_union.graph)

    #def test_unions_of_a_single_argument_vanish(self):
    #    @torch.jit.script
    #    def fn_with_int(x: int) -> str:
    #        return "foo"

    #    @torch.jit.script
    #    def fn_with_union_of_int(x: Union[int]) -> str:
    #        return "foo"

    #    self.assertEqual(fn_with_int.graph, fn_with_union_of_int.graph)

    #def test_union_redundant_arguments_are_skipped(self):
    #    @torch.jit.script
    #    def fn(x: Union[int, str]) -> str:
    #        return "foo"

    #    @torch.jit.script
    #    def redundant_fn(x: Union[int, str, int]) -> str:
    #        return "foo"

    #    self.assertEqual(fn.graph, redundant_fn.graph)

    #def test_union_argument_order_is_ignored(self):
    #    @torch.jit.script
    #    def fn1(x: Union[int, str]) -> str:
    #        return "foo"

    #    @torch.jit.script
    #    def fn2(x: Union[str, int]) -> str:
    #        return "foo"

    #    self.assertEqual(fn1.graph, fn2.graph)

    #def test_union_cannot_be_subclassed(self):
    #    with self.assertRaisesRegex(RuntimeError, "poop"):
    #        @torch.jit.script
    #        class A(Union):
    #            def __init__(self, x):
    #                self.x = x

    #def test_union_cannot_be_instantiated(self):
    #    @torch.jit.script
    #    def fn_with_str_annotation() -> str:
    #        x: str = "foo"
    #        return x

    #    @torch.jit.script
    #    def fn_with_union_annotation() -> str:
    #        x: Union[int, str] = "foo"
    #        return x

    #    fn_with_str_annotation()        # This should work fine

    #    with self.assertRaisesRegex(RuntimeError, "poop"):
    #        fn_with_union_annotation()

    #def test_union_argument_cannot_be_doubly_subscripted(self):
    #    @torch.jit.script
    #    def fn(x: Union[int][str]) -> str:
    #        return "foo"
    #    with self.assertRaisesRegex(RuntimeError, "poop"):
    #        fn("bar")

    #def test_union_return_value_cannot_be_doubly_subscripted(self):
    #    @torch.jit.script
    #    def fn() -> Union[int][str]:
    #        return "foo"
    #    with self.assertRaisesRegex(RuntimeError, "poop"):
    #        fn()

    ## TODO: The following tests cannot be implemented because they rely
    ## on constructs that are not yet implemented in TorchScript:
    ##   - Union accepts child classes in place of parent classes
    ##     (relies on polymorphism)
    ##   - Union accepts aliased types (relies on aliased types)
