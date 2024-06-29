# Owner(s): ["oncall: jit"]

import io
import os
import sys
import unittest
from enum import Enum
from textwrap import dedent
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, make_global


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


@unittest.skipIf(sys.version_info < (3, 10), "Requires Python 3.10")
class TestUnion(JitTestCase):
    """
    This class tests the functionality of `Union`.

    Note: It's important to be able to refine the type of a `Union` to
    one of its internal types. Currently, there are differences in the
    way Python expects `isinstance` checks and the way TorchScript
    expects `isinstance` checks. This means that we can't use
    `checkScript` in our test cases because either the eager mode or the
    script mode wouldn't run! So, some test cases have separate but
    equivalent functions to emulate `checkScript`.
    """

    def test_check_union_annotation(self):
        def test_func(a: int | float, b: Optional[int]):
            return 0

        scripted_func = torch.jit.script(test_func)
        graph_rep = str(scripted_func.graph)
        code_rep = str(scripted_func.code)
        # TS graph IR for Union should be annotated as Union()
        FileCheck().check("Union(").check("int?").run(graph_rep)
        # Serialized code for Union should be annotated as Union[]
        FileCheck().check("Union[").check("Optional[int]").run(code_rep)
        self.checkScript(test_func, (5, 6))
        # this shouldn't error out
        torch._C.parse_ir(str(scripted_func.graph))

    def test_union_with_scalar_values(self):
        def fn(x: int | float) -> str:
            return "foo"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (1.0,))

        scripted = torch.jit.script(fn)

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[float, int\] but "
            "instead found type str",
        ):
            scripted("1")

    def test_union_with_collections(self):
        def fn(x: Dict[str, int] | List[int]) -> str:
            return "foo"

        self.checkScript(fn, ({"foo": 1, "bar": 2, "baz": 3},))
        self.checkScript(fn, ([1, 2, 3],))

        scripted = torch.jit.script(fn)

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[List\[int\], Dict\[str, "
            r"int\]\] but instead found type "
            r"Dict\[str, str\]",
        ):
            scripted({"foo": "bar", "baz": "qux"})

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[List\[int\], Dict\[str, "
            r"int\]\] but instead found type "
            r"List\[str\]",
        ):
            scripted(["foo", "bar", "baz"])

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[List\[int\], Dict\[str, "
            r"int\]\] but instead found type "
            "str",
        ):
            scripted("1")

    def test_union_with_enum(self):
        class Color(Enum):
            RED = 1
            GREEN = 2

        make_global(Color)

        def fn(x: str | Color) -> str:
            return "foo"

        self.checkScript(fn, (Color.RED,))
        self.checkScript(fn, ("red",))

        scripted = torch.jit.script(fn)

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[__torch__.jit.test_union_pep604."
            r"Color, str\] but instead found "
            "type int",
        ):
            scripted(1)

    def test_union_in_class_constructor(self):
        @torch.jit.script  # noqa: B903
        class A:  # noqa: B903
            def __init__(self, x: int | str) -> None:
                self.x = x

        def fn(x: str | int) -> A:
            return A(x)

        self.assertEqual(fn("foo").x, "foo")
        self.assertEqual(fn(1).x, 1)

        scripted = torch.jit.script(fn)

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[int, str\] but instead "
            r"found type List\[str\]",
        ):
            scripted(["foo", "bar", "baz"])

    def test_union_return_type(self):
        def fn(x: int) -> int | str:
            return "foo"

        self.checkScript(fn, (1,))

    def test_union_as_annotation(self):
        def fn() -> int | str:
            x: int | str = "foo"
            return x

        self.checkScript(fn, ())

    def test_union_as_annotation_in_typed_container(self):
        def fn() -> None:
            l: List[int | str] = []
            u1: int | str = "foo"
            u2: int | str = 1
            l.append(u1)
            l.append(u2)

        self.checkScript(fn, ())

    def test_union_as_annotation_py2(self):
        def fn():
            # type: () -> int | str
            x: int | str = "foo"
            return x

        self.checkScript(fn, ())

    def test_union_as_internal_tuple_type(self):
        def fn():
            t: Tuple[int | str, int | str] = (1, "foo")
            return t

        self.checkScript(fn, ())

    def test_union_variable_can_be_reassigned(self):
        @torch.jit.script
        def aux1(i: int):
            return int(i**2)

        @torch.jit.script
        def aux2(s: str):
            return s + s

        def fn() -> int | str:
            x: int | str = "foo"
            i: int = 1
            x = i
            y: int = aux1(x)
            z: str = aux2(str(y))
            x = z
            return x

        self.checkScript(fn, ())

    def test_union_does_not_replace_existing_annotated_type(self):
        def fn():
            x: List[int] = [1, 2, 3]
            x.append("foo")
            return x

        with self.assertRaisesRegex(RuntimeError, "Could not match type str"):
            scripted = torch.jit.script(fn)
            scripted()

    def test_union_does_not_replace_existing_annotated_type_union(self):
        def fn():
            x: List[int | str] = [1, "foo", 3]
            x.append(2.0)
            return x

        with self.assertRaisesRegex(RuntimeError, "Could not match type float"):
            scripted = torch.jit.script(fn)
            scripted()

    def test_union_does_not_replace_existing_annotated_type_empty_container(self):
        def fn():
            x: List[int] = []
            x.append("foo")
            return x

        with self.assertRaisesRegex(RuntimeError, "Could not match type str"):
            scripted = torch.jit.script(fn)
            scripted()

    def test_unions_of_unions_are_flattened(self):
        @torch.jit.script
        def fn(x: (int | str) | float) -> str:
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union(float, int, str)").run(s)

    def test_unions_of_a_single_argument_vanish(self):
        @torch.jit.script
        def fn(x: Union[int]) -> str:
            return "foo"

        s = fn.graph

        FileCheck().check("x : int").run(s)

    def test_union_redundant_arguments_are_skipped(self):
        @torch.jit.script
        def fn(x: int | str | int) -> str:
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union(int, str)").run(s)

    def test_union_redundant_arguments_are_skipped_optional(self):
        @torch.jit.script
        def fn(x: int | Optional[float] | Optional[int]) -> str:
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union(float, int, NoneType)").run(s)

    def test_union_redundant_arguments_are_skipped_subtyping(self):
        @torch.jit.script
        def fn(x: str | Tuple[Optional[int], int] | Tuple[int, int]) -> str:
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union((int?, int), str)").run(s)

    def test_union_redundant_arguments_are_skipped_container(self):
        @torch.jit.script
        def fn(x: List[str] | List[float] | List[str]) -> str:
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union(float[], str[])").run(s)

    def test_union_argument_order_is_ignored(self):
        @torch.jit.script
        def fn1(x: int | str) -> str:
            return "foo"

        @torch.jit.script
        def fn2(x: str | int) -> str:
            return "foo"

        for s in (fn1.graph, fn2.graph):
            FileCheck().check("x : Union(int, str)").run(s)

    def test_union_argument_order_is_ignored_container(self):
        @torch.jit.script
        def fn1(x: List[str] | List[int]) -> str:
            return "foo"

        @torch.jit.script
        def fn2(x: List[int] | List[str]) -> str:
            return "foo"

        for s in (fn1.graph, fn2.graph):
            FileCheck().check("x : Union(int[], str[])").run(s)

    def test_union_T_None_is_equivalent_to_optional_T(self):
        @torch.jit.script
        def inner(x: int | None) -> int:
            if x is not None:
                return x
            else:
                return 5

        @torch.jit.script
        def fn1() -> int:
            a: Optional[int] = 5
            b: Optional[int] = None
            a_ = inner(a)
            b_ = inner(b)
            return a_ + b_

        self.assertEqual(fn1(), 10)

        @torch.jit.script
        def inner2(x: Optional[int]) -> int:
            if x is not None:
                return x
            else:
                return 5

        @torch.jit.script
        def fn2() -> int:
            a: int | None = 5
            b: int | None = None
            a_ = inner(a)
            b_ = inner(b)
            return a_ + b_

        self.assertEqual(fn2(), 10)

    @unittest.expectedFailure
    def test_union_optional_of_union_return(self):
        @torch.jit.script
        def fn() -> None | str | int:
            y: Optional[int | str] = "foo"
            return y

    @unittest.expectedFailure
    def test_union_optional_of_union_is_flattened(self):
        @torch.jit.script
        def fn(flag: int) -> str | int | None:
            y: int | str | None = "foo"
            if flag == 0:
                x: Optional[int | str] = y
            elif flag == 1:
                x: Optional[int | str] = 1
            else:
                x: Optional[int | str] = None
            return x

        # Can't use `checkScript` because it will flag the fact that
        # the original code has `Optional[Union[int, str]]` but the
        # saved/loaded code has `Union[int, NoneType, str]` (even
        # though this is exactly what we want)
        self.assertEqual(fn(0), "foo")
        self.assertEqual(fn(1), 1)
        self.assertEqual(fn(2), None)

        buffer = io.BytesIO()
        torch.jit.save(fn, buffer)
        buffer = io.BytesIO(buffer.getvalue())
        l = torch.jit.load(buffer)

        s = l.code

        FileCheck().check("Union[int, NoneType, str]").check(
            "Union[int, NoneType, str]"
        ).run(s)

    def test_union_subclasses_larger_union(self):
        def fn() -> int | str | torch.Tensor:
            x: int | str = "foo"
            return x

        self.checkScript(fn, ())

    # TODO: We would like to eventually support this. The issue is being
    # tracked at https://github.com/pytorch/pytorch/issues/58167
    def test_union_as_dict_key(self):
        def fn():
            x: Dict[int | str, str] = {}
            x["foo"] = "bar"
            x[1] = 2
            return x[1]

        with self.assertRaisesRegex(
            RuntimeError,
            "only int, float, "
            "complex, Tensor, device and string keys "
            "are supported",
        ):
            torch.jit.script(fn)

    def test_union_as_dict_value(self):
        def fn():
            x: Dict[str, int | str] = {}
            x["foo"] = "bar"
            x["baz"] = 2
            return x["baz"]

        self.checkScript(fn, ())

    def test_union_module_with_union_instance_variable(self):
        class M(torch.nn.Module):
            x: int | str

            def __init__(self, x: int | str):
                super().__init__()
                self.x: int | str = x

            def forward(self, y: int | str):
                self.x = y
                return self.x

        self.checkModule(
            M(
                2,
            ),
            (1,),
        )
        self.checkModule(M("bar"), ("foo",))

    def test_union_module_with_union_class_variable(self):
        class M(torch.nn.Module):
            x: int | str = "foo"

            def __init__(self, y: int):
                super().__init__()
                x = y

            def forward(self, z: str):
                x = z
                return x

        self.checkModule(M(1), ("foo",))

    def test_union_type_refinement(self):
        def fn(x: int | str) -> str:
            if isinstance(x, str):
                z = x + "bar"
                return x
            else:
                return "baz"

        self.checkScript(fn, ("foo",))
        self.checkScript(fn, (1,))

    def test_union_type_refinement_union_rhs(self):
        def fn(x: int) -> str:
            if torch.jit.isinstance(x, int | str):
                return "bar"
            else:
                return "baz"

        self.checkScript(fn, (1,))

    def test_union_type_refinement_tuple_rhs(self):
        def fn(x: int | float | List[str]) -> str:
            if isinstance(x, (int, float)):
                if isinstance(x, int):
                    return str(x)
                else:
                    return "foo"
            else:
                if len(x):
                    return x[0]
                else:
                    return "bar"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (1.0,))
        self.checkScript(fn, (["a", "b", "c"],))

    def test_union_type_refinement_tuple_rhs_noncontained_type(self):
        def fn(x: int | List[str]) -> str:
            if isinstance(x, (int, float)):
                y = x + x
                return str(y)
            else:
                if len(x):
                    return x[0]
                else:
                    return "bar"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (["a", "b", "c"],))

    def test_union_type_refinement_tuple_rhs_union(self):
        @torch.jit.script
        def fn(x: int) -> str:
            if torch.jit.isinstance(x, (int | str, float)):
                y = x + x
                return str(y)
            else:
                return "foo"

        # TODO: There's currently an unrelated bug in
        # `torch.jit.isinstance` that makes it fail for tuple literals.
        # Posted here: https://github.com/pytorch/pytorch/issues/60095
        # Change `assertEqual` to `checkScript` when the bug is fixed
        self.assertEqual(fn(1), "2")

    def test_union_type_refinement_statically_false(self):
        @torch.jit.script
        def fn(x: int) -> str:
            if torch.jit.isinstance(x, (str | float, List[str], str)):
                z = x + "foo"
                return z
            else:
                return "bar"

        s = fn.graph

        # Check that we don't have any branching statements
        FileCheck().check_not("block0()").check_not("block1()").run(s)

    def test_union_type_refinement_statically_true(self):
        @torch.jit.script
        def fn(x: List[int] | int) -> List[int] | int:
            if not torch.jit.isinstance(x, (int, List[int])):
                return x
            else:
                l = [1, 2, 3]
                y: List[int] | int = l
                return y

        s = fn.graph

        # Check that we don't have any branching statements
        FileCheck().check_not("block0()").check_not("block1()").run(s)

    def test_union_type_refinement_partial_static_refinement_tuple_rhs(self):
        def fn(x: List[int] | int) -> int:
            if torch.jit.isinstance(x, (int, float, str)):
                # We should know that `x` is an `int` here
                z = x + 1
                return z
            else:
                return 100

        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn, (1,))

    def test_union_type_refinement_partial_static_refinement_union_rhs(self):
        def fn(x: List[int] | int) -> int:
            if torch.jit.isinstance(x, int | float | str):
                # We should know that `x` is an `int` here
                z = x + 1
                return z
            else:
                return 100

        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn, (1,))

    def test_union_type_refinement_internal_declaration(self):
        def fn(flag: bool) -> str:
            x: int | str | None = None
            if flag:
                y = "foo"
            else:
                y = 1
            if isinstance(x, str):
                return x
            else:
                return "bar"

        self.checkScript(fn, (True,))
        self.checkScript(fn, (False,))

    def test_union_branching_with_union_return_and_homogenous_types(self):
        def fn(x: int) -> int | str:
            if x % 2:
                return "foo"
            else:
                return "bar"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (8,))

    def test_union_branching_does_not_autoinfer_undeclared_union(self):
        def fn(x: int) -> str:
            if x % 2:
                y = "foo"
            else:
                y = x
            if isinstance(y, str):
                return y
            else:
                return "bar"

        with self.assertRaisesRegex(
            RuntimeError,
            "y is set to type str"
            " in the true branch and type int "
            "in the false branch",
        ):
            torch.jit.script(fn)

    def test_union_branching_does_not_widen_existing_inferred_type(self):
        def fn(x: int) -> str:
            y = "foo"
            if x % 2:
                y = "bar"
            else:
                y = x
            if isinstance(y, str):
                return y
            else:
                return "baz"

        with self.assertRaisesRegex(
            RuntimeError,
            "previously had type "
            "str but is now being assigned to a"
            " value of type int",
        ):
            torch.jit.script(fn)

    def test_union_schema_matching_on_internal_type(self):
        def fn(x: List[int] | Dict[str, int]) -> int:
            if torch.jit.isinstance(x, List[int]):
                return x[0]
            else:
                return list(x.values())[0]

        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn, ({"foo": 1, "bar": 2, "baz": 3},))

    def test_union_subtractive_refinement(self):
        def fn(x: List[int] | int) -> int:
            if not isinstance(x, int):
                x.append(1)
                return x[0]
            else:
                return x

        self.checkScript(fn, (1,))
        self.checkScript(fn, ([1, 2, 3],))

    def test_union_subtractive_refinement_with_container(self):
        def fn(x: List[int] | int) -> int:
            if not torch.jit.isinstance(x, List[int]):
                return x
            else:
                x.append(1)
                return x[0]

        self.checkScript(fn, (1,))
        self.checkScript(fn, ([1, 2, 3],))

    def test_union_memory_aliasing(self):
        def fn():
            x: List[torch.Tensor] = []
            z: List[Optional[List[torch.Tensor]]] = []
            z.append(x)
            x_alias = z[0]
            if torch.jit.isinstance(x_alias, List[torch.Tensor]):
                x_alias.append(torch.tensor(3))
            return x

        self.checkScript(fn, ())

    def test_union_serialization_preserves_type_annotations(self):
        # This function will fail after being torch.jit.save'd and
        # torch.jit.load'd if the type annotations aren't preserved
        # for Union during serialization. We need the `Union[str, int]`
        # annotation to make sure that `y` is typed as a Union instead
        # of as a str in one branch and an int in the other
        def fn(x: int) -> str:
            if x % 2:
                y: str | int = "bar"
            else:
                y: str | int = x
            if isinstance(y, str):
                return y
            else:
                return "baz"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (8,))

    def _assert_passes(self, template: str, ann: str, lhs: str):
        code = template.format(ann=ann, lhs=lhs)
        self.checkScript(code, (), name="fn")

    def _assert_raises(self, template: str, ann: str, lhs: str, msg: str):
        code = template.format(ann=ann, lhs=lhs)
        with self.assertRaisesRegex(RuntimeError, msg):
            cu = torch.jit.CompilationUnit(code, _frames_up=1)
            string_frontend = getattr(cu, "fn")  # noqa: B009

    def test_union_with_list_assignment(self):
        template = dedent(
            """
            def fn():
                x: {ann} = {lhs}
                if torch.jit.isinstance(x, List[torch.Tensor]):
                    x.append(torch.tensor(3))
                return x
        """
        )

        lhs = {
            "list_literal_empty": "[]",
            "list_literal_of_tensor": "[torch.arange(3), torch.arange(5)]",
            "list_literal_of_str": '["foo", "bar", "baz"]',
            "list_literal_of_mixed": "[torch.arange(5), 1]",
            "list_comprehension_of_tensor": "[torch.add(x, 1) for x in [torch.arange(3), torch.arange(5)]]",
            "list_comprehension_of_str": '[x + "!" for x in ["foo", "bar", "baz"]]',
            "list_comprehension_of_mixed": "[torch.add(1, x) for x in [torch.arange(5), 1]]",
        }

        """
        List[str] | List[torch.Tensor]
        """
        self._assert_raises(
            template,
            "List[str] | List[torch.Tensor]",
            lhs["list_literal_empty"],
            "there are multiple possible List type "
            "candidates in the Union annotation",
        )

        self._assert_passes(
            template, "List[str] | List[torch.Tensor]", lhs["list_literal_of_tensor"]
        )

        self._assert_passes(
            template, "List[str] | List[torch.Tensor]", lhs["list_literal_of_str"]
        )

        self._assert_raises(
            template,
            "List[str] | List[torch.Tensor]",
            lhs["list_literal_of_mixed"],
            "none of those types match the types of the" " given list elements",
        )

        self._assert_passes(
            template,
            "List[str] | List[torch.Tensor]",
            lhs["list_comprehension_of_tensor"],
        )

        self._assert_passes(
            template, "List[str] | List[torch.Tensor]", lhs["list_comprehension_of_str"]
        )

        # TODO: Support mixed list comprehensions
        self._assert_raises(
            template,
            "List[str] | List[torch.Tensor]",
            lhs["list_comprehension_of_mixed"],
            "Arguments for call are not valid",
        )

        """
        int | torch.Tensor
        """
        self._assert_raises(
            template,
            "int | torch.Tensor",
            lhs["list_literal_empty"],
            "Expected an Union type annotation with an " "inner List type",
        )

        self._assert_raises(
            template,
            "int | torch.Tensor",
            lhs["list_literal_of_tensor"],
            "Expected an Union type annotation with an " "inner List type",
        )

        self._assert_raises(
            template,
            "int | torch.Tensor",
            lhs["list_comprehension_of_tensor"],
            "Expected an Union type annotation with an " "inner List type",
        )

        """
        List[torch.Tensor] | int
        """
        self._assert_passes(
            template, "List[torch.Tensor] | int", lhs["list_literal_empty"]
        )

        self._assert_passes(
            template, "List[torch.Tensor] | int", lhs["list_literal_of_tensor"]
        )

        self._assert_raises(
            template,
            "List[torch.Tensor] | int",
            lhs["list_literal_of_str"],
            r"List type annotation `List\[Tensor\]` did "
            "not match the types of the given list "
            "elements",
        )

        self._assert_raises(
            template,
            "List[torch.Tensor] | int",
            lhs["list_literal_of_mixed"],
            r"List type annotation `List\[Tensor\]` did "
            "not match the types of the given list "
            "elements",
        )

        self._assert_passes(
            template, "List[torch.Tensor] | int", lhs["list_comprehension_of_tensor"]
        )

        self._assert_raises(
            template,
            "List[torch.Tensor] | int",
            lhs["list_comprehension_of_str"],
            r"List type annotation `List\[Tensor\]` did "
            "not match the types of the given list "
            "elements",
        )

        # TODO(@ansley): Support mixed list comprehensions
        self._assert_raises(
            template,
            "List[torch.Tensor] | int",
            lhs["list_comprehension_of_mixed"],
            "Arguments for call are not valid",
        )

    def test_union_with_dict_assignment(self):
        template = dedent(
            """
            def fn():
                x: {ann} = {lhs}
                if torch.jit.isinstance(x, Dict[str, torch.Tensor]):
                    x["foo"] = torch.tensor(3)
                return x
        """
        )

        lhs = {
            "dict_literal_empty": "{}",
            "dict_literal_of_str_tensor": '{"foo" : torch.arange(3), "bar" : torch.arange(5)}',
            "dict_literal_of_str_int": '{"foo" : 1, "bar" : 2}',
            "dict_literal_of_mixed": '{"foo" : torch.arange(3), "bar" : 2}',
            "dict_comprehension_of_str_tensor": '{x : torch.add(y, 1) for x, y in \
                    zip(["foo", "bar"], [torch.arange(3), torch.arange(5)])}',
            "dict_comprehension_of_str_int": '{x : torch.add(y, 1) for x, y in \
                    zip(["foo", "bar"], [1, 2]}',
            "dict_comprehension_of_mixed": '{x : torch.add(y, 1) for x, y in \
                    zip(["foo", "bar"], [torch.arange(3), 2])}',
            "dict_keyword": "dict(foo=torch.arange(3), baz=torch.arange(5))",
            "dict_keyword_with_iterable": 'dict([("foo", torch.arange(3)), ("bar", torch.arange(5))])',
            "dict_keyword_with_empty_iterable": "dict([])",
            "dict_keyword_with_internal_aggregate_function": 'dict(zip(["foo", "bar"], [torch.arange(3), torch.arange(5)])',
            "dict_keyword_with_mapping": 'dict({"foo" : torch.arange(3), "bar" : torch.arange(5)})',
            "dict_keyword_with_mapping_and_kwargs": 'dict({"foo" : torch.arange(3), "bar" : torch.arange(5)}, baz=torch.arange(7))',
        }

        """
        Dict[str, torch.Tensor] | Dict[str, int]
        """
        self._assert_raises(
            template,
            "List[str] | List[torch.Tensor]",
            lhs["dict_literal_empty"],
            "Expected an Union type annotation with an " "inner Dict type",
        )

        self._assert_passes(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_literal_of_str_tensor"],
        )

        self._assert_passes(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_literal_of_str_int"],
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_literal_of_mixed"],
            "none of those dict types can hold the "
            "types of the given keys and values",
        )

        # TODO: String frontend does not support tuple unpacking
        # https://github.com/pytorch/pytorch/issues/64096
        # self._assert_passes(template, "Dict[str, torch.Tensor] | Dict[str, int]",
        #              lhs["dict_comprehension_of_str_tensor"])

        # self._assert_passes(template, "Dict[str, torch.Tensor] | Dict[str, int]",
        #              lhs["dict_comprehension_of_str_int"])

        # self._assert_raises(template, "Dict[str, torch.Tensor] | Dict[str, int]",
        #              lhs["dict_comprehension_of_mixed"],
        #              "foobar")

        # self._assert_passes(template,
        #                    "Dict[str, torch.Tensor] | Dict[str, int]",
        #                    lhs["dict_keyword_with_internal_aggregate_function"])

        # TODO(@ansley): Follow-up project needed for full type
        # inference with dict keyword (supported for dict comprehension
        # and dict literal already; should not be a blocker for anyone)
        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_keyword"],
            "full type inference is not yet supported",
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_keyword_with_iterable"],
            "full type inference is not yet supported",
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_keyword_with_empty_iterable"],
            "full type inference is not yet supported",
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_keyword_with_mapping"],
            "full type inference is not yet supported",
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_keyword_with_mapping_and_kwargs"],
            "full type inference is not yet supported",
        )

        """
        int | torch.Tensor
        """
        self._assert_raises(
            template,
            "int | torch.Tensor",
            lhs["dict_literal_empty"],
            "Expected an Union type annotation with " "an inner Dict type",
        )

        self._assert_raises(
            template,
            "int | torch.Tensor",
            lhs["dict_literal_of_str_tensor"],
            "Expected an Union type annotation with " "an inner Dict type",
        )

        # See above--string frontend does not support tuple unpacking
        # self._assert_raises(template, "int | torch.Tensor",
        #              lhs["dict_comprehension_of_tensor"],
        #              "foobar")

        """
        Dict[str, torch.Tensor] | int
        """
        self._assert_passes(
            template, "Dict[str, torch.Tensor] | int", lhs["dict_literal_empty"]
        )

        self._assert_passes(
            template, "Dict[str, torch.Tensor] | int", lhs["dict_literal_of_str_tensor"]
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | int",
            lhs["dict_literal_of_str_int"],
            "Type annotation was inferred to be "
            r"`Dict\[str, Tensor\]`, but the type of "
            "values given by the dict literal is",
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | int",
            lhs["dict_literal_of_mixed"],
            "Type annotation was inferred to be "
            r"`Dict\[str, Tensor\]`, but the type of "
            "values given by the dict literal is",
        )

        self._assert_passes(
            template, "Dict[str, torch.Tensor] | int", lhs["dict_keyword"]
        )

        self._assert_passes(
            template, "Dict[str, torch.Tensor] | int", lhs["dict_keyword_with_iterable"]
        )

        self._assert_passes(
            template,
            "Dict[str, torch.Tensor] | int",
            lhs["dict_keyword_with_empty_iterable"],
        )

        self._assert_passes(
            template, "Dict[str, torch.Tensor] | int", lhs["dict_keyword_with_mapping"]
        )

        self._assert_passes(
            template,
            "Dict[str, torch.Tensor] | int",
            lhs["dict_keyword_with_mapping_and_kwargs"],
        )

        # See above--string frontend does not support tuple unpacking
        # self._assert_passes(template,
        #                    "Dict[str, torch.Tensor] | int",
        #                    lhs["dict_keyword_with_internal_aggregate_function"])
        #
        # self._assert_passes(template,
        #                    "Dict[str, torch.Tensor] | int",
        #                    lhs["dict_comprehension_of_str_tensor"])

        # self._assert_raises(template,
        #                    "Dict[str, torch.Tensor] | int",
        #                    lhs["dict_comprehension_of_str_int"],
        #                    "foobar")

        # self._assert_raises(template,
        #                    "Dict[str, torch.Tensor] | int",
        #                    lhs["dict_comprehension_of_mixed"],
        #                    "foobar")
