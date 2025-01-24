# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import inspect
import os
import sys
from collections import namedtuple
from textwrap import dedent
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.testing._internal.jit_utils
from jit.test_module_interface import TestModuleInterface  # noqa: F401
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestTypesAndAnnotation(JitTestCase):
    def test_pep585_type(self):
        # TODO add test to use PEP585 type annotation for return type after py3.9
        # see: https://www.python.org/dev/peps/pep-0585/#id5
        def fn(x: torch.Tensor) -> Tuple[Tuple[torch.Tensor], Dict[str, int]]:
            xl: list[tuple[torch.Tensor]] = []
            xd: dict[str, int] = {}
            xl.append((x,))
            xd["foo"] = 1
            return xl.pop(), xd

        self.checkScript(fn, [torch.randn(2, 2)])

        x = torch.randn(2, 2)
        expected = fn(x)
        scripted = torch.jit.script(fn)(x)

        self.assertEqual(expected, scripted)

    def test_types_as_values(self):
        def fn(m: torch.Tensor) -> torch.device:
            return m.device

        self.checkScript(fn, [torch.randn(2, 2)])

        GG = namedtuple("GG", ["f", "g"])

        class Foo(torch.nn.Module):
            @torch.jit.ignore
            def foo(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[GG, GG]:
                return GG(x, z), GG(x, z)

            def forward(self, x, z):
                return self.foo(x, z)

        foo = torch.jit.script(Foo())
        y = foo(torch.randn(2, 2), torch.randn(2, 2))

        class Foo(torch.nn.Module):
            @torch.jit.ignore
            def foo(self, x, z) -> Tuple[GG, GG]:
                return GG(x, z)

            def forward(self, x, z):
                return self.foo(x, z)

        foo = torch.jit.script(Foo())
        y = foo(torch.randn(2, 2), torch.randn(2, 2))

    def test_ignore_with_types(self):
        @torch.jit.ignore
        def fn(x: Dict[str, Optional[torch.Tensor]]):
            return x + 10

        class M(torch.nn.Module):
            def forward(
                self, in_batch: Dict[str, Optional[torch.Tensor]]
            ) -> torch.Tensor:
                self.dropout_modality(in_batch)
                fn(in_batch)
                return torch.tensor(1)

            @torch.jit.ignore
            def dropout_modality(
                self, in_batch: Dict[str, Optional[torch.Tensor]]
            ) -> Dict[str, Optional[torch.Tensor]]:
                return in_batch

        sm = torch.jit.script(M())
        FileCheck().check("dropout_modality").check("in_batch").run(str(sm.graph))

    def test_python_callable(self):
        class MyPythonClass:
            @torch.jit.ignore
            def __call__(self, *args) -> str:
                return str(type(args[0]))

        the_class = MyPythonClass()

        @torch.jit.script
        def fn(x):
            return the_class(x)

        # This doesn't involve the string frontend, so don't use checkScript
        x = torch.ones(2)
        self.assertEqual(fn(x), the_class(x))

    def test_bad_types(self):
        @torch.jit.ignore
        def fn(my_arg):
            return my_arg + 10

        with self.assertRaisesRegex(RuntimeError, "argument 'my_arg'"):

            @torch.jit.script
            def other_fn(x):
                return fn("2")

    def test_type_annotate_py3(self):
        def fn():
            a: List[int] = []
            b: torch.Tensor = torch.ones(2, 2)
            c: Optional[torch.Tensor] = None
            d: Optional[torch.Tensor] = torch.ones(3, 4)
            for _ in range(10):
                a.append(4)
                c = torch.ones(2, 2)
                d = None
            return a, b, c, d

        self.checkScript(fn, ())

        def wrong_type():
            wrong: List[int] = [0.5]
            return wrong

        with self.assertRaisesRegex(
            RuntimeError,
            "List type annotation"
            r" `List\[int\]` did not match the "
            "types of the given list elements",
        ):
            torch.jit.script(wrong_type)

    def test_optional_no_element_type_annotation(self):
        """
        Test that using an optional with no contained types produces an error.
        """

        def fn_with_comment(x: torch.Tensor) -> Optional:
            return (x, x)

        def annotated_fn(x: torch.Tensor) -> Optional:
            return (x, x)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            torch.jit.script(annotated_fn)

    def test_tuple_no_element_type_annotation(self):
        """
        Test that using a tuple with no contained types produces an error.
        """

        def fn_with_comment(x: torch.Tensor) -> Tuple:
            return (x, x)

        def annotated_fn(x: torch.Tensor) -> Tuple:
            return (x, x)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            torch.jit.script(annotated_fn)

    def test_ignoring_module_attributes(self):
        """
        Test that module attributes can be ignored.
        """

        class Sub(torch.nn.Module):
            def forward(self, a: int) -> int:
                return sum([a])

        class ModuleWithIgnoredAttr(torch.nn.Module):
            __jit_ignored_attributes__ = ["a", "sub"]

            def __init__(self, a: int, b: int):
                super().__init__()
                self.a = a
                self.b = b
                self.sub = Sub()

            def forward(self) -> int:
                return self.b

            @torch.jit.ignore
            def ignored_fn(self) -> int:
                return self.sub.forward(self.a)

        mod = ModuleWithIgnoredAttr(1, 4)
        scripted_mod = torch.jit.script(mod)
        self.assertEqual(scripted_mod(), 4)
        self.assertEqual(scripted_mod.ignored_fn(), 1)

        # Test the error message for ignored attributes.
        class ModuleUsesIgnoredAttr(torch.nn.Module):
            __jit_ignored_attributes__ = ["a", "sub"]

            def __init__(self, a: int):
                super().__init__()
                self.a = a
                self.sub = Sub()

            def forward(self) -> int:
                return self.sub(self.b)

        mod = ModuleUsesIgnoredAttr(1)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"attribute was ignored during compilation", "self.sub"
        ):
            scripted_mod = torch.jit.script(mod)

    def test_ignoring_fn_with_nonscriptable_types(self):
        class CFX:
            def __init__(self, a: List[torch.Tensor]) -> None:
                self.a = a

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sin(x)

            @torch.jit._drop
            def __iter__(self) -> Iterator[torch.Tensor]:
                return iter(self.a)

            @torch.jit._drop
            def __fx_create_arg__(
                self, tracer: torch.fx.Tracer
            ) -> torch.fx.node.Argument:
                # torch.fx classes are not scriptable
                return tracer.create_node(
                    "call_function",
                    CFX,
                    args=(tracer.create_arg(self.features),),
                    kwargs={},
                )

        torch.jit.script(CFX)

    def test_unimported_type_resolution(self):
        # verify fallback from the python resolver to the c++ resolver

        @torch.jit.script
        def fn(x):
            # type: (number) -> number
            return x + 1

        FileCheck().check("Scalar").run(fn.graph)

    def test_parser_bug(self):
        def parser_bug(o: Optional[torch.Tensor]):
            pass

    def test_mismatched_annotation(self):
        with self.assertRaisesRegex(RuntimeError, "annotated with type"):

            @torch.jit.script
            def foo():
                x: str = 4
                return x

    def test_reannotate(self):
        with self.assertRaisesRegex(RuntimeError, "declare and annotate"):

            @torch.jit.script
            def foo():
                x = 5
                if 1 == 1:
                    x: Optional[int] = 7

    def test_annotate_outside_init(self):
        msg = "annotations on instance attributes must be declared in __init__"
        highlight = "self.x: int"

        # Simple case
        with self.assertRaisesRegexWithHighlight(ValueError, msg, highlight):

            @torch.jit.script
            class BadModule:
                def __init__(self, x: int):
                    self.x = x

                def set(self, val: int):
                    self.x: int = val

        # Type annotation in a loop
        with self.assertRaisesRegexWithHighlight(ValueError, msg, highlight):

            @torch.jit.script
            class BadModuleLoop:
                def __init__(self, x: int):
                    self.x = x

                def set(self, val: int):
                    for i in range(3):
                        self.x: int = val

        # Type annotation in __init__, should not fail
        @torch.jit.script
        class GoodModule:
            def __init__(self, x: int):
                self.x: int = x

            def set(self, val: int):
                self.x = val

    def test_inferred_type_error_message(self):
        inferred_type = torch._C.InferredType("ErrorReason")

        with self.assertRaisesRegex(
            RuntimeError,
            "Tried to get the type from an InferredType but the type is null.",
        ):
            t = inferred_type.type()

        with self.assertRaisesRegex(RuntimeError, "ErrorReason"):
            t = inferred_type.type()
