from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase, make_global
from torch.testing import FileCheck
from torch import jit
from textwrap import dedent
from jit.test_module_interface import TestModuleInterface  # noqa: F401
import inspect
import unittest
import sys
import torch
import torch.testing._internal.jit_utils
import torch.nn as nn

class TestScriptPy3(JitTestCase):
    def test_joined_str(self):
        def func(x):
            hello, test = "Hello", "test"
            print(f"{hello + ' ' + test}, I'm a {test}") # noqa E999
            print(f"format blank") # noqa F541
            hi = 'hi'
            print(f"stuff before {hi}")
            print(f"{hi} stuff after")
            return x + 1

        x = torch.arange(4., requires_grad=True)
        # TODO: Add support for f-strings in string parser frontend
        # self.checkScript(func, [x], optimize=True, capture_output=True)

        with self.capture_stdout() as captured:
            out = func(x)

        scripted = torch.jit.script(func)
        with self.capture_stdout() as captured_script:
            out_script = func(x)

        self.assertEqual(out, out_script)
        self.assertEqual(captured, captured_script)

    @unittest.skipIf(sys.version_info[:2] < (3, 7), "`dataclasses` module not present on < 3.7")
    def test_dataclass_error(self):
        from dataclasses import dataclass

        @dataclass
        class NormalizationInfo(object):
            mean: float = 0.0

            def compute(self, total_rows):
                return self.mean

        def fn():
            return NormalizationInfo(1, 2, 3, 4, 5)

        with self.assertRaisesRegex(OSError, "could not get source code"):
            torch.jit.script(fn)


    def test_kwarg_support(self):
        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, "variable number of arguments"):
            class M(torch.nn.Module):
                def forward(self, *, n_tokens: int, device_name: str = 2):
                    pass
            torch.jit.script(M())

        class M(torch.nn.Module):
            def forward(self, *, n_tokens: int, device_name: str):
                return n_tokens, device_name

        sm = torch.jit.script(M())

        with self.assertRaisesRegex(RuntimeError, "missing value for argument 'n_tokens'"):
            sm()

        input = (3, 'hello')
        self.assertEqual(sm(*input), input)

    def test_types_as_values(self):
        def fn(m: torch.Tensor) -> torch.device:
            return m.device

        self.checkScript(fn, [torch.randn(2, 2)])

        GG = namedtuple('GG', ['f', 'g'])

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            @torch.jit.ignore
            def foo(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[GG, GG]:
                return GG(x, z), GG(x, z)

            def forward(self, x, z):
                return self.foo(x, z)

        foo = torch.jit.script(Foo())
        y = foo(torch.randn(2, 2), torch.randn(2, 2))

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

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
            def __init__(self):
                super(M, self).__init__()

            def forward(self, in_batch: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
                self.dropout_modality(in_batch)
                fn(in_batch)
                return torch.tensor(1)

            @torch.jit.ignore
            def dropout_modality(self, in_batch: Dict[str, Optional[torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
                return in_batch

        sm = torch.jit.script(M())
        FileCheck().check("dropout_modality").check("in_batch").run(str(sm.graph))

    def test_python_callable(self):
        class MyPythonClass(object):
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
                return fn('2')

    def test_type_annotate_py3(self):
        def fn():
            a : List[int] = []
            b : torch.Tensor = torch.ones(2, 2)
            c : Optional[torch.Tensor] = None
            d : Optional[torch.Tensor] = torch.ones(3, 4)
            for _ in range(10):
                a.append(4)
                c = torch.ones(2, 2)
                d = None
            return a, b, c, d

        self.checkScript(fn, ())

        def wrong_type():
            wrong : List[int] = [0.5]
            return wrong

        with self.assertRaisesRegex(RuntimeError, "Lists must contain only a single type"):
            torch.jit.script(wrong_type)

    def test_optional_no_element_type_annotation(self):
        """
        Test that using an optional with no contained types produces an error.
        """
        def fn_with_comment(x: torch.Tensor) -> Optional:
            return (x, x)

        def annotated_fn(x: torch.Tensor) -> Optional:
            return (x, x)

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Optional without a contained type"):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Optional without a contained type"):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Optional without a contained type"):
            torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Optional without a contained type"):
            torch.jit.script(annotated_fn)

    def test_tuple_no_element_type_annotation(self):
        """
        Test that using a tuple with no contained types produces an error.
        """
        def fn_with_comment(x: torch.Tensor) -> Tuple:
            return (x, x)

        def annotated_fn(x: torch.Tensor) -> Tuple:
            return (x, x)

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Tuple without a contained type"):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Tuple without a contained type"):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Tuple without a contained type"):
            torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(RuntimeError, r"Attempted to use Tuple without a contained type"):
            torch.jit.script(annotated_fn)

    def test_tuple_subscripted_assign(self):
        with self.assertRaisesRegex(RuntimeError, "subscripted assignment"):
            @torch.jit.script
            def foo(a: Tuple[int, int]) -> None:
                a[0] = a[1]

        with self.assertRaisesRegex(RuntimeError, "augmented assignment"):
            @torch.jit.script
            def bar(a: Tuple[int, int]) -> None:
                a[0] += a[1]

    def test_subexpression_List_Future(self):

        @torch.jit.script
        def fn(x: List[torch.jit.Future[int]]) -> torch.jit.Future[int]:
            return x[0]

        FileCheck().check('Future[int]').check('Future[int]').run(fn.graph)

    def test_subexpression_Future_annotate(self):
        @torch.jit.script
        def fn() -> torch.jit.Future[int]:
            x: List[torch.jit.Future[int]] = []
            return x[0]

        FileCheck().check("Future[int][]").run(fn.graph)

    def test_future_isinstance(self):
        @torch.jit.script
        def fn(x: Any) -> torch.jit.Future[int]:
            assert isinstance(x, jit.Future[int])
            return x

        FileCheck().check("Future[int]").run(fn.graph)

    def test_str_refine_any(self):
        def forward(x: Any) -> str:
            if isinstance(x, str):
                return x
            return "foo"
        forward = torch.jit.script(forward)
        self.assertEqual(forward(1), "foo")
        self.assertEqual(forward("bar"), "bar")

    def test_subexpression_Tuple_int_int_Future(self):

        @torch.jit.script
        def fn(x: Tuple[int, int, torch.jit.Future[int]]) -> Tuple[int, torch.jit.Future[int]]:
            return x[0], x[2]

        FileCheck().check('(int, int, Future[int])').check('(int, Future[int])').run(fn.graph)

    def test_subexpression_Dict_int_Future(self):

        @torch.jit.script
        def fn(x: Dict[int, torch.jit.Future[int]], y: int) -> torch.jit.Future[int]:
            return x[y]

        FileCheck().check('Dict(int, Future(int))').check('Future[int]').run(fn.graph)

    def test_subexpression_Optional(self):

        @torch.jit.script
        def fn(x: Optional[Dict[int, torch.jit.Future[int]]]) -> Optional[torch.jit.Future[int]]:
            if x is not None:
                return x[0]
            else:
                return None

        FileCheck().check('Dict(int, Future(int))?').run(fn.graph)

    def test_unimported_type_resolution(self):
        # verify fallback from the python resolver to the c++ resolver

        @ torch.jit.script
        def fn(x):
            # type: (number) -> number
            return x + 1

        FileCheck().check('Scalar').run(fn.graph)

    def test_parser_bug(self):
        def parser_bug(o: Optional[torch.Tensor]):
            pass

    def test_mismatched_annotation(self):
        with self.assertRaisesRegex(RuntimeError, 'annotated with type'):
            @torch.jit.script
            def foo():
                x : str = 4
                return x

    def test_reannotate(self):
        with self.assertRaisesRegex(RuntimeError, 'declare and annotate'):
            @torch.jit.script
            def foo():
                x = 5
                if 1 == 1:
                    x : Optional[int] = 7

    def test_module_inplace_construct(self):
        class M(nn.Module):
            def __init__(self, start: int):
                super().__init__()
                self.linear = nn.Linear(3, 3)
                self.attribute = start
                self.parameter = nn.Parameter(torch.tensor(3, dtype=torch.float))

            def method(self) -> int:
                return self.attribute

            @torch.jit.unused
            def unused_method(self):
                return self.attribute + self.attribute

            def forward(self, x):
                return self.linear(self.linear(x))


        class N(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            @torch.jit.ignore
            def ignored_method(self, x):
                return x

            def forward(self, x):
                return self.linear(x)

        m = torch.jit.script(M(3))
        n = torch.jit.script(N())

        n._reconstruct(m._c)

        inp = torch.rand((3))

        # Check that both modules produce the same output.
        with torch.no_grad():
            m_out = m(inp)
            n_out = n(inp)
            self.assertEqual(m_out, n_out)

        # Check that ignored method is still intact.
        self.assertEqual(inp, n.ignored_method(inp))

    def test_if_returning_any(self):
        """
        Check that an if statement can return different
        types early from each branch when the return
        type of the function is Any.
        """
        def if_function(inp: torch.Tensor) -> Any:
            if inp.shape[0] == 1:
                return inp * inp
            else:
                return "str"

        self.checkScript(if_function, (torch.randn(5),))

    def test_module_properties(self):
        class ModuleWithProperties(torch.nn.Module):
            __jit_unused_properties__ = ["ignored_attr"]

            def __init__(self, a: int):
                super().__init__()
                self.a = a

            def forward(self, a: int, b: int):
                self.attr = a + b
                return self.attr

            @property
            def attr(self):
                return self.a

            @property
            def ignored_attr(self):
                return sum([self.a])

            @torch.jit.unused
            @property
            def ignored_attr_2(self):
                return sum([self.a])

            @ignored_attr_2.setter
            def ignored_attr_2(self, value):
                self.a = sum([self.a])

            @attr.setter
            def attr(self, a: int):
                if a > 0:
                    self.a = a
                else:
                    self.a = 0

        class ModuleWithNoSetter(torch.nn.Module):
            def __init__(self, a: int):
                super().__init__()
                self.a = a

            def forward(self, a: int, b: int):
                self.attr + a + b

            @property
            def attr(self):
                return self.a + 1

        self.checkModule(ModuleWithProperties(5), (5, 6,))
        self.checkModule(ModuleWithProperties(5), (-5, -6,))
        self.checkModule(ModuleWithNoSetter(5), (5, 6,))
        self.checkModule(ModuleWithNoSetter(5), (-5, -6,))

        mod = ModuleWithProperties(3)
        scripted_mod = torch.jit.script(mod)

        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            scripted_mod.ignored_attr

    def test_ignoring_module_attributes(self):
        """
        Test that module attributes can be ignored.
        """
        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()

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

        with self.assertRaisesRegexWithHighlight(RuntimeError, r"attribute was ignored during compilation", "self.sub"):
            scripted_mod = torch.jit.script(mod)


    def test_export_opnames_interface(self):

        @torch.jit.interface
        class OneTwoModule(nn.Module):
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                pass

            def two(self, x: torch.Tensor) -> torch.Tensor:
                pass

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                pass

        class FooMod(nn.Module):
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            def two(self, x: torch.Tensor) -> torch.Tensor:
                return 2 * x

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.one(self.two(x), x)

        class BarMod(nn.Module):
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x * y

            def two(self, x: torch.Tensor) -> torch.Tensor:
                return 2 / x

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.two(self.one(x, x))

        make_global(OneTwoModule)

        class M(nn.Module):
            sub : OneTwoModule

            def __init__(self):
                super(M, self).__init__()
                self.sub = BarMod()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.sub.forward(x)

        def use_module_interface(mod_list: List[OneTwoModule], x: torch.Tensor):
            return mod_list[0].forward(x) + mod_list[1].forward(x)

        scripted_M_mod = torch.jit.script(M())
        # Temporarily test empty output because lite interpreter does not support interface call
        # Replace it with the issubset call when interface call is supported.
        self.assertTrue(len(torch.jit.export_opnames(scripted_M_mod)) == 0)
        # self.assertTrue(set(['aten::mul.Scalar', 'aten::mul.Tensor', 'aten::reciprocal']).issubset(
        #     set(torch.jit.export_opnames(scripted_M_mod))))

        scripted_M_mod.sub = torch.jit.script(FooMod())
        self.assertTrue(len(torch.jit.export_opnames(scripted_M_mod)) == 0)
        # self.assertTrue(set(['aten::add.Tensor', 'aten::mul.Scalar']).issubset(
        #     set(torch.jit.export_opnames(scripted_M_mod))))

    def test_broadcasting_list(self):
        """
        Test BroadcastingList and torch.nn._size_N_t alias
        """
        from torch._jit_internal import BroadcastingList2
        from torch.nn.common_types import _size_2_t

        def sum_i(x: _size_2_t) -> int:
            return x[0] + x[1]

        def sum_f(x: BroadcastingList2[float]) -> float:
            return x[0] + x[1]

        self.assertTrue(torch.jit.script(sum_i)(4) == 8)
        self.assertTrue(torch.jit.script(sum_f)(4.5) == 9.)


if __name__ == '__main__':
    run_tests()
