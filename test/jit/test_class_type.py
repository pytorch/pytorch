import io
import os
import sys
import unittest

import torch
import torch.nn as nn
from torch.testing import FileCheck
from typing import Any

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, make_global
import torch.testing._internal.jit_utils
from torch.testing._internal.common_utils import IS_SANDCASTLE
from typing import List, Tuple, Iterable, Optional, Dict

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestClassType(JitTestCase):
    def test_reference_semantics(self):
        """
        Test that modifications made to a class instance in TorchScript
        are visible in eager.
        """
        class Foo(object):
            def __init__(self, a: int):
                self.a = a

            def set_a(self, value: int):
                self.a = value

            def get_a(self) -> int:
                return self.a

            @property
            def attr(self):
                return self.a

        make_global(Foo)  # see [local resolution in python]

        def test_fn(obj: Foo):
            obj.set_a(2)

        scripted_fn = torch.jit.script(test_fn)
        obj = torch.jit.script(Foo(1))
        self.assertEqual(obj.get_a(), 1)
        self.assertEqual(obj.attr, 1)

        scripted_fn(obj)

        self.assertEqual(obj.get_a(), 2)
        self.assertEqual(obj.attr, 2)

    def test_get_with_method(self):
        class FooTest(object):
            def __init__(self, x):
                self.foo = x

            def getFooTest(self):
                return self.foo

        def fn(x):
            foo = FooTest(x)
            return foo.getFooTest()

        input = torch.ones(2, 3)
        self.assertEqual(fn(input), input)

    def test_get_attr(self):
        class FooTest(object):  # noqa: B903
            def __init__(self, x):
                self.foo = x

        @torch.jit.script
        def fn(x):
            foo = FooTest(x)
            return foo.foo

        input = torch.ones(2, 3)
        self.assertEqual(fn(input), input)

    def test_in(self):
        class FooTest(object):  # noqa: B903
            def __init__(self):
                pass

            def __contains__(self, key: str) -> bool:
                return key == 'hi'

        @torch.jit.script
        def fn():
            foo = FooTest()
            return 'hi' in foo, 'no' in foo

        self.assertEqual(fn(), (True, False))

    def test_set_attr_in_method(self):
        class FooTest(object):
            def __init__(self, x: int) -> None:
                self.foo = x

            def incFooTest(self, y: int) -> None:
                self.foo = self.foo + y

        @torch.jit.script
        def fn(x: int) -> int:
            foo = FooTest(x)
            foo.incFooTest(2)
            return foo.foo

        self.assertEqual(fn(1), 3)

    def test_set_attr_type_mismatch(self):
        with self.assertRaisesRegexWithHighlight(RuntimeError, "Wrong type for attribute assignment", "self.foo = 10"):
            @torch.jit.script
            class FooTest(object):
                def __init__(self, x):
                    self.foo = x
                    self.foo = 10  # should error since int != Tensor

    def test_get_attr_not_initialized(self):
        with self.assertRaisesRegexWithHighlight(RuntimeError, "object has no attribute or method", "self.asdf"):
            @torch.jit.script
            class FooTest(object):
                def __init__(self, x):
                    self.foo = x

                def get_non_initialized(self):
                    return self.asdf  # asdf isn't an attr

    def test_set_attr_non_initialized(self):
        with self.assertRaisesRegexWithHighlight(RuntimeError, "Tried to set nonexistent attribute", "self.bar = y"):
            @torch.jit.script
            class FooTest(object):
                def __init__(self, x):
                    self.foo = x

                def set_non_initialized(self, y):
                    self.bar = y  # can't assign to non-initialized attr

    def test_schema_human_readable(self):
        """
        Make sure that the schema is human readable, ie the mode parameter should read "nearest" instead of being displayed in octal
        aten::__interpolate(Tensor input, int? size=None, float[]? scale_factor=None,
        str mode='\156\145\141\162\145\163\164', bool? align_corners=None) -> (Tensor):
        Expected a value of type 'Optional[int]' for argument 'size' but instead found type 'Tensor'.
        """
        with self.assertRaisesRegexWithHighlight(RuntimeError, "nearest", ""):
            @torch.jit.script
            def FooTest(x):
                return torch.nn.functional.interpolate(x, 'bad')

    def test_type_annotations(self):
        with self.assertRaisesRegexWithHighlight(RuntimeError, "Expected a value of type \'bool", ""):
            @torch.jit.script  # noqa: B903
            class FooTest(object):  # noqa: B903
                def __init__(self, x: bool) -> None:
                    self.foo = x

            @torch.jit.script
            def fn(x):
                FooTest(x)

            fn(2)

    def test_conditional_set_attr(self):
        with self.assertRaisesRegexWithHighlight(RuntimeError, "assignment cannot be in a control-flow block", ""):
            @torch.jit.script
            class FooTest(object):
                def __init__(self, x):
                    if 1 == 1:
                        self.attr = x

    def test_class_type_as_param(self):
        class FooTest(object):  # noqa: B903
            def __init__(self, x):
                self.attr = x

        make_global(FooTest)  # see [local resolution in python]

        @torch.jit.script
        def fn(foo: FooTest) -> torch.Tensor:
            return foo.attr

        @torch.jit.script
        def fn2(x):
            foo = FooTest(x)
            return fn(foo)

        input = torch.ones(1)
        self.assertEqual(fn2(input), input)

    def test_out_of_order_methods(self):
        class FooTest(object):
            def __init__(self, x):
                self.x = x
                self.x = self.get_stuff(x)

            def get_stuff(self, y):
                return self.x + y

        @torch.jit.script
        def fn(x):
            f = FooTest(x)
            return f.x

        input = torch.ones(1)
        self.assertEqual(fn(input), input + input)

    def test_save_load_with_classes(self):
        class FooTest(object):
            def __init__(self, x):
                self.x = x

            def get_x(self):
                return self.x

        class MyMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                foo = FooTest(a)
                return foo.get_x()

        m = MyMod()

        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # classes are globally registered for now, so we need to clear the JIT
        # registry to simulate loading a new model


        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(input, output)

    def test_save_load_with_classes_returned(self):
        class FooTest(object):
            def __init__(self, x):
                self.x = x

            def clone(self):
                clone = FooTest(self.x)
                return clone

        class MyMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                foo = FooTest(a)
                foo_clone = foo.clone()
                return foo_clone.x

        m = MyMod()

        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # classes are globally registered for now, so we need to clear the JIT
        # registry to simulate loading a new model
        torch.testing._internal.jit_utils.clear_class_registry()

        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(input, output)

    def test_save_load_with_classes_nested(self):
        class FooNestedTest(object):  # noqa: B903
            def __init__(self, y):
                self.y = y

        class FooNestedTest2(object):
            def __init__(self, y):
                self.y = y
                self.nested = FooNestedTest(y)

        class FooTest(object):
            def __init__(self, x):
                self.class_attr = FooNestedTest(x)
                self.class_attr2 = FooNestedTest2(x)
                self.x = self.class_attr.y + self.class_attr2.y

        class MyMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                foo = FooTest(a)
                return foo.x

        m = MyMod()

        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # classes are globally registered for now, so we need to clear the JIT
        # registry to simulate loading a new model
        torch.testing._internal.jit_utils.clear_class_registry()

        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(2 * input, output)

    def test_python_interop(self):
        class Foo(object):  # noqa: B903
            def __init__(self, x, y):
                self.x = x
                self.y = y

        make_global(Foo)   # see [local resolution in python]

        @torch.jit.script
        def use_foo(foo: Foo) -> Foo:
            return foo

        # create from python
        x = torch.ones(2, 3)
        y = torch.zeros(2, 3)
        f = Foo(x, y)

        self.assertEqual(x, f.x)
        self.assertEqual(y, f.y)

        # pass in and out of script
        f2 = use_foo(f)

        self.assertEqual(x, f2.x)
        self.assertEqual(y, f2.y)

    def test_class_specialization(self):
        class Foo(object):  # noqa: B903
            def __init__(self, x, y):
                self.x = x
                self.y = y

        make_global(Foo)  # see [local resolution in python]

        def use_foo(foo: Foo, foo2: Foo, tup: Tuple[Foo, Foo]) -> torch.Tensor:
            a, b = tup
            return foo.x + foo2.y + a.x + b.y

        # create from python
        x = torch.ones(2, 3)
        y = torch.zeros(2, 3)
        f = Foo(x, y)
        f2 = Foo(x * 2, y * 3)
        f3 = Foo(x * 4, y * 4)

        input = (f, f2, (f, f3))
        sfoo = self.checkScript(use_foo, input)
        graphstr = str(sfoo.graph_for(*input))
        FileCheck().check_count("prim::GetAttr", 4).run(graphstr)

    def test_class_sorting(self):
        class Foo(object):  # noqa: B903
            def __init__(self, x: int) -> None:
                self.x = x

            def __lt__(self, other) -> bool:
                # type: (Foo) -> bool
                return self.x < other.x

            def getVal(self):
                return self.x

        make_global(Foo)  # see [local resolution in python]

        def test(li: List[Foo], reverse: bool = False) -> Tuple[List[int], List[int]]:
            li_sorted = sorted(li)
            ret_sorted = torch.jit.annotate(List[int], [])
            for foo in li_sorted:
                ret_sorted.append(foo.getVal())

            li.sort(reverse=reverse)
            ret_sort = torch.jit.annotate(List[int], [])
            for foo in li:
                ret_sort.append(foo.getVal())
            return ret_sorted, ret_sort

        self.checkScript(test, ([Foo(2), Foo(1), Foo(3)],))
        self.checkScript(test, ([Foo(2), Foo(1), Foo(3)], True))
        self.checkScript(test, ([Foo(2)],))
        self.checkScript(test, ([],))

        @torch.jit.script
        def test_list_no_reverse():
            li = [Foo(3), Foo(1)]
            li.sort()
            return li[0].getVal()

        self.assertEqual(test_list_no_reverse(), 1)

        @torch.jit.script
        def test_sorted_copies():
            li = [Foo(3), Foo(1)]
            li_sorted = sorted(li)
            return li[0].getVal(), li_sorted[0].getVal()

        self.assertEqual(test_sorted_copies(), (3, 1))

        @torch.jit.script
        def test_nested_inside_tuple():
            li = [(1, Foo(12)), (1, Foo(11))]
            li.sort()
            return [(li[0][0], li[0][1].getVal()), (li[1][0], li[1][1].getVal())]

        self.assertEqual(test_nested_inside_tuple(), [(1, 11), (1, 12)])

        with self.assertRaisesRegexWithHighlight(RuntimeError, "bool\' for argument \'reverse", ""):
            @torch.jit.script
            def test():
                li = [Foo(1)]
                li.sort(li)
                return li
            test()

        with self.assertRaisesRegexWithHighlight(RuntimeError, "must define a __lt__", ""):
            @torch.jit.script
            class NoMethod(object):
                def __init__(self):
                    pass

            @torch.jit.script
            def test():
                li = [NoMethod(), NoMethod()]
                li.sort()
                return li
            test()

        @torch.jit.script
        class WrongLt(object):
            def __init__(self):
                pass

            # lt method defined with the wrong signature
            def __lt__(self, other):
                pass

        with self.assertRaisesRegexWithHighlight(RuntimeError, "must define a __lt__", ""):
            @torch.jit.script
            def test():
                li = [WrongLt(), WrongLt()]
                li.sort()
                return li
            test()

    def test_class_inheritance(self):
        @torch.jit.script
        class Base(object):
            def __init__(self):
                self.b = 2

            def two(self, x):
                return x + self.b

        with self.assertRaisesRegexWithHighlight(RuntimeError, "does not support inheritance", ""):
            @torch.jit.script
            class Derived(Base):
                def two(self, x):
                    return x + self.b + 2


    def test_class_inheritance_implicit(self):
        """
        Test that inheritance is detected in
        implicit scripting codepaths (e.g. try_ann_to_type).
        """
        class A:
            def __init__(self, t):
                self.t = t

            @staticmethod
            def f(a: torch.Tensor):
                return A(a + 1)

        class B(A):
            def __init__(self, t):
                self.t = t + 10

            @staticmethod
            def f(a: torch.Tensor):
                return A(a + 1)

        x = A(torch.tensor([3]))

        def fun(x: Any):
            if isinstance(x, A):
                return A.f(x.t)
            else:
                return B.f(x.t)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "object has no attribute or method", ""):
            sc = torch.jit.script(fun)

    @unittest.skipIf(IS_SANDCASTLE, "Importing like this doesn't work in fbcode")
    def test_imported_classes(self):
        import jit._imported_class_test.foo
        import jit._imported_class_test.bar
        import jit._imported_class_test.very.very.nested

        class MyMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                foo = jit._imported_class_test.foo.FooSameName(a)
                bar = jit._imported_class_test.bar.FooSameName(a)
                three = jit._imported_class_test.very.very.nested.FooUniqueName(a)
                return foo.x + bar.y + three.y

        m = MyMod()

        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # classes are globally registered for now, so we need to clear the JIT
        # registry to simulate loading a new model
        torch.testing._internal.jit_utils.clear_class_registry()

        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(3 * input, output)

    def test_interface(self):
        @torch.jit.script
        class Foo(object):
            def __init__(self):
                pass

            def one(self, x, y):
                return x + y

            def two(self, x):
                return 2 * x

        @torch.jit.script
        class Bar(object):
            def __init__(self):
                pass

            def one(self, x, y):
                return x * y

            def two(self, x):
                return 2 / x

        @torch.jit.interface
        class OneTwo(object):
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                pass

            def two(self, x: torch.Tensor) -> torch.Tensor:
                pass

        @torch.jit.interface
        class OneTwoThree(object):
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                pass

            def two(self, x: torch.Tensor) -> torch.Tensor:
                pass

            def three(self, x: torch.Tensor) -> torch.Tensor:
                pass

        @torch.jit.interface
        class OneTwoWrong(object):
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                pass

            def two(self, x: int) -> int:
                pass

        @torch.jit.script
        class NotMember(object):
            def __init__(self):
                pass

            def one(self, x, y):
                return x + y
            # missing two

        @torch.jit.script
        class NotMember2(object):
            def __init__(self):
                pass

            def one(self, x, y):
                return x + y

            def two(self, x: int) -> int:
                return 3

        make_global(Foo, Bar, OneTwo, OneTwoThree, OneTwoWrong, NotMember, NotMember2)

        def use_them(x):
            a = Foo()
            b = Bar()
            c = torch.jit.annotate(List[OneTwo], [a, b])
            for i in range(len(c)):
                x = c[i].one(x, x)
                x = c[i].two(x)
            return x
        self.checkScript(use_them, (torch.rand(3, 4),))

        @torch.jit.script
        def as_interface(x: OneTwo) -> OneTwo:
            return x

        @torch.jit.script
        def inherit(x: OneTwoThree) -> OneTwo:
            return as_interface(x)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "does not have method", ""):
            @torch.jit.script
            def wrong1():
                return as_interface(NotMember())

        with self.assertRaisesRegexWithHighlight(RuntimeError, "is not compatible with interface", ""):
            @torch.jit.script
            def wrong2():
                return as_interface(NotMember2())

        with self.assertRaisesRegexWithHighlight(RuntimeError, "does not have method", ""):
            @torch.jit.script
            def wrong3():
                return inherit(as_interface(Foo()))

        with self.assertRaisesRegexWithHighlight(RuntimeError, "is not compatible with interface", ""):

            @torch.jit.script
            def wrong4(x: OneTwoWrong) -> int:
                return as_interface(x)

        # Test interface/class python assignment
        class TestPyAssign(nn.Module):
            def __init__(self):
                super(TestPyAssign, self).__init__()
                self.proxy_mod = Foo()

            def forward(self, x):
                return self.proxy_mod.two(x)

        TestPyAssign.__annotations__ = {'proxy_mod': OneTwo}

        input = torch.rand(3, 4)
        scripted_pyassign_mod = torch.jit.script(TestPyAssign())
        imported_mod = self.getExportImportCopy(scripted_pyassign_mod)
        self.assertEqual(scripted_pyassign_mod(input), imported_mod(input))

        class TestPyAssignError(nn.Module):
            def __init__(self, obj):
                super(TestPyAssignError, self).__init__()
                self.proxy_mod = obj

            def forward(self, x):
                return self.proxy_mod.two(x)

        TestPyAssignError.__annotations__ = {'proxy_mod': OneTwoThree}

        with self.assertRaisesRegexWithHighlight(RuntimeError,
                                                 "is not compatible with interface __torch__", ""):
            torch.jit.script(TestPyAssignError(Foo()))

        # test pure python object assignment to interface fails
        class PyClass(object):
            def __init__(self):
                pass

        with self.assertRaisesRegexWithHighlight(RuntimeError,
                                                 "the value is not a TorchScript compatible type", ""):
            torch.jit.script(TestPyAssignError(PyClass()))
        # TODO test: interface-interface class-interface inheritance errors,
        # NamedTuple inheritance errors

    def test_overloaded_fn(self):
        @torch.jit.script
        class Foo(object):
            def __init__(self, x):
                self.x = x

            def __len__(self) -> int:
                return len(self.x)

            def __neg__(self):
                self.x = -self.x
                return self

            def __mul__(self, other: torch.Tensor) -> torch.Tensor:
                return self.x * other

        def test_overload():
            a = Foo(torch.ones([3, 3]))
            return len(a), -a * torch.zeros([3, 3])

        make_global(Foo)  # see [local resolution in python]

        self.checkScript(test_overload, ())
        # unary ops tested above

        # TODO - support compiling classes from strings in jit.CompilationUnit
        @torch.jit.script
        class MyClass(object):
            def __init__(self, x: int) -> None:
                self.x = x

            def __add__(self, other: int) -> int:
                return self.x + other

            def __sub__(self, other: int) -> int:
                return self.x - other

            def __mul__(self, other: int) -> int:
                return self.x * other

            def __pow__(self, other: int) -> int:
                return int(self.x ** other)

            def __truediv__(self, other: int) -> float:
                return self.x / other

            def __mod__(self, other: int) -> int:
                return self.x % other

            def __ne__(self, other: int) -> bool:
                return self.x != other

            def __eq__(self, other: int) -> bool:
                return self.x == other

            def __lt__(self, other: int) -> bool:
                return self.x < other

            def __gt__(self, other: int) -> bool:
                return self.x > other

            def __le__(self, other: int) -> bool:
                return self.x <= other

            def __ge__(self, other: int) -> bool:
                return self.x >= other

            def __and__(self, other: int) -> int:
                return self.x & other

            def __or__(self, other: int) -> int:
                return self.x | other

            def __xor__(self, other: int) -> int:
                return self.x ^ other

            def __getitem__(self, other: int) -> int:
                return other + 1

            def __setitem__(self, idx: int, val: int) -> None:
                self.x = val * idx

            def __call__(self, val: int) -> int:
                return self.x * val * 3


        make_global(Foo)  # see [local resolution in python]

        def add():
            return MyClass(4) + 3
        def sub():  # noqa: E306
            return MyClass(4) - 3
        def mul():  # noqa: E306
            return MyClass(4) * 3
        def pow():  # noqa: E306
            return MyClass(4) ** 3
        def truediv():  # noqa: E306
            return MyClass(4) / 3
        def ne():  # noqa: E306
            return MyClass(4) != 3
        def eq():  # noqa: E306
            return MyClass(4) == 3
        def lt():  # noqa: E306
            return MyClass(4) < 3
        def gt():  # noqa: E306
            return MyClass(4) > 3
        def le():  # noqa: E306
            return MyClass(4) <= 3
        def ge():  # noqa: E306
            return MyClass(4) >= 3
        def _and():  # noqa: E306
            return MyClass(4) & 3
        def _or():  # noqa: E306
            return MyClass(4) | 3
        def _xor():  # noqa: E306
            return MyClass(4) ^ 3
        def getitem():  # noqa: E306
            return MyClass(4)[1]
        def setitem():  # noqa: E306
            a = MyClass(4)
            a[1] = 5
            return a.x
        def call():  # noqa: E306
            a = MyClass(5)
            return a(2)

        ops = [add, sub, mul, pow, ne, eq, lt, gt, le, ge, _and, _or, _xor, getitem, setitem, call]

        ops.append(truediv)
        for func in ops:
            self.checkScript(func, ())

        with self.assertRaisesRegexWithHighlight(RuntimeError, "object has no attribute or method", ""):
            @torch.jit.script
            def test():
                return Foo(torch.tensor(1)) + Foo(torch.tensor(1))

    def test_cast_overloads(self):
        @torch.jit.script
        class Foo(object):
            def __init__(self, val: float) -> None:
                self.val = val

            def __int__(self):
                return int(self.val)

            def __float__(self):
                return self.val

            def __bool__(self):
                return bool(self.val)

            def __str__(self):
                return str(self.val)

        make_global(Foo)  # see [local resolution in python]

        def test(foo: Foo) -> Tuple[int, float, bool]:
            if foo:
                pass
            return int(foo), float(foo), bool(foo)

        fn = torch.jit.script(test)
        self.assertEqual(fn(Foo(0.5)), test(0.5))
        self.assertEqual(fn(Foo(0.)), test(0.0))
        # str has slightly different formatting
        self.assertTrue("0.5" in (str(Foo(0.5))))
        self.assertTrue("0." in (str(Foo(0.0))))

        @torch.jit.script
        class BadBool(object):
            def __init__(self):
                pass

            def __bool__(self):
                return (1, 2)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "expected a bool expression for condition", ""):
            @torch.jit.script
            def test():
                if BadBool():
                    print(1)
                    pass

    def test_init_compiled_first(self):
        @torch.jit.script  # noqa: B903
        class Foo(object):  # noqa: B903
            def __before_init__(self):
                # accessing this field should not throw, since __init__ should be compiled
                return self.x

            def __init__(self, x, y):
                self.x = x
                self.y = y

    def test_class_constructs_itself(self):
        @torch.jit.script  # noqa: B903
        class LSTMStateStack(object):  # noqa: B903
            def __init__(self, num_layers: int, hidden_size: int) -> None:
                self.num_layers = num_layers
                self.hidden_size = hidden_size
                self.last_state = (
                    torch.zeros(num_layers, 1, hidden_size),
                    torch.zeros(num_layers, 1, hidden_size),
                )
                self.stack = [(self.last_state[0][-1], self.last_state[0][-1])]

            def copy(self):
                # should be able to construct a class inside its own methods
                other = LSTMStateStack(self.num_layers, self.hidden_size)
                other.stack = list(self.stack)
                return other

    def test_optional_type_promotion(self):
        @torch.jit.script
        class Leaf(object):
            def __init__(self):
                self.x = 1

        # should not throw
        @torch.jit.script  # noqa: B903
        class Tree(object):  # noqa: B903
            def __init__(self):
                self.child = torch.jit.annotate(Optional[Leaf], None)

            def add_child(self, child: Leaf) -> None:
                self.child = child

    def test_recursive_class(self):
        """
        Recursive class types not yet supported. We should give a good error message.
        """
        with self.assertRaises(RuntimeError):
            @torch.jit.script  # noqa: B903
            class Tree(object):  # noqa: B903
                def __init__(self):
                    self.parent = torch.jit.annotate(Optional[Tree], None)

    def test_class_constant(self):
        class M(torch.nn.Module):
            __constants__ = ["w"]

            def __init__(self, w):
                super(M, self).__init__()
                self.w = w

            def forward(self, x):
                # Make sure class constant is accessible in method
                y = self.w
                return x, y

        # Test serialization/deserialization of class constant
        for c in (2, 1.0, None, True, 'str', (2, 3), [5.9, 7.3]):
            m = torch.jit.script(M(c))
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)

            buffer.seek(0)
            m_loaded = torch.jit.load(buffer)
            input = torch.rand(2, 3)
            self.assertEqual(m(input), m_loaded(input))
            # Make sure class constant is accessible from module
            self.assertEqual(m.w, m_loaded.w)

    def test_py_class_to_ivalue_missing_attribute(self):
        class Foo(object):
            i : int
            f : float

            def __init__(self, i : int, f : float):
                self.i = i
                self.f = f

        make_global(Foo)  # see [local resolution in python]

        @torch.jit.script
        def test_fn(x : Foo) -> float:
            return x.i + x.f

        test_fn(Foo(3, 4.0))

        with self.assertRaisesRegexWithHighlight(RuntimeError, 'missing attribute i', ""):
            test_fn(torch.rand(3, 4))

    def test_unused_method(self):
        """
        Test unused methods on scripted classes.
        """
        @torch.jit.script
        class Unused(object):
            def __init__(self):
                self.count: int = 0
                self.items: List[int] = []

            def used(self):
                self.count += 1
                return self.count

            @torch.jit.unused
            def unused(self, x: int, y: Iterable[int], **kwargs) -> int:
                a = next(self.items)
                return a

            def uses_unused(self) -> int:
                return self.unused(y="hi", x=3)

        class ModuleWithUnused(nn.Module):
            def __init__(self):
                super().__init__()
                self.obj = Unused()

            def forward(self):
                return self.obj.used()

            @torch.jit.export
            def calls_unused(self):
                return self.obj.unused(3, "hi")

            @torch.jit.export
            def calls_unused_indirectly(self):
                return self.obj.uses_unused()

        python_module = ModuleWithUnused()
        script_module = torch.jit.script(ModuleWithUnused())

        # Forward should work because it does not used any methods marked unused.
        self.assertEqual(python_module.forward(), script_module.forward())

        # Calling a method marked unused should throw.
        with self.assertRaises(torch.jit.Error):
            script_module.calls_unused()

        with self.assertRaises(torch.jit.Error):
            script_module.calls_unused_indirectly()

    def test_self_referential_method(self):
        """
        Test that a scripted class can have a method that refers to the class itself
        in its type annotations.
        """
        @torch.jit.script
        class Meta(object):
            def __init__(self, a: int):
                self.a = a

            def method(self, other: List['Meta']) -> 'Meta':
                return Meta(len(other))

        class ModuleWithMeta(torch.nn.Module):
            def __init__(self, a: int):
                super().__init__()
                self.meta = Meta(a)

            def forward(self):
                new_obj = self.meta.method([self.meta])
                return new_obj.a

        self.checkModule(ModuleWithMeta(5), ())

    def test_type_annotation(self):
        """
        Test that annotating container attributes with types works correctly
        """
        @torch.jit.script
        class CompetitiveLinkingTokenReplacementUtils:
            def __init__(self):
                self.my_list : List[Tuple[float, int, int]] = []
                self.my_dict : Dict[int, int] = {}

        @torch.jit.script
        def foo():
            y = CompetitiveLinkingTokenReplacementUtils()
            new_dict : Dict[int, int] = {1: 1, 2: 2}
            y.my_dict = new_dict

            new_list : List[Tuple[float, int, int]] = [(1.0, 1, 1)]
            y.my_list = new_list
            return y

    def test_default_args(self):
        """
        Test that methods on class types can have default arguments.
        """
        @torch.jit.script
        class ClassWithDefaultArgs:
            def __init__(
                self,
                a: int = 1,
                b: Optional[List[int]] = None,
                c: Tuple[int, int, int] = (1, 2, 3),
                d: Optional[Dict[int, int]] = None,
                e: Optional[str] = None,
            ):
                self.int = a
                self.tup = c
                self.str = e

                self.list = [1, 2, 3]
                if b is not None:
                    self.list = b

                self.dict = {1: 2, 3: 4}
                if d is not None:
                    self.dict = d

            def add(self, b: int, scale: float = 1.0) -> float:
                return self.int * scale + b

        def all_defaults() -> int:
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs()
            return obj.int + obj.list[2] + obj.tup[1]

        def some_defaults() -> int:
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs(b=[5, 6, 7])
            return obj.int + obj.list[2] + obj.dict[1]

        def override_defaults() -> int:
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs(3, [9, 10, 11], (12, 13, 14), {3: 4}, "str")
            s: int = obj.int

            for x in obj.list:
                s += x

            for y in obj.tup:
                s += y

            s += obj.dict[3]

            st = obj.str
            if st is not None:
                s += len(st)

            return s

        def method_defaults() -> float:
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs()
            return obj.add(3) + obj.add(3, 0.25)

        self.checkScript(all_defaults, ())
        self.checkScript(some_defaults, ())
        self.checkScript(override_defaults, ())
        self.checkScript(method_defaults, ())

        # The constructor of this class below has some arguments without default values.
        class ClassWithSomeDefaultArgs:  # noqa: B903
            def __init__(
                self,
                a: int,
                b: int = 1,
            ):
                self.a = a
                self.b = b

        def default_b() -> int:
            obj: ClassWithSomeDefaultArgs = ClassWithSomeDefaultArgs(1)
            return obj.a + obj.b

        def set_b() -> int:
            obj: ClassWithSomeDefaultArgs = ClassWithSomeDefaultArgs(1, 4)
            return obj.a + obj.b

        self.checkScript(default_b, ())
        self.checkScript(set_b, ())

        # The constructor of this class below has mutable arguments. This should throw
        # an error.
        class ClassWithMutableArgs:   # noqa: B903
            def __init__(
                self,
                a: List[int] = [1, 2, 3],  # noqa: B006
            ):
                self.a = a

        def should_fail():
            obj: ClassWithMutableArgs = ClassWithMutableArgs()

        with self.assertRaisesRegexWithHighlight(RuntimeError, "Mutable default parameters are not supported", ""):
            torch.jit.script(should_fail)

    def test_staticmethod(self):
        """
        Test static methods on class types.
        """
        @torch.jit.script
        class ClassWithStaticMethod:
            def __init__(self, a: int, b: int):
                self.a: int = a
                self.b: int = b

            def get_a(self):
                return self.a

            def get_b(self):
                return self.b

            def __eq__(self, other: 'ClassWithStaticMethod'):
                return self.a == other.a and self.b == other.b

            # staticmethod that calls constructor.
            @staticmethod
            def create(args: List['ClassWithStaticMethod']) -> 'ClassWithStaticMethod':
                return ClassWithStaticMethod(args[0].a, args[0].b)

            # staticmethod that calls another staticmethod.
            @staticmethod
            def create_from(a: int, b: int) -> 'ClassWithStaticMethod':
                a = ClassWithStaticMethod(a, b)
                return ClassWithStaticMethod.create([a])

        # Script function that calls staticmethod.
        def test_function(a: int, b: int) -> 'ClassWithStaticMethod':
            return ClassWithStaticMethod.create_from(a, b)

        make_global(ClassWithStaticMethod)

        self.checkScript(test_function, (1, 2))

    def test_classmethod(self):
        """
        Test classmethods on class types.
        """
        @torch.jit.script
        class ClassWithClassMethod:
            def __init__(self, a: int):
                self.a: int = a

            def __eq__(self, other: 'ClassWithClassMethod'):
                return self.a == other.a

            @classmethod
            def create(cls, a: int) -> 'ClassWithClassMethod':
                return cls(a)

        make_global(ClassWithClassMethod)

        def test_function(a: int) -> 'ClassWithClassMethod':
            x = ClassWithClassMethod(a)
            # Support calling classmethod with an instance
            # Calling with the class is not supported.
            return x.create(a)

        self.checkScript(test_function, (1,))

    def test_properties(self):
        """
        Test that a scripted class can make use of the @property decorator.
        """
        def free_function(x: int) -> int:
            return x + 1

        @torch.jit.script
        class Properties(object):
            __jit_unused_properties__ = ["unsupported"]

            def __init__(self, a: int):
                self.a = a

            @property
            def attr(self) -> int:
                return self.a - 1

            @property
            def unsupported(self) -> int:
                return sum([self.a])

            @torch.jit.unused
            @property
            def unsupported_2(self) -> int:
                return sum([self.a])

            @unsupported_2.setter
            def unsupported_2(self, value):
                self.a = sum([self.a])

            @attr.setter
            def attr(self, value: int):
                self.a = value + 3

        @torch.jit.script
        class NoSetter(object):
            def __init__(self, a: int):
                self.a = a

            @property
            def attr(self) -> int:
                return free_function(self.a)

        @torch.jit.script
        class MethodThatUsesProperty(object):
            def __init__(self, a: int):
                self.a = a

            @property
            def attr(self) -> int:
                return self.a - 2

            @attr.setter
            def attr(self, value: int):
                self.a = value + 4

            def forward(self):
                return self.attr

        class ModuleWithProperties(torch.nn.Module):
            def __init__(self, a: int):
                super().__init__()
                self.props = Properties(a)

            def forward(self, a: int, b: int, c: int, d: int):
                self.props.attr = a
                props = Properties(b)
                no_setter = NoSetter(c)
                method_uses_property = MethodThatUsesProperty(a + b)

                props.attr = c
                method_uses_property.attr = d

                return self.props.attr + no_setter.attr + method_uses_property.forward()

        self.checkModule(ModuleWithProperties(5), (5, 6, 7, 8,))

    def test_custom_delete(self):
        """
        Test that del can be called on an instance of a class that
        overrides __delitem__.
        """
        class Example(object):
            def __init__(self):
                self._data: Dict[str, torch.Tensor] = {"1": torch.tensor(1.0)}

            def check(self, key: str) -> bool:
                return key in self._data

            def __delitem__(self, key: str):
                del self._data[key]

        def fn() -> bool:
            example = Example()
            del example["1"]
            return example.check("1")

        self.checkScript(fn, ())

        # Test the case in which the class does not have __delitem__ defined.
        class NoDelItem(object):
            def __init__(self):
                self._data: Dict[str, torch.Tensor] = {"1": torch.tensor(1.0)}

            def check(self, key: str) -> bool:
                return key in self._data

        def fn() -> bool:
            example = NoDelItem()
            key = "1"
            del example[key]
            return example.check(key)

        with self.assertRaisesRegexWithHighlight(RuntimeError, r"Class does not define __delitem__", "example[key]"):
            self.checkScript(fn, ())

    def test_recursive_script_builtin_type_resolution(self):
        """
        Test resolution of built-in torch types(e.g. torch.Tensor, torch.device) when a class is recursively compiled.
        """
        # A will be implicitly compiled because it is not annotated with @torch.jit.script
        # but is used in g() below.
        tensor_t = torch.Tensor
        device_t = torch.device
        device_ty = torch.device

        class A(object):
            def __init__(self):
                pass

            def f(self, x: tensor_t, y: torch.device) -> tensor_t:
                return x.to(device=y)

            def g(self, x: device_t) -> device_ty:
                return x

            def h(self, a: 'A') -> 'A':
                return A()

            def i(self, a: List[int]) -> int:
                return a[0]

            def j(self, l: List[device_t]) -> device_ty:
                return l[0]

        def call_f():
            a = A()
            return a.f(torch.tensor([1]), torch.device("cpu"))

        def call_g():
            a = A()
            return a.g(torch.device("cpu"))

        def call_i():
            a = A()
            return a.i([3])

        def call_j():
            a = A()
            return a.j([torch.device("cpu"), torch.device("cpu")])

        for fn in [call_f, call_g, call_i, call_j]:
            self.checkScript(fn, ())
            s = self.getExportImportCopy(torch.jit.script(fn))
            self.assertEqual(s(), fn())

    def test_recursive_script_module_builtin_type_resolution(self):
        """
        Test resolution of built-in torch types(e.g. torch.Tensor, torch.device) when a class is recursively compiled
        when compiling a module.
        """
        class Wrapper():
            def __init__(self, t):
                self.t = t

            def to(self, l: List[torch.device], device: Optional[torch.device] = None):
                return self.t.to(device=device)


        class A(nn.Module):
            def forward(self):
                return Wrapper(torch.rand(4, 4))

        scripted = torch.jit.script(A())
        self.getExportImportCopy(scripted)

    def test_class_attribute_wrong_type(self):
        """
        Test that the error message displayed when convering a class type
        to an IValue that has an attribute of the wrong type.
        """
        @torch.jit.script
        class ValHolder(object):  # noqa: B903
            def __init__(self, val):
                self.val = val

        class Mod(nn.Module):
            def __init__(self):
                super(Mod, self).__init__()
                self.mod1 = ValHolder(1)
                self.mod2 = ValHolder(2)

            def forward(self, cond: bool):
                if cond:
                    mod = self.mod1
                else:
                    mod = self.mod2
                return mod.val

        with self.assertRaisesRegexWithHighlight(RuntimeError, "Could not cast attribute 'val' to type Tensor", ""):
            torch.jit.script(Mod())

    def test_recursive_scripting(self):
        """
        Test that class types are recursively scripted when an Python instance of one
        is encountered as a module attribute.
        """
        class Class(object):
            def __init__(self, a: int):
                self.a = a

            def get_a(self) -> int:
                return self.a

        class M(torch.nn.Module):
            def __init__(self, obj):
                super().__init__()
                self.obj = obj

            def forward(self) -> int:
                return self.obj.get_a()

        self.checkModule(M(Class(4)), ())

    def test_recursive_scripting_failed(self):
        """
        Test that class types module attributes that fail to script
        are added as failed attributes and do not cause compilation itself
        to fail unless they are used in scripted code.
        """
        class UnscriptableClass(object):
            def __init__(self, a: int):
                self.a = a

            def get_a(self) -> bool:
                return issubclass(self.a, int)

        # This Module has an attribute of type UnscriptableClass
        # and tries to use it in scripted code. This should fail.
        class ShouldNotCompile(torch.nn.Module):
            def __init__(self, obj):
                super().__init__()
                self.obj = obj

            def forward(self) -> bool:
                return self.obj.get_a()

        with self.assertRaisesRegexWithHighlight(RuntimeError, "failed to convert Python type", ""):
            torch.jit.script(ShouldNotCompile(UnscriptableClass(4)))

        # This Module has an attribute of type UnscriptableClass
        # and does not try to use it in scripted code. This should not fail.
        class ShouldCompile(torch.nn.Module):
            def __init__(self, obj):
                super().__init__()
                self.obj = obj

            @torch.jit.ignore
            def ignored_method(self) -> bool:
                return self.obj.get_a()

            def forward(self, x: int) -> int:
                return x + x

        self.checkModule(ShouldCompile(UnscriptableClass(4)), (4,))


    def test_unresolved_class_attributes(self):
        class UnresolvedAttrClass(object):
            def __init__(self):
                pass

            (attr_a, attr_b), [attr_c, attr_d] = ("", ""), ["", ""]
            attr_e: int = 0

        def fn_a():
            u = UnresolvedAttrClass()
            return u.attr_a

        def fn_b():
            u = UnresolvedAttrClass()
            return u.attr_b

        def fn_c():
            u = UnresolvedAttrClass()
            return u.attr_c

        def fn_d():
            u = UnresolvedAttrClass()
            return u.attr_d

        def fn_e():
            u = UnresolvedAttrClass()
            return u.attr_e

        error_message_regex = "object has no attribute or method.*is defined as a class attribute"
        for fn in (fn_a, fn_b, fn_c, fn_d, fn_e):
            with self.assertRaisesRegex(RuntimeError, error_message_regex):
                torch.jit.script(fn)
