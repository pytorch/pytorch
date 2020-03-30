from __future__ import division
import io
import os
import sys
import unittest

import torch
import torch.nn as nn
from torch._six import PY2
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
import torch.testing._internal.jit_utils
from torch.testing._internal.common_utils import IS_SANDCASTLE
from typing import List

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestClassType(JitTestCase):
    def test_get_with_method(self):
        @torch.jit.script
        class FooTest(object):
            def __init__(self, x):
                self.foo = x

            def getFooTest(self):
                return self.foo

        @torch.jit.script
        def fn(x):
            foo = FooTest(x)
            return foo.getFooTest()

        input = torch.ones(2, 3)
        self.assertEqual(fn(input), input)

    def test_get_attr(self):
        @torch.jit.script  # noqa: B903
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
        @torch.jit.script  # noqa: B903
        class FooTest(object):  # noqa: B903
            def __init__(self):
                pass

            def __contains__(self, key):
                # type: (str) -> bool
                return key == 'hi'

        @torch.jit.script
        def fn():
            foo = FooTest()
            return 'hi' in foo, 'no' in foo

        self.assertEqual(fn(), (True, False))

    def test_set_attr_in_method(self):
        @torch.jit.script
        class FooTest(object):
            def __init__(self, x):
                # type: (int) -> None
                self.foo = x

            def incFooTest(self, y):
                # type: (int) -> None
                self.foo = self.foo + y

        @torch.jit.script
        def fn(x):
            # type: (int) -> int
            foo = FooTest(x)
            foo.incFooTest(2)
            return foo.foo

        self.assertEqual(fn(1), 3)

    def test_staticmethod(self):
        class X(object):
            def __init__(self, x):
                # type: (int) -> None
                self.x = x

            @staticmethod
            def identity(x):
                return x

        def fn(x, y):
            return X.identity(x)

        self.checkScript(fn, (torch.randn(2, 2), torch.randn(2, 2)))

    def test_set_attr_type_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "Wrong type for attribute assignment"):
            @torch.jit.script
            class FooTest(object):
                def __init__(self, x):
                    self.foo = x
                    self.foo = 10  # should error since int != Tensor

    def test_get_attr_not_initialized(self):
        with self.assertRaisesRegex(RuntimeError, "Tried to access nonexistent attribute"):
            @torch.jit.script
            class FooTest(object):
                def __init__(self, x):
                    self.foo = x

                def get_non_initialized(self):
                    return self.asdf  # asdf isn't an attr

    def test_set_attr_non_initialized(self):
        with self.assertRaisesRegex(RuntimeError, "Tried to set nonexistent attribute"):
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
        with self.assertRaisesRegex(RuntimeError, "nearest"):
            @torch.jit.script
            def FooTest(x):
                return torch.nn.functional.interpolate(x, 'bad')

    def test_type_annotations(self):
        with self.assertRaisesRegex(RuntimeError, "Expected a value of type \'bool"):
            @torch.jit.script  # noqa: B903
            class FooTest(object):  # noqa: B903
                def __init__(self, x):
                    # type: (bool) -> None
                    self.foo = x

            @torch.jit.script
            def fn(x):
                FooTest(x)

            fn(2)

    def test_conditional_set_attr(self):
        with self.assertRaisesRegex(RuntimeError, "assignment cannot be in a control-flow block"):
            @torch.jit.script
            class FooTest(object):
                def __init__(self, x):
                    if True:
                        self.attr = x

    def test_class_type_as_param(self):
        global FooTest  # see [local resolution in python]
        @torch.jit.script  # noqa: B903
        class FooTest(object):  # noqa: B903
            def __init__(self, x):
                self.attr = x

        @torch.jit.script
        def fn(foo):
            # type: (FooTest) -> Tensor
            return foo.attr

        @torch.jit.script
        def fn2(x):
            foo = FooTest(x)
            return fn(foo)

        input = torch.ones(1)
        self.assertEqual(fn2(input), input)

    def test_out_of_order_methods(self):
        @torch.jit.script
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
        @torch.jit.script
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
        @torch.jit.script
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
        @torch.jit.script  # noqa: B903
        class FooNestedTest(object):  # noqa: B903
            def __init__(self, y):
                self.y = y

        @torch.jit.script
        class FooNestedTest2(object):
            def __init__(self, y):
                self.y = y
                self.nested = FooNestedTest(y)

        @torch.jit.script
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
        global Foo   # see [local resolution in python]
        @torch.jit.script  # noqa: B903
        class Foo(object):  # noqa: B903
            def __init__(self, x, y):
                self.x = x
                self.y = y

        @torch.jit.script
        def use_foo(foo):
            # type: (Foo) -> Foo
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
        global Foo  # see [local resolution in python]
        @torch.jit.script  # noqa: B903
        class Foo(object):  # noqa: B903
            def __init__(self, x, y):
                self.x = x
                self.y = y

        def use_foo(foo, foo2, tup):
            # type: (Foo, Foo, Tuple[Foo, Foo]) -> Tensor
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
        global Foo  # see [local resolution in python]
        @torch.jit.script  # noqa: B903
        class Foo(object):  # noqa: B903
            def __init__(self, x):
                # type: (int) -> None
                self.x = x

            def __lt__(self, other):
                # type: (Foo) -> bool
                return self.x < other.x

            def getVal(self):
                return self.x

        def test(li, reverse=False):
            # type: (List[Foo], bool)
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

        with self.assertRaisesRegex(RuntimeError, "bool\' for argument \'reverse"):
            @torch.jit.script
            def test():
                li = [Foo(1)]
                li.sort(li)
                return li

        with self.assertRaisesRegex(RuntimeError, "must define a __lt__"):
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

        with self.assertRaisesRegex(RuntimeError, "must define a __lt__"):
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

        with self.assertRaisesRegex(RuntimeError, "does not support inheritance"):
            @torch.jit.script
            class Derived(Base):
                def two(self, x):
                    return x + self.b + 2

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
        global Foo, Bar, OneTwo, OneTwoThree, OneTwoWrong, NotMember, NotMember2
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
            def one(self, x, y):
                # type: (Tensor, Tensor) -> Tensor
                pass

            def two(self, x):
                # type: (Tensor) -> Tensor
                pass

        @torch.jit.interface
        class OneTwoThree(object):
            def one(self, x, y):
                # type: (Tensor, Tensor) -> Tensor
                pass

            def two(self, x):
                # type: (Tensor) -> Tensor
                pass

            def three(self, x):
                # type: (Tensor) -> Tensor
                pass

        @torch.jit.interface
        class OneTwoWrong(object):
            def one(self, x, y):
                # type: (Tensor, Tensor) -> Tensor
                pass

            def two(self, x):
                # type: (int) -> int
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

            def two(self, x):
                # type: (int) -> int
                return 3

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
        def as_interface(x):
            # type: (OneTwo) -> OneTwo
            return x

        @torch.jit.script
        def inherit(x):
            # type: (OneTwoThree) -> OneTwo
            return as_interface(x)

        with self.assertRaisesRegex(RuntimeError, "does not have method"):
            @torch.jit.script
            def wrong1():
                return as_interface(NotMember())

        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            @torch.jit.script
            def wrong2():
                return as_interface(NotMember2())

        with self.assertRaisesRegex(RuntimeError, "does not have method"):
            @torch.jit.script
            def wrong3():
                return inherit(as_interface(Foo()))

        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):

            @torch.jit.script
            def wrong4(x):
                # type: (OneTwoWrong) -> int
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

        with self.assertRaisesRegex(RuntimeError,
                                    "is not compatible with interface __torch__"):
            torch.jit.script(TestPyAssignError(Foo()))

        # test pure python object assignment to interface fails
        class PyClass(object):
            def __init__(self):
                pass

        with self.assertRaisesRegex(RuntimeError,
                                    "the value is not a TorchScript compatible type"):
            torch.jit.script(TestPyAssignError(PyClass()))
        # TODO test: interface-interface class-interface inheritance errors,
        # NamedTuple inheritance errors

    def test_overloaded_fn(self):
        global Foo, MyClass  # see [local resolution in python]
        @torch.jit.script
        class Foo(object):
            def __init__(self, x):
                self.x = x

            def __len__(self):
                # type: () -> int
                return len(self.x)

            def __neg__(self):
                self.x = -self.x
                return self

            def __mul__(self, other):
                # type: (Tensor) -> Tensor
                return self.x * other

        def test_overload():
            a = Foo(torch.ones([3, 3]))
            return len(a), -a * torch.zeros([3, 3])

        self.checkScript(test_overload, ())
        # unary ops tested above

        # TODO - support compiling classes from strings in jit.CompilationUnit
        @torch.jit.script
        class MyClass(object):
            def __init__(self, x):
                # type: (int) -> None
                self.x = x

            def __add__(self, other):
                # type: (int) -> int
                return self.x + other

            def __sub__(self, other):
                # type: (int) -> int
                return self.x - other

            def __mul__(self, other):
                # type: (int) -> int
                return self.x * other

            def __pow__(self, other):
                # type: (int) -> int
                return int(self.x ** other)

            def __truediv__(self, other):
                # type: (int) -> float
                return self.x / other

            def __mod__(self, other):
                # type: (int) -> int
                return self.x % other

            def __ne__(self, other):  # noqa T484
                # type: (int) -> bool
                return self.x != other

            def __eq__(self, other):  # noqa T484
                # type: (int) -> bool
                return self.x == other

            def __lt__(self, other):
                # type: (int) -> bool
                return self.x < other

            def __gt__(self, other):
                # type: (int) -> bool
                return self.x > other

            def __le__(self, other):
                # type: (int) -> bool
                return self.x <= other

            def __ge__(self, other):
                # type: (int) -> bool
                return self.x >= other

            def __and__(self, other):
                # type: (int) -> int
                return self.x & other

            def __or__(self, other):
                # type: (int) -> int
                return self.x | other

            def __xor__(self, other):
                # type: (int) -> int
                return self.x ^ other

            def __getitem__(self, other):
                # type: (int) -> int
                return other + 1

            def __setitem__(self, idx, val):
                # type: (int, int) -> None
                self.x = val * idx

            def __call__(self, val):
                # type: (int) -> int
                return self.x * val * 3


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

        if not PY2:
            ops.append(truediv)
        for func in ops:
            self.checkScript(func, ())

        with self.assertRaisesRegex(RuntimeError, "nonexistent attribute"):
            @torch.jit.script
            def test():
                return Foo(torch.tensor(1)) + Foo(torch.tensor(1))

    def test_cast_overloads(self):
        global Foo  # see [local resolution in python]
        @torch.jit.script
        class Foo(object):
            def __init__(self, val):
                # type: (float) -> None
                self.val = val

            def __int__(self):
                return int(self.val)

            def __float__(self):
                return self.val

            def __bool__(self):
                return bool(self.val)

            def __str__(self):
                return str(self.val)

        def test(foo):
            # type: (Foo) -> Tuple[int, float, bool]
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

        with self.assertRaisesRegex(RuntimeError, "expected a bool expression for condition"):
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
            def __init__(self, num_layers, hidden_size):
                # type: (int, int) -> None
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

            def add_child(self, child):
                # type: (Leaf) -> None
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
