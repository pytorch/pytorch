from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import io
from common_utils import run_tests
from test_jit import JitTestCase
from torch.testing import FileCheck


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
        class FooTest(object):
            def __init__(self, x):
                self.foo = x

        @torch.jit.script
        def fn(x):
            foo = FooTest(x)
            return foo.foo

        input = torch.ones(2, 3)
        self.assertEqual(fn(input), input)

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

    def test_set_attr_type_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "Wrong type for attribute assignment"):
            @torch.jit.script
            class FooTest(object):
                def __init__(self, x):
                    self.foo = x
                    self.foo = 10  # should error since int != Tensor

    def test_get_attr_not_initialized(self):
        with self.assertRaisesRegex(RuntimeError, "Tried to access to nonexistent attribute"):
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

    def test_type_annotations(self):
        with self.assertRaisesRegex(RuntimeError, "expected a value of type bool"):
            @torch.jit.script  # noqa: B903
            class FooTest(object):
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
        @torch.jit.script  # noqa: B903
        class FooTest(object):
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
        torch._C._jit_clear_class_registry()

        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(input, output)

    def test_save_load_with_classes_nested(self):
        @torch.jit.script  # noqa: B903
        class FooNestedTest(object):
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
        torch._C._jit_clear_class_registry()

        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(2 * input, output)

    def test_python_interop(self):
        @torch.jit.script  # noqa: B903
        class Foo(object):
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
        @torch.jit.script  # noqa: B903
        class Foo(object):
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
        FileCheck().check_count("Double(*, *) = prim::GetAttr", 4).run(graphstr)

if __name__ == '__main__':
    run_tests()
