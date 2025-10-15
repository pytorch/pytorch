# Owner(s): ["oncall: jit"]

import io
import os
import sys

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    suppress_warnings,
)
from torch.testing._internal.jit_utils import JitTestCase


class TestTypeSharing(JitTestCase):
    def assertSameType(self, m1, m2):
        if not isinstance(m1, torch.jit.ScriptModule):
            m1 = torch.jit.script(m1)
        if not isinstance(m2, torch.jit.ScriptModule):
            m2 = torch.jit.script(m2)
        self.assertEqual(m1._c._type(), m2._c._type())

    def assertDifferentType(self, m1, m2):
        if not isinstance(m1, torch.jit.ScriptModule):
            m1 = torch.jit.script(m1)
        if not isinstance(m2, torch.jit.ScriptModule):
            m2 = torch.jit.script(m2)
        self.assertNotEqual(m1._c._type(), m2._c._type())

    def test_basic(self):
        class M(torch.nn.Module):
            def __init__(self, a, b, c):
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                return x

        a = torch.rand(2, 3)
        b = torch.rand(2, 3)
        c = torch.rand(2, 3)
        m1 = M(a, b, c)
        m2 = M(a, b, c)
        self.assertSameType(m1, m2)

    def test_diff_attr_values(self):
        """
        Types should be shared even if attribute values differ
        """

        class M(torch.nn.Module):
            def __init__(self, a, b, c):
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                return x

        a = torch.rand(2, 3)
        b = torch.rand(2, 3)
        c = torch.rand(2, 3)
        m1 = M(a, b, c)
        m2 = M(a * 2, b * 3, c * 4)
        self.assertSameType(m1, m2)

    def test_constants(self):
        """
        Types should be shared for identical constant values, and different for different constant values
        """

        class M(torch.nn.Module):
            __constants__ = ["const"]

            def __init__(self, attr, const):
                super().__init__()
                self.attr = attr
                self.const = const

            def forward(self):
                return self.const

        attr = torch.rand(2, 3)
        m1 = M(attr, 1)
        m2 = M(attr, 1)
        self.assertSameType(m1, m2)

        # a different constant value
        m3 = M(attr, 2)
        self.assertDifferentType(m1, m3)

    def test_linear(self):
        """
        Simple example with a real nn Module
        """
        a = torch.nn.Linear(5, 5)
        b = torch.nn.Linear(5, 5)
        c = torch.nn.Linear(10, 10)
        a = torch.jit.script(a)
        b = torch.jit.script(b)
        c = torch.jit.script(c)

        self.assertSameType(a, b)
        self.assertDifferentType(a, c)

    def test_submodules(self):
        """
        If submodules differ, the types should differ.
        """

        class M(torch.nn.Module):
            def __init__(self, in1, out1, in2, out2):
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)

            def forward(self, x):
                x = self.submod1(x)
                x = self.submod2(x)
                return x

        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        self.assertSameType(a, b)
        self.assertSameType(a.submod1, b.submod1)
        c = M(2, 2, 2, 2)
        self.assertDifferentType(a, c)

        self.assertSameType(b.submod2, c.submod1)
        self.assertDifferentType(a.submod1, b.submod2)

    def test_param_vs_attribute(self):
        """
        The same module with an `foo` as a parameter vs. attribute shouldn't
        share types
        """

        class M(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def forward(self, x):
                return x + self.foo

        as_param = torch.nn.Parameter(torch.ones(2, 2))
        as_attr = torch.ones(2, 2)
        param_mod = M(as_param)
        attr_mod = M(as_attr)
        self.assertDifferentType(attr_mod, param_mod)

    def test_same_but_different_classes(self):
        """
        Even if everything about the module is the same, different originating
        classes should prevent type sharing.
        """

        class A(torch.nn.Module):
            __constants__ = ["const"]

            def __init__(self, in1, out1, in2, out2):
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.const = 5

            def forward(self, x):
                x = self.submod1(x)
                x = self.submod2(x)
                return x * self.const

        class B(torch.nn.Module):
            __constants__ = ["const"]

            def __init__(self, in1, out1, in2, out2):
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.const = 5

            def forward(self, x):
                x = self.submod1(x)
                x = self.submod2(x)
                return x * self.const

        a = A(1, 1, 2, 2)
        b = B(1, 1, 2, 2)
        self.assertDifferentType(a, b)

    def test_mutate_attr_value(self):
        """
        Mutating the value of an attribute should not change type sharing
        """

        class M(torch.nn.Module):
            def __init__(self, in1, out1, in2, out2):
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.foo = torch.ones(in1, in1)

            def forward(self, x):
                x = self.submod1(x)
                x = self.submod2(x)
                return x + self.foo

        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        a.foo = torch.ones(2, 2)
        b.foo = torch.rand(2, 2)
        self.assertSameType(a, b)

    def test_assign_python_attr(self):
        """
        Assigning a new (python-only) attribute should not change type sharing
        """

        class M(torch.nn.Module):
            def __init__(self, in1, out1, in2, out2):
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.foo = torch.ones(in1, in1)

            def forward(self, x):
                x = self.submod1(x)
                x = self.submod2(x)
                return x + self.foo

        # explicitly call script() to freeze the type
        a = torch.jit.script(M(1, 1, 2, 2))
        b = torch.jit.script(M(1, 1, 2, 2))
        a.new_attr = "foo bar baz"
        self.assertSameType(a, b)

        # but if we assign attributes *before* calling script(), the types
        # should be different, since `new_attr` should be turned into a Script
        # attribute
        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        a.new_attr = "foo bar baz"
        self.assertDifferentType(a, b)

    def test_failed_attribute_compilation(self):
        """
        Attributes whose type cannot be inferred should fail cleanly with nice hints
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # assign a type we know can't be converted to TorchScript
                self.foo = object

            def forward(self):
                # try to use it in forward
                return self.foo

        m = M()
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "failed to convert Python type", "self.foo"
        ):
            torch.jit.script(m)

    def test_script_function_attribute_different(self):
        """
        Different functions passed in should lead to different types
        """

        @torch.jit.script
        def fn1(x):
            return x + x

        @torch.jit.script
        def fn2(x):
            return x - x

        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x):
                return self.fn(x)

        fn1_mod = M(fn1)
        fn2_mod = M(fn2)

        self.assertDifferentType(fn1_mod, fn2_mod)

    def test_builtin_function_same(self):
        class Caller(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, input):
                return self.fn(input, input)

        c1 = Caller(torch.add)
        c2 = Caller(torch.add)

        self.assertSameType(c1, c2)

    def test_builtin_function_different(self):
        class Caller(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, input):
                return self.fn(input, input)

        c1 = Caller(torch.add)
        c2 = Caller(torch.sub)

        self.assertDifferentType(c1, c2)

    def test_script_function_attribute_same(self):
        """
        Same functions passed in should lead to same types
        """

        @torch.jit.script
        def fn(x):
            return x + x

        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x):
                return self.fn(x)

        fn1_mod = M(fn)
        fn2_mod = M(fn)

        self.assertSameType(fn1_mod, fn2_mod)

    def test_python_function_attribute_different(self):
        """
        Different functions passed in should lead to different types
        """

        def fn1(x):
            return x + x

        def fn2(x):
            return x - x

        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x):
                return self.fn(x)

        fn1_mod = M(fn1)
        fn2_mod = M(fn2)

        self.assertDifferentType(fn1_mod, fn2_mod)

    def test_python_function_attribute_same(self):
        """
        Same functions passed in should lead to same types
        """

        def fn(x):
            return x + x

        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x):
                return self.fn(x)

        fn1_mod = M(fn)
        fn2_mod = M(fn)

        self.assertSameType(fn1_mod, fn2_mod)

    @suppress_warnings
    def test_tracing_gives_different_types(self):
        """
        Since we can't guarantee that methods are the same between different
        trace runs, tracing must always generate a unique type.
        """

        class M(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > y.sum():
                    return x
                else:
                    return y

        a = torch.jit.trace(M(), (torch.zeros(1, 1), torch.ones(1, 1)))
        b = torch.jit.trace(M(), (torch.ones(1, 1), torch.zeros(1, 1)))
        self.assertDifferentType(a, b)

    def test_ignored_fns(self):
        class M(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            @torch.jit.ignore
            def ignored(self):
                return self.foo

            def forward(self):
                return self.ignored()

        a = torch.jit.script(M(torch.ones(1)))
        b = torch.jit.script(M(torch.ones(2)))
        self.assertSameType(a, b)
        self.assertNotEqual(a(), b())

    @suppress_warnings
    def test_script_module_containing_traced_module(self):
        class Traced(torch.nn.Module):
            def forward(self, x):
                if x.sum() > 0:
                    return x
                else:
                    return x + x

        class M(torch.nn.Module):
            def __init__(self, input):
                super().__init__()
                self.traced = torch.jit.trace(Traced(), input)

            def forward(self, x):
                return self.traced(x)

        a = M((torch.ones(1),))
        b = M((torch.zeros(1),))
        self.assertDifferentType(a, b)

    def test_loaded_modules_work(self):
        class AB(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 1
                self.b = 1

            def forward(self):
                return self.a + self.b

        class A(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 1

            def forward(self):
                return self.a

        class Wrapper(torch.nn.Module):
            def __init__(self, sub):
                super().__init__()
                self.sub = sub

            def forward(self):
                return self.sub()

        def package(x):
            buffer = io.BytesIO()
            torch.jit.save(torch.jit.script(x), buffer)
            buffer.seek(0)
            return torch.jit.script(Wrapper(torch.jit.load(buffer)))

        a = package(AB())
        a()
        b = package(A())
        b()

    def test_module_dict_same_type_different_name(self):
        """
        We should be able to differentiate between two ModuleDict instances
        that have different keys but the same value types.
        """

        class A(torch.nn.Module):
            def forward(self, x):
                return x

        class Foo(torch.nn.Module):
            def __init__(self, s):
                super().__init__()
                self.dict = torch.nn.ModuleDict(s)

            def forward(self, x):
                return x

        a = Foo({"foo": A()})
        b = Foo({"bar": A()})
        c = Foo({"bar": A()})
        self.assertDifferentType(a, b)
        self.assertSameType(b, c)

    def test_type_sharing_define_in_init(self):
        """
        Tests that types between instances of a ScriptModule
        subclass that defines methods in its __init__ are not
        shared.
        """

        class A(torch.jit.ScriptModule):
            def __init__(self, val):
                super().__init__()
                self.define(
                    f"""
                def forward(self) -> int:
                    return {val}
                """
                )

        one = A(1)
        two = A(2)

        self.assertEqual(one(), 1)
        self.assertEqual(two(), 2)

    def test_type_sharing_disabled(self):
        """
        Test that type sharing can be disabled.
        """

        class A(torch.nn.Module):
            def __init__(self, sub):
                super().__init__()
                self.sub = sub

            def forward(self, x):
                return x

        class B(torch.nn.Module):
            def forward(self, x):
                return x

        top1 = A(A(B()))
        top2 = A(A(B()))

        top1_s = torch.jit._recursive.create_script_module(
            top1,
            torch.jit._recursive.infer_methods_to_compile,
            share_types=False,
        )
        top2_s = torch.jit._recursive.create_script_module(
            top2,
            torch.jit._recursive.infer_methods_to_compile,
            share_types=False,
        )

        self.assertDifferentType(top1_s, top2_s)
        self.assertDifferentType(top1_s, top1_s.sub)
        self.assertDifferentType(top1_s, top2_s.sub)
        self.assertDifferentType(top2_s, top2_s.sub)
        self.assertDifferentType(top2_s, top1_s.sub)

    def test_type_shared_ignored_attributes(self):
        """
        Test that types are shared if the exclusion of their
        ignored attributes makes them equal.
        """

        class A(torch.nn.Module):
            __jit_ignored_attributes__ = ["a"]

            def __init__(self, a, b):
                super().__init__()
                self.a = a
                self.b = b

            def forward(self, x):
                return x

        a_with_linear = A(torch.nn.Linear(5, 5), 5)
        a_with_string = A("string", 10)

        # Both should have the same type because the attribute
        # that differs in type is ignored and the common attribute
        # has the same type.
        self.assertSameType(a_with_linear, a_with_string)

    def test_type_not_shared_ignored_attributes(self):
        """
        Test that types are not shared if the exclusion of their
        ignored attributes makes them not equal.
        """

        class A(torch.nn.Module):
            __jit_ignored_attributes__ = ["a"]

            def __init__(self, a, b, c):
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                return x

        mod = A(torch.nn.Linear(5, 5), 5, "string")
        s1 = torch.jit.script(mod)
        A.__jit_ignored_attributes__ = ["a", "b"]
        s2 = torch.jit.script(mod)

        # The types of s1 and s2 should differ. Although they are instances
        # of A, __jit_ignored_attributes__ was modified before scripting s2,
        # so the set of ignored attributes is different between s1 and s2.
        self.assertDifferentType(s1, s2)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
