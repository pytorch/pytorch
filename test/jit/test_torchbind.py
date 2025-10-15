# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import copy
import io
import os
import sys
from typing import Optional

import torch
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    skipIfTorchDynamo,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.torchbind_impls import load_torchbind_test_lib


@skipIfTorchDynamo("skipping as a precaution")
class TestTorchbind(JitTestCase):
    def setUp(self):
        load_torchbind_test_lib()

    def test_torchbind(self):
        def test_equality(f, cmp_key):
            obj1 = f()
            obj2 = torch.jit.script(f)()
            return (cmp_key(obj1), cmp_key(obj2))

        def f():
            val = torch.classes._TorchScriptTesting._Foo(5, 3)
            val.increment(1)
            return val

        test_equality(f, lambda x: x)

        with self.assertRaisesRegex(RuntimeError, "Expected a value of type 'int'"):
            val = torch.classes._TorchScriptTesting._Foo(5, 3)
            val.increment("foo")

        def f():
            ss = torch.classes._TorchScriptTesting._StackString(["asdf", "bruh"])
            return ss.pop()

        test_equality(f, lambda x: x)

        def f():
            ss1 = torch.classes._TorchScriptTesting._StackString(["asdf", "bruh"])
            ss2 = torch.classes._TorchScriptTesting._StackString(["111", "222"])
            ss1.push(ss2.pop())
            return ss1.pop() + ss2.pop()

        test_equality(f, lambda x: x)

        # test nn module with prepare_scriptable function
        class NonJitableClass:
            def __init__(self, int1, int2):
                self.int1 = int1
                self.int2 = int2

            def return_vals(self):
                return self.int1, self.int2

        class CustomWrapper(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def forward(self) -> None:
                self.foo.increment(1)
                return

            def __prepare_scriptable__(self):
                int1, int2 = self.foo.return_vals()
                foo = torch.classes._TorchScriptTesting._Foo(int1, int2)
                return CustomWrapper(foo)

        foo = CustomWrapper(NonJitableClass(1, 2))
        jit_foo = torch.jit.script(foo)

    def test_torchbind_take_as_arg(self):
        global StackString  # see [local resolution in python]
        StackString = torch.classes._TorchScriptTesting._StackString

        def foo(stackstring):
            # type: (StackString)
            stackstring.push("lel")
            return stackstring

        script_input = torch.classes._TorchScriptTesting._StackString([])
        scripted = torch.jit.script(foo)
        script_output = scripted(script_input)
        self.assertEqual(script_output.pop(), "lel")

    def test_torchbind_return_instance(self):
        def foo():
            ss = torch.classes._TorchScriptTesting._StackString(["hi", "mom"])
            return ss

        scripted = torch.jit.script(foo)
        # Ensure we are creating the object and calling __init__
        # rather than calling the __init__wrapper nonsense
        fc = (
            FileCheck()
            .check("prim::CreateObject()")
            .check('prim::CallMethod[name="__init__"]')
        )
        fc.run(str(scripted.graph))
        out = scripted()
        self.assertEqual(out.pop(), "mom")
        self.assertEqual(out.pop(), "hi")

    def test_torchbind_return_instance_from_method(self):
        def foo():
            ss = torch.classes._TorchScriptTesting._StackString(["hi", "mom"])
            clone = ss.clone()
            ss.pop()
            return ss, clone

        scripted = torch.jit.script(foo)
        out = scripted()
        self.assertEqual(out[0].pop(), "hi")
        self.assertEqual(out[1].pop(), "mom")
        self.assertEqual(out[1].pop(), "hi")

    def test_torchbind_def_property_getter_setter(self):
        def foo_getter_setter_full():
            fooGetterSetter = torch.classes._TorchScriptTesting._FooGetterSetter(5, 6)
            # getX method intentionally adds 2 to x
            old = fooGetterSetter.x
            # setX method intentionally adds 2 to x
            fooGetterSetter.x = old + 4
            new = fooGetterSetter.x
            return old, new

        self.checkScript(foo_getter_setter_full, ())

        def foo_getter_setter_lambda():
            foo = torch.classes._TorchScriptTesting._FooGetterSetterLambda(5)
            old = foo.x
            foo.x = old + 4
            new = foo.x
            return old, new

        self.checkScript(foo_getter_setter_lambda, ())

    def test_torchbind_def_property_just_getter(self):
        def foo_just_getter():
            fooGetterSetter = torch.classes._TorchScriptTesting._FooGetterSetter(5, 6)
            # getY method intentionally adds 4 to x
            return fooGetterSetter, fooGetterSetter.y

        scripted = torch.jit.script(foo_just_getter)
        out, result = scripted()
        self.assertEqual(result, 10)

        with self.assertRaisesRegex(RuntimeError, "can't set attribute"):
            out.y = 5

        def foo_not_setter():
            fooGetterSetter = torch.classes._TorchScriptTesting._FooGetterSetter(5, 6)
            old = fooGetterSetter.y
            fooGetterSetter.y = old + 4
            # getY method intentionally adds 4 to x
            return fooGetterSetter.y

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Tried to set read-only attribute: y",
            "fooGetterSetter.y = old + 4",
        ):
            scripted = torch.jit.script(foo_not_setter)

    def test_torchbind_def_property_readwrite(self):
        def foo_readwrite():
            fooReadWrite = torch.classes._TorchScriptTesting._FooReadWrite(5, 6)
            old = fooReadWrite.x
            fooReadWrite.x = old + 4
            return fooReadWrite.x, fooReadWrite.y

        self.checkScript(foo_readwrite, ())

        def foo_readwrite_error():
            fooReadWrite = torch.classes._TorchScriptTesting._FooReadWrite(5, 6)
            fooReadWrite.y = 5
            return fooReadWrite

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set read-only attribute: y", "fooReadWrite.y = 5"
        ):
            scripted = torch.jit.script(foo_readwrite_error)

    def test_torchbind_take_instance_as_method_arg(self):
        def foo():
            ss = torch.classes._TorchScriptTesting._StackString(["mom"])
            ss2 = torch.classes._TorchScriptTesting._StackString(["hi"])
            ss.merge(ss2)
            return ss

        scripted = torch.jit.script(foo)
        out = scripted()
        self.assertEqual(out.pop(), "hi")
        self.assertEqual(out.pop(), "mom")

    def test_torchbind_return_tuple(self):
        def f():
            val = torch.classes._TorchScriptTesting._StackString(["3", "5"])
            return val.return_a_tuple()

        scripted = torch.jit.script(f)
        tup = scripted()
        self.assertEqual(tup, (1337.0, 123))

    def test_torchbind_save_load(self):
        def foo():
            ss = torch.classes._TorchScriptTesting._StackString(["mom"])
            ss2 = torch.classes._TorchScriptTesting._StackString(["hi"])
            ss.merge(ss2)
            return ss

        scripted = torch.jit.script(foo)
        self.getExportImportCopy(scripted)

    def test_torchbind_lambda_method(self):
        def foo():
            ss = torch.classes._TorchScriptTesting._StackString(["mom"])
            return ss.top()

        scripted = torch.jit.script(foo)
        self.assertEqual(scripted(), "mom")

    def test_torchbind_class_attr_recursive(self):
        class FooBar(torch.nn.Module):
            def __init__(self, foo_model):
                super().__init__()
                self.foo_mod = foo_model

            def forward(self) -> int:
                return self.foo_mod.info()

            def to_ivalue(self):
                torchbind_model = torch.classes._TorchScriptTesting._Foo(
                    self.foo_mod.info(), 1
                )
                return FooBar(torchbind_model)

        inst = FooBar(torch.classes._TorchScriptTesting._Foo(2, 3))
        scripted = torch.jit.script(inst.to_ivalue())
        self.assertEqual(scripted(), 6)

    def test_torchbind_class_attribute(self):
        class FooBar1234(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._StackString(["3", "4"])

            def forward(self):
                return self.f.top()

        inst = FooBar1234()
        scripted = torch.jit.script(inst)
        eic = self.getExportImportCopy(scripted)
        assert eic() == "deserialized"
        for expected in ["deserialized", "was", "i"]:
            assert eic.f.pop() == expected

    def test_torchbind_getstate(self):
        class FooBar4321(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            def forward(self):
                return self.f.top()

        inst = FooBar4321()
        scripted = torch.jit.script(inst)
        eic = self.getExportImportCopy(scripted)
        # NB: we expect the values {7, 3, 3, 1} as __getstate__ is defined to
        # return {1, 3, 3, 7}. I tried to make this actually depend on the
        # values at instantiation in the test with some transformation, but
        # because it seems we serialize/deserialize multiple times, that
        # transformation isn't as you would it expect it to be.
        assert eic() == 7
        for expected in [7, 3, 3, 1]:
            assert eic.f.pop() == expected

    def test_torchbind_deepcopy(self):
        class FooBar4321(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            def forward(self):
                return self.f.top()

        inst = FooBar4321()
        scripted = torch.jit.script(inst)
        copied = copy.deepcopy(scripted)
        assert copied.forward() == 7
        for expected in [7, 3, 3, 1]:
            assert copied.f.pop() == expected

    def test_torchbind_python_deepcopy(self):
        class FooBar4321(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            def forward(self):
                return self.f.top()

        inst = FooBar4321()
        copied = copy.deepcopy(inst)
        assert copied() == 7
        for expected in [7, 3, 3, 1]:
            assert copied.f.pop() == expected

    def test_torchbind_tracing(self):
        class TryTracing(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            def forward(self):
                return torch.ops._TorchScriptTesting.take_an_instance(self.f)

        traced = torch.jit.trace(TryTracing(), ())
        self.assertEqual(torch.zeros(4, 4), traced())

    def test_torchbind_pass_wrong_type(self):
        with self.assertRaisesRegex(RuntimeError, "but instead found type 'Tensor'"):
            torch.ops._TorchScriptTesting.take_an_instance(torch.rand(3, 4))

    def test_torchbind_tracing_nested(self):
        class TryTracingNest(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        class TryTracing123(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nest = TryTracingNest()

            def forward(self):
                return torch.ops._TorchScriptTesting.take_an_instance(self.nest.f)

        traced = torch.jit.trace(TryTracing123(), ())
        self.assertEqual(torch.zeros(4, 4), traced())

    def test_torchbind_pickle_serialization(self):
        nt = torch.classes._TorchScriptTesting._PickleTester([3, 4])
        b = io.BytesIO()
        torch.save(nt, b)
        b.seek(0)
        # weights_only=False as trying to load ScriptObject
        nt_loaded = torch.load(b, weights_only=False)
        for exp in [7, 3, 3, 1]:
            self.assertEqual(nt_loaded.pop(), exp)

    def test_torchbind_instantiate_missing_class(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Tried to instantiate class 'foo.IDontExist', but it does not exist!",
        ):
            torch.classes.foo.IDontExist(3, 4, 5)

    def test_torchbind_optional_explicit_attr(self):
        class TorchBindOptionalExplicitAttr(torch.nn.Module):
            foo: Optional[torch.classes._TorchScriptTesting._StackString]

            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.classes._TorchScriptTesting._StackString(["test"])

            def forward(self) -> str:
                foo_obj = self.foo
                if foo_obj is not None:
                    return foo_obj.pop()
                else:
                    return "<None>"

        mod = TorchBindOptionalExplicitAttr()
        scripted = torch.jit.script(mod)

    def test_torchbind_no_init(self):
        with self.assertRaisesRegex(RuntimeError, "torch::init"):
            x = torch.classes._TorchScriptTesting._NoInit()

    def test_profiler_custom_op(self):
        inst = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        with torch.autograd.profiler.profile() as prof:
            torch.ops._TorchScriptTesting.take_an_instance(inst)

        found_event = False
        for e in prof.function_events:
            if e.name == "_TorchScriptTesting::take_an_instance":
                found_event = True
        self.assertTrue(found_event)

    def test_torchbind_getattr(self):
        foo = torch.classes._TorchScriptTesting._StackString(["test"])
        self.assertEqual(None, getattr(foo, "bar", None))

    def test_torchbind_attr_exception(self):
        foo = torch.classes._TorchScriptTesting._StackString(["test"])
        with self.assertRaisesRegex(AttributeError, "does not have a field"):
            foo.bar

    def test_lambda_as_constructor(self):
        obj_no_swap = torch.classes._TorchScriptTesting._LambdaInit(4, 3, False)
        self.assertEqual(obj_no_swap.diff(), 1)

        obj_swap = torch.classes._TorchScriptTesting._LambdaInit(4, 3, True)
        self.assertEqual(obj_swap.diff(), -1)

    def test_staticmethod(self):
        def fn(inp: int) -> int:
            return torch.classes._TorchScriptTesting._StaticMethod.staticMethod(inp)

        self.checkScript(fn, (1,))

    def test_hasattr(self):
        ss = torch.classes._TorchScriptTesting._StackString(["foo", "bar"])
        self.assertFalse(hasattr(ss, "baz"))

    def test_default_args(self):
        def fn() -> int:
            obj = torch.classes._TorchScriptTesting._DefaultArgs()
            obj.increment(5)
            obj.decrement()
            obj.decrement(2)
            obj.divide()
            obj.scale_add(5)
            obj.scale_add(3, 2)
            obj.divide(3)
            return obj.increment()

        self.checkScript(fn, ())

        def gn() -> int:
            obj = torch.classes._TorchScriptTesting._DefaultArgs(5)
            obj.increment(3)
            obj.increment()
            obj.decrement(2)
            obj.divide()
            obj.scale_add(3)
            obj.scale_add(3, 2)
            obj.divide(2)
            return obj.decrement()

        self.checkScript(gn, ())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
