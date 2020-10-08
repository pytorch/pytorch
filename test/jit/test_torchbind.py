import io
import os
import sys
import copy
import unittest

import torch
from typing import Optional
from pathlib import Path

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_utils import TEST_WITH_ROCM, IS_WINDOWS, IS_SANDCASTLE, IS_MACOS, IS_FBCODE
from torch.testing import FileCheck

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

def test_equality(f, cmp_key):
    obj1 = f()
    obj2 = torch.jit.script(f)()
    return (cmp_key(obj1), cmp_key(obj2))

class TestTorchbind(JitTestCase):
    def setUp(self):
        if IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE:
            raise unittest.SkipTest("non-portable load_library call used in test")
        if TEST_WITH_ROCM:
            torch_root = Path(torch.__file__).resolve().parent
            p = torch_root / 'lib' / 'libtorchbind_test.so'
        else:
            torch_root = Path(__file__).resolve().parent.parent.parent
            p = torch_root / 'build' / 'lib' / 'libtorchbind_test.so'
        torch.ops.load_library(str(p))

    def test_torchbind(self):
        def f():
            val = torch.classes._TorchScriptTesting._Foo(5, 3)
            val.increment(1)
            return val
        test_equality(f, lambda x: x)

        with self.assertRaisesRegex(RuntimeError, "Expected a value of type 'int'"):
            val = torch.classes._TorchScriptTesting._Foo(5, 3)
            val.increment('foo')

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

    def test_torchbind_static(self):
        def f():
            val = torch.classes._TorchScriptTesting._Foo.func_mul(3, 5)
            return val

        # test_equality(f, lambda x: x)
        # obj1 = torch.jit.script(f)()
        # print(obj1)


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
        fc = FileCheck().check('prim::CreateObject()')\
                        .check('prim::CallMethod[name="__init__"]')
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

    def test_torchbind_class_attribute(self):
        class FooBar1234(torch.nn.Module):
            def __init__(self):
                super(FooBar1234, self).__init__()
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
            def __init__(self):
                super(FooBar4321, self).__init__()
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
            def __init__(self):
                super(FooBar4321, self).__init__()
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
            def __init__(self):
                super(FooBar4321, self).__init__()
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
            def __init__(self):
                super(TryTracing, self).__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            def forward(self):
                return torch.ops._TorchScriptTesting.take_an_instance(self.f)

        traced = torch.jit.trace(TryTracing(), ())
        self.assertEqual(torch.zeros(4, 4), traced())

    def test_torchbind_tracing_nested(self):
        class TryTracingNest(torch.nn.Module):
            def __init__(self):
                super(TryTracingNest, self).__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        class TryTracing123(torch.nn.Module):
            def __init__(self):
                super(TryTracing123, self).__init__()
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
        nt_loaded = torch.load(b)
        for exp in [7, 3, 3, 1]:
            self.assertEqual(nt_loaded.pop(), exp)

    def test_torchbind_instantiate_missing_class(self):
        with self.assertRaisesRegex(RuntimeError, 'Tried to instantiate class \'foo.IDontExist\', but it does not exist!'):
            torch.classes.foo.IDontExist(3, 4, 5)

    def test_torchbind_optional_explicit_attr(self):
        class TorchBindOptionalExplicitAttr(torch.nn.Module):
            foo : Optional[torch.classes._TorchScriptTesting._StackString]

            def __init__(self):
                super().__init__()
                self.foo = torch.classes._TorchScriptTesting._StackString(["test"])

            def forward(self) -> str:
                foo_obj = self.foo
                if foo_obj is not None:
                    return foo_obj.pop()
                else:
                    return '<None>'

        mod = TorchBindOptionalExplicitAttr()
        scripted = torch.jit.script(mod)

    def test_torchbind_no_init(self):
        with self.assertRaisesRegex(RuntimeError, 'torch::init'):
            x = torch.classes._TorchScriptTesting._NoInit()

    def test_profiler_custom_op(self):
        inst = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        with torch.autograd.profiler.profile() as prof:
            torch.ops._TorchScriptTesting.take_an_instance(inst)

        found_event = False
        for e in prof.function_events:
            if e.name == '_TorchScriptTesting::take_an_instance':
                found_event = True
        self.assertTrue(found_event)

    def test_torchbind_getattr(self):
        foo = torch.classes._TorchScriptTesting._StackString(["test"])
        self.assertEqual(None, getattr(foo, 'bar', None))

    def test_torchbind_attr_exception(self):
        foo = torch.classes._TorchScriptTesting._StackString(["test"])
        with self.assertRaisesRegex(AttributeError, 'does not have a field'):
            foo.bar
