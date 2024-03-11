# Owner(s): ["oncall: export"]
import unittest

import torch
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch.export import export
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


@skipIfTorchDynamo("torchbind not supported with dynamo yet")
class TestExportTorchbind(TestCase):
    def setUp(self):
        if IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        elif IS_SANDCASTLE or IS_FBCODE:
            torch.ops.load_library(
                "//caffe2/test/cpp/jit:test_custom_class_registrations"
            )
        elif IS_WINDOWS:
            lib_file_path = find_library_location("torchbind_test.dll")
            torch.ops.load_library(str(lib_file_path))
        else:
            lib_file_path = find_library_location("libtorchbind_test.so")
            torch.ops.load_library(str(lib_file_path))

    def _test_export_same_as_eager(self, f, args, kwargs=None, strict=True):
        kwargs = kwargs or {}
        with enable_torchbind_tracing():
            exported_program = export(f, args, kwargs, strict=strict)
        reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}
        self.assertEqual(exported_program.module()(*args, **kwargs), f(*args, **kwargs))
        self.assertEqual(
            exported_program.module()(*args, **reversed_kwargs),
            f(*args, **reversed_kwargs),
        )

    def test_none(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x, n):
                return x + self.attr.add_tensor(x)

        self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), None), strict=False
        )

    def test_attribute(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return x + self.attr.add_tensor(x)

        self._test_export_same_as_eager(MyModule(), (torch.ones(2, 3),), strict=False)

    def test_attribute_as_custom_op_argument(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x)

        self._test_export_same_as_eager(MyModule(), (torch.ones(2, 3),), strict=False)

    def test_input(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, cc):
                return x + cc.add_tensor(x)

        cc = torch.classes._TorchScriptTesting._Foo(10, 20)
        self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), cc), strict=False
        )

    def test_input_as_custom_op_argument(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, cc):
                return x + torch.ops._TorchScriptTesting.takes_foo(cc, x)

        cc = torch.classes._TorchScriptTesting._Foo(10, 20)
        self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), cc), strict=False
        )

    def test_unlift_custom_obj(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo(self.attr, x)
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, a)
                return x + b

        m = MyModule()
        input = torch.ones(2, 3)
        with enable_torchbind_tracing():
            ep = torch.export.export(m, (input,), strict=False)

        unlifted = ep.module()
        self.assertEqual(m(input), unlifted(input))

        with enable_torchbind_tracing():
            ep2 = torch.export.export(unlifted, (input,), strict=False)

        self.assertEqual(m(input), ep2.module()(input))

    def test_custom_obj_list_out(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo_list_return(self.attr, x)
                y = a[0] + a[1] + a[2]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        m = MyModule()
        input = torch.ones(2, 3)
        with enable_torchbind_tracing():
            ep = torch.export.export(m, (input,), strict=False)

        unlifted = ep.module()
        self.assertEqual(m(input), unlifted(input))

        with enable_torchbind_tracing():
            ep2 = torch.export.export(unlifted, (input,), strict=False)

        self.assertEqual(m(input), ep2.module()(input))

    def test_custom_obj_tuple_out(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        m = MyModule()
        input = torch.ones(2, 3)
        with enable_torchbind_tracing():
            ep = torch.export.export(m, (input,), strict=False)

        unlifted = ep.module()
        self.assertEqual(m(input), unlifted(input))

        with enable_torchbind_tracing():
            ep2 = torch.export.export(unlifted, (input,), strict=False)

        self.assertEqual(m(input), ep2.module()(input))


if __name__ == "__main__":
    run_tests()
