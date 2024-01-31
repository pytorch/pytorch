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
        return exported_program

    def test_none(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x, n):
                return x + self.attr.add_tensor(x)

        ep_nonstrict = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), None), strict=False
        )
        self.assertExpectedInline(
            ep_nonstrict.module().code.strip(),
            """\
def forward(self, arg_0, arg_1):
    arg0_1, arg1_1, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    _lifted_custom_obj0_1 = self._lifted_custom_obj0
    call_torchbind = torch.ops.higher_order.call_torchbind(_lifted_custom_obj0_1, 'add_tensor', arg0_1);  _lifted_custom_obj0_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, call_torchbind);  arg0_1 = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",  # noqa:B950
        )
        ep_strict = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), None), strict=True
        )
        self.assertExpectedInline(
            ep_strict.module().code.strip(),
            """\
def forward(self, arg_0, arg_1):
    l_x_, arg1, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    _lifted_custom_obj0_1 = self._lifted_custom_obj0
    call_torchbind = torch.ops.higher_order.call_torchbind(_lifted_custom_obj0_1, 'add_tensor', l_x_);  _lifted_custom_obj0_1 = None
    add = torch.ops.aten.add.Tensor(l_x_, call_torchbind);  l_x_ = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",  # noqa:B950
        )

    def test_attribute(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return x + self.attr.add_tensor(x)

        ep_non_strict = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3),), strict=False
        )
        self.assertExpectedInline(
            ep_non_strict.module().code.strip(),
            """\
def forward(self, arg_0):
    arg0_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    _lifted_custom_obj0_1 = self._lifted_custom_obj0
    call_torchbind = torch.ops.higher_order.call_torchbind(_lifted_custom_obj0_1, 'add_tensor', arg0_1);  _lifted_custom_obj0_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, call_torchbind);  arg0_1 = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",  # noqa:B950
        )
        ep_strict = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3),), strict=True
        )
        self.assertExpectedInline(
            ep_strict.module().code.strip(),
            """\
def forward(self, arg_0):
    l_x_, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    _lifted_custom_obj0_1 = self._lifted_custom_obj0
    call_torchbind = torch.ops.higher_order.call_torchbind(_lifted_custom_obj0_1, 'add_tensor', l_x_);  _lifted_custom_obj0_1 = None
    add = torch.ops.aten.add.Tensor(l_x_, call_torchbind);  l_x_ = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",  # noqa:B950
        )

    def test_attribute_as_custom_op_argument(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x)

        ep_non_strict = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3),), strict=False
        )
        self.assertExpectedInline(
            ep_non_strict.module().code.strip(),
            """\
def forward(self, arg_0):
    arg0_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    _lifted_custom_obj0_1 = self._lifted_custom_obj0
    takes_foo = torch.ops._TorchScriptTesting.takes_foo.default(_lifted_custom_obj0_1, arg0_1);  _lifted_custom_obj0_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, takes_foo);  arg0_1 = takes_foo = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )

        ep_strict = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3),), strict=True
        )
        self.assertExpectedInline(
            ep_strict.module().code.strip(),
            """\
def forward(self, arg_0):
    l_x_, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    _lifted_custom_obj0_1 = self._lifted_custom_obj0
    takes_foo = torch.ops._TorchScriptTesting.takes_foo.default(_lifted_custom_obj0_1, l_x_);  _lifted_custom_obj0_1 = None
    add = torch.ops.aten.add.Tensor(l_x_, takes_foo);  l_x_ = takes_foo = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )

    def test_input(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, cc):
                return x + cc.add_tensor(x)

        cc = torch.classes._TorchScriptTesting._Foo(10, 20)
        ep_nonstrict = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), cc), strict=False
        )
        self.assertExpectedInline(
            ep_nonstrict.module().code.strip(),
            """\
def forward(self, arg_0, arg_1):
    arg0_1, arg1_1, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    call_torchbind = torch.ops.higher_order.call_torchbind(arg1_1, 'add_tensor', arg0_1);  arg1_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, call_torchbind);  arg0_1 = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        ep_strict = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), cc), strict=True
        )
        self.assertExpectedInline(
            ep_strict.module().code.strip(),
            """\
def forward(self, arg_0, arg_1):
    l_x_, l_cc_, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    call_torchbind = torch.ops.higher_order.call_torchbind(l_cc_, 'add_tensor', l_x_);  l_cc_ = None
    add = torch.ops.aten.add.Tensor(l_x_, call_torchbind);  l_x_ = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )

    def test_input_as_custom_op_argument(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, cc):
                return x + torch.ops._TorchScriptTesting.takes_foo(cc, x)

        cc = torch.classes._TorchScriptTesting._Foo(10, 20)
        ep_nonstrict = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), cc), strict=False
        )
        self.assertExpectedInline(
            ep_nonstrict.module().code.strip(),
            """\
def forward(self, arg_0, arg_1):
    arg0_1, arg1_1, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    takes_foo = torch.ops._TorchScriptTesting.takes_foo.default(arg1_1, arg0_1);  arg1_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, takes_foo);  arg0_1 = takes_foo = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        ep_strict = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), cc), strict=True
        )
        self.assertExpectedInline(
            ep_strict.module().code.strip(),
            """\
def forward(self, arg_0, arg_1):
    l_x_, l_cc_, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    takes_foo = torch.ops._TorchScriptTesting.takes_foo.default(l_cc_, l_x_);  l_cc_ = None
    add = torch.ops.aten.add.Tensor(l_x_, takes_foo);  l_x_ = takes_foo = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )

    def test_unlift_custom_obj(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x)

        input = torch.ones(2, 3)
        ep_nonstrict = self._test_export_same_as_eager(
            MyModule(), (input,), strict=False
        )
        self.assertExpectedInline(
            ep_nonstrict.module().code.strip(),
            """\
def forward(self, arg_0):
    arg0_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    _lifted_custom_obj0_1 = self._lifted_custom_obj0
    takes_foo = torch.ops._TorchScriptTesting.takes_foo.default(_lifted_custom_obj0_1, arg0_1);  _lifted_custom_obj0_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, takes_foo);  arg0_1 = takes_foo = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )

        ep_strict = self._test_export_same_as_eager(MyModule(), (input,), strict=True)
        self.assertExpectedInline(
            ep_strict.module().code.strip(),
            """\
def forward(self, arg_0):
    l_x_, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    _lifted_custom_obj0_1 = self._lifted_custom_obj0
    takes_foo = torch.ops._TorchScriptTesting.takes_foo.default(_lifted_custom_obj0_1, l_x_);  _lifted_custom_obj0_1 = None
    add = torch.ops.aten.add.Tensor(l_x_, takes_foo);  l_x_ = takes_foo = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )


if __name__ == "__main__":
    run_tests()
