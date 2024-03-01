# Owner(s): ["oncall: export"]

import torch
import torch.testing._internal.torchbind_impls  # noqa: F401
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch.export import export
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


@skipIfTorchDynamo("torchbind not supported with dynamo yet")
class TestExportTorchbind(TestCase):
    def setUp(self):
        @torch._library.impl_abstract_class("_TorchScriptTesting::_Foo")
        class FakeFoo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            @staticmethod
            def from_metadata(foo_meta):
                x, y = foo_meta
                return FakeFoo(x, y)

            def add_tensor(self, z):
                return (self.x + self.y) * z

    def tearDown(self):
        torch._library.abstract_impl_class.deregister_abstract_impl(
            "_TorchScriptTesting::_Foo"
        )

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
                return x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x)

        m = MyModule()
        input = torch.ones(2, 3)
        with enable_torchbind_tracing():
            ep = torch.export.export(m, (input,), strict=False)

        unlifted = ep.module()
        self.assertEqual(m(input), unlifted(input))


if __name__ == "__main__":
    run_tests()
