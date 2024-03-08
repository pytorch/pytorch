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

            @classmethod
            def from_real(cls, foo):
                (x, y), _ = foo.__getstate__()
                return cls(x, y)

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

    def test_tensor_queue(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, tq, x):
                torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
                x_sin = torch.ops._TorchScriptTesting.queue_pop(tq)
                x_cos = torch.ops._TorchScriptTesting.queue_pop(tq)
                return x_sin, x_cos, tq

        @torch._library.impl_abstract_class("_TorchScriptTesting::_TensorQueue")
        class TensorQueue:
            def __init__(self, q):
                self.queue = q

            @classmethod
            def from_real(cls, real_tq):
                return cls(real_tq.clone_queue())

            def push(self, x):
                self.queue.append(x)

            def pop(self):
                self.queue.pop(0)

            def size(self):
                return len(self.queue)

        mod = Model()

        tq = torch.classes._TorchScriptTesting._TensorQueue(torch.empty(0,).fill_(-1))
        x = torch.ones(2, 3)
        # mod(tq, x)
        ep = self._test_export_same_as_eager(Model(), (tq, x), strict=False)
        pass


@skipIfTorchDynamo("torchbind not supported with dynamo yet")
class TestImplAbstractClass(TestCase):
    def tearDown(self):
        torch._library.abstract_impl_class.global_abstract_class_registry.clear()

    def test_impl_abstract_class_no_torch_bind_class(self):
        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate class"):

            @torch._library.impl_abstract_class("_TorchScriptTesting::NOT_A_VALID_NAME")
            class Invalid:
                pass

    def test_impl_abstract_class_no_from_real(self):
        with self.assertRaisesRegex(
            RuntimeError, "must define a classmethod from_real"
        ):

            @torch._library.impl_abstract_class("_TorchScriptTesting::_Foo")
            class InvalidFakeFoo:
                def __init__(self):
                    pass

    def test_impl_abstract_class_from_real_not_classmethod(self):
        with self.assertRaisesRegex(
            RuntimeError, "must define a classmethod from_real"
        ):

            @torch._library.impl_abstract_class("_TorchScriptTesting::_Foo")
            class FakeFoo:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                def from_real(self, foo_obj):
                    x, y = foo_obj.__getstate__()
                    return FakeFoo(x, y)

    def test_impl_abstract_class_valid(self):
        class FakeFoo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            @classmethod
            def from_real(cls, foo_obj):
                x, y = foo_obj.__getstate__()
                return cls(x, y)

        torch._library.impl_abstract_class("_TorchScriptTesting::_Foo", FakeFoo)

    def test_impl_abstract_class_duplicate_registration(self):
        @torch._library.impl_abstract_class("_TorchScriptTesting::_Foo")
        class FakeFoo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            @classmethod
            def from_real(cls, foo_obj):
                x, y = foo_obj.__getstate__()
                return cls(x, y)

        with self.assertRaisesRegex(RuntimeError, "already registered"):
            torch._library.impl_abstract_class("_TorchScriptTesting::_Foo", FakeFoo)


if __name__ == "__main__":
    run_tests()
