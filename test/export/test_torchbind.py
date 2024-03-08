# Owner(s): ["oncall: export"]

import torch
import torch.testing._internal.torchbind_impls  # noqa: F401
import torch.utils._pytree as pytree
from torch._functorch.aot_autograd import aot_export_module
from torch._higher_order_ops.effects import _EffectType, SIDE_EFFECTS
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch.export import export
from torch.fx.experimental.proxy_tensor import make_fx
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

        @torch._library.impl_abstract_class("_TorchScriptTesting::_TensorQueue")
        class FakeTensorQueue:
            def __init__(self, q):
                self.queue = q

            @classmethod
            def from_real(cls, real_tq):
                return cls(real_tq.clone_queue())

            def push(self, x):
                self.queue.append(x)

            def pop(self):
                return self.queue.pop(0)

            def size(self):
                return len(self.queue)

    def tearDown(self):
        torch._library.abstract_impl_class.deregister_abstract_impl(
            "_TorchScriptTesting::_Foo"
        )
        torch._library.abstract_impl_class.deregister_abstract_impl(
            "_TorchScriptTesting::_TensorQueue"
        )

    def _assert_equal_skip_script_object(self, exp, actual):
        flat_exp = pytree.tree_leaves(exp)
        flat_actual = pytree.tree_leaves(actual)
        self.assertEqual(len(flat_exp), len(flat_actual))
        for a, b in zip(flat_exp, flat_actual):
            if isinstance(a, torch.ScriptObject) and isinstance(b, torch.ScriptObject):
                continue
            self.assertEqual(a, b)

    def _test_export_same_as_eager_skip_script_object(
        self, f, args, kwargs=None, strict=False
    ):
        kwargs = kwargs or {}
        with enable_torchbind_tracing():
            exported_program = export(f, args, kwargs, strict=strict)
        gm = exported_program.module()
        reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}

        self._assert_equal_skip_script_object(gm(*args, **kwargs), f(*args, **kwargs))
        self._assert_equal_skip_script_object(
            gm(*args, **reversed_kwargs),
            f(*args, **reversed_kwargs),
        )
        return exported_program

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

    def test_tensor_queue_non_strict(self):
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

        mod = Model()

        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        x = torch.ones(2, 3)

        SIDE_EFFECTS[
            torch.ops._TorchScriptTesting.queue_push.default
        ] = _EffectType.ORDERED
        SIDE_EFFECTS[
            torch.ops._TorchScriptTesting.queue_pop.default
        ] = _EffectType.ORDERED
        # This is caused by newly created token not handled in export yet.
        with self.assertRaisesRegex(IndexError, "list index out of range"):
            ep = self._test_export_same_as_eager_skip_script_object(
                mod, (tq, x), strict=False
            )
        del SIDE_EFFECTS[torch.ops._TorchScriptTesting.queue_push.default]
        del SIDE_EFFECTS[torch.ops._TorchScriptTesting.queue_pop.default]

    def test_tensor_queue_make_fx(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)

            def forward(self, tq, x):
                torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
                x_sin = torch.ops._TorchScriptTesting.queue_pop(tq)
                x_cos = torch.ops._TorchScriptTesting.queue_pop(tq)
                return self.linear(x_sin), self.linear(x_cos), tq

        mod = Model()
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        x = torch.ones(2, 3)
        gm = make_fx(mod)(tq, x)
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1):
    cos = torch.ops.aten.cos.default(arg1_1)
    queue_push = torch.ops._TorchScriptTesting.queue_push.default(arg0_1, cos);  cos = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    queue_push_1 = torch.ops._TorchScriptTesting.queue_push.default(arg0_1, sin);  sin = None
    queue_pop = torch.ops._TorchScriptTesting.queue_pop.default(arg0_1)
    queue_pop_1 = torch.ops._TorchScriptTesting.queue_pop.default(arg0_1)
    _param_constant0 = self._param_constant0
    t = torch.ops.aten.t.default(_param_constant0);  _param_constant0 = None
    _param_constant1 = self._param_constant1
    addmm = torch.ops.aten.addmm.default(_param_constant1, queue_pop, t);  _param_constant1 = queue_pop = t = None
    _param_constant0_1 = self._param_constant0
    t_1 = torch.ops.aten.t.default(_param_constant0_1);  _param_constant0_1 = None
    _param_constant1_1 = self._param_constant1
    addmm_1 = torch.ops.aten.addmm.default(_param_constant1_1, queue_pop_1, t_1);  _param_constant1_1 = queue_pop_1 = t_1 = None
    return (addmm, addmm_1, arg0_1)
    """,
        )
        self._assert_equal_skip_script_object(mod(tq, x), gm(tq, x))

    def test_tensor_queue_aot_export(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)

            def forward(self, tq, x):
                torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
                x_sin = torch.ops._TorchScriptTesting.queue_pop(tq)
                x_cos = torch.ops._TorchScriptTesting.queue_pop(tq)
                return self.linear(x_sin), self.linear(x_cos), tq

        mod = Model()
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        x = torch.ones(2, 3)
        with torch._higher_order_ops.torchbind.enable_torchbind_tracing():
            SIDE_EFFECTS[
                torch.ops._TorchScriptTesting.queue_push.default
            ] = _EffectType.ORDERED
            SIDE_EFFECTS[
                torch.ops._TorchScriptTesting.queue_pop.default
            ] = _EffectType.ORDERED
            gm, _ = aot_export_module(mod, (tq, x), trace_joint=False)
            del SIDE_EFFECTS[torch.ops._TorchScriptTesting.queue_push.default]
            del SIDE_EFFECTS[torch.ops._TorchScriptTesting.queue_pop.default]
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    cos = torch.ops.aten.cos.default(arg4_1)
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops._TorchScriptTesting.queue_push.default, arg3_1, cos);  arg0_1 = cos = None
    getitem = with_effects[0];  with_effects = None
    sin = torch.ops.aten.sin.default(arg4_1);  arg4_1 = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops._TorchScriptTesting.queue_push.default, arg3_1, sin);  getitem = sin = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    with_effects_2 = torch._higher_order_ops.effects.with_effects(getitem_2, torch.ops._TorchScriptTesting.queue_pop.default, arg3_1);  getitem_2 = None
    getitem_4 = with_effects_2[0]
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    with_effects_3 = torch._higher_order_ops.effects.with_effects(getitem_4, torch.ops._TorchScriptTesting.queue_pop.default, arg3_1);  getitem_4 = None
    getitem_6 = with_effects_3[0]
    getitem_7 = with_effects_3[1];  with_effects_3 = None
    t = torch.ops.aten.t.default(arg1_1)
    addmm = torch.ops.aten.addmm.default(arg2_1, getitem_5, t);  getitem_5 = t = None
    t_1 = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    addmm_1 = torch.ops.aten.addmm.default(arg2_1, getitem_7, t_1);  arg2_1 = getitem_7 = t_1 = None
    return (getitem_6, addmm, addmm_1, arg3_1)
    """,  # noqa: B950
        )


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
