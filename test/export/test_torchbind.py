# Owner(s): ["oncall: export"]

import unittest

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch._library.fake_class_registry import FakeScriptObject
from torch.export import export
from torch.export._trace import _export
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    find_library_location,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.torchbind_impls import register_fake_operators


def load_torchbind_test_lib():
    if IS_SANDCASTLE or IS_FBCODE:
        torch.ops.load_library("//caffe2/test/cpp/jit:test_custom_class_registrations")
    elif IS_MACOS:
        raise unittest.SkipTest("non-portable load_library call used in test")
    else:
        lib_file_path = find_library_location("libtorchbind_test.so")
        if IS_WINDOWS:
            lib_file_path = find_library_location("torchbind_test.dll")
        torch.ops.load_library(str(lib_file_path))

    register_fake_operators()


@skipIfTorchDynamo("torchbind not supported with dynamo yet")
class TestExportTorchbind(TestCase):
    def setUp(self):
        load_torchbind_test_lib()

        @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
        class FakeFoo:
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

            @classmethod
            def from_real(cls, foo):
                (x, y), _ = foo.__getstate__()
                return cls(x, y)

            def add_tensor(self, z):
                return (self.x + self.y) * z

        test = self
        test.tq_push_counter = 0
        test.tq_pop_counter = 0
        test.tq_size_counter = 0

        @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
        class FakeTensorQueue:
            def __init__(self, q):
                self.queue = q

            @classmethod
            def from_real(cls, real_tq):
                ctx = torch.library.get_ctx()
                fake_queue = [ctx.to_fake_tensor(t) for t in real_tq.get_raw_queue()]
                return cls(fake_queue)

            def push(self, x):
                test.tq_push_counter += 1
                self.queue.append(x)

            def pop(self):
                test.tq_pop_counter += 1
                return self.queue.pop(0)

            def size(self):
                test.tq_size_counter += 1
                return len(self.queue)

    def tearDown(self):
        torch._library.fake_class_registry.deregister_fake_class(
            "_TorchScriptTesting::_Foo"
        )
        torch._library.fake_class_registry.deregister_fake_class(
            "_TorchScriptTesting::_TensorQueue"
        )

    def _assertEqualSkipScriptObject(self, exp, actual):
        flat_exp = pytree.tree_leaves(exp)
        flat_actual = pytree.tree_leaves(actual)
        self.assertEqual(len(flat_exp), len(flat_actual))
        for a, b in zip(flat_exp, flat_actual):
            if isinstance(a, torch.ScriptObject) and isinstance(b, torch.ScriptObject):
                continue
            self.assertEqual(a, b)

    def _test_export_same_as_eager(
        self, f, args, kwargs=None, strict=True, pre_dispatch=False
    ):
        kwargs = kwargs or {}

        def export_wrapper(f, args, kwargs, strcit, pre_dispatch):
            with enable_torchbind_tracing():
                if pre_dispatch:
                    exported_program = _export(
                        f, args, kwargs, strict=strict, pre_dispatch=True
                    )
                else:
                    exported_program = export(f, args, kwargs, strict=strict)
            return exported_program

        exported_program = export_wrapper(f, args, kwargs, strict, pre_dispatch)
        reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}
        unlifted = exported_program.module()
        exp = f(*args, **kwargs)
        self.assertEqual(unlifted(*args, **kwargs), exp)
        self.assertEqual(
            unlifted(*args, **reversed_kwargs),
            exp,
        )

        # check re-tracing
        retraced_ep = export_wrapper(unlifted, args, kwargs, strict, pre_dispatch)
        self.assertEqual(retraced_ep.module()(*args, **kwargs), exp)
        return exported_program

    @parametrize("pre_dispatch", [True, False])
    def test_none(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x, n):
                return x + self.attr.add_tensor(x)

        ep = self._test_export_same_as_eager(
            MyModule(),
            (torch.ones(2, 3), None),
            strict=False,
            pre_dispatch=pre_dispatch,
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, arg_0, arg_1):
    arg0_1, arg1_1, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    attr_1 = self.attr
    call_torchbind = torch.ops.higher_order.call_torchbind(attr_1, 'add_tensor', arg0_1);  attr_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, call_torchbind);  arg0_1 = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, attr, arg0_1, arg1_1):
    call_torchbind = torch.ops.higher_order.call_torchbind(attr, 'add_tensor', arg0_1);  attr = None
    add = torch.ops.aten.add.Tensor(arg0_1, call_torchbind);  arg0_1 = call_torchbind = None
    return (add,)""",
        )

    @parametrize("pre_dispatch", [True, False])
    def test_attribute(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return x + self.attr.add_tensor(x)

        ep = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3),), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, arg_0):
    arg0_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    attr_1 = self.attr
    call_torchbind = torch.ops.higher_order.call_torchbind(attr_1, 'add_tensor', arg0_1);  attr_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, call_torchbind);  arg0_1 = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, attr, arg0_1):
    call_torchbind = torch.ops.higher_order.call_torchbind(attr, 'add_tensor', arg0_1);  attr = None
    add = torch.ops.aten.add.Tensor(arg0_1, call_torchbind);  arg0_1 = call_torchbind = None
    return (add,)""",
        )

    @parametrize("pre_dispatch", [True, False])
    def test_attribute_as_custom_op_argument(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x)

        ep = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3),), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, arg_0):
    arg1_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    attr_1 = self.attr
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr_1, arg1_1);  attr_1 = None
    add = torch.ops.aten.add.Tensor(arg1_1, takes_foo_default);  arg1_1 = takes_foo_default = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, arg0_1, attr, arg1_1):
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops._TorchScriptTesting.takes_foo.default, attr, arg1_1);  arg0_1 = attr = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, getitem_1);  arg1_1 = getitem_1 = None
    return (getitem, add)""",  # noqa: B950
        )

    @parametrize("pre_dispatch", [True, False])
    def test_input(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, cc):
                return x + cc.add_tensor(x)

        cc = torch.classes._TorchScriptTesting._Foo(10, 20)
        ep = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), cc), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, arg_0, arg_1):
    arg0_1, arg1_1, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    call_torchbind = torch.ops.higher_order.call_torchbind(arg1_1, 'add_tensor', arg0_1);  arg1_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, call_torchbind);  arg0_1 = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    call_torchbind = torch.ops.higher_order.call_torchbind(arg1_1, 'add_tensor', arg0_1);  arg1_1 = None
    add = torch.ops.aten.add.Tensor(arg0_1, call_torchbind);  arg0_1 = call_torchbind = None
    return (add,)""",
        )

    @parametrize("pre_dispatch", [True, False])
    def test_input_as_custom_op_argument(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, cc):
                return x + torch.ops._TorchScriptTesting.takes_foo(cc, x)

        cc = torch.classes._TorchScriptTesting._Foo(10, 20)
        ep = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), cc), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, arg_0, arg_1):
    arg1_1, arg2_1, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(arg2_1, arg1_1);  arg2_1 = None
    add = torch.ops.aten.add.Tensor(arg1_1, takes_foo_default);  arg1_1 = takes_foo_default = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops._TorchScriptTesting.takes_foo.default, arg2_1, arg1_1);  arg0_1 = arg2_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, getitem_1);  arg1_1 = getitem_1 = None
    return (getitem, add)""",  # noqa: B950
        )

    @parametrize("pre_dispatch", [True, False])
    def test_unlift_custom_obj(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo(self.attr, x)
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, a)
                return x + b

        input = torch.ones(2, 3)
        ep = self._test_export_same_as_eager(
            MyModule(), (input,), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, arg_0):
    arg1_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    attr_1 = self.attr
    takes_foo_default_1 = torch.ops._TorchScriptTesting.takes_foo.default(attr_1, arg1_1)
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr_1, takes_foo_default_1);  attr_1 = takes_foo_default_1 = None
    add = torch.ops.aten.add.Tensor(arg1_1, takes_foo_default);  arg1_1 = takes_foo_default = None
    return pytree.tree_unflatten((add,), self._out_spec)""",  # noqa: B950
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, arg0_1, attr, arg1_1):
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops._TorchScriptTesting.takes_foo.default, attr, arg1_1);  arg0_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, attr, getitem_1);  getitem = attr = getitem_1 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(arg1_1, getitem_3);  arg1_1 = getitem_3 = None
    return (getitem_2, add)""",  # noqa: B950
        )

    @parametrize("pre_dispatch", [True, False])
    def test_custom_obj_list_out(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo_list_return(self.attr, x)
                y = a[0] + a[1] + a[2]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        input = torch.ones(2, 3)
        ep = self._test_export_same_as_eager(
            MyModule(), (input,), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, arg_0):
    arg1_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    attr_1 = self.attr
    takes_foo_list_return_default = torch.ops._TorchScriptTesting.takes_foo_list_return.default(attr_1, arg1_1)
    getitem_2 = takes_foo_list_return_default[0]
    getitem_3 = takes_foo_list_return_default[1]
    getitem_4 = takes_foo_list_return_default[2];  takes_foo_list_return_default = None
    add = torch.ops.aten.add.Tensor(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    add_1 = torch.ops.aten.add.Tensor(add, getitem_4);  add = getitem_4 = None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr_1, add_1);  attr_1 = add_1 = None
    add_2 = torch.ops.aten.add.Tensor(arg1_1, takes_foo_default);  arg1_1 = takes_foo_default = None
    return pytree.tree_unflatten((add_2,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, arg0_1, attr, arg1_1):
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops._TorchScriptTesting.takes_foo_list_return.default, attr, arg1_1);  arg0_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    getitem_2 = getitem_1[0]
    getitem_3 = getitem_1[1]
    getitem_4 = getitem_1[2];  getitem_1 = None
    add = torch.ops.aten.add.Tensor(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    add_1 = torch.ops.aten.add.Tensor(add, getitem_4);  add = getitem_4 = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, attr, add_1);  getitem = attr = add_1 = None
    getitem_5 = with_effects_1[0]
    getitem_6 = with_effects_1[1];  with_effects_1 = None
    add_2 = torch.ops.aten.add.Tensor(arg1_1, getitem_6);  arg1_1 = getitem_6 = None
    return (getitem_5, add_2)""",  # noqa: B950
        )

    @parametrize("pre_dispatch", [True, False])
    def test_custom_obj_tuple_out(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        input = torch.ones(2, 3)
        ep = self._test_export_same_as_eager(
            MyModule(), (input,), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, arg_0):
    arg1_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    attr_1 = self.attr
    takes_foo_tuple_return_default = torch.ops._TorchScriptTesting.takes_foo_tuple_return.default(attr_1, arg1_1)
    getitem_1 = takes_foo_tuple_return_default[0]
    getitem_2 = takes_foo_tuple_return_default[1];  takes_foo_tuple_return_default = None
    add = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr_1, add);  attr_1 = add = None
    add_1 = torch.ops.aten.add.Tensor(arg1_1, takes_foo_default);  arg1_1 = takes_foo_default = None
    return pytree.tree_unflatten((add_1,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, arg0_1, attr, arg1_1):
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops._TorchScriptTesting.takes_foo_tuple_return.default, attr, arg1_1);  arg0_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1]
    getitem_2 = with_effects[2];  with_effects = None
    add = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, attr, add);  getitem = attr = add = None
    getitem_3 = with_effects_1[0]
    getitem_4 = with_effects_1[1];  with_effects_1 = None
    add_1 = torch.ops.aten.add.Tensor(arg1_1, getitem_4);  arg1_1 = getitem_4 = None
    return (getitem_3, add_1)""",  # noqa: B950
        )

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx_tensor_queue_methods(self, make_fx_tracing_mode):
        test = self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)
                self.check_tq_is_fake = True

            def forward(self, tq, x):
                if self.check_tq_is_fake:
                    test.assertTrue(isinstance(tq, FakeScriptObject))
                tq.push(x.cos())
                tq.push(x.sin())
                x_cos = tq.pop() + tq.size()
                x_sin = tq.pop() - tq.size()
                return x_sin, x_cos, tq

        mod = Model()
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        x = torch.ones(2, 3)
        gm = make_fx(mod, tracing_mode=make_fx_tracing_mode)(tq, x)
        self.assertEqual(self.tq_push_counter, 2)
        self.assertEqual(self.tq_pop_counter, 2)
        self.assertEqual(self.tq_size_counter, 2)
        self.assertEqual(tq.size(), 0)
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1):
    cos = torch.ops.aten.cos.default(arg1_1)
    call_torchbind = torch.ops.higher_order.call_torchbind(arg0_1, 'push', cos);  cos = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    call_torchbind_1 = torch.ops.higher_order.call_torchbind(arg0_1, 'push', sin);  sin = None
    call_torchbind_2 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    call_torchbind_3 = torch.ops.higher_order.call_torchbind(arg0_1, 'size')
    add = torch.ops.aten.add.Tensor(call_torchbind_2, 1);  call_torchbind_2 = None
    call_torchbind_4 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    call_torchbind_5 = torch.ops.higher_order.call_torchbind(arg0_1, 'size')
    sub = torch.ops.aten.sub.Tensor(call_torchbind_4, 0);  call_torchbind_4 = None
    return (sub, add, arg0_1)
    """,
        )
        mod.check_tq_is_fake = False
        self._assertEqualSkipScriptObject(gm(tq, x), mod(tq1, x))

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx_tensor_queue_methods_fakify_internal_states(
        self, make_fx_tracing_mode
    ):
        test = self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)
                self.check_tq_is_fake = True
                self.current_test = test

            def forward(self, tq, x):
                if self.check_tq_is_fake:
                    self.current_test.assertTrue(isinstance(tq, FakeScriptObject))
                x_cos = tq.pop() + tq.size() + x
                x_sin = tq.pop() - tq.size() + x
                return x_sin, x_cos, tq

        mod = Model()
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        for _ in range(2):
            tq.push(torch.ones(2, 3))
            tq1.push(torch.ones(2, 3))
        x = torch.ones(2, 3)
        prev_size = tq.size()
        gm = make_fx(mod, tracing_mode=make_fx_tracing_mode)(tq, x)
        self.assertEqual(self.tq_push_counter, 0)
        self.assertEqual(self.tq_pop_counter, 2)
        self.assertEqual(self.tq_size_counter, 2)
        self.assertEqual(tq.size(), prev_size)
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1):
    call_torchbind = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    call_torchbind_1 = torch.ops.higher_order.call_torchbind(arg0_1, 'size')
    add = torch.ops.aten.add.Tensor(call_torchbind, 1);  call_torchbind = None
    add_1 = torch.ops.aten.add.Tensor(add, arg1_1);  add = None
    call_torchbind_2 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    call_torchbind_3 = torch.ops.higher_order.call_torchbind(arg0_1, 'size')
    sub = torch.ops.aten.sub.Tensor(call_torchbind_2, 0);  call_torchbind_2 = None
    add_2 = torch.ops.aten.add.Tensor(sub, arg1_1);  sub = arg1_1 = None
    return (add_2, add_1, arg0_1)
    """,
        )
        # turn off tq type checking in eager execution
        mod.check_tq_is_fake = False
        self._assertEqualSkipScriptObject(gm(tq, x), mod(tq1, x))
        self.assertEqual(tq.size(), 0)
        self.assertEqual(tq1.size(), 0)


@skipIfTorchDynamo("torchbind not supported with dynamo yet")
class TestRegisterFakeClass(TestCase):
    def setUp(self):
        load_torchbind_test_lib()

    def tearDown(self):
        torch._library.fake_class_registry.global_fake_class_registry.clear()

    def test_register_fake_class_no_torch_bind_class(self):
        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate class"):

            @torch._library.register_fake_class("_TorchScriptTesting::NOT_A_VALID_NAME")
            class Invalid:
                pass

    def test_register_fake_class_no_from_real(self):
        with self.assertRaisesRegex(RuntimeError, "define a classmethod from_real"):

            @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
            class InvalidFakeFoo:
                def __init__(self):
                    pass

    def test_register_fake_class_from_real_not_classmethod(self):
        with self.assertRaisesRegex(RuntimeError, "is not a classmethod"):

            @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
            class FakeFoo:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                def from_real(self, foo_obj):
                    x, y = foo_obj.__getstate__()
                    return FakeFoo(x, y)

    def test_register_fake_class_valid(self):
        class FakeFoo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            @classmethod
            def from_real(cls, foo_obj):
                x, y = foo_obj.__getstate__()
                return cls(x, y)

        torch._library.register_fake_class("_TorchScriptTesting::_Foo", FakeFoo)

    def test_register_fake_class_duplicate_registration(self):
        @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
        class FakeFoo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            @classmethod
            def from_real(cls, foo_obj):
                x, y = foo_obj.__getstate__()
                return cls(x, y)

        with self.assertWarnsRegex(UserWarning, "already registered"):
            torch._library.register_fake_class("_TorchScriptTesting::_Foo", FakeFoo)


instantiate_parametrized_tests(TestExportTorchbind)

if __name__ == "__main__":
    run_tests()
