# Owner(s): ["oncall: export"]

import unittest

import torch
import torch.utils._pytree as pytree
from torch._functorch.aot_autograd import aot_export_module
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

        self.torch_bind_ops = [
            torch.ops._TorchScriptTesting.takes_foo,
            torch.ops._TorchScriptTesting.takes_foo_python_meta,
            torch.ops._TorchScriptTesting.takes_foo_list_return,
            torch.ops._TorchScriptTesting.takes_foo_tuple_return,
            torch.ops._TorchScriptTesting.take_an_instance,
            torch.ops._TorchScriptTesting.take_an_instance_inferred,
            torch.ops._TorchScriptTesting.takes_foo_cia,
            torch.ops._TorchScriptTesting.queue_pop,
            torch.ops._TorchScriptTesting.queue_push,
            torch.ops._TorchScriptTesting.queue_size,
        ]

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
    x, n, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    attr = self.attr
    call_torchbind = torch.ops.higher_order.call_torchbind(attr, 'add_tensor', x);  attr = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, obj_attr, x, n):
    call_torchbind = torch.ops.higher_order.call_torchbind(obj_attr, 'add_tensor', x);  obj_attr = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
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
    x, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    attr = self.attr
    call_torchbind = torch.ops.higher_order.call_torchbind(attr, 'add_tensor', x);  attr = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, obj_attr, x):
    call_torchbind = torch.ops.higher_order.call_torchbind(obj_attr, 'add_tensor', x);  obj_attr = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
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
    x, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    attr = self.attr
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, x);  attr = None
    add = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x):
    with_effects = torch._higher_order_ops.effects.with_effects(token, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, x);  token = obj_attr = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
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
    x, cc, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    call_torchbind = torch.ops.higher_order.call_torchbind(cc, 'add_tensor', x);  cc = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, x, cc):
    call_torchbind = torch.ops.higher_order.call_torchbind(cc, 'add_tensor', x);  cc = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
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
    x, cc, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(cc, x);  cc = None
    add = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, x, cc):
    with_effects = torch._higher_order_ops.effects.with_effects(token, torch.ops._TorchScriptTesting.takes_foo.default, cc, x);  token = cc = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
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
    x, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    attr = self.attr
    takes_foo_default_1 = torch.ops._TorchScriptTesting.takes_foo.default(attr, x)
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, takes_foo_default_1);  attr = takes_foo_default_1 = None
    add = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add,), self._out_spec)""",  # noqa: B950
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x):
    with_effects = torch._higher_order_ops.effects.with_effects(token, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, x);  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, getitem_1);  getitem = obj_attr = getitem_1 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(x, getitem_3);  x = getitem_3 = None
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
    x, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    attr = self.attr
    takes_foo_list_return_default = torch.ops._TorchScriptTesting.takes_foo_list_return.default(attr, x)
    getitem_2 = takes_foo_list_return_default[0]
    getitem_3 = takes_foo_list_return_default[1]
    getitem_4 = takes_foo_list_return_default[2];  takes_foo_list_return_default = None
    add = torch.ops.aten.add.Tensor(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    add_1 = torch.ops.aten.add.Tensor(add, getitem_4);  add = getitem_4 = None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, add_1);  attr = add_1 = None
    add_2 = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add_2,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x):
    with_effects = torch._higher_order_ops.effects.with_effects(token, torch.ops._TorchScriptTesting.takes_foo_list_return.default, obj_attr, x);  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    getitem_2 = getitem_1[0]
    getitem_3 = getitem_1[1]
    getitem_4 = getitem_1[2];  getitem_1 = None
    add = torch.ops.aten.add.Tensor(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    add_1 = torch.ops.aten.add.Tensor(add, getitem_4);  add = getitem_4 = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, add_1);  getitem = obj_attr = add_1 = None
    getitem_5 = with_effects_1[0]
    getitem_6 = with_effects_1[1];  with_effects_1 = None
    add_2 = torch.ops.aten.add.Tensor(x, getitem_6);  x = getitem_6 = None
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
    x, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    attr = self.attr
    takes_foo_tuple_return_default = torch.ops._TorchScriptTesting.takes_foo_tuple_return.default(attr, x)
    getitem_1 = takes_foo_tuple_return_default[0]
    getitem_2 = takes_foo_tuple_return_default[1];  takes_foo_tuple_return_default = None
    add = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, add);  attr = add = None
    add_1 = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add_1,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x):
    with_effects = torch._higher_order_ops.effects.with_effects(token, torch.ops._TorchScriptTesting.takes_foo_tuple_return.default, obj_attr, x);  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1]
    getitem_2 = with_effects[2];  with_effects = None
    add = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, add);  getitem = obj_attr = add = None
    getitem_3 = with_effects_1[0]
    getitem_4 = with_effects_1[1];  with_effects_1 = None
    add_1 = torch.ops.aten.add.Tensor(x, getitem_4);  x = getitem_4 = None
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

    def test_identifying_torchbind_ops(self):
        for op in self.torch_bind_ops:
            self.assertTrue(op._has_torchbind_op_overload)

        for op in [
            torch.ops.aten.add,
            torch.ops.aten.cos,
        ]:
            self.assertFalse(op._has_torchbind_op_overload)

    def test_torchbind_op_register_fallthrough(self):
        TEST_DISPATCH_KEY = torch._C.DispatchKey.AutocastCPU
        TEST_DISPATCH_KEY_STR = "AutocastCPU"

        for op_packet in self.torch_bind_ops:
            op = op_packet.default
            ns, _ = torch._library.utils.parse_namespace(op_packet._qualified_op_name)
            with torch.library._scoped_library(ns, "FRAGMENT") as lib:
                lib.impl(
                    op.name(), torch.library.fallthrough_kernel, TEST_DISPATCH_KEY_STR
                )
                self.assertTrue(
                    torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough(
                        op.name(), TEST_DISPATCH_KEY
                    )
                )

    def test_torchbind_op_fallthrough_keys_respects_lib_impl(self):
        TEST_DISPATCH_KEY = torch._C.DispatchKey.AutogradCPU
        TEST_DISPATCH_KEY_STR = "AutogradCPU"

        tested = 0
        for op_packet in self.torch_bind_ops:
            op = op_packet.default
            ns, _ = torch._library.utils.parse_namespace(op_packet._qualified_op_name)
            if (
                not torch._C._dispatch_has_kernel_for_dispatch_key(
                    op.name(), TEST_DISPATCH_KEY
                )
                and TEST_DISPATCH_KEY not in op.py_kernels
            ):
                tested += 1
                with torch.library._scoped_library(ns, "FRAGMENT") as lib:
                    lib.impl(
                        op.name(), lambda *args, **kwargs: args, TEST_DISPATCH_KEY_STR
                    )
                    self.assertTrue(TEST_DISPATCH_KEY not in op._fallthrough_keys())

                with torch.library._scoped_library(ns, "FRAGMENT") as lib:
                    lib.impl(
                        op.name(),
                        torch.library.fallthrough_kernel,
                        TEST_DISPATCH_KEY_STR,
                    )
                    self.assertTrue(TEST_DISPATCH_KEY in op._fallthrough_keys())
        self.assertTrue(tested > 0)

    def test_make_fx_schema_checking_script_object(self):
        class Model(torch.nn.Module):
            def forward(self, tq, x, foo):
                torch.ops._TorchScriptTesting.queue_push(foo, x.cos())
                return tq

        class ModelCallByKW(torch.nn.Module):
            def forward(self, tq, x, foo):
                torch.ops._TorchScriptTesting.queue_push(x=x.cos(), foo=foo)
                return tq

        mod = Model()
        modkw = ModelCallByKW()

        foo = torch.classes._TorchScriptTesting._Foo(10, 20)
        x = torch.ones(3, 3)
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        ns = "_TorchScriptTesting"
        with torch.library._scoped_library(ns, "FRAGMENT") as lib:
            op = torch.ops._TorchScriptTesting.queue_push
            lib.impl(op.__name__, torch.library.fallthrough_kernel, "AutogradCPU")
            lib.impl(op.__name__, torch.library.fallthrough_kernel, "ADInplaceOrView")
            lib.impl(
                op.__name__,
                torch.library.fallthrough_kernel,
                "PythonTLSSnapshot",
            )

            with self.assertRaisesRegex(
                RuntimeError, "is expected to be a FakeScriptObject"
            ):
                _ = make_fx(mod, tracing_mode="fake")(tq, x, foo)

            with self.assertRaisesRegex(
                RuntimeError, "is expected to be a FakeScriptObject"
            ):
                _ = make_fx(modkw, tracing_mode="fake")(tq, x, foo)

    @parametrize("fallthrough_via", ["lib_impl", "py_impl"])
    def test_make_fx_tensor_queue_operators(self, fallthrough_via):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, tq, x):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                    torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
                    x_sin = torch.ops._TorchScriptTesting.queue_pop(
                        tq
                    ) - torch.ops._TorchScriptTesting.queue_size(tq)
                    x_cos = torch.ops._TorchScriptTesting.queue_pop(
                        tq
                    ) + torch.ops._TorchScriptTesting.queue_size(tq)
                    return x_sin, x_cos, tq

        mod = Model()

        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq2 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        x = torch.ones(2, 3)

        mod(tq1, x)

        ops = [
            torch.ops._TorchScriptTesting.queue_push,
            torch.ops._TorchScriptTesting.queue_pop,
            torch.ops._TorchScriptTesting.queue_size,
        ]
        if fallthrough_via == "lib_impl":
            ns = "_TorchScriptTesting"
            with torch.library._scoped_library(ns, "FRAGMENT") as lib:
                for op in ops:
                    lib.impl(
                        op.__name__, torch.library.fallthrough_kernel, "AutocastCUDA"
                    )

                gm = make_fx(mod, tracing_mode="fake")(tq1, x)
        else:
            for op in ops:
                op.default.py_impl(torch._C.DispatchKey.AutocastCUDA)(
                    torch.library.fallthrough_kernel
                )
            gm = make_fx(mod, tracing_mode="fake")(tq1, x)
            for op in ops:
                op.default._dispatch_cache.clear()
                del op.default.py_kernels[torch._C.DispatchKey.AutocastCUDA]

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    cos = torch.ops.aten.cos.default(arg1_1)
    queue_push = torch.ops._TorchScriptTesting.queue_push.default(arg0_1, cos);  cos = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    queue_push_1 = torch.ops._TorchScriptTesting.queue_push.default(arg0_1, sin);  sin = None
    queue_pop = torch.ops._TorchScriptTesting.queue_pop.default(arg0_1)
    queue_size = torch.ops._TorchScriptTesting.queue_size.default(arg0_1)
    sub = torch.ops.aten.sub.Tensor(queue_pop, 1);  queue_pop = None
    queue_pop_1 = torch.ops._TorchScriptTesting.queue_pop.default(arg0_1)
    queue_size_1 = torch.ops._TorchScriptTesting.queue_size.default(arg0_1)
    add = torch.ops.aten.add.Tensor(queue_pop_1, 0);  queue_pop_1 = None
    return (sub, add, arg0_1)""",
        )
        self._assertEqualSkipScriptObject(gm(tq1, x), mod(tq2, x))

    def test_aot_export_tensor_queue_operators(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, tq, x):
                torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
                x_sin = torch.ops._TorchScriptTesting.queue_pop(
                    tq
                ) - torch.ops._TorchScriptTesting.queue_size(tq)
                x_cos = torch.ops._TorchScriptTesting.queue_pop(
                    tq
                ) + torch.ops._TorchScriptTesting.queue_size(tq)
                return x_sin, x_cos, tq

        mod = Model()

        tq1 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        x = torch.ones(2, 3)

        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        fake_tq1 = torch._library.fake_class_registry.to_fake_obj(fake_mode, tq1)
        fake_x = fake_mode.from_tensor(x)
        gm = aot_export_module(mod, (fake_tq1, fake_x), trace_joint=False)[0]

        # inputs: token, tq, x
        # return: token, x_sin, x_cos, tq
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    cos = torch.ops.aten.cos.default(arg2_1)
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops._TorchScriptTesting.queue_push.default, arg1_1, cos);  arg0_1 = cos = None
    getitem = with_effects[0];  with_effects = None
    sin = torch.ops.aten.sin.default(arg2_1);  arg2_1 = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops._TorchScriptTesting.queue_push.default, arg1_1, sin);  getitem = sin = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    with_effects_2 = torch._higher_order_ops.effects.with_effects(getitem_2, torch.ops._TorchScriptTesting.queue_pop.default, arg1_1);  getitem_2 = None
    getitem_4 = with_effects_2[0]
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    with_effects_3 = torch._higher_order_ops.effects.with_effects(getitem_4, torch.ops._TorchScriptTesting.queue_size.default, arg1_1);  getitem_4 = None
    getitem_6 = with_effects_3[0];  with_effects_3 = None
    sub = torch.ops.aten.sub.Tensor(getitem_5, 1);  getitem_5 = None
    with_effects_4 = torch._higher_order_ops.effects.with_effects(getitem_6, torch.ops._TorchScriptTesting.queue_pop.default, arg1_1);  getitem_6 = None
    getitem_8 = with_effects_4[0]
    getitem_9 = with_effects_4[1];  with_effects_4 = None
    with_effects_5 = torch._higher_order_ops.effects.with_effects(getitem_8, torch.ops._TorchScriptTesting.queue_size.default, arg1_1);  getitem_8 = None
    getitem_10 = with_effects_5[0];  with_effects_5 = None
    add = torch.ops.aten.add.Tensor(getitem_9, 0);  getitem_9 = None
    return (getitem_10, sub, add, arg1_1)""",  # noqa: B950
        )


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
