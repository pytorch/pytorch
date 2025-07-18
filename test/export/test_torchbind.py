# Owner(s): ["oncall: export"]
# ruff: noqa: F841

import copy

import torch
import torch.utils._pytree as pytree
from torch._dynamo.testing import EagerAndRecordGraphs
from torch._functorch.aot_autograd import aot_export_module
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch._higher_order_ops.wrap import wrap
from torch._library.fake_class_registry import FakeScriptObject
from torch.export._trace import _export
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.torchbind_impls import (
    _empty_tensor_queue,
    init_torchbind_implementations,
)
from torch.testing._internal.triton_utils import requires_gpu


def _assertEqualSkipScriptObject(test_case, exp, actual):
    flat_exp = pytree.tree_leaves(exp)
    flat_actual = pytree.tree_leaves(actual)
    test_case.assertEqual(len(flat_exp), len(flat_actual))
    for a, b in zip(flat_exp, flat_actual):
        if isinstance(a, torch.ScriptObject) and isinstance(b, torch.ScriptObject):
            continue
        test_case.assertEqual(a, b)


def _check_script_obj_equal(test_case, a: torch.ScriptObject, b: torch.ScriptObject):
    return test_case.assertEqual(
        a._type().qualified_name(), b._type().qualified_name()
    ) and test_case.assertEqual(a.__obj_flatten__(), b.__obj_flatten__())


def _assertEqualScriptObject(
    test_case, exp, actual, check_obj_eq=_check_script_obj_equal
):
    flat_exp = pytree.tree_leaves(exp)
    flat_actual = pytree.tree_leaves(actual)
    test_case.assertEqual(len(flat_exp), len(flat_actual))
    for a, b in zip(flat_exp, flat_actual):
        if isinstance(a, torch.ScriptObject) and isinstance(b, torch.ScriptObject):
            check_obj_eq(test_case, a, b)
        else:
            test_case.assertEqual(a, b)


@skipIfTorchDynamo("torchbind not supported with dynamo yet")
class TestExportTorchbind(TestCase):
    def setUp(self):
        init_torchbind_implementations()

        test = self
        test.tq_push_counter = 0
        test.tq_pop_counter = 0
        test.tq_size_counter = 0
        test.foo_add_tensor_counter = 0

        # We need different fake classes, which update the counters
        @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
        class FakeFoo:
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

            @classmethod
            def __obj_unflatten__(cls, flattend_foo):
                return cls(**dict(flattend_foo))

            def add_tensor(self, z):
                test.foo_add_tensor_counter += 1
                return (self.x + self.y) * z

        @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
        class FakeTensorQueue:
            def __init__(self, queue):
                self.queue = queue

            @classmethod
            def __obj_unflatten__(cls, flattened_ctx):
                return cls(**dict(flattened_ctx))

            def push(self, x):
                test.tq_push_counter += 1
                self.queue.append(x)

            def pop(self):
                test.tq_pop_counter += 1
                return self.queue.pop(0)

            def size(self):
                test.tq_size_counter += 1
                return len(self.queue)

            def is_empty(self):
                return len(self.queue) == 0

            def float_size(self):
                return float(len(self.queue))

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

    def _test_export_same_as_eager(
        self, f, args, kwargs=None, strict=True, pre_dispatch=False
    ):
        kwargs = kwargs or {}

        def export_wrapper(f, args, kwargs, strict, pre_dispatch):
            with enable_torchbind_tracing():
                if pre_dispatch:
                    exported_program = torch.export.export_for_training(
                        f, args, kwargs, strict=strict
                    ).run_decompositions({})
                else:
                    exported_program = _export(
                        f, args, kwargs, strict=strict, pre_dispatch=False
                    )
            return exported_program

        exported_program = export_wrapper(f, args, kwargs, strict, pre_dispatch)
        reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}
        unlifted = exported_program.module()
        exp = f(*args, **kwargs)
        _assertEqualScriptObject(self, unlifted(*args, **kwargs), exp)
        _assertEqualScriptObject(
            self,
            unlifted(*args, **reversed_kwargs),
            exp,
        )

        # check re-tracing
        retraced_ep = export_wrapper(unlifted, args, kwargs, strict, pre_dispatch)
        _assertEqualScriptObject(self, retraced_ep.module()(*args, **kwargs), exp)
        return exported_program

    @parametrize("pre_dispatch", [True, False])
    def test_none(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
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
def forward(self, x, n):
    x, n, = fx_pytree.tree_flatten_spec(([x, n], {}), self._in_spec)
    attr = self.attr
    call_torchbind = torch.ops.higher_order.call_torchbind(attr, 'add_tensor', x);  attr = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x, n):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, obj_attr, 'add_tensor', x);  token = obj_attr = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
    return (getitem, add)""",  # noqa: B950
        )

    def test_method_schema(self):
        tq = _empty_tensor_queue()
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        fake_obj = torch._library.fake_class_registry.maybe_to_fake_obj(fake_mode, tq)
        self.assertExpectedInline(
            str(fake_obj.push.schema),
            """push(__torch__.torch.classes._TorchScriptTesting._TensorQueue _0, Tensor _1) -> NoneType _0""",
        )
        self.assertExpectedInline(
            str(fake_obj.pop.schema),
            """pop(__torch__.torch.classes._TorchScriptTesting._TensorQueue _0) -> Tensor _0""",
        )

    @parametrize("pre_dispatch", [True, False])
    def test_attribute(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
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
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    attr = self.attr
    call_torchbind = torch.ops.higher_order.call_torchbind(attr, 'add_tensor', x);  attr = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, obj_attr, 'add_tensor', x);  token = obj_attr = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
    return (getitem, add)""",  # noqa: B950
        )

    @parametrize("pre_dispatch", [True, False])
    def test_attribute_as_custom_op_argument(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
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
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    attr = self.attr
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(attr, x);  attr = None
    add = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj_attr, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, x);  token = obj_attr = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
    return (getitem, add)""",  # noqa: B950
        )

    @parametrize("pre_dispatch", [True, False])
    def test_input(self, pre_dispatch):
        cc = torch.classes._TorchScriptTesting._Foo(10, 20)

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, cc):
                return x + cc.add_tensor(x)

        ep = self._test_export_same_as_eager(
            MyModule(), (torch.ones(2, 3), cc), strict=False, pre_dispatch=pre_dispatch
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x, cc):
    x, cc, = fx_pytree.tree_flatten_spec(([x, cc], {}), self._in_spec)
    call_torchbind = torch.ops.higher_order.call_torchbind(cc, 'add_tensor', x);  cc = None
    add = torch.ops.aten.add.Tensor(x, call_torchbind);  x = call_torchbind = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, x, cc):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, cc, 'add_tensor', x);  token = cc = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
    return (getitem, add)""",  # noqa: B950
        )
        # aot_export_function runs the program twice
        # in run_functionalized_fw_and_collect_metadata and create_aot_dispatcher_function
        # We also have a re-tracing test, which doubles the count.
        if pre_dispatch:
            self.assertEqual(self.foo_add_tensor_counter, 6)
        else:
            self.assertEqual(self.foo_add_tensor_counter, 4)

    @parametrize("pre_dispatch", [True, False])
    def test_input_as_custom_op_argument(self, pre_dispatch):
        cc = torch.classes._TorchScriptTesting._Foo(10, 20)

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, cc):
                return x + torch.ops._TorchScriptTesting.takes_foo(cc, x)

        del torch.ops._TorchScriptTesting.takes_foo.default.py_kernels[
            torch._C.DispatchKey.Meta
        ]
        torch.ops._TorchScriptTesting.takes_foo.default._dispatch_cache.clear()
        # Even though a C++ implementation for takes_foo.default is registered,
        # we still need the python implementation for takes_foo.default to trace with FakeFoo.
        with self.assertRaisesRegex(RuntimeError, "no python implementation is found"):
            self._test_export_same_as_eager(
                MyModule(),
                (torch.ones(2, 3), cc),
                strict=False,
                pre_dispatch=pre_dispatch,
            )

        torch.ops._TorchScriptTesting.takes_foo.default.py_impl(
            torch._C.DispatchKey.Meta
        )(lambda cc, x: cc.add_tensor(x))
        ep = self._test_export_same_as_eager(
            MyModule(),
            (torch.ones(2, 3), cc),
            strict=False,
            pre_dispatch=pre_dispatch,
        )

        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, x, cc):
    x, cc, = fx_pytree.tree_flatten_spec(([x, cc], {}), self._in_spec)
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(cc, x);  cc = None
    add = torch.ops.aten.add.Tensor(x, takes_foo_default);  x = takes_foo_default = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, x, cc):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.takes_foo.default, cc, x);  token = cc = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(x, getitem_1);  x = getitem_1 = None
    return (getitem, add)""",  # noqa: B950
        )

    @parametrize("pre_dispatch", [True, False])
    def test_torchbind_alias(self, pre_dispatch):
        class F2(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def forward(self, x):
                return x + torch.ops._TorchScriptTesting.takes_foo(self.foo, x)

        class F1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.alpha = torch.classes._TorchScriptTesting._Foo(10, 20)
                self.beta = self.alpha
                self.gamma = self.alpha
                self.foo = F2(self.gamma)

            def forward(self, x):
                return (
                    x
                    + torch.ops._TorchScriptTesting.takes_foo(self.gamma, x)
                    + self.foo(x)
                )

        self._test_export_same_as_eager(
            F1(), (torch.ones(2, 3),), strict=False, pre_dispatch=pre_dispatch
        )

    def test_torchbind_register_attr_at_runtime_error(self):
        # alias as model attribute
        class F3(torch.nn.Module):
            def forward(self, x, foo):
                self.foo = foo
                return x + self.foo.add_tensor(x)

        foo = torch.classes._TorchScriptTesting._Foo(10, 20)
        with self.assertRaisesRegex(
            ValueError, "following attrs were created in the model"
        ):
            torch.export.export(F3(), (torch.ones(2, 3), foo))

    @parametrize("pre_dispatch", [True, False])
    def test_torchbind_input_and_alias(self, pre_dispatch):
        # alias as model attribute
        class F3(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def forward(self, x):
                return x + self.foo.add_tensor(x)

        foo = torch.classes._TorchScriptTesting._Foo(10, 20)
        self._test_export_same_as_eager(
            F3(foo), (torch.ones(2, 3),), strict=False, pre_dispatch=pre_dispatch
        )

    @parametrize("pre_dispatch", [True, False])
    def test_unlift_custom_obj(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
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
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
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
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, x);  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, getitem_1);  getitem = obj_attr = getitem_1 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(x, getitem_3);  x = getitem_3 = None
    return (getitem_2, add)""",  # noqa: B950
        )

    @parametrize("pre_dispatch", [True, False])
    def test_custom_obj_list_out(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
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
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
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
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.takes_foo_list_return.default, obj_attr, x);  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    getitem_2 = getitem_1[0]
    getitem_3 = getitem_1[1]
    getitem_4 = getitem_1[2];  getitem_1 = None
    add = torch.ops.aten.add.Tensor(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    add_1 = torch.ops.aten.add.Tensor(add, getitem_4);  add = getitem_4 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, add_1);  getitem = obj_attr = add_1 = None
    getitem_5 = with_effects_1[0]
    getitem_6 = with_effects_1[1];  with_effects_1 = None
    add_2 = torch.ops.aten.add.Tensor(x, getitem_6);  x = getitem_6 = None
    return (getitem_5, add_2)""",  # noqa: B950
        )

    @parametrize("pre_dispatch", [True, False])
    def test_custom_obj_tuple_out(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
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
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
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
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.takes_foo_tuple_return.default, obj_attr, x);  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1]
    getitem_2 = with_effects[2];  with_effects = None
    add = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TorchScriptTesting.takes_foo.default, obj_attr, add);  getitem = obj_attr = add = None
    getitem_3 = with_effects_1[0]
    getitem_4 = with_effects_1[1];  with_effects_1 = None
    add_1 = torch.ops.aten.add.Tensor(x, getitem_4);  x = getitem_4 = None
    return (getitem_3, add_1)""",  # noqa: B950
        )

    @parametrize("pre_dispatch", [True, False])
    def test_custom_obj_unbacked_symint(self, pre_dispatch):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(2, 3)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo_tensor_return(self.attr, x)
                return a

        input = torch.ones(2, 3)
        ep = self._test_export_same_as_eager(
            MyModule(), (input,), strict=False, pre_dispatch=pre_dispatch
        )
        gm = ep.module()
        foo_node = next(
            n
            for n in gm.graph.nodes
            if n.target == torch.ops._TorchScriptTesting.takes_foo_tensor_return.default
        )
        unbacked_bindings = foo_node.meta["unbacked_bindings"]
        self.assertEqual(len(unbacked_bindings), 2)
        u = next(iter(unbacked_bindings.keys()))
        path = unbacked_bindings[u]
        # the unbacked bindings should be CallMethodKey(name='size'), SequenceKey(idx=0)
        # it should not include the effect token in the path
        self.assertEqual(
            type(u).__name__, "Symbol"
        )  # check binding is symbol, not expr
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0].name, "size")
        self.assertEqual(path[1].idx, 0)

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx_tensor_queue_methods(self, make_fx_tracing_mode):
        test = self

        class Model(torch.nn.Module):
            def __init__(self) -> None:
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
    call_torchbind = torch.ops.higher_order.call_torchbind(arg0_1, 'push', cos);  cos = call_torchbind = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    call_torchbind_1 = torch.ops.higher_order.call_torchbind(arg0_1, 'push', sin);  sin = call_torchbind_1 = None
    call_torchbind_2 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    call_torchbind_3 = torch.ops.higher_order.call_torchbind(arg0_1, 'size');  call_torchbind_3 = None
    add = torch.ops.aten.add.Tensor(call_torchbind_2, 1);  call_torchbind_2 = None
    call_torchbind_4 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    call_torchbind_5 = torch.ops.higher_order.call_torchbind(arg0_1, 'size');  call_torchbind_5 = None
    sub = torch.ops.aten.sub.Tensor(call_torchbind_4, 0);  call_torchbind_4 = None
    return (sub, add, arg0_1)
    """,
        )
        mod.check_tq_is_fake = False
        _assertEqualSkipScriptObject(self, gm(tq, x), mod(tq1, x))

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx_tensor_queue_methods_fakify_internal_states(
        self, make_fx_tracing_mode
    ):
        test = self

        class Model(torch.nn.Module):
            def __init__(self) -> None:
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
    call_torchbind_1 = torch.ops.higher_order.call_torchbind(arg0_1, 'size');  call_torchbind_1 = None
    add = torch.ops.aten.add.Tensor(call_torchbind, 1);  call_torchbind = None
    add_1 = torch.ops.aten.add.Tensor(add, arg1_1);  add = None
    call_torchbind_2 = torch.ops.higher_order.call_torchbind(arg0_1, 'pop')
    call_torchbind_3 = torch.ops.higher_order.call_torchbind(arg0_1, 'size');  call_torchbind_3 = None
    sub = torch.ops.aten.sub.Tensor(call_torchbind_2, 0);  call_torchbind_2 = None
    add_2 = torch.ops.aten.add.Tensor(sub, arg1_1);  sub = arg1_1 = None
    return (add_2, add_1, arg0_1)
    """,
        )
        # turn off tq type checking in eager execution
        mod.check_tq_is_fake = False
        _assertEqualSkipScriptObject(self, gm(tq, x), mod(tq1, x))
        self.assertEqual(tq.size(), 0)
        self.assertEqual(tq1.size(), 0)

    def test_non_strict_export_methods(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, tq, x):
                x_cos = tq.pop() + tq.float_size() + self.linear(x)
                if tq.is_empty():
                    x_sin = self.linear(tq.pop()) - tq.size() + x
                else:
                    x_sin = tq.pop() + tq.size() + x
                return x_sin, x_cos, tq

        mod = Model()
        tq = _empty_tensor_queue()
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        tq.push(a)
        tq.push(b)
        ep = torch.export.export_for_training(
            mod, (tq, torch.randn(2, 2)), strict=False
        ).run_decompositions({})
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, p_linear_weight, p_linear_bias, tq, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, tq, 'pop');  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.higher_order.call_torchbind, tq, 'float_size');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(getitem_1, 1.0);  getitem_1 = None
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  p_linear_weight = p_linear_bias = None
    add_1 = torch.ops.aten.add.Tensor(add, linear);  add = linear = None
    with_effects_2 = torch.ops.higher_order.with_effects(getitem_2, torch.ops.higher_order.call_torchbind, tq, 'is_empty');  getitem_2 = None
    getitem_4 = with_effects_2[0];  with_effects_2 = None
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_4, torch.ops.higher_order.call_torchbind, tq, 'pop');  getitem_4 = None
    getitem_6 = with_effects_3[0]
    getitem_7 = with_effects_3[1];  with_effects_3 = None
    with_effects_4 = torch.ops.higher_order.with_effects(getitem_6, torch.ops.higher_order.call_torchbind, tq, 'size');  getitem_6 = None
    getitem_8 = with_effects_4[0];  with_effects_4 = None
    add_2 = torch.ops.aten.add.Tensor(getitem_7, 0);  getitem_7 = None
    add_3 = torch.ops.aten.add.Tensor(add_2, x);  add_2 = x = None
    return (getitem_8, add_3, add_1, tq)""",  # noqa: B950
        )
        self.assertEqual(tq.size(), 2)
        self.assertTrue(tq.pop() is a)
        self.assertTrue(tq.pop() is b)

    @skipIfCrossRef  # arg names change with torch function mode
    def test_safe_to_trace_with_real(self):
        x = torch.randn(3, 3)
        safe_obj = torch.classes._TorchScriptTesting._ConstantTensorContainer(x)

        class Mod(torch.nn.Module):
            def forward(self, safe_obj: torch.ScriptObject) -> None:
                return safe_obj.get().sin()

        mod = Mod()
        backend = EagerAndRecordGraphs()
        out = torch.compile(mod, backend=backend, fullgraph=True)(safe_obj)
        self.assertEqual(out, mod(safe_obj))
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_safe_obj_ : torch.ScriptObject):
    l_safe_obj_ = L_safe_obj_
    call_torchbind = torch.ops.higher_order.call_torchbind(l_safe_obj_, 'get');  l_safe_obj_ = None
    sin = call_torchbind.sin();  call_torchbind = None
    return (sin,)""",
        )

        with enable_torchbind_tracing():
            ep = torch.export.export_for_training(
                mod, (safe_obj,), strict=False
            ).run_decompositions({})
            self.assertExpectedInline(
                ep.graph_module.code.strip(),
                """\
def forward(self, token, safe_obj):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, safe_obj, 'get');  token = safe_obj = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    sin = torch.ops.aten.sin.default(getitem_1);  getitem_1 = None
    return (getitem, sin)""",  # noqa: B950
            )

    def test_identifying_torchbind_ops(self):
        for op in self.torch_bind_ops:
            self.assertTrue(op._has_torchbind_op_overload)

        for op in [
            torch.ops.aten.add,
            torch.ops.aten.cos,
        ]:
            self.assertFalse(op._has_torchbind_op_overload)

    def test_torchbind_op_register_fallthrough(self):
        TEST_DISPATCH_KEY = torch._C.DispatchKey.AutogradCPU
        TEST_DISPATCH_KEY_STR = "AutogradCPU"

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
            def __init__(self) -> None:
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
                        op.__name__, torch.library.fallthrough_kernel, "AutogradCPU"
                    )

                gm = make_fx(mod, tracing_mode="fake")(tq1, x)
        else:
            for op in ops:
                op.default.py_impl(torch._C.DispatchKey.AutogradCPU)(
                    torch.library.fallthrough_kernel
                )
            gm = make_fx(mod, tracing_mode="fake")(tq1, x)
            for op in ops:
                op.default._dispatch_cache.clear()
                del op.default.py_kernels[torch._C.DispatchKey.AutogradCPU]

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    cos = torch.ops.aten.cos.default(arg1_1)
    queue_push = torch.ops._TorchScriptTesting.queue_push.default(arg0_1, cos);  cos = queue_push = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    queue_push_1 = torch.ops._TorchScriptTesting.queue_push.default(arg0_1, sin);  sin = queue_push_1 = None
    queue_pop = torch.ops._TorchScriptTesting.queue_pop.default(arg0_1)
    queue_size = torch.ops._TorchScriptTesting.queue_size.default(arg0_1);  queue_size = None
    sub = torch.ops.aten.sub.Tensor(queue_pop, 1);  queue_pop = None
    queue_pop_1 = torch.ops._TorchScriptTesting.queue_pop.default(arg0_1)
    queue_size_1 = torch.ops._TorchScriptTesting.queue_size.default(arg0_1);  queue_size_1 = None
    add = torch.ops.aten.add.Tensor(queue_pop_1, 0);  queue_pop_1 = None
    return (sub, add, arg0_1)""",
        )
        _assertEqualSkipScriptObject(self, gm(tq1, x), mod(tq2, x))

    def test_aot_export_tensor_queue_operators(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
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
        fake_tq1 = torch._library.fake_class_registry.maybe_to_fake_obj(fake_mode, tq1)
        fake_x = fake_mode.from_tensor(x)
        gm = aot_export_module(mod, (fake_tq1, fake_x), trace_joint=False)[0]

        # inputs: token, tq, x
        # return: token, x_sin, x_cos, tq
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    cos = torch.ops.aten.cos.default(arg2_1)
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops._TorchScriptTesting.queue_push.default, arg1_1, cos);  arg0_1 = cos = None
    getitem = with_effects[0];  with_effects = None
    sin = torch.ops.aten.sin.default(arg2_1);  arg2_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._TorchScriptTesting.queue_push.default, arg1_1, sin);  getitem = sin = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    with_effects_2 = torch.ops.higher_order.with_effects(getitem_2, torch.ops._TorchScriptTesting.queue_pop.default, arg1_1);  getitem_2 = None
    getitem_4 = with_effects_2[0]
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_4, torch.ops._TorchScriptTesting.queue_size.default, arg1_1);  getitem_4 = None
    getitem_6 = with_effects_3[0];  with_effects_3 = None
    sub = torch.ops.aten.sub.Tensor(getitem_5, 1);  getitem_5 = None
    with_effects_4 = torch.ops.higher_order.with_effects(getitem_6, torch.ops._TorchScriptTesting.queue_pop.default, arg1_1);  getitem_6 = None
    getitem_8 = with_effects_4[0]
    getitem_9 = with_effects_4[1];  with_effects_4 = None
    with_effects_5 = torch.ops.higher_order.with_effects(getitem_8, torch.ops._TorchScriptTesting.queue_size.default, arg1_1);  getitem_8 = None
    getitem_10 = with_effects_5[0];  with_effects_5 = None
    add = torch.ops.aten.add.Tensor(getitem_9, 0);  getitem_9 = None
    return (getitem_10, sub, add, arg1_1)""",  # noqa: B950
        )

    def test_export_inplace_custom_op(self):
        class Model(torch.nn.Module):
            def forward(self, tq: torch.ScriptObject, x: torch.Tensor) -> None:
                torch.ops._TorchScriptTesting.queue_push(tq, x)
                return tq

        mod = Model()
        ep = self._test_export_same_as_eager(
            mod,
            (_empty_tensor_queue(), torch.randn(3, 3)),
            strict=False,
            pre_dispatch=True,
        )
        self.assertExpectedInline(
            ep.module().code.strip(),
            """\
def forward(self, tq, x):
    tq, x, = fx_pytree.tree_flatten_spec(([tq, x], {}), self._in_spec)
    queue_push_default = torch.ops._TorchScriptTesting.queue_push.default(tq, x);  x = queue_push_default = None
    return pytree.tree_unflatten((tq,), self._out_spec)""",
        )
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, tq, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops._TorchScriptTesting.queue_push.default, tq, x);  token = x = None
    getitem = with_effects[0];  with_effects = None
    return (getitem, tq)""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(ep.graph_module.graph).strip(),
            """\
graph():
    %tq : [num_users=2] = placeholder[target=tq]
    %x : [num_users=1] = placeholder[target=x]
    %queue_push_default : [num_users=0] = call_function[target=torch.ops._TorchScriptTesting.queue_push.default](args = (%tq, %x), kwargs = {})
    return (tq,)""",  # noqa: B950
        )

    def test_deepcopy(self):
        tq = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq_0 = copy.deepcopy(tq)
        tq.push(torch.zeros(2, 2))
        tq.push(torch.ones(2, 2))
        tq_1 = copy.deepcopy(tq)
        tq.push(torch.ones(2, 2) * 2)
        self.assertEqual(tq_0.size(), 0)
        self.assertEqual(tq_1.size(), 2)
        self.assertEqual(tq.size(), 3)

        foo = torch.classes._TorchScriptTesting._Foo(1, 2)
        foo_0 = copy.deepcopy(foo)
        foo.increment(1)
        foo_1 = copy.deepcopy(foo)
        foo.increment(1)
        self.assertEqual(foo_0.add(1), 3)
        self.assertEqual(foo_1.add(1), 5)
        self.assertEqual(foo.add(1), 7)


class TestCompileTorchbind(TestCase):
    def setUp(self):
        init_torchbind_implementations()

        @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
        class FakeTensorQueue:
            def __init__(self, queue):
                self.queue = queue

            @classmethod
            def __obj_unflatten__(cls, flattened_ctx):
                return cls(**dict(flattened_ctx))

            def push(self, x):
                self.queue.append(x)

            def pop(self):
                return self.queue.pop(0)

            def size(self):
                return len(self.queue)

        @torch._library.register_fake_class("_TorchScriptTesting::_FlattenWithTensorOp")
        class FakeFlatten:
            def __init__(self, t):
                self.t = t

            def get(self):
                return self.t

            @classmethod
            def __obj_unflatten__(cls, flattened_ctx):
                return cls(**dict(flattened_ctx))

        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_script_object_input(self, backend):
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.check_tq_is_fake = True

            def forward(self, tq, x):
                tq.push(x.cos())
                tq.push(x.sin())
                x_sin = tq.pop() - tq.size()
                return x_sin, tq

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
        tq3 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        tq4 = torch.classes._TorchScriptTesting._TensorQueue(
            torch.empty(
                0,
            ).fill_(-1)
        )
        x = torch.randn(2, 3)
        ret = torch.compile(mod, backend=backend)(tq1, x)
        eager_ret = mod(tq2, x)
        _assertEqualSkipScriptObject(self, ret, eager_ret)
        self.assertEqual(ret[1].size(), eager_ret[1].size())
        self.assertEqual(ret[1].pop(), eager_ret[1].pop())
        # Note that dynamo captured graph
        # does not return L_tq_ as output. This is because it's able
        # to detect that L_tq_ is an input therefore don't return
        # it as graph output. Related logic is in dynamo/codegen.py
        if backend == "eager":
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
    def forward(self, L_tq_ : torch.ScriptObject, L_x_ : torch.Tensor):
        l_tq_ = L_tq_
        l_x_ = L_x_
        cos = l_x_.cos()
        call_torchbind = torch.ops.higher_order.call_torchbind(l_tq_, 'push', cos);  cos = None
        sin = l_x_.sin();  l_x_ = None
        call_torchbind_1 = torch.ops.higher_order.call_torchbind(l_tq_, 'push', sin);  sin = None
        call_torchbind_2 = torch.ops.higher_order.call_torchbind(l_tq_, 'pop')
        call_torchbind_3 = torch.ops.higher_order.call_torchbind(l_tq_, 'size');  l_tq_ = None
        x_sin = call_torchbind_2 - 1;  call_torchbind_2 = None
        return (x_sin,)""",
            )

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_script_object_input_guards(self, backend):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.check_tq_is_fake = True

            def forward(self, tq, x):
                tq.push(x.cos())
                tq.push(x.sin())
                x_sin = tq.pop() - tq.size()
                return x_sin, tq

        mod = Model()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
        x = torch.randn(2, 3)

        tq1 = _empty_tensor_queue()
        torch.compile(mod, backend=cnt)(tq1, x)
        self.assertEqual(cnt.frame_count, 1)

        tq2 = _empty_tensor_queue()
        for _ in range(10):
            tq2.push(torch.randn(4, 5, requires_grad=False))
        torch.compile(mod, backend=cnt)(tq2, x)
        # Queue length change causes re-compile
        self.assertEqual(cnt.frame_count, 2)

        tq3 = _empty_tensor_queue()
        tq3.push(torch.randn(2, 3, requires_grad=False))
        torch.compile(mod, backend=cnt)(tq3, x)
        # Tensor in queue changes shape causes re-compile
        self.assertEqual(cnt.frame_count, 3)

        tq4 = _empty_tensor_queue()
        tq4.push(torch.randn(2, 3, requires_grad=False))
        torch.compile(mod, backend=cnt)(tq4, x)
        # No recompile
        self.assertEqual(cnt.frame_count, 3)

        tq5 = _empty_tensor_queue()
        tq5.push(torch.randn(2, 3, requires_grad=True))
        torch.compile(mod, backend=cnt)(tq5, x)
        # Tensor in queue changes dispatch key causes re-compile
        self.assertEqual(cnt.frame_count, 4)

        tq6 = _empty_tensor_queue()
        tq6.push(torch.randn(2, 3, requires_grad=True, dtype=torch.float64))
        torch.compile(mod, backend=cnt)(tq6, x)
        # Tensor in queue changes dtype causes re-compile
        self.assertEqual(cnt.frame_count, 5)

    def test_compile_script_object_input_automatic_dynamic_shape(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.check_tq_is_fake = True

            def forward(self, tq, x):
                tq.push(x.cos())
                tq.push(x.sin())
                x_sin = tq.pop() - tq.size()
                return x_sin, tq

        mod = Model()
        cnt = torch._dynamo.testing.CompileCounter()
        x = torch.randn(2, 3)

        tq1 = _empty_tensor_queue()
        tq1.push(torch.randn(2, 3, requires_grad=False))
        torch.compile(mod, backend=cnt)(tq1, x)
        self.assertEqual(cnt.frame_count, 1)

        tq2 = _empty_tensor_queue()
        # make first tensor's secon dim dynamic
        tq2.push(torch.randn(2, 4, requires_grad=False))
        torch.compile(mod, backend=cnt)(tq2, x)
        self.assertEqual(cnt.frame_count, 2)

        tq3 = _empty_tensor_queue()
        tq3.push(torch.randn(2, 5, requires_grad=False))
        # should have no-recompilation
        torch.compile(mod, backend=cnt)(tq3, x)
        self.assertEqual(cnt.frame_count, 2)

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_error_on_input_aliasing_contents(self, backend):
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.check_tq_is_fake = True

            def forward(self, tq, x):
                return x.sin(), tq.pop().cos()

        x = torch.randn(2, 3)
        mod = Model()

        tq1 = _empty_tensor_queue()
        tq1.push(x)
        with self.assertRaisesRegex(RuntimeError, "is aliasing"):
            torch.compile(mod, backend=backend)(tq1, x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_compile_error_on_script_obj_setattr(self, backend):
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        def setattr_f(tq):
            tq.a = 1
            return tq

        with self.assertRaisesRegex(
            RuntimeError, "call method __setattr__ on script object is not safe"
        ):
            torch.compile(setattr_f, backend=backend)(_empty_tensor_queue())

    @parametrize("backend", ["eager", "aot_eager"])
    def test_compile_error_on_script_obj_missing_attr(self, backend):
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        def setattr_f(tq):
            return tq._not_defined_attr

        with self.assertRaisesRegex(
            RuntimeError, "doesn't define method _not_defined_attr"
        ):
            torch.compile(setattr_f, backend=backend)(_empty_tensor_queue())

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_body_aliasing_contents(self, backend):
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        def f(tq, x):
            x1 = x.view(-1)
            x2 = x.permute(1, 0)
            tq.push(x1)
            tq.push(x2)
            return x1 - tq.size(), x2 + tq.size(), tq

        x = torch.randn(2, 3)
        _assertEqualScriptObject(
            self,
            f(_empty_tensor_queue(), x),
            torch.compile(f, backend=backend)(_empty_tensor_queue(), x),
        )
        if not torch._dynamo.is_compiling() and backend == "eager":
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
def forward(self, L_x_ : torch.Tensor, L_tq_ : torch.ScriptObject):
    l_x_ = L_x_
    l_tq_ = L_tq_
    x1 = l_x_.view(-1)
    x2 = l_x_.permute(1, 0);  l_x_ = None
    call_torchbind = torch.ops.higher_order.call_torchbind(l_tq_, 'push', x1)
    call_torchbind_1 = torch.ops.higher_order.call_torchbind(l_tq_, 'push', x2)
    call_torchbind_2 = torch.ops.higher_order.call_torchbind(l_tq_, 'size')
    sub = x1 - 2;  x1 = None
    call_torchbind_3 = torch.ops.higher_order.call_torchbind(l_tq_, 'size');  l_tq_ = None
    add = x2 + 2;  x2 = None
    return (sub, add)""",
            )

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_tensor_op_in_tensor_flatten(self, backend):
        test_obj = torch.classes._TorchScriptTesting._FlattenWithTensorOp(
            torch.randn(3, 2)
        )

        class TestMod(torch.nn.Module):
            def forward(self, obj, x):
                return obj.get() + x + obj.get().size(0)

        mod = TestMod()

        x = torch.randn(3, 1)
        eager_out = mod(test_obj, x)
        compiled_out = torch.compile(mod, backend=backend, fullgraph=True)(test_obj, x)
        ep = torch.export.export_for_training(
            mod, (test_obj, x), strict=False
        ).run_decompositions({})
        self.assertExpectedInline(
            ep.graph_module.code.strip(),
            """\
def forward(self, token, obj, x):
    with_effects = torch.ops.higher_order.with_effects(token, torch.ops.higher_order.call_torchbind, obj, 'get');  token = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(getitem_1, x);  getitem_1 = x = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.higher_order.call_torchbind, obj, 'get');  getitem = obj = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    add_1 = torch.ops.aten.add.Tensor(add, 3);  add = None
    return (getitem_2, add_1)""",  # noqa: B950
        )
        self.assertEqual(eager_out, compiled_out)
        self.assertEqual(eager_out, ep.module()(test_obj, x))

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_error_on_non_fakified_method(self, backend):
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        def f(tq, x):
            x1 = x.view(-1)
            x2 = x.permute(1, 0)
            tq.push(x1)
            tq.push(x2)
            # though real tensor queue implemented a method clone_queue,
            # The fakified version doesn't.
            flat_obj = tq.clone_queue()
            return flat_obj

        x = torch.randn(2, 3)
        with self.assertRaisesRegex(
            RuntimeError, "FakeScriptObject doesn't define method"
        ):
            torch.compile(f, backend=backend)(_empty_tensor_queue(), x)

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_obj_as_hop_input(self, backend):
        def f(tq, x):
            def fn(tq, x):
                tq.push(x)
                return x.sin()

            return wrap(fn, tq, x)

        x = torch.randn(2, 3)
        _assertEqualScriptObject(
            self,
            f(_empty_tensor_queue(), x),
            torch.compile(f, backend=backend)(_empty_tensor_queue(), x),
        )

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_obj_closure(self, backend):
        def f(x):
            def inner_f(x):
                tq.push(x.sin())

            inner_f(x)
            return tq.pop(), tq

        opt_f = torch.compile(f, backend="eager")

        tq = _empty_tensor_queue()
        x = torch.randn(3, 2)
        _assertEqualScriptObject(self, f(x), opt_f(x))

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_global_obj(self, backend):
        global _TENSOR_QUEUE_GLOBAL_TEST
        _TENSOR_QUEUE_GLOBAL_TEST = _empty_tensor_queue()

        def f(x):
            _TENSOR_QUEUE_GLOBAL_TEST.push(x.sin())
            return _TENSOR_QUEUE_GLOBAL_TEST.pop(), _TENSOR_QUEUE_GLOBAL_TEST

        opt_f = torch.compile(f, backend=backend)
        x = torch.randn(3, 2)
        eager_ret = f(x)
        opt_ret = opt_f(x)
        _assertEqualScriptObject(self, eager_ret, opt_ret)

    def test_compile_obj_graph_breaks(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def f(tq, x):
            tq.push(x.sin())
            tq.push(x.sin())
            torch._dynamo.graph_break()
            tq.pop()
            torch._dynamo.graph_break()
            tq.push(x.cos() + tq.size())
            torch._dynamo.graph_break()
            tq.push(x.cos() - tq.size())
            return x, tq.pop(), tq

        opt_f = torch.compile(f, backend=cnt)
        x = torch.randn(3, 2)
        _assertEqualScriptObject(
            self, f(_empty_tensor_queue(), x), opt_f(_empty_tensor_queue(), x)
        )
        self.assertEqual(cnt.frame_count, 4)

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_obj_attributes(self, backend):
        if backend == "eager":
            backend = EagerAndRecordGraphs()

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tq = _empty_tensor_queue()

            def forward(self, x):
                self.tq.push(x)
                return self.tq.pop()

        x = torch.randn(2, 3)
        opt_f = torch.compile(Model(), backend=backend)
        _assertEqualScriptObject(self, Model()(x), opt_f(x))
        if backend == "eager":
            self.assertEqual(len(backend.graphs), 1)
            # lifted as input. In the future, we would want to cosolidate this
            # with non-strict behavior, where they're set as attributes.
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
    def forward(self, L_self_tq : torch.ScriptObject, L_x_ : torch.Tensor):
        l_self_tq = L_self_tq
        l_x_ = L_x_
        call_torchbind = torch.ops.higher_order.call_torchbind(l_self_tq, 'push', l_x_);  l_x_ = None
        call_torchbind_1 = torch.ops.higher_order.call_torchbind(l_self_tq, 'pop');  l_self_tq = None
        return (call_torchbind_1,)""",
            )

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_obj_torchbind_op(self, backend):
        def f(tq, x):
            torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
            torch.ops._TorchScriptTesting.queue_push(tq, x.cos() + 1)
            torch.ops._TorchScriptTesting.queue_pop(tq)
            torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
            return tq.pop(), tq.pop() + tq.size(), tq

        opt_f = torch.compile(f, backend=backend)
        x = torch.randn(2)
        _assertEqualScriptObject(
            self, f(_empty_tensor_queue(), x), opt_f(_empty_tensor_queue(), x)
        )

    @requires_gpu
    @parametrize("device", ["cpu", "cuda"])
    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_compile_obj_torchbind_op_with_autocast(self, backend, device):
        def f(tq, x):
            with torch.autocast(device_type=device):
                torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                torch.ops._TorchScriptTesting.queue_push(tq, x.cos() + 1)
                torch.ops._TorchScriptTesting.queue_pop(tq)
                torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
            return tq.pop(), tq.pop() + tq.size(), tq

        opt_f = torch.compile(f, backend=backend)
        x = torch.randn(2, device=device)
        _assertEqualScriptObject(
            self, f(_empty_tensor_queue(), x), opt_f(_empty_tensor_queue(), x)
        )

    @requires_gpu
    @parametrize("device", ["cpu", "cuda"])
    def test_export_obj_torchbind_op_with_autocast(self, device):
        class Mod(torch.nn.Module):
            def forward(self, x, tq):
                with torch.autocast(device_type=device):
                    torch.ops._TorchScriptTesting.queue_push(tq, x.cos())
                    torch.ops._TorchScriptTesting.queue_push(tq, x.cos() + 1)
                    torch.ops._TorchScriptTesting.queue_pop(tq)
                    torch.ops._TorchScriptTesting.queue_push(tq, x.sin())
                return tq.pop(), tq.pop() + tq.size(), tq

        x = torch.randn(2, device=device)
        args = (x,)
        mod = Mod()
        ep = torch.export.export(mod, (x, _empty_tensor_queue()))
        _assertEqualScriptObject(
            self, ep.module()(x, _empty_tensor_queue()), mod(x, _empty_tensor_queue())
        )


@skipIfTorchDynamo("torchbind not supported with dynamo yet")
class TestRegisterFakeClass(TestCase):
    def setUp(self):
        init_torchbind_implementations()

    def tearDown(self):
        torch._library.fake_class_registry.global_fake_class_registry.clear()

    def test_register_fake_class_no_torch_bind_class(self):
        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate class"):

            @torch._library.register_fake_class("_TorchScriptTesting::NOT_A_VALID_NAME")
            class Invalid:
                pass

    def test_register_fake_class_no_from_real(self):
        with self.assertRaisesRegex(
            RuntimeError, "define a classmethod __obj_unflatten__"
        ):

            @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
            class InvalidFakeFoo:
                def __init__(self) -> None:
                    pass

    def test_register_fake_class_from_real_not_classmethod(self):
        with self.assertRaisesRegex(RuntimeError, "is not a classmethod"):

            @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
            class FakeFoo:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                def __obj_unflatten__(cls, flattend_foo):  # noqa: B902
                    return cls(**dict(flattend_foo))

    def test_register_fake_class_valid(self):
        class FakeFoo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            @classmethod
            def __obj_unflatten__(cls, flattend_foo):
                return cls(**dict(flattend_foo))

        torch._library.register_fake_class("_TorchScriptTesting::_Foo", FakeFoo)


instantiate_parametrized_tests(TestExportTorchbind)
instantiate_parametrized_tests(TestCompileTorchbind)

if __name__ == "__main__":
    run_tests()
