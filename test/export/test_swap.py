# Owner(s): ["oncall: export"]
# flake8: noqa
import copy
import dataclasses
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from re import escape
from typing import Any, List

from parameterized import parameterized_class

import torch
import torch._dynamo as torchdynamo
from functorch.experimental.control_flow import cond, map
from torch import Tensor
from torch._export.utils import (
    get_buffer,
    get_param,
    is_buffer,
    is_param,
    register_dataclass_as_pytree_node,
)
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch.export import Constraint, Dim, export, FlatArgsAdapter, unflatten
from torch.export._swap import _swap_modules
from torch.export._trace import DEFAULT_EXPORT_DYNAMO_CONFIG
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
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
from torch.testing._internal.torchbind_impls import init_torchbind_implementations
from torch.utils._pytree import (
    LeafSpec,
    tree_flatten,
    tree_unflatten,
    TreeSpec,
    treespec_dumps,
    treespec_loads,
)


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
@parameterized_class(
    [
        {"strict": False},
        {"strict": True},
    ],
    class_name_func=lambda cls, _, params: f"{cls.__name__}_{'strict' if params['strict'] else 'nonstrict'}",
)
class TestSwap(TestCase):
    def test_unflatten_preserve_signature(self):
        class NestedChild(torch.nn.Module):
            def forward(self, zx, y):
                return {"x": y["key"] + zx[1], "w": y["key"] * zx[1]}

        class Child1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = NestedChild()

            def forward(self, x, y):
                z = torch.ones_like(x)
                xw = self.nested((z, x), y={"key": y})
                return xw["w"] + z - xw["x"]

        class Child2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x - 1

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()

            def forward(self, x, y):
                x = self.foo(x, y)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        inps = torch.rand(2, 3), torch.rand(2, 3)

        ep = export(
            orig_eager,
            inps,
            {},
            preserve_module_call_signature=("foo.nested", "bar"),
            strict=self.strict,
        )

        swapped_gm = _swap_modules(
            ep,
            {"foo.nested": NestedChild(), "bar": Child2()},
        )

        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))

    def test_unflatten_preserve_with_unused_input(self):
        class M1(torch.nn.Module):
            def forward(self, x, a, b):
                return x + a, b

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = M1()

            def forward(self, x, y):
                a, b = torch.topk(y, 2)
                return self.m1(x, a, b)[0]

        ep = torch.export.export(
            M(),
            (torch.randn(2), torch.randn(5)),
            preserve_module_call_signature=("m1",),
            strict=self.strict,
        )

        swapped_gm = _swap_modules(
            ep,
            {"m1": M1()},
        )

        inps = (torch.randn(2), torch.randn(5))
        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))

    def test_nested_leaf(self):
        class Leaf(torch.nn.Module):
            def forward(self, x):
                return x + 1

        class Nested(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.leaf = Leaf()

            def forward(self, x):
                return self.leaf(x) + 2

        class TopLevel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = Nested()

            def forward(self, x):
                return self.nested(x) + 3

        ep = torch.export.export(
            TopLevel(),
            (torch.randn(3),),
            strict=self.strict,
            preserve_module_call_signature=("nested",),
        )

        swapped_gm = _swap_modules(
            ep,
            {"nested": Nested()},
        )

        inps = (torch.randn(3),)
        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))

    def test_dedup_sym_size(self):
        # Here, sym_size & floor div are used in 3 subgraphs (top-level, m1, m2),
        # but only one copy of sym_size is created in the initial export graph.
        # For m1, sym_size & floordiv should be copied as recompute since we preserve the call signature,
        # but for m2 floordiv should be passed in as a placeholder.
        # Test that this is preserved, and the unflattened module runs correctly.
        class M1(torch.nn.Module):
            def forward(self, x, y):
                d = x.size(0) // 2
                return y[:d]

        class M2(torch.nn.Module):
            def forward(self, x, y):
                d = x.size(0) // 2
                return y[:d]

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = M1()
                self.m2 = M2()

            def forward(self, x, y):
                d = x.size(0) // 2
                m1_res = self.m1(x, y)
                m2_res = self.m2(x, y)
                return y[d:] + m1_res + m2_res

        inputs = (torch.ones(10), torch.ones(10))
        d_ = torch.export.Dim("foo", max=2048)
        d = 2 * d_
        ep = torch.export.export(
            M(),
            inputs,
            dynamic_shapes=((d,), (d,)),
            strict=self.strict,
            preserve_module_call_signature=("m1",),
        )

        swapped_gm = _swap_modules(
            ep,
            {"m1": M1()},
        )

        inps = (torch.randn(10), torch.randn(10))
        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))

        inps = (torch.randn(20), torch.randn(20))
        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))

    def test_remove_duplicate_pytree_simple(self):
        class Child1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                z = torch.ones_like(x)
                w = y + z[1]
                x = y * z[1]
                return {"res1": x + y, "res2": x * y}

        class Child2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x["res2"] + x["res1"] - 1

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()

            def forward(self, x, y):
                x = self.foo(x, y)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        inps = torch.rand(2, 3), torch.rand(2, 3)

        ep = export(
            orig_eager,
            inps,
            {},
            preserve_module_call_signature=("foo", "bar"),
            strict=self.strict,
        )

        swapped_gm = _swap_modules(
            ep,
            {"foo": Child1(), "bar": Child2()},
        )

        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))
        self.assertExpectedInline(
            swapped_gm.code.strip(),
            """\
def forward(self, x, y):
    x_1 = x
    y_1 = y
    _spec_0 = self._spec_0
    _spec_1 = self._spec_1
    _spec_4 = self._spec_4
    tree_flatten = torch.utils._pytree.tree_flatten((x_1, y_1));  x_1 = y_1 = None
    getitem = tree_flatten[0];  tree_flatten = None
    x = getitem[0]
    y = getitem[1];  getitem = None
    tree_unflatten_1 = torch.utils._pytree.tree_unflatten([x, y], _spec_1);  x = y = _spec_1 = None
    getitem_1 = tree_unflatten_1[0];  tree_unflatten_1 = None
    getitem_2 = getitem_1[0]
    getitem_3 = getitem_1[1];  getitem_1 = None
    foo = self.foo(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    bar = self.bar(foo);  foo = None
    tree_flatten_spec_1 = torch.fx._pytree.tree_flatten_spec(bar, _spec_4);  bar = _spec_4 = None
    getitem_10 = tree_flatten_spec_1[0];  tree_flatten_spec_1 = None
    tree_unflatten = torch.utils._pytree.tree_unflatten((getitem_10,), _spec_0);  getitem_10 = _spec_0 = None
    return tree_unflatten""",
        )

    @unittest.expectedFailure
    def test_remove_duplicate_pytree_different_order(self):
        """
        This is not supported yet because module `foo`s outputs are not all
        directly used in as inputs to `bar` in the same order as outputted from
        `foo`. To support this, we would have to do some sort of ordering.
        """

        class Child1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                return {"res1": x + y}, {"res2": x * y, "res3": x * x}

        class Child2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, y, x):
                y = y["res2"] * y["res3"]
                x = x["res1"] + x["res1"]
                return y - x

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()

            def forward(self, x, y):
                x, y = self.foo(x, y)
                x = self.bar(y, x)
                return x

        orig_eager = MyModule()
        inps = torch.rand(2, 3), torch.rand(2, 3)

        ep = export(
            orig_eager,
            inps,
            {},
            preserve_module_call_signature=("foo", "bar"),
            strict=self.strict,
        )

        swapped_gm = _swap_modules(
            ep,
            {"foo": Child1(), "bar": Child2()},
        )

        self.assertTrue(torch.allclose(ep.module()(*inps), swapped_gm(*inps)))
        self.assertExpectedInline(
            swapped_gm.code.strip(),
            """\
def forward(self, x, y):
    x, y, = fx_pytree.tree_flatten_spec(([x, y], {}), self._in_spec)
    _spec_0 = self._spec_0
    _spec_3 = self._spec_3
    tree_unflatten = torch.utils._pytree.tree_unflatten([x, y], _spec_0);  x = y = _spec_0 = None
    getitem = tree_unflatten[0];  tree_unflatten = None
    getitem_1 = getitem[0]
    getitem_2 = getitem[1];  getitem = None
    foo = self.foo(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    getitem_3 = foo[0]
    getitem_4 = foo[1];
    bar = self.bar(getitem_4, getitem_3);  foo = None
    tree_flatten_spec_1 = torch.fx._pytree.tree_flatten_spec(bar, _spec_3);  bar = _spec_3 = None
    getitem_9 = tree_flatten_spec_1[0];  tree_flatten_spec_1 = None
    return pytree.tree_unflatten((getitem_9,), self._out_spec)""",
        )

    def test_custom_input_args(self):
        @dataclass
        class CustomInput:
            a: Tensor
            b: Tensor

        register_dataclass_as_pytree_node(
            CustomInput,
            serialized_type_name="test_swap.test_custom_input.CustomInput",
        )

        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs.a, inputs.b)

        ep = export(
            Foo(),
            (CustomInput(torch.randn(2, 3), torch.randn(3, 2)),),
            strict=self.strict,
        )
        swapped = _swap_modules(ep, {})
        inp = (CustomInput(torch.randn(2, 3), torch.randn(3, 2)),)
        res1 = torch.fx.Interpreter(swapped).run(*inp)
        res2 = swapped(*inp)
        self.assertTrue(torch.allclose(res1, res2))

    def test_custom_input_kwargs(self):
        @dataclass
        class CustomInput:
            a: Tensor
            b: Tensor

        register_dataclass_as_pytree_node(
            CustomInput,
            serialized_type_name="test_swap.test_custom_input.CustomInput",
        )

        class Foo(torch.nn.Module):
            def forward(self, x, *, inputs):
                return x + torch.matmul(inputs.a, inputs.b)

        ep = export(
            Foo(),
            (torch.randn(2, 2),),
            {"inputs": CustomInput(torch.randn(2, 3), torch.randn(3, 2))},
            strict=self.strict,
        )
        swapped = _swap_modules(ep, {})
        inp_args = (torch.randn(2, 2),)
        inp_kwargs = {"inputs": CustomInput(torch.randn(2, 3), torch.randn(3, 2))}
        res1 = torch.fx.Interpreter(swapped).run(*(*inp_args, *inp_kwargs.values()))
        res2 = swapped(*inp_args, **inp_kwargs)
        self.assertTrue(torch.allclose(res1, res2))

    def test_custom_output(self):
        @dataclass
        class CustomOutput:
            a: Tensor
            b: Tensor

        register_dataclass_as_pytree_node(
            CustomOutput,
            serialized_type_name="test_swap.test_custom_input.CustomInput",
        )

        class Foo(torch.nn.Module):
            def forward(self, a, b):
                return (CustomOutput(a * a, b * b), CustomOutput(a * b.T, a + b.T))

        ep = export(Foo(), (torch.randn(2, 3), torch.randn(3, 2)))
        swapped = _swap_modules(ep, {})
        inp = (torch.randn(2, 3), torch.randn(3, 2))
        res1 = torch.fx.Interpreter(swapped).run(*inp)
        res2 = swapped(*inp)
        self.assertTrue(torch.allclose(res1[0].a, res2[0].a))
        self.assertTrue(torch.allclose(res1[0].b, res2[0].b))
        self.assertTrue(torch.allclose(res1[1].a, res2[1].a))
        self.assertTrue(torch.allclose(res1[1].b, res2[1].b))


if __name__ == "__main__":
    run_tests()
