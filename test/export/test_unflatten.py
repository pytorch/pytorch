# Owner(s): ["oncall: export"]
# flake8: noqa
import copy
import unittest
from re import escape
from typing import Any, List, Optional

import torch
import torch._dynamo as torchdynamo
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch.export import export, FlatArgsAdapter, unflatten
from torch.export.unflatten import _disable_interpreter
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.torchbind_impls import init_torchbind_implementations
from torch.utils._pytree import TreeSpec


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestUnflatten(TestCase):
    def compare_outputs(self, eager, unflattened, args):
        orig_output = eager(*args)
        unflattened_output = unflattened(*args)
        self.assertTrue(torch.allclose(orig_output, unflattened_output))

    def test_unflatten_nested(self):
        class NestedChild(torch.nn.Module):
            def forward(self, x):
                return x / x

        class Child1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3))

            def forward(self, x):
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        export_module = export(orig_eager, (torch.rand(2, 3),), {}, strict=True)
        unflattened = unflatten(export_module)

        inputs = (torch.rand(2, 3),)

        # Compare the root modules and all submodules
        self.compare_outputs(orig_eager, unflattened, inputs)
        self.compare_outputs(orig_eager.foo, unflattened.foo, inputs)
        self.compare_outputs(orig_eager.bar, unflattened.bar, inputs)
        self.compare_outputs(orig_eager.foo.nested, unflattened.foo.nested, inputs)

        # Check state dicts are equal
        orig_state_dict = orig_eager.state_dict()
        exported_state_dict = unflattened.state_dict()
        for name, value in orig_state_dict.items():
            self.assertTrue(torch.allclose(value, exported_state_dict[name]))

    def test_unflatten_buffer_mutation(self):
        class Child(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3))

            def forward(self, x):
                self.child2buffer.add_(x)
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Child()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = self.foo(x)
                return x * self.rootparam

        eager_module = MyModule()
        export_module = export(eager_module, (torch.rand(2, 3),), {}, strict=True)
        unflattened_module = unflatten(export_module)

        # Buffer should look the same before and after one run
        eager_buffer = eager_module.foo.child2buffer
        unflattened_buffer = unflattened_module.foo.child2buffer
        self.assertTrue(torch.allclose(eager_buffer, unflattened_buffer))

        inputs = (torch.rand(2, 3),)
        eager_module(*inputs)
        unflattened_module(*inputs)
        self.assertTrue(torch.allclose(eager_buffer, unflattened_buffer))

    def test_unflatten_nested_access(self):
        class Child(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3))

            def forward(self, x):
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Child()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = x + self.foo.child2buffer
                x = self.foo(x)
                return x

        eager_module = MyModule()
        export_module = export(eager_module, (torch.rand(2, 3),), {}, strict=True)
        unflattened_module = unflatten(export_module)

        inputs = (torch.rand(2, 3),)
        self.compare_outputs(eager_module, unflattened_module, inputs)

    def test_unflatten_shared_submodule(self):
        class Shared(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layernorm = torch.nn.LayerNorm(10)
                self.sub_net = torch.nn.Sequential(
                    layernorm,
                    torch.nn.ReLU(),
                    layernorm,
                    torch.nn.ReLU(),
                )

            def forward(self, x):
                return self.sub_net(x)

        eager_module = Shared()
        inps = (torch.rand(10),)
        export_module = export(eager_module, inps, {}, strict=True)
        unflattened_module = unflatten(export_module)
        self.compare_outputs(eager_module, unflattened_module, inps)
        self.assertTrue(hasattr(unflattened_module, "sub_net"))
        for i in range(len(eager_module.sub_net)):
            self.assertTrue(hasattr(unflattened_module.sub_net, str(i)))
        self.assertEqual(
            id(getattr(unflattened_module.sub_net, "0")),
            id(getattr(unflattened_module.sub_net, "2")),
        )

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    @skipIfTorchDynamo("Non strict mode is not meant to run with dynamo")
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
        for strict in [True, False]:
            export_module = export(
                orig_eager,
                inps,
                {},
                preserve_module_call_signature=("foo.nested",),
                strict=strict,
            )
            unflattened = unflatten(export_module)
            self.compare_outputs(export_module.module(), unflattened, inps)
            unflattened.foo.nested = NestedChild()
            self.compare_outputs(export_module.module(), unflattened, inps)

            # Test tree spec mismatched input
            orig_outs = export_module.module()(*inps)
            new_inps = *inps, torch.rand(2, 3)
            with self.assertRaisesRegex(
                TypeError,
                "There is no flat args adapter sepcified. Are you sure you are calling this with the right arguments?",
            ):
                unflattened(new_inps)

            # With flat args adapter
            class KeepTwoFlatArgsAdapter(FlatArgsAdapter):
                def adapt(
                    self,
                    target_spec: TreeSpec,
                    input_spec: TreeSpec,
                    input_args: List[Any],
                    metadata: dict[str, Any],
                    obj: Optional[Any] = None,
                ) -> List[Any]:
                    while len(input_args) > 2:
                        input_args.pop(-1)
                    return input_args

            unflattened = unflatten(export_module, KeepTwoFlatArgsAdapter())
            new_outs = unflattened(*new_inps)
            self.assertTrue(torch.allclose(orig_outs, new_outs))

    def test_unflatten_param_list_dict(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param_list = torch.nn.ParameterList()
                self.param_dict = torch.nn.ParameterDict()
                for i in range(2):
                    self.param_list.append(torch.nn.Parameter(torch.randn((2, 3))))
                    self.param_dict[f"key_{i}"] = torch.nn.Parameter(
                        torch.randn((2, 3))
                    )

            def forward(self, x):
                for i in range(2):
                    x = x + self.param_list[i]
                    x = x + self.param_dict[f"key_{i}"]
                return x

        export_module = torch.export.export(Mod(), (torch.randn((2, 3)),), strict=True)
        unflattened = unflatten(export_module)

        self.compare_outputs(
            export_module.module(), unflattened, (torch.randn((2, 3)),)
        )

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
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
            strict=False,
        )
        ep.graph.eliminate_dead_code()
        unflattened = unflatten(ep)
        self.compare_outputs(ep.module(), unflattened, (torch.randn(2), torch.randn(5)))

    def test_unflatten_wrong_input(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param_list = torch.nn.ParameterList()
                self.param_dict = torch.nn.ParameterDict()
                for i in range(2):
                    self.param_list.append(torch.nn.Parameter(torch.randn((2, 3))))
                    self.param_dict[f"key_{i}"] = torch.nn.Parameter(
                        torch.randn((2, 3))
                    )

            def forward(self, x):
                a = x.sum()
                for i in range(2):
                    a = a + self.param_list[i].sum()
                    a = a + self.param_dict[f"key_{i}"].sum()
                return a

        export_module = torch.export.export(Mod(), (torch.randn((2, 3)),), strict=True)
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[0].shape[0] to be equal to 2, but got 6"),
        ):
            export_module.module()(torch.randn(6, 6))

        unflattened = unflatten(export_module)
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[0].shape[0] to be equal to 2, but got 6"),
        ):
            unflattened(torch.randn(6, 6))

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    def test_unflatten_with_inplace_compile(self):
        class NestedChild(torch.nn.Module):
            def forward(self, x):
                return x / x

        class Child1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3))

            def forward(self, x):
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        export_module = torch.export.export(
            orig_eager, (torch.rand(2, 3),), {}, strict=True
        )
        unflattened = unflatten(export_module)

        # in-place compilation should work. Pass fullgraph to ensure no graph breaks.
        from torch._dynamo.backends.debugging import ExplainWithBackend

        eb = ExplainWithBackend("inductor")
        unflattened.foo.compile(backend=eb, fullgraph=True)
        inputs = (torch.randn(2, 3),)
        self.compare_outputs(orig_eager, unflattened, inputs)
        self.assertEqual(len(eb.graphs), 1)

        unflattened.compile()
        self.compare_outputs(orig_eager, unflattened, inputs)

    def test_fx_trace(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                x = x[0] + x[1]
                x = x + y["foo"]
                return x

        orig_eager = MyModule()
        inputs = ((torch.rand(2, 3), torch.rand(2, 3)), {"foo": torch.rand(2, 3)})
        export_module = export(orig_eager, inputs, {}, strict=True)

        unflattened = unflatten(export_module)
        torch.fx.symbolic_trace(
            unflattened, concrete_args=(torch.fx.PH, torch.fx.PH, torch.fx.PH)
        )

    def test_double_nested_submodule(self):
        class SubSubMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x * x

        class SubMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.subsubmod = SubSubMod()

            def forward(self, x):
                return x - x

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.submod = SubMod()

            def forward(self, x):
                return x + self.submod.subsubmod(x)

        orig_eager = MyModule()
        export_module = torch.export.export(
            orig_eager, (torch.rand(2, 3),), {}, strict=True
        )
        unflattened = unflatten(export_module)

        inputs = (torch.rand(2, 3),)
        self.compare_outputs(orig_eager, unflattened, inputs)

    def test_unflatten_container_type(self):
        class Leaf(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.leaf = Leaf()
                self.buffer = torch.nn.Buffer(torch.randn(4, 4))

            def forward(self, x, z):
                return self.buffer.sum() + self.leaf(x).sum() + z[0].sum() + z[1].sum()

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bar = Bar()

            def forward(self, x, z):
                y = self.bar.buffer + x + z[0] + z[1]
                return self.bar(x, z) + y.sum()

        inp = (torch.randn(4, 4), [torch.randn(4, 4), torch.randn(4, 4)])
        mod = Foo()

        ep_strict = torch.export.export(mod, inp, strict=True)  # noqa: F841
        ep_non_strict = torch.export.export(mod, inp, strict=False)

        gm_unflat_non_strict = unflatten(ep_non_strict)
        ep = torch.export.export(gm_unflat_non_strict, inp, strict=False)
        self.assertTrue(torch.allclose(ep.module()(*inp), mod(*inp)))

    def test_unflattened_module_nodes_has_meta_val(self):
        class SubMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x + x, x * x

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.submod = SubMod()

            def forward(self, x):
                return x + sum(self.submod(x))

        orig_eager = MyModule()
        export_module = torch.export.export(
            orig_eager, (torch.rand(2, 3),), {}, strict=True
        )
        unflattened = unflatten(export_module)

        inputs = (torch.rand(2, 3),)
        self.compare_outputs(orig_eager, unflattened, inputs)

        def check_meta(gm):
            for n in gm.graph.nodes:
                if n.op == "output":
                    continue
                self.assertTrue(n.meta.get("val") is not None)

        for m in unflattened.modules():
            check_meta(m)

    def test_unflatten_requires_grad_param(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(3, 3), requires_grad=False)

            def forward(self, x):
                return self.p + x

        with torch.device("meta"):
            mod = M()

        inputs = (torch.randn(3, 3, device="meta"),)
        ep = export(mod, inputs, strict=True)
        unflattened = unflatten(ep)
        self.assertTrue(unflattened.state_dict()["p"].requires_grad is False)
        self.assertTrue(unflattened.p.requires_grad is False)

    def test_placeholder_and_get_attr_ordering_after_unflattened(self):
        class TransposeModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            def forward(self, x):
                x = self.conv(x)
                return x.transpose(0, 1)

        x = torch.randn(32, 3, 64, 64)
        exported_program = export(TransposeModule(), args=(x,), strict=True)
        unflattened_module = unflatten(exported_program)

        # Check the inputs of the created call_module node are in order
        call_module_input_order = []
        for node in unflattened_module.graph.nodes:
            if node.op == "call_module":
                transpose_module = unflattened_module.get_submodule(node.target)
                for sub_node in transpose_module.graph.nodes:
                    if sub_node.op == "placeholder" or sub_node.op == "get_attr":
                        call_module_input_order.append(sub_node.op)
        self.assertEqual(
            call_module_input_order, ["placeholder", "get_attr", "get_attr"]
        )

    def test_unflatten_constant_tensor(self):
        class SubMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.initializer = 0.1

            def forward(self, x):
                return x + torch.tensor(self.initializer)

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.submod = SubMod()

            def forward(self, x):
                return x + self.submod(x)

        export_module = torch.export.export(Mod(), (torch.randn((2, 3)),), strict=True)
        unflattened = unflatten(export_module)

        self.compare_outputs(
            export_module.module(), unflattened, (torch.randn((2, 3)),)
        )

    @skipIfTorchDynamo("custom objects not supported in dynamo yet")
    def test_unflatten_constant_obj(self):
        init_torchbind_implementations()

        @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
        class FakeFoo:  # noqa: F841
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

            @classmethod
            def __obj_unflatten__(cls, flat_ctx):
                return cls(**dict(flat_ctx))

            def add_tensor(self, z):
                return (self.x + self.y) * z

        class SubMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return x + self.attr.add_tensor(x)

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.submod = SubMod()

            def forward(self, x):
                return x + self.submod(x)

        with enable_torchbind_tracing():
            export_module = torch.export.export(
                Mod(), (torch.randn((2, 3)),), strict=False
            )
        unflattened = unflatten(export_module)

        self.compare_outputs(
            export_module.module(), unflattened, (torch.randn((2, 3)),)
        )

    # skip connection is not supported yet
    @unittest.expectedFailure
    def test_unflatten_skipped_call_module(self):
        class C(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return a.d(x.cos())

        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = C()

            def forward(self, x):
                return self.c(x) + x

        class D(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.sin()

        class A(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = B()
                self.d = D()

            def forward(self, x):
                return self.b(x)

        a = A()

        # The call chain looks like this:
        # A -> B -> C -> A.d
        ep = torch.export.export(a, (torch.randn(3),), strict=False)
        unflatten(ep)

    def test_nested_leaf_non_strict(self):
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
            strict=False,
            preserve_module_call_signature=("nested",),
        )

        torch.export.unflatten(ep)

    def test_unflatten_submodule_ordering(self):
        class Module2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.rand(3, 4))
                self.register_parameter("param", torch.nn.Parameter(torch.rand(3, 4)))

            def forward(self, x):
                return x + self.buffer + self.param

        class Module1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.rand(3, 4))
                self.register_parameter("param", torch.nn.Parameter(torch.rand(3, 4)))

            def forward(self, x):
                return x + self.buffer + self.param

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod2 = Module2()
                self.mod3 = self.mod2
                self.mod1 = Module1()

            def forward(self, x):
                return self.mod3(self.mod2(self.mod1(x)))

        mod = Module()

        ep = torch.export.export(mod, (torch.randn(3, 4),), strict=True)

        unflattened = torch.export.unflatten(ep)
        fqn_list = [x for x, _ in unflattened.named_modules(remove_duplicate=False)]
        self.assertEqual(len(fqn_list), 4)
        self.assertEqual(
            [x for x, _ in mod.named_modules(remove_duplicate=False)],
            fqn_list,
        )

    def test_duplicate_placeholder(self):
        N, C, H, W = 1, 2, 2, 3

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layer = torch.nn.LayerNorm([C, H, W])
                self.norms = torch.nn.ModuleList(
                    [
                        layer,  # reuse layer norm
                        layer,
                        layer,
                    ]
                )

            def forward(self, input_):
                for i in range(len(self.norms)):
                    output = self.norms[i](input_)
                    input_ = output
                return output

        mod = MyModule()
        input_ = torch.randn(N, C, H, W)

        ep_strict = export(copy.deepcopy(mod), (input_,), strict=True)
        umod = unflatten(ep_strict)
        self.assertTrue(torch.allclose(umod(input_), mod(input_)))

        ep_non_strict = export(copy.deepcopy(mod), (input_,), strict=False)
        umod = unflatten(ep_non_strict)
        self.assertTrue(torch.allclose(umod(input_), mod(input_)))

    def test_simple_alias(self):
        # handle weight sharing, check tensor ids after unflattening
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # alias param
                self.bias = torch.nn.Parameter(torch.randn(4))
                self.m = torch.nn.Linear(4, 4)
                self.m.bias = self.bias

            def forward(self, x):
                return self.m(x) + self.bias

        m = Foo()
        inps = (torch.randn(4, 4),)
        ep = export(m, inps, strict=True)
        unep = unflatten(ep)
        self.assertTrue(id(unep.m.bias) == id(unep.bias))

        # handle aliasing where one alias is unused
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.randn(4))
                self.m = torch.nn.Linear(4, 4)
                self.m.bias = (
                    self.bias
                )  # self.bias is unused, aliasing should be handled

            def forward(self, x):
                return self.m(x)

        m = Foo()
        inps = (torch.randn(4, 4),)
        ep = export(m, inps, strict=True)
        unep = unflatten(ep)
        self.assertTrue(torch.allclose(unep(*inps), m(*inps)))

    def test_attr_as_submod_input(self):
        class layer(torch.nn.Module):
            def forward(self, x, const) -> torch.Tensor:
                return x + const

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.const = torch.nn.Buffer(torch.ones(4, 8))
                self.layers = torch.nn.ModuleList([layer() for _ in range(2)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.layers:
                    x = layer(x, self.const)
                return x

        mod = M()
        x = torch.randn(4, 8)
        ep = export(mod, (x,), strict=True)
        unflattened = unflatten(ep)
        torch.testing.assert_close(unflattened(x), mod(x))

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
            strict=False,
            preserve_module_call_signature=("m1",),
        )
        unflat = unflatten(ep)
        unflat(*inputs)

        fn_count_sym_size = lambda graph: [node.target for node in graph.nodes].count(
            torch.ops.aten.sym_size.int
        )
        self.assertEqual(fn_count_sym_size(unflat.graph), 1)
        self.assertEqual(fn_count_sym_size(unflat.m1.graph), 1)
        self.assertEqual(fn_count_sym_size(unflat.m2.graph), 0)

    def test_unflatten_eager(self):
        class NestedChild(torch.nn.Module):
            def forward(self, x):
                return x / x

        class Child1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3))

            def forward(self, x):
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        orig_eager = MyModule()
        export_module = export(orig_eager, (torch.rand(2, 3),), {}, strict=True)
        with _disable_interpreter():
            unflattened = unflatten(export_module)

        self.assertEqual(unflattened._run_with_interpreter, False)
        self.assertEqual(unflattened.foo._run_with_interpreter, False)

        inputs = (torch.rand(2, 3),)

        # Compare the root modules and all submodules
        self.compare_outputs(orig_eager, unflattened, inputs)
        self.compare_outputs(orig_eager.foo, unflattened.foo, inputs)
        self.compare_outputs(orig_eager.bar, unflattened.bar, inputs)
        self.compare_outputs(orig_eager.foo.nested, unflattened.foo.nested, inputs)

        # Check state dicts are equal
        orig_state_dict = orig_eager.state_dict()
        exported_state_dict = unflattened.state_dict()
        for name, value in orig_state_dict.items():
            self.assertTrue(torch.allclose(value, exported_state_dict[name]))

        # Check composability with symbolic trace, as torchrec ddp uses symbolic
        # tracer
        symbolic_traced = torch.fx.symbolic_trace(unflattened, concrete_args=inputs)
        self.assertTrue(torch.allclose(orig_eager(*inputs), symbolic_traced(*inputs)))

        # torch.compile submodule
        unflattened.foo = torch.compile(unflattened.foo, fullgraph=True)
        self.compare_outputs(orig_eager, unflattened, inputs)


if __name__ == "__main__":
    run_tests()
