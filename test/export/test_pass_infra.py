# Owner(s): ["oncall: export"]
import copy
import unittest

import torch
from functorch.experimental import control_flow
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch.export import export, unflatten
from torch.fx.passes.infra.pass_base import PassResult
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)


_CANONICALIZE_VALUES = (False, True)


@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPassInfra(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_export_pass_base(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x):
                y = torch.cat([x, x])
                return torch.ops.aten.tensor_split.sections(y, 2)

        f = Foo()

        class NullPass(_ExportPassBaseDeprecatedDoNotUse):
            pass

        ep = export(f, (torch.ones(3, 2),), strict=True)
        old_nodes = ep.graph.nodes

        ep = ep._transform_do_not_use(NullPass())
        new_nodes = ep.graph.nodes

        for node in new_nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(hasattr(node, "stack_trace"))
            self.assertIsNotNone(node.stack_trace)

        self.assertEqual(len(new_nodes), len(old_nodes))
        for new_node, old_node in zip(new_nodes, old_nodes):
            self.assertEqual(new_node.op, old_node.op)
            self.assertEqual(new_node.target, old_node.target)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported")
    def test_cond(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, pred, x, y):
                def true_fn(x, y):
                    b = x.item()
                    torch._check(b >= 2)
                    torch._check(b <= 5)
                    return x - y

                def false_fn(x, y):
                    c = y.item()
                    torch._check(c >= 2)
                    torch._check(c <= 5)
                    return x + y

                ret = control_flow.cond(pred, true_fn, false_fn, [x, y])
                return ret

        x = torch.tensor([2])
        y = torch.tensor([5])
        mod = M()
        _ = export(mod, (torch.tensor(True), x, y), strict=True)._transform_do_not_use(
            _ExportPassBaseDeprecatedDoNotUse()
        )

    @parametrize("canonicalize", _CANONICALIZE_VALUES)
    def test_node_name_stability(self, canonicalize) -> None:
        # Tests that graph nodes stay the same for nodes that are not touched
        # during transformation
        class CustomModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                # Define a parameter
                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                # Define two buffers
                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)

                return output

        inps = (torch.rand(1), torch.rand(1))
        m = CustomModule()

        with torch._dynamo.config.patch(
            canonicalize_output_graph_node_order=canonicalize
        ):
            ep_before = export(m, inps, strict=True)

        # No op transformation that doesn't perform any meaningful changes to node
        ep_after = ep_before._transform_do_not_use(_ExportPassBaseDeprecatedDoNotUse())

        self.assertEqual(
            [node.name for node in ep_before.graph.nodes],
            [node.name for node in ep_after.graph.nodes],
        )

    @parametrize("canonicalize", _CANONICALIZE_VALUES)
    def test_preserved_module_call_signature_after_noop_transform(
        self, canonicalize
    ) -> None:
        class Child(torch.nn.Module):
            def forward(self, input):
                return input + 1

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = Child()

            def forward(self, input):
                return self.child(input) * 2

        model = Model()
        inputs = (torch.ones(2),)
        with torch._dynamo.config.patch(
            canonicalize_output_graph_node_order=canonicalize
        ):
            ep_before = export(
                model,
                inputs,
                preserve_module_call_signature=("child",),
            )
        ep_after = ep_before._transform_do_not_use(_ExportPassBaseDeprecatedDoNotUse())

        self.assertEqual(ep_after.module_call_graph, ep_before.module_call_graph)
        self.assertEqual(ep_after.module()(*inputs), model(*inputs))

    @parametrize("canonicalize", _CANONICALIZE_VALUES)
    def test_preserved_name_uses_returned_node(self, canonicalize) -> None:
        class Child(torch.nn.Module):
            def forward(self, x):
                return torch.cat([x, x])

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = Child()

            def forward(self, x):
                return self.child(x) * 2

        class AddAuxiliaryCat(_ExportPassBaseDeprecatedDoNotUse):
            def call_operator(self, op, args, kwargs, meta):
                if op is torch.ops.aten.cat.default:
                    auxiliary = super().call_operator(op, ([args[0][0]],), kwargs, meta)
                    auxiliary.node.meta["is_auxiliary"] = True
                result = super().call_operator(op, args, kwargs, meta)
                if op is torch.ops.aten.cat.default:
                    result.node.meta["is_returned"] = True
                return result

        model = Model()
        inputs = (torch.tensor([2.0, 3.0]),)
        with torch._dynamo.config.patch(
            canonicalize_output_graph_node_order=canonicalize
        ):
            ep_before = export(
                model,
                inputs,
                preserve_module_call_signature=("child",),
            )
        ep_after = ep_before._transform_do_not_use(AddAuxiliaryCat())

        child_signature = next(
            entry.signature
            for entry in ep_after.module_call_graph
            if entry.fqn == "child"
        )
        if child_signature is None:
            raise AssertionError("child signature must not be None")
        returned = next(
            node for node in ep_after.graph.nodes if node.meta.get("is_returned")
        )
        auxiliary = next(
            node for node in ep_after.graph.nodes if node.meta.get("is_auxiliary")
        )
        self.assertEqual(returned.name, child_signature.outputs[0].name)
        self.assertNotEqual(auxiliary.name, returned.name)

        outlined = unflatten(ep_after)
        self.assertEqual(outlined(*inputs), model(*inputs))
        self.assertEqual(outlined.child(*inputs), model.child(*inputs))

    def test_intentional_node_rename_is_preserved(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        class RenameAdd(_ExportPassBaseDeprecatedDoNotUse):
            def call_operator(self, op, args, kwargs, meta):
                result = super().call_operator(op, args, kwargs, meta)
                if op is torch.ops.aten.add.Tensor:
                    result.node.name = "intentional_name"
                return result

        ep = export(M(), (torch.ones(2),))
        result = RenameAdd().call(ep.graph_module)
        add = next(
            node
            for node in result.graph_module.graph.nodes
            if node.target is torch.ops.aten.add.Tensor
        )
        self.assertEqual(add.name, "intentional_name")
        graph = result.graph_module.graph
        placeholder = next(node for node in graph.nodes if node.op == "placeholder")
        with graph.inserting_before(graph.output_node()):
            new_add = graph.call_function(
                torch.ops.aten.add.Tensor,
                (placeholder, 2),
                name="add",
            )
        self.assertEqual(new_add.name, "add")
        graph.lint()

    def test_custom_tracer_node_rename_is_preserved(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        class PrefixNames(_ExportPassBaseDeprecatedDoNotUse):
            class ExportTracer(_ExportPassBaseDeprecatedDoNotUse.ExportTracer):
                def create_node(
                    self, kind, target, args, kwargs, name=None, type_expr=None
                ):
                    if kind == "call_function":
                        name = f"custom_{name or self.graph._target_to_str(target)}"
                    return super().create_node(
                        kind, target, args, kwargs, name, type_expr
                    )

        ep = export(M(), (torch.ones(2),))
        result = PrefixNames().call(ep.graph_module)
        add = next(
            node
            for node in result.graph_module.graph.nodes
            if node.target is torch.ops.aten.add.Tensor
        )
        self.assertEqual(add.name, "custom_add")

    def test_replaced_node_does_not_preserve_source_name(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        class ReplaceAdd(_ExportPassBaseDeprecatedDoNotUse):
            def call_operator(self, op, args, kwargs, meta):
                if op is torch.ops.aten.add.Tensor:
                    op = torch.ops.aten.sub.Tensor
                return super().call_operator(op, args, kwargs, meta)

        ep = export(M(), (torch.ones(2),))
        source_name = next(
            node.name
            for node in ep.graph.nodes
            if node.target is torch.ops.aten.add.Tensor
        )
        result = ReplaceAdd().call(ep.graph_module)
        replacement = next(
            node
            for node in result.graph_module.graph.nodes
            if node.target is torch.ops.aten.sub.Tensor
        )
        self.assertNotEqual(replacement.name, source_name)

    def test_graph_signature_updated_after_transformation(self) -> None:
        # Checks that pass infra correctly updates graph signature
        # after transformations.
        class CustomModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        my_module = CustomModule()

        # Test the custom module with two input tensors
        input_tensor1 = torch.tensor(5.0)
        input_tensor2 = torch.tensor(6.0)

        ep_before = torch.export.export(
            my_module, (input_tensor1, input_tensor2), strict=True
        )
        from torch.fx.passes.infra.pass_base import PassResult

        def modify_input_output_pass(gm):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        ep_after = ep_before._transform_do_not_use(modify_input_output_pass)
        new_signature = ep_after.graph_signature

        for node_name in new_signature.user_outputs:
            self.assertTrue("_modified" in node_name)

        old_signature = ep_before.graph_signature
        self.assertNotEqual(new_signature.user_outputs, old_signature.user_outputs)

    def test_replace_hook_basic(self) -> None:
        class CustomModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        my_module = CustomModule()
        inputs = (torch.tensor(6.0), torch.tensor(7.0))
        ep_before = export(my_module, inputs, strict=True)

        def replace_pass(gm):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        gm = copy.deepcopy(ep_before.graph_module)
        sig = copy.deepcopy(ep_before.graph_signature)

        with gm._set_replace_hook(sig.get_replace_hook()):
            replace_pass(gm)

        for node_name in sig.user_outputs:
            self.assertTrue("_modified" in node_name)

        old_signature = ep_before.graph_signature
        self.assertNotEqual(sig.user_outputs, old_signature.user_outputs)


instantiate_parametrized_tests(TestPassInfra)


if __name__ == "__main__":
    run_tests()
