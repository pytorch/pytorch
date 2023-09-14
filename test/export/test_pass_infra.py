# Owner(s): ["module: dynamo"]
import unittest
from typing import List

import torch
from functorch.experimental import control_flow
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export import export
from torch._export.constraints import constrain_as_value
from torch._export.pass_base import _ExportPassBase
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPassInfra(TestCase):
    def test_export_pass_base(self) -> None:
        def f(x: torch.Tensor) -> List[torch.Tensor]:
            y = torch.cat([x, x])
            return torch.ops.aten.tensor_split.sections(y, 2)

        class NullPass(_ExportPassBase):
            pass

        ep = export(f, (torch.ones(3, 2),))
        old_nodes = ep.graph.nodes

        ep = ep._transform(NullPass())
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

    def test_cond(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred, x, y):
                def true_fn(x, y):
                    b = x.item()
                    constrain_as_value(b, min=2, max=5)
                    return x - y

                def false_fn(x, y):
                    c = y.item()
                    constrain_as_value(c, min=2, max=5)
                    return x + y

                ret = control_flow.cond(pred, true_fn, false_fn, [x, y])
                return ret

        x = torch.tensor([2])
        y = torch.tensor([5])
        mod = M()
        _ = export(mod, (torch.tensor(True), x, y))._transform(_ExportPassBase())

    def test_node_name_stability(self) -> None:
        # Tests that graph nodes stay the same for nodes that are not touched
        # during transformation
        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

                # Define a parameter
                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                # Define two buffers
                self.register_buffer('my_buffer1', torch.tensor(3.0))
                self.register_buffer('my_buffer2', torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (x1 + self.my_parameter) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)

                return output

        inps = (torch.rand(1), torch.rand(1))
        m = CustomModule()

        ep_before = export(m, inps)

        # No op transformation that doesn't perform any meaningful changes to node
        ep_after = ep_before._transform(_ExportPassBase())

        for before_node, after_node in zip(ep_before.graph.nodes, ep_after.graph.nodes):
            self.assertEqual(before_node.name, after_node.name)

    def test_graph_signature_updated_after_transformation(self) -> None:
        # Checks that pass infra correctly updates graph signature
        # after transformations.
        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.register_buffer("my_buffer1", torch.tensor(3.0))
                self.register_buffer("my_buffer2", torch.tensor(4.0))

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

        ep_before = torch._export.export(my_module, (input_tensor1, input_tensor2))
        from torch.fx.passes.infra.pass_base import PassResult

        def modify_input_output_pass(gm):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        ep_after = ep_before._transform(modify_input_output_pass)
        new_signature = ep_after.graph_signature

        for node_name in new_signature.user_outputs:
            self.assertTrue("_modified" in node_name)

        old_signature = ep_before.graph_signature
        self.assertNotEqual(new_signature.user_outputs, old_signature.user_outputs)


if __name__ == '__main__':
    run_tests()
