# Owner(s): ["oncall: export"]

import torch
from torch._dynamo.test_case import TestCase
from torch.testing._internal.common_utils import run_tests


class TestExportTools(TestCase):
    def test_source_id(self):
        forward_context = (torch.randn(3, 2),)

        class Model(torch.nn.Module):
            def forward(self, x):
                return x + forward_context[0]

        model = Model()
        inputs = (torch.randn(3, 2),)

        ep = torch.export.export(model, inputs, strict=False)
        lifted_node = next(iter(ep.graph.nodes))
        self.assertEqual(lifted_node.op, "placeholder")
        self.assertEqual(lifted_node.meta["source_id"], id(forward_context[0]))

        # rename the node
        lifted_node.name += "_kv_cache"
        lifted_node.target = lifted_node.name
        gm = ep.graph_module
        gm.recompile()
        self.assertExpectedInline(
            str(gm.graph),
            """\
graph():
    %c_lifted_tensor_0_kv_cache : [num_users=1] = placeholder[target=c_lifted_tensor_0_kv_cache]
    %x : [num_users=1] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %c_lifted_tensor_0_kv_cache), kwargs = {})
    return (add,)""",
        )


if __name__ == "__main__":
    run_tests()
