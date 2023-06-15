import unittest

import torch
from torch._export import export
from torch._export.serde.serialize import GraphModuleOpUpgrader

TEST_UPGRADERS = {
    "aten::div__Scalar_mode_0_3": (
        "div.Scalar_mode(Tensor self, Scalar other, *, str rounding_mode) -> Tensor",
        """
def div__Scalar_mode_0_3(self: torch.Tensor, other: Any,  *, rounding_mode: Optional[str]=None) -> torch.Tensor:
    return self.divide(other, rounding_mode=rounding_mode)
        """,
    ),
    "aten::gelu_0_9": (
        "gelu(Tensor self) -> Tensor",
        """
def gelu_0_9(self: Tensor) -> Tensor:
  return torch.gelu(self, approximate='none')
        """,
    ),
}


def count_op(graph, target_str):
    return len(
        [n for n in graph.nodes if isinstance(n.target, torch._ops.OpOverload) and n.target.name() == target_str])


class TestUpgrade(unittest.TestCase):
    def test_creates_upgrader_pass(self):
        compiler_opset_version = {"aten": 4}
        model_opset_version = {"aten": 3}
        upgrader = GraphModuleOpUpgrader(compiler_opset_version, model_opset_version, TEST_UPGRADERS)
        self.assertEqual(len(upgrader.upgrader_passes), 1)

    def test_div_upgrader_replaces_op_with_old_version(self):
        def fn(a: torch.Tensor, b):
            return torch.ops.aten.div.Scalar_mode(a, b, rounding_mode='trunc')

        inputs = (torch.ones([2, 3]) * 4, 2.)
        ep = export(fn, inputs, [])
        compiler_opset_version = {"aten": 4}
        model_opset_version = {"aten": 3}
        upgrader = GraphModuleOpUpgrader(compiler_opset_version, model_opset_version, TEST_UPGRADERS)
        upgraded = ep.transform(*upgrader.upgrader_passes)
        upgraded.graph_module.print_readable()

        count = count_op(upgraded.graph, "aten::div.Scalar_mode")
        self.assertEqual(count, 0)
        custom_op_count = count_op(upgraded.graph, "aten::div__Scalar_mode_0_3")
        self.assertEqual(custom_op_count, 1)

    def test_div_upgrader_pass_return_new_op_after_retrace(self):
        def fn(a: torch.Tensor, b):
            return torch.ops.aten.div.Scalar_mode(a, b, rounding_mode='trunc')

        inputs = (torch.ones([2, 3]) * 4, 2.)
        ep = export(fn, inputs, [])
        compiler_opset_version = {"aten": 4}
        model_opset_version = {"aten": 3}
        upgrader = GraphModuleOpUpgrader(compiler_opset_version, model_opset_version, TEST_UPGRADERS)

        count = count_op(ep.graph, "aten::div.Scalar_mode")
        self.assertEqual(count, 1)

        # upgrade: replace op (div.Scalar_mode -> div__Scalar_mode_0_3) then retrace
        upgraded_ep = upgrader.upgrade(ep)
        upgraded_ep.graph_module.print_readable()

        # no old version of op (div__Scalar_mode_0_3) anymore.
        custom_op_count = count_op(upgraded_ep.graph, "aten::div__Scalar_mode_0_3")
        self.assertEqual(custom_op_count, 0)

        # div__Scalar_mode_0_3 decomposes into div.Tensor.
        decomposed_op_count = count_op(upgraded_ep.graph, "aten::div.Tensor_mode")
        self.assertEqual(decomposed_op_count, 1)


if __name__ == '__main__':
    unittest.main()
