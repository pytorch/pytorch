# Owner(s): ["module: onnx"]

"""Tests for `torch.onnx.symbolic_opset9`."""
import torch
from torch import _C
from torch.onnx import symbolic_opset9 as opset9
from torch.testing._internal import common_utils


def create_empty_graph(opset_version: int) -> _C.Graph:
    graph = _C.Graph()
    # kick off initialization for ConstantMap.
    # _C._jit_pass_onnx_graph_shape_type_inference(graph, {}, opset_version)
    return graph


class TestPrim(common_utils.TestCase):
    def setUp(self):
        super().setUp()
        self.graph = create_empty_graph(opset_version=9)

    def test_list_unpack_returns_all_list_elements_when_previous_node_is_list_construct(self):
        # Build the graph
        self.graph.create(prim, args, outputs)
        input_1 = self.graph.addInput()
        input_1.setType(input_1.type().with_dtype(torch.float).with_sizes([2, 42]))
        input_2 = self.graph.addInput()
        input_2.setType(input_2.type().with_dtype(torch.float).with_sizes([3, 42]))
        constructed_list = self.graph.op("prim::ListConstruct", input_1, input_2)

        outputs = opset9.Prim.ListUnpack(self.graph, constructed_list)
        self.assertNotEqual(outputs, None)
        self.assertEqual(outputs[0], input_1)
        self.assertEqual(outputs[1], input_1)


if __name__ == "__main__":
    common_utils.run_tests()
