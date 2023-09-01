
import torch
import torch._export as export
import copy
from torch._decomp import core_aten_decompositions, get_decompositions
from unittest.mock import patch

from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    skip_if_no_torchvision,
    skipIfNoQNNPACK,
)
from torch.testing._internal.common_utils import TestCase

def _get_ops_list(m: torch.fx.GraphModule):
    op_list = []
    for n in m.graph.nodes:
        if n.op == "call_function":
            op_list.append(n.target)
    return op_list

class TestQuantizePT2EModels(TestCase):

    @skip_if_no_torchvision
    def test_vit_aten_export(self):
        from torchvision.models import vit_b_16  # @manual
        m = vit_b_16(weights="IMAGENET1K_V1")
        m = m.eval()
        input_shape = (1, 3, 224, 224)
        example_inputs = (torch.randn(input_shape),)
        m = export.capture_pre_autograd_graph(m, copy.deepcopy(example_inputs))
        m(*example_inputs)
        m = export.export(m, copy.deepcopy(example_inputs))
        ops = _get_ops_list(m.graph_module)
        non_core_aten_op_found = False
        for op in ops:
            if "scaled_dot_product" in str(op):
                non_core_aten_op_found = True
        self.assertFalse(non_core_aten_op_found)
