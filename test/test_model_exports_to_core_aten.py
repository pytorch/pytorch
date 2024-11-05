# Owner(s): ["oncall: mobile"]
import copy

import pytest

import torch
import torch._export as export
from torch.testing._internal.common_quantization import skip_if_no_torchvision
from torch.testing._internal.common_utils import TestCase


def _get_ops_list(m: torch.fx.GraphModule):
    op_list = []
    for n in m.graph.nodes:
        if n.op == "call_function":
            op_list.append(n.target)
    return op_list


class TestQuantizePT2EModels(TestCase):
    @pytest.mark.xfail
    @skip_if_no_torchvision
    def test_vit_aten_export(self):
        from torchvision.models import vit_b_16  # @manual

        m = vit_b_16(weights="IMAGENET1K_V1")
        m = m.eval()
        input_shape = (1, 3, 224, 224)
        example_inputs = (torch.randn(input_shape),)
        m = torch.export.export_for_training(m, copy.deepcopy(example_inputs)).module()
        m(*example_inputs)
        m = export.export(m, copy.deepcopy(example_inputs))
        ops = _get_ops_list(m.graph_module)
        non_core_aten_op_found = False
        for op in ops:
            if "scaled_dot_product" in str(op):
                non_core_aten_op_found = True
        self.assertFalse(non_core_aten_op_found)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
