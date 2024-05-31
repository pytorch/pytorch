# Owner(s): ["oncall: quantization"]

import unittest

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization import generate_numeric_debug_handle
from torch.ao.quantization.pt2e.export_utils import _WrapperModule
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.fx import Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)
from torch.testing._internal.common_quantization import TestHelperModules
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase


def _extract_conv2d_pattern_debug_handle_map(model):
    """Returns a debug_handle_map from input/weight/bias/output to numeric_debug_handle
    for conv2d pattern, extracted from the model
    """

    def conv_pattern(input, weight, bias):
        output = torch.nn.functional.conv2d(input, weight, bias)
        return output, {
            "input": input,
            "weight": weight,
            "bias": bias,
            "output": output,
        }

    conv_pattern_example_inputs = (
        torch.randn(1, 1, 3, 3),  # input
        torch.randn(1, 1, 1, 1),  # weight
        torch.randn(1),  # bias
    )
    conv_gm = capture_pre_autograd_graph(
        _WrapperModule(conv_pattern), conv_pattern_example_inputs
    )
    conv_pm = SubgraphMatcherWithNameNodeMap(conv_gm)
    matches = conv_pm.match(model.graph)
    assert len(matches) == 1, "Expecting to have one match"
    match = matches[0]
    name_node_map = match.name_node_map
    input_node = name_node_map["input"]
    weight_node = name_node_map["weight"]
    bias_node = name_node_map["bias"]
    output_node = name_node_map["output"]

    debug_handle_map = {}
    conv_node = output_node
    if input_node not in conv_node.meta["numeric_debug_handle"]:
        return debug_handle_map
    debug_handle_map["input"] = conv_node.meta["numeric_debug_handle"][input_node]
    debug_handle_map["weight"] = conv_node.meta["numeric_debug_handle"][weight_node]
    if bias_node is not None:
        debug_handle_map["bias"] = conv_node.meta["numeric_debug_handle"][bias_node]
    debug_handle_map["output"] = conv_node.meta["numeric_debug_handle"]["output"]
    return debug_handle_map


@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestGenerateNumericDebugHandle(TestCase):
    def test_simple(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = capture_pre_autograd_graph(m, example_inputs)
        generate_numeric_debug_handle(m)
        unique_ids = set()
        count = 0
        for n in m.graph.nodes:
            if "numeric_debug_handle" in n.meta:
                for arg in n.args:
                    if isinstance(arg, Node):
                        unique_ids.add(n.meta["numeric_debug_handle"][arg])
                        count += 1
                unique_ids.add(n.meta["numeric_debug_handle"]["output"])
                count += 1
        self.assertEqual(len(unique_ids), count)

    def test_quantize_pt2e_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = capture_pre_autograd_graph(m, example_inputs)
        generate_numeric_debug_handle(m)

        debug_handle_map_ref = _extract_conv2d_pattern_debug_handle_map(m)

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        m = prepare_pt2e(m, quantizer)
        debug_handle_map = _extract_conv2d_pattern_debug_handle_map(m)
        self.assertEqual(debug_handle_map, debug_handle_map_ref)
        m(*example_inputs)
        m = convert_pt2e(m)
        debug_handle_map = _extract_conv2d_pattern_debug_handle_map(m)
        self.assertEqual(debug_handle_map, debug_handle_map_ref)
