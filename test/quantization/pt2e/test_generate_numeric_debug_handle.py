# Owner(s): ["oncall: quantization"]

import unittest

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization import (
    generate_numeric_debug_handle,
    NUMERIC_DEBUG_HANDLE_KEY,
)
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

    debug_handle_map = {}
    names = ["output"]
    for name in names:
        node = name_node_map[name]
        if NUMERIC_DEBUG_HANDLE_KEY in node.meta:
            debug_handle_map[name] = node.meta[NUMERIC_DEBUG_HANDLE_KEY]

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
            if NUMERIC_DEBUG_HANDLE_KEY in n.meta:
                unique_ids.add(n.meta[NUMERIC_DEBUG_HANDLE_KEY])
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
