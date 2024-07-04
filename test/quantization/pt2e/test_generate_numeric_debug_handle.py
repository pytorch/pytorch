# Owner(s): ["oncall: quantization"]

import unittest
from collections import Counter
from typing import Dict

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization import (
    generate_numeric_debug_handle,
    NUMERIC_DEBUG_HANDLE_KEY,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.fx import Node
import copy
from torch.testing._internal.common_quantization import TestHelperModules
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase


def _extract_debug_handles(model) -> Dict[torch.fx.Node, int]:
    debug_handle_map: Dict[torch.fx.Node, int] = {}

    for node in model.graph.nodes:
        if NUMERIC_DEBUG_HANDLE_KEY in node.meta:
            debug_handle_map[str(node)] = node.meta[NUMERIC_DEBUG_HANDLE_KEY]

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

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        m = prepare_pt2e(m, quantizer)
        debug_handle_map = _extract_debug_handles(m)
        res_counter = Counter(debug_handle_map.values())
        repeated_debug_handle_ids = [2, 3, 6]
        # 3 ids were repeated because we copy over the id from node to its output observer
        # torch.ops.aten.conv2d.default, torch.ops.aten.squeeze.dim, torch.ops.aten.conv1d.default
        for dh_id in repeated_debug_handle_ids:
            self.assertEqual(res_counter[dh_id], 2)

        m(*example_inputs)
        m = convert_pt2e(m)
        debug_handle_map = _extract_debug_handles(m)
        res_counter = Counter(debug_handle_map.values())
        # same set of ids where repeated, because we copy over the id from observer/fake_quant to
        # dequantize node
        repeated_debug_handle_ids = [2, 3, 6]
        for dh_id in repeated_debug_handle_ids:
            self.assertEqual(res_counter[dh_id], 2)

    def test_copy_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = capture_pre_autograd_graph(m, example_inputs)
        generate_numeric_debug_handle(m)

        debug_handle_map_ref = _extract_debug_handles(m)

        m_copy = copy.copy(m)
        debug_handle_map = _extract_debug_handles(m_copy)

        self.assertEqual(debug_handle_map, debug_handle_map_ref)

    def test_deepcopy_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = capture_pre_autograd_graph(m, example_inputs)
        generate_numeric_debug_handle(m)

        debug_handle_map_ref = _extract_debug_handles(m)
        m_copy = copy.deepcopy(m)
        debug_handle_map = _extract_debug_handles(m_copy)

        self.assertEqual(debug_handle_map, debug_handle_map_ref)

    @unittest.skip("reexport is not fully supported yet, need to add support for output node")
    def test_re_export_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = capture_pre_autograd_graph(m, example_inputs)
        generate_numeric_debug_handle(m)

        debug_handle_map_ref = _extract_debug_handles(m)
        m_export = capture_pre_autograd_graph(m, example_inputs)
        debug_handle_map = _extract_debug_handles(m_export)

        self.assertEqual(debug_handle_map, debug_handle_map_ref)
