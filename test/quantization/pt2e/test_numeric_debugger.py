# Owner(s): ["oncall: quantization"]

import copy
import unittest
from collections import Counter
from typing import Dict

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization import (
    compare_results,
    CUSTOM_KEY,
    extract_results_from_loggers,
    generate_numeric_debug_handle,
    NUMERIC_DEBUG_HANDLE_KEY,
    prepare_for_propagation_comparison,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.testing._internal.common_quantization import TestHelperModules
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase


def _extract_debug_handles(model) -> Dict[torch.fx.Node, int]:
    debug_handle_map: Dict[torch.fx.Node, int] = {}

    for node in model.graph.nodes:
        if (
            CUSTOM_KEY in node.meta
            and NUMERIC_DEBUG_HANDLE_KEY in node.meta[CUSTOM_KEY]
        ):
            debug_handle_map[str(node)] = node.meta[CUSTOM_KEY][
                NUMERIC_DEBUG_HANDLE_KEY
            ]

    return debug_handle_map


def is_fbcode():
    return not hasattr(torch.version, "git_version")


@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestNumericDebugger(TestCase):
    def test_simple(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = torch.export.export(m, example_inputs)
        generate_numeric_debug_handle(m)
        unique_ids = set()
        count = 0
        for n in m.graph.nodes:
            if CUSTOM_KEY in n.meta and NUMERIC_DEBUG_HANDLE_KEY in n.meta[CUSTOM_KEY]:
                unique_ids.add(n.meta[CUSTOM_KEY][NUMERIC_DEBUG_HANDLE_KEY])
                count += 1
        self.assertEqual(len(unique_ids), count)

    @unittest.skipIf(
        is_fbcode(),
        "fbcode changes the code path for `capture_pre_autograd_graph` "
        "we can enable the test in fbcode after we remove `capture_pre_autograd_graph`",
    )
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
        repeated_debug_handle_ids = [3, 4, 7]
        # 3 ids were repeated because we copy over the id from node to its output observer
        # torch.ops.aten.conv2d.default, torch.ops.aten.squeeze.dim and torch.ops.aten.conv1d.default
        for dh_id in repeated_debug_handle_ids:
            self.assertEqual(res_counter[dh_id], 2)

        m(*example_inputs)
        m = convert_pt2e(m)
        debug_handle_map = _extract_debug_handles(m)
        res_counter = Counter(debug_handle_map.values())
        # same set of ids where repeated, because we copy over the id from observer/fake_quant to
        # dequantize node
        repeated_debug_handle_ids = [3, 4, 7]
        for dh_id in repeated_debug_handle_ids:
            self.assertEqual(res_counter[dh_id], 2)

    def test_copy_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = torch.export.export(m, example_inputs)
        generate_numeric_debug_handle(m)

        debug_handle_map_ref = _extract_debug_handles(m)

        m_copy = copy.copy(m)
        debug_handle_map = _extract_debug_handles(m_copy)

        self.assertEqual(debug_handle_map, debug_handle_map_ref)

    def test_deepcopy_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = torch.export.export(m, example_inputs)
        generate_numeric_debug_handle(m)

        debug_handle_map_ref = _extract_debug_handles(m)
        m_copy = copy.deepcopy(m)
        debug_handle_map = _extract_debug_handles(m_copy)

        self.assertEqual(debug_handle_map, debug_handle_map_ref)

    @unittest.skip("All nodes' meta are preserved but get_attr nodes' meta are wrong.")
    def test_re_export_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = capture_pre_autograd_graph(m, example_inputs)
        generate_numeric_debug_handle(m)

        debug_handle_map_ref = _extract_debug_handles(m)
        m_export = capture_pre_autograd_graph(m, example_inputs)
        debug_handle_map = _extract_debug_handles(m_export)

        self.assertEqual(debug_handle_map, debug_handle_map_ref)

    @unittest.skip(
        "All nodes' meta are preserved but the first arg for the first node seems to be dropped"
    )
    def test_run_decompositions_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = torch.export.export(m, example_inputs)
        generate_numeric_debug_handle(m)

        debug_handle_map_ref = _extract_debug_handles(m)

        m_copy = copy.copy(m)
        m_copy = m_copy.run_decompositions()
        debug_handle_map = _extract_debug_handles(m_copy)

        # checking the map still has the same ids, the node may change
        self.assertEqual(
            set(debug_handle_map.values()), set(debug_handle_map_ref.values())
        )

    def test_prepare_for_propagation_comparison(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = capture_pre_autograd_graph(m, example_inputs)
        generate_numeric_debug_handle(m)
        m_logger = prepare_for_propagation_comparison(m)
        ref = m(*example_inputs)
        res = m_logger(*example_inputs)

        from torch.ao.quantization.pt2e._numeric_debugger import OutputLogger

        loggers = [m for m in m_logger.modules() if isinstance(m, OutputLogger)]
        self.assertEqual(len(loggers), 8)
        self.assertTrue("conv2d" in [logger.node_name for logger in loggers])
        self.assertEqual(res, ref)

    def test_extract_results_from_loggers(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        m = capture_pre_autograd_graph(m, example_inputs)
        generate_numeric_debug_handle(m)
        m_ref_logger = prepare_for_propagation_comparison(m)

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        m_quant_logger = prepare_for_propagation_comparison(m)

        m_ref_logger(*example_inputs)
        m_quant_logger(*example_inputs)
        ref_results = extract_results_from_loggers(m_ref_logger)
        quant_results = extract_results_from_loggers(m_quant_logger)
        comparison_results = compare_results(ref_results, quant_results)
        for node_summary in comparison_results.values():
            if len(node_summary.results) > 0:
                self.assertGreaterEqual(node_summary.results[0].sqnr, 35)
