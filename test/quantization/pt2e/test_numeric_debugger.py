# Owner(s): ["oncall: quantization"]

import copy
import unittest
from collections import Counter
from typing import Dict

import torch
from torch.ao.quantization import (
    compare_results,
    CUSTOM_KEY,
    extract_results_from_loggers,
    generate_numeric_debug_handle,
    NUMERIC_DEBUG_HANDLE_KEY,
    prepare_for_propagation_comparison,
)
from torch.ao.quantization.pt2e.graph_utils import bfs_trace_with_node_process
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.export import export_for_training
from torch.testing._internal.common_quantization import TestHelperModules
from torch.testing._internal.common_utils import IS_WINDOWS, skipIfCrossRef, TestCase


@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestNumericDebugger(TestCase):
    def _assert_each_node_has_debug_handle(self, model) -> None:
        def _assert_node_has_debug_handle(node):
            self.assertTrue(
                CUSTOM_KEY in node.meta
                and NUMERIC_DEBUG_HANDLE_KEY in node.meta[CUSTOM_KEY],
                f"Node {node} doesn't have debug handle",
            )

        bfs_trace_with_node_process(model, _assert_node_has_debug_handle)

    def _extract_debug_handles(self, model) -> Dict[str, int]:
        debug_handle_map: Dict[str, int] = {}

        def _extract_debug_handles_from_node(node):
            nonlocal debug_handle_map
            if (
                CUSTOM_KEY in node.meta
                and NUMERIC_DEBUG_HANDLE_KEY in node.meta[CUSTOM_KEY]
            ):
                debug_handle_map[str(node)] = node.meta[CUSTOM_KEY][
                    NUMERIC_DEBUG_HANDLE_KEY
                ]

        bfs_trace_with_node_process(model, _extract_debug_handles_from_node)

        return debug_handle_map

    def _extract_debug_handles_with_prev_decomp_op(self, model) -> Dict[str, int]:
        prev_decomp_op_to_debug_handle_map: Dict[str, int] = {}

        def _extract_debug_handles_with_prev_decomp_op_from_node(node):
            nonlocal prev_decomp_op_to_debug_handle_map
            if (
                CUSTOM_KEY in node.meta
                and NUMERIC_DEBUG_HANDLE_KEY in node.meta[CUSTOM_KEY]
            ):
                prev_decomp_op = str(node.meta.get("nn_module_stack"))
                debug_handle = node.meta[CUSTOM_KEY][NUMERIC_DEBUG_HANDLE_KEY]
                if prev_decomp_op not in prev_decomp_op_to_debug_handle_map:
                    prev_decomp_op_to_debug_handle_map[prev_decomp_op] = debug_handle
                else:
                    assert (
                        prev_decomp_op_to_debug_handle_map[prev_decomp_op]
                        == debug_handle
                    ), f"Node {node} has different debug handle {debug_handle}"
                    "than previous node sharing the same decomp op {prev_decomp_op}"

        bfs_trace_with_node_process(
            model, _extract_debug_handles_with_prev_decomp_op_from_node
        )
        return prev_decomp_op_to_debug_handle_map

    def test_simple(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs)
        generate_numeric_debug_handle(ep)
        self._assert_each_node_has_debug_handle(ep)
        debug_handle_map = self._extract_debug_handles(ep)

        self.assertEqual(len(set(debug_handle_map.values())), len(debug_handle_map))

    def test_control_flow(self):
        m = TestHelperModules.ControlFlow()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs)
        generate_numeric_debug_handle(ep)

        self._assert_each_node_has_debug_handle(ep)
        debug_handle_map = self._extract_debug_handles(ep)

        self.assertEqual(len(set(debug_handle_map.values())), len(debug_handle_map))

    def test_quantize_pt2e_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs)
        generate_numeric_debug_handle(ep)
        m = ep.module()

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False)
        )
        m = prepare_pt2e(m, quantizer)
        debug_handle_map = self._extract_debug_handles(m)
        res_counter = Counter(debug_handle_map.values())
        repeated_debug_handle_ids = [1, 2, 3]
        # 3 ids were repeated because we copy over the id from node to its output observer
        # torch.ops.aten.conv2d.default, torch.ops.aten.squeeze.dim and torch.ops.aten.conv1d.default
        for dh_id in repeated_debug_handle_ids:
            self.assertEqual(res_counter[dh_id], 2)

        m(*example_inputs)
        m = convert_pt2e(m)
        self._assert_each_node_has_debug_handle(ep)
        debug_handle_map = self._extract_debug_handles(m)
        res_counter = Counter(debug_handle_map.values())
        # same set of ids where repeated, because we copy over the id from observer/fake_quant to
        # dequantize node
        repeated_debug_handle_ids = [1, 2, 3]
        for dh_id in repeated_debug_handle_ids:
            self.assertEqual(res_counter[dh_id], 2)

    def test_copy_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = torch.export.export(m, example_inputs)
        generate_numeric_debug_handle(ep)

        self._assert_each_node_has_debug_handle(ep)
        debug_handle_map_ref = self._extract_debug_handles(ep)

        ep_copy = copy.copy(ep)
        debug_handle_map = self._extract_debug_handles(ep_copy)

        self._assert_each_node_has_debug_handle(ep)
        self.assertEqual(debug_handle_map, debug_handle_map_ref)

    def test_deepcopy_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = torch.export.export(m, example_inputs)
        generate_numeric_debug_handle(ep)

        debug_handle_map_ref = self._extract_debug_handles(ep)
        ep_copy = copy.deepcopy(ep)
        debug_handle_map = self._extract_debug_handles(ep_copy)

        self._assert_each_node_has_debug_handle(ep)
        self.assertEqual(debug_handle_map, debug_handle_map_ref)

    @skipIfCrossRef  # mlazos: retracing FX graph with torch function mode doesn't propagate metadata, because the stack
    # trace of the mode torch function impl doesn't match the traced graph stored lineno.
    def test_re_export_preserve_handle(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs)
        generate_numeric_debug_handle(ep)
        m = ep.module()

        self._assert_each_node_has_debug_handle(ep)
        debug_handle_map_ref = self._extract_debug_handles(ep)

        ep_reexport = export_for_training(m, example_inputs)

        self._assert_each_node_has_debug_handle(ep_reexport)
        debug_handle_map = self._extract_debug_handles(ep_reexport)

        self.assertEqual(debug_handle_map, debug_handle_map_ref)

    def test_run_decompositions_same_handle_id(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs)
        generate_numeric_debug_handle(ep)

        self._assert_each_node_has_debug_handle(ep)
        debug_handle_map_ref = self._extract_debug_handles(ep)

        ep_copy = copy.copy(ep)
        ep_copy = ep_copy.run_decompositions()

        self._assert_each_node_has_debug_handle(ep_copy)
        debug_handle_map = self._extract_debug_handles(ep_copy)

        # checking the map still has the same ids, the node may change
        self.assertEqual(
            set(debug_handle_map.values()), set(debug_handle_map_ref.values())
        )

    def test_run_decompositions_map_handle_to_new_nodes(self):
        test_models = [
            TestHelperModules.TwoLinearModule(),
            TestHelperModules.Conv2dThenConv1d(),
        ]

        for m in test_models:
            example_inputs = m.example_inputs()
            ep = export_for_training(m, example_inputs)
            generate_numeric_debug_handle(ep)

            self._assert_each_node_has_debug_handle(ep)
            pre_decomp_to_debug_handle_map_ref = (
                self._extract_debug_handles_with_prev_decomp_op(ep)
            )

            ep_copy = copy.copy(ep)
            ep_copy = ep_copy.run_decompositions()
            self._assert_each_node_has_debug_handle(ep_copy)
            pre_decomp_to_debug_handle_map = (
                self._extract_debug_handles_with_prev_decomp_op(ep_copy)
            )

            # checking the map still has the same ids, the node may change
            self.assertEqual(
                pre_decomp_to_debug_handle_map, pre_decomp_to_debug_handle_map_ref
            )

    def test_prepare_for_propagation_comparison(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs)
        generate_numeric_debug_handle(ep)
        m = ep.module()
        m_logger = prepare_for_propagation_comparison(m)
        ref = m(*example_inputs)
        res = m_logger(*example_inputs)

        from torch.ao.quantization.pt2e._numeric_debugger import OutputLogger

        loggers = [m for m in m_logger.modules() if isinstance(m, OutputLogger)]
        self.assertEqual(len(loggers), 3)
        self.assertTrue("conv2d" in [logger.node_name for logger in loggers])
        self.assertEqual(res, ref)

    def test_extract_results_from_loggers(self):
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs)
        generate_numeric_debug_handle(ep)
        m = ep.module()
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

    def test_added_node_gets_unique_id(self) -> None:
        m = TestHelperModules.Conv2dThenConv1d()
        example_inputs = m.example_inputs()
        ep = export_for_training(m, example_inputs)
        generate_numeric_debug_handle(ep)
        ref_handles = self._extract_debug_handles(ep)
        ref_counter = Counter(ref_handles.values())
        for k, v in ref_counter.items():
            self.assertEqual(
                v,
                1,
                msg=f"For handle {k}, there were {v} nodes with that handle, but expected only 1",
            )

        # Now that we have unique ids, add a new node into the graph and re-generate
        # to make sure that the new node gets a unique id.
        last_node = next(iter(reversed(ep.graph.nodes)))
        with ep.graph.inserting_before(last_node):
            arg = last_node.args[0]
            self.assertIsInstance(arg, (list, tuple))
            arg = arg[0]
            # Add a function that only requires a single tensor input.
            n = ep.graph.call_function(torch.ops.aten.relu.default, args=(arg,))
            arg.replace_all_uses_with(n, lambda x: x != n)
        ep.graph_module.recompile()

        # Regenerate handles, make sure only the new relu node has a new id, and
        # it doesn't clash with any of the existing ids.
        generate_numeric_debug_handle(ep)

        self._assert_each_node_has_debug_handle(ep)
        handles_after_modification = self._extract_debug_handles(ep)
        handles_counter = Counter(handles_after_modification.values())
        for name, handle in ref_handles.items():
            self.assertIn(name, handles_after_modification)
            # Check that handle was unchanged.
            self.assertEqual(handles_after_modification[name], handle)
            # Check that total count was unchanged.
            ref_count = ref_counter[handle]
            after_count = handles_counter[handle]
            self.assertEqual(
                after_count,
                ref_count,
                msg=f"For handle {handle}, there were {after_count} nodes with that handle, but expected only {ref_count}",
            )

        # Check for relu specifically. Avoid hardcoding the handle id since it
        # may change with future node ordering changes.
        self.assertNotEqual(handles_after_modification["relu_default"], 0)
        self.assertEqual(handles_counter[handles_after_modification["relu_default"]], 1)
