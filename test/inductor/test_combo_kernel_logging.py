# Owner(s): ["module: inductor"]

import logging
from types import SimpleNamespace

from torch._inductor.codegen.simd import NodeInfo
from torch._inductor.codegen.triton_combo_kernel import (
    _default_custom_combo_kernel_horizontal_partition,
    _log_large_pointwise_separation_once,
    LARGE_NUMELS,
)
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import run_tests, TestCase


class ComboKernelPartitionLoggingTests(TestCase):
    def test_large_pointwise_separation_debug_log_uses_partition_context(self):
        class Features:
            def is_reduction(self):
                return False

        def optimization_hint(expr, fallback=None):
            return expr

        def make_node_info(x_hint):
            return NodeInfo(
                node_schedule=[],
                tiling={"x": x_hint, "y": 1},
                tiling_scores=None,
                numel=x_hint,
                rnumel=1,
                features=Features(),
                is_persistent_reduction=False,
            )

        large_x = int(LARGE_NUMELS) + 1
        different_large_x = large_x + 1024
        large_node = object()
        small_node = object()
        equivalent_large_node = object()
        equivalent_small_node = object()
        different_large_node = object()
        different_small_node = object()
        node_info_map = {
            large_node: make_node_info(large_x),
            small_node: make_node_info(1024),
            equivalent_large_node: make_node_info(large_x),
            equivalent_small_node: make_node_info(1024),
            different_large_node: make_node_info(different_large_x),
            different_small_node: make_node_info(1024),
        }
        graph = SimpleNamespace(
            sizevars=SimpleNamespace(optimization_hint=optimization_hint)
        )
        logger = logging.getLogger("torch._inductor.codegen.triton_combo_kernel")

        _log_large_pointwise_separation_once.cache_clear()
        try:
            with (
                V.set_graph_handler(graph),
                self.assertLogs(logger, level=logging.DEBUG) as cm,
            ):
                first_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [large_node, small_node], None, node_info_map
                )
                second_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [equivalent_large_node, equivalent_small_node],
                    None,
                    node_info_map,
                )
                third_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [different_large_node, different_small_node], None, node_info_map
                )
        finally:
            _log_large_pointwise_separation_once.cache_clear()

        self.assertEqual(first_partitions, [[large_node], [small_node]])
        self.assertEqual(
            second_partitions, [[equivalent_large_node], [equivalent_small_node]]
        )
        self.assertEqual(
            third_partitions, [[different_large_node], [different_small_node]]
        )
        large_pointwise_logs = [
            msg
            for msg in cm.output
            if "ComboKernels: 1 large pointwise nodes are separated" in msg
        ]
        self.assertEqual(len(large_pointwise_logs), 2)


if __name__ == "__main__":
    run_tests()
