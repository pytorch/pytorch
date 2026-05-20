# Owner(s): ["module: inductor"]

import logging
from types import SimpleNamespace

from torch._inductor.codegen.simd import NodeInfo
from torch._inductor.codegen.triton_combo_kernel import (
    _default_custom_combo_kernel_horizontal_partition,
    _log_partition_separation_once,
    LARGE_NUMELS,
)
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import run_tests, TestCase


class _Features:
    def __init__(self, is_reduction=False):
        self._is_reduction = is_reduction

    def is_reduction(self):
        return self._is_reduction


class _Node:
    def __init__(self, rnumel=1):
        self.group = [(None, rnumel)]


def _optimization_hint(expr, fallback=None):
    return expr


def _make_node_info(x_hint, rnumel=1, is_reduction=False):
    return NodeInfo(
        node_schedule=[],
        tiling={"x": x_hint, "y": rnumel},
        tiling_scores=None,
        numel=x_hint,
        rnumel=rnumel,
        features=_Features(is_reduction),
        is_persistent_reduction=False,
    )


class ComboKernelPartitionLoggingTests(TestCase):
    def test_large_pointwise_separation_debug_log_uses_companion_context(self):
        large_x = int(LARGE_NUMELS) + 1
        large_node = _Node()
        small_node = _Node()
        equivalent_large_node = _Node()
        equivalent_small_node = _Node()
        different_companion_large_node = _Node()
        different_small_node = _Node()
        node_info_map = {
            large_node: _make_node_info(large_x),
            small_node: _make_node_info(1024),
            equivalent_large_node: _make_node_info(large_x),
            equivalent_small_node: _make_node_info(1024),
            different_companion_large_node: _make_node_info(large_x),
            different_small_node: _make_node_info(2048),
        }
        graph = SimpleNamespace(
            sizevars=SimpleNamespace(optimization_hint=_optimization_hint)
        )
        logger = logging.getLogger("torch._inductor.codegen.triton_combo_kernel")

        _log_partition_separation_once.cache_clear()
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
                    [different_companion_large_node, different_small_node],
                    None,
                    node_info_map,
                )
        finally:
            _log_partition_separation_once.cache_clear()

        self.assertEqual(first_partitions, [[large_node], [small_node]])
        self.assertEqual(
            second_partitions, [[equivalent_large_node], [equivalent_small_node]]
        )
        self.assertEqual(
            third_partitions, [[different_companion_large_node], [different_small_node]]
        )
        large_pointwise_logs = [
            msg
            for msg in cm.output
            if "ComboKernels: 1 large pointwise nodes are separated" in msg
        ]
        self.assertEqual(len(large_pointwise_logs), 2)

    def test_long_reduction_separation_debug_log_uses_companion_context(self):
        long_node = _Node(rnumel=4096)
        short_node = _Node(rnumel=1024)
        equivalent_long_node = _Node(rnumel=4096)
        equivalent_short_node = _Node(rnumel=1024)
        different_companion_long_node = _Node(rnumel=4096)
        different_short_node = _Node(rnumel=1536)
        node_info_map = {
            long_node: _make_node_info(1024, rnumel=4096, is_reduction=True),
            short_node: _make_node_info(1024, rnumel=1024, is_reduction=True),
            equivalent_long_node: _make_node_info(1024, rnumel=4096, is_reduction=True),
            equivalent_short_node: _make_node_info(
                1024, rnumel=1024, is_reduction=True
            ),
            different_companion_long_node: _make_node_info(
                1024, rnumel=4096, is_reduction=True
            ),
            different_short_node: _make_node_info(1024, rnumel=1536, is_reduction=True),
        }
        graph = SimpleNamespace(
            sizevars=SimpleNamespace(optimization_hint=_optimization_hint)
        )
        logger = logging.getLogger("torch._inductor.codegen.triton_combo_kernel")

        _log_partition_separation_once.cache_clear()
        try:
            with (
                V.set_graph_handler(graph),
                self.assertLogs(logger, level=logging.DEBUG) as cm,
            ):
                first_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [long_node, short_node], None, node_info_map
                )
                second_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [equivalent_long_node, equivalent_short_node],
                    None,
                    node_info_map,
                )
                third_partitions = _default_custom_combo_kernel_horizontal_partition(
                    [different_companion_long_node, different_short_node],
                    None,
                    node_info_map,
                )
        finally:
            _log_partition_separation_once.cache_clear()

        self.assertEqual(first_partitions, [[short_node], [long_node]])
        self.assertEqual(
            second_partitions, [[equivalent_short_node], [equivalent_long_node]]
        )
        self.assertEqual(
            third_partitions,
            [[different_short_node], [different_companion_long_node]],
        )
        long_reduction_logs = [
            msg
            for msg in cm.output
            if "ComboKernels: 1 long reduction nodes are separated" in msg
        ]
        self.assertEqual(len(long_reduction_logs), 2)


if __name__ == "__main__":
    run_tests()
