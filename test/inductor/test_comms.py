# Owner(s): ["module: inductor"]
from unittest import mock

import torch._inductor.comms as comms
from torch._inductor.dependencies import StarDep
from torch._inductor.memory import SNodeMemory
from torch._inductor.scheduler import BaseSchedulerNode
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestSinkWaits(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @comms.config_comms.patch(sink_iterative_use_runtime_estimations=True)
    @parametrize("count_group_compute", [False, True])
    @parametrize("compute_runtime", [0.0, 1.0])
    def test_accounts_for_grouped_compute(self, compute_runtime, count_group_compute):
        def make_node(name, kind, deps=()):
            node = mock.Mock(spec=BaseSchedulerNode)
            node.node = node
            node.kind = kind
            node.get_name.return_value = name
            node.get_outputs.return_value = [mock.Mock(get_name=lambda: name)]
            node.unmet_dependencies = [StarDep(dep) for dep in deps]
            return node

        ag = make_node("ag", "collective")
        rs = make_node("rs", "collective")
        ag_wait = make_node("ag_wait", "wait", ["ag"])
        compute = make_node("compute", "compute", ["ag_wait"])
        rs_wait = make_node("rs_wait", "wait", ["rs"])
        snodes = [ag, rs, ag_wait, compute, rs_wait]
        graph = mock.Mock(graph_inputs={})
        graph.get_output_names.return_value = []

        def is_collective(node):
            return node.kind == "collective"

        def is_wait(node):
            return node.kind == "wait"

        with (
            comms.config_comms.patch(
                sink_waits_iterative_count_group_compute=count_group_compute
            ),
            mock.patch.object(comms, "V", mock.Mock(graph=graph)),
            mock.patch.object(
                comms,
                "_initialize_memory_tracking",
                return_value=(
                    0,
                    dict.fromkeys(snodes, (0, 0)),
                    {node: SNodeMemory(0, 0) for node in snodes},
                    {},
                    {},
                    {},
                ),
            ),
            mock.patch.object(comms, "contains_collective", is_collective),
            mock.patch.object(comms, "contains_async_collective", is_collective),
            mock.patch.object(comms, "contains_wait", is_wait),
            mock.patch.object(comms, "is_wait", is_wait),
            mock.patch.object(
                comms,
                "estimate_op_runtime",
                lambda node: compute_runtime if node is compute else 1.0,
            ),
            mock.patch.object(
                comms, "_wait_exposed_communication_time", return_value=(1.0, 1.0, "")
            ),
        ):
            reordered, stats = comms._sink_waits_iterative_internal(snodes)

        blocked = compute_runtime and count_group_compute
        expected = snodes if blocked else [ag, rs, rs_wait, ag_wait, compute]
        self.assertEqual(reordered, expected)
        self.assertEqual(stats[ag_wait].grouped, 1)


if __name__ == "__main__":
    run_tests()
