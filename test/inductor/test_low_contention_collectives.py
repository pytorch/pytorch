# Owner(s): ["oncall: pt2"]

import unittest

import torch
import torch.distributed as dist
from torch._inductor import config
from torch._inductor.fx_passes.low_contention_collectives import (
    replace_collectives_with_low_contention,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


if dist.is_available():
    from torch.distributed._symmetric_memory import _test_mode


@unittest.skipIf(not dist.is_available(), "requires distributed")
class TestLowContentionCollectives(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _build_ag_graph(self):
        """Build an FX graph: input -> all_gather -> wait_tensor -> output (no compute)."""
        c10d_fn = torch.ops._c10d_functional
        graph = torch.fx.Graph()
        inp = graph.placeholder("input")
        inp.meta["val"] = torch.empty(1024, 1024)

        ag = graph.call_function(
            c10d_fn.all_gather_into_tensor.default,
            args=(inp, 2, "test_group"),
        )
        ag.meta["val"] = torch.empty(2048, 1024)

        wait = graph.call_function(c10d_fn.wait_tensor.default, args=(ag,))
        wait.meta["val"] = torch.empty(2048, 1024)

        graph.output(wait)
        return graph

    def test_skip_overlap_check(self):
        c10d_fn = torch.ops._c10d_functional
        symm_mem = torch.ops.symm_mem
        lc_config = {
            "aten_distributed_optimizations.low_contention_min_bytes_per_rank": 0,
        }

        def get_targets(graph):
            return [n.target for n in graph.nodes if n.op == "call_function"]

        # Default: no compute overlap -> collective NOT replaced
        graph = self._build_ag_graph()
        with (
            _test_mode(),
            config.patch(
                {
                    **lc_config,
                    "aten_distributed_optimizations.low_contention_skip_overlap_check": False,
                }
            ),
        ):
            replace_collectives_with_low_contention(graph)
        self.assertIn(c10d_fn.all_gather_into_tensor.default, get_targets(graph))

        # skip_overlap_check=True: collective replaced despite no compute
        graph = self._build_ag_graph()
        with (
            _test_mode(),
            config.patch(
                {
                    **lc_config,
                    "aten_distributed_optimizations.low_contention_skip_overlap_check": True,
                }
            ),
        ):
            replace_collectives_with_low_contention(graph)
        self.assertIn(symm_mem._low_contention_all_gather.default, get_targets(graph))
        self.assertNotIn(c10d_fn.all_gather_into_tensor.default, get_targets(graph))


if __name__ == "__main__":
    run_tests()
