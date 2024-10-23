# Owner(s): ["module: unknown"]
from typing import Dict

from torch.distributed._tools.fsdp_ilp import CommParams, CommType, fsdp_milp
from torch.distributed._tools.ilp_utils import ModuleInfo, parse_module_info
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFSDPILP(TestCase):
    """
    Test the fsdp ilp formulation on a LLM model with transformation blocks.
    ``mod_info`` and ``comm_params`` are hard coded instead of traced to avoid machine dependency.
    """

    def setUp(self):
        super().setUp()
        self.comm_params = self._get_test_comm_params()
        self.comm_params_low_bw = self._get_test_comm_params(True)
        self.mod_info = self._get_mod_info()
        self.g = parse_module_info(self.mod_info)

    def _get_test_comm_params(
        self, comm_bound: bool = False
    ) -> Dict[CommType, CommParams]:
        if comm_bound:
            return {
                CommType.ALL_GATHER: CommParams(latency=0.01, bandwidth=1e7),
                CommType.REDUCE_SCATTER: CommParams(latency=0.01, bandwidth=1e7),
            }
        else:
            return {
                CommType.ALL_GATHER: CommParams(latency=0.01, bandwidth=1e8),
                CommType.REDUCE_SCATTER: CommParams(latency=0.01, bandwidth=1e8),
            }

    def _get_mod_info(self) -> ModuleInfo:
        mod_info = {
            "mod_order": {
                "fw_pre_order": [
                    "Transformer",
                    "Transformer.layers.0",
                    "Transformer.layers.0.attention",
                    "Transformer.layers.0.feed_forward",
                    "Transformer.layers.1",
                    "Transformer.layers.1.attention",
                    "Transformer.layers.1.feed_forward",
                    "Transformer.layers.2",
                    "Transformer.layers.2.attention",
                    "Transformer.layers.2.feed_forward",
                    "Transformer.layers.3",
                    "Transformer.layers.3.attention",
                    "Transformer.layers.3.feed_forward",
                    "Transformer.output",
                ],
                "bw_pre_order": [
                    "Transformer",
                    "Transformer.output",
                    "Transformer.layers.3",
                    "Transformer.layers.3.feed_forward",
                    "Transformer.layers.3.attention",
                    "Transformer.layers.2",
                    "Transformer.layers.2.feed_forward",
                    "Transformer.layers.2.attention",
                    "Transformer.layers.1",
                    "Transformer.layers.1.feed_forward",
                    "Transformer.layers.1.attention",
                    "Transformer.layers.0",
                    "Transformer.layers.0.feed_forward",
                    "Transformer.layers.0.attention",
                ],
                "fw_post_order": [
                    "Transformer.layers.0.attention",
                    "Transformer.layers.0.feed_forward",
                    "Transformer.layers.0",
                    "Transformer.layers.1.attention",
                    "Transformer.layers.1.feed_forward",
                    "Transformer.layers.1",
                    "Transformer.layers.2.attention",
                    "Transformer.layers.2.feed_forward",
                    "Transformer.layers.2",
                    "Transformer.layers.3.attention",
                    "Transformer.layers.3.feed_forward",
                    "Transformer.layers.3",
                    "Transformer.output",
                    "Transformer",
                ],
                "bw_post_order": [
                    "Transformer.output",
                    "Transformer.layers.3.feed_forward",
                    "Transformer.layers.3.attention",
                    "Transformer.layers.3",
                    "Transformer.layers.2.feed_forward",
                    "Transformer.layers.2.attention",
                    "Transformer.layers.2",
                    "Transformer.layers.1.feed_forward",
                    "Transformer.layers.1.attention",
                    "Transformer.layers.1",
                    "Transformer.layers.0.feed_forward",
                    "Transformer.layers.0.attention",
                    "Transformer.layers.0",
                    "Transformer",
                ],
            },
            "mod_stats": [
                {
                    "fqn": "Transformer",
                    "param_per_module": 1960000000,
                    "grad_per_module": 1960000000,
                    "grad_total": 0,
                    "act_fw_per_module": 2548856832,
                    "act_bw_per_module": 482486272,
                    "act_grad_per_module": 402665472,
                    "act_total": 2683074560,
                    "input_per_module": 0,
                    "output_per_module": 67108864,
                    "fw_runtime_per_module": 115.51510375623819,
                    "bw_runtime_per_module": 262.8396350763604,
                },
                {
                    "fqn": "Transformer.layers.0",
                    "param_per_module": 453095424,
                    "grad_per_module": 453095424,
                    "grad_total": 2265501696,
                    "act_fw_per_module": 390202368,
                    "act_bw_per_module": 482486272,
                    "act_grad_per_module": 276836352,
                    "act_total": 526525440,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 18.586358962812866,
                    "bw_runtime_per_module": 42.31444884235838,
                },
                {
                    "fqn": "Transformer.layers.0.attention",
                    "param_per_module": 150994944,
                    "grad_per_module": 150994944,
                    "grad_total": 2567577600,
                    "act_fw_per_module": 107054080,
                    "act_bw_per_module": 224520192,
                    "act_grad_per_module": 100663296,
                    "act_total": 268559360,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 6.914146061765929,
                    "bw_runtime_per_module": 16.190205050846142,
                },
                {
                    "fqn": "Transformer.layers.0.feed_forward",
                    "param_per_module": 302051328,
                    "grad_per_module": 302051328,
                    "grad_total": 2265501696,
                    "act_fw_per_module": 207618048,
                    "act_bw_per_module": 482486272,
                    "act_grad_per_module": 276836352,
                    "act_total": 526525440,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 11.49012612856017,
                    "bw_runtime_per_module": 25.499871304739376,
                },
                {
                    "fqn": "Transformer.layers.1",
                    "param_per_module": 453095424,
                    "grad_per_module": 453095424,
                    "grad_total": 1812406272,
                    "act_fw_per_module": 390202368,
                    "act_bw_per_module": 482486272,
                    "act_grad_per_module": 276836352,
                    "act_total": 941893632,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 18.586358962812866,
                    "bw_runtime_per_module": 42.31444884235838,
                },
                {
                    "fqn": "Transformer.layers.1.attention",
                    "param_per_module": 150994944,
                    "grad_per_module": 150994944,
                    "grad_total": 2114482176,
                    "act_fw_per_module": 107054080,
                    "act_bw_per_module": 224520192,
                    "act_grad_per_module": 100663296,
                    "act_total": 683927552,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 6.914146061765929,
                    "bw_runtime_per_module": 16.190205050846142,
                },
                {
                    "fqn": "Transformer.layers.1.feed_forward",
                    "param_per_module": 302051328,
                    "grad_per_module": 302051328,
                    "grad_total": 1812406272,
                    "act_fw_per_module": 207618048,
                    "act_bw_per_module": 482486272,
                    "act_grad_per_module": 276836352,
                    "act_total": 526525440,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 11.49012612856017,
                    "bw_runtime_per_module": 25.499871304739376,
                },
                {
                    "fqn": "Transformer.layers.2",
                    "param_per_module": 453095424,
                    "grad_per_module": 453095424,
                    "grad_total": 1359310848,
                    "act_fw_per_module": 390202368,
                    "act_bw_per_module": 482486272,
                    "act_grad_per_module": 276836352,
                    "act_total": 1357261824,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 18.586358962812866,
                    "bw_runtime_per_module": 42.31444884235838,
                },
                {
                    "fqn": "Transformer.layers.2.attention",
                    "param_per_module": 150994944,
                    "grad_per_module": 150994944,
                    "grad_total": 1661386752,
                    "act_fw_per_module": 107054080,
                    "act_bw_per_module": 224520192,
                    "act_grad_per_module": 100663296,
                    "act_total": 1099295744,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 6.914146061765929,
                    "bw_runtime_per_module": 16.190205050846142,
                },
                {
                    "fqn": "Transformer.layers.2.feed_forward",
                    "param_per_module": 302051328,
                    "grad_per_module": 302051328,
                    "grad_total": 1359310848,
                    "act_fw_per_module": 207618048,
                    "act_bw_per_module": 482486272,
                    "act_grad_per_module": 276836352,
                    "act_total": 1357261824,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 11.49012612856017,
                    "bw_runtime_per_module": 25.499871304739376,
                },
                {
                    "fqn": "Transformer.layers.3",
                    "param_per_module": 453095424,
                    "grad_per_module": 453095424,
                    "grad_total": 906215424,
                    "act_fw_per_module": 390202368,
                    "act_bw_per_module": 482486272,
                    "act_grad_per_module": 276836352,
                    "act_total": 1772630016,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 18.586358962812866,
                    "bw_runtime_per_module": 42.31444884235838,
                },
                {
                    "fqn": "Transformer.layers.3.attention",
                    "param_per_module": 150994944,
                    "grad_per_module": 150994944,
                    "grad_total": 1208291328,
                    "act_fw_per_module": 107054080,
                    "act_bw_per_module": 224520192,
                    "act_grad_per_module": 100663296,
                    "act_total": 1514663936,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 6.914146061765929,
                    "bw_runtime_per_module": 16.190205050846142,
                },
                {
                    "fqn": "Transformer.layers.3.feed_forward",
                    "param_per_module": 302051328,
                    "grad_per_module": 302051328,
                    "grad_total": 906215424,
                    "act_fw_per_module": 207618048,
                    "act_bw_per_module": 482486272,
                    "act_grad_per_module": 276836352,
                    "act_total": 1772630016,
                    "input_per_module": 25165824,
                    "output_per_module": 25165824,
                    "fw_runtime_per_module": 11.49012612856017,
                    "bw_runtime_per_module": 25.499871304739376,
                },
                {
                    "fqn": "Transformer.output",
                    "param_per_module": 100663296,
                    "grad_per_module": 100663296,
                    "grad_total": 0,
                    "act_fw_per_module": 0,
                    "act_bw_per_module": 2615966720,
                    "act_grad_per_module": 125829120,
                    "act_total": 2695657472,
                    "input_per_module": 25165824,
                    "output_per_module": 67108864,
                    "fw_runtime_per_module": 3.7249330481443956,
                    "bw_runtime_per_module": 8.000336731209435,
                },
            ],
        }

        return mod_info

    def test_fsdp_ilp_case1(self):
        """a standard case with memory budget that is not too tight"""

        fsdp_decisions, exposed_comm_time, peak_mem = fsdp_milp(
            self.g,
            world_size=4,
            comm_params=self.comm_params,
            memory_budget=4.75,
        )
        self.assertEqual(
            fsdp_decisions,
            {
                "Transformer",
                "Transformer.layers.0.attention",
                "Transformer.layers.0.feed_forward",
                "Transformer.layers.1",
                "Transformer.layers.2",
                "Transformer.layers.3",
                "Transformer.output",
            },
        )
        self.assertAlmostEqual(exposed_comm_time / 4.0, 1, delta=0.05)
        self.assertAlmostEqual(peak_mem / 4672410203, 1, delta=0.05)

    def test_fsdp_ilp_case2(self):
        """with user specified fsdp units"""

        fsdp_decisions, exposed_comm_time, peak_mem = fsdp_milp(
            self.g,
            world_size=4,
            comm_params=self.comm_params,
            memory_budget=4.75,
            fsdp_units=[
                "Transformer.layers.0",
                "Transformer.layers.1",
                "Transformer.layers.2",
                "Transformer.layers.3",
                "Transformer.output",
            ],
        )
        self.assertEqual(
            fsdp_decisions,
            {
                "Transformer",
                "Transformer.layers.0",
                "Transformer.layers.1",
                "Transformer.layers.2",
                "Transformer.layers.3",
                "Transformer.output",
            },
        )
        self.assertAlmostEqual(exposed_comm_time / 10.041, 1, delta=0.05)
        self.assertAlmostEqual(peak_mem / 4672311956, 1, delta=0.05)

    def test_fsdp_ilp_case3(self):
        """a case with tight memory budget"""

        fsdp_decisions, exposed_comm_time, peak_mem = fsdp_milp(
            self.g,
            world_size=4,
            comm_params=self.comm_params,
            memory_budget=4,
        )
        self.assertEqual(
            fsdp_decisions,
            {
                "Transformer",
                "Transformer.layers.0.attention",
                "Transformer.layers.0.feed_forward",
                "Transformer.layers.1.attention",
                "Transformer.layers.1.feed_forward",
                "Transformer.layers.2.attention",
                "Transformer.layers.2.feed_forward",
                "Transformer.layers.3.attention",
                "Transformer.layers.3.feed_forward",
                "Transformer.output",
            },
        )
        self.assertAlmostEqual(exposed_comm_time / 4.0029, 1, delta=0.05)
        self.assertAlmostEqual(peak_mem / 4274145874, 1, delta=0.05)

    def test_fsdp_ilp_case4(self):
        """a case with extremely tight memory budget but no feasible solution is possible"""

        fsdp_decisions, exposed_comm_time, peak_mem = fsdp_milp(
            self.g,
            world_size=4,
            comm_params=self.comm_params,
            memory_budget=3.5,
        )
        self.assertEqual(fsdp_decisions, set())
        self.assertEqual(peak_mem, -1)

    def test_fsdp_ilp_case5(self):
        """a case similar to case 1 but with low communication bandwidth"""

        fsdp_decisions, exposed_comm_time, peak_mem = fsdp_milp(
            self.g,
            world_size=4,
            comm_params=self.comm_params_low_bw,
            memory_budget=4.75,
        )
        self.assertEqual(
            fsdp_decisions,
            {
                "Transformer",
                "Transformer.layers.0",
                "Transformer.layers.1",
                "Transformer.layers.2",
                "Transformer.layers.3",
            },
        )
        self.assertAlmostEqual(exposed_comm_time / 303.0618, 1, delta=0.05)
        self.assertAlmostEqual(peak_mem / 4873638548, 1, delta=0.05)


if __name__ == "__main__":
    run_tests()
