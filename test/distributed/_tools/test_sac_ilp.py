# Owner(s): ["module: unknown"]
import unittest
from typing import Tuple

import torch
from torch._inductor.test_case import TestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.collect_stats import aggregate_stats
from torch.distributed._tools.ilp_utils import (
    get_peak_memory_runtime_baseline,
    parse_module_info,
)
from torch.distributed._tools.sac_ilp import sac_milp
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)
from torch.testing._internal.inductor_utils import HAS_GPU


def loss_fn(logits: torch.Tensor, targets: torch.Tensor):
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
    )


class TestSACILP(TestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device(torch.cuda.current_device())

    def _init_model_and_args(
        self,
    ) -> Tuple[
        torch.nn.Module, torch.optim.Optimizer, Tuple[torch.Tensor, torch.Tensor]
    ]:
        bsz = 8
        model_args = ModelArgs(
            n_layers=4,
            n_heads=12,
            vocab_size=8192,
            max_seq_len=1024,
            dim=768,
            dropout_p=0.1,
        )

        with torch.device(self.device):
            model = Transformer(model_args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        inp = torch.randint(
            0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=self.device
        )
        tgt = torch.randint(
            0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=self.device
        )
        return (model, optimizer, (inp, tgt))

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_collect_stats_and_baseline_estimation(self):
        with FakeTensorMode():
            model, optimizer, (inp, tgt) = self._init_model_and_args()
            mod_info = aggregate_stats(
                model, optimizer, (inp, tgt), loss_fn, self.device
            )
            g = parse_module_info(mod_info)
            peak_mem, compute_time = get_peak_memory_runtime_baseline(g)
            self.assertAlmostEqual(peak_mem / 2183395840, 1, delta=0.1)
            self.assertAlmostEqual(compute_time / 101.82, 1, delta=0.1)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sac_ilp_case1(self):
        with FakeTensorMode():
            model, optimizer, (inp, tgt) = self._init_model_and_args()
            mod_info = aggregate_stats(
                model, optimizer, (inp, tgt), loss_fn, self.device
            )
            g = parse_module_info(mod_info)
            sol = sac_milp(g, memory_budget=1.5, world_size=4)
            self.assertDictEqual(
                sol.ac_decisions,
                {
                    "Transformer.layers.0": 0.4791,
                    "Transformer.layers.1": 0.4791,
                    "Transformer.layers.2": 0.3321,
                    "Transformer.layers.3": 0.4791,
                },
            )
            self.assertAlmostEqual(sol.recomputation_time, 1.08, delta=0.1)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sac_ilp_case2(self):
        """
        This is a case where the memory budget is not binding, meaning that no
        AC is needed to fit the model into memory.
        """
        with FakeTensorMode():
            model, optimizer, (inp, tgt) = self._init_model_and_args()
            mod_info = aggregate_stats(
                model, optimizer, (inp, tgt), loss_fn, self.device
            )
            g = parse_module_info(mod_info)
            sol = sac_milp(g, memory_budget=5, world_size=4)
            print(sol)
            self.assertDictEqual(sol.ac_decisions, {})
            self.assertEqual(sol.recomputation_time, 0)
            self.assertGreater(sol.peak_mem, 1)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sac_ilp_case3(self):
        """
        This is a case where the memory budget is binding, meaning that even with
        aggressive AC, the model cannot fit into memory.
        """
        with FakeTensorMode():
            model, optimizer, (inp, tgt) = self._init_model_and_args()
            mod_info = aggregate_stats(
                model, optimizer, (inp, tgt), loss_fn, self.device
            )
            g = parse_module_info(mod_info)
            sol = sac_milp(g, memory_budget=0.5, world_size=4)
            self.assertEqual(sol.ac_decisions, {})
            self.assertEqual(sol.recomputation_time, 0)
            self.assertEqual(sol.peak_mem, -1)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
