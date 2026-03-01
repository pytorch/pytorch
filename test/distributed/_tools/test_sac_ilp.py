# Owner(s): ["oncall: distributed"]
import copy
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.ilp_utils import (
    aggregate_stats,
    get_peak_memory_runtime_baseline,
    ModuleInfo,
    parse_module_info,
)
from torch.distributed._tools.mem_tracker import _ModState, MemTracker
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.distributed._tools.sac_estimator import SACEstimator, SACStats
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


# sac_ilp depends on the pulp package which may not be installed
# See: https://github.com/pytorch/pytorch/issues/162453
HAS_PULP = True
try:
    from torch.distributed._tools.sac_ilp import (
        get_optimal_checkpointing_policy_per_module,
        sac_milp,
    )
except ImportError:
    HAS_PULP = False
    get_optimal_checkpointing_policy_per_module = None  # type: ignore[assignment, misc]
    sac_milp = None  # type: ignore[assignment]


class TestSACILP(TestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.cuda.current_device()
        self.estimate_mode = "operator-level-cost-model"

    def _init_model_input_optimizer(
        self,
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.Tensor]:
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
        return (model, optimizer, inp)

    def _run_and_get_memTracker(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        inp: torch.Tensor,
    ) -> MemTracker:
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optimizer)
        with mem_tracker as mt:
            for iter_idx in range(2):  # running twice to initialize optimizer
                output = model(inp)
                output.sum().backward()
                if iter_idx == 1:
                    last_snapshot = mt.get_tracker_snapshot("current")
                optimizer.step()
                optimizer.zero_grad()
                if iter_idx == 0:
                    mt.reset_mod_stats()
        if last_snapshot is None:
            raise AssertionError("Expected last_snapshot to not be None")
        for mod_stats in mem_tracker.memory_tracking.values():
            # postprocessing due to the fact that for ModTracker, the post backward hook
            # is not being called for modules whose inputs don't require gradients
            # TODO: fix this in ModTracker and ensure it does not lead to any perf regression
            if _ModState.POST_BW not in mod_stats.snapshots:
                mod_stats.snapshots.setdefault(_ModState.POST_BW, []).append(
                    copy.deepcopy(last_snapshot)
                )
        return mem_tracker

    def _run_and_get_runtime_estimator(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        inp: torch.Tensor,
    ) -> RuntimeEstimator:
        def _run_one_step() -> None:
            output = model(inp)
            output.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        # Initializing optimizer states and warm-up
        _run_one_step()

        runtime_estimator = RuntimeEstimator()
        with runtime_estimator(estimate_mode_type=self.estimate_mode):
            _run_one_step()  # We use only one iteration for estimation
        return runtime_estimator

    def _run_and_get_sac_estimator(
        self,
        model: torch.nn.Module,
        inp: torch.Tensor,
    ) -> SACEstimator:
        sac_estimator = SACEstimator()
        with sac_estimator(estimate_mode_type=self.estimate_mode):
            loss = model(inp).sum()
        loss.backward()
        return sac_estimator

    def _collect_module_info_with_fake_tensor_mode(self) -> ModuleInfo:
        with FakeTensorMode():
            model, optimizer, inp = self._init_model_input_optimizer()
            mem_tracker = self._run_and_get_memTracker(model, optimizer, inp)
            runtime_estimator = self._run_and_get_runtime_estimator(
                model, optimizer, inp
            )
            sac_estimator = self._run_and_get_sac_estimator(model, inp)
            mod_info = aggregate_stats(
                model,
                mem_tracker,
                runtime_estimator,
                sac_estimator,
                torch.device(self.device),
            )
        return mod_info

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not HAS_PULP, "pulp package not installed")
    def test_sac_ilp_case1(self):
        """
        This is a case where the memory budget is either binding or too tight,
        meaning that with some AC, the model can fit into GPU memory.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)

        peak_mem, compute_time = get_peak_memory_runtime_baseline(g)

        # Validate baseline peak memory is reasonable (>2GB for this model)
        self.assertGreater(peak_mem, 2e9)
        # Validate baseline compute time is reasonable (>50ms for this model)
        self.assertGreater(compute_time, 50)

        ac_decisions, recomputation_time, _ = sac_milp(
            g, memory_budget=1.6, world_size=4
        )

        # The solution should AC all four transformer layers to meet the memory budget.
        # This is the key validation: the ILP solver should identify that all 4 transformer
        # layers need activation checkpointing to fit within the 1.6GB memory budget.
        modules_to_ac = set(ac_decisions.keys())
        self.assertEqual(
            modules_to_ac,
            {"Transformer.layers." + str(i) for i in range(4)},  # n_layers=4
        )

        # Validate discard ratios are reasonable (between 0.4 and 0.9).
        # The exact values depend on the device's FLOPS and memory bandwidth which are
        # architecture-specific (see torch._inductor.analysis.device_info).
        # Different devices will produce different but valid ratios.
        sorted_discard_ratio = sorted(ac_decisions.values())
        for ratio in sorted_discard_ratio:
            self.assertGreater(ratio, 0.4)
            self.assertLess(ratio, 0.9)

        # Validate sum of discard ratios is reasonable (between 1.8 and 2.8).
        # This ensures the solver is making balanced decisions across all layers.
        ratio_sum = sum(sorted_discard_ratio)
        self.assertGreater(ratio_sum, 1.8)
        self.assertLess(ratio_sum, 2.8)

        # Validate recomputation time is positive but less than compute time.
        # The ILP solver should only recommend AC if it reduces peak memory without
        # adding excessive recomputation overhead.
        self.assertGreater(recomputation_time, 0)
        self.assertLess(recomputation_time, compute_time)
        # Recomputation overhead should be reasonable (<20% of compute time)
        self.assertLess(recomputation_time / compute_time, 0.20)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not HAS_PULP, "pulp package not installed")
    def test_sac_ilp_case2(self):
        """
        This is a case where the memory budget is not binding, meaning that no
        AC is needed to fit the model into memory.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)
        ac_decisions, recomputation_time, peak_mem = sac_milp(
            g, memory_budget=2.4, world_size=4
        )
        self.assertDictEqual(ac_decisions, {})
        self.assertEqual(recomputation_time, 0)
        self.assertGreater(peak_mem, 1)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not HAS_PULP, "pulp package not installed")
    def test_sac_ilp_case3(self):
        """
        This is a case where the memory budget is too tight, meaning that even with
        aggressive AC, the model cannot fit into memory.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)
        ac_decisions, recomputation_time, peak_mem = sac_milp(
            g, memory_budget=0.8, world_size=4
        )
        self.assertEqual(ac_decisions, {})
        self.assertEqual(recomputation_time, 0)
        self.assertEqual(peak_mem, -1)


class TestOptimalCheckpointingPolicy(TestCase):
    # tests are adapted from tests in xformers
    # https://github.com/facebookresearch/xformers/blob/c6c0ac31f1b08542a0bc27278c6ed10f825f6963/tests/test_checkpoint.py#L222
    def setUp(self):
        super().setUp()
        data = [
            ("aten.copy_", 5, 0),
            ("aten.add", 5, 100),
            ("aten.div", 8, 100),
            ("aten.mm", 15, 120),
            ("aten.native_dropout", 15, 0),
            ("aten.linear", 9, 100),
            ("aten.t", 1, 0),
            ("aten.relu_", 5, 0),
        ]
        self.sac_stats = SACStats(
            func_names=[x[0] for x in data],
            runtimes=[x[1] for x in data],
            memory=[x[2] for x in data],
            view_like_ops=[6],
            rand_ops=[4],
            saved_autograd_ops=[],  # not needed for SAC decisions
            inplace_ops=[(0, 0), (7, 5)],
            force_store_random=False,
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not HAS_PULP, "pulp package not installed")
    def test_get_optimial_checkpointing_policy_per_module(self):
        for memory_budget, optimal_soln in [
            (0, [1, 0, 0, 0, 1, 0, 0, 0]),
            (100 / 420, [1, 0, 0, 0, 1, 1, 0, 1]),
            (120 / 420, [1, 0, 0, 1, 1, 0, 0, 0]),
            (200 / 420, [1, 0, 1, 0, 1, 1, 0, 1]),
            (220 / 420, [1, 0, 0, 1, 1, 1, 0, 1]),
            (320 / 420, [1, 0, 1, 1, 1, 1, 0, 1]),
            (420 / 420, [1, 1, 1, 1, 1, 1, 0, 1]),
        ]:
            soln = get_optimal_checkpointing_policy_per_module(
                sac_stats=self.sac_stats, memory_budget=memory_budget
            )
            self.assertEqual(optimal_soln, soln)


if __name__ == "__main__":
    run_tests()
