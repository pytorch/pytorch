# Owner(s): ["module: unknown"]
import unittest
from functools import partial
from typing import Callable, List, Optional, Set, Tuple

import torch
from torch import nn, optim
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tools.auto_sac import (
    apply_auto_sac_policies,
    get_auto_sac_policies,
    get_greedy_checkpointing_policy_per_module,
    SACAlgorithm,
)
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
from torch.distributed._tools.ilp_utils import (
    aggregate_stats,
    collect_stats,
    parse_module_info,
)
from torch.distributed._tools.mem_tracker import MemTracker
from torch.distributed._tools.sac_estimator import SACEstimator, SACStats
from torch.distributed._tools.sac_ilp import (
    get_optimal_checkpointing_policy_per_module,
    sac_milp,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


def _init_model_input_optimizer(
    dev: torch.device,
) -> Tuple[Callable, List[nn.Module], List[optim.Optimizer], torch.Tensor]:
    bsz = 8
    model_args = ModelArgs(
        n_layers=4,
        n_heads=12,
        vocab_size=8192,
        max_seq_len=1024,
        dim=768,
        dropout_p=0.1,
    )
    with torch.device(dev):
        model = Transformer(model_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
    inp = torch.randint(
        0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=dev
    )

    def train_step(
        models: List[nn.Module], optimizers: List[optim.Optimizer], inputs: torch.Tensor
    ) -> None:
        model = models[0]
        optimizer = optimizers[0]
        loss = model(inputs).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return (train_step, [model], [optimizer], inp)


def _run_and_get_memtracker(
    train_step: Callable,
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    inp: torch.Tensor,
) -> MemTracker:
    train_step(models, optimizers, inp)
    mem_tracker = MemTracker()
    mem_tracker.track_external(*models, *optimizers, inp)
    with mem_tracker:
        train_step(models, optimizers, inp)
    return mem_tracker


def _run_and_get_fsdp_memtracker(
    train_step: Callable,
    models: List[nn.Module],
    optimizers: List[optim.Optimizer],
    inp: torch.Tensor,
) -> FSDPMemTracker:
    fsdp_memtracker = FSDPMemTracker(models[0], optimizers[0])
    fsdp_memtracker.track_inputs((inp,))
    with fsdp_memtracker as fmt:
        for iter_idx in range(2):
            train_step(models, optimizers, inp)
            if iter_idx == 0:
                fmt.reset_mod_stats()
    if torch.distributed.group.WORLD:
        torch.distributed.destroy_process_group()
    return fsdp_memtracker


def _init_distributed(world_size: int) -> DeviceMesh:
    store = FakeStore()
    torch.distributed.init_process_group(
        "fake", rank=0, world_size=world_size, store=store
    )
    mesh = DeviceMesh("cuda", torch.arange(0, world_size))
    return mesh


def _apply_fsdp(model: torch.nn.Module, mesh: DeviceMesh):
    fully_shard_fn = partial(fully_shard, mesh=mesh)
    for layer_id, transformer_block in enumerate(model.layers):
        reshard_after_forward = int(layer_id) < len(model.layers) - 1
        fully_shard_fn(transformer_block, reshard_after_forward=reshard_after_forward)
    fully_shard_fn(model)


class TestSACILP(TestCase):
    """
    Unit tests for Selective Activation Checkpointing (SAC) optimization using Integer Linear Programming (ILP).

    This class tests various scenarios to ensure SAC ILP solutions are valid under different memory budgets,
    shard degrees, and FSDP unit configurations.

    Attributes:
        device (torch.device): The GPU device used for testing.
        estimate_mode (str): Mode for estimating operator-level costs.
        gpu_type (str): Type of GPU used for testing (e.g., "H100_SXM_80GB").
    """

    def setUp(self):
        super().setUp()
        self.device = torch.device(type="cuda", index=torch.cuda.current_device())
        self.estimate_mode = "operator-level-cost-model"
        self.gpu_type = "H100_SXM_80GB"

    def _test_sac_ilp(
        self,
        memory_budget: float,
        fsdp_units: Optional[Set[str]] = None,
        shard_degree: int = 1,
    ):
        """
        Internal helper function to test SAC ILP solutions.

        Args:
            memory_budget (float): Memory budget in GiB.
            fsdp_units (Optional[Set[str]]): FSDP unit Fully Qualified Names (FQNs). Defaults to None.
            shard_degree (int): Number of GPUs used for sharding. Defaults to 1.

        Returns:
            Tuple:
                - ac_decisions (Dict[str, float]): Activation discard ratios per module.
                - peak_mem (float): Peak memory usage of the model without SAC.
                - expected_peak_mem (float): Expected peak memory after applying SAC.
                - compute_time (float): Total compute time without SAC.
                - recomputation_time (float): Total recomputation time with SAC.
        """
        with FakeTensorMode():
            train_step, models, optimizers, inp = _init_model_input_optimizer(
                self.device
            )
            mem_tracker, runtime_estimator, sac_estimator = collect_stats(
                train_step,
                models,
                optimizers,
                inp,
                runtime_kwargs={
                    "estimate_mode": self.estimate_mode,
                    "gpu_type": self.gpu_type,
                },
            )

        mod_info = aggregate_stats(
            models,
            optimizers,
            mem_tracker,
            runtime_estimator,
            sac_estimator,
            self.device,
        )
        g = parse_module_info(mod_info)
        peak_mem = mem_tracker.get_tracker_snapshot("peak")[self.device]["Total"]
        compute_time = runtime_estimator.total_compute_time
        ac_decisions, recomputation_time, expected_peak_mem = sac_milp(
            g,
            memory_budget=memory_budget,
            shard_degree=shard_degree,
            fsdp_units=fsdp_units,
        )
        return (
            ac_decisions,
            peak_mem,
            expected_peak_mem,
            compute_time,
            recomputation_time,
        )

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sac_ilp_case1(self):
        """
        This is a case where the memory budget is neither binding nor too tight,
        meaning that with some AC, the model can fit into GPU memory.
        Validates:
        - Modules selected for AC.
        - Discard ratios for each module.
        - Memory and computation time metrics.
        """
        (
            ac_decisions,
            _,
            expected_peak_mem,
            compute_time,
            recomputation_time,
        ) = self._test_sac_ilp(memory_budget=2.0)
        modules_to_ac = set(ac_decisions.keys())
        sorted_discard_ratio = sorted(ac_decisions.values())
        self.assertEqual(
            modules_to_ac,
            {"Transformer.layers." + str(i) for i in range(4)},  # n_layers=4
        )
        self.assertAlmostEqual(sorted_discard_ratio[0], 0.2447, delta=0.05)
        self.assertAlmostEqual(sorted_discard_ratio[1], 0.4979, delta=0.05)
        self.assertAlmostEqual(sorted_discard_ratio[2], 0.4979, delta=0.05)
        self.assertAlmostEqual(sum(sorted_discard_ratio), 1.7384, delta=0.05)

        self.assertAlmostEqual(
            (recomputation_time / compute_time) / (1.31 / 42.016), 1, delta=0.1
        )
        GiB = 2**30
        self.assertLessEqual(expected_peak_mem / GiB, 2.01)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sac_ilp_case2(self):
        """
        This is a case where the memory budget is neither binding nor too tight,
        meaning that with some AC, the model can fit into GPU memory.
        FSDP units have been pre-determined.
        Validates:
        - Modules selected for AC.
        - Discard ratios for each module.
        - Memory and computation time metrics.
        """
        (
            ac_decisions,
            _,
            expected_peak_mem,
            compute_time,
            recomputation_time,
        ) = self._test_sac_ilp(
            memory_budget=1.6,
            fsdp_units={"Transformer.layers." + str(i) for i in range(4)}
            | {"Transformer"},
            shard_degree=4,
        )
        modules_to_ac = set(ac_decisions.keys())
        sorted_discard_ratio = sorted(ac_decisions.values())
        self.assertEqual(
            modules_to_ac,
            {"Transformer.layers." + str(i) for i in range(4)},  # n_layers=4
        )
        self.assertAlmostEqual(sorted_discard_ratio[0], 0.4979, delta=0.05)
        self.assertAlmostEqual(sorted_discard_ratio[1], 0.4979, delta=0.05)
        self.assertAlmostEqual(sorted_discard_ratio[2], 0.4979, delta=0.05)
        self.assertAlmostEqual(sum(sorted_discard_ratio), 2.1946, delta=0.05)

        self.assertAlmostEqual(
            (recomputation_time / compute_time) / (2.37 / 42.016), 1, delta=0.1
        )
        GiB = 2**30
        self.assertLessEqual(expected_peak_mem / GiB, 1.61)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sac_ilp_case3(self):
        """
        This is a case where the memory budget is not binding, meaning that no
        AC is needed to fit the model into memory.
        FSDP units have been pre-determined.
        Validates:
        - No modules are selected for AC.
        - Peak memory remains under the budget.
        - Recomputation time is zero.
        """
        ac_decisions, _, expected_peak_mem, _, recomputation_time = self._test_sac_ilp(
            memory_budget=2.7,
            fsdp_units={"Transformer.layers." + str(i) for i in range(4)}
            | {"Transformer"},
            shard_degree=4,
        )
        self.assertDictEqual(ac_decisions, {})
        self.assertEqual(recomputation_time, 0)
        GiB = 2**30
        self.assertLessEqual(expected_peak_mem / GiB, 2.71)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sac_ilp_case4(self):
        """
        This is a case where the memory budget is too tight, meaning that even with
        aggressive AC, the model cannot fit into memory.
        FSDP units have been pre-determined.
        Validates:
        - No valid SAC solution is found.
        - Peak memory is set to -1, indicating failure.
        - Recomputation time is zero.
        """
        ac_decisions, _, expected_peak_mem, _, recomputation_time = self._test_sac_ilp(
            memory_budget=0.8,
            fsdp_units={"Transformer.layers." + str(i) for i in range(4)}
            | {"Transformer"},
            shard_degree=4,
        )
        self.assertEqual(ac_decisions, {})
        self.assertEqual(recomputation_time, 0)
        self.assertEqual(expected_peak_mem, -1)


class TestCheckpointingPolicy(TestCase):
    """
    Unit tests for Selective Activation Checkpointing (SAC) checkpointing policies.

    This class validates the behavior of optimal and greedy checkpointing policies
    under different memory budgets using pre-defined SAC statistics.

    Attributes:
        sac_stats (SACStats): Predefined statistics for SAC, including operator
            runtimes, memory usage, and operator relationships.
        greedy_order_meta: Metadata for evaluating greedy checkpointing policies.
    """

    # tests are adpated from tests in xformers
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
        self.greedy_order_meta = SACEstimator._get_greedy_order_meta(self.sac_stats)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
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

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_get_greedy_checkpointing_policy_per_module(self):
        for memory_budget, optimal_soln in [
            (0, [1, 0, 0, 0, 1, 0, 0, 0]),
            (100 / 420, [1, 0, 0, 0, 1, 1, 0, 1]),
            (120 / 420, [1, 0, 0, 0, 1, 1, 0, 1]),
            (200 / 420, [1, 0, 0, 0, 1, 1, 0, 1]),
            (220 / 420, [1, 0, 0, 1, 1, 1, 0, 1]),
            (320 / 420, [1, 0, 1, 1, 1, 1, 0, 1]),
            (420 / 420, [1, 1, 1, 1, 1, 1, 0, 1]),
        ]:
            soln = get_greedy_checkpointing_policy_per_module(
                sac_stats=self.sac_stats,
                sac_greedy_order_meta=self.greedy_order_meta,
                memory_budget=memory_budget,
            )
            self.assertEqual(optimal_soln, soln)


class TestAutoSAC(TestCase):
    """
    Unit tests for the Auto Selective Activation Checkpointing (Auto-SAC) mechanism.

    This class validates the behavior of Auto-SAC under different memory budgets,
    shard degrees, and SAC algorithms (Optimal and Greedy).

    Attributes:
        device (torch.device): The GPU device used for testing.
        estimate_mode (str): Mode for estimating operator-level costs.
        gpu_type (str): Type of GPU used for testing (e.g., "H100_SXM_80GB").
        fake_mode (FakeTensorMode): Fake tensor mode used for efficient memory simulation.
    """

    def setUp(self):
        super().setUp()
        self.device = torch.device(type="cuda", index=torch.cuda.current_device())
        self.estimate_mode = "operator-level-cost-model"
        self.gpu_type = "H100_SXM_80GB"
        self.fake_mode = FakeTensorMode()

    def _test_auto_sac(
        self,
        memory_budget: float,
        sac_algo: SACAlgorithm,
        fsdp_units: Optional[Set[str]] = None,
        shard_degree: int = 1,
    ) -> Tuple[int, int]:
        """
        Internal helper function to test Auto-SAC behavior.

        Args:
            memory_budget (float): The memory budget in GiB.
            sac_algo (SACAlgorithm): The SAC algorithm to use (Optimal or Greedy).
            fsdp_units (Optional[Set[str]], optional): Fully Sharded Data Parallel
                (FSDP) unit FQNs. Defaults to None.
            shard_degree (int, optional): Number of GPUs used for sharding. Defaults to 1.

        Returns:
            Tuple[int, int]:
                - Returns a tuple of the SAC-estimated peak memory
                  and actual peak memory after applying Auto-SAC.
        """
        with self.fake_mode:
            train_step, models, optimizers, inp = _init_model_input_optimizer(
                self.device
            )

            auto_sac_result = get_auto_sac_policies(
                train_step,
                models,
                optimizers,
                inp,
                self.device,
                memory_budget=memory_budget,
                sac_algo=sac_algo,
                shard_degree=shard_degree,
                fsdp_units=fsdp_units,
                runtime_kwargs={
                    "estimate_mode": self.estimate_mode,
                    "gpu_type": self.gpu_type,
                },
            )
            for model in models:
                apply_auto_sac_policies(
                    model, auto_sac_result.sac_policies, preserve_rng_state=False
                )

        if shard_degree > 1:
            mesh = _init_distributed(shard_degree)
            with self.fake_mode:
                del optimizers[0]
                _apply_fsdp(model, mesh)
                optimizers.append(
                    torch.optim.Adam(models[0].parameters(), lr=1e-2, foreach=True)
                )
                fsdp_mem_tracker = _run_and_get_fsdp_memtracker(
                    train_step, models, optimizers, inp
                )
                peak_mem_after = fsdp_mem_tracker.get_tracker_snapshot("peak")[
                    self.device
                ]["Total"]
        else:
            with self.fake_mode:
                mem_tracker_sac = _run_and_get_memtracker(
                    train_step, models, optimizers, inp
                )
                peak_mem_after = mem_tracker_sac.get_tracker_snapshot("peak")[
                    self.device
                ]["Total"]
        return (auto_sac_result.peak_mem, peak_mem_after)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_auto_sac_optimal(self):
        """
        Tests Auto-SAC using the Optimal algorithm with a sufficient memory budget.

        Validates:
            - SAC-estimated peak memory is within the memory budget.
            - Actual peak memory does not exceed the SAC estimate.
            - Actual peak memory fits within the memory budget.

        Memory Budget: 2.0 GiB
        SAC Algorithm: Optimal
        """
        memory_budget = 2.0
        delta = 0.01
        sac_algo = SACAlgorithm.OPTIMAL
        GiB = 2**30

        sac_est_peak_mem, actual_peak_mem = self._test_auto_sac(memory_budget, sac_algo)
        self.assertLessEqual(sac_est_peak_mem / GiB, memory_budget + delta)
        self.assertLessEqual(actual_peak_mem, sac_est_peak_mem)
        self.assertLessEqual(actual_peak_mem / GiB, memory_budget)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_auto_sac_optimal_fsdp(self):
        """
        Tests Auto-SAC with the Optimal algorithm in an FSDP setup.

        Validates:
            - SAC-estimated peak memory is within the memory budget.
            - Actual peak memory does not exceed the SAC estimate.
            - Actual peak memory fits within the memory budget.

        Memory Budget: 1.6 GiB
        SAC Algorithm: Optimal
        Shard Degree: 4
        FSDP Units: Transformer layers (0-3)
        """
        memory_budget = 1.6
        sac_algo = SACAlgorithm.OPTIMAL
        GiB = 2**30
        fsdp_units = {"Transformer.layers." + str(i) for i in range(4)} | {
            "Transformer"
        }
        shard_degree = 4
        delta = 0.01

        sac_est_peak_mem, actual_peak_mem = self._test_auto_sac(
            memory_budget, sac_algo, fsdp_units=fsdp_units, shard_degree=shard_degree
        )
        self.assertLessEqual(sac_est_peak_mem / GiB, memory_budget + delta)
        self.assertLessEqual(actual_peak_mem, sac_est_peak_mem)
        self.assertLessEqual(actual_peak_mem / GiB, memory_budget)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_auto_sac_greedy(self):
        """
        Tests Auto-SAC using the Greedy algorithm with a sufficient memory budget.

        Validates:
            - SAC-estimated peak memory is within the memory budget.
            - Actual peak memory does not exceed the SAC estimate.
            - Actual peak memory fits within the memory budget.

        Memory Budget: 2.0 GiB
        SAC Algorithm: Greedy
        """
        memory_budget = 2.0
        sac_algo = SACAlgorithm.GREEDY
        GiB = 2**30
        delta = 0.01

        sac_est_peak_mem, actual_peak_mem = self._test_auto_sac(memory_budget, sac_algo)
        self.assertLessEqual(sac_est_peak_mem / GiB, memory_budget + delta)
        self.assertLessEqual(actual_peak_mem, sac_est_peak_mem)
        self.assertLessEqual(actual_peak_mem / GiB, memory_budget)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_auto_sac_greedy_fsdp(self):
        """
        Tests Auto-SAC with the Greedy algorithm in an FSDP setup.

        Validates:
            - SAC-estimated peak memory is within the memory budget.
            - Actual peak memory does not exceed the SAC estimate.
            - Actual peak memory fits within the memory budget.

        Memory Budget: 1.6 GiB
        SAC Algorithm: Greedy
        Shard Degree: 4
        FSDP Units: Transformer layers (0-3)
        """
        memory_budget = 1.6
        sac_algo = SACAlgorithm.GREEDY
        GiB = 2**30
        fsdp_units = {"Transformer.layers." + str(i) for i in range(4)} | {
            "Transformer"
        }
        shard_degree = 4
        delta = 0.01

        sac_est_peak_mem, actual_peak_mem = self._test_auto_sac(
            memory_budget, sac_algo, fsdp_units=fsdp_units, shard_degree=shard_degree
        )
        self.assertLessEqual(sac_est_peak_mem / GiB, memory_budget + delta)
        self.assertLessEqual(actual_peak_mem, sac_est_peak_mem)
        self.assertLessEqual(actual_peak_mem / GiB, memory_budget)


if __name__ == "__main__":
    run_tests()
