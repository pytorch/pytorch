import logging
from copy import deepcopy
from enum import StrEnum
from typing import Dict, List, Optional, Set

import torch
from torch.distributed._tools import MemTracker, RuntimeEstimator, SACEstimator
from torch.distributed._tools.ilp_utils import aggregate_stats, parse_module_info
from torch.distributed._tools.sac_estimator import (
    OPS_TO_ALWAYS_SKIP,
    SACGreedyOrderMeta,
    SACStats,
)
from torch.distributed._tools.sac_ilp import (
    get_optimal_checkpointing_policy_per_module,
    sac_milp,
)
from torch.utils.checkpoint import CheckpointPolicy


# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level to INFO
logger.setLevel(logging.INFO)


class SACAlgorithm(StrEnum):
    """
    Enum for SAC algorithms.

    Attributes:
        GREEDY (str): Greedy algorithm.
        OPTIMAL (str): Optimal algorithm.
    """

    GREEDY = "greedy"
    OPTIMAL = "optimal"


class SACPolicy:
    """
    SAC Policy class.

    Attributes:
        counter (int): Counter for tracking policy output.
        policy_output (List[int]): Policy output as a list of integers (1: save, 0: discard).

    Methods:
        __call__: Evaluates the checkpoint policy for a given function.
    """

    def __init__(self, policy_output: List[int]):
        self.counter = 0
        self.policy_output = policy_output

    def __call__(self, ctx, func, *args, **kwargs) -> CheckpointPolicy:  # type: ignore[no-untyped-def]
        if func in OPS_TO_ALWAYS_SKIP:
            return CheckpointPolicy.MUST_RECOMPUTE
        count = self.counter
        self.counter += 1
        if self.policy_output[count] == 1:
            return CheckpointPolicy.PREFER_SAVE
        else:
            return CheckpointPolicy.MUST_RECOMPUTE


def get_greedy_checkpoint_policy_per_module(
    sac_stats: SACStats, sac_greedy_order_meta: SACGreedyOrderMeta, memory_budget: float
) -> List[int]:
    """
    Compute greedy checkpoint policy per module.

    Args:
        sac_stats (SACStats): SAC statistics.
        sac_greedy_order_meta (SACGreedyOrderMeta): SAC greedy order metadata.
        memory_budget (float): Memory budget as a fraction of total memory (0 <= memory_budget <= 1).

    Returns:
        List[int]: Policy output as a list of integers (1: save, 0: discard).

    Raises:
        ValueError: If memory_budget is not within the valid range (0 <= memory_budget <= 1).
    """

    # Validate memory budget range
    if not (0 <= memory_budget <= 1):
        raise ValueError(
            f"`memory_budget` must be a float between 0 and 1. Got {memory_budget}."
        )

    # Initialize policy output with all ops saved
    policy_output = [1 for _ in range(len(sac_stats.memory))]

    sac_memory = sum(sac_stats.memory)
    sac_memory_budget = memory_budget * sac_memory

    stored_ops, recomputed_ops, inplace_op_groups, random_inplace_ops, msps_meta = (
        sac_greedy_order_meta.stored_ops,
        sac_greedy_order_meta.recomputed_ops,
        sac_greedy_order_meta.inplace_op_groups,
        sac_greedy_order_meta.random_inplace_ops,
        sac_greedy_order_meta.msps_meta,
    )

    stored_indices: Set[int] = set()
    for s_idx in stored_ops:
        stored_indices.add(s_idx)
        if s_idx in inplace_op_groups:
            stored_indices.update(inplace_op_groups[s_idx])
        if s_idx in random_inplace_ops:
            stored_indices.update(random_inplace_ops)

    saved_memory = sum(sac_stats.memory[op_idx] for op_idx in stored_indices)

    # Check if saved ops exceed memory budget
    if saved_memory > sac_memory_budget:
        logger.error(
            "Ops that need to be saved already exceed the given memory budget.\n"
            "Ops: %s\n"
            "Budget: %s Saved Ops Memory: %s",
            [sac_stats.func_names[i] for i in stored_ops],
            sac_memory_budget,
            saved_memory,
        )
        return [
            1 if idx in stored_indices else 0 for idx in range(len(sac_stats.memory))
        ]

    recompute_indices = set(recomputed_ops)
    discarded_memory = sum(sac_stats.memory[i] for i in recompute_indices)

    sac_memory_budget -= saved_memory
    msps_meta = deepcopy(msps_meta)

    # Discard ops until memory budget is met
    while (sac_memory - discarded_memory) > sac_memory_budget:
        try:
            msps = msps_meta.pop(0)
        except IndexError:
            logger.error("Exhausted the Ops to recompute, cannot satisfy budget.")
            return [
                1 if idx in stored_indices else 0
                for idx in range(len(sac_stats.memory))
            ]
        recompute_indices.add(msps.op_idx)
        if msps.op_idx in random_inplace_ops:
            recompute_indices.update(random_inplace_ops)
        if inplace_op_group := inplace_op_groups.get(msps.op_idx, None):
            recompute_indices.update(inplace_op_group)
        discarded_memory += msps.memory

    # Update policy output with recompute ops
    for i in recompute_indices:
        policy_output[i] = 0

    return policy_output


def get_auto_sac_policies(
    model: torch.nn.Module,
    sac_estimator: SACEstimator,
    mem_tracker: MemTracker,
    runtime_estimator: RuntimeEstimator,
    dev: torch.device,
    memory_budget: float,
    sac_algo: SACAlgorithm = SACAlgorithm.GREEDY,
    shard_degree: int = 1,
    ac_units: Optional[Set[str]] = None,
    fsdp_units: Optional[Set[str]] = None,
) -> Dict[str, SACPolicy]:
    """
    Compute auto-SAC policies for a given model.

    Args:
        model (torch.nn.Module): Input model.
        sac_estimator (SACEstimator): `SACEstimator` instance.
        mem_tracker (MemTracker): `MemTracker` instance.
        runtime_estimator (RuntimeEstimator): `RuntimeEstimator` instance.
        dev (torch.device): Device for which stats were captured.
        memory_budget (float): Memory budget in GiB.
        sac_algo (SACAlgorithm, optional): `SACAlgorithm`. Defaults to `SACAlgorithm.GREEDY`.
        shard_degree: number of GPUs across which the model is sharded. In the case of FSDP,
            shard_degree will be used to compute the amount of parameter, gradient and optimizer
            memory on each rank. Defaults to 1.
        ac_units: a set of user-specified AC unit FQNs.
        fsdp_units: a set of FSDP units. AC units cannot be supermodules of FSDP unit FQNs.

    Returns:
        Dict[str, SACPolicy]: Dictionary of Module FQN to `SACPolicy` for each module.

    Raises:
        ValueError: If sac_algo is not a valid SAC algorithm.
    """

    # Aggregate model statistics
    mod_info = aggregate_stats(
        model, mem_tracker, runtime_estimator, sac_estimator, dev
    )
    # Parse module information into a graph
    graph = parse_module_info(mod_info)
    # Solve SAC MILP problem
    ac_decisions, _, _ = sac_milp(
        graph, memory_budget, shard_degree, ac_units, fsdp_units
    )
    sac_policies: Dict[str, SACPolicy] = {}

    # Compute SAC policies for each module
    for mod_name, discard_ratio in ac_decisions.items():
        sac_stats = sac_estimator.sac_mod_stats[mod_name]
        budget = 1 - discard_ratio
        sac_greedy_order_meta = sac_estimator.sac_mod_greedy_order_meta[mod_name]
        if sac_algo == SACAlgorithm.GREEDY:
            policy_output = get_greedy_checkpoint_policy_per_module(
                sac_stats, sac_greedy_order_meta, budget
            )
        else:
            policy_output = get_optimal_checkpointing_policy_per_module(
                sac_stats, budget
            )
        # Create and store SAC policy in dictionary
        sac_policy = SACPolicy(policy_output)
        sac_policies[mod_name] = sac_policy

    return sac_policies
