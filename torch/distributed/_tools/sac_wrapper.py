from copy import deepcopy
from typing import  Dict, List, Set, Optional
from enum import StrEnum
import logging
import torch
from torch.distributed._tools import MemTracker, RuntimeEstimator, SACEstimator
from torch.distributed._tools.sac_estimator import OPS_TO_ALWAYS_SKIP, SACStats, SACGreedyOrderMeta, SACTradeOffStats
from torch.distributed._tools.ilp_utils import aggregate_stats, parse_module_info
from torch.distributed._tools.sac_ilp import get_optimal_checkpointing_policy_per_module, SACDecision, sac_milp
from torch.utils.checkpoint import CheckpointPolicy

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level to INFO
logger.setLevel(logging.INFO)

class SACAlgorithm(StrEnum):
    GREEDY = "greedy"
    DP = "dp"
    OPTIMAL = "optimal"

class SACPolicy:
    def __init__(self, policy_output: List[int]):
        self.counter = 0
        self.policy_output = policy_output

    def __call__(self, ctx, func, *args, **kwargs) -> bool:
        if func in OPS_TO_ALWAYS_SKIP:
            return CheckpointPolicy.MUST_RECOMPUTE
        count = self.counter
        self.counter += 1
        if self.policy_output[count] == 1:
            return CheckpointPolicy.MUST_SAVE
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE


def get_greedy_checkpoint_policy_per_module(sac_stats: SACStats, sac_greedy_order_meta: SACGreedyOrderMeta, memory_budget: float) -> List[int]:
    policy_output = [1 for _ in range(len(sac_stats.memory))]
    sac_memory = sum(sac_stats.memory)
    sac_memory_budget = memory_budget * sac_memory
    saved_memory = sum(sac_stats.memory[i] for i in sac_greedy_order_meta.stored_ops)
    if saved_memory > sac_memory_budget:
        logger.error(
            f"Ops that need to be saved already exceed the given memory budget.\n"
            f"Ops: {[sac_stats.func_names[i] for i in sac_greedy_order_meta.stored_ops]}\n"
            f"Budget: {sac_memory_budget} Saved Ops Memory: {saved_memory}"
        )
    discard_op_indices = set(sac_greedy_order_meta.recomputed_ops)
    discarded_memory = sum(sac_stats.memory[i] for i in sac_greedy_order_meta.recomputed_ops)
    sac_memory_budget -= saved_memory
    msps_meta = deepcopy(sac_greedy_order_meta.msps_meta)

    while (sac_memory - discarded_memory) > sac_memory_budget:
        try:
            msps = msps_meta.pop(0)
        except IndexError as e:
            raise IndexError(f"msps_meta is empty") from e
        discard_op_indices.add(msps.op_idx)
        if msps.op_idx in sac_greedy_order_meta.random_inplace_ops:
            discard_op_indices.update(sac_greedy_order_meta.random_inplace_ops)

        if inplace_op_group := sac_greedy_order_meta.inplace_op_groups.get(msps.op_idx, None):
            discard_op_indices.update(inplace_op_group)

        discarded_memory += msps.memory

    for i in discard_op_indices:
        policy_output[i] = 0
    return policy_output


def get_dp_checkpoint_policy_per_module(sac_stats: SACStats, memory_budget) -> List[int]:
    # TODO @sanketpurandare: Write a pseudo-polynomial knapsack DP algo
    pass

def get_auto_sac_policies(
        model: torch.nn.Module,
        sac_estimator: SACEstimator,
        mem_tracker: MemTracker,
        runtime_estimator: RuntimeEstimator,
        dev: torch.device,
        memory_budget: float,
        sac_algo: SACAlgorithm = SACAlgorithm.GREEDY,
        ac_units: Optional[List[str]] = None,
        fsdp_units: Optional[List[str]] = None,
        shard_degree: int = 1,
    ) -> Dict[str, SACPolicy]:
    mod_info = aggregate_stats(model, mem_tracker, runtime_estimator, sac_estimator, dev)
    graph = parse_module_info(mod_info)
    ac_decisions, _, _ = sac_milp(graph, memory_budget, shard_degree, ac_units, fsdp_units)
    sac_policies: Dict[str, SACPolicy] = {}
    for mod_name, discard_ratio in ac_decisions.items():
        sac_stats = sac_estimator.sac_mod_stats[mod_name]
        budget = 1 - discard_ratio
        match sac_algo:
            case SACAlgorithm.GREEDY:
                sac_greedy_order_meta = sac_estimator.sac_mod_greedy_order_meta[mod_name]
                policy_output = get_greedy_checkpoint_policy_per_module(sac_stats, sac_greedy_order_meta, budget)
            case SACAlgorithm.DP:
                policy_output = get_dp_checkpoint_policy_per_module(sac_stats, budget)
            case SACAlgorithm.OPTIMAL:
                policy_output = get_optimal_checkpointing_policy_per_module(sac_stats, budget)
        sac_policy = SACPolicy(policy_output)
        sac_policies[mod_name] = sac_policy
    return sac_policies
        



