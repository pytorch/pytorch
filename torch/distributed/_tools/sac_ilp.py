import logging
import math
import os
from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from torch.distributed._tools.ilp_utils import Graph, is_self_or_submodule, is_submodule
from torch.distributed._tools.sac_estimator import SACStats


try:
    from pulp import (  # type: ignore[import-untyped,import-not-found]
        lpDot,
        LpInteger,
        LpMaximize,
        LpMinimize,
        LpProblem,
        LpStatus,
        lpSum,
        LpVariable,
        PULP_CBC_CMD,
        value,
    )
except ImportError as err:
    raise ImportError(
        "Please install pulp package. See: https://github.com/coin-or/pulp."
    ) from err

# Create a logger object
logger = logging.getLogger()

# Set the logging level to according to env variable
_DEBUG_ILP = int(os.environ.get("DEBUG_AUTO_SAC", 0))
if _DEBUG_ILP == 1:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def sac_milp(
    graph: Graph,
    memory_budget: float,
    shard_degree: int = 1,
    ac_units: Optional[Set[str]] = None,
    fsdp_units: Optional[Set[str]] = None,
) -> Tuple[Dict[str, float], float, int]:
    """
    MILP to decide which modules to AC and how much memory to discard.
    The objective is to minimize recomputation time.
    The constraint is to ensure peak memory is under budget.

    Args:
        graph: graph representation of the model as a module submodule tree
            where each node is a submodule with memory & runtime stats
        memory_budget: memory budget in GiB
        shard_degree: number of GPUs across which the model is sharded. In the case of FSDP,
            shard_degree will be used to compute the amount of parameter, gradient and optimizer
            memory on each rank.
        ac_units: a set of user-specified AC unit FQNs.
        fsdp_units: a set of FSDP units. AC units cannot be supermodules of FSDP unit FQNs.

    Returns:
        Dict[str, float]: the optimal SAC solution, mapping from module fqn to
            the percentage of activation memory to **discard**
        float: the recomputation time of the optimal SAC solution
        int: upper bound on the peak memory of the optimal SAC solution.
            note that value of -1 means that the ILP solver failed to find a solution.

    """
    num_nodes = len(graph.nodes)
    MEM_MULTIPLIER = 2**30
    # total_param_memory/total_opt_memory/total_grad_memory represents the total sharded + non-sharded param/opt/grad
    # memory for the root modules
    total_fwd_runtime = 0.0
    total_param_memory = 0.0
    total_opt_memory = 0.0
    root_grad_mem: Dict[str, float] = defaultdict(float)
    for root_fqn in graph.root_fqns:
        if fsdp_units and root_fqn in fsdp_units:
            total_opt_memory += graph.root_opt_mem[root_fqn] / (MEM_MULTIPLIER * shard_degree)
            root_grad_mem[root_fqn] = graph.name2node[root_fqn]["grad_per_module"] / (
                MEM_MULTIPLIER * shard_degree
            )
            total_param_memory += graph.name2node[root_fqn]["param_per_module"] / (
                MEM_MULTIPLIER * shard_degree
            )
        else:
            total_opt_memory += graph.root_opt_mem[root_fqn] / MEM_MULTIPLIER
            root_grad_mem[root_fqn] = (
                graph.name2node[root_fqn]["grad_per_module"] / MEM_MULTIPLIER
            )
            total_param_memory += (
                graph.name2node[root_fqn]["param_per_module"] / MEM_MULTIPLIER
            )
        total_fwd_runtime += graph.name2node[root_fqn]["fw_runtime_per_module"]
    M = total_fwd_runtime  # note: numerical issue may occur if M is too big

    max_fsdp_unit_memory: Dict[str, float] = defaultdict(float)

    # Create a MILP problem
    prob = LpProblem("SAC", LpMinimize)

    # Create decision variables
    # y_i: indicator for if module i is AC'ed
    y = LpVariable.matrix("y", list(range(num_nodes)), 0, 1, LpInteger)
    # r_i: percentage of discarded activation memory
    r = LpVariable.matrix("r", list(range(num_nodes)), 0, 1)
    # d_i: discarded activation memory for module i
    d = LpVariable.matrix("d", list(range(num_nodes)), 0)
    # a_i: total activation memory at module i
    a = LpVariable.matrix("a", list(range(num_nodes)), 0)
    # m_i: memory at module i, combining parameters, gradients, and activations
    m = LpVariable.matrix("m", list(range(num_nodes)), 0)
    # rcp_i: percentage of recomputation time
    rcp = LpVariable.matrix("rcp", list(range(num_nodes)), 0)
    # rct_i: recomputation time for module i (in ms)
    rct = LpVariable.matrix("rct", list(range(num_nodes)), 0)
    # max_m: peak memory
    max_m = LpVariable("max_m", 0)

    # Add constraints
    # [Constraint] User specified AC units
    if ac_units:
        for i in range(num_nodes):
            if graph.nodes[i]["fqn"] not in ac_units:
                prob += y[i] == 0

    # [Constraint] AC units cannot be supmodules of user specified FSDP units
    if fsdp_units:
        for i in range(num_nodes):
            if any(
                is_submodule(fsdp_unit, graph.nodes[i]["fqn"])
                for fsdp_unit in fsdp_units
            ):
                prob += y[i] == 0

        node_to_param_size: Dict[str, int] = {
            node["fqn"]: node["param_per_module"] for node in graph.nodes
        }

        for (
            fqn
        ) in graph.fw_post_order:  # using post order so that children are visited first
            if fqn in fsdp_units:
                if node := graph.name2node.get(fqn, None):
                    j = node["index"]
                    # its ancestor nodes will no longer need to take care of these parameters, so subtracting for them
                    for i in range(j):
                        if graph.ad_matrix[i][j]:
                            node_to_param_size[
                                graph.nodes[i]["fqn"]
                            ] -= node_to_param_size[fqn]

        # Find maximum parameter count among FSDP units for each root module
        for root_fqn in graph.root_fqns:
            if root_fqn in fsdp_units:
                max_unit_memory = max(
                    node_to_param_size[fsdp_unit]
                    for fsdp_unit in fsdp_units
                    if is_self_or_submodule(fsdp_unit, root_fqn)
                )
                max_fsdp_unit_memory[root_fqn] = max_unit_memory / MEM_MULTIPLIER

    # [Constraint] No nested AC units
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if graph.ad_matrix[i][j] == 1:
                prob += y[i] + y[j] <= 1

    # [Constraint] Do not AC leaf modules
    for i in range(num_nodes):
        if graph.nodes[i]["is_leaf"]:
            prob += y[i] == 0

    # [Constraint] Do not AC modules that don't call backward
    for i in range(num_nodes):
        if not graph.nodes[i]["requires_grad"]:
            prob += y[i] == 0

    # [Constraint] Express amount of discarded activation memory
    for i in range(num_nodes):
        # There are two measures for activation memory: ACM and IA
        # 1. IA is the activation memory saved when not using AC
        # 2. ACM is the total activation memory, including those
        #    that are not typically saved when not using AC
        # Note: ACM >= IA
        if (not graph.nodes[i]["is_leaf"]) and graph.nodes[i][
            "sac_memory"
        ] < graph.nodes[i]["act_fw_per_module"]:
            logger.warning(
                "For module {%s}: activation memory from memory tracker is {%d}, activation memory from SAC estimator is {%d}. "
                "Something is wrong. Please check! Overriding the latter with the former.",
                graph.nodes[i]["fqn"],
                graph.nodes[i]["act_fw_per_module"],
                graph.nodes[i]["sac_memory"],
            )
            graph.nodes[i]["sac_memory"] = graph.nodes[i]["act_fw_per_module"]
        ACM_i = graph.nodes[i]["sac_memory"] / MEM_MULTIPLIER
        IA_i = graph.nodes[i]["act_fw_per_module"] / MEM_MULTIPLIER
        prob += d[i] == ACM_i * r[i] - (ACM_i - IA_i) * y[i]

    # [Constraint] Ensure correctness of r_i
    # There are two parts to its correctness
    # 1. r_i > 0 only if y_i == 1 (discard only if it is an AC unit)
    # 2. r_i needs to be large enough to cover the difference between
    #    ACM and IA. Otherwise, we are not saving any memory
    # 3. r_i cannot be larger than the total amount of ACM available
    #   for discarding. Hence, we subtract the mandatorily saved memory (SAV).
    for i in range(num_nodes):
        prob += y[i] >= r[i]
        if graph.nodes[i]["is_leaf"] or not graph.nodes[i]["requires_grad"]:
            continue
        SAV_i = graph.nodes[i]["saved_memory"] / MEM_MULTIPLIER
        ACM_i = graph.nodes[i]["sac_memory"] / MEM_MULTIPLIER
        IA_i = graph.nodes[i]["act_fw_per_module"] / MEM_MULTIPLIER
        if ACM_i > 0:
            prob += r[i] >= (ACM_i - IA_i) / ACM_i * y[i]
            prob += r[i] <= (ACM_i - SAV_i) / ACM_i
        else:
            prob += y[i] == 0

    # [Constraint] Express total activation memory in the backward pass
    for i in range(num_nodes):
        AG_i = graph.nodes[i]["act_grad_per_module"] / MEM_MULTIPLIER
        TA_i = graph.nodes[i]["act_total"] / MEM_MULTIPLIER
        # related to discarded amount of memory
        pos = graph.nodes[i]["pos_fw_post_order"]
        coeff = [0] * num_nodes
        for p in range(pos):
            if fw_po_node := graph.name2node.get(graph.fw_post_order[p], None):
                j = fw_po_node["index"]
                coeff[j] = 1
        prob += a[i] == TA_i + AG_i - lpDot(coeff, d)

    # [Constraint] Express the total amount of memory at each module
    # ACC_i represents the grad memory accumulated so far for the root module of i
    # RG_i represents the accumulated grad memory of other root modules the have completed their backward
    # total_param_memory/total_opt_memory represents the total sharded + non-sharded param/opt memory for
    #  all modules that stays static throughtout the training
    for i in range(num_nodes):
        root_node = graph.nodes[graph.get_root_idx(i)]
        root_fqn = root_node["fqn"]
        dtype_factors = graph.root_dtype_factors[root_fqn]
        ACC_i = (graph.nodes[i]["grad_total"] - root_node["grad_total"]) / MEM_MULTIPLIER
        grad_shard_degree = (
            shard_degree if fsdp_units and root_fqn in fsdp_units else 1
        )
        RG_i = 0.0
        for other_root_fqn, grad_mem in root_grad_mem.items():
            if graph.name2node[other_root_fqn]["index"] > root_node["index"]:
                RG_i += grad_mem
        prob += m[i] == a[i] * dtype_factors["act"] \
        + 3 * max_fsdp_unit_memory[root_fqn] * dtype_factors["param"] \
        + max_fsdp_unit_memory[root_fqn] * dtype_factors["reduce"] \
        + total_opt_memory + total_param_memory + RG_i + (ACC_i / grad_shard_degree)

    # [Constraint] Express peak memory
    for i in range(num_nodes):
        prob += max_m >= m[i]

    # [Constraint] Express percentage of recomputation time
    for i in range(num_nodes):
        for s in range(graph.nodes[i]["n_segments"]):
            slope = graph.nodes[i]["slopes"][s]
            intercept = graph.nodes[i]["intercepts"][s]
            prob += rcp[i] >= slope * r[i] + intercept

    # [Constraint] Express recomputation time
    # rct_i = (rcp_i * ACT_i) if y_i == 1 else 0
    for i in range(num_nodes):
        ACT_i = graph.nodes[i]["sac_runtime"]
        prob += rct[i] <= M * y[i]
        prob += rct[i] <= ACT_i * rcp[i]
        prob += rct[i] >= ACT_i * rcp[i] - M * (1 - y[i])

    # [Constraint] Peak memory should be below budget
    prob += max_m <= memory_budget

    # Set Objeictive
    prob += lpSum(rct)

    # Solve
    solver = PULP_CBC_CMD(gapRel=0.05, timeLimit=180, msg=0, options=[f"RandomS {42}"])
    status = prob.solve(solver)

    # If solver fails, print status and return empty solution
    if status != 1:
        logger.error("Solver failed to find a solution: %s", LpStatus[status])
        return {}, 0, -1

    # Gather and return solution if optimal solution is found
    ac_decisions = {}
    for i in range(num_nodes):
        if round(y[i].varValue) == 1:
            ac_decisions[graph.nodes[i]["fqn"]] = round(r[i].varValue, 4)
    recomputation_time = round(value(prob.objective), 2)
    peak_mem = round(max_m.varValue * MEM_MULTIPLIER)

    return ac_decisions, recomputation_time, peak_mem


class SACDecision(IntEnum):
    RECOMPUTE = 0
    SAVE = 1


def get_optimal_checkpointing_policy_per_module(
    sac_stats: SACStats, memory_budget: float
) -> List[int]:
    """
    This is adapted from --
    https://github.com/facebookresearch/xformers/blob/c6c0ac31f1b08542a0bc27278c6ed10f825f6963/xformers/checkpoint.py#L375

    Given the SACStats of a module, including list of operators, their memory, runtimes, and metadata,
    decide via MILP an optimal set of operators to checkpoint under a given ``memory_budget``.

    Args:
        sac_stats: the SACStats object of the module
        memory_budget: a float between zero and one

    Returns:
        List[int]: the decision whether each operator should be saved (1) or recomptued (0).
    """
    if not (0 <= memory_budget <= 1):
        raise ValueError(
            f"`memory_budget` must be a float between 0 and 1. Got {memory_budget}."
        )
    clone_ops = [op_idx for op_idx, f_name in enumerate(sac_stats.func_names) if f_name == "clone"]
    num_ops = len(sac_stats.func_names)

    # Create a MILP problem
    prob = LpProblem("SAC-per-module", LpMaximize)

    # Create decision variables
    # x[i] = 1 means the i-th operator should be saved, otherwise it should be recomputed
    x = LpVariable.matrix("x", list(range(num_ops)), 0, 1, LpInteger)

    # Add constraints
    # [Constraint] random ops should be saved if ``force_store_random`` is True
    # otherwise, random ops should either be all recomputed or all saved
    if sac_stats.force_store_random:
        for i in sac_stats.rand_ops:
            prob += x[i] == SACDecision.SAVE.value
    else:
        for i1, i2 in zip(sac_stats.rand_ops[:-1], sac_stats.rand_ops[1:]):
            prob += x[i1] == x[i2]

    # [Constraint] view-like and clone ops should always be recomputed
    view_or_clone_ops = sac_stats.view_like_ops + clone_ops
    for i in view_or_clone_ops:
        prob += x[i] == SACDecision.RECOMPUTE.value

    # [Constraint] inplace ops should always be done in conjunction with its parent op
    for op, op_parent in sac_stats.inplace_ops:
        if op != op_parent:
            prob += x[op] == x[op_parent]
        else:
            prob += x[op] == SACDecision.SAVE.value

    # [Constraint] saved memory should be under the ``memory_budget``
    max_memory = math.ceil(memory_budget * sum(sac_stats.memory))
    prob += lpDot(x, sac_stats.memory) <= max_memory

    # [Objective] minimize recomputation time, note the ILP is a maximization problem
    # because x[i] == 1 means the op is saved (not recomputed), and thus recomputation
    # time is sum(sac_stats.runtimes) - lpDot(x, sac_stats.runtimes)
    prob += lpDot(x, sac_stats.runtimes)

    # Solve
    solver = PULP_CBC_CMD(gapRel=0.05, timeLimit=20, msg=0, options=[f"RandomS {42}"])
    status = prob.solve(solver)

    # If solver fails, print status and return empty solution
    if status != 1:
        logger.error("Solver failed to find a solution: %s", LpStatus[status])
        return []

    # Gather and return solution if optimal solution is found
    return [round(x[i].varValue) for i in range(num_ops)]
