import logging
from typing import Dict, List, Optional, Tuple

from torch.distributed._tools.ilp_utils import Graph, is_submodule


try:
    from pulp import (  # type: ignore[import-untyped,import-not-found]
        lpDot,
        LpInteger,
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
logger = logging.getLogger(__name__)

# Set the logging level to INFO
logger.setLevel(logging.INFO)


def sac_milp(
    graph: Graph,
    memory_budget: float,
    world_size: int = 1,
    ac_units: Optional[List[str]] = None,
    fsdp_units: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], float, int]:
    """
    MILP to decide which modules to AC and how much memory to discard.
    The objective is to minimize recomputation time.
    The constraint is to ensure peak memory is under budget.

    Args:
        graph: graph representation of the model as a module submodule tree
            where each node is a submodule with memory & runtime stats
        memory_budget: memory budget in GiB
        world_size: number of GPUs. In the case of FSDP, world_size will be
            used to compute the amount of parameter and gradient memory on each rank
        ac_units: a list of user-specified AC units.
        fsdp_units: a list of FSDP units. AC units cannot be supermodules of FSDP units.

    Returns:
        Dict[str, float]: the optimal SAC solution, mapping from module fqn to
            the percentage of activation memory to discard
        float: the recomputation time of the optimal SAC solution
        int: upper bound on the peak memory of the optimal SAC solution.
            note that value of -1 means that the ILP solver failed to find a solution.

    """
    num_nodes = len(graph.nodes)
    M = 10**2  # note: numerical issue may occur if M is too big
    MEM_MULTIPLIER = 2**30

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
        ac_units_set = set(ac_units)
        for i in range(num_nodes):
            if graph.nodes[i]["fqn"] not in ac_units_set:
                prob += y[i] == 0

    # [Constraint] AC units cannot be supmodules of user specified FSDP units
    if fsdp_units:
        for i in range(num_nodes):
            if any(
                is_submodule(fsdp_unit, graph.nodes[i]["fqn"])
                for fsdp_unit in fsdp_units
            ):
                prob += y[i] == 0

    # [Constraint] No nested AC units
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if graph.ad_matrix[i][j] == 1:
                prob += y[i] + y[j] <= 1

    # [Constraint] Do not AC leaf modules
    for i in range(num_nodes):
        if graph.nodes[i]["is_leaf"]:
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
            logger.warning("For module {%s}: ", graph.nodes[i]["fqn"])
            logger.warning(
                "activation memory from memory tracker is {%d},",
                graph.nodes[i]["act_fw_per_module"],
            )
            logger.warning(
                "activation memory from SAC estimator is {%d}.",
                graph.nodes[i]["sac_memory"],
            )
            logger.warning("Something is wrong. Please check!")
            logger.warning("Overriding the latter with the former.")
            graph.nodes[i]["sac_memory"] = graph.nodes[i]["act_fw_per_module"]
        ACM_i = graph.nodes[i]["sac_memory"] / MEM_MULTIPLIER
        IA_i = graph.nodes[i]["act_fw_per_module"] / MEM_MULTIPLIER
        prob += d[i] == ACM_i * r[i] - (ACM_i - IA_i) * y[i]

    # [Constraint] Ensure correctness of r_i
    # There are two parts to its correctness
    # 1. r_i > 0 only if y_i == 1 (discard only if it is an AC unit)
    # 2. r_i needs to be large enough to cover the difference between
    #    ACM and IA. Otherwise, we are not saving any memory
    for i in range(num_nodes):
        prob += y[i] >= r[i]
        if graph.nodes[i]["is_leaf"]:
            continue
        ACM_i = graph.nodes[i]["sac_memory"] / MEM_MULTIPLIER
        IA_i = graph.nodes[i]["act_fw_per_module"] / MEM_MULTIPLIER
        prob += r[i] >= (ACM_i - IA_i) / ACM_i * y[i]

    # [Constraint] Express total activation memory in the backward pass
    for i in range(num_nodes):
        AG_i = graph.nodes[i]["act_grad_per_module"] / MEM_MULTIPLIER
        TA_i = graph.nodes[i]["act_total"] / MEM_MULTIPLIER
        # related to discarded amount of memory
        pos = graph.nodes[i]["pos_fw_post_order"]
        coeff = [0] * num_nodes
        for p in range(pos):
            j = graph.name2node[graph.fw_post_order[p]]["index"]
            coeff[j] = 1
        prob += a[i] == TA_i + AG_i - lpDot(coeff, d)

    # [Constraint] Express the total amount of memory at each module
    # Note that unsharded parameters and gradients are not included here
    P_1 = graph.nodes[0]["param_per_module"] / MEM_MULTIPLIER
    for i in range(num_nodes):
        TG_i = graph.nodes[i]["grad_total"] / MEM_MULTIPLIER
        prob += m[i] == a[i] + (P_1 + TG_i) / world_size

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
    solver = PULP_CBC_CMD(gapRel=0.05, timeLimit=180)
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
