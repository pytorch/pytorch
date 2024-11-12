import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from torch.distributed._tools.ilp_utils import display_bytes, Graph


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


class CommType(Enum):
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"


@dataclass
class CommParams:
    latency: float  # in ms
    bandwidth: float  # in bytes / ms


def fsdp_milp(
    graph: Graph,
    world_size: int,
    comm_params: Dict[CommType, CommParams],
    memory_budget: float,
    fsdp_units: Optional[List[str]] = None,
) -> Tuple[Set[str], float, int]:
    """
    MILP to decide FSDP units.
    The objective is to minimize exposed computation time.
    The constraint is to ensure peak memory is under budget.

    Args:
        graph: graph representation of the model as a module submodule tree
            where each node is a submodule with memory & runtime stats
        world_size: number of GPUs parameters and gradients are sharded across for FSDP.
        comm_params: a dictionary of communication parameters, including latency and bandwidth.
        memory_budget: memory budget in GiB
        fsdp_units: a list of user-specified FSDP units.
        selective_ac: whether to use selective AC jointly with FSDP.

    Returns:
        Set[str]: the set of FSDP units
        float: the per-iteration exposed communication time of the returned FSDP solution.
        int: upper bound on the peak memory of the returned FSDP solution
            note that value of -1 means that the ILP solver failed to find a solution.
    """

    num_nodes = len(graph.nodes)
    BIG_M = 1000
    MEM_MULTIPLIER = 2**30

    # Create a MILP problem
    prob = LpProblem("FSDP", LpMinimize)

    # Create decision variables
    # x_i: indicator if module i is an fsdp unit
    x = LpVariable.matrix("x", list(range(num_nodes)), 0, 1, LpInteger)
    # p_i: parameter memory during module i
    p = LpVariable.matrix("p", list(range(num_nodes)), 0)
    # g_i: gradient memory during module i
    g = LpVariable.matrix("g", list(range(num_nodes)), 0)
    # a_i: activation(-related) memory during module i
    a = LpVariable.matrix("a", list(range(num_nodes)), 0)
    # m_i: total memory during module i (including params, grads, and activations)
    m = LpVariable.matrix("m", list(range(num_nodes)), 0)
    # max_m: peak memory
    max_m = LpVariable("max_m", 0)
    # max_p: maximum fsdp shard
    max_p = LpVariable("max_p", 0)
    # ag_i: all gather communication time of parameters for module i
    ag = LpVariable.matrix("ag", list(range(num_nodes)), 0)
    # t0_i: helper variable for the forward prefetch all gather communication time
    t0 = LpVariable.matrix("t0", list(range(num_nodes)), 0)
    # fw_ag_i: all gather communication time at module i during forward
    # this is the prefetch for the next fsdp unit
    fw_ag = LpVariable.matrix("fw_ag", list(range(num_nodes)), 0)
    # t1_i: helper variable for the backward prefetch all gather communication time
    t1 = LpVariable.matrix("t1", list(range(num_nodes)), 0)
    # bw_ag_i: all gather communication time at module i during backward
    # this is the prefetch for the next fsdp unit
    bw_ag = LpVariable.matrix("bw_ag", list(range(num_nodes)), 0)
    # rs_i: reduce scatter communication time of parameters for module i
    rs = LpVariable.matrix("rs", list(range(num_nodes)), 0)
    # t2_i: helper variable for the backward prefetch reduce scatter communication time
    t2 = LpVariable.matrix("t2", list(range(num_nodes)), 0)
    # bw_rs_i: reduce scatter communication time at module i during backward
    # this is the prefetch for the next fsdp unit
    bw_rs = LpVariable.matrix("bw_rs", list(range(num_nodes)), 0)
    # t3_i: helpr variable for the exposed communication time in the forward pass
    t3 = LpVariable.matrix("t3", list(range(num_nodes)), 0)
    # fw_e_i: exposed communication time in the forward pass for module i if fsdp unit
    fw_e = LpVariable.matrix("fw_e", list(range(num_nodes)), 0)
    # t4_i: helper variable for the exposed communication time in the backward pass
    t4 = LpVariable.matrix("t4", list(range(num_nodes)), 0)
    # bw_e_i: exposed communication time in the backward pass for module i if fsdp unit
    bw_e = LpVariable.matrix("bw_e", list(range(num_nodes)), 0)

    # Add constraints
    # [Constraint] Root module is always an FSDP unit
    prob += x[0] == 1

    # [Constraint] Use user specified FSDP units if provided
    if fsdp_units:
        fsdp_units_set = set(fsdp_units)
        for i in range(1, num_nodes):
            if graph.nodes[i]["fqn"] in fsdp_units_set:
                prob += x[i] == 1
            else:
                prob += x[i] == 0

    # [Constraint] No nested FSDP unit
    # This is not a necessary constraint for the application of FSDP. But having it does not
    # significantly affect the solution qulity and improves the speed of the solver.
    for i in range(1, num_nodes):
        for j in range(i + 1, num_nodes):
            if graph.ad_matrix[i][j] == 1:
                prob += x[i] + x[j] <= 1

    # [Constraint] Express param size of each module if it is an FSDP unit, zero otherwise
    for i in range(1, num_nodes):
        P_i = graph.nodes[i]["param_per_module"] / MEM_MULTIPLIER
        prob += p[i] == P_i * x[i]
    P_1 = graph.nodes[0]["param_per_module"] / MEM_MULTIPLIER  # total parameter size
    prob += p[0] == P_1 - lpSum(p[1:])

    # [Constraint] Express grad size of each module if it is an FSDP unit, zero otherwise
    for i in range(1, num_nodes):
        G_i = graph.nodes[i]["grad_per_module"] / MEM_MULTIPLIER
        prob += g[i] == G_i * x[i]
    G_1 = graph.nodes[0]["grad_per_module"] / MEM_MULTIPLIER  # total gradient size
    prob += g[0] == G_1 - lpSum(g[1:])

    # [Constraint] Express total activation memory of each module in the bwd pass
    for i in range(num_nodes):
        AG_i = graph.nodes[i]["act_grad_per_module"] / MEM_MULTIPLIER
        TA_i = graph.nodes[i]["act_total"] / MEM_MULTIPLIER
        prob += a[i] == TA_i + AG_i

    # [Constraint] Express the total amount memory at each module
    # It includes: sharded parameters and gradients; unsharded parameters and gradients, activations
    for i in range(num_nodes):
        TG_i = graph.nodes[i]["grad_total"] / MEM_MULTIPLIER
        coeff = [0] * num_nodes
        for j in range(num_nodes):
            if graph.ad_matrix[j][i] == 1:
                coeff[j] = 1
        prob += (
            m[i] == (P_1 + TG_i) / world_size + lpDot(p, coeff) + lpDot(g, coeff) + a[i]
        )

    # [Constraint] Express peak memory
    for i in range(num_nodes):
        prob += max_m >= m[i]

    # [Constraint] Express the maximum size of an FSDP shard
    for i in range(num_nodes):
        prob += max_p >= p[i]

    # [Constraint] Respect memory budget
    # `2 * max_p` is the hacky way to deal with prefetched all-gathered parameter memory
    prob += max_m + 2 * max_p <= memory_budget

    # [Constraint] Express the all gather communication time of each FSDP unit
    comm_model = comm_params[CommType.ALL_GATHER]
    for i in range(num_nodes):
        prob += ag[i] == comm_model.latency + p[i] * (
            MEM_MULTIPLIER / comm_model.bandwidth  # convert from bytes/ms to GiB/ms
        )

    # [Constraint] Express the reduce scatter communication time of each FSDP unit
    comm_model = comm_params[CommType.REDUCE_SCATTER]
    for i in range(num_nodes):
        prob += rs[i] == comm_model.latency + g[i] * (
            MEM_MULTIPLIER / comm_model.bandwidth  # convert from bytes/ms to GiB/ms
        )

    # [Constraint] Express the forward prefetch all gather communication time
    # E.g., each FSDP unit will prefetch the parameters for the next FSDP unit
    # The constraints below are to linearize the following non-linear constraints:
    #    t0_i = ag_i * x_i + t0_{i+1} * (1 - x_i)
    #    fw-ag_i = t0_{i+1} * x_i
    # Note that t0 is a helper decision variable, expressing the all-gather communication
    # time of the next fsdp unit (self included).
    prob += t0[num_nodes - 1] == ag[num_nodes - 1]
    for i in range(1, num_nodes - 1):
        prob += t0[i] <= t0[i + 1] + BIG_M * x[i]
        prob += t0[i] >= t0[i + 1] - BIG_M * x[i]
        prob += t0[i] <= ag[i] + BIG_M * (1 - x[i])
        prob += t0[i] >= ag[i] - BIG_M * (1 - x[i])
    prob += fw_ag[num_nodes - 1] == 0
    for i in range(num_nodes - 1):
        prob += fw_ag[i] <= BIG_M * x[i]
        prob += fw_ag[i] <= t0[i + 1]
        prob += fw_ag[i] >= t0[i + 1] - BIG_M * (1 - x[i])

    # [Constraint] Express the backward prefetch all gather communication time
    # E.g., each FSDP unit will prefetch the parameters for the next FSDP unit
    # The constraints below are to linearize the following non-linear constraints:
    #    t1_{o1(k)} = ag_{o1(k)} * x_{o1(k)} + t1_{o1(k+1)} * (1 - x_{o1(k)})
    #    bw-ag_i = t1_{o1(k+1)} * x_{o1(k)}
    # Note that t1 is a helper decision variable, expressing the all-gather communication
    # time of the next fsdp unit (self included).
    # Note the order of module traversal is different in the backward pass. Thus, needing
    # ``o1`` which is the index of modules in the backward pre order.
    o1 = [graph.name2node[fqn]["index"] for fqn in reversed(graph.fw_post_order)]
    prob += t1[o1[num_nodes - 1]] == ag[o1[num_nodes - 1]]
    for k in range(1, num_nodes - 1):
        i = o1[k]
        i_next = o1[k + 1]
        prob += t1[i] <= t1[i_next] + BIG_M * x[i]
        prob += t1[i] >= t1[i_next] - BIG_M * x[i]
        prob += t1[i] <= ag[i] + BIG_M * (1 - x[i])
        prob += t1[i] >= ag[i] - BIG_M * (1 - x[i])
    prob += bw_ag[o1[num_nodes - 1]] == 0
    for k in range(1, num_nodes - 1):
        i = o1[k]
        i_next = o1[k + 1]
        prob += bw_ag[i] <= BIG_M * x[i]
        prob += bw_ag[i] <= t1[i_next]
        prob += bw_ag[i] >= t1[i_next] - BIG_M * (1 - x[i])

    # [Constraint] Express the previous module's reduce scatter communication time
    # E.g., each FSDP unit's all-gather call follows the reduce-scatter call of the previous FSDP unit
    # The constraints below are to linearize the following non-linear constraints:
    #    t2_i = rs_i * x_i + t2_{i+1} * (1 - x_i)
    #    bw-rs_i = t2_{i+1} * x_i
    # Note that t2 is a helper decision variable, expressing the reduce communication
    # time of the next fsdp unit (self included).
    prob += t2[num_nodes - 1] == rs[num_nodes - 1]
    for i in range(1, num_nodes - 1):
        prob += t2[i] <= t2[i + 1] + BIG_M * x[i]
        prob += t2[i] >= t2[i + 1] - BIG_M * x[i]
        prob += t2[i] <= rs[i] + BIG_M * (1 - x[i])
        prob += t2[i] >= rs[i] - BIG_M * (1 - x[i])
    prob += bw_rs[num_nodes - 1] == 0
    for i in range(num_nodes - 1):
        prob += bw_rs[i] <= BIG_M * x[i]
        prob += bw_rs[i] <= t2[i + 1]
        prob += bw_rs[i] >= t2[i + 1] - BIG_M * (1 - x[i])

    # [Constraint] Express the exposed communication time in the forward pass for
    # The constraints below are to linearize the following non-linear constraints:
    #    t3_i = max(0, fw-ag_i - FCP_i)
    #    fw_e_i = t3_i * x_i
    for i in range(1, num_nodes):
        FCP_i = graph.nodes[i]["fw_runtime_per_module"]
        prob += t3[i] >= fw_ag[i] - FCP_i
        prob += fw_e[i] <= BIG_M * x[i]
        prob += fw_e[i] <= t3[i]
        prob += fw_e[i] >= t3[i] - BIG_M * (1 - x[i])
    prob += fw_e[0] == 0

    # [Constraint] Express the exposed communication time in the backward pass
    # The constraints below are to linearize the following non-linear constraints:
    #    t4_i = max(0, bw-ag_i + bw-rs_i - FCP_i)
    #    bw_e_i = t4_i * x_i
    for i in range(1, num_nodes):
        BCP_i = graph.nodes[i]["bw_runtime_per_module"]
        prob += t4[i] >= bw_ag[i] + bw_rs[i] - BCP_i
        prob += bw_e[i] <= BIG_M * x[i]
        prob += bw_e[i] <= t4[i]
        prob += bw_e[i] >= t4[i] - BIG_M * (1 - x[i])
    prob += bw_e[0] == 0

    # Set objeictive -- minimize total exposed communication time
    prob += lpSum(fw_e[1:]) + lpSum(bw_e[1:]) + ag[0] + rs[0] + fw_ag[0] + bw_rs[0]

    # Solve
    solver = PULP_CBC_CMD(gapRel=0.05, timeLimit=180, msg=0)
    status = prob.solve(solver)

    # If solver fails, print status and return empty solution
    if status != 1:
        logger.error("Solver failed to find a solution: %s", LpStatus[status])
        return set(), 0, -1

    # Gather and return solution if optimal solution is found
    fsdp_decisions = set()
    for i in range(num_nodes):
        if round(value(x[i]) if x[i] else 0) == 1:
            fsdp_decisions.add(graph.nodes[i]["fqn"])
    peak_mem = round((max_m.varValue + 2 * max_p.varValue) * MEM_MULTIPLIER)
    exposed_comm_time = round(value(prob.objective), 4)

    # debugging info
    fqn_len = min(30, max(len(graph.nodes[i]["fqn"]) for i in range(num_nodes)))
    for i in range(num_nodes):
        fqn = graph.nodes[i]["fqn"][-fqn_len:].ljust(fqn_len)
        x_i = value(x[i]) if x[i] else 0
        p_i = p[i].varValue * MEM_MULTIPLIER
        g_i = g[i].varValue * MEM_MULTIPLIER
        TG_i = graph.nodes[i]["grad_total"]
        a_i = a[i].varValue * MEM_MULTIPLIER
        m_i = m[i].varValue * MEM_MULTIPLIER
        ag_i = ag[i].varValue if ag[i] else 0
        fw_ag_i = fw_ag[i].varValue if fw_ag[i] else 0
        bw_ag_i = bw_ag[i].varValue if bw_ag[i] else 0
        rs_i = rs[i].varValue if rs[i] else 0
        bw_rs_i = bw_rs[i].varValue if bw_rs[i] else 0
        FCP_i = graph.nodes[i]["fw_runtime_per_module"]
        BCP_i = graph.nodes[i]["bw_runtime_per_module"]
        fw_e_i = fw_e[i].varValue if fw_e[i] else 0
        bw_e_i = bw_e[i].varValue if bw_e[i] else 0
        debug_str = (
            ("FSDP" if round(x_i) == 1 else "    ")
            + f" {fqn} : "
            + f"p_i = {display_bytes(p_i, 'GiB'):<10} "
            + f"g_i = {display_bytes(g_i, 'GiB'):<10} "
            + f"TG_i = {display_bytes(TG_i, 'GiB'):<10} "
            + f"a_i = {display_bytes(a_i, 'GiB'):<10} "
            + f"m_i = {display_bytes(m_i, 'GiB'):<10} "
            + f"ag_i = {round(ag_i, 2):5.2f} ms "
            + f"fw_ag_i = {round(fw_ag_i, 2):5.2f} ms "
            + f"bw_ag_i = {round(bw_ag_i, 2):5.2f} ms "
            + f"rs_i = {round(rs_i, 2):5.2f} ms "
            + f"bw_rs_i = {round(bw_rs_i, 2):5.2f} ms "
            + f"FCP_i = {FCP_i:8.2f} ms "
            + f"BCP_i = {BCP_i:8.2f} ms "
            + f"fw_e_i = {round(fw_e_i, 2):5.2f} ms "
            + f"bw_e_i = {round(bw_e_i, 2):5.2f} ms "
        )
        logger.debug(debug_str)

    return fsdp_decisions, exposed_comm_time, peak_mem
