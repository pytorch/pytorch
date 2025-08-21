# mypy: ignore-errors
import functools
import time
from collections import defaultdict

import torch
from torch.utils._ordered_set import OrderedSet

from .. import config, memory
from ..utils import is_collective
from ..virtualized import V
from .auto_bucket_utils import (
    calibrate_with_cache,
    get_ag_node_pg_info,
    get_node_tensor_info,
    get_rs_node_pg_info,
)
from .bucket_utils import (
    _schedule_fallback_operation,
    bucket_all_gathers,
    bucket_reduce_scatters,
)
from .estimator import get_data_size
from .reorder import _check_ir_node_fsdp


def get_dynamic_memory_threshold(
    peak_memory,
    peak_memory_dict,
    current_step,
    last_release_step,
    forward,
):
    # this function calculates the memory gap from the current step's peak memory
    # to the peak memory criteria
    # it calculates how much memory can be filled to meet peak memory criteria
    left_peak_memory = 0
    right_peak_memory = 0
    for idx, memory in peak_memory_dict.items():
        if idx <= current_step:
            left_peak_memory = max(memory, left_peak_memory)
        if idx >= current_step:
            right_peak_memory = max(memory, right_peak_memory)
    current_peak_memory = min(left_peak_memory, right_peak_memory)
    return peak_memory - current_peak_memory, current_peak_memory


def estimate_bucketed_node_list(
    current_node_bucket,
    schedule_fallback_operation,
    group_size,
    group_name,
    name_to_buf,
    comm_func,
    comm_cache,
    reduce_op=None,
):
    input_ir_nodes = [n.node.inputs[0] for n in current_node_bucket]
    if len(input_ir_nodes) == 1:
        # standalone node, no need to bucket
        comm_node = current_node_bucket[0].node
        comm_size_inp, comm_size_out = (
            comm_node.inputs[0].layout.size,
            comm_node.layout.size,
        )
        estimated_comm = comm_cache.get_comm_time(
            comm_size_inp,
            comm_size_out,
            comm_func,
            calibrated=True,
        )
        return estimated_comm, comm_size_inp, comm_size_out

    if comm_func == "torch.ops._c10d_functional.all_gather_into_tensor.default":
        bucked_node = bucket_all_gathers(
            schedule_fallback_operation,
            group_size,
            group_name,
            input_ir_nodes,
            current_node_bucket,
            name_to_buf,
            return_ag_only=True,
        )
        comm_size_inp = bucked_node[0].layout.size
        comm_size_out = bucked_node[1].layout.size
    elif comm_func == "torch.ops._c10d_functional.reduce_scatter_tensor.default":
        bucked_node = bucket_reduce_scatters(
            schedule_fallback_operation,
            group_size,
            group_name,
            reduce_op,
            input_ir_nodes,
            current_node_bucket,
            name_to_buf,
            return_rs_only=True,
        )
        comm_size_inp = bucked_node[0].layout.size
        comm_size_out = bucked_node[1].layout.size
    estimated_comm = comm_cache.get_comm_time(
        bucked_node[0].layout.size,
        bucked_node[1].layout.size,
        comm_func,
        calibrated=True,
    )
    return estimated_comm, comm_size_inp, comm_size_out


def estimate_hetero_bucketed_node(
    current_node_bucket_dict,
    schedule_fallback_operation,
    name_to_buf,
    comm_func,
    comm_cache,
    reduce_op=None,
):
    estimated_comm, comm_size_inp, comm_size_out = 0, 0, 0
    for node_info, node_list in current_node_bucket_dict.items():
        group_size, group_name = node_info[-2], node_info[-1]
        local_comm, local_comm_size_inp, local_comm_size_out = (
            estimate_bucketed_node_list(
                node_list,
                schedule_fallback_operation,
                group_size,
                group_name,
                name_to_buf,
                comm_func,
                comm_cache,
                reduce_op,
            )
        )
        estimated_comm += local_comm
        comm_size_inp += get_data_size(local_comm_size_inp)
        comm_size_out += get_data_size(local_comm_size_out)
    return estimated_comm, comm_size_inp, comm_size_out


def get_bucketing_plan(
    sched: "scheduler.Scheduler",
    snodes: list["scheduler.BaseSchedulerNode"],
    name_to_buf,
    name_to_fused_node,
    has_reduce_scatter: bool,
    comm_cache,
    comp_cache,
    non_bucketable_pg,
    verbose: bool = False,
) -> list[list["scheduler.BaseSchedulerNode"]]:
    all_gather_plan = []
    reduce_scatter_plan = []
    current_ag_bucket = defaultdict(list)
    current_rs_bucket = defaultdict(list)
    heuristic_info = {
        "last_step_rs_comm": 0,
        "last_step_rs_comm_size": 0,
        "this_step_rs_comm_size": 0,
        "rs_comm_size_accumulated": 0,
        "this_step_rs_comm": 0,
        "this_step_comp": 0,
        "this_step_memory": 0,
        "next_step_comp": 0,
        "next_step_memory": 0,
        "next_step_nonfsdp_comm": 0,
    }

    graph_outputs = OrderedSet(V.graph.get_output_names())
    graph_inputs = OrderedSet(V.graph.graph_inputs.keys())
    _, name_to_freeable_input_buf = memory.prepare_planning_info(
        snodes,
        name_to_buf,
        name_to_fused_node,
        graph_inputs,
        graph_outputs,
    )
    _, memories_at_nodes = memory.estimate_peak_memory(
        snodes, name_to_freeable_input_buf, graph_outputs
    )
    assert len(memories_at_nodes) == len(snodes) + 1

    # get basic info of ag/rs nodes
    has_fsdp_comm = False
    fsdp_world_size = 0
    fsdp_rs_reduce_op = None
    for idx, snode in enumerate(snodes):
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ) and _check_ir_node_fsdp(snode.node, non_bucketable_pg):
            has_fsdp_comm = True
            pg_info = get_ag_node_pg_info(snode)
            if pg_info is None:
                continue
            fsdp_world_size = pg_info[0]
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ) and _check_ir_node_fsdp(snode.node, non_bucketable_pg):
            has_fsdp_comm = True
            pg_info = get_rs_node_pg_info(
                snode, return_reduce_op=True
            )
            if pg_info is None:
                continue
            fsdp_world_size, _, fsdp_rs_reduce_op = pg_info
            break

    # if there is no fsdp comm., return
    if not has_fsdp_comm:
        return [[]], [[]]

    comp_time_dict, memory_dict, peak_memory_dict = calibrate_with_cache(
        sched, snodes, comm_cache, comp_cache, memories_at_nodes, has_reduce_scatter, non_bucketable_pg
    )
    total_comp_time = sum(comp_time_dict.values())
    peak_memory = 0
    for idx, cur_memory in peak_memory_dict.items():
        peak_memory = max(cur_memory, peak_memory)
    # add memory offset if user wants to trade memory for more overlapping
    peak_memory = peak_memory + config.simplefsdp.peak_memory_offset

    schedule_fallback_operation = functools.partial(
        _schedule_fallback_operation,
        scheduler=sched,
        name_to_buf=name_to_buf,
        name_to_fused_node=name_to_fused_node,
    )
    release_steps = [0]

    # auto-bucketing plan
    st_time = time.time()
    fsdp_ag_idx = -1
    seen_new_fsdp_ag = True
    for idx, snode in enumerate(snodes):
        # we only bucket on FSDP comm
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ) and _check_ir_node_fsdp(snode.node, non_bucketable_pg):
            fsdp_ag_idx += 1
            seen_new_fsdp_ag = True
            total_comp_time -= comp_time_dict[fsdp_ag_idx]
            node_info = get_node_tensor_info(snode)[:-2] + get_ag_node_pg_info(snode)
            current_ag_bucket[node_info].append(snode)

            estimated_comm, comm_size_inp, comm_size_out = (
                estimate_hetero_bucketed_node(
                    current_ag_bucket,
                    schedule_fallback_operation,
                    name_to_buf,
                    "torch.ops._c10d_functional.all_gather_into_tensor.default",
                    comm_cache,
                )
            )
            break_overlap_criteria = heuristic_info["this_step_comp"] * (1 + config.simplefsdp.relax_ratio) < (
                estimated_comm
                + heuristic_info["last_step_rs_comm"]
            )
            if not has_reduce_scatter:
                break_comm_size_criteria = comm_cache.ag_max_inp_size < comm_size_inp
            else:
                break_comm_size_criteria = (
                    comm_cache.ag_max_inp_size < comm_size_inp
                    or comm_cache.rs_max_out_size
                    < heuristic_info["this_step_rs_comm_size"]
                )
            memory_threshold, current_peak_dynamic = get_dynamic_memory_threshold(
                peak_memory,
                peak_memory_dict,
                fsdp_ag_idx,
                # we have the last two steps because ag will be reordered to the previous ag-wait
                # TODO(ruisizhang123): this is a hacky way to ensure the memory budget is safe
                # probably need to update here if we have a better fine-grained memory estimation
                release_steps[-1] if len(release_steps) <= 1 else release_steps[-2],
                not has_reduce_scatter,
            )
            accumulated_comm_memory = (
                2
                * comm_size_inp  # copy-in (comm_size_inp) & copy-out (comm_size_out) memory created for AG
                + 2 * comm_size_out
                + heuristic_info[
                    "rs_comm_size_accumulated"
                ]  # accumulated gradient from reduce scatter
                + heuristic_info["this_step_rs_comm_size"]
                * 2
                * (1 + 1 * fsdp_world_size)
                + heuristic_info["last_step_rs_comm_size"]
            )
            break_memory_criteria = (
                memory_threshold
                < heuristic_info["next_step_memory"] + accumulated_comm_memory
            )
            if has_reduce_scatter and len(all_gather_plan) == 0:
                break_overlap_criteria = (
                    heuristic_info["next_step_comp"]
                    + heuristic_info["next_step_nonfsdp_comm"]
                ) * (1+config.simplefsdp.relax_ratio) < (
                    estimated_comm
                    + heuristic_info["last_step_rs_comm"]
                )
            if (
                break_overlap_criteria
                or break_memory_criteria
                or break_comm_size_criteria
            ):
                if heuristic_info["this_step_comp"] > 0:
                    overflow_ag = current_ag_bucket[node_info].pop()
                    all_gather_plan.append(current_ag_bucket)
                    current_ag_bucket = defaultdict(list)
                    current_ag_bucket[node_info].append(overflow_ag)
                else:
                    all_gather_plan.append(current_ag_bucket)
                    current_ag_bucket = defaultdict(list)
                if verbose:
                    print("********************")
                    print(
                        "break_overlap_criteria",
                        break_overlap_criteria,
                        heuristic_info["this_step_comp"],
                        "comm",
                        estimated_comm + heuristic_info["last_step_rs_comm"],
                    )
                    print(
                        "break_memory_criteria",
                        break_memory_criteria,
                        memory_threshold,
                        heuristic_info["next_step_memory"] + accumulated_comm_memory,
                    )
                    print("current_ag_bucket", all_gather_plan[-1])
                    for key, value in all_gather_plan[-1].items():
                        print("sub info current_ag_bucket", key, len(value), [v.node.get_name() for v in value])
                    print(
                        "current_peak_dynamic",
                        current_peak_dynamic,
                        "peak_memory",
                        peak_memory,
                        "config.simplefsdp.peak_memory_offset",
                        config.simplefsdp.peak_memory_offset,
                    )
                release_steps.append(idx + 1)
                if len(current_rs_bucket) > 0:
                    current_estimated_rs, rs_comm_size_inp, rs_comm_size_out = (
                        estimate_hetero_bucketed_node(
                            current_rs_bucket,
                            schedule_fallback_operation,
                            name_to_buf,
                            "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                            comm_cache,
                            fsdp_rs_reduce_op,
                        )
                    )
                    heuristic_info["last_step_rs_comm"] = current_estimated_rs
                    reduce_scatter_plan.append(current_rs_bucket)
                    for key, value in reduce_scatter_plan[-1].items():
                        print("sub info current_rs_bucket", key, len(value), [v.node.get_name() for v in value])
                    heuristic_info["last_step_rs_comm_size"] = 2 * (
                        rs_comm_size_inp + rs_comm_size_out
                    )  # rs copy-in + rs data
                    current_rs_bucket = defaultdict(list)

                (
                    heuristic_info["this_step_comp"],
                    heuristic_info["this_step_memory"],
                ) = (
                    heuristic_info["next_step_comp"]
                    + heuristic_info["next_step_nonfsdp_comm"],
                    heuristic_info["next_step_memory"],
                )
                (
                    heuristic_info["next_step_comp"],
                    heuristic_info["next_step_memory"],
                ) = 0, 0
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ) and _check_ir_node_fsdp(snode.node, non_bucketable_pg):
            node_info = get_node_tensor_info(snode)[:-2] + get_rs_node_pg_info(snode)
            current_rs_bucket[node_info].append(snode)

            heuristic_info["this_step_rs_comm"], _, rs_comm_size_out = (
                estimate_hetero_bucketed_node(
                    current_rs_bucket,
                    schedule_fallback_operation,
                    name_to_buf,
                    "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                    comm_cache,
                    fsdp_rs_reduce_op,
                )
            )
            heuristic_info["this_step_rs_comm_size"] = rs_comm_size_out
            # accumulated gradient from rs
            heuristic_info["rs_comm_size_accumulated"] += get_data_size(
                snode.node.layout.size
            )
            break_rs_overlap_criteria = (
                total_comp_time < heuristic_info["this_step_rs_comm"] * 5
            )
            if break_rs_overlap_criteria:
                heuristic_info["last_step_rs_comm"] = heuristic_info[
                    "this_step_rs_comm"
                ]
                heuristic_info["this_step_rs_comm"] = 0
                reduce_scatter_plan.append(current_rs_bucket)
                current_rs_bucket = defaultdict(list)
                for key, value in reduce_scatter_plan[-1].items():
                    print("sub info current_rs_bucket", key, len(value), [v.node.get_name() for v in value])
        else:
            # [TODO]ruisizhang: for now, we only consider TP and CP, whose comm are AG & RS & All_Reduce
            # For TP and CP, we consider the node as a "COMP" node with exposed communication as Comp time
            # the memory is the data fetched by the communication.
            if is_collective(snode.node):
                current_comm = comm_cache.get_comm_time(
                    snode.node.inputs[0].layout.size,
                    snode.node.layout.size,
                    getattr(snode.node, "python_kernel_name", ""),
                    calibrated=True,
                )
                current_memory = get_data_size(snode.node.layout.size)
                heuristic_info["next_step_nonfsdp_comm"] += current_comm
            else:
                # print("seen_new_fsdp_ag", seen_new_fsdp_ag, "fsdp_ag_idx", fsdp_ag_idx, "at comp_time_dict", comp_time_dict[fsdp_ag_idx], "next_step_comp", heuristic_info["next_step_comp"])
                if seen_new_fsdp_ag:
                    heuristic_info["next_step_memory"] += memory_dict[fsdp_ag_idx]
                    heuristic_info["next_step_comp"] += comp_time_dict[fsdp_ag_idx]
                    seen_new_fsdp_ag = False

    if len(current_ag_bucket) > 0 or len(all_gather_plan) == 0:
        all_gather_plan.append(current_ag_bucket)

    if len(current_rs_bucket) > 0 or len(reduce_scatter_plan) == 0:
        reduce_scatter_plan.append(current_rs_bucket)
    et_time = time.time()
    print("algorithm takes", et_time - st_time)
    return all_gather_plan, reduce_scatter_plan
