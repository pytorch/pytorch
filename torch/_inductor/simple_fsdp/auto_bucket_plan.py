# mypy: ignore-errors
import functools
import os
import pickle
import time
from collections import defaultdict

import torch
import torch.distributed as c10d
from torch.utils._ordered_set import OrderedSet
from torch.distributed.distributed_c10d import _resolve_process_group

from .. import config, memory
from ..utils import is_collective
from ..virtualized import V
from .bucket_utils import (
    _schedule_fallback_operation,
    bucket_all_gathers,
    bucket_reduce_scatters,
    get_fx_node,
    sync_dict_across_ranks,
    get_ag_node_pg_info,
    get_rs_node_pg_info,
)
from .estimator import (
    _create_real_tensor,
    benchmark_comm_func,
    estimate_comp_time,
    get_data_size,
    get_sample_list,
)
from .reorder import _check_ir_node_fsdp


def benchmark_and_cache_comm(
    size_list, cali_num_samples, tensor_info, comm_cache, comm_func_name
):
    samples = get_sample_list(size_list, cali_num_samples)
    input_dtype, input_device, output_dtype, output_device, group_size, process_group = tensor_info
    aggregated_samples = [s * group_size for s in samples]

    for sample, agg_sample in zip(samples, aggregated_samples):
        if (
            comm_func_name
            == "torch.ops._c10d_functional.all_gather_into_tensor.default"
        ):
            inp = sample
            out = agg_sample
        elif (
            comm_func_name == "torch.ops._c10d_functional.reduce_scatter_tensor.default"
        ):
            inp = agg_sample
            out = sample
        elif comm_func_name == "torch.ops._c10d_functional.all_reduce_.default":
            inp = sample
            out = sample

        inp = _create_real_tensor(torch.Size((inp,)), input_dtype, input_device)
        out = _create_real_tensor(torch.Size((out,)), output_dtype, output_device)
        time = benchmark_comm_func(
            inp,
            out,
            comm_func_name,
            comm_cache,
            group_size,
            process_group,
            estimate=False,
        )
        print(comm_func_name, "inp", inp.size(), "out", out.size(), "time", time)
        inp.cpu()
        out.cpu()
        del inp, out


def calibrate_with_cache(sched, snodes, comm_cache, comp_cache, has_reduce_scatter):
    world_size = c10d.distributed_c10d.get_world_size()

    fsdp_ag_input_size_list = []
    fsdp_rs_output_size_list = []
    if not config.simplefsdp.simplefsdp_only:
        non_fsdp_ag_input_size_dict = defaultdict(list)
        non_fsdp_rs_input_size_dict = defaultdict(list)
        all_reduce_input_size_dict = defaultdict(list)

    total_comp_time = 0

    cali_num_samples = config.simplefsdp.estimate_calibrate_number
    st_time = time.time()
    for idx, snode in enumerate(snodes):
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            if _check_ir_node_fsdp(snode.node):
                # For FSDP, we assume they have all have the same group size
                example_ag_fx_node = get_fx_node(
                    snode,
                    expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default,
                )
                fsdp_ag_group_size = example_ag_fx_node.args[1]
                fsdp_ag_process_group = _resolve_process_group(example_ag_fx_node.args[2])
                fsdp_ag_input_dtype, fsdp_ag_input_device = (
                    snode.node.inputs[0].layout.dtype,
                    snode.node.inputs[0].layout.device,
                )
                fsdp_ag_output_dtype, fsdp_ag_output_device = (
                    snode.node.layout.dtype,
                    snode.node.layout.device,
                )
                input_size = get_data_size(snode.node.inputs[0].layout.size)
                fsdp_ag_input_size_list.append(input_size)
            else:
                ag_fx_node = get_fx_node(
                    snode,
                    expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default,
                )
                group_size = ag_fx_node.args[1]
                process_group = _resolve_process_group(ag_fx_node.args[2])
                input_size = get_data_size(snode.node.inputs[0].data.get_size())
                tensor_info = tuple(
                    [
                        snode.node.inputs[0].layout.dtype,
                        snode.node.inputs[0].layout.device,
                        snode.node.layout.dtype,
                        snode.node.layout.device,
                        group_size,
                        process_group,
                    ]
                )
                non_fsdp_ag_input_size_dict[tensor_info].append(input_size)
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            if _check_ir_node_fsdp(snode.node):
                # For FSDP, we assume they have all have the same group size
                example_rs_fx_node = get_fx_node(
                    snode,
                    expected_op=torch.ops._c10d_functional.reduce_scatter_tensor.default,
                )
                fsdp_rs_group_size = example_rs_fx_node.args[2]
                fsdp_rs_process_group = _resolve_process_group(example_rs_fx_node.args[3])
                fsdp_rs_input_dtype, fsdp_rs_input_device = (
                    snode.node.inputs[0].layout.dtype,
                    snode.node.inputs[0].layout.device,
                )
                fsdp_rs_output_dtype, fsdp_rs_output_device = (
                    snode.node.layout.dtype,
                    snode.node.layout.device,
                )
                output_size = get_data_size(snode.node.layout.size)
                fsdp_rs_output_size_list.append(output_size)
            else:
                rs_fx_node = get_fx_node(
                    snode,
                    expected_op=torch.ops._c10d_functional.reduce_scatter_tensor.default,
                )
                group_size = rs_fx_node.args[2]
                process_group = _resolve_process_group(rs_fx_node.args[3])
                input_size = get_data_size(snode.node.layout.size)
                tensor_info = tuple(
                    [
                        snode.node.inputs[0].layout.dtype,
                        snode.node.inputs[0].layout.device,
                        snode.node.layout.dtype,
                        snode.node.layout.device,
                        group_size,
                        process_group,
                    ]
                )
                non_fsdp_rs_input_size_dict[tensor_info].append(input_size)
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.all_reduce_.default
        ):
            all_reduce_fx_node = get_fx_node(
                snode,
                expected_op=torch.ops._c10d_functional.all_reduce_.default,
            )
            all_reduce_process_group = _resolve_process_group(all_reduce_fx_node.args[2])
            group_size = all_reduce_fx_node.args[1]
            input_size = get_data_size(snode.node.inputs[0].layout.size)
            tensor_info = tuple(
                [
                    snode.node.inputs[0].layout.dtype,
                    snode.node.inputs[0].layout.device,
                    snode.node.inputs[0].layout.dtype,
                    snode.node.inputs[0].layout.device,
                    group_size,
                    all_reduce_process_group,
                ]
            )
            all_reduce_input_size_dict[tensor_info].append(input_size)
        else:
            if not is_collective(snode.node):
                comp_time = estimate_comp_time(
                    sched, snode, verbose=False, comp_cache=comp_cache
                )
                total_comp_time += comp_time
            else:
                print("[Relaxed Setting] untracked communication", snode.node.python_kernel_name)

    et_time = time.time()
    print("computation time estimation take", et_time - st_time)
    # Sync extern nodes
    extern_runtimes = sync_dict_across_ranks(comp_cache.extern_cache, world_size)
    comp_cache.extern_cache = extern_runtimes

    # Sync triton code
    triton_runtims = sync_dict_across_ranks(comp_cache.triton_cache, world_size)
    comp_cache.triton_cache = triton_runtims

    # Sync total compute time
    gathered_total_comp_time = [None for _ in range(world_size)]
    c10d.all_gather_object(gathered_total_comp_time, total_comp_time)
    total_comp_time = torch.median(torch.tensor(gathered_total_comp_time), dim=0).values

    if config.simplefsdp.load_cache and os.path.exists(
        config.simplefsdp.save_estimation_path
    ):
        with open(config.simplefsdp.save_estimation_path, "rb") as file:
            cache = pickle.load(file)
        comm_cache.cache = cache
        comm_cache._update_max_size()
        return total_comp_time

    # benchmark only in fwd to improve efficiency
    st_time = time.time()
    if len(fsdp_ag_input_size_list) > 0 and not has_reduce_scatter:
        tensor_info = [
            fsdp_ag_input_dtype,
            fsdp_ag_input_device,
            fsdp_ag_output_dtype,
            fsdp_ag_output_device,
            fsdp_ag_group_size,
            fsdp_ag_process_group,
        ]
        benchmark_and_cache_comm(
            fsdp_ag_input_size_list,
            cali_num_samples,
            tensor_info,
            comm_cache,
            "torch.ops._c10d_functional.all_gather_into_tensor.default",
        )

    if len(fsdp_rs_output_size_list) > 0:
        tensor_info = [
            fsdp_rs_input_dtype,
            fsdp_rs_input_device,
            fsdp_rs_output_dtype,
            fsdp_rs_output_device,
            fsdp_rs_group_size,
            fsdp_rs_process_group,
        ]
        benchmark_and_cache_comm(
            fsdp_rs_output_size_list,
            cali_num_samples,
            tensor_info,
            comm_cache,
            "torch.ops._c10d_functional.reduce_scatter_tensor.default",
        )

    if not config.simplefsdp.simplefsdp_only:
        if len(non_fsdp_ag_input_size_dict) > 0:
            for tensor_info, comm_list in non_fsdp_ag_input_size_dict.items():
                benchmark_and_cache_comm(
                    comm_list,
                    3,
                    list(tensor_info),
                    comm_cache,
                    "torch.ops._c10d_functional.all_gather_into_tensor.default",
                )

        if len(non_fsdp_rs_input_size_dict) > 0:
            for tensor_info, comm_list in non_fsdp_rs_input_size_dict.items():
                benchmark_and_cache_comm(
                    comm_list,
                    3,
                    list(tensor_info),
                    comm_cache,
                    "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                )

        if len(all_reduce_input_size_dict) > 0:
            for tensor_info, comm_list in all_reduce_input_size_dict.items():
                benchmark_and_cache_comm(
                    comm_list,
                    3,
                    list(tensor_info),
                    comm_cache,
                    "torch.ops._c10d_functional.all_reduce_.default",
                )

    et_time = time.time()

    print("communication time estimation takes", et_time - st_time)
    print("comm_cache.cache", len(comm_cache.cache))
    median_runtimes = sync_dict_across_ranks(comm_cache.cache, world_size)
    comm_cache.cache = median_runtimes
    comm_cache._update_max_size()
    with open(config.simplefsdp.save_estimation_path, "wb") as file:
        pickle.dump(comm_cache.cache, file)
    return total_comp_time


def get_dynamic_memory_threshold(
    peak_memory,
    memories_at_nodes,
    current_step,
    last_release_step,
    forward,
):
    # this function calculates the memory gap from the current step's peak memory
    # to the peak memory criteria
    if forward:
        # it calculates how much memory can be filled to meet peak memory criteria
        current_peak_memory = max(memories_at_nodes[:current_step])
    else:
        # maximum memory from last_release_step -> end in backward pass
        current_peak_memory = max(memories_at_nodes[last_release_step:])
    return peak_memory - current_peak_memory, current_peak_memory


def estimate_bucketed_node_list(
    current_node_bucket,
    schedule_fallback_operation,
    group_size,
    group_name,
    process_group,
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
            process_group,
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
        process_group,
        calibrated=True,
    )
    return estimated_comm, comm_size_inp, comm_size_out


def estimate_hetero_bucketed_node(
    current_node_bucket_dict,
    schedule_fallback_operation,
    process_group,
    name_to_buf,
    comm_func,
    comm_cache,
    reduce_op=None,
):
    estimated_comm, comm_size_inp, comm_size_out = 0, 0, 0
    for node_info, node_list in current_node_bucket_dict.items():
        group_size, group_name, input_dtype = node_info
        local_comm, local_comm_size_inp, local_comm_size_out = estimate_bucketed_node_list(
            node_list,
            schedule_fallback_operation,
            group_size,
            group_name,
            process_group,
            name_to_buf,
            comm_func,
            comm_cache,
            reduce_op,
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
        "multi_dim_comm_mem": 0,
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
    peak_memory, memories_at_nodes = memory.estimate_peak_memory(
        snodes, name_to_freeable_input_buf, graph_outputs
    )
    # add memory offset if user wants to trade memory for more overlapping
    peak_memory = peak_memory + config.simplefsdp.peak_memory_offset
    assert len(memories_at_nodes) == len(snodes) + 1

    # get basic info of ag/rs nodes
    world_size = c10d.distributed_c10d.get_world_size()
    group_size, group_name, reduce_op = None, None, None
    for idx, snode in enumerate(snodes):
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ) and _check_ir_node_fsdp(snode.node):
            example_ag_fx_node = get_fx_node(
                snode,
                expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default,
            )
            _, group_size, group_name = example_ag_fx_node.args
            fsdp_process_group = _resolve_process_group(example_ag_fx_node.args[2])
            if not has_reduce_scatter:
                break
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ) and _check_ir_node_fsdp(snode.node):
            example_rs_fx_node = get_fx_node(
                snode,
                expected_op=torch.ops._c10d_functional.reduce_scatter_tensor.default,
            )
            _, reduce_op, group_size, group_name = example_rs_fx_node.args
            fsdp_process_group = _resolve_process_group(example_rs_fx_node.args[3])
            break

    # if there is no collective comm., return dummy plan
    if group_size is None:
        return [[]], [[]]

    total_comp_time = calibrate_with_cache(
        sched, snodes, comm_cache, comp_cache, has_reduce_scatter
    )
    print("comm_cache", comm_cache.cache)
    schedule_fallback_operation = functools.partial(
        _schedule_fallback_operation,
        scheduler=sched,
        name_to_buf=name_to_buf,
        name_to_fused_node=name_to_fused_node,
    )
    release_steps = [0]

    # auto-bucketing plan
    st_time = time.time()
    for idx, snode in enumerate(snodes):
        # we only bucket on FSDP comm
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ) and _check_ir_node_fsdp(snode.node):
            node_info = get_ag_node_pg_info(snode)
            current_ag_bucket[node_info].append(snode)

            estimated_comm, comm_size_inp, comm_size_out = estimate_hetero_bucketed_node(
                current_ag_bucket,
                schedule_fallback_operation,
                fsdp_process_group,
                name_to_buf,
                "torch.ops._c10d_functional.all_gather_into_tensor.default",
                comm_cache,
            )
            break_overlap_criteria = heuristic_info["this_step_comp"] < (
                estimated_comm
                + heuristic_info["last_step_rs_comm"]
                * (1 + config.simplefsdp.relax_ratio)
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
                memories_at_nodes,
                idx + 1,
                # we have the last two steps because ag will be reordered to the previous ag-wait
                # TODO(ruisizhang123): this is a hacky way to ensure the memory budget is safe
                # probably need to update here if we have a better fine-grained memory estimation
                release_steps[-1] if len(release_steps) <= 1 else release_steps[-2],
                not has_reduce_scatter,
            )
            accumulated_comm_memory = (
                2 * comm_size_inp # copy-in (comm_size_inp) & copy-out (comm_size_out) memory created for AG
                + 2 * comm_size_out
                + heuristic_info[
                    "rs_comm_size_accumulated"
                ]  # accumulated gradient from reduce scatter
                + heuristic_info["this_step_rs_comm_size"] * 2 * (1 + 1 * world_size)
                + heuristic_info["last_step_rs_comm_size"]
            )
            break_memory_criteria = (
                memory_threshold
                < heuristic_info["next_step_memory"]
                + heuristic_info["multi_dim_comm_mem"]
                + accumulated_comm_memory
            )
            if has_reduce_scatter and len(all_gather_plan) == 0:
                break_overlap_criteria = heuristic_info["next_step_comp"] < (
                    estimated_comm
                    + heuristic_info["last_step_rs_comm"]
                    * (1 + config.simplefsdp.relax_ratio)
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
                            fsdp_process_group,
                            name_to_buf,
                            "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                            comm_cache,
                            reduce_op,
                        )
                    )
                    heuristic_info["last_step_rs_comm"] = current_estimated_rs
                    reduce_scatter_plan.append(current_rs_bucket)
                    heuristic_info["last_step_rs_comm_size"] = 2 * (
                        rs_comm_size_inp + rs_comm_size_out
                    )  # rs copy-in + rs data
                    current_rs_bucket = defaultdict(list)

                (
                    heuristic_info["this_step_comp"],
                    heuristic_info["this_step_memory"],
                ) = (
                    heuristic_info["next_step_comp"],
                    heuristic_info["next_step_memory"],
                )
                (
                    heuristic_info["next_step_comp"],
                    heuristic_info["next_step_memory"],
                ) = 0, 0
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ) and _check_ir_node_fsdp(snode.node):
            node_info = get_rs_node_pg_info(snode)
            current_rs_bucket[node_info].append(snode)

            heuristic_info["this_step_rs_comm"], _, rs_comm_size_out = (
                estimate_hetero_bucketed_node(
                    current_rs_bucket,
                    schedule_fallback_operation,
                    fsdp_process_group,
                    name_to_buf,
                    "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                    comm_cache,
                    reduce_op,
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
        else:
            # [TODO]ruisizhang: for now, we only consider TP and CP, whose comm are AG & RS & All_Reduce
            # For TP and CP, we consider the node as a "COMP" node with exposed communication as Comp time
            # the memory is the data fetched by the communication.
            if is_collective(snode.node, op=torch.ops._c10d_functional.all_reduce_.default):
                all_reduce_fx_node = get_fx_node(
                    snode,
                    expected_op=torch.ops._c10d_functional.all_reduce_.default,
                )
                process_group =  _resolve_process_group(all_reduce_fx_node.args[2])
                current_comp = comm_cache.get_comm_time(
                    snode.node.inputs[0].data.get_size(),
                    snode.node.layout.size,
                    getattr(snode.node, "python_kernel_name", ""),
                    process_group,
                    calibrated=True,
                )
                current_memory = get_data_size(snode.node.layout.size)
                #heuristic_info["multi_dim_comm_mem"] += current_memory
            else:
                current_comp = estimate_comp_time(
                    sched, snode, verbose=False, comp_cache=comp_cache
                )
                current_memory = max(
                    abs(
                        memories_at_nodes[idx + 1]
                        - memories_at_nodes[release_steps[-1]]
                    ),
                    heuristic_info["next_step_memory"],
                )
                heuristic_info["next_step_memory"] = current_memory
                total_comp_time -= current_comp
            heuristic_info["next_step_comp"] += current_comp

    if len(current_ag_bucket) > 0 or len(all_gather_plan) == 0:
        all_gather_plan.append(current_ag_bucket)

    if len(current_rs_bucket) > 0 or len(reduce_scatter_plan) == 0:
        reduce_scatter_plan.append(current_rs_bucket)
    et_time = time.time()
    print("algorithm takes", et_time - st_time)
    return all_gather_plan, reduce_scatter_plan
