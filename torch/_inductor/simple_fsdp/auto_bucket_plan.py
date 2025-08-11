# mypy: ignore-errors
import functools
import os
import pickle
import statistics
import time

import torch
import torch.distributed as c10d
from torch.utils._ordered_set import OrderedSet

from .. import config, memory
from ..utils import is_collective
from ..virtualized import V
from .bucket_utils import (
    _schedule_fallback_operation,
    bucket_all_gathers,
    bucket_reduce_scatters,
    get_fx_node,
)


def get_dynamic_memory_threshold(
    peak_memory,
    memories_at_nodes,
    current_step,
    last_release_step,
    forward,
):
    if forward:
        current_peak_memory = max(memories_at_nodes[:current_step])
    else:
        current_peak_memory = max(memories_at_nodes[last_release_step:])
    return peak_memory - current_peak_memory


def get_sample_list(input_size_list, cali_num_samples):
    input_size_min, input_size_max = (
        min(input_size_list),
        int(0.3 * sum(input_size_list)),
    )

    sample_list = [
        int(
            input_size_min
            + i * (input_size_max - input_size_min) / (cali_num_samples - 1)
        )
        for i in range(cali_num_samples)
    ]
    sample_list = [s // 100 * 100 for s in sample_list]
    return sample_list


def calibrate_with_cache(sched, snodes, comm_cache, comp_cache, has_reduce_scatter):
    from .estimator import (
        _create_real_tensor,
        benchmark_comm_func,
        estimate_comp_time,
        get_data_size,
    )
    world_size = c10d.distributed_c10d.get_world_size()

    ag_input_size_list = []
    rs_output_size_list = []
    total_comp_time = 0

    cali_num_samples = config.simplefsdp.estimate_calibrate_number
    st_time = time.time()
    for idx, snode in enumerate(snodes):
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            ag_input_dtype, ag_input_device = (
                snode.node.inputs[0].data.get_dtype(),
                snode.node.inputs[0].data.get_device(),
            )
            ag_output_dtype, ag_output_device = (
                snode.node.layout.dtype,
                snode.node.layout.device,
            )
            input_size = get_data_size(snode.node.inputs[0].data.get_size())
            ag_input_size_list.append(input_size)
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            rs_input_dtype, rs_input_device = (
                snode.node.inputs[0].data.get_dtype(),
                snode.node.inputs[0].data.get_device(),
            )
            rs_output_dtype, rs_output_device = (
                snode.node.layout.dtype,
                snode.node.layout.device,
            )
            output_size = get_data_size(snode.node.layout.size)
            rs_output_size_list.append(output_size)
        else:
            comp_time = estimate_comp_time(
                sched, snode, verbose=False, comp_cache=comp_cache
            )
            total_comp_time += comp_time
    et_time = time.time()
    print("computation time estimation take", et_time - st_time)
    # Sync extern nodes
    extern_runtimes = comp_cache.extern_cache
    gathered_extern_runtimes = [{} for _ in range(world_size)]
    c10d.all_gather_object(
        gathered_extern_runtimes,
        comp_cache.extern_cache,
        group=c10d.distributed_c10d._get_default_group(),
    )
    assert [
        gathered_runtime is not None for gathered_runtime in gathered_extern_runtimes
    ]
    for key in list(extern_runtimes.keys()):
        comm_value = [
            gathered_runtime[key] for gathered_runtime in gathered_extern_runtimes
        ]
        extern_runtimes[key] = statistics.median(comm_value)
    comp_cache.extern_cache = extern_runtimes

    # Sync triton code
    triton_runtims = comp_cache.triton_cache
    gathered_lists = [None for _ in range(world_size)]
    c10d.all_gather_object(gathered_lists, list(triton_runtims.values()))
    median_triton_time = torch.median(torch.tensor(gathered_lists), dim=0).values
    for idx, (key, value) in enumerate(triton_runtims.items()):
        triton_runtims[key] = median_triton_time[idx]
    comp_cache.triton_cache = triton_runtims

    gathered_total_comp_time = [None for _ in range(world_size)]
    c10d.all_gather_object(gathered_total_comp_time, total_comp_time)
    total_comp_time = torch.median(torch.tensor(gathered_total_comp_time), dim=0).values
    print("total_comp_time", total_comp_time)

    if config.simplefsdp.load_cache and os.path.exists(
        config.simplefsdp.save_estimation_path
    ):
        with open(config.simplefsdp.save_estimation_path, "rb") as file:
            cache = pickle.load(file)
        comm_cache.cache = cache
        comm_cache._update_max_size()
        print(
            "comm_cache.max numbers", comm_cache.ag_max_inp_size, comm_cache.rs_max_out_size
        )
        return total_comp_time

    # benchmark only fwd ag to improve efficiency
    st_time = time.time()
    if not has_reduce_scatter:
        ag_input_samples = get_sample_list(ag_input_size_list, cali_num_samples)
        ag_output_samples = [s * world_size for s in ag_input_samples]

        for inp, out in zip(ag_input_samples, ag_output_samples):
            comm_time = comm_cache.get_comm_time(
                torch.Size((inp,)),
                torch.Size((out,)),
                "torch.ops._c10d_functional.all_gather_into_tensor.default",
            )
            if comm_time is not None:
                continue
            inp = _create_real_tensor(
                torch.Size((inp,)), ag_input_dtype, ag_input_device
            )
            out = _create_real_tensor(
                torch.Size((out,)), ag_output_dtype, ag_output_device
            )
            comm_time = benchmark_comm_func(
                inp,
                out,
                "torch.ops._c10d_functional.all_gather_into_tensor.default",
                comm_cache,
                False,
            )
            print("AG inp", inp.size(), "out", out.size(), "comm", comm_time)

    # benchmark bwd rs
    if has_reduce_scatter and len(rs_output_size_list) > 0:
        rs_output_samples = get_sample_list(rs_output_size_list, cali_num_samples)
        rs_input_samples = [s * world_size for s in rs_output_samples]

        for inp, out in zip(rs_input_samples, rs_output_samples):
            inp = _create_real_tensor(
                torch.Size((inp,)), rs_input_dtype, rs_input_device
            )
            out = _create_real_tensor(
                torch.Size((out,)), rs_output_dtype, rs_output_device
            )
            comm_time = benchmark_comm_func(
                inp,
                out,
                "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                comm_cache,
                False,
            )
            print("RS inp", inp.size(), "out", out.size(), "comm", comm_time)
    et_time = time.time()
    print("communication time estimation takes", et_time - st_time)

    median_runtimes = comm_cache.cache
    gathered_runtimes = [{} for _ in range(world_size)]
    c10d.all_gather_object(
        gathered_runtimes,
        comm_cache.cache,
        group=c10d.distributed_c10d._get_default_group(),
    )
    assert [gathered_runtime is not None for gathered_runtime in gathered_runtimes]

    for key in list(median_runtimes.keys()):
        comm_value = [gathered_runtime[key] for gathered_runtime in gathered_runtimes]
        median_runtimes[key] = statistics.median(comm_value)
    comm_cache.cache = median_runtimes
    comm_cache._update_max_size()
    print(
        "comm_cache.max numbers", comm_cache.ag_max_inp_size, comm_cache.rs_max_out_size
    )
    with open(config.simplefsdp.save_estimation_path, "wb") as file:
        pickle.dump(comm_cache.cache, file)
    return total_comp_time


def estimate_bucketed_node(
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
    from .estimator import estimate_comp_time, get_data_size

    all_gather_plan = []
    reduce_scatter_plan = []
    current_ag_bucket = []
    current_rs_bucket = []
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
    assert len(memories_at_nodes) == len(snodes) + 1

    # get basic info of ag/rs nodes
    world_size = c10d.distributed_c10d.get_world_size()
    group_size, group_name, reduce_op = None, None, None
    for idx, snode in enumerate(snodes):
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            example_ag_fx_node = get_fx_node(
                snode,
                expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default,
            )
            _, group_size, group_name = example_ag_fx_node.args
            if not has_reduce_scatter:
                break
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            example_rs_fx_node = get_fx_node(
                snode,
                expected_op=torch.ops._c10d_functional.reduce_scatter_tensor.default,
            )
            _, reduce_op, group_size, group_name = example_rs_fx_node.args
            break

    # if there is no collective comm., return dummy plan
    if group_size is None:
        return [[]], [[]]

    total_comp_time = calibrate_with_cache(
        sched, snodes, comm_cache, comp_cache, has_reduce_scatter
    )
    schedule_fallback_operation = functools.partial(
        _schedule_fallback_operation,
        scheduler=sched,
        name_to_buf=name_to_buf,
        name_to_fused_node=name_to_fused_node,
    )
    release_steps = [0]

    # auto-bucketing plan
    st_time = time.time()
    if has_reduce_scatter:
        peak_memory = peak_memory + config.simplefsdp.peak_memory_offset
    for idx, snode in enumerate(snodes):
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            current_ag_bucket.append(snode)
            estimated_comm, comm_size_inp, comm_size_out = estimate_bucketed_node(
                current_ag_bucket,
                schedule_fallback_operation,
                group_size,
                group_name,
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
                break_comm_size_criteria = comm_cache.ag_max_inp_size < get_data_size(comm_size_inp)
            else:
                break_comm_size_criteria = (
                    comm_cache.ag_max_inp_size < get_data_size(comm_size_inp)
                    or comm_cache.rs_max_out_size < heuristic_info["this_step_rs_comm_size"]
                )
            if len(release_steps) > 1:
                last_step = release_steps[-2]
            else:
                last_step = release_steps[-1]
            memory_threshold = get_dynamic_memory_threshold(
                peak_memory,
                memories_at_nodes,
                idx + 1,
                last_step,
                not has_reduce_scatter,
            )
            accumulated_comm_memory_before = (
                2* get_data_size(comm_size_inp) # copy-in (comm_size_inp) & copy-out (comm_size_out) memory created for AG
                + 2* get_data_size(comm_size_out)
                + heuristic_info["rs_comm_size_accumulated"] # accumulated gradient from reduce scatter
                + heuristic_info["last_step_rs_comm_size"]
            )
            accumulated_comm_memory_after = (
                2* get_data_size(comm_size_inp) # copy-in (comm_size_inp) & copy-out (comm_size_out) memory created for AG
                + 2* get_data_size(comm_size_out)
                + heuristic_info["rs_comm_size_accumulated"] # accumulated gradient from reduce scatter
                + heuristic_info["this_step_rs_comm_size"] * 2 * (1+1*world_size) #+ heuristic_info["last_step_rs_comm_size"]
            )
            break_memory_criteria = (
                memory_threshold
                < heuristic_info["next_step_memory"] + accumulated_comm_memory_before or
                memory_threshold
                < heuristic_info["next_step_memory"] + accumulated_comm_memory_after
            )
            if has_reduce_scatter and len(all_gather_plan) == 0:
                break_overlap_criteria = heuristic_info["next_step_comp"] < (
                    estimated_comm
                    + heuristic_info["last_step_rs_comm"]
                    * (1 + config.simplefsdp.relax_ratio)
                )
            #if has_reduce_scatter and len(all_gather_plan) == 2:
            #    break_memory_criteria = True if len(current_ag_bucket) >= 3 else False
            if (
                break_overlap_criteria
                or break_memory_criteria
                or break_comm_size_criteria
            ):
                if heuristic_info["this_step_comp"] > 0:
                    overflow_ag = current_ag_bucket.pop()
                    all_gather_plan.append(current_ag_bucket)
                    current_ag_bucket = [overflow_ag]
                else:
                    all_gather_plan.append(current_ag_bucket)
                    current_ag_bucket = []
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
                        heuristic_info["next_step_memory"] + accumulated_comm_memory_before ,
                        heuristic_info["next_step_memory"] + accumulated_comm_memory_after,
                    )
                    print("current_ag_bucket", all_gather_plan[-1])
                release_steps.append(idx + 1)
                if len(current_rs_bucket) > 0:
                    current_estimated_rs, rs_comm_size_inp, rs_comm_size_out = (
                        estimate_bucketed_node(
                            current_rs_bucket,
                            schedule_fallback_operation,
                            group_size,
                            group_name,
                            name_to_buf,
                            "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                            comm_cache,
                            reduce_op,
                        )
                    )
                    heuristic_info["last_step_rs_comm"] = current_estimated_rs
                    reduce_scatter_plan.append(current_rs_bucket)
                    heuristic_info["last_step_rs_comm_size"] = 2*(get_data_size(rs_comm_size_inp) + get_data_size(rs_comm_size_out)) # rs copy-in + rs data
                    heuristic_info["rs_comm_size_accumulated"] += get_data_size(rs_comm_size_out) + get_data_size(rs_comm_size_inp) # accumulated gradient from rs
                    current_rs_bucket = []

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
        ):
            current_rs_bucket.append(snode)
            heuristic_info["this_step_rs_comm"], _, rs_comm_size_out = (
                estimate_bucketed_node(
                    current_rs_bucket,
                    schedule_fallback_operation,
                    group_size,
                    group_name,
                    name_to_buf,
                    "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                    comm_cache,
                    reduce_op,
                )
            )
            heuristic_info["this_step_rs_comm_size"] = get_data_size(rs_comm_size_out)
            break_rs_overlap_criteria = (
                total_comp_time < heuristic_info["this_step_rs_comm"] * 5
            )
            if break_rs_overlap_criteria:
                heuristic_info["last_step_rs_comm"] = heuristic_info[
                    "this_step_rs_comm"
                ]
                heuristic_info["this_step_rs_comm"] = 0
                reduce_scatter_plan.append(current_rs_bucket)
                current_rs_bucket = []
        else:
            comp = estimate_comp_time(
                sched, snode, verbose=False, comp_cache=comp_cache
            )
            heuristic_info["next_step_comp"] += comp
            heuristic_info["next_step_memory"] = max(
                abs(memories_at_nodes[idx + 1] - memories_at_nodes[release_steps[-1]]),
                heuristic_info["next_step_memory"],
            )
            total_comp_time -= comp

    if len(current_ag_bucket) > 0 or len(all_gather_plan) == 0:
        all_gather_plan.append(current_ag_bucket)

    if len(current_rs_bucket) > 0 or len(reduce_scatter_plan) == 0:
        reduce_scatter_plan.append(current_rs_bucket)
    print("release_steps", release_steps)
    et_time = time.time()

    print("algorithm takes", et_time - st_time)
    return all_gather_plan, reduce_scatter_plan
