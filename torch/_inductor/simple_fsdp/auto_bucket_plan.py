import functools
import statistics

import torch
import torch.distributed as c10d
from torch.utils._ordered_set import OrderedSet

from .. import config, memory
from ..utils import is_collective
from ..virtualized import V
from .bucket_utils import _schedule_fallback_operation, bucket_all_gathers, get_fx_node
from .estimator import (
    _create_real_tensor,
    benchmark_comm_func,
    estimate_comp_time,
    get_data_size,
)


def get_dynamic_memory_threshold(
    peak_memory,
    memories_at_nodes,
    current_step,
    last_release_step,
):
    increased_memory = (
        memories_at_nodes[current_step] - memories_at_nodes[last_release_step]
    )
    return peak_memory - increased_memory


def get_sample_list(input_size_list, cali_num_samples):
    input_size_min, input_size_thirdmax, input_size_secmax, input_size_max = (
        min(input_size_list),
        max(input_size_list),
        int(0.5 * sum(input_size_list)),
        int(0.9 * sum(input_size_list)),
    )

    sample_list = (
        [
            int(
                input_size_min
                + i
                * (input_size_thirdmax - input_size_min)
                / (cali_num_samples // 2 - 1)
            )
            for i in range(cali_num_samples // 4)
        ]
        + [
            int(
                input_size_thirdmax
                + i
                * (input_size_secmax - input_size_thirdmax)
                / (cali_num_samples // 2 - 1)
            )
            for i in range(cali_num_samples // 2)
        ]
        + [
            int(
                input_size_secmax
                + i * (input_size_max - input_size_secmax) / (cali_num_samples // 2 - 1)
            )
            for i in range(cali_num_samples // 4)
        ]
    )
    sample_list = [s // 100 * 100 for s in sample_list]
    return sample_list


def calibrate_with_cache(sched, snodes, comm_cache, comp_cache):
    world_size = c10d.distributed_c10d.get_world_size()

    ag_input_size_list = []
    rs_output_size_list = []
    cali_num_samples = config.simplefsdp.estimate_calibrate_number
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
            estimate_comp_time(sched, snode, verbose=False, comp_cache=comp_cache)

    if len(ag_input_size_list) == 0:
        return

    ag_input_samples = get_sample_list(ag_input_size_list, cali_num_samples)
    ag_output_samples = [s * world_size for s in ag_input_samples]

    for inp, out in zip(ag_input_samples, ag_output_samples):
        inp = _create_real_tensor(torch.Size((inp, 1)), ag_input_dtype, ag_input_device)
        out = _create_real_tensor(
            torch.Size((out, 1)), ag_output_dtype, ag_output_device
        )
        comm_time = benchmark_comm_func(
            inp,
            out,
            "torch.ops._c10d_functional.all_gather_into_tensor.default",
            comm_cache,
            False,
        )

    if len(rs_output_size_list) > 0:
        rs_output_samples = get_sample_list(rs_output_size_list, cali_num_samples)
        rs_input_samples = [s * world_size for s in rs_output_samples]

        for inp, out in zip(rs_input_samples, rs_output_samples):
            inp = _create_real_tensor(
                torch.Size((inp, 1)), rs_input_dtype, rs_input_device
            )
            out = _create_real_tensor(
                torch.Size((out, 1)), rs_output_dtype, rs_output_device
            )
            comm_time = benchmark_comm_func(
                inp,
                out,
                "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                comm_cache,
                False,
            )

    median_runtimes = comm_cache.cache
    if world_size > 1:
        gathered_runtimes = [{} for _ in range(world_size)]
        c10d.all_gather_object(
            gathered_runtimes,
            comm_cache.cache,
            group=c10d.distributed_c10d._get_default_group(),
        )
        assert [gathered_runtime is not None for gathered_runtime in gathered_runtimes]

        for key in list(median_runtimes.keys()):
            comm_value = [
                gathered_runtime[key] for gathered_runtime in gathered_runtimes
            ]
            median_runtimes[key] = statistics.median(comm_value)
    comm_cache.cache = median_runtimes

    extern_runtimes = comp_cache.extern_cache
    if world_size > 1:
        gathered_triton_runtimes = [{} for _ in range(world_size)]
        gathered_extern_runtimes = [{} for _ in range(world_size)]
        c10d.all_gather_object(
            gathered_triton_runtimes,
            comp_cache.triton_cache,
            group=c10d.distributed_c10d._get_default_group(),
        )
        c10d.all_gather_object(
            gathered_extern_runtimes,
            comp_cache.extern_cache,
            group=c10d.distributed_c10d._get_default_group(),
        )
        assert [
            gathered_runtime is not None
            for gathered_runtime in gathered_triton_runtimes
        ]
        assert [
            gathered_runtime is not None
            for gathered_runtime in gathered_extern_runtimes
        ]

        # for key in list(triton_runtimes.keys()):
        #     comm_value = [
        #         gathered_runtime[key] for gathered_runtime in gathered_triton_runtimes
        #     ]
        #     triton_runtimes[key] = statistics.median(comm_value)
        for key in list(extern_runtimes.keys()):
            comm_value = [
                gathered_runtime[key] for gathered_runtime in gathered_extern_runtimes
            ]
            extern_runtimes[key] = statistics.median(comm_value)
    # comp_cache.triton_cache = triton_runtimes
    comp_cache.extern_cache = extern_runtimes


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
    current_ag_bucket = []
    current_rs_bucket = []

    graph_outputs = OrderedSet(V.graph.get_output_names())
    graph_inputs = OrderedSet(V.graph.graph_inputs.keys())
    estimated_peak_memory, name_to_freeable_input_buf = memory.prepare_planning_info(
        snodes,
        name_to_buf,
        name_to_fused_node,
        graph_inputs,
        graph_outputs,
    )
    peak_memory, memories_at_nodes = memory.estimate_peak_memory(
        snodes, name_to_freeable_input_buf, graph_outputs
    )

    if config.simplefsdp.estimate_type == "calibrate":
        calibrate_with_cache(sched, snodes, comm_cache, comp_cache)

    heuristic_info = {
        "this_step_comm": 0,
        "this_step_comp": 0,
        "this_step_memory": 0,
        "next_step_comm": 0,
        "next_step_comp": 0,
        "next_step_memory": min(memories_at_nodes),
    }
    assert len(memories_at_nodes) == len(snodes) + 1

    schedule_fallback_operation = functools.partial(
        _schedule_fallback_operation,
        scheduler=sched,
        name_to_buf=name_to_buf,
        name_to_fused_node=name_to_fused_node,
    )
    group_size, group_name = None, None
    bucket_flag = False
    last_release_step = 0

    for idx, snode in enumerate(snodes):
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            if len(current_ag_bucket) == 0:
                current_ag_bucket.append(snode)
                if len(all_gather_plan) == 0:
                    example_ag_fx_node = get_fx_node(
                        snode,
                        expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default,
                    )
                    _, group_size, group_name = example_ag_fx_node.args
                    all_gather_plan.append(current_ag_bucket)
                    last_release_step = idx + 1
                    current_ag_bucket = []

                    if len(current_rs_bucket) > 0:
                        reduce_scatter_plan.append(current_rs_bucket)
                        current_rs_bucket = []
            else:
                current_ag_bucket.append(snode)
                ag_input_ir_nodes = [ag.node.inputs[0] for ag in current_ag_bucket]
                bucked_ag = bucket_all_gathers(
                    schedule_fallback_operation,
                    group_size,
                    group_name,
                    ag_input_ir_nodes,
                    current_ag_bucket,
                    name_to_buf,
                    return_ag_only=True,
                )
                # add cache here to accelerate?
                estimated_comm = comm_cache.get_comm_time(
                    bucked_ag[1].layout.size,
                    bucked_ag[2].layout.size,
                    "torch.ops._c10d_functional.all_gather_into_tensor.default",
                    calibrated=True,
                )
                break_comm_criteria = (
                    heuristic_info["this_step_comp"]
                    * (1 + config.simplefsdp.relax_ratio)
                    < estimated_comm
                )
                memory_threshold = get_dynamic_memory_threshold(
                    peak_memory, memories_at_nodes, idx + 1, last_release_step
                )
                break_memory_criteria = (
                    memory_threshold < heuristic_info["next_step_memory"]
                )
                if break_comm_criteria or break_memory_criteria:
                    if verbose:
                        print(
                            "break_comm_criteria",
                            break_comm_criteria,
                            "break_memory_criteria",
                            break_memory_criteria,
                        )
                    overflow_ag = current_ag_bucket.pop()
                    all_gather_plan.append(current_ag_bucket)
                    last_release_step = idx + 1
                    current_ag_bucket = [overflow_ag]
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
                    ) = 0, min(memories_at_nodes)

                    if len(current_rs_bucket) > 0:
                        reduce_scatter_plan.append(current_rs_bucket)
                        current_rs_bucket = []
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            if len(reduce_scatter_plan) == 0 and len(current_rs_bucket) == 0:
                current_rs_bucket.append(snode)
                reduce_scatter_plan.append(current_rs_bucket)
                current_rs_bucket = []
            else:
                current_rs_bucket.append(snode)
        else:
            # include  exclude ops to accelerate?
            comp = estimate_comp_time(
                sched, snode, verbose=False, comp_cache=comp_cache
            )
            heuristic_info["next_step_comp"] += comp
            heuristic_info["next_step_memory"] = (
                memories_at_nodes[idx + 1] - memories_at_nodes[last_release_step]
            )

    if len(current_ag_bucket) > 0:
        all_gather_plan.append(current_ag_bucket)

    if len(current_rs_bucket) > 0:
        reduce_scatter_plan.append(current_rs_bucket)

    if has_reduce_scatter:
        return all_gather_plan, reduce_scatter_plan
    return all_gather_plan
