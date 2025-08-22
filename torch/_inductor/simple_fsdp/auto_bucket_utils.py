import os
import pickle
import time
from collections import defaultdict

import torch
import torch.distributed as c10d
from torch.distributed.distributed_c10d import _resolve_process_group

from .. import config
from ..ir import NoneLayout
from ..utils import is_collective
from .bucket_utils import get_fx_node
from .estimator import (
    _create_real_tensor,
    benchmark_comm_func,
    estimate_comp_time,
    get_data_size,
    get_sample_list,
)
from .reorder import _check_ir_node_fsdp


def get_ag_node_pg_info(snode, resolve_pg=False):
    ag_fx_node = get_fx_node(
        snode,
        expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default,
    )
    if ag_fx_node is None:
        return None
    group_size, group_name = snode.node.constant_args[0], snode.node.constant_args[1]
    if resolve_pg:
        group_name = _resolve_process_group(group_name)
    return group_size, group_name


def get_rs_node_pg_info(snode, resolve_pg=False, return_reduce_op=False):
    rs_fx_node = get_fx_node(
        snode,
        expected_op=torch.ops._c10d_functional.reduce_scatter_tensor.default,
    )
    if rs_fx_node is None:
        return None
    group_size, group_name = snode.node.constant_args[1], snode.node.constant_args[2]
    if resolve_pg:
        group_name = _resolve_process_group(group_name)
    if return_reduce_op:
        reduce_op = rs_fx_node.args[1]
        return group_size, group_name, reduce_op
    return group_size, group_name


def get_all_reduce_node_pg_info(snode, resolve_pg=False):
    all_reduce_fx_node = get_fx_node(
        snode,
        expected_op=torch.ops._c10d_functional.all_reduce_.default,
    )
    if all_reduce_fx_node is None:
        return None
    group_size, group_name = all_reduce_fx_node.args[1], all_reduce_fx_node.args[2]
    if resolve_pg:
        group_name = _resolve_process_group(group_name)
    return group_size, group_name


def get_all_to_all_node_pg_info(snode, resolve_pg=False):
    all_to_all_fx_node = get_fx_node(
        snode,
        expected_op=torch.ops._c10d_functional.all_to_all_single.default,
    )
    if all_to_all_fx_node is None:
        return None
    group_size, group_name = 0, all_to_all_fx_node.args[3]
    if resolve_pg:
        group_name = _resolve_process_group(group_name)
    return group_size, group_name


def sync_dict_across_ranks(runtime_dict, world_size, group=None):
    gathered_lists = [None for _ in range(world_size)]
    c10d.all_gather_object(gathered_lists, list(runtime_dict.values()), group=group)
    median_gathered_time = torch.median(torch.tensor(gathered_lists), dim=0).values
    for idx, (key, value) in enumerate(runtime_dict.items()):
        runtime_dict[key] = median_gathered_time[idx]
    return runtime_dict


def get_node_tensor_info(snode):
    input_dtype, input_device = (
        snode.node.inputs[0].layout.dtype,
        snode.node.inputs[0].layout.device,
    )
    input_size = get_data_size(snode.node.inputs[0].layout.size)

    if not isinstance(snode.node.layout, NoneLayout):
        output_dtype, output_device = (
            snode.node.layout.dtype,
            snode.node.layout.device,
        )
        output_size = get_data_size(snode.node.layout.size)
    else:
        # In all_reduce, layout is NoneLayout
        # We set output info to be the same as input info as a special treatment
        output_dtype, output_device, output_size = input_dtype, input_device, input_size
    return (
        input_dtype,
        input_device,
        output_dtype,
        output_device,
        input_size,
        output_size,
    )


def benchmark_and_cache_comm(
    size_list, cali_num_samples, tensor_info, comm_cache, comm_func_name
):
    print("size_list", size_list)
    samples = get_sample_list(size_list, cali_num_samples)
    (
        input_dtype,
        input_device,
        output_dtype,
        output_device,
        group_size,
        process_group,
    ) = tensor_info
    aggregated_samples = [s * group_size for s in samples]
    print("samples", samples, "process_group", process_group)

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
        elif comm_func_name == "torch.ops._c10d_functional.all_to_all_single.default":
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


def benchmark_and_cache_comm_dicts(
    comm_cache, comm_dict, comm_func_name, cali_num_samples
):
    for node_info, comm_list in comm_dict.items():
        benchmark_and_cache_comm(
            comm_list,
            cali_num_samples,
            list(node_info),
            comm_cache,
            comm_func_name,
            # process_group,
        )
        print("bench node info", node_info, "comm_list size", len(comm_list))


def calibrate_with_cache(
    sched, snodes, comm_cache, comp_cache, memories_at_nodes, has_reduce_scatter, non_bucketable_pg
):
    world_size = c10d.distributed_c10d.get_world_size()

    fsdp_ag_input_size_dict = defaultdict(list)
    fsdp_rs_output_size_dict = defaultdict(list)
    if not config.simplefsdp.simplefsdp_only:
        non_fsdp_ag_input_size_dict = defaultdict(list)
        non_fsdp_rs_input_size_dict = defaultdict(list)
        all_reduce_input_size_dict = defaultdict(list)
        all_to_all_input_size_dict = defaultdict(list)

    cali_num_samples = config.simplefsdp.estimate_calibrate_number
    st_time = time.time()
    comp_time_dict = defaultdict(int)
    memory_dict = defaultdict(int)
    peak_memory_dict = defaultdict(int)
    fsdp_ag_idx = -1
    release_steps = [0]
    for idx, snode in enumerate(snodes):
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            fsdp_ag_idx += 1
            release_steps.append(idx)
            node_tensor_info = get_node_tensor_info(snode)
            node_pg_info = get_ag_node_pg_info(snode, resolve_pg=True)
            if node_pg_info is None:
                continue
            node_info = node_tensor_info[:-2] + node_pg_info
            input_size = node_tensor_info[-2]
            if _check_ir_node_fsdp(snode.node, non_bucketable_pg):
                # For FSDP, we assume they have all have the
                fsdp_ag_input_size_dict[node_info].append(input_size)
            else:
                non_fsdp_ag_input_size_dict[node_info].append(input_size)
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            node_tensor_info = get_node_tensor_info(snode)
            node_pg_info = get_rs_node_pg_info(snode, resolve_pg=True)
            if node_pg_info is None:
                continue
            node_info = node_tensor_info[:-2] + node_pg_info
            output_size = node_tensor_info[-1]
            if _check_ir_node_fsdp(snode.node, non_bucketable_pg):
                # For FSDP, we assume they have all have the same group size
                fsdp_rs_output_size_dict[node_info].append(output_size)
            else:
                non_fsdp_rs_input_size_dict[node_info].append(output_size)
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.all_reduce_.default
        ):
            node_tensor_info = get_node_tensor_info(snode)
            node_pg_info = get_all_reduce_node_pg_info(snode, resolve_pg=True)
            if node_pg_info is None:
                continue
            node_info = node_tensor_info[:-2] + node_pg_info
            input_size = node_tensor_info[-2]
            all_reduce_input_size_dict[node_info].append(input_size)
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.all_to_all_single.default
        ):
            node_tensor_info = get_node_tensor_info(snode)
            node_pg_info = get_all_to_all_node_pg_info(snode, resolve_pg=True)
            if node_pg_info is None:
                continue
            node_info = node_tensor_info[:-2] + node_pg_info
            input_size = node_tensor_info[-2]
            all_to_all_input_size_dict[node_info].append(input_size)
        else:
            if not is_collective(snode.node):
                comp_time = estimate_comp_time(
                    sched, snode, verbose=False, comp_cache=comp_cache
                )
                comp_time_dict[fsdp_ag_idx] += comp_time
                memory_dict[fsdp_ag_idx] = max(
                    abs(
                        memories_at_nodes[idx + 1]
                        - memories_at_nodes[release_steps[-1]]
                    ),
                    memory_dict[fsdp_ag_idx],
                )
                peak_memory_dict[fsdp_ag_idx] = max(
                    memories_at_nodes[idx + 1], peak_memory_dict[fsdp_ag_idx]
                )
            else:
                print(
                    "[Relaxed Setting] untracked communication",
                    snode.node.python_kernel_name,
                )

    et_time = time.time()
    print("computation time estimation take", et_time - st_time)
    # Sync total compute time
    comp_time_dict = sync_dict_across_ranks(comp_time_dict, world_size)
    memory_dict = sync_dict_across_ranks(memory_dict, world_size)
    peak_memory_dict = sync_dict_across_ranks(peak_memory_dict, world_size)

    if config.simplefsdp.load_cache and os.path.exists(
        config.simplefsdp.save_estimation_path
    ):
        with open(config.simplefsdp.save_estimation_path, "rb") as file:
            cache = pickle.load(file)
        comm_cache.cache = cache
        comm_cache._update_max_size()
        return comp_time_dict, memory_dict, peak_memory_dict

    # benchmark only in fwd to improve efficiency
    st_time = time.time()

    if len(fsdp_ag_input_size_dict) > 0 and not has_reduce_scatter:
        benchmark_and_cache_comm_dicts(
            comm_cache,
            fsdp_ag_input_size_dict,
            "torch.ops._c10d_functional.all_gather_into_tensor.default",
            cali_num_samples,
        )

    if len(fsdp_rs_output_size_dict) > 0:
        benchmark_and_cache_comm_dicts(
            comm_cache,
            fsdp_rs_output_size_dict,
            "torch.ops._c10d_functional.reduce_scatter_tensor.default",
            cali_num_samples,
        )

    if not config.simplefsdp.simplefsdp_only:
        if len(non_fsdp_ag_input_size_dict) > 0:
            benchmark_and_cache_comm_dicts(
                comm_cache,
                non_fsdp_ag_input_size_dict,
                "torch.ops._c10d_functional.all_gather_into_tensor.default",
                3,
            )

        if len(non_fsdp_rs_input_size_dict) > 0:
            benchmark_and_cache_comm_dicts(
                comm_cache,
                non_fsdp_rs_input_size_dict,
                "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                3,
            )

        if len(all_reduce_input_size_dict) > 0:
            benchmark_and_cache_comm_dicts(
                comm_cache,
                all_reduce_input_size_dict,
                "torch.ops._c10d_functional.all_reduce_.default",
                3,
            )

        if len(all_to_all_input_size_dict) > 0:
            benchmark_and_cache_comm_dicts(
                comm_cache,
                all_to_all_input_size_dict,
                "torch.ops._c10d_functional.all_to_all_single.default",
                3,
            )

    et_time = time.time()

    print("communication time estimation takes", et_time - st_time)
    median_runtimes = sync_dict_across_ranks(comm_cache.cache, world_size)
    comm_cache.cache = median_runtimes
    comm_cache._update_max_size()
    with open(config.simplefsdp.save_estimation_path, "wb") as file:
        pickle.dump(comm_cache.cache, file)
    return comp_time_dict, memory_dict, peak_memory_dict
