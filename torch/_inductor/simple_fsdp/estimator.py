from typing import TYPE_CHECKING

import os
import itertools
import torch
import time
import pickle
from typing import List

import torch.distributed as c10d
import torch.utils._pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch._inductor.codecache import PyCodeCache
from .. import scheduler, ir
from ..utils import (
    contains_collective,
    contains_wait,
    is_collective,
    is_fallback_op,
    is_wait,
)
from torch._C import TensorType
from torch.utils._ordered_set import OrderedSet



# TODO (ruisizhang123): add more communication ops here
kernel_name_to_comm_op = {
    "torch.ops._c10d_functional.all_gather_into_tensor.default": c10d.all_gather_into_tensor,
    "torch.ops._c10d_functional.reduce_scatter_tensor.default": c10d.reduce_scatter_tensor,
}


kernel_name_to_comp_op = {
    "extern_kernels.mm": torch.ops.aten.mm,
    "extern_kernels.bmm": torch.ops.aten.bmm,
    "extern_kernels.addmm": torch.ops.aten.addmm,
}


def convert_str_to_op(full_name):
    module_names= full_name.split(".")
    target_kernel = torch
    for module_name in module_names:
        target_kernel = getattr(target_kernel, module_name)
    return target_kernel


def create_real_tensor(size, dtype, device):
    out = torch.randn(size, dtype=dtype).to(device)
    return out


def estimate_runtime(sched: "scheduler.Scheduler", snodes: list["scheduler.BaseSchedulerNode"], verbose=False):
    runtimes = {}
    for idx, snode in enumerate(snodes):
        runtimes[snode.get_name()] = estimate_op_runtime(sched, snode, verbose=verbose)
    return runtimes


def estimate_op_runtime(sched: "scheduler.Scheduler", snode: "scheduler.BaseSchedulerNode", verbose=False):
    runtime = {"COMM": 0, "COMP": 0, "MEMORY": 0}

    if contains_collective(snode):
        ## benchmark the communication time here
        runtime["COMM"] = estimate_comm_time(sched, snode, verbose=verbose)
        return runtime
    elif contains_wait(snode):
        ## a wait node here
        return runtime

    runtime["COMP"], runtime["MEMORY"] = estimate_comp_time(sched, snode, verbose=verbose)
    return runtime


def estimate_comm_time(sched: "scheduler.Scheduler", snode: "scheduler.Scheduler", estimate=False, verbose=False):
    kernel = snode.node
    world_size = c10d.distributed_c10d.get_world_size()
    rank = c10d.distributed_c10d.get_rank()
    assert len(kernel.inputs) == 1
    inputs = kernel.inputs[0]
    comm_func = kernel_name_to_comm_op.get(getattr(kernel, "python_kernel_name", ""))

    device = torch.device(f"cuda:{rank:d}")
    process_group = c10d.distributed_c10d._get_default_group()

    tensor_input = create_real_tensor(inputs.data.get_size(), inputs.data.get_dtype(), inputs.data.get_device())
    tensor_output = create_real_tensor(kernel.layout.size, kernel.layout.dtype, kernel.layout.device)

    if estimate:
        with c10d._time_estimator(group=process_group, device=device) as cm:
            comm_func(tensor_output, tensor_input)
        comm_time = cm.estimated_time
    else:
        torch.cuda.synchronize()
        nwarms = 2
        for _ in range(nwarms):
            c10d.barrier()
            comm_func(tensor_output, tensor_input)
            torch.cuda.synchronize()

        nruns = 4
        comm_time = 0
        test_list = []

        for _ in range(nruns):
            c10d.barrier()
            torch.cuda.synchronize()

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            comm_func(tensor_output, tensor_input)
            end_evt.record()
            end_evt.synchronize()

            torch.cuda.synchronize()

            current_run_time = start_evt.elapsed_time(end_evt)
            comm_time += current_run_time
            test_list.append(current_run_time)
        comm_time = comm_time / nruns * 1e3
    if verbose:
        print("[COMM Node]", getattr(kernel, "python_kernel_name", ""), "level", get_block_level(sched, snode), "time", comm_time)
    del tensor_input, tensor_output
    return comm_time


def estimate_comp_time(sched: "scheduler.Scheduler", snode: "scheduler.BaseSchedulerNode", verbose=False):
    device = snode.get_device()

    kernel_name = ""
    if isinstance(snode, scheduler.FusedSchedulerNode):
        node_list = snode.snodes
        for n in node_list:
            kernel_name += str(n.node.origin_node)
    elif isinstance(snode, scheduler.ExternKernelSchedulerNode):
        time, _, extern_kernel_name = benchmark_extern_node(snode.node)
        if verbose and time != 0:
            print("[COMP Node] EXTERN", extern_kernel_name, "level", get_block_level(sched, snode), "time", time)
        return time, 0
    elif isinstance(snode, scheduler.BaseSchedulerNode):
        node_list = [snode]
        kernel_name = str(snode.node.origin_node)
    else:
        raise ValueError(f"Unsupported snode type {type(snode)}")

    src_code = sched.generate_kernel_code_from_nodes(
        node_list, benchmark_kernel=True
    )
    module = PyCodeCache.load(src_code)

    time, _ = sched.benchmark_codegened_module(module=module, device=device)
    time = time * 1000
    if verbose and time != 0:
        print("[COMP Node] BASE/FUSE", kernel_name, "level", get_block_level(sched, snode), "time", time)
    return time, 0


def benchmark_extern_node(node):
    cls = node.__class__
    if isinstance(node, ir.MultiOutput):
        return 0, 0, ""

    python_kernel_name = getattr(node, "python_kernel_name", "")

    if python_kernel_name.startswith("extern_kernels"):
        func = kernel_name_to_comp_op.get(
            python_kernel_name, None
        )
    elif python_kernel_name.startswith("torch.ops.aten"):
        func = convert_str_to_op(python_kernel_name)
    else:
        func = None

    if func is None:
        return 0, 0, ""
    else:
        if isinstance(node, ir.FallbackKernel):
            args = node.export_extern_kernel_node()
            ordered_kwargs_length = len(node.ordered_kwargs_for_cpp_kernel)
            if ordered_kwargs_length > 0:
                args, ordered_kwargs = args[:-1*ordered_kwargs_length], args[-1*ordered_kwargs_length:]
                ordered_kwargs = {k: v for k, v in zip(node.ordered_kwargs_for_cpp_kernel, ordered_kwargs)}
                node.kwargs.update(ordered_kwargs)
        else:
            args = node.inputs
            args = node.fill_non_provided_args([*args, *node.constant_args], node.kwargs)

        flat_args, args_property = pytree.tree_flatten((args, node.kwargs))
        flat_args = [
            ir.ir_node_to_tensor(input, guard_shape=False) if isinstance(input, ir.IRNode) and not isinstance(input, ir.GeneratorState) else input
            for input in flat_args
        ]

        with no_dispatch():

            def to_real_tensor(e):
                if not isinstance(e, torch.Tensor):
                    return e
                if torch.is_floating_point(e):
                    out = torch.rand_like(e, device=e.device)
                else:
                    out = torch.ones_like(e, device=e.device)
                if e.is_sparse:
                    out._coalesced_(e.is_coalesced())
                return out

            flat_args = [to_real_tensor(a) for a in flat_args]
            args, kwargs = pytree.tree_unflatten(flat_args, args_property)

            func(*args, **kwargs)
            num_iters = 3
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            cpu_start = time.time()
            start_event.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                r = None
                r = func(*args, **kwargs)
            end_event.record(torch.cuda.current_stream())
            cpu_end = time.time()
            torch.cuda.synchronize()
            cpu_time = (cpu_end - cpu_start)
            total_op_time = start_event.elapsed_time(end_event) - cpu_time
            mean_op_time = total_op_time / num_iters
            del flat_args

    mean_op_time = mean_op_time * 1e3
    return mean_op_time, 0, getattr(node, "python_kernel_name", "")


def get_block_level(sched: "scheduler.Scheduler", node: "scheduler.BaseSchedulerNode") -> str:
    """
    Get the node's block name
    """
    if isinstance(node, scheduler.FusedSchedulerNode):
        node_list = node.snodes
    else:
        node_list = [node]

    node_origin_list = [origin for n in node_list for origin in list(n.node.origins)]
    module_list = []
    for n in node_origin_list:
        module_stack = n.meta.get("nn_module_stack", {})
        fwd_nn_module_stack = n.meta.get("fwd_nn_module_stack", {})
        if module_stack != {}:
            layer_info, block_info = list(module_stack.values())[0]
            module_list.append(layer_info)
        elif fwd_nn_module_stack != {}:
            layer_info, block_info = list(fwd_nn_module_stack.values())[0]
            module_list.append(layer_info)
    node_module = list(OrderedSet(module_list))
    if len(node_module) > 0:
        return node_module[0]
    return ""
