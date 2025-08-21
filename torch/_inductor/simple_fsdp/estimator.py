import statistics
import time
from collections.abc import Sequence
from functools import reduce
from typing import Any, Callable, cast, Union

from sympy import Expr

import torch
import torch.distributed as c10d
import torch.utils._pytree as pytree
from torch._inductor.codecache import PyCodeCache
from torch.utils._mode_utils import no_dispatch

from .. import config, ir
from ..utils import contains_collective, contains_wait


kernel_name_to_comm_op: dict[str, Callable[..., Any]] = {
    "torch.ops._c10d_functional.all_gather_into_tensor.default": c10d.all_gather_into_tensor,
    "torch.ops._c10d_functional.reduce_scatter_tensor.default": c10d.reduce_scatter_tensor,
    "torch.ops._c10d_functional.all_gather_into_tensor_out.default": c10d.all_gather_into_tensor,
    "torch.ops._c10d_functional.all_reduce_.default": c10d.all_reduce,
    "torch.ops._c10d_functional.all_to_all_single.default": c10d.all_to_all_single,
}


kernel_name_to_comp_op: dict[str, Callable[..., Any]] = {
    "extern_kernels.mm": torch.ops.aten.mm,
    "extern_kernels.bmm": torch.ops.aten.bmm,
    "extern_kernels.addmm": torch.ops.aten.addmm,
}

OpType = Union[
    torch._ops.OpOverload,
    torch._ops.OpOverloadPacket,
    torch._ops.HigherOrderOperator,
]


def get_sample_list(input_size_list, cali_num_samples):
    input_size_min, input_size_max = (
        min(input_size_list),
        int(config.simplefsdp.benchmark_ratio * sum(input_size_list)),
    )
    # ensure the min transmitted data volume is not 0
    input_size_min = max(100, input_size_min)
    sample_list = [
        int(
            input_size_min
            + i * (input_size_max - input_size_min) / (cali_num_samples - 1)
        )
        for i in range(cali_num_samples)
    ]
    sample_list = [s // 100 * 100 for s in sample_list]
    return sample_list


def _convert_str_to_op(full_name: str) -> OpType:
    module_names = full_name.split(".")
    target_kernel = torch
    for module_name in module_names:
        target_kernel = getattr(target_kernel, module_name)
    assert isinstance(
        target_kernel,
        (
            torch._ops.OpOverload,
            torch._ops.OpOverloadPacket,
            torch._ops.HigherOrderOperator,
        ),
    )
    return target_kernel


def _create_real_tensor(
    size: Union[torch.Size, Sequence[Expr]],
    dtype: torch.dtype,
    device: Union[torch.device, None],
) -> torch.Tensor:
    if dtype.is_floating_point:
        out = torch.randn(size, dtype=dtype).to(device)
    else:
        out = torch.ones(size, dtype=dtype).to(device)
    return out


def get_data_size(size):
    return reduce(lambda x, y: x * y, size)


class CommPerfCache:
    def __init__(self, threshold=3000):
        self.cache = {}
        self.threshold = threshold
        self.ag_max_inp_size = -1
        self.rs_max_out_size = -1
        self.all_reduce_max_input_size = -1
        self.all_to_all_max_input_size = -1

    def _calculate_distance(self, size1, size2):
        word_size1 = get_data_size(size1)
        word_size2 = get_data_size(size2)
        return abs(word_size1 - word_size2)

    def _update_max_size(self):
        for k in self.cache.keys():
            if k[2] == "torch.ops._c10d_functional.all_gather_into_tensor.default":
                self.ag_max_inp_size = max(
                    self.ag_max_inp_size, get_data_size(list(k[0]))
                )
            if k[2] == "torch.ops._c10d_functional.reduce_scatter_tensor.default":
                self.rs_max_out_size = max(
                    self.rs_max_out_size, get_data_size(list(k[1]))
                )
            if k[2] == "torch.ops._c10d_functional.all_reduce_.default":
                self.all_reduce_max_input_size = max(
                    self.all_reduce_max_input_size, get_data_size(list(k[0]))
                )
            if k[2] == "torch.ops._c10d_functional.all_to_all_single.default":
                self.all_to_all_max_input_size = max(
                    self.all_to_all_max_input_size, get_data_size(list(k[0]))
                )

    def add_comm_time(self, tensor_input_size, tensor_output_size, comm_func, value):
        key = (tuple(tensor_input_size), tuple(tensor_output_size), comm_func)
        self.cache[key] = value
        if comm_func == "torch.ops._c10d_functional.all_gather_into_tensor.default":
            self.ag_max_inp_size = max(
                self.ag_max_inp_size, get_data_size(tensor_input_size)
            )
        if comm_func == "torch.ops._c10d_functional.reduce_scatter_tensor.default":
            self.rs_max_out_size = max(
                self.rs_max_out_size, get_data_size(tensor_output_size)
            )
        if comm_func == "torch.ops._c10d_functional.all_reduce_.default":
            self.all_reduce_max_input_size = max(
                self.all_reduce_max_input_size, get_data_size(tensor_input_size)
            )
        if comm_func == "torch.ops._c10d_functional.all_to_all_single.default":
            self.all_to_all_max_input_size = max(
                self.all_to_all_max_input_size, get_data_size(tensor_input_size)
            )

    def get_comm_time(
        self, tensor_input_size, tensor_output_size, comm_func, calibrated=False
    ):
        key = (tuple(tensor_input_size), tuple(tensor_output_size), comm_func)
        if key in self.cache:
            return self.cache[key]

        if calibrated:
            threshold = float("inf")
        else:
            threshold = self.threshold
        closest_key = None
        closest_distance = float("inf")

        for k in self.cache.keys():
            if k[2] == comm_func:
                input_distance = self._calculate_distance(tensor_input_size, k[0])
                output_distance = self._calculate_distance(tensor_output_size, k[1])
                total_distance = input_distance + output_distance
                if (
                    input_distance <= threshold
                    and output_distance <= threshold
                    and total_distance < closest_distance
                ):
                    closest_distance = total_distance
                    closest_key = k

        if closest_key:
            return self.cache[closest_key]

        # fall back to 0 if the data has been calibrated, but we cannot find a match
        if calibrated:
            return 0
        return None


class CompPerfCache:
    def __init__(self):
        self.triton_cache = {}
        self.extern_cache = {}

    def add_triton_runtime(self, triton_code, runtime):
        self.triton_cache[triton_code] = runtime

    def add_extern_runtime(self, kernel_args, runtime):
        self.extern_cache[kernel_args] = runtime

    def get_runtime_by_triton(self, triton_code):
        return self.triton_cache.get(triton_code, None)

    def get_runtime_by_extern(self, kernel_args):
        return self.extern_cache.get(kernel_args, None)


def estimate_runtime(
    sched: "scheduler.Scheduler",
    snodes: list["scheduler.BaseSchedulerNode"],
    verbose: bool = False,
) -> dict[str, dict[str, float]]:
    # The runtimes dict containts the estimated runtime of each node
    # For each node, the key is the node name, and the value is a dict of {"COMM": comm_time, "COMP": comp_time}
    # If the node is a collective node, the value is {"COMM": comm_time, "COMP": 0.}
    # If the node is a compute node, the value is {"COMM": 0., "COMP": comp_time}
    # If the node is a wait node, the value is {"COMM": 0., "COMP": 0.}
    runtimes = {}

    # Get the runtime of each rank
    for _, snode in enumerate(snodes):
        runtimes[snode.get_name()] = estimate_op_runtime(sched, snode, verbose=verbose)

    # If world_size is larger than 1, gather runtimes from each rank and sync the median runtime across ranks
    world_size = c10d.distributed_c10d.get_world_size()
    median_runtimes = runtimes
    if world_size > 1:
        gathered_runtimes: list[dict[str, dict[str, float]]] = [
            {} for _ in range(world_size)
        ]
        c10d.all_gather_object(
            gathered_runtimes,
            runtimes,
            group=c10d.distributed_c10d._get_default_group(),
        )
        assert [len(gathered_runtime) > 0 for gathered_runtime in gathered_runtimes]

        for key in list(runtimes.keys()):
            comm_value = [
                gathered_runtime[key]["COMM"] for gathered_runtime in gathered_runtimes
            ]
            comp_value = [
                gathered_runtime[key]["COMP"] for gathered_runtime in gathered_runtimes
            ]
            median_runtimes[key] = {
                "COMM": statistics.median(comm_value),
                "COMP": statistics.median(comp_value),
            }

    return median_runtimes


def estimate_op_runtime(
    sched: "scheduler.Scheduler",
    snode: "scheduler.BaseSchedulerNode",
    verbose: bool = False,
) -> dict[str, float]:
    runtime = {"COMM": 0.0, "COMP": 0.0}

    if contains_collective(snode):
        # benchmark communication node runtime
        runtime["COMM"] = estimate_comm_time(sched, snode, verbose=verbose)
        return runtime
    elif contains_wait(snode):
        # wait node
        return runtime

    runtime["COMP"] = estimate_comp_time(sched, snode, verbose=verbose)
    return runtime


def estimate_comm_time(
    sched: "scheduler.Scheduler",
    snode: Union[tuple["ir.IRNode"], "scheduler.BaseSchedulerNode"],
    estimate: bool = False,
    verbose: bool = False,
    comm_cache: "CommPerfCache" = None,
) -> float:
    # TODO (ruisizhang123): add more types of collective communication.
    # Currently, it only supports all_gather and reduce_scatter
    # estimate set to True: return NCCL's estimated comm time (https://github.com/pytorch/pytorch/pull/149343)
    # estimate set to False: run the collective communication and return the actual comm time
    from ..scheduler import BaseSchedulerNode

    # for node with collective kernel estimation
    if isinstance(snode, BaseSchedulerNode):
        kernel = snode.node
        assert hasattr(kernel.inputs[0], "data")
        assert isinstance(kernel.layout, ir.Layout)
        inputs = kernel.inputs[0]
    # for node with fallbackkernel in bucketing estimation
    elif isinstance(snode, tuple):
        node, inputs, outputs = snode
        kernel = node.inputs[0]

    if isinstance(snode, BaseSchedulerNode):
        tensor_input = _create_real_tensor(
            inputs.data.get_size(), inputs.data.get_dtype(), inputs.data.get_device()
        )
        tensor_output = _create_real_tensor(
            kernel.layout.size, kernel.layout.dtype, kernel.layout.device
        )
    elif isinstance(snode, tuple):
        tensor_input = _create_real_tensor(
            inputs.layout.size, inputs.layout.dtype, inputs.layout.device
        )
        tensor_output = _create_real_tensor(
            outputs.layout.size, outputs.layout.dtype, outputs.layout.device
        )

    comm_time = benchmark_comm_func(
        tensor_input,
        tensor_output,
        getattr(kernel, "python_kernel_name", ""),
        comm_cache,
        estimate,
        verbose=verbose,
    )
    tensor_input.cpu()
    tensor_output.cpu()
    del tensor_input, tensor_output
    return comm_time


def benchmark_comm_func(
    tensor_input,
    tensor_output,
    comm_func_name,
    comm_cache,
    group_size,
    process_group,
    estimate,
    verbose=False,
):
    rank = c10d.distributed_c10d.get_rank()
    device = torch.device(f"cuda:{rank:d}")

    if comm_func_name == "torch.ops._c10d_functional.all_gather_into_tensor.default":
        input_args = {"input_tensor": tensor_input, "output_tensor": tensor_output}
    elif comm_func_name == "torch.ops._c10d_functional.reduce_scatter_tensor.default":
        input_args = {"input": tensor_input, "output": tensor_output}
    elif comm_func_name == "torch.ops._c10d_functional.all_reduce_.default":
        input_args = {"tensor": tensor_input}
    elif comm_func_name == "torch.ops._c10d_functional.all_to_all_single.default":
        input_args = {"input": tensor_input, "output": tensor_output}

    comm_func = kernel_name_to_comm_op.get(comm_func_name, None)
    assert comm_func is not None, f"Unsupported comm op {comm_func}"

    if estimate:
        with c10d._time_estimator(group=process_group, device=device) as cm:
            comm_func(**input_args)
        comm_time = cm.estimated_time
    else:
        torch.cuda.synchronize()
        comm_func(**input_args, group=process_group)

        nruns = 2
        comm_time = 0
        for _ in range(nruns):
            c10d.barrier()
            torch.cuda.synchronize()

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            comm_func(**input_args, group=process_group)
            end_evt.record()
            end_evt.synchronize()

            current_run_time = start_evt.elapsed_time(end_evt)
            comm_time += current_run_time
        comm_time = comm_time / nruns * 1e3
    if verbose:
        print(
            "[COMM Node]",
            getattr(comm_func_name, "python_kernel_name", ""),
            "time",
            comm_time,
        )
    if comm_cache is not None:
        comm_cache.add_comm_time(
            tensor_input.size(), tensor_output.size(), comm_func_name, comm_time
        )
    tensor_input.cpu()
    tensor_output.cpu()
    del tensor_input, tensor_output
    return comm_time


def estimate_comp_time(
    sched: "scheduler.Scheduler",
    snode: "scheduler.BaseSchedulerNode",
    verbose: bool = False,
    comp_cache: "CompPerfCache" = None,
) -> float:
    # Estimate the runtime of a compute node
    # FusedSchedulerNode & BaseSchedulerNode: get the generated triton code and use `do_bench` mode to obtain runtime
    # ExternKernelSchedulerNode: get python kernel and run the kernel to obtain runtime
    from ..scheduler import (
        BaseSchedulerNode,
        ExternKernelSchedulerNode,
        FusedSchedulerNode,
    )

    device = cast(torch.device, snode.get_device())

    if isinstance(snode, FusedSchedulerNode):
        node_list = snode.snodes
    elif isinstance(snode, ExternKernelSchedulerNode):
        time = benchmark_extern_node(snode.node, comp_cache)
        if verbose and time != 0:
            print("[COMP Node] EXTERN", "time", time)
        return time
    elif isinstance(snode, BaseSchedulerNode):
        node_list = [snode]
    else:
        raise ValueError(f"Unsupported snode type {type(snode)}")

    # this part code is from triton's bench code:
    # https://github.com/pytorch/pytorch/blob/85111cd165f108ffabb4a90083d59d7a867ebd9f/torch/_inductor/codegen/triton.py#L4234
    src_code = sched.generate_kernel_code_from_nodes(node_list, benchmark_kernel=True)
    if comp_cache is not None:
        time = comp_cache.get_runtime_by_triton(src_code)
        if time is not None:
            return time
    module = PyCodeCache.load(src_code)
    time, _ = sched.benchmark_codegened_module(module=module, device=device)
    time = time * 1e3

    if comp_cache is not None:
        comp_cache.add_triton_runtime(src_code, time)
    if verbose and time != 0:
        print("[COMP Node] BASE/FUSE", "time", time)
    return time


def benchmark_extern_node(
    node: ir._NodeOrNodes, comp_cache: "CompPerfCache" = None
) -> float:
    if isinstance(node, ir.MultiOutput):
        return 0

    python_kernel_name = getattr(node, "python_kernel_name", "")
    if python_kernel_name is None:
        return 0
    if python_kernel_name.startswith("extern_kernels"):
        func = kernel_name_to_comp_op.get(python_kernel_name, None)
    elif python_kernel_name.startswith("torch.ops.aten"):
        func = _convert_str_to_op(python_kernel_name)
    else:
        func = None

    if func is None:
        return 0
    else:
        if isinstance(node, ir.FallbackKernel):
            args = node.export_extern_kernel_node()
            ordered_kwargs_length = len(list(node.ordered_kwargs_for_cpp_kernel))
            if ordered_kwargs_length > 0:
                args, ordered_kwargs = (
                    args[: -1 * ordered_kwargs_length],
                    args[-1 * ordered_kwargs_length :],
                )
                ordered_kwargs = dict(
                    zip(node.ordered_kwargs_for_cpp_kernel, ordered_kwargs)
                )
                node.kwargs.update(ordered_kwargs)
        elif isinstance(node, ir.ExternKernel):
            args = node.inputs
            args = node.fill_non_provided_args(
                [*args, *node.constant_args], node.kwargs
            )
        else:
            raise ValueError(f"Unsupported node type {type(node)}")

        flat_args, args_property = pytree.tree_flatten((args, node.kwargs))

        if comp_cache is not None:
            flat_args_info = [
                input.get_size()
                if isinstance(input, ir.IRNode)
                and not isinstance(input, ir.GeneratorState)
                else input
                for input in flat_args
            ]
            flat_args_info = (
                tuple(a) if isinstance(a, list) else a for a in flat_args_info
            )
            kernel_flat_args = (python_kernel_name,) + tuple(flat_args_info)
            op_time = comp_cache.get_runtime_by_extern(kernel_flat_args)
            if op_time is not None:
                return op_time

        flat_args = [
            ir.ir_node_to_tensor(input, guard_shape=False)
            if isinstance(input, ir.IRNode) and not isinstance(input, ir.GeneratorState)
            else input
            for input in flat_args
        ]

        # this part code is from https://fburl.com/3xpyoq93
        with no_dispatch():

            def to_real_tensor(e: Any) -> Any:
                if not isinstance(e, torch.Tensor):
                    return e
                out = _create_real_tensor(e.size(), e.dtype, e.device)
                if e.is_sparse:
                    out._coalesced_(e.is_coalesced())
                return out

            def delete_tensor_in_list(l: list[Any]) -> None:
                for i in range(len(l)):
                    if isinstance(l[i], torch.Tensor):
                        l[i].cpu()
                        del l[i]
                        break

            flat_args = [to_real_tensor(a) for a in flat_args]
            args, kwargs = pytree.tree_unflatten(flat_args, args_property)
            func(*args, **kwargs)
            num_iters = 3
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            cpu_start = time.time()
            start_event.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                func(*args, **kwargs)
            end_event.record(torch.cuda.current_stream())
            cpu_end = time.time()
            torch.cuda.synchronize()
            cpu_time = cpu_end - cpu_start
            total_op_time = start_event.elapsed_time(end_event) - cpu_time
            mean_op_time = total_op_time / num_iters
            delete_tensor_in_list(flat_args)
            delete_tensor_in_list(args)
            del flat_args
            del args
            del kwargs

    mean_op_time = mean_op_time * 1e3
    if comp_cache is not None:
        comp_cache.add_extern_runtime(kernel_flat_args, mean_op_time)

    return mean_op_time
