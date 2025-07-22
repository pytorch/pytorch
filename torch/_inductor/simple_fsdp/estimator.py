import statistics
import time
from collections.abc import Sequence
from typing import Any, Callable, cast, Union

from sympy import Expr

import torch
import torch.distributed as c10d
import torch.utils._pytree as pytree
from torch._inductor.codecache import PyCodeCache
from torch.utils._mode_utils import no_dispatch

from .. import ir, scheduler
from ..utils import contains_collective, contains_wait


kernel_name_to_comm_op: dict[str, Callable[..., Any]] = {
    "torch.ops._c10d_functional.all_gather_into_tensor.default": c10d.all_gather_into_tensor,
    "torch.ops._c10d_functional.reduce_scatter_tensor.default": c10d.reduce_scatter_tensor,
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
    snode: "scheduler.BaseSchedulerNode",
    estimate: bool = False,
    verbose: bool = False,
) -> float:
    # TODO (ruisizhang123): add more types of collective communication.
    # Currently, it only supports all_gather and reduce_scatter
    # estimate set to True: return NCCL's estimated comm time (https://github.com/pytorch/pytorch/pull/149343)
    # estimate set to False: run the collective communication and return the actual comm time

    kernel = snode.node
    assert isinstance(kernel, ir._CollectiveKernel)
    assert isinstance(kernel.layout, ir.Layout)
    assert hasattr(kernel.inputs[0], "data")
    inputs = kernel.inputs[0]

    comm_func = kernel_name_to_comm_op.get(getattr(kernel, "python_kernel_name", ""))
    assert comm_func is not None, (
        f"Unsupported comm op {getattr(kernel, 'python_kernel_name', '')}"
    )

    rank = c10d.distributed_c10d.get_rank()
    device = torch.device(f"cuda:{rank:d}")
    process_group = c10d.distributed_c10d._get_default_group()

    tensor_input = _create_real_tensor(
        inputs.data.get_size(), inputs.data.get_dtype(), inputs.data.get_device()
    )
    tensor_output = _create_real_tensor(
        kernel.layout.size, kernel.layout.dtype, kernel.layout.device
    )

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

        for _ in range(nruns):
            c10d.barrier()
            torch.cuda.synchronize()

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            comm_func(tensor_output, tensor_input)
            end_evt.record()
            end_evt.synchronize()

            current_run_time = start_evt.elapsed_time(end_evt)
            comm_time += current_run_time
        comm_time = comm_time / nruns * 1e3
    if verbose:
        print(
            "[COMM Node]", getattr(kernel, "python_kernel_name", ""), "time", comm_time
        )
    del tensor_input, tensor_output
    return comm_time


def estimate_comp_time(
    sched: "scheduler.Scheduler",
    snode: "scheduler.BaseSchedulerNode",
    verbose: bool = False,
) -> float:
    # Estimate the runtime of a compute node
    # FusedSchedulerNode & BaseSchedulerNode: get the generated triton code and use `do_bench` mode to obtain runtime
    # ExternKernelSchedulerNode: get python kernel and run the kernel to obtain runtime
    device = cast(torch.device, snode.get_device())

    if isinstance(snode, scheduler.FusedSchedulerNode):
        node_list = snode.snodes
    elif isinstance(snode, scheduler.ExternKernelSchedulerNode):
        time = benchmark_extern_node(snode.node)
        if verbose and time != 0:
            print("[COMP Node] EXTERN", "time", time)
        return time
    elif isinstance(snode, scheduler.BaseSchedulerNode):
        node_list = [snode]
    else:
        raise ValueError(f"Unsupported snode type {type(snode)}")

    # this part code is from triton's bench code:
    # https://github.com/pytorch/pytorch/blob/85111cd165f108ffabb4a90083d59d7a867ebd9f/torch/_inductor/codegen/triton.py#L4234
    src_code = sched.generate_kernel_code_from_nodes(node_list, benchmark_kernel=True)
    module = PyCodeCache.load(src_code)

    time, _ = sched.benchmark_codegened_module(module=module, device=device)
    time = time * 1e3
    if verbose and time != 0:
        print("[COMP Node] BASE/FUSE", "time", time)
    return time


def benchmark_extern_node(node: ir._NodeOrNodes) -> float:
    if isinstance(node, ir.MultiOutput):
        return 0

    python_kernel_name = getattr(node, "python_kernel_name", "")

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
            del flat_args

    mean_op_time = mean_op_time * 1e3
    return mean_op_time
