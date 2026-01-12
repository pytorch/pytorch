import torch
from torch._inductor.utils import get_device_tflops, get_gpu_dram_gbps
from torch.fx.experimental.symbolic_shapes import (
    has_hint,
    size_hint,
    statically_known_true,
)
from torch.utils._ordered_set import OrderedSet
from .flop_counter import flop_registry


aten = torch.ops.aten

_FLOAT_TYPES = OrderedSet(
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ]
)

# No fall-back kernel needed/exists for view ops
_VIEW_OPS = OrderedSet(
    [
        aten.lift_fresh,
        aten.t,
        aten.transpose,
        aten.view,
        aten.detach,
        aten._unsafe_view,
        aten.split,
        aten.adjoint,
        aten.as_strided,
        aten.diagonal,
        aten.expand,
        aten.expand_as,
        aten.movedim,
        aten.permute,
        aten.select,
        aten.squeeze,
        aten.mT,
        aten.mH,
        aten.real,
        aten.imag,
        aten.view_as,
        aten.unflatten,
        aten.unfold,
        aten.unbind,
        aten.unsqueeze,
        aten.vsplit,
        aten.hsplit,
        aten.split_with_sizes,
        aten.swapaxes,
        aten.swapdims,
        aten.chunk,
    ]
)
# We can ignore benchmarking tensor create ops
_CREATE_OPS = OrderedSet(
    [
        aten.randint,
        aten.randn,
        aten.rand,
        aten.randn_like,
        aten.rand_like,
        aten.randint_like,
        aten.arange,
        aten.ones_like,
        aten.zeros_like,
    ]
)

_IGNORE_OPS = _VIEW_OPS | _CREATE_OPS


def get_compute_time(func_packet, args, kwargs, out, out_dtypes) -> float:  # type: ignore[no-untyped-def]
    """
    Estimates the compute time of an aten operator.

    Args:
        func_packet: The operator overload packet.
        args: The arguments to the operator.
        kwargs: The keyword arguments to the operator.
        out: The output of the operator.
        out_dtypes: The output data types.

    Returns:
        float: The estimated compute time in nanoseconds.
    """
    if func_packet in flop_registry:
        if len(out_dtypes) != 1:
            raise AssertionError(
                f"Only support single out dtype got {out_dtypes} for {func_packet}"
            )
        dtype = out_dtypes.pop()
        # This actually gives peta-FLOPs/s hence multiply by 1e15 to get the FLOPs/s
        peak_gpu_flops = get_device_tflops(dtype) * 1e15
        # We can expect to achieve 75% of theoretical peak flops
        factor = 0.75
        peak_empirical_flops = factor * peak_gpu_flops
        flop_count_func = flop_registry[func_packet]
        # We divide by a factor of 2 to get the MACs (multiply and accumulate)
        flop_count = flop_count_func(*args, **kwargs, out_val=out) / 2
        # We multiply by 1e9 to get the time in nano seconds
        compute_time = (flop_count / peak_empirical_flops) * 1e9
        return compute_time
    return 0.0


def get_num_bytes(t: torch.Tensor) -> int:
    """
    Calculates the memory consumption of a tensor.

    Args:
        t (torch.Tensor): The input tensor.

    Returns:
        int: The memory consumption of the tensor in bytes.
    """
    real_numel = 1
    for size, stride in zip(t.shape, t.stride()):
        if not has_hint(size) or not has_hint(stride):
            return 0

        # For dims with stride=0 (expanded/broadcast), only 1 element accessed
        if not statically_known_true(stride == 0):
            real_numel *= size_hint(size)

    return real_numel * t.element_size()


def get_transfer_time(flat_args_kwargs, flat_outs) -> float:  # type: ignore[no-untyped-def]
    """
    Estimates the memory transfer time of input and output tensors.

    Args:
        flat_args_kwargs (List[torch.Tensor]): The flat list of arguments and keyword arguments.
        flat_outs (List[torch.Tensor]): The flat list of outputs.

    Returns:
        float: The estimated memory transfer time in nanoseconds.
    """
    gpu_memory_bandwidth = get_gpu_dram_gbps()
    read_bytes = sum(
        get_num_bytes(t) for t in flat_args_kwargs if isinstance(t, torch.Tensor)
    )
    write_bytes = sum(
        get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor)
    )
    counted_bytes = read_bytes + write_bytes
    # The GPU memory bandwidth is in GB/s so the transfer time is in nanoseconds
    transfer_time = counted_bytes / gpu_memory_bandwidth
    return transfer_time
