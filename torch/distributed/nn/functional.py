# mypy: allow-untyped-defs
"""
.. warning::
    ``torch.distributed.nn.functional`` is deprecated. Use
    :mod:`torch.distributed.functional_collectives` instead.
"""

import warnings

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

# The two imports below are not always available depending on the
# USE_DISTRIBUTED compile flag. Make sure they raise import error
# if we're trying to use them.
from torch.distributed import group, ReduceOp
from torch.distributed.distributed_c10d import _rank_not_in_group


def _not_supported_under_compile(name, *, suggestion=None):
    msg = (
        f"torch.distributed.nn.functional.{name} is not supported under torch.compile."
    )
    if suggestion:
        msg += f" Use {suggestion} instead."
    raise RuntimeError(msg)


def _deprecated(name, suggestion):
    warnings.warn(
        f"torch.distributed.nn.functional.{name} is deprecated, "
        f"use {suggestion} instead.",
        category=FutureWarning,
        stacklevel=3,
    )


def _resolve_group(group):
    return dist.group.WORLD if group is None else group


def broadcast(tensor, src, group=group.WORLD):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Received tensor from the broadcast op.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "broadcast",
            suggestion="torch.distributed.functional_collectives.broadcast",
        )
    _deprecated("broadcast", "torch.distributed.functional_collectives.broadcast")
    group = _resolve_group(group)
    if _rank_not_in_group(group):
        return tensor.clone()
    return funcol.wait_tensor(
        funcol.broadcast(tensor, dist.get_group_rank(group, src), group)
    )


def gather(tensor, dst=0, group=group.WORLD):
    """
    Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        dst (int, optional): Destination rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple[Tensor]: List of appropriately-sized tensors with the gathered data.
    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "gather",
            suggestion="torch.distributed.functional_collectives.all_gather_single",
        )
    _deprecated("gather", "torch.distributed.functional_collectives.all_gather_single")
    group = _resolve_group(group)
    if _rank_not_in_group(group):
        return ()
    world_size = dist.get_world_size(group)
    group_dst = dist.get_group_rank(group, dst)
    is_dst = dist.get_rank(group) == group_dst
    out = funcol.wait_tensor(
        funcol.all_to_all_single(
            tensor.unsqueeze(0),
            [int(is_dst)] * world_size,
            [int(i == group_dst) for i in range(world_size)],
            group,
        )
    )
    if not is_dst:
        out = torch.cat([out, tensor.new_zeros(world_size, *tensor.shape)])
    return tuple(t.clone() for t in out.unbind(0))


def scatter(tensors, src=0, group=group.WORLD):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensors (list[Tensor]): List of tensors to scatter on the source rank.
            Receivers must pass ``None``.
        src (int, optional): Source rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output tensor from the scatter operation.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "scatter",
            suggestion="torch.distributed.functional_collectives.all_to_all_single",
        )
    _deprecated("scatter", "torch.distributed.functional_collectives.all_to_all_single")
    group = _resolve_group(group)
    if _rank_not_in_group(group):
        return torch.zeros_like(tensors[0])
    world_size = dist.get_world_size(group)
    group_src = dist.get_group_rank(group, src)
    is_src = dist.get_rank(group) == group_src
    input = torch.stack(tensors)
    if not is_src:
        # Send an empty slice that keeps the inputs in the graph.
        input = input[:0]
    out = funcol.all_to_all_single(
        input,
        [int(i == group_src) for i in range(world_size)],
        [int(is_src)] * world_size,
        group,
    )
    return funcol.wait_tensor(out)[0]


def reduce(tensor, dst, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input of the collective.
        dst (int): Destination rank.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "reduce", suggestion="torch.distributed.functional_collectives.all_reduce"
        )
    _deprecated("reduce", "torch.distributed.functional_collectives.all_reduce")
    group = _resolve_group(group)
    if _rank_not_in_group(group):
        return tensor.clone()
    out = funcol.wait_tensor(funcol.all_reduce(tensor, op, group))
    if dist.get_rank() == dst:
        return out
    # Non-dst ranks return their input. Keeping ``out`` in the graph with a zero
    # gradient makes the backward all_reduce run on every rank.
    false_mask = torch.zeros((), dtype=torch.bool, device=out.device)
    return torch.where(false_mask, out, tensor.detach())


def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Arguments:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "reduce_scatter",
            suggestion="torch.distributed.functional_collectives.reduce_scatter_single",
        )
    _deprecated(
        "reduce_scatter",
        "torch.distributed.functional_collectives.reduce_scatter_single",
    )
    group = _resolve_group(group)
    if _rank_not_in_group(group):
        return output
    return funcol.reduce_scatter_inplace(
        output, list(input_list), funcol.REDUCE_OP_TO_STR[op], group
    )


def all_gather(tensor, group=group.WORLD):
    """
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "all_gather",
            suggestion="torch.distributed.functional_collectives.all_gather_single",
        )
    _deprecated(
        "all_gather", "torch.distributed.functional_collectives.all_gather_single"
    )
    group = _resolve_group(group)
    if _rank_not_in_group(group):
        return ()
    out = funcol.wait_tensor(funcol.all_gather_single(tensor.unsqueeze(0), 0, group))
    return tuple(t.clone() for t in out.unbind(0))


def _all_gather_base(output_tensor, input_tensor, group=group.WORLD):
    """
    Single tensor all gather. Gathers a single tensor from all ranks, and puts them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Examples:
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 ranks.
        >>> # xdoctest: +SKIP("incorrect want text")
        >>> output_tensor = torch.zeros(2, dtype=torch.int64)
        >>> output_tensor
        [tensor([0, 0])] # Rank 0 and 1
        >>> tensor = torch.arange(1, dtype=torch.int64) + 1 + rank
        >>> tensor
        tensor([1]) # Rank 0
        tensor([2]) # Rank 1
        >>> dist.all_gather_base(output_tensor, tensor)
        >>> output_tensor
        tensor([1,2]) # Rank 0
        tensor([1,2]) # Rank 1

    .. warning::
        `_all_gather_base` is experimental and subject to change.
        It is the caller's responsibility to ensure the output_tensor
        is correctly sized.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "_all_gather_base",
            suggestion="torch.distributed.functional_collectives.all_gather_single",
        )
    _deprecated(
        "_all_gather_base", "torch.distributed.functional_collectives.all_gather_single"
    )
    group = _resolve_group(group)
    if _rank_not_in_group(group):
        return output_tensor
    return funcol.all_gather_tensor_inplace(output_tensor, input_tensor, group)


def all_to_all(output_tensor_list, input_tensor_list, group=group.WORLD):
    """
    Each process scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

    Arguments:
        output_tensor_list (list[Tensor]): list of tensors to gather one per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "all_to_all",
            suggestion="torch.distributed.functional_collectives.all_to_all_single",
        )
    _deprecated(
        "all_to_all", "torch.distributed.functional_collectives.all_to_all_single"
    )
    group = _resolve_group(group)
    if _rank_not_in_group(group):
        return tuple(output_tensor_list)
    output_split_sizes = [t.numel() for t in output_tensor_list]
    out = funcol.all_to_all_single(
        torch.cat([t.reshape(-1) for t in input_tensor_list]),
        output_split_sizes,
        [t.numel() for t in input_tensor_list],
        group,
    )
    outputs = funcol.wait_tensor(out).split(output_split_sizes)
    for o, t in zip(output_tensor_list, outputs):
        o.copy_(t.view_as(o))
    return tuple(output_tensor_list)


def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=group.WORLD,
):
    """
    Each process splits input tensor and then scatters the split list to all processes in a group.

    Then concatenate the received tensors from all the processes in the group and return single output tensor.

    Arguments:
        output (Tensor): Gathered concatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.

    Returns:
        Tensor: Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "all_to_all_single",
            suggestion="torch.distributed.functional_collectives.all_to_all_single",
        )
    _deprecated(
        "all_to_all_single",
        "torch.distributed.functional_collectives.all_to_all_single",
    )
    group = _resolve_group(group)
    if _rank_not_in_group(group):
        return output
    world_size = dist.get_world_size(group)
    output_split_sizes = (
        output_split_sizes or [output.size(0) // world_size] * world_size
    )
    input_split_sizes = input_split_sizes or [input.size(0) // world_size] * world_size
    return funcol.all_to_all_inplace(
        output, input, output_split_sizes, input_split_sizes, group
    )


def all_reduce(tensor, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces the tensor data across all machines in such a way that all get the final result.

    After the call the returned tensor is going to be bitwise
    identical in all processes.

    Arguments:
        tensor (Tensor): Input of the collective.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "all_reduce",
            suggestion="torch.distributed.functional_collectives.all_reduce",
        )
    _deprecated("all_reduce", "torch.distributed.functional_collectives.all_reduce")
    group = _resolve_group(group)
    if _rank_not_in_group(group):
        return tensor.clone()
    return funcol.wait_tensor(funcol.all_reduce(tensor, op, group))
