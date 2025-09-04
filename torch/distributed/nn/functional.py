# mypy: allow-untyped-defs
import torch
import torch.distributed as dist
from torch.autograd import Function

# The two imports below are not always available depending on the
# USE_DISTRIBUTED compile flag. Make sure they raise import error
# if we're trying to use them.
from torch.distributed import group, ReduceOp


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
    return _Broadcast.apply(src, group, tensor)


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
    return _Gather.apply(dst, group, tensor)


def scatter(tensors, src=0, group=group.WORLD):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensors (list[Tensor]): List of tensors to scatter on the source rank.
            Receivers must pass ``None`.
        src (int, optional): Source rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output tensor from the scatter operation.

    """
    return _Scatter.apply(src, group, *tensors)


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
    return _Reduce.apply(dst, op, group, tensor)


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
    return _Reduce_Scatter.apply(op, group, output, *input_list)


def all_gather(tensor, group=group.WORLD):
    """
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
    return _AllGather.apply(group, tensor)


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
        >>> # We have 2 process groups, 2 ranks.
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
    return _AllGatherBase.apply(output_tensor, input_tensor, group)


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
    return _AlltoAll.apply(group, output_tensor_list, *input_tensor_list)


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
    return _AlltoAllSingle.apply(
        group, output, output_split_sizes, input_split_sizes, input
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
    return _AllReduce.apply(op, group, tensor)


class _Broadcast(Function):
    @staticmethod
    def forward(ctx, src, group, tensor):
        ctx.src = src
        ctx.group = group
        ctx.rank = dist.get_rank(group=group)
        # torch.distributed makes all the calls in place
        # we allocate new tensors to avoid this
        tensor = tensor.clone()
        dist.broadcast(tensor, src, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        gx = _Reduce.apply(ctx.src, ReduceOp.SUM, ctx.group, grad_output)
        if ctx.src != ctx.rank:
            gx.zero_()
        return (None, None, gx)


class _Gather(Function):
    @staticmethod
    def forward(ctx, dst, group, tensor):
        ctx.dst = dst
        ctx.group = group
        # Need to create a list of tensors here to do the
        # aggregation, get it from the group size
        # tensor should be correctly sized for the method
        # gathering
        tensor_list = [
            torch.zeros_like(tensor) for i in range(dist.get_world_size(group=group))
        ]

        tensor = tensor.contiguous()
        if dist.get_rank(group=group) == dst:
            dist.gather(tensor, tensor_list, dst, group=group)
        else:
            dist.gather(tensor, None, dst, group=group)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None) + (_Scatter.apply(ctx.dst, ctx.group, *grad_outputs),)


class _Scatter(Function):
    @staticmethod
    def forward(ctx, src, group, *tensors):
        ctx.src = src
        ctx.group = group
        assert all(t.size() == tensors[0].size() for t in tensors)
        output = torch.zeros_like(tensors[0])
        if dist.get_rank(group=group) == src:
            dist.scatter(output, list(tensors), src, group=group)
        else:
            dist.scatter(output, None, src, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + _Gather.apply(ctx.src, ctx.group, grad_output)


class _Reduce(Function):
    @staticmethod
    def forward(ctx, src, op, group, tensor):
        ctx.src = src
        ctx.group = group
        tensor = tensor.clone()
        dist.reduce(tensor, src, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (_Broadcast.apply(ctx.src, ctx.group, grad_output),)


class _Reduce_Scatter(Function):
    @staticmethod
    def forward(ctx, op, group, tensor, *input_tensor_list):
        ctx.group = group
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()
        input_tensor_list = tuple(t.contiguous() for t in input_tensor_list)
        dist.reduce_scatter(tensor, list(input_tensor_list), op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + _AllGather.apply(ctx.group, grad_output)


class _AllGather(Function):
    @staticmethod
    def forward(ctx, group, tensor):
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()

        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]

        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if dist.get_backend(group=ctx.group) is dist.Backend.NCCL:
            rank = dist.get_rank(group=ctx.group)
            gx = torch.empty_like(grad_outputs[rank])
            gx = _Reduce_Scatter.apply(ReduceOp.SUM, ctx.group, gx, *grad_outputs)
        else:
            # As many backends doesn't support ReduceScatter, we use AlltoAll with .sum()
            # to emulate the ReduceScatter behavior
            tensor_list = [torch.empty_like(tensor) for tensor in grad_outputs]
            gxs = _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)
            gx = torch.sum(torch.stack(gxs), dim=0)
        return (None, gx)


class _AllGatherBase(Function):
    @staticmethod
    def forward(ctx, output_tensor, input_tensor, group):
        ctx.group = group
        dist._all_gather_base(output_tensor, input_tensor.contiguous(), group=group)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        if dist.get_backend(group=ctx.group) is dist.Backend.NCCL:
            world_size = dist.get_world_size(group=ctx.group)
            out_size = list(grad_output.size())
            if out_size[0] % world_size != 0:
                raise RuntimeError(
                    f"Tensor with dimensions: {out_size} does "
                    f"not have first dimension divisible by world_size: {world_size}"
                )
            out_size[0] = out_size[0] // dist.get_world_size(group=ctx.group)
            gx = torch.empty(
                out_size, device=grad_output.device, dtype=grad_output.dtype
            )
            dist._reduce_scatter_base(gx, grad_output, ReduceOp.SUM, ctx.group)
        else:
            raise RuntimeError("Backend not supported!")
        return (None, gx, None)


class _AlltoAll(Function):
    @staticmethod
    def forward(ctx, group, out_tensor_list, *tensors):
        ctx.group = group
        ctx.input_tensor_size_list = [
            tensors[i].size() for i in range(dist.get_world_size(group=group))
        ]
        my_rank = dist.get_rank(group=group)
        tensors = tuple(t.contiguous() for t in tensors)
        # Implement it on means of scatter/gather, send/recv async operations have issues
        if dist.get_backend(group=group) is dist.Backend.GLOO:
            for i in range(dist.get_world_size(group=group)):
                to_send = None
                if i == my_rank:
                    to_send = list(tensors)
                dist.scatter(out_tensor_list[i], to_send, i, group=group)
        else:
            dist.all_to_all(
                out_tensor_list,
                list(tensors),
                group=group,
            )
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        tensor_list = [
            torch.empty(
                size, device=grad_outputs[0].device, dtype=grad_outputs[0].dtype
            )
            for size in ctx.input_tensor_size_list
        ]
        return (None, None) + _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)


class _AlltoAllSingle(Function):
    @staticmethod
    def forward(ctx, group, output, output_split_sizes, input_split_sizes, input):
        ctx.group = group
        ctx.input_size = input.size()
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tensor = torch.empty(
            ctx.input_size, device=grad_output.device, dtype=grad_output.dtype
        )
        return (None, None, None, None) + (
            _AlltoAllSingle.apply(
                ctx.group,
                tensor,
                ctx.output_split_sizes,
                ctx.input_split_sizes,
                grad_output.contiguous(),
            ),
        )


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group = group
        ctx.op = op
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)
