import torch
from torch.autograd import Function
import torch.distributed as dist


def broadcast(tensor, src, group=dist.group.WORLD):
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


def gather(tensor, dst=0, group=dist.group.WORLD):
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


def scatter(tensors, src=0, group=dist.group.WORLD):
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


def reduce(tensor, dst, op=dist.ReduceOp.SUM, group=dist.group.WORLD):
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


def all_gather(tensor, group=dist.group.WORLD):
    """
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple[Tensor]): Output of the collective.

    """    
    return _AllGather.apply(group, tensor)


def all_to_all(tensors, group=dist.group.WORLD):
    """
    Each process scatters list of input tensors to all processes in a group and
    return gathered list of tensors in output list.

    Arguments:
        tensors (list[Tensor]): List of tensors to scatter one per rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple[Tensor]): Output of the collective.

    """
    return _AlltoAll.apply(group, *tensors)


def all_reduce(tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

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
        ctx.rank = dist.get_rank()
        # torch.distributed makes all the calls in place
        # we allocate new tensors to avoid this
        tensor = tensor.clone()
        dist.broadcast(tensor, src, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        gx = _Reduce.apply(ctx.src, dist.ReduceOp.SUM, ctx.group, grad_output)
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


class _AllGather(Function):
    @staticmethod
    def forward(ctx, group, tensor):
        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensor) for i in range(dist.get_world_size(group=group))
        ]
        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        gxs = _AlltoAll.apply(ctx.group, *grad_outputs)
        gx = torch.sum(torch.stack(gxs), dim=0)
        return (None, gx)


class _AlltoAll(Function):
    @staticmethod
    def forward(ctx, group, *tensors):
        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensors[i]) for i in range(dist.get_world_size(group=group))
        ]
        reqs = [None] * dist.get_world_size(group=group)
        my_rank = dist.get_rank(group=group)
        # Implement it on means of scatter/gather, send/recv async operations have issues
        if dist.get_backend(group=group) is dist.Backend.GLOO:
            for i in range(dist.get_world_size(group=group)):
                to_send = None
                if i == my_rank:
                    to_send = list(tensors)
                dist.scatter(out_tensor_list[i], to_send, i, group=group)
        else:
            dist.all_to_all(out_tensor_list, list(tensors), group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + _AlltoAll.apply(ctx.group, *grad_outputs)


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group = group
        ctx.op = op
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)
