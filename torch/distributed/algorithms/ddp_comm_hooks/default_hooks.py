import torch
import torch.distributed as dist


def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    "Averages the input gradient tensor by allreduce and returns a future."
    fut = dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future()

    def div_by_group_size(fut):
        return [fut.value()[0].div_(group_to_use.size())]

    return fut.then(div_by_group_size)


def allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist._GradBucket
) -> torch.futures.Future:
    """
    This DDP communication hook just calls ``allreduce`` using ``GradBucket``
    tensors. Once gradient tensors are aggregated across all workers, its ``then``
    callback takes the mean and returns the result. If user registers this hook,
    DDP results is expected to be same as the case where no hook was registered.
    Hence, this won't change behavior of DDP and user can use this as a reference
    or modify this hook to log useful information or any other purposes while
    unaffecting DDP behavior.

    Example::
        >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
    """
    return _allreduce_fut(process_group, bucket.get_tensors()[0])


def fp16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist._GradBucket
) -> torch.futures.Future:
    """
    This DDP communication hook implements a simple gradient compression
    approach that converts ``GradBucket`` tensors whose type is assumed to be
    ``torch.float32`` to half-precision floating point format (``torch.float16``).
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, its then callback called ``decompress`` converts the
    aggregated result back to ``float32`` and takes the mean.

    Example::
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    compressed_tensor = bucket.get_tensors()[0].to(torch.float16)

    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.get_tensors()[0]
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        decompressed_tensor.copy_(fut.value()[0].div_(world_size))
        return [decompressed_tensor]

    return fut.then(decompress)


def _get_allgather_out_list(all_gather_in_list, world_size):
    out_list = [
        torch.zeros_like(
            all_gather_in_list,
            device=all_gather_in_list.device,
            dtype=all_gather_in_list.dtype,
        )
        for _ in range(world_size)
    ]
    return out_list


def _allgather_then_aggregate_hook(
    process_group: dist.ProcessGroup, bucket: dist._GradBucket
) -> torch.futures.Future:
    """
    Similar to ``allreduce_hook``, this hook first gathers ``GradBucket`` tensors
    and its ``then`` callback aggregates the gathered gradient tensors and takes
    mean. Instead of ``allreduce`` this hook uses ``allgather``. Note that with
    W workers, both the computation and communication time scale as O(W) for
    allgather compared to O(logW) for allreduce. Therefore, this hook is expected
    to be much slower than ``allreduce_hook`` although both essentially do the
    same thing with the gradients.

    .. warning ::
        This is for test and experiments. User is suggested to use a faster
        alternative called ``allreduce_hook``  that uses ``allreduce`` protocol
        instead of ``allgather`` protocol.

    Example::
        >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    world_size = group_to_use.size()

    tensor = bucket.get_tensors()[0]
    fut = dist.all_gather(
        _get_allgather_out_list(tensor, world_size),
        tensor,
        group=group_to_use,
        async_op=True,
    ).get_future()

    def aggregate(fut):
        all_ranks_tensor = fut.value()[0]
        tensor = bucket.get_tensors()[0]
        for r, gathered_tensor in enumerate(all_ranks_tensor):
            if r != rank:
                tensor += gathered_tensor

        return [tensor.div_(world_size)]

    return fut.then(aggregate)
