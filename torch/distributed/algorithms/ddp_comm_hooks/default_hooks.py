from typing import Any, Callable

import torch
import torch.distributed as dist

__all__ = ["allreduce_hook", "fp16_compress_hook", "bf16_compress_hook", "fp16_compress_wrapper", "bf16_compress_wrapper"]

def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


def allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook just calls ``allreduce`` using ``GradBucket``
    tensors. Once gradient tensors are aggregated across all workers, its ``then``
    callback takes the mean and returns the result. If user registers this hook,
    DDP results is expected to be same as the case where no hook was registered.
    Hence, this won't change behavior of DDP and user can use this as a reference
    or modify this hook to log useful information or any other purposes while
    unaffecting DDP behavior.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
    """
    return _allreduce_fut(process_group, bucket.buffer())


def fp16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    compressed_tensor = bucket.buffer().to(torch.float16).div_(world_size)

    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.buffer()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        decompressed_tensor.copy_(fut.value()[0])
        return decompressed_tensor

    return fut.then(decompress)

# TODO: create an internal helper function and extract the duplicate code in FP16_compress and BF16_compress.
def bf16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_ (``torch.bfloat16``)
    and then divides it by the process group size.
    It allreduces those ``bfloat16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, bf16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    compressed_tensor = bucket.buffer().to(torch.bfloat16).div_(world_size)

    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.buffer()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        decompressed_tensor.copy_(fut.value()[0])
        return decompressed_tensor

    return fut.then(decompress)


def fp16_compress_wrapper(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    floating point format (``torch.float16``), and casts the resulting tensor of the given hook back to
    the input data type, such as ``float32``.

    Therefore, ``fp16_compress_hook`` is equivalent to ``fp16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, fp16_compress_wrapper(powerSGD_hook))
    """

    def fp16_compress_wrapper_hook(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # Cast bucket tensor to FP16.
        bucket.set_buffer(bucket.buffer().to(torch.float16))

        fut = hook(hook_state, bucket)

        def decompress(fut):
            decompressed_tensor = bucket.buffer()
            # Decompress in place to reduce the peak memory.
            # See: https://github.com/pytorch/pytorch/issues/45968
            decompressed_tensor.copy_(fut.value())
            return decompressed_tensor

        # Decompress after hook has run.
        return fut.then(decompress)

    return fp16_compress_wrapper_hook

def bf16_compress_wrapper(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format> `_  (``torch.bfloat16``),
    and casts the resulting tensor of the given hook back to the input data type, such as ``float32``.

    Therefore, ``bf16_compress_hook`` is equivalent to ``bf16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, bf16_compress_wrapper(powerSGD_hook))
    """

    def bf16_compress_wrapper_hook(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # Cast bucket tensor to BF16.
        bucket.set_buffer(bucket.buffer().to(torch.bfloat16))

        fut = hook(hook_state, bucket)

        def decompress(fut):
            decompressed_tensor = bucket.buffer()
            # Decompress in place to reduce the peak memory.
            # See: https://github.com/pytorch/pytorch/issues/45968
            decompressed_tensor.copy_(fut.value())
            return decompressed_tensor

        # Decompress after hook has run.
        return fut.then(decompress)

    return bf16_compress_wrapper_hook
