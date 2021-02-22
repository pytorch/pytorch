import logging

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


class AllReduceFirstKStepsState(object):
    """
    State information required by allreduce_first_k_steps_hook to run gradient averaging for the first K steps.
    Need to specify ``k``.
    """

    __slots__ = [
        "process_group",
        "k",
        "step",
    ]

    def __init__(
        self,
        process_group,
        k,
    ):
        self.process_group = process_group
        self.k = k
        self.step = 0

        logging.info(
            "Allreduce-first-k-steps hook runs allreduce for the first {} steps.".format(
                self.k,
            )
        )

    def maybe_increase_step(self, bucket):
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `step` when bucket 0 is processed.
        if bucket.get_index() == 0:
            self.step += 1

        if self.step == self.k:
            logging.info("No more gradient averaging after {} steps.".format(self.k))


def allreduce_first_k_steps_hook(
    state: AllReduceFirstKStepsState, bucket: dist._GradBucket
) -> torch.futures.Future:
    """
    This DDP communication hook runs allreduce to average gradients for the first K steps,
    and then stops any communication afterwards.
    This hook is designed to support post-local SGD, by running gradient averaging before local SGD stage.
    It is user's responsibility to perform model parameter averaging at local SGD stage after K steps.

    Args:
        state (AllReduceFirstKStepsState): State information to run allreduce for the first K steps.
        bucket (dist._GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode at this time,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        state = AllReduceFirstKStepsState(process_group=process_group, k=1000)
        >>> ddp_model.register_comm_hook(state, allreduce_first_k_steps_hook)
    """
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # The input tensors in the bucket only has a single tensor.
    input_tensors = bucket.get_tensors()

    # Run allreduce in the first K steps.
    if state.step < state.k:
        state.maybe_increase_step(bucket)
        return _allreduce_fut(group_to_use, input_tensors[0])
    else:
        # Afterwards K steps, directly return the input tensors as the output future.
        ret_fut = torch.futures.Future()
        ret_fut.set_result(input_tensors)
        return ret_fut
