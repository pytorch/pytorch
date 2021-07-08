import logging

import torch
import torch.distributed as dist

from . import default_hooks as default


class PostLocalSGDState(object):
    r"""
    Stores the state for all-reducing gradients globally using ``process_group`` until step ``start_localSGD_iter``,
    and all-reducing gradients locally using ``subgroup`` afterwards.
    """

    __slots__ = [
        "process_group",
        "subgroup",
        "start_localSGD_iter",
        "iter",
    ]

    def __init__(
        self,
        process_group,
        subgroup,
        start_localSGD_iter,
    ):
        logging.info(
            "Local SGD will be started after {} iterations".format(start_localSGD_iter)
        )

        # The group used for all-reducing gradients globally.
        self.process_group = process_group
        # The group used for all-reducing gradients locally.
        self.subgroup = subgroup
        self.start_localSGD_iter = start_localSGD_iter
        # Iteration/step in the training loop.
        self.iter = 0

    def maybe_increase_iter(self, bucket):
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_the_last_bucket_to_allreduce():
            self.iter += 1

        if self.iter == self.start_localSGD_iter:
            logging.info(
                "Start to apply local SGD after {} iterations.".format(self.iter)
            )


def post_localSGD_hook(
    state: PostLocalSGDState, bucket: dist.GradBucket
) -> torch.futures.Future:
    """
    This DDP communication hook is used for running post-localSGD algorithm,
    by combining with a model averaging component (e.g.,
    :class:`~torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager`)
    that runs after the optimizer step.

    Args:
        state (PostLocalSGDState): State information to run post-localSGD.
            Users mainly need to tune ``start_localSGD_iter`` to determine when to start local SGD.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode at this time,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> state = PostLocalSGDState(process_group=process_group, subgroup=subgroup,
                                  start_localSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, post_localSGD_hook)
        >>> # Also need to establish a model averaging module and run model averaging after ``optimizer.step()``.
        >>> # Please refer to the examples in ``torch.distributed.algorithms.model_averaging.averagers`` module.
    """
    global_group_to_use = (
        state.process_group if state.process_group is not None else dist.group.WORLD
    )
    world_size = global_group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.get_tensor()

    # Run allreduce using `global_group_to_use` in the first `start_localSGD_iter` iterations.
    if state.iter < state.start_localSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(global_group_to_use, input_tensor)

    # Run allreduce using `subgroup` after the first `start_localSGD_iter` iterations.
    # From this moment, model averaging should run after the optimizer step,
    # to globally allreduce all the parameters.
    return default._allreduce_fut(state.subgroup, input_tensor)
