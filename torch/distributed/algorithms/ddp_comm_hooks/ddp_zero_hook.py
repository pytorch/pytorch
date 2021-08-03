from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import _OverlapStatus
from torch.nn.parallel.distributed import DistributedDataParallel

# Functional optimizers require passing a list of gradients to their `step()`
# method, and ZeRO requires a functional optimizer to overlap with DDP
# Passing a `None` instead of an actual gradient indicates to the optimizer
# to not update the corresponding parameter
_NO_PARAM_UPDATE = None


def _perform_local_step(
    bucket: dist.GradBucket,
    zero: ZeroRedundancyOptimizer,
    rank: int,
):
    r"""
    Performs a local optimizer step using the gradients provided by ``bucket``.

    Arguments:
        bucket (dist.GradBucket): the bucket providing the gradients.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to perform the :meth:`_local_step`.
        rank (int): the calling process's rank.

    .. warning::
        This function assumes that appropriate synchronization has taken place
        so that the bucket's gradients can be used.
    """
    overlap_info = zero._overlap_info
    bucket_index = bucket.index()
    assert len(zero.optim.param_groups) == 1, \
        "Overlapping DDP with ZeRO only supports a single parameter group"

    # Construct the `gradients` input for the local optimizer step, which
    # expects `None` in a list position to indicate that the corresponding
    # parameter should not be updated
    num_local_optim_params = len(zero.optim.param_groups[0]["params"])
    gradients: List[Optional[torch.Tensor]] = \
        [_NO_PARAM_UPDATE for _ in range(num_local_optim_params)]
    assert bucket_index in overlap_info.offsets, \
        f"Bucket index {bucket_index} was not assigned to rank {rank}"
    offset = overlap_info.offsets[bucket_index]
    for i, grad in enumerate(bucket.gradients()):
        gradients[offset + i] = grad

    zero._local_step(gradients)


def _broadcast_bucket(
    bucket_index: int,
    zero: ZeroRedundancyOptimizer,
    assigned_rank: int,
):
    r"""
    Broadcasts a bucket's parameters.

    Arguments:
        bucket_index (int): the index of the bucket corresponding to the
            parameters to broadcast.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
        assigned_rank (int): the rank assigned to the bucket; it has the
            updated parameters and serves as the source for the broadcast.
    """
    overlap_info = zero._overlap_info
    device = overlap_info.params_per_bucket[bucket_index][0].device
    device_index = zero._device_to_device_index[device]
    assert bucket_index in zero._buckets[device_index][assigned_rank]
    overlap_info.broadcast_handles.append(
        dist.broadcast(
            zero._buckets[device_index][assigned_rank][bucket_index],
            src=assigned_rank,
            async_op=True
        )
    )

def _collect_ddp_bucket_info(
    bucket: dist.GradBucket,
    zero: ZeroRedundancyOptimizer,
    rank: int,
    assigned_rank: int,
):
    r"""
    Collects :class:`DistributedDataParallel` gradient bucket information for
    the :class:`ZeroRedundancyOptimizer` instance ``zero`` to use when
    overlapping.

    Arguments:
        bucket (dist.GradBucket): the current gradient bucket.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
        rank (int): the calling process's rank.
        assigned_rank (int): the rank assigned to update the parameters
            corresponding to ``bucket``.
    """
    overlap_info = zero._overlap_info
    bucket_index = bucket.index()
    bucket_params = bucket.parameters()
    assert len(bucket_params) > 0, "Bucket {bucket_index} is empty"
    params_per_rank = overlap_info.params_per_rank
    params_per_bucket = overlap_info.params_per_bucket

    # Collect relevant information
    if assigned_rank == rank:
        overlap_info.offsets[bucket_index] = len(params_per_rank[assigned_rank])
    params_per_rank[assigned_rank].extend(bucket_params)
    params_per_bucket.append(bucket_params)


def hook_with_zero_step(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future],
    ddp: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""
    Modifies the given ``hook`` to overlap the :class:`ZeroRedundancyOptimizer`
    optimizer step with the :class:`DistributedDataParallel` backward pass,
    where the optimizer step computation begins after the last gradient bucket
    computation has finished.

    This approach overlaps the optimizer computation and communication with the
    backward communication. In particular, the backward computation proceeds
    contiguously, and the optimizer computation follows, overlapping with
    outstanding backward communication (i.e. all-reduces) and possibly other
    optimizer communication (i.e. broadcasts).

    This approach may be preferred over :meth:`hook_with_zero_step_interleaved`
    if communication is relatively slow compared to computation.

    Arguments:
        hook (Callable[[Any, dist.GradBucket], torch.futures.Future]): the hook
            to modify.
        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
            instance to use.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to use.

    Returns:
        The modified hook.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
        RuntimeError: if using any backend other than NCCL since currently
            Gloo may hang.

    .. warning::
        Given the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two training iterations do not perform parameter updates in the
        optimizer step. This is because it needs information about the gradient
        bucketing strategy used by :class:`DistributedDataParallel`, which is
        not finalized until the second forward pass.
    """
    if not zero._overlap_with_ddp:
        raise ValueError(
            "ZeroRedundancyOptimizer must be constructed with "
            "`overlap_with_ddp=True` to use this hook properly"
        )

    # NOTE: Gloo may hang with this overlapping approach, so we require
    # NCCL backend for now; see https://github.com/pytorch/pytorch/issues/62300
    if dist.get_backend() != dist.Backend.NCCL:
        raise RuntimeError(
            "Overlapping DDP with ZeRO using this approach currently requires "
            "NCCL backend to avoid hangs"
        )

    def hook_with_zero_fn(
        state: Any,
        bucket: dist.GradBucket,
    ) -> torch.futures.Future[torch.Tensor]:
        r"""
        Returns a :class:`Future` that gives a gradient bucket tensor and
        performs the equivalent of a :class:`ZeroRedundancyOptimizer`
        :meth:`step` if ``bucket`` is the last gradient bucket.

        The function performs additional computation on the iteration that
        the :class:`DistributedDataParallel` buckets are rebuilt to collect
        information used to implement the modified hook.

        Arguments:
            state (Any): any state for the hook.
            bucket (dist.GradBucket): the :class:`DistributedDataParallel`
                gradient bucket.
        """
        fut = hook(state, bucket)
        overlap_info = zero._overlap_info
        bucket_index = bucket.index()

        # Proceed as normal until the DDP buckets have been rebuilt
        if not ddp._has_rebuilt_buckets:
            assert overlap_info.status == _OverlapStatus.UNINITIALIZED
            return fut

        if overlap_info.status == _OverlapStatus.UNINITIALIZED:
            overlap_info.status = _OverlapStatus.DDP_HAS_REBUILT_BUCKETS

        rank = zero.global_rank
        assigned_rank = zero._ddp_bucket_index_to_rank(bucket_index)

        # Once DDP buckets have been rebuilt but ZeRO has not been
        # properly initialized yet, collect the information needed
        if overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
            _collect_ddp_bucket_info(bucket, zero, rank, assigned_rank)
            return fut

        assert overlap_info.status == _OverlapStatus.INITIALIZED

        # Save the bucket reference and all-reduce future for the final bucket
        if assigned_rank == rank:
            overlap_info.bucket_index_to_bucket[bucket_index] = bucket
            overlap_info.bucket_index_to_future[bucket_index] = fut

        # Check that buckets are indexed incrementally starting from 0 in the
        # order of their autograd hooks firing
        if len(overlap_info.bucket_indices_seen) > 0:
            assert overlap_info.bucket_indices_seen[-1] == bucket_index - 1, \
                "Bucket indices are not in incremental order"
        else:
            assert bucket_index == 0, "Bucket indices do not start from 0"
        overlap_info.bucket_indices_seen.append(bucket_index)

        # Directly return the future without any optimizer computation if this
        # is not the last bucket
        num_buckets = len(overlap_info.params_per_bucket)
        is_last_bucket = bucket_index == num_buckets - 1
        if not is_last_bucket:
            return fut

        # Perform partial optimizer step on all buckets after the final
        # bucket has been computed
        # NOTE: This should not be chained as a callback to the last bucket's
        # all-reduce future since that would add synchronization that delays
        # all optimizer computation to wait for that last all-reduce
        for bucket_index in range(num_buckets):
            assigned_rank = zero._ddp_bucket_index_to_rank(bucket_index)
            if assigned_rank == rank:
                # Wait on the bucket's all-reduce future to ensure correct
                # gradients
                assert bucket_index in overlap_info.bucket_index_to_future, \
                    f"All-reduce future for bucket {bucket_index} not saved " \
                    f"on rank {rank}"
                allreduce_future = overlap_info.bucket_index_to_future[bucket_index]
                allreduce_future.wait()

                # Perform the partial optimizer step
                curr_bucket = overlap_info.bucket_index_to_bucket[bucket_index]
                _perform_local_step(curr_bucket, zero, rank)

            _broadcast_bucket(bucket_index, zero, assigned_rank)

        # Ensure that all parameter updates are finished before the
        # next forward pass
        overlap_info.wait_for_broadcasts(num_buckets, rank)
        overlap_info.clear_per_iter_info()

        return fut

    return hook_with_zero_fn


def hook_with_zero_step_interleaved(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future],
    ddp: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""
    Modifies the given ``hook`` to overlap the :class:`ZeroRedundancyOptimizer`
    optimizer step with the :class:`DistributedDataParallel` backward pass,
    where the optimizer step computation interleaves with the backward
    computation.

    This approach overlaps the optimizer computation and communication with the
    backward computation and communication. In particular, once a bucket's
    gradients have been computed, the optimizer computation using those
    gradients is launched (though the actual computation must wait for the
    bucket's all-reduce to complete). This yields an interleaving of all-
    reduces and broadcasts in the communication stream.

    This approach may be preferred over :meth:`hook_with_zero_step` if
    communication is relatively fast compared to computation.

    Arguments:
        hook (Any * dist.GradBucket -> torch.futures.Future): the hook to
            modify.
        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
            instance to use.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to use.

    Returns:
        The modified hook.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
        RuntimeError: if using any backend other than NCCL since currently
            Gloo may hang.

    .. warning::
        Given the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two training iterations do not perform parameter updates in the
        optimizer step. This is because it needs information about the gradient
        bucketing strategy used by :class:`DistributedDataParallel`, which is
        not finalized until the second forward pass.
    """
    if not zero._overlap_with_ddp:
        raise ValueError(
            "ZeroRedundancyOptimizer must be constructed with "
            "`overlap_with_ddp=True` to use this hook properly"
        )

    # NOTE: Gloo may hang with this overlapping approach, so we require
    # NCCL backend for now; see https://github.com/pytorch/pytorch/issues/62300
    if dist.get_backend() != dist.Backend.NCCL:
        raise RuntimeError(
            "Overlapping DDP with ZeRO using this approach currently requires "
            "NCCL backend to avoid hangs"
        )

    def hook_with_zero_interleaved_fn(
        state,
        bucket: dist.GradBucket,
    ) -> torch.futures.Future[torch.Tensor]:
        r"""
        Returns a :class:`Future` that gives a gradient bucket tensor and
        performs a partial :class:`ZeroRedundancyOptimizer` :meth:`step` using
        the gradients in that bucket.
        Arguments:
            state: any state for the hook.
            bucket (dist.GradBucket): the :class:`DistributedDataParallel`
                gradient bucket.
        """
        fut = hook(state, bucket)

        # Proceed as normal until the DDP buckets have been rebuilt
        if not ddp._has_rebuilt_buckets:
            assert zero._overlap_info.status == _OverlapStatus.UNINITIALIZED
            return fut

        def zero_step(fut: torch.futures.Future) -> torch.Tensor:
            r"""
            Performs a partial :class:`ZeroRedundancyOptimizer` :meth:`step`
            using the gradients in the given :class:`DistributedDataParallel`
            gradient bucket.

            Returns:
                A :class:`torch.Tensor` representing the contents of the
                gradient bucket.
            """
            assert ddp._has_rebuilt_buckets

            bucket_index = bucket.index()
            rank = zero.global_rank
            assigned_rank = zero._ddp_bucket_index_to_rank(bucket_index)
            overlap_info = zero._overlap_info
            if overlap_info.status == _OverlapStatus.UNINITIALIZED:
                overlap_info.status = _OverlapStatus.DDP_HAS_REBUILT_BUCKETS

            # Once DDP buckets have been rebuilt but ZeRO has not been
            # properly initialized yet, collect the information needed
            if overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
                _collect_ddp_bucket_info(bucket, zero, rank, assigned_rank)
                return bucket.get_tensor()

            overlap_info.bucket_indices_seen.append(bucket_index)
            if assigned_rank == rank:
                _perform_local_step(bucket, zero, rank)

            _broadcast_bucket(bucket_index, zero, assigned_rank)

            num_buckets = len(overlap_info.params_per_bucket)
            if len(overlap_info.bucket_indices_seen) == num_buckets:
                # Ensure that all parameter updates are finished before the
                # next forward pass
                overlap_info.wait_for_broadcasts(num_buckets, rank)
                overlap_info.clear_per_iter_info()

            return bucket.get_tensor()

        return fut.then(zero_step)

    return hook_with_zero_interleaved_fn
