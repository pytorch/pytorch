# mypy: allow-untyped-defs
import weakref
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import _OverlapStatus
from torch.nn.parallel.distributed import DistributedDataParallel


__all__ = ["hook_with_zero_step", "hook_with_zero_step_interleaved"]

# Functional optimizers require passing a list of gradients to their `step()`
# method, and ZeRO requires a functional optimizer to overlap with DDP
# Passing a `None` instead of an actual gradient indicates to the optimizer
# to not update the corresponding parameter
_NO_PARAM_UPDATE: None = None


def _perform_local_step(
    bucket: dist.GradBucket,
    zero: ZeroRedundancyOptimizer,
    rank: int,
):
    r"""
    Perform a local optimizer step using the gradients provided by ``bucket``.

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
    assert (
        len(zero.optim.param_groups) == 1
    ), "Overlapping DDP with ZeRO only supports a single parameter group"

    # Construct the `gradients` input for the local optimizer step, which
    # expects `None` in a list position to indicate that the corresponding
    # parameter should not be updated
    num_local_optim_params = len(zero.optim.param_groups[0]["params"])
    gradients: List[Optional[torch.Tensor]] = [
        _NO_PARAM_UPDATE for _ in range(num_local_optim_params)
    ]
    assert (
        bucket_index in overlap_info.offsets
    ), f"Bucket index {bucket_index} was not assigned to rank {rank}"
    gradients_offset = overlap_info.offsets[bucket_index]
    bucket_assignment = zero._bucket_assignments_per_rank[rank][bucket_index]
    bucket_offset = bucket_assignment.offset
    length = len(bucket_assignment.parameters)
    bucket_gradients = bucket.gradients()[bucket_offset : bucket_offset + length]
    for i, grad in enumerate(bucket_gradients):
        gradients[gradients_offset + i] = grad

    zero._local_step(gradients)


def _broadcast_bucket(
    bucket_index: int,
    zero: ZeroRedundancyOptimizer,
):
    r"""
    Broadcasts a bucket's parameters.

    Arguments:
        bucket_index (int): the index of the bucket corresponding to the
            parameters to broadcast.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
    """
    overlap_info = zero._overlap_info
    assert (
        len(overlap_info.assigned_ranks_per_bucket) > bucket_index
    ), "`assigned_ranks_per_bucket` is not fully constructed"
    # Sort to ensure the same ordering across ranks
    assigned_ranks = sorted(overlap_info.assigned_ranks_per_bucket[bucket_index])
    assert len(assigned_ranks) > 0, (
        f"Bucket {bucket_index} should be " "assigned to at least one rank"
    )
    for assigned_rank in assigned_ranks:
        bucket_assignments = zero._bucket_assignments_per_rank[assigned_rank]
        if bucket_index in bucket_assignments:
            send_tensor = bucket_assignments[bucket_index].tensor
            assert send_tensor is not None
            overlap_info.broadcast_handles.append(
                dist.broadcast(
                    send_tensor,
                    src=dist.get_global_rank(zero.process_group, assigned_rank),
                    group=zero.process_group,
                    async_op=True,
                )
            )


def _save_ddp_bucket_info(
    bucket: dist.GradBucket,
    zero: ZeroRedundancyOptimizer,
):
    r"""
    Save :class:`DistributedDataParallel` gradient bucket information for :class:`ZeroRedundancyOptimizer` instance ``zero``.

    In particular, this function is meant to be called upon seeing each
    gradient bucket to use when overlapping, meaning it does not save or compute any global
    information.

    Arguments:
        bucket (dist.GradBucket): the current gradient bucket.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
    """
    overlap_info = zero._overlap_info
    bucket_params = bucket.parameters()
    assert len(bucket_params) > 0, "Empty bucket"

    # Save the parameters in the bucket
    overlap_info.params_per_bucket.append(bucket_params)
    if overlap_info.shard_buckets:
        # Additionally save the bucket size for the assignment heuristic to use
        bucket_size = 0
        for param in bucket_params:
            bucket_size += param.numel()
        assert overlap_info.total_size is not None
        overlap_info.total_size += bucket_size


def _hook_with_zero_step_setup(
    ddp_ref: weakref.ReferenceType,
    zero: ZeroRedundancyOptimizer,
    bucket: dist.GradBucket,
):
    r"""
    Encapsulate the setup logic for :func:`hook_with_zero_step` and :func:`hook_with_zero_step_interleaved`.

    This means the logic to run in the
    hook before the backward pass and optimizer step can actually be
    overlapped. This is factored out since it is common to both
    :func:`hook_with_zero_step` and :func:`hook_with_zero_step_interleaved`.

    Arguments:
        ddp_ref (weakref.ReferenceType): weak reference to the process's
            :class:`DistributedDataParallel` instance.
        zero (ZeroRedundancyOptimizer): the calling process's
            :class:`ZeroRedundancyOptimizer` instance.
        bucket (dist.GradBucket): the current gradient bucket.
    """
    # Proceed as normal until the DDP buckets have been rebuilt
    if not ddp_ref()._has_rebuilt_buckets:  # type: ignore[union-attr]
        assert zero._overlap_info.status == _OverlapStatus.UNINITIALIZED
        return

    bucket_index = bucket.index()
    overlap_info = zero._overlap_info
    if overlap_info.status == _OverlapStatus.UNINITIALIZED:
        overlap_info.status = _OverlapStatus.DDP_HAS_REBUILT_BUCKETS

    if overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
        if bucket_index == 0 and len(overlap_info.params_per_bucket) > 0:
            # This corresponds to the first bucket of the backward pass
            # immediately after all information has been saved, so we
            # can perform the delayed ZeRO initialization
            zero._init_zero_for_overlap()
        else:
            # Once DDP buckets have been rebuilt but ZeRO has not been
            # properly initialized yet, save the information needed
            _save_ddp_bucket_info(bucket, zero)


def hook_with_zero_step(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future],
    ddp: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
    shard_buckets: bool = False,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""
    Modify ``hook`` to overlap :class:`ZeroRedundancyOptimizer` optimizer step with :class:`DistributedDataParallel` backward pass.

    This approach overlaps the optimizer computation and communication with the
    backward communication. In particular, the backward computation proceeds
    contiguously, and the optimizer computation follows, overlapping with
    outstanding backward communication (i.e. all-reduces) and possibly other
    optimizer communication (i.e. broadcasts).
    The optimizer step computation begins after the last gradient bucket computation has finished.

    This approach may be preferred over :meth:`hook_with_zero_step_interleaved`
    if communication is relatively slow compared to computation.

    Arguments:
        hook (Callable[[Any, dist.GradBucket], torch.futures.Future]): the hook
            to modify.
        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
            instance to use.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to use.
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity; if
            ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).

    Returns:
        The modified hook.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
        RuntimeError: if using any backend other than NCCL/HCCL since currently
            Gloo may hang.

    .. warning::
        Given the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``.
    """
    if not zero._overlap_with_ddp:
        raise ValueError(
            "ZeroRedundancyOptimizer must be constructed with "
            "`overlap_with_ddp=True` to use this hook properly"
        )
    ddp_ref = weakref.ref(ddp)

    # NOTE: Gloo may hang with this overlapping approach, so we require
    # NCCL/HCCL backend for now; see https://github.com/pytorch/pytorch/issues/62300
    pg = dist.get_backend(ddp_ref().process_group)  # type: ignore[union-attr]
    if (pg != dist.Backend.NCCL) and (pg != "hccl"):
        raise RuntimeError(
            "Overlapping DDP with ZeRO using this approach currently requires "
            "NCCL/HCCL backend to avoid hangs"
        )

    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    def hook_with_zero_fn(
        state: Any,
        bucket: dist.GradBucket,
    ) -> torch.futures.Future[torch.Tensor]:
        r"""
        Return :class:`Future` that runs the optimizer step if this corresponds to the last gradient bucket.

        Perform equivalent of :class:`ZeroRedundancyOptimizer` :meth:`step` if ``bucket`` is last gradient bucket.
        The function gives a gradient bucket tensor and
        performs additional computation on the iteration that
        the :class:`DistributedDataParallel` buckets are rebuilt to collect
        information used to implement the modified hook.

        Arguments:
            state (Any): any state for the hook.
            bucket (dist.GradBucket): the :class:`DistributedDataParallel`
                gradient bucket.
        """
        fut = hook(state, bucket)
        _hook_with_zero_step_setup(ddp_ref, zero, bucket)
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut

        overlap_info = zero._overlap_info
        bucket_index = bucket.index()
        rank = zero.global_rank

        assert overlap_info.status == _OverlapStatus.INITIALIZED
        assert (
            len(overlap_info.assigned_ranks_per_bucket) > bucket_index
        ), "`assigned_ranks_per_bucket` is not fully constructed"
        assigned_to_bucket = (
            rank in overlap_info.assigned_ranks_per_bucket[bucket_index]
        )

        # Save the bucket reference and all-reduce future for the final bucket
        if assigned_to_bucket:
            overlap_info.bucket_index_to_bucket[bucket_index] = bucket
            overlap_info.bucket_index_to_future[bucket_index] = fut

        # Check that buckets are indexed incrementally starting from 0 in the
        # order of their autograd hooks firing
        if len(overlap_info.bucket_indices_seen) > 0:
            assert (
                overlap_info.bucket_indices_seen[-1] == bucket_index - 1
            ), "Bucket indices are not in incremental order"
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
            assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
            if rank in assigned_ranks:
                # Wait on the bucket's all-reduce future to ensure correct
                # gradients
                assert bucket_index in overlap_info.bucket_index_to_future, (
                    f"All-reduce future for bucket {bucket_index} not saved "
                    f"on rank {rank}"
                )
                allreduce_future = overlap_info.bucket_index_to_future[bucket_index]
                allreduce_future.wait()

                # Perform the partial optimizer step
                curr_bucket = overlap_info.bucket_index_to_bucket[bucket_index]
                _perform_local_step(curr_bucket, zero, rank)

            _broadcast_bucket(bucket_index, zero)

        # Ensure that all parameter updates are finished before the
        # next forward pass
        overlap_info.wait_for_broadcasts()
        overlap_info.clear_per_iter_info()

        return fut

    return hook_with_zero_fn


def hook_with_zero_step_interleaved(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future],
    ddp: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
    shard_buckets: bool = False,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""
    Modify ``hook`` to overlap :class:`ZeroRedundancyOptimizer` optimizer step with :class:`DistributedDataParallel` backward pass

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
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity; if
            ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).

    Returns:
        The modified hook.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
        RuntimeError: if using any backend other than NCCL since currently
            Gloo may hang.

    .. warning::
        Given the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``.
    """
    if not zero._overlap_with_ddp:
        raise ValueError(
            "ZeroRedundancyOptimizer must be constructed with "
            "`overlap_with_ddp=True` to use this hook properly"
        )
    ddp_ref = weakref.ref(ddp)

    # NOTE: Gloo may hang with this overlapping approach, so we require
    # NCCL/HCCL backend for now; see https://github.com/pytorch/pytorch/issues/62300
    pg = dist.get_backend(ddp_ref().process_group)  # type: ignore[union-attr]
    if (pg != dist.Backend.NCCL) and (pg != "hccl"):
        raise RuntimeError(
            "Overlapping DDP with ZeRO using this approach currently requires "
            "NCCL/HCCL backend to avoid hangs"
        )

    if shard_buckets:
        zero._overlap_info.shard_buckets = True
        zero._overlap_info.total_size = 0

    def hook_with_zero_interleaved_fn(
        state,
        bucket: dist.GradBucket,
    ) -> torch.futures.Future[torch.Tensor]:
        r"""
        Return :class:`Future` that gives gradient bucket tensor and performs partial :class:`ZeroRedundancyOptimizer` :meth:`step`.

        This function uses the gradients in gradient in given bucket to perform a partial
        :class:`ZeroRedundancyOptimizer` :meth:`step`

        Arguments:
            state: any state for the hook.
            bucket (dist.GradBucket): the :class:`DistributedDataParallel`
                gradient bucket.
        """
        fut = hook(state, bucket)
        _hook_with_zero_step_setup(ddp_ref, zero, bucket)
        if zero._overlap_info.status != _OverlapStatus.INITIALIZED:
            return fut

        def zero_step(fut: torch.futures.Future) -> torch.Tensor:
            r"""
            Perform partial :class:`ZeroRedundancyOptimizer` :meth:`step` using gradients in the :class:`DistributedDataParallel`.

            Returns:
                A :class:`torch.Tensor` representing the contents of the
                gradient bucket.
            """
            overlap_info = zero._overlap_info
            bucket_index = bucket.index()
            rank = zero.global_rank

            assigned_ranks = overlap_info.assigned_ranks_per_bucket[bucket_index]
            overlap_info.bucket_indices_seen.append(bucket_index)
            if rank in assigned_ranks:
                _perform_local_step(bucket, zero, rank)

            _broadcast_bucket(bucket_index, zero)

            num_buckets = len(overlap_info.params_per_bucket)
            if len(overlap_info.bucket_indices_seen) == num_buckets:
                # Ensure that all parameter updates are finished before the
                # next forward pass
                overlap_info.wait_for_broadcasts()
                overlap_info.clear_per_iter_info()

            return bucket.buffer()

        return fut.then(zero_step)

    return hook_with_zero_interleaved_fn
