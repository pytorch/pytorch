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


def hook_then_zero_step(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future],
    ddp: DistributedDataParallel,
    zero: ZeroRedundancyOptimizer,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future]:
    r"""
    Modifies the given ``hook`` to additionally perform a partial
    :class:`ZeroRedundancyOptimizer` :meth:`step` using the gradients in the
    :class:`DistributedDataParallel` gradient bucket provided by the ``hook``.

    Arguments:
        hook (Any * dist.GradBucket -> torch.futures.Future): the hook to
            modify.
        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
            instance to use.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to use.

    Returns:
        The modified hook.
    """
    assert zero._is_functional_optim, "ZeroRedundancyOptimizer must be " \
        "constructed with a functional optimizer class to use this function"

    def hook_then_zero_fn(
        state,
        bucket: dist.GradBucket,
    ) -> torch.futures.Future:
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
        overlap_info = zero._overlap_info
        bucket_index = bucket.get_index()

        # Proceed as normal until the DDP buckets have been rebuilt
        if not ddp._has_rebuilt_buckets:
            assert overlap_info.status == _OverlapStatus.UNINITIALIZED
            return fut

        if overlap_info.status == _OverlapStatus.UNINITIALIZED:
            overlap_info.status = _OverlapStatus.DDP_HAS_REBUILT_BUCKETS

        rank = zero.global_rank
        rank_to_update = zero._ddp_bucket_index_to_rank(bucket_index)

        # Once DDP buckets have been rebuilt but ZeRO has not been
        # properly initialized yet, collect the information needed
        if overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
            bucket_params = bucket.get_model_params_for_bucket()
            assert len(bucket_params) > 0, "Empty bucket"
            params_per_rank = overlap_info.params_per_rank
            params_per_bucket = overlap_info.params_per_bucket
            if rank_to_update == rank:
                overlap_info.offsets[bucket_index] = len(params_per_rank[rank_to_update])
            params_per_rank[rank_to_update].extend(bucket_params)
            params_per_bucket.append(bucket_params)
            return fut

        assert overlap_info.status == _OverlapStatus.INITIALIZED

        # Save the bucket reference and all-reduce future for the final bucket
        if rank_to_update == rank:
            overlap_info.bucket_index_to_bucket[bucket_index] = bucket
            overlap_info.bucket_index_to_future[bucket_index] = fut

        num_buckets = len(overlap_info.params_per_bucket)
        # NOTE: The implementation from this point forward assumes that the
        # buckets are indexed incrementally starting from 0 in the order of
        # their autograd hooks firing
        is_last_bucket = bucket_index == num_buckets - 1
        if not is_last_bucket:
            return fut

        # Perform partial optimizer step on all buckets after the final
        # bucket has been computed
        for bucket_index in range(num_buckets):
            rank_to_update = zero._ddp_bucket_index_to_rank(bucket_index)
            num_local_optim_params = len(zero.optim.param_groups[0]["params"])
            if rank_to_update == rank:
                gradients: List[Optional[torch.Tensor]] = \
                    [_NO_PARAM_UPDATE for _ in range(num_local_optim_params)]
                assert bucket_index in overlap_info.offsets, \
                    f"Bucket index {bucket_index} was not assigned to rank {rank}"
                offset = overlap_info.offsets[bucket_index]
                # Ensure that the all-reduce completes before performing the
                # the parameter update
                assert bucket_index in overlap_info.bucket_index_to_future, \
                    f"All-reduce future for bucket {bucket_index} not saved " \
                    f"on rank {rank}"
                allreduce_future = overlap_info.bucket_index_to_future[bucket_index]
                allreduce_future.wait()
                bucket_gradients = overlap_info.bucket_index_to_bucket[bucket_index].get_per_parameter_tensors()
                for i, grad in enumerate(bucket_gradients):
                    gradients[offset + i] = grad
                zero._local_step(gradients)
            device = overlap_info.params_per_bucket[bucket_index][0].device
            device_index = zero._device_to_device_index[device]
            assert bucket_index in zero._buckets[device_index][rank_to_update]
            overlap_info.broadcast_handles.append(
                dist.broadcast(
                    zero._buckets[device_index][rank_to_update][bucket_index],
                    src=rank_to_update,
                    async_op=True
                )
            )

        # Zero each parameter's gradient if needed
        if zero._to_zero_grad:
            ZeroRedundancyOptimizer._zero_grad(zero._all_params)

        # Ensure that all parameter updates are finished before the
        # next forward pass
        _ = list(map(lambda x: x.wait(), overlap_info.broadcast_handles))
        overlap_info.broadcast_handles.clear()

        # Reset per-iteration information
        overlap_info.bucket_index_to_future.clear()
        overlap_info.bucket_index_to_bucket.clear()

        return fut

    return hook_then_zero_fn
