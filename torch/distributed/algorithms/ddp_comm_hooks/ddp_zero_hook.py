import contextlib
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
_PARAM_NO_UPDATE = None


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

        # Proceed as normal until the DDP buckets have been rebuilt
        if not ddp._has_rebuilt_buckets:
            assert overlap_info.status == _OverlapStatus.UNINITIALIZED
            return fut

        if overlap_info.status == _OverlapStatus.UNINITIALIZED:
            overlap_info.status = _OverlapStatus.DDP_HAS_REBUILT_BUCKETS

        # Once DDP buckets have been rebuilt but ZeRO has not been
        # properly initialized yet, collect the information needed
        if overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
            bucket_index = bucket.get_index()
            rank = zero.global_rank
            rank_to_update = zero._ddp_bucket_index_to_rank(bucket_index)

            bucket_params = bucket.get_model_params_for_bucket()
            assert len(bucket_params) > 0, "Empty bucket"
            params_per_rank = overlap_info.params_per_rank
            params_per_bucket = overlap_info.params_per_bucket
            if rank_to_update == rank:
                overlap_info.offsets[bucket_index] = len(params_per_rank[rank_to_update])
            params_per_rank[rank_to_update].extend(bucket_params)
            params_per_bucket.append(bucket_params)
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
            assert overlap_info.status == _OverlapStatus.INITIALIZED

            bucket_index = bucket.get_index()
            overlap_info.bucket_indices_seen.append(bucket_index)
            rank = zero.global_rank
            rank_to_update = zero._ddp_bucket_index_to_rank(bucket_index)

            if rank_to_update == rank:
                # Construct the `gradients` input for the local optimizer step,
                # which expects `None` in a list position to indicate that the
                # corresponding parameter should not be updated
                num_local_optim_params = len(zero.optim.param_groups[0]["params"])
                gradients = [_PARAM_NO_UPDATE for _ in range(num_local_optim_params)]
                assert bucket_index in overlap_info.offsets, \
                    f"Bucket index {bucket_index} was not assigned to rank " \
                    f"{rank}"
                offset = overlap_info.offsets[bucket_index]
                bucket_gradients = bucket.get_per_parameter_tensors()
                for i, grad in enumerate(bucket_gradients):
                    gradients[offset + i] = grad
                assert bucket_index not in overlap_info.bucket_to_gradients, \
                    f"Already a gradient list for bucket index {bucket_index}"

                # Save the `gradients` input and the all-reduce future
                overlap_info.bucket_to_gradients[bucket_index] = gradients
                overlap_info.bucket_to_allreduce_future[bucket_index] = fut

            # `bucket_index` does not refer to the argument `bucket` 's index
            # from this point forward
            del bucket_index

            num_buckets = len(overlap_info.params_per_bucket)
            is_last_bucket = len(overlap_info.bucket_indices_seen) == num_buckets

            # Perform partial optimizer step on all buckets
            if is_last_bucket:
                for bucket_index in range(num_buckets):
                    rank_to_update = zero._ddp_bucket_index_to_rank(bucket_index)
                    if rank_to_update == rank:
                        assert bucket_index in overlap_info.bucket_to_gradients, \
                            f"Bucket index {bucket_index} assigned to rank {rank} is not present"
                        gradients = overlap_info.bucket_to_gradients[bucket_index]
                        # Ensure that the all-reduce completes before
                        # performing the parameter update
                        allreduce_future = overlap_info.bucket_to_allreduce_future[bucket_index]
                        allreduce_future.wait()
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
                if zero._zero_grad:
                    ZeroRedundancyOptimizer._zero_grad(zero._all_params)

                # Ensure that all parameter updates are finished before the
                # next forward pass
                _ = list(map(lambda x: x.wait(), overlap_info.broadcast_handles))
                overlap_info.broadcast_handles.clear()

                # Reset per-iteration information
                overlap_info.bucket_to_gradients.clear()
                overlap_info.bucket_to_allreduce_future.clear()
                overlap_info.bucket_indices_seen.clear()

            return bucket.get_tensor()

        return fut.then(zero_step)

    return hook_then_zero_fn
