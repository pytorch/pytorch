import contextlib
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.optim.zero_redundancy_optimizer import _OverlapStatus
from torch.nn.parallel.distributed import DistributedDataParallel


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
        if zero._use_extra_stream:
            fut.wait()

        with torch.cuda.stream(zero._optim_stream) if zero._use_extra_stream else contextlib.suppress():
            def zero_step(fut: torch.futures.Future) -> torch.Tensor:
                r"""
                Performs a partial :class:`ZeroRedundancyOptimizer` :meth:`step`
                using the gradients in the given :class:`DistributedDataParallel`
                gradient bucket.
                """
                # Proceed as normal until the DDP buckets have been rebuilt
                if not ddp._has_rebuilt_buckets:
                    return fut.wait()[0] if zero._use_extra_stream else bucket.get_tensor()

                bucket_index = bucket.get_index()
                rank = zero.global_rank
                overlap_info = zero._overlap_info
                if overlap_info.status == _OverlapStatus.UNINITIALIZED:
                    overlap_info.status = _OverlapStatus.DDP_HAS_REBUILT_BUCKETS

                bucket_params = bucket.get_model_params_for_bucket()
                assert len(bucket_params) > 0, "Empty bucket"
                rank_to_update = zero._ddp_bucket_index_to_rank(bucket_index)

                # Once DDP buckets have been rebuilt but ZeRO has not been
                # properly initialized yet, collect the information needed
                if overlap_info.status == _OverlapStatus.DDP_HAS_REBUILT_BUCKETS:
                    params_per_rank = overlap_info.params_per_rank
                    params_per_bucket = overlap_info.params_per_bucket
                    if rank_to_update == rank:
                        overlap_info.offsets[bucket_index] = len(params_per_rank[rank_to_update])
                    params_per_rank[rank_to_update].extend(bucket_params)
                    params_per_bucket.append(bucket_params)

                    return fut.wait()[0] if zero._use_extra_stream else bucket.get_tensor()

                if rank_to_update == rank:
                    assert len(zero.optim.param_groups) == 1, \
                        "Overlapping DDP with ZeRO only supports a single " \
                        "parameter group"
                    # Construct the `gradients` input for the local optimizer step,
                    # which expects `None` in a list position to indicate that the
                    # corresponding parameter should not be updated
                    num_local_optim_params = len(zero.optim.param_groups[0]["params"])
                    gradients: List[Optional[torch.Tensor]] = \
                        [None for _ in range(num_local_optim_params)]
                    assert bucket_index in overlap_info.offsets, \
                        f"Bucket index {bucket_index} was not assigned to rank " \
                        f"{rank}"
                    offset = overlap_info.offsets[bucket_index]
                    bucket_gradients = bucket.get_per_parameter_tensors()
                    for i, grad in enumerate(bucket_gradients):
                        gradients[offset + i] = grad
                    zero._local_step(gradients)

                device = bucket_params[0].device
                device_index = zero._device_to_device_index[device]
                assert bucket_index in zero._buckets[device_index][rank_to_update]
                dist.broadcast(zero._buckets[device_index][rank_to_update][bucket_index], src=rank_to_update, async_op=True)

                return fut.wait()[0] if zero._use_extra_stream else bucket.get_tensor()

        return fut.then(zero_step)

    return hook_then_zero_fn
