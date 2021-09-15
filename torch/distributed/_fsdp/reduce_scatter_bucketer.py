# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup


class Bucket:
    def __init__(self, data: Tensor, group: ProcessGroup):
        self.data = data
        self.group = group
        self.offset = 0
        self.callbacks: List[Callable] = []
        self.output_shard = torch.zeros_like(data[0])

    def flush(self) -> None:
        """Flush content of the bucket."""
        if self.offset == 0:
            assert len(self.callbacks) == 0
            return
        # reduce-scatter bucket
        if hasattr(dist, "_reduce_scatter_base"):
            dist._reduce_scatter_base(
                self.output_shard[: self.offset], self.data[:, : self.offset].contiguous(), group=self.group
            )
        else:
            dist.reduce_scatter(
                self.output_shard[: self.offset], list(self.data[:, : self.offset].unbind(0)), group=self.group
            )
        # execute post-reduction callbacks
        for callback_fn in self.callbacks:
            callback_fn()
        # reuse input bucket but allocate a fresh output shard
        self.data[:, : self.offset].zero_()
        self.offset = 0
        self.callbacks.clear()
        self.output_shard = torch.zeros_like(self.data[0])

    def setup(self) -> None:
        """Setup the buffers if they are not allocated.
        Using ``setup`` and ``teardown``, we can ensure that the bucket
        buffers are only allocated during the backward pass, hence saving more
        memory to other parts of the training process, such as the forward pass
        for activation memory.
        """
        for tensor in [self.data, self.output_shard]:
            if tensor.storage().size() == 0:
                tensor.storage().resize_(tensor.size().numel())  # type: ignore[attr-defined]

    def teardown(self) -> None:
        """Tear down the bucket by freeing the memory"""
        assert self.offset == 0 and self.callbacks == [], "Incorrect call of teardown"
        for tensor in [self.data, self.output_shard]:
            tensor.storage().resize_(0)  # type: ignore[attr-defined]


class ReduceScatterBucketer:
    """
    Helper for bucketing multiple reduce-scatter operations on small tensors
    into larger reduce-scatter ops to improve communication efficiency.
    Usage::
        bucketer = ReduceScatterBucketer()
        bucketer.reduce_scatter_async(
            small_tensors, callback_fn=lambda result: print("small")
        )
        bucketer.reduce_scatter_async(
            big_tensors, callback_fn=lambda result: print("big")
        )
        bucketer.reduce_scatter_async(
            more_small_tensors, callback_fn=lambda result: print("small2")
        )
        bucketer.flush()  # callbacks only guaranteed to be called after flush()
        # Example output (note that it is out of order, due to bucketing):
        # big
        # small
        # small2
    Args:
        bucket_cap_mb (int, Optional): bucket size for communicating. Buckets
            are sub-divided based on world_size. Values <= 0 disable bucketing.
    """

    def __init__(self, bucket_cap_mb: int = 25):
        self.bucket_cap_mb = bucket_cap_mb
        self.buckets: Dict[Tuple[torch.dtype, torch.device, ProcessGroup], Bucket] = {}

    @torch.no_grad()
    def reduce_scatter_async(
        self, input_list: List[Tensor], group: ProcessGroup, callback_fn: Optional[Callable] = None,
    ) -> None:
        """
        Reduce-scatter a list of tensors asynchronously, so smaller reductions
        can be bucketed together. The given callback (``callback_fn``) will be
        called with the reduced result at some later time. Call ``flush()`` to
        force all queued ops and callbacks to be executed.
        Note that large inputs will be reduced immediately, and this function
        may also flush the relevant bucket to make room for ``input_list``.
        Args:
            input_list (List[Tensor]): list of tensors to reduce-scatter. List
                should contain ``group.size()`` tensors and each tensor should
                have identical shape, dtype and device.
            group (ProcessGroup): process group for reduction
            callback_fn (Callable, Optional): callback function to call after
                the reduction executes. Function will be called with a single
                argument corresponding to the reduced result.
        """
        world_size = group.size()

        assert (
            len(input_list) == world_size
        ), f"reduce_scatter received {len(input_list)} inputs, expected group.size() ({world_size})"

        first_input = input_list[0]
        first_input_size = first_input.numel()

        bucket_shard_size = self._get_shard_size(first_input.element_size(), world_size)
        if first_input_size > bucket_shard_size:
            # TODO: investigate how to avoid using torch.cat (because it seems to be slow for CPU tensors)
            # input is too big to fit in the bucket, reduce-scatter directly
            output = torch.zeros_like(input_list[0])
            if hasattr(dist, "_reduce_scatter_base"):
                input_flattened = torch.cat(input_list)
                dist._reduce_scatter_base(output, input_flattened, group=group)
            else:
                # fallback
                dist.reduce_scatter(output, input_list, group=group)
            if callback_fn is not None:
                callback_fn(output)
            return

        bucket = self._get_bucket(first_input, group)
        if first_input_size > bucket.data.size(1) - bucket.offset:
            # not enough space remaining in bucket, flush it now
            bucket.flush()

        # copy data from input_list into bucket
        stacked_input = torch.stack(input_list).view(world_size, first_input_size)
        offset = bucket.offset
        bucket.data[:, offset : offset + first_input_size].copy_(stacked_input)
        bucket.offset += first_input_size

        # callback will be given the reduced result
        if callback_fn is not None:
            result_view = bucket.output_shard[offset : offset + first_input_size].view_as(first_input)
            bucket.callbacks.append(functools.partial(callback_fn, result_view))

    @torch.no_grad()
    def flush(self) -> None:
        """Reduce-scatter any partial buckets."""
        for bucket in self.buckets.values():
            bucket.flush()

    @torch.no_grad()
    def teardown(self) -> None:
        """Free buffers from all buckets."""
        for bucket in self.buckets.values():
            bucket.teardown()

    @functools.lru_cache()
    def _get_shard_size(self, element_size: int, num_shards: int) -> int:
        if self.bucket_cap_mb <= 0:  # Values <= 0 disable bucketing.
            return 0
        MB = 1024 * 1024
        bucket_size = self.bucket_cap_mb * MB / element_size
        return int(bucket_size // num_shards)

    def _get_bucket(self, tensor: Tensor, group: ProcessGroup) -> Bucket:
        # TODO (Min): the `group` used here in the key is the object hash, not the content
        #     hash. That means if FSDP instances are initialized with different process groups,
        #     even when the group members are in fact the same, we end up creating different
        #     buckets here.
        key = (tensor.dtype, tensor.device, group)
        if key not in self.buckets:
            # buckets are divided into world_size pieces, bucket.data shaped (world_size, shard_size)
            world_size = group.size()
            shard_size = self._get_shard_size(tensor.element_size(), world_size)
            data = tensor.new_zeros((world_size, shard_size))
            self.buckets[key] = Bucket(data, group)
        self.buckets[key].setup()
        return self.buckets[key]
