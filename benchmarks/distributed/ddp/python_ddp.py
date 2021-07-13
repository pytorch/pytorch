import functools
import torch
import torch.distributed as dist
import torch.nn as nn
import utils

from collections import OrderedDict

class PythonDDP(nn.Module):

    class Bucket:
        def __init__(self, max_buffer_size):
            self.param_to_offset = {}
            # buffer would be allocated in backward callback.
            self.buffer = None
            self.ready_param_grad_count = 0
            self.total_elements = 0
            self._MAX_BUFFER_SIZE = max_buffer_size

        def __str__(self):
            return "bucket: num_param={}, total_elements={}, ready_param_grad_count={}".format(
                len(self.param_to_offset),
                self.total_elements,
                self.ready_param_grad_count
            )

        def is_full(self):
            assert self.ready_param_grad_count >= 0
            assert self.ready_param_grad_count <= len(self.param_to_offset)
            return len(self.param_to_offset) == self.ready_param_grad_count

        def empty(self):
            self.ready_param_grad_count = 0

        # Checks whether current bucket hold the param.
        # Returns true if current bucket can hold current param.
        # Otherwise False.
        def try_hold_param(self, param):
            if self.total_elements + param.numel() <= self._MAX_BUFFER_SIZE :
                self.param_to_offset[param] = self.total_elements
                self.total_elements += param.numel()
                return True
            else:
                return False

    def __init__(self, module, process_group, async_reduction=True, buffer_size=2 ** 22):
        super(PythonDDP, self).__init__()

        self.module = module
        self.process_group = process_group
        self.world_size = utils.get_world_size(self.process_group)
        self.async_reduction = async_reduction
        # Holds all_reduce handls, valid when async_reduction is True
        self.async_handles = set()

        # Ensure buffer size is large enough to hold largest param
        max_numel = max(p.numel() for p in module.parameters())
        assert buffer_size > max_numel, "buffer_size: {} should be larger than largest param: {}".format(buffer_size, max_numel)

        # Build buckets for params
        self.param_to_bucket, self.buckets = self.build_buckets_for_params(buffer_size)

        # Register per-parameter backward grad ready hook
        for p in self.module.parameters():
            assert p.requires_grad is True
            p.register_hook(functools.partial(self._on_param_grad_ready, p))

    # Distribute params into list of buckets.
    # Return (param_to_buckets, buckets)
    def build_buckets_for_params(self, max_buffer_size):
        print("build_buckets_for_params ... ")
        params_to_buckets = {}
        buckets = set()
        cur_bucket = self.Bucket(max_buffer_size)
        total_param = 0
        for param in self.module.parameters():
            total_param += 1
            assert param.requires_grad, "param.requires_grad must be True"
            if cur_bucket.try_hold_param(param):
                params_to_buckets[param] = cur_bucket
                buckets.add(cur_bucket)
            else:
                new_bucket = self.Bucket(max_buffer_size)
                assert new_bucket.try_hold_param(param), "param must be holded in a empty bucket!"
                params_to_buckets[param] = new_bucket
                buckets.add(new_bucket)
                cur_bucket = new_bucket

        print('allocating buffer for buckets')
        first_param = next(self.module.parameters())
        for bucket in buckets:
            bucket.buffer = first_param.new(bucket.total_elements)
            assert bucket.buffer is not None, 'bucket.buffer should not be None'
        print("len(param_to_bucket)={}, len(buckets)={}".format(
            len(params_to_buckets), len(buckets)))

        print("bowangbj model.params count = {}".format(total_param))
        total_params_in_buckets = 0
        for bucket in buckets:
            total_params_in_buckets += len(bucket.param_to_offset)
        print(f"bowangbj total_params_in_buckets = {total_params_in_buckets}")

        return params_to_buckets, buckets

    # Callback when param.grad is ready. Note during callback, param.grad won't
    # be ready yet, we MUST use the given ''grad'' which would be passed upon
    # callback.
    def _on_param_grad_ready(self, param, grad):
        # print('== bowangbj _on_param_grad_ready invoked')

        # Find the bucket holding param and corresponding offset
        bucket = self.param_to_bucket.get(param)
        assert bucket is not None, "Failed to find bucket for param"
        offset = bucket.param_to_offset.get(param)
        assert offset is not None, "offset must be set for param"
        assert bucket.buffer is not None, "buffer not allocated"

        # Copy grad to bucket, note param.grad isn't ready yet.
        # print('bowang copying grad to bucket')
        sz = param.numel()
        # assert param.grad is None
        assert grad is not None, 'grad should be set'
        assert param.requires_grad
        assert param.numel() == grad.numel(), 'sz of param and grad must be equal'
        bucket.buffer[offset : offset + sz].copy_(grad.data.view(-1))
        bucket.ready_param_grad_count += 1

        # Start reducing current bucket when it's full. This ensures grad reduction
        # and grad calculation runs in parallel.
        if self.async_reduction and bucket.is_full():
            print('bowangbj bucket is full, all_reducing async ... ')
            bucket.buffer.div_(self.world_size)
            handle = dist.all_reduce(bucket.buffer, dist.ReduceOp.SUM, self.process_group, True)
            self.async_handles.add(handle)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    # TODO(bowangbj): Call Wait for all bucket reduction handle.
    # TODO(bowangbj): Update with Async All Reduce
    def all_reduce_grads(self):
        print('bowangbj all_reduce_grads invoked, {} buckets, async_reduction={}'.format(
            len(self.buckets), self.async_reduction))

        if self.async_reduction:
            print('bowangbj waiting async all reduction')
            for handle in self.async_handles:
                handle.wait()
            self.async_handles.clear()
            print('bowangbj all work from async handle done')
        else:
            print('bowangbj async_reduction=false')
            # Defer grad reduction post backward.
            for bucket in self.buckets:
                assert bucket.is_full(), 'bucket should be full'
                bucket.buffer.div_(self.world_size)
                utils.all_reduce(bucket.buffer, self.process_group)

        print('bowangbj copying reduced-grads back to original params.')
        # Copy reduced-grad back into original place
        for bucket in self.buckets:
            assert bucket.is_full(), 'bucket should be full'
            for cur_p, cur_offset in bucket.param_to_offset.items():
                sz = cur_p.numel()
                if cur_p.grad is not None:
                    cur_p.grad.data.copy_(bucket.buffer[cur_offset : cur_offset + sz].view_as(cur_p))
                else:
                    cur_p.grad = bucket.buffer[cur_offset : cur_offset + sz].view_as(cur_p).clone()

        # Empty bucket for next epoch
        for bucket in self.buckets:
            bucket.empty()
