import functools
import torch
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

    def __init__(self, module, process_group, buffer_size=2 ** 22):
        super(PythonDDP, self).__init__()

        self.module = module
        self.process_group = process_group
        self.world_size = utils.get_world_size(self.process_group)

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

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    # TODO(bowangbj): Call Wait for all bucket reduction handle.
    # TODO(bowangbj): Update with Async All Reduce
    def all_reduce_grads(self):
        print('bowangbj all_reduce_grads invoked, {} buckets'.format(len(self.buckets)))
        for bucket in self.buckets:
            print('bowangbj bucket is full: {}'.format(bucket.is_full()))
            assert bucket.is_full()

            # TODO(bowangbj): use ASYNC flow.
            bucket.buffer.div_(self.world_size)
            utils.all_reduce(bucket.buffer, self.process_group)
            # Copy reduced-grad back into original place
            for cur_p, cur_offset in bucket.param_to_offset.items():
                sz = cur_p.numel()
                if cur_p.grad is not None:
                    # print('bowangbj copy_')
                    cur_p.grad.data.copy_(bucket.buffer[cur_offset : cur_offset + sz].view_as(cur_p))
                else:
                    # print('bowang clone')
                    cur_p.grad = bucket.buffer[cur_offset : cur_offset + sz].view_as(cur_p).clone()

        print('resetting buffer for next iteration')
        for bucket in self.buckets:
            bucket.ready_param_grad_count = 0

        # print('bowangbj === all_reduce_grads === invoked')
        # for p, bucket in self.param_to_bucket.items():
        #     n = p.numel()
        #     bucket_is_full = bucket.ready_param_grad_count == len(bucket.param_to_offset)
        #     assert bucket_is_full
        pass
        # def all_reduce_params(params):
        #     buffer = self.buffer
        #     nonzero_buffer = False
        #     if len(params) > 1:
        #         offset = 0
        #         for p in params:
        #             sz = p.numel()
        #             if p.grad is not None:
        #                 buffer[offset : offset + sz].copy_(p.grad.data.view(-1))
        #                 nonzero_buffer = True
        #             else:
        #                 buffer[offset : offset + sz].zero_()
        #             offset += sz
        #     else:
        #         # we only have a single grad to all-reduce
        #         p = params[0]
        #         if p.grad is not None:
        #             buffer = p.grad.data
        #             nonzero_buffer = True
        #         elif p.numel() <= self.buffer.numel():
        #             buffer = buffer[: p.numel()]
        #             buffer.zero_()
        #         else:
        #             buffer = torch.zeros_like(p)

        #     if nonzero_buffer:
        #         buffer.div_(self.world_size)

        #     utils.all_reduce(buffer, self.process_group)

        #     # copy all-reduced grads back into their original place
        #     offset = 0
        #     for p in params:
        #         sz = p.numel()
        #         if p.grad is not None:
        #             p.grad.data.copy_(buffer[offset : offset + sz].view_as(p))
        #         else:
        #             p.grad = buffer[offset : offset + sz].view_as(p).clone()
        #         offset += sz

        # def reduction_fn():
        #     if self.buffer is None:
        #         self.buffer = next(self.module.parameters()).new(self.buffer_size)

        #     for params in self.per_device_params:
        #         # All-reduce the gradients in buckets
        #         offset = 0
        #         buffered_params = []
        #         for param in params:
        #             if not param.requires_grad:
        #                 continue
        #             if param.grad is None:
        #                 param.grad = torch.zeros_like(param)

        #             if hasattr(param, 'expert'):
        #                 # Skip gradient sync for unshared parameters
        #                 continue

        #             if param.grad.requires_grad:
        #                 raise RuntimeError(
        #                     "DistributedDataParallel only works "
        #                     "with gradients that don't require "
        #                     "grad"
        #                 )
        #             sz = param.numel()
        #             if sz > self.buffer.numel():
        #                 # all-reduce big params directly
        #                 all_reduce_params([param])
        #             else:
        #                 if offset + sz > self.buffer.numel():
        #                     all_reduce_params(buffered_params)
        #                     offset = 0
        #                     buffered_params.clear()
        #                 buffered_params.append(param)
        #                 offset += sz

        #         if len(buffered_params) > 0:
        #             all_reduce_params(buffered_params)

        # reduction_fn()
