import torch
import torch.nn as nn
import utils

from collections import OrderedDict

# TODO(bowangbj): move into util
class Bucket:
    def __init__(self, max_buffer_size):
        self.param_to_offset = {}
        # buffer will be allocated in param.grad hook if None
        self.buffer = None
        self.ready_param_grad_count = 0
        self.total_elements = 0
        self._MAX_BUFFER_SIZE = max_buffer_size

    def __str__(self):
        return "num_param={}, total_elements={}".format(
            len(self.param_to_offset), self.total_elements)

    # Checks whether current bucket hold the param.
    # Returns true if current bucket can hold current param.
    # Otherwise False.
    def find_space_for_param(self, param):
        if self.total_elements + param.numel() <= self._MAX_BUFFER_SIZE :
            self.param_to_offset[param] = self.total_elements
            self.total_elements += param.numel()
            return True
        else:
            return False

# TODO(bowangbj): move into class member func
# Distribute params into list of buckets.
# Returns params to bucket which contains actual position for the corresponding param.
def build_buckets_for_params(params, max_buffer_size):
    params_to_buckets = {}
    cur_bucket = Bucket(max_buffer_size)
    for param in params:
        print('bowang_bj param.sumel ' + str(param.numel()))
        if not param.requires_grad:
            print('Warning: param.requires_grad should be True')
            continue
        if cur_bucket.find_space_for_param(param):
           params_to_buckets[param] = cur_bucket
        else:
            new_bucket = Bucket(max_buffer_size)
            assert new_bucket.find_space_for_param(param)
            params_to_buckets[param] = new_bucket
            cur_bucket = new_bucket
    return params_to_buckets

class PythonDDP(nn.Module):

    def __init__(self, module, process_group, buffer_size=2 ** 28):
        super(PythonDDP, self).__init__()

        self.module = module
        self.process_group = process_group
        self.world_size = utils.get_world_size(self.process_group)

        # TODO(bowangbj): Remove unused params
        # Never use a bigger buffer than the number of model params (elements)
        self.buffer_size = min(buffer_size, sum(p.numel() for p in module.parameters()))
        self.buffer = None

        # Ensure buffer size is large enough to hold largest param
        self.max_buffer_size = max(buffer_size, max(p.numel() for p in module.parameters()))

        # Distribute params into multiple buckets.
        # param_to_bucket maps param to corresponding bucket, which has detailed position info
        self.param_to_bucket = build_buckets_for_params(self.module.parameters(), self.max_buffer_size)
        # debug only
        for bucket in self.param_to_bucket.values():
            num = len(bucket.param_to_offset)
            print(f'bowangbj bucket has {num} params, toal={bucket.total_elements}')

        # Register per-parameter hook
        for param in self.module.parameters():
            if param.requires_grad:
                p_tmp = param.expand_as(param)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._on_param_grad_ready(param))
            else:
                print('Warning current param should requires_grad as True')

    # TODO(bowangbj): Rename for better name.
    def _on_param_grad_ready(self, param):
        print('== bowangbj _on_param_grad_ready invoked')

        bucket = self.param_to_bucket.get(param)
        assert bucket is not None
        offset = bucket.param_to_offset.get(param)
        assert offset is not None
        print('bowangbj bucket is ' + str(bucket))

        # TODO(bowangbj): Copy grad from param to bucket
        # TODO(bowangbj): increate n_ready_grad count to bucket
        # TODO(bowangbj): kickoff all_reduce ASYNC flow

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    # TODO(bowangbj): Call Wait for all bucket reduction handle.
    # TODO(bowangbj): Update with Async All Reduce
    def all_reduce_grads(self):
        """
        This function must be called explicitly after backward to reduce
        gradients. There is no automatic hook like c10d.
        """

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
