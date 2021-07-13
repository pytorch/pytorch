"""
A modified version of the legacy DistributedDataParallel module that uses c10d
communication primitives. This version is simpler than the latest PyTorch
version and is useful for debugging. Notably it does not overlap gradient
communication with the backward pass, which makes it slower but more robust
than the PyTorch version.

This version also supports the *no_sync* context manager, which allows faster
training with `--update-freq`.
"""

from collections import OrderedDict
from contextlib import contextmanager

import utils
import torch
import torch.nn as nn

class LegacyDistributedDataParallel(nn.Module):
    """Implements distributed data parallelism at the module level.

    A simplified version of :class:`torch.nn.parallel.DistributedDataParallel`.
    This version uses a c10d process group for communication and does not
    broadcast buffers.

    Args:
        module (~torch.nn.Module): module to be parallelized
        process_group: the c10d process group to be used for distributed data
            parallel all-reduction.
        buffer_size (int, optional): number of elements to buffer before
            performing all-reduce (default: 256M).
    """

    def __init__(self, module, process_group, buffer_size=2 ** 28):
        super(LegacyDistributedDataParallel, self).__init__()

        self.module = module
        self.process_group = process_group
        self.world_size = utils.get_world_size(self.process_group)

        # Never use a bigger buffer than the number of model params
        self.buffer_size = min(buffer_size, sum(p.numel() for p in module.parameters()))
        self.buffer = None

        # We can also forcibly accumulate grads locally and only do the
        # all-reduce at some later time.
        self.accumulate_grads = False

        # make per-device lists of parameters
        paramlists = OrderedDict()
        for param in self.module.parameters():
            device = param.device
            if paramlists.get(device) is None:
                paramlists[device] = []
            paramlists[device] += [param]
        self.per_device_params = list(paramlists.values())

    @contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        old_accumulate_grads = self.accumulate_grads
        self.accumulate_grads = True
        yield
        self.accumulate_grads = old_accumulate_grads

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def all_reduce_grads(self):
        """
        This function must be called explicitly after backward to reduce
        gradients. There is no automatic hook like c10d.
        """

        def all_reduce_params(params):
            buffer = self.buffer
            nonzero_buffer = False
            if len(params) > 1:
                offset = 0
                for p in params:
                    sz = p.numel()
                    if p.grad is not None:
                        buffer[offset : offset + sz].copy_(p.grad.data.view(-1))
                        nonzero_buffer = True
                    else:
                        buffer[offset : offset + sz].zero_()
                    offset += sz
            else:
                # we only have a single grad to all-reduce
                p = params[0]
                if p.grad is not None:
                    buffer = p.grad.data
                    nonzero_buffer = True
                elif p.numel() <= self.buffer.numel():
                    buffer = buffer[: p.numel()]
                    buffer.zero_()
                else:
                    buffer = torch.zeros_like(p)

            if nonzero_buffer:
                buffer.div_(self.world_size)

            utils.all_reduce(buffer, self.process_group)

            # copy all-reduced grads back into their original place
            offset = 0
            for p in params:
                sz = p.numel()
                if p.grad is not None:
                    p.grad.data.copy_(buffer[offset : offset + sz].view_as(p))
                else:
                    p.grad = buffer[offset : offset + sz].view_as(p).clone()
                offset += sz

        def reduction_fn():
            # This function only needs to be called once
            if self.accumulate_grads:
                return

            if self.buffer is None:
                self.buffer = next(self.module.parameters()).new(self.buffer_size)

            for params in self.per_device_params:
                # All-reduce the gradients in buckets
                offset = 0
                buffered_params = []
                for param in params:
                    if not param.requires_grad:
                        continue
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)

                    if hasattr(param, 'expert'):
                        # Skip gradient sync for unshared parameters
                        continue

                    if param.grad.requires_grad:
                        raise RuntimeError(
                            "DistributedDataParallel only works "
                            "with gradients that don't require "
                            "grad"
                        )
                    sz = param.numel()
                    if sz > self.buffer.numel():
                        # all-reduce big params directly
                        all_reduce_params([param])
                    else:
                        if offset + sz > self.buffer.numel():
                            all_reduce_params(buffered_params)
                            offset = 0
                            buffered_params.clear()
                        buffered_params.append(param)
                        offset += sz

                if len(buffered_params) > 0:
                    all_reduce_params(buffered_params)

        reduction_fn()
