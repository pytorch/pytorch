import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module
from collections import defaultdict
from torch.autograd import Variable
import torch.utils.hooks


class DistributedDataParallelCPU(Module):
    r"""Implements distributed data parallelism for CPU at the module level.

    This module supports the ``mpi`` and ``gloo`` backends.

    This container parallelizes the application of the given module by splitting
    the input across the specified devices by chunking in the batch
    dimension. The module is replicated on each machine, and each such replica
    handles a portion of the input. During the backwards pass, gradients from
    each node are averaged.

    This module could be used in conjunction with the DistributedSampler,
    (see :class:`~torch.utils.data.distributed.DistributedSampler`)
    which will load a subset of the original dataset for each node with the same
    batch size. So strong scaling should be configured like this:

    n = 1, batch size = 12

    n = 2, batch size = 64

    n = 4, batch size = 32

    n = 8, batch size = 16

    Creation of this class requires the distributed package to be already
    initialized in the process group mode
    (see :func:`torch.distributed.init_process_group`).

    .. warning::
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) is a distributed synchronization
        point. Take that into account in case different node might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.

    .. warning::
        This module assumes all gradients are dense.

    .. warning::
        This module doesn't work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        won't be invoked anymore, unless the hooks are initialized in the
        :meth:`forward` method.

    .. note::
        Parameters are broadcast between nodes in the __init__() function. The
        module performs an all-reduce step on gradients and assumes that they
        will be modified by the optimizer in all nodes in the same way.

    Args:
        module: module to be parallelized

    Example::

        >>> torch.distributed.init_process_group(world_size=4, init_method='...')
        >>> net = torch.nn.DistributedDataParallelCPU(model)
    """

    def __init__(self, module):
        super(DistributedDataParallelCPU, self).__init__()
        self.module = module
        self.sync_parameters()

        def allreduce_params():
            if self.needs_reduction:
                self.needs_reduction = False
                buckets = defaultdict(list)
                for param in self.module.parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = type(param.data)
                        buckets[tp].append(param)

                for bucket in buckets.values():
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    dist.all_reduce(coalesced)
                    coalesced /= dist.get_world_size()
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)

        for param in list(self.module.parameters()):
            @torch.utils.hooks.unserializable_hook
            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(allreduce_params)

            if param.requires_grad:
                param.register_hook(allreduce_hook)

    def sync_parameters(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, 0)

    def forward(self, *inputs, **kwargs):
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)
