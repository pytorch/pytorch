import copy

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors

from torch.cuda.comm import broadcast_coalesced
from torch.cuda import nccl
import torch.distributed.c10d as c10d

from ..modules import Module
from .replicate import replicate
from .scatter_gather import scatter_kwargs, gather
from .parallel_apply import parallel_apply


class _DistributedDataParallelC10d(Module):
    r"""Implements distributed data parallelism that is based on c10d at the
    module level.

    Currently this module is EXPERIMENTAL ONLY and should not be
    used by normal users. Instead, please use DistributedDataParallel.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. The module is replicated on each machine and each device, and
    each such replica handles a portion of the input. During the backwards
    pass, gradients from each node are averaged.

    The batch size should be larger than the number of GPUs used locally. It
    should also be an integer multiple of the number of GPUs so that each chunk
    is the same size (so that each GPU processes the same number of samples).

    See also: :ref:`distributed-basics` and :ref:`cuda-nn-dataparallel-instead`.
    The same constraints on input as in :class:`torch.nn.DataParallel` apply.

    Creation of this class requires the c10d process group to be already
    initialized. This class will basically operate on the provided c10d
    process group.

    .. warning::
        This module works only with the ``gloo`` and ``nccl`` process groups.

    .. warning::
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) is a distributed synchronization
        point. Take that into account in case different processes might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.
        Same applies to buffers.

    -- warning::
        This module assumes all parameters are registered in the model of each
        distributed processes are in the same order. The module itself will
        conduct gradient all-reduction following the reverse order of the
        registered parameters of the model. In other wise, it is users'
        responsibility to ensure that each distributed process has the exact
        same model and thus the exact parameter registeration order.

    .. warning::
        This module assumes all buffers and gradients are dense.

    .. warning::
        This module doesn't work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. warning::
        If you plan on using this module with a ``nccl`` process group or
        a ``gloo`` process group (that uses Infiniband), together with a
        DataLoader that uses multiple workers, please change the multiprocessing
        start method to ``forkserver`` (Python 3 only) or ``spawn``.
        Unfortunately Gloo (that uses Infiniband) and NCCL2 are not fork safe,
        and you will likely experience deadlocks if you don't change this
        setting.

    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in process of rank
        0, to all other replicas in the system in every iteration.

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        won't be invoked anymore, unless the hooks are initialized in the
        :meth:`forward` method.

    Args:
        module: module to be parallelized
        process_group: the c10d process group to be used for distributed data
                       all-reduction
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])
        broadcast_buffers: flag that enables syncing (broadcasting) buffers of
                           the module at beginning of the forward function.
                           (default: True)
        bucket_cap_mb: DistributedDataParallelC10d will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       bucket_cap_mb controls the bucket size in MegaBytes (MB)
                       (default: 25)

    Attributes:
        module (Module): the module to be parallelized

    Example::
        >>> store = torch.distributed.c10d.FileStore("/tmp/tempfile.txt")
        >>> pg = torch.distributed.c10d.ProcessGroupGloo(store, rank, world_size)
        >>> net = torch.nn._DistributedDataParallelC10d(model, pg)
    """
    def __init__(self, module, process_group, device_ids=None,
                 output_device=None, dim=0, broadcast_buffers=True,
                 bucket_cap_mb=25):

        super(_DistributedDataParallelC10d, self).__init__()

        # Use all devices by default
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))

        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.process_group = process_group
        self.device_ids = device_ids
        self.output_device = output_device
        self.broadcast_buffers = broadcast_buffers

        MB = 1024 * 1024

        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = 25 * MB

        # Sync params and buffers
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._dist_broadcast_coalesced(module_states,
                                           self.broadcast_bucket_size)

        if len(device_ids) > 1:
            # TODO: we don't need to replicate params in here. they're always going to
            # be broadcasted using larger blocks in broadcast_coalesced, so it might be
            # better to not pollute the caches with these small blocks
            self._module_copies = replicate(self.module, self.device_ids, detach=True)
            self._module_copies[0] = self.module

            for module_copy in self._module_copies[1:]:
                for param, copy_param in zip(self.module.parameters(), module_copy.parameters()):
                    copy_param.requires_grad = param.requires_grad

        else:
            self._module_copies = [self.module]

        self.modules_params_data = [[] for _ in range(len(self.device_ids))]
        self.modules_buffers_data = [[] for _ in range(len(self.device_ids))]

        for dev_idx, module in enumerate(self._module_copies):
            self.modules_params_data[dev_idx] = [p.data for p in module.parameters()]
            self.modules_buffers_data[dev_idx] = [b.data for b in module.buffers()]

        bucket_bytes_cap = bucket_cap_mb * MB

        # This is a triply-nested list where the "dimensions" are: devices, buckets, bucket_elems
        param_buckets = []
        # Split the parameters into buckets and by types as well
        param_buckets = [list(_take_tensors(m.parameters(), bucket_bytes_cap)) for m in self._module_copies]

        self.bucket_sizes = []
        self.bucket_map = {}

        # We transpose param_buckets, so the loop is over buckets.
        # param_buckets_tuple is a doubly-nested list with "dims": devices, bucket_elems
        for bucket_idx, param_buckets_tuple in enumerate(zip(*param_buckets)):
            self.bucket_sizes.append(0)
            # Now, we transpose again, so we iterate over bucket_elems, but getting tuples
            # of params from each device.
            for idx, param_tuple in enumerate(zip(*param_buckets_tuple)):
                if not param_tuple[0].requires_grad:
                    continue
                for p in param_tuple:
                    self.bucket_map[p] = (bucket_idx, idx)
                self.bucket_sizes[bucket_idx] += 1

        self.buckets = [[[None for _ in range(self.bucket_sizes[i])]
                        for _ in range(len(self.device_ids))] for i in range(len(self.bucket_sizes))]
        # The number of params ready in each bucket
        self.buckets_ready_size = [[0 for _ in range(len(self.device_ids))] for i in range(len(self.bucket_sizes))]

        # coalesced bucket for only device 0
        self.buckets_coalesced = [[] for _ in range(len(self.bucket_sizes))]
        # We will always reduce the bucket following the reverse order
        # that is, alway reduces following the order of: n - 1, n - 2, ..., 0
        self.next_bucket = len(self.bucket_sizes) - 1
        self.ready_buckets_not_reduced = set()
        self.reduction_works = [None for _ in range(len(self.bucket_sizes))]

        self.devs_ready = [0 for _ in range(len(self.bucket_sizes))]

        # default stream tracking to launch nccl reduce kernels
        self.default_streams = []
        for dev_id in self.device_ids:
            with torch.cuda.device(dev_id):
                self.default_streams.append(torch.cuda.current_stream())

        self._register_grad_hooks()

    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        del attrs['_grad_accs']
        return attrs

    def __setstate__(self, state):
        super(_DistributedDataParallelC10d, self).__setstate__(state)
        self._register_grad_hooks()

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        self._sync_params()
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def train(self, mode=True):
        super(_DistributedDataParallelC10d, self).train(mode)
        for module in self._module_copies[1:]:
            module.train(mode)

    def _dist_broadcast_coalesced(self, tensors, buffer_size):
        c10d._dist_broadcast_coalesced(tensors, buffer_size, self.process_group)

    def _sync_params(self):
        if len(self.device_ids) > 1:
            # intra-node parameter sync
            result = broadcast_coalesced(self.modules_params_data[0],
                                         self.device_ids,
                                         self.broadcast_bucket_size)
            for tensors, module_params_data in zip(result[1:], self.modules_params_data[1:]):
                for tensor, param_data in zip(tensors, module_params_data):
                    param_data.set_(tensor)

        # module buffer sync
        if self.broadcast_buffers:
            if len(self.modules_buffers_data[0]) > 0:
                # cross-node buffer sync
                self._dist_broadcast_coalesced(self.modules_buffers_data[0],
                                               self.broadcast_bucket_size)
                if len(self.device_ids) > 1:
                    # intra-node buffer sync
                    result = broadcast_coalesced(self.modules_buffers_data[0],
                                                 self.device_ids,
                                                 self.broadcast_bucket_size)
                    for tensors, module_buffers_data in zip(result[1:], self.modules_buffers_data[1:]):
                        for tensor, buffer_data in zip(tensors, module_buffers_data):
                            buffer_data.set_(tensor)

    def _register_grad_hooks(self):
        self._grad_accs = []  # need to keep them in scope
        for device_idx, module in enumerate(self._module_copies):
            for p in module.parameters():
                if p.requires_grad:
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(p, device_idx))
                    self._grad_accs.append(grad_acc)

    def _make_param_hook(self, param, device_idx):
        bucket_idx, bucket_offset = self.bucket_map[param]

        def distributed_data_parallel_hook(*unused):
            if param.grad.requires_grad:
                raise RuntimeError("DistributedDataParallelC10d only works "
                                   "with gradients that don't require grad")
            bucket = self.buckets[bucket_idx][device_idx]
            bucket[bucket_offset] = param.grad.data
            self.buckets_ready_size[bucket_idx][device_idx] += 1

            # We can flush these and save memory for replicas
            if device_idx > 0:
                param.grad = None
                param.data.set_()

            # Current device's bucket is full
            if self.buckets_ready_size[bucket_idx][device_idx] == self.bucket_sizes[bucket_idx]:
                self.devs_ready[bucket_idx] += 1
                if self.devs_ready[bucket_idx] < len(self.device_ids):
                    return

                # Now all devices's buckets with index: bucket_idx are ready
                if bucket_idx == self.next_bucket:
                    self._queue_reduction(bucket_idx)
                    self.next_bucket -= 1
                    # Now reduce anything that is ready but not yet reduced
                    if len(self.ready_buckets_not_reduced) > 0:
                        sorted_todo = sorted(self.ready_buckets_not_reduced, reverse=True)
                        for i in sorted_todo:
                            # Nothing can be reduced now
                            if i < self.next_bucket:
                                break
                            self._queue_reduction(i)
                            self.ready_buckets_not_reduced.remove(i)
                            if i == self.next_bucket:
                                self.next_bucket -= 1
                else:
                    self.ready_buckets_not_reduced.add(bucket_idx)

                # When all devices' buckets
                if self.next_bucket == -1:
                    # A final sync for all the reduction works
                    self._sync_reduction_works()

        return distributed_data_parallel_hook

    def _queue_reduction(self, bucket_idx):
        grads_batch = self.buckets[bucket_idx]
        grads_batch_coalesced = []

        # coalesce the bucket
        for dev_id, dev_grads_batch in zip(self.device_ids, grads_batch):
            with torch.cuda.device(dev_id):
                dev_grads_batch_coalesced = _flatten_dense_tensors(dev_grads_batch)
                grads_batch_coalesced.append(dev_grads_batch_coalesced)

        # reduce to the first GPU in self.device_ids
        if len(self.device_ids) > 1:
            nccl.reduce(grads_batch_coalesced, root=0, streams=self.default_streams)

        # now work on the first gpu
        reduction_work = c10d.all_reduce(grads_batch_coalesced[0], self.process_group)
        self.reduction_works[bucket_idx] = reduction_work
        self.buckets_coalesced[bucket_idx] = grads_batch_coalesced[0]

    def _sync_reduction_works(self):
        # Now only work on the first GPU of self.device_ids, uncoalesce
        # the gradients for each bucket
        for bucket_idx, grads_batch in enumerate(self.buckets):
            # wait will let current stream wait on the c10d reduction stream
            self.reduction_works[bucket_idx].wait()

            self.buckets_coalesced[bucket_idx] /= self.process_group.size()
            grads_batch_reduced = _unflatten_dense_tensors(
                self.buckets_coalesced[bucket_idx], grads_batch[0])

            for grad, reduced in zip(grads_batch[0], grads_batch_reduced):
                grad.copy_(reduced)

        # Reset the module states
        self.next_bucket = len(self.bucket_sizes) - 1
        self.ready_buckets_not_reduced = set()
        self.reduction_works = [None for _ in range(len(self.bucket_sizes))]
        self.devs_ready = [0 for _ in range(len(self.bucket_sizes))]

        self.buckets = [[[None for _ in range(self.bucket_sizes[i])]
                        for _ in range(len(self.device_ids))] for i in range(len(self.bucket_sizes))]
        self.buckets_coalesced = [[] for _ in range(len(self.bucket_sizes))]
        self.buckets_ready_size = [[0 for _ in range(len(self.device_ids))] for i in range(len(self.bucket_sizes))]
