import copy

import torch

from torch.cuda.comm import broadcast_coalesced
from torch.cuda import nccl
import torch.distributed as dist

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group

from ..modules import Module
from .replicate import replicate
from .scatter_gather import scatter_kwargs, gather
from .parallel_apply import parallel_apply
from torch.cuda._utils import _get_device_index


class DistributedDataParallel(Module):
    r"""Implements distributed data parallelism that is based on
    torch.distributed package at the module level.

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

    Creation of this class requires that ``torch.distributed`` to be already
    initialized, by calling :func:`torch.distributed.init_process_group`

    ``DistributedDataParallel`` can be used in the following two ways:

    (1) Single-Process Multi-GPU

    In this case, a single process will be
    spawned on each host/node and each process will operate on all the GPUs
    of the node where it's running. To use ``DistributedDataParallel`` in
    this way, you can simply construct the model as the following:

        >>> torch.distributed.init_process_group(backend="nccl")
        >>> model = DistributedDataParallel(model) # device_ids will include all GPU devices be default

    (2) Multi-Process Single-GPU

    This is the highly recommended way to use ``DistributedDataParallel``, with
    multiple processes, each of which operates on a single GPU. This is
    currently the fastest approach to do data parallel training using PyTorch
    and applies to both single-node(multi-GPU) and multi-node data
    parallel training. It is proven to be significantly faster than
    :class:`torch.nn.DataParallel` for single-node multi-GPU data
    parallel training.

    Here is how to use it: on each host with N GPUs, you should spawn up N
    processes, while ensuring that each process invidually works on a single GPU
    from 0 to N-1. Therefore, it is your job to ensure that your training script
    operates on a single given GPU by calling:

        >>> torch.cuda.set_device(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``

    .. note:: ``nccl`` backend is currently the fastest and
        highly recommended backend to be used with Multi-Process Single-GPU
        distributed training and this applies to both single-node and multi-node
        distributed training

    .. warning::
        This module works only with the ``gloo`` and ``nccl`` backends.

    .. warning::
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) is a distributed synchronization
        point. Take that into account in case different processes might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.
        Same applies to buffers.

    .. warning::
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

        If you plan on using this module with a ``nccl`` backend or a ``gloo``
        backend (that uses Infiniband), together with a DataLoader that uses
        multiple workers, please change the multiprocessing start method to
        ``forkserver`` (Python 3 only) or ``spawn``. Unfortunately
        Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will
        likely experience deadlocks if you don't change this setting.

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        won't be invoked anymore, unless the hooks are initialized in the
        :meth:`forward` method.

    .. warning::
        You should never try to change your model's parameters after wrapping
        up your model with DistributedDataParallel. In other words, when
        wrapping up your model with DistributedDataParallel, the constructor of
        DistributedDataParallel will register the additional gradient
        reduction functions on all the parameters of the model itself at the
        time of construction. If you change the model's parameters after
        the DistributedDataParallel construction, this is not supported and
        unexpected behaviors can happen, since some parameters' gradient
        reduction functions might not get called.

    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in process of rank
        0, to all other replicas in the system in every iteration.

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices (default: all devices)
        output_device (int or torch.device): device location of output (default: device_ids[0])
        broadcast_buffers (bool): flag that enables syncing (broadcasting) buffers of
                           the module at beginning of the forward function.
                           (default: True)
        process_group: the process group to be used for distributed data
                       all-reduction. If None, the default process group, which
                       is created by ```torch.distributed.init_process_group```,
                       will be used. (default: None)
        bucket_cap_mb: DistributedDataParallel will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       bucket_cap_mb controls the bucket size in MegaBytes (MB)
                       (default: 25)
        check_reduction: when setting to True, it enables DistributedDataParallel
                         to automatically check if the previous iteration's
                         backward reductions were successfully issued at the
                         beginning of every iteration's forward function.
                         You normally don't need this option enabled unless you
                         are observing weird behaviors such as different ranks
                         are getting different gradients, which should not
                         happen if DistributedDataParallel is corrected used.
                         (default: False)

    Attributes:
        module (Module): the module to be parallelized

    Example::
        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = torch.nn.DistributedDataParallel(model, pg)
    """
    def __init__(self, module, device_ids=None,
                 output_device=None, dim=0, broadcast_buffers=True,
                 process_group=None, bucket_cap_mb=25,
                 check_reduction=False):

        super(DistributedDataParallel, self).__init__()

        # Use all devices by default
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))

        if output_device is None:
            output_device = device_ids[0]

        if process_group is None:
            self.process_group = _get_default_group()
        else:
            self.process_group = process_group

        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.broadcast_buffers = broadcast_buffers
        self.check_reduction = check_reduction

        MB = 1024 * 1024

        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = 250 * MB

        # reduction bucket size
        self.bucket_bytes_cap = bucket_cap_mb * MB

        # Sync params and buffers
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._dist_broadcast_coalesced(module_states,
                                           self.broadcast_bucket_size)

        self._ddp_init_helper()

    def _ddp_init_helper(self):
        """
        Initialization helper function that does the following:

        (1) replicating the module from device[0] to the other devices
        (2) bucketing the parameters for reductions
        (3) resetting the bucketing states
        (4) registering the grad hooks
        """
        if len(self.device_ids) > 1:
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

        # This is a triply-nested list where the "dimensions" are: devices, buckets, bucket_elems
        param_buckets = []

        # Split the parameters into buckets and by types as well
        # We only need to bucket and reduce parameters that require grad and
        # this is also true for backward since only the backward hooks for
        # parameters that require grad will be registered with gradient
        # reduction functions
        params_to_bucket = [[] for _ in self._module_copies]
        for dev_idx, m in enumerate(self._module_copies):
            for p in m.parameters():
                if p.requires_grad:
                    params_to_bucket[dev_idx].append(p)

        param_buckets = [dist._dist_bucket_tensors(dev_params_to_bucket,
                                                   int(self.bucket_bytes_cap),
                                                   fine_grained=False)
                         for dev_params_to_bucket in params_to_bucket]

        self.bucket_sizes = []
        self.bucket_map = {}

        # We transpose param_buckets, so the loop is over buckets.
        # param_buckets_tuple is a doubly-nested list with "dims": devices, bucket_elems
        for bucket_idx, param_buckets_tuple in enumerate(zip(*param_buckets)):
            self.bucket_sizes.append(0)
            # Now, we transpose again, so we iterate over bucket_elems, but getting tuples
            # of params from each device.
            for param_tuple in zip(*param_buckets_tuple):
                if not param_tuple[0].requires_grad:
                    continue
                for p in param_tuple:
                    self.bucket_map[p] = (bucket_idx, self.bucket_sizes[bucket_idx])
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
        # When all buckets are reduced, this will be set to True. This flag is
        # useful for sanity checks to ensure that each iteration's backward has
        # always reduced all buckets
        self.all_buckets_reduced = False
        self.check_previous_reduction = False
        self.ready_buckets_not_reduced = set()
        self.reduction_works = [None for _ in range(len(self.bucket_sizes))]
        self.devs_ready = [0 for _ in range(len(self.bucket_sizes))]
        self._register_grad_hooks()

    def __getstate__(self):
        self._check_default_group()
        attrs = copy.copy(self.__dict__)
        del attrs['process_group'], \
            attrs['default_streams'], \
            attrs['_grad_accs']
        return attrs

    def __setstate__(self, state):
        # If serializable, then the process group should be the default one
        self.process_group = _get_default_group()
        self.check_previous_reduction = False
        super(DistributedDataParallel, self).__setstate__(state)
        self._ddp_init_helper()

    def _check_default_group(self):
        pickle_not_supported = False
        try:
            if self.process_group != _get_default_group():
                pickle_not_supported = True
        except RuntimeError:
            pickle_not_supported = True

        if pickle_not_supported:
            raise RuntimeError("DDP Pickling/Unpickling are only supported "
                               "when using DDP with the default process "
                               "group. That is, when you have called "
                               "init_process_group and have not passed "
                               "process_group argument to DDP constructor")

    def _check_previous_reduction(self):
        if not self.training:
            return
        # self.check_previous_reduction will be False in the first iteration
        # and is then toggled to True for all future iterations.
        if self.check_previous_reduction is False:
            self.check_previous_reduction = True
        else:
            if not self.all_buckets_reduced:
                raise RuntimeError("Not all gradients have been reduced from "
                                   "the backward of the previous iteration. "
                                   "This is unexpected and fatal error. Please "
                                   "check and ensure that the model's "
                                   "parameters are not changed after you wrap "
                                   "up the model with DistributedDataParallel.")
        self.all_buckets_reduced = False

    def forward(self, *inputs, **kwargs):
        if self.check_reduction:
            self._check_previous_reduction()
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
        self.check_previous_reduction = False
        super(DistributedDataParallel, self).train(mode)
        for module in self._module_copies[1:]:
            module.train(mode)

    def _dist_broadcast_coalesced(self, tensors, buffer_size):
        dist._dist_broadcast_coalesced(self.process_group, tensors, buffer_size, False)

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

        # default stream tracking to launch nccl reduce kernels
        self.default_streams = []
        for dev_id in self.device_ids:
            with torch.cuda.device(dev_id):
                self.default_streams.append(torch.cuda.current_stream())

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
                raise RuntimeError("DistributedDataParallel only works "
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
                    self.all_buckets_reduced = True

        return distributed_data_parallel_hook

    def _queue_reduction(self, bucket_idx):
        # _queue_reduction will use a seperate CUDA stream to coalesce
        # the small tensors to achieve more parallelisms, before passing the
        # coalesced tensor into the c10d CUDA stream for reduction
        result = dist._queue_reduction(self.process_group,
                                       self.buckets[bucket_idx],
                                       self.device_ids)
        self.reduction_works[bucket_idx] = result[0]
        self.buckets_coalesced[bucket_idx] = result[1]

    def _sync_reduction_works(self):
        # Now only work on the first GPU of self.device_ids
        # _sync_reduction will use a seperate CUDA stream to uncoalesce
        # the coalesced tensors to achieve more parallelisms
        for bucket_idx, grads_batch in enumerate(self.buckets):
            dist._sync_reduction(self.reduction_works[bucket_idx],
                                 grads_batch[0],
                                 self.buckets_coalesced[bucket_idx])

        # Reset the module states
        self.next_bucket = len(self.bucket_sizes) - 1
        self.ready_buckets_not_reduced = set()
        self.reduction_works = [None for _ in range(len(self.bucket_sizes))]
        self.devs_ready = [0 for _ in range(len(self.bucket_sizes))]

        self.buckets = [[[None for _ in range(self.bucket_sizes[i])]
                        for _ in range(len(self.device_ids))] for i in range(len(self.bucket_sizes))]
        self.buckets_coalesced = [[] for _ in range(len(self.bucket_sizes))]
        self.buckets_ready_size = [[0 for _ in range(len(self.device_ids))] for i in range(len(self.bucket_sizes))]
