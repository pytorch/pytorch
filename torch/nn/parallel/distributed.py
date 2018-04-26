import sys
import math
import threading
import copy

import torch
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors

from torch.cuda.comm import broadcast_coalesced
from torch.cuda import nccl
import torch.distributed as dist

from ..modules import Module
from .replicate import replicate
from .scatter_gather import scatter_kwargs, gather
from .parallel_apply import parallel_apply

if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue


class DistributedDataParallel(Module):
    r"""Implements distributed data parallelism at the module level.

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

    Creation of this class requires the distributed package to be already
    initialized in the process group mode
    (see :func:`torch.distributed.init_process_group`).

    .. warning::
        This module works only with the ``nccl`` and ``gloo`` backends.

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
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])
        broadcast_buffers: flag that enables syncing (broadcasting) buffers of
                           the module at beginning of the forward function.
                           (default: True)

    Example::

        >>> torch.distributed.init_process_group(world_size=4, init_method='...')
        >>> net = torch.nn.DistributedDataParallel(model)
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0,
                 broadcast_buffers=True):
        super(DistributedDataParallel, self).__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.broadcast_buffers = broadcast_buffers

        # Flag used by the NCCL backend to make sure we only reduce gradients
        # one time in the execution engine
        self.need_reduction = False

        MB = 1024 * 1024
        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = 10 * MB
        self.nccl_reduce_bucket_size = 256 * MB

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

        # For NCCL backend, since every single NCCL call is asynchoronous, we
        # therefore directly enqueue all the NCCL reduction calls to the
        # default CUDA stream without spawning up other reduction threads.
        # This achieves the best performance.
        if dist._backend == dist.dist_backend.NCCL:
            self._register_nccl_grad_hook()
            return

        bucket_bytes_cap = 1 * MB

        # This is a triply-nested list where the "dimensions" are: devices, buckets, bucket_elems
        param_buckets = []
        # Split the parameters into buckets and by types as well
        for dev_idx, module in enumerate(self._module_copies):
            param_buckets.append(list(_take_tensors(module.parameters(), bucket_bytes_cap)))

        self.bucket_sizes = []
        self.bucket_map = {}

        # We transpose param_buckets, so the loop is over buckets.
        # param_buckets_tuple is a doubly-nested list with "dims": devices, bucket_elems
        for bucket_idx, param_buckets_tuple in enumerate(zip(*param_buckets)):
            self.bucket_sizes.append(0)
            # Now, we transpose again, so we iterate over bucket_elems, but getting tuples
            # of params from each device.
            for idx, param_tuple in enumerate(zip(*param_buckets_tuple)):
                if idx == 0:
                    # Bucket parameter type tracking
                    bucket_param_type = param_tuple[0].type()
                    # Only gloo and nccl support half-precision
                    if bucket_param_type == torch.cuda.HalfTensor and \
                            dist._backend != dist.dist_backend.GLOO:
                        raise RuntimeError("DistributedDataParallel currently only "
                                           "supports half precision parameters "
                                           "with Nccl and Gloo backend")
                if not param_tuple[0].requires_grad:
                    continue
                for p in param_tuple:
                    self.bucket_map[p] = bucket_idx
                self.bucket_sizes[bucket_idx] += 1

        self.buckets = [[[] for _ in range(len(self.device_ids))] for _ in range(len(self.bucket_sizes))]
        self.bucket_events = [[None] * len(self.device_ids) for _ in range(len(self.bucket_sizes))]
        self.reduced = [False] * len(self.bucket_sizes)

        self._register_grad_hooks()

        self.dispatch_lock = threading.Lock()
        self._start_reduction_threads()

    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        if dist._backend != dist.dist_backend.NCCL:
            del attrs['_grad_accs'], attrs['_reduction_queues'], \
                attrs['_reduction_streams'], attrs['_reduction_threads'], \
                attrs['_nccl_streams'], attrs['_default_streams']
        return attrs

    def __setstate__(self, state):
        super(DistributedDataParallel, self).__setstate__(state)
        if dist._backend == dist.dist_backend.NCCL:
            self._register_nccl_grad_hook()
        else:
            self._register_grad_hooks()
            self._start_reduction_threads()

    def forward(self, *inputs, **kwargs):
        self.need_reduction = True
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
        super(DistributedDataParallel, self).train(mode)
        for module in self._module_copies[1:]:
            module.train(mode)

    def _dist_broadcast_coalesced(self, tensors, buffer_size):
        """
        Broadcast a sequence of tensors to the default group from rank 0.
        Small tensors are first coalesced into a buffer to reduce the number of
        broadcasts.

        tensors (sequence): tensors to broadcast. Each tensor needs to be on the
                            same GPU.
        buffer_size (int): maximum size of the buffer for coalescing
        """
        for tensors in _take_tensors(tensors, buffer_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.broadcast(flat_tensors, 0)
            for tensor, synced in zip(tensors,
                                      _unflatten_dense_tensors(flat_tensors, tensors)):
                tensor.copy_(synced)

    def _sync_params(self):
        if len(self.device_ids) > 1:
            # intra-node parameter sync
            params = [p.data for p in self.module.parameters()]
            result = broadcast_coalesced(params, self.device_ids, self.broadcast_bucket_size)
            for tensors, module in zip(result[1:], self._module_copies[1:]):
                for tensor, param in zip(tensors, module.parameters()):
                    param.data.set_(tensor)

        # module buffer sync
        if self.broadcast_buffers:
            buffers = list(self.module._all_buffers())
            if len(buffers) > 0:
                # cross-node buffer sync
                self._dist_broadcast_coalesced(buffers, self.broadcast_bucket_size)

                if len(self.device_ids) > 1:
                    # intra-node buffer sync
                    result = broadcast_coalesced(buffers, self.device_ids, self.broadcast_bucket_size)
                    for tensors, module in zip(result[1:], self._module_copies[1:]):
                        for tensor, buf in zip(tensors, module._all_buffers()):
                            buf.set_(tensor)

    def _register_grad_hooks(self):
        self._grad_accs = []  # need to keep them in scope
        for device_idx, module in enumerate(self._module_copies):
            for p in module.parameters():
                if p.requires_grad:
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(p, device_idx))
                    self._grad_accs.append(grad_acc)

    def _register_nccl_grad_hook(self):
        """
        This function registers the callback all-reduction function for the
        NCCL backend. All gradients will be all reduced in one single step.
        The NCCL reduction will directly be enqueued into the
        default CUDA stream. Therefore, no synchronization is needed.
        """
        # Creating a new group
        self.nccl_reduction_group_id = dist.new_group()

        def reduction_fn_nccl():
            # This function only needs to be called once
            if not self.need_reduction:
                return

            self.need_reduction = False
            all_grads = [[] for _ in range(len(self._module_copies))]
            all_grads_buckets_iters = []

            # Bucketing all the gradients
            for dev_idx, module in enumerate(self._module_copies):
                for param in module.parameters():
                    if not param.requires_grad or param.grad is None:
                        continue
                    if param.grad.requires_grad:
                        raise RuntimeError("DistributedDataParallel only works "
                                           "with gradients that don't require "
                                           "grad")
                    # Adding the gradients for reduction
                    all_grads[dev_idx].append(param.grad.data)

                # Now bucketing the parameters
                dev_grads_buckets = _take_tensors(all_grads[dev_idx],
                                                  self.nccl_reduce_bucket_size)

                all_grads_buckets_iters.append(dev_grads_buckets)

            # Now reduce each bucket one after another
            for grads_batch in zip(*all_grads_buckets_iters):
                grads_batch_coalesced = []
                # Coalesce each bucket
                for dev_idx, dev_grads_batch in enumerate(grads_batch):
                    dev_id = self.device_ids[dev_idx]
                    with torch.cuda.device(dev_id):
                        dev_grads_batch_coalesced = _flatten_dense_tensors(dev_grads_batch)
                        grads_batch_coalesced.append(dev_grads_batch_coalesced)

                # We will only use device 0's results, but this single op should be
                # faster than doing the following two operation sequentially:
                # (1) intra-node reduce to lead GPU, followed by
                # (2) inter-node allreduce for all the first lead GPUs in all nodes
                dist.all_reduce_multigpu(grads_batch_coalesced,
                                         group=self.nccl_reduction_group_id)

                # Now only work on the first device of self.device_ids, uncoalesce
                # the gradients for each bucket
                grads_batch_coalesced[0] /= dist.get_world_size()
                grads_batch_reduced = _unflatten_dense_tensors(grads_batch_coalesced[0], grads_batch[0])
                for grad, reduced in zip(grads_batch[0], grads_batch_reduced):
                    grad.copy_(reduced)

            # clear the gradients and save memory for replicas
            for module in self._module_copies[1:]:
                for param in module.parameters():
                    if param.requires_grad:
                        param.grad = None
                        param.data.set_()

        # Now register the reduction hook on the parameters
        for p in self.module.parameters():
            if not p.requires_grad:
                continue

            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(reduction_fn_nccl)

            p.register_hook(allreduce_hook)

    def _make_param_hook(self, param, device_idx):

        bucket_idx = self.bucket_map[param]

        def distributed_data_parallel_hook(*unused):
            if param.grad.requires_grad:
                raise RuntimeError("DistributedDataParallel only works with "
                                   "gradients that don't require grad")
            bucket = self.buckets[bucket_idx][device_idx]
            bucket.append(param.grad.data)

            # We can flush these and save memory for replicas
            if device_idx > 0:
                param.grad = None
                param.data.set_()

            # Current device's bucket is full
            if len(bucket) == self.bucket_sizes[bucket_idx]:
                with torch.cuda.device(self.device_ids[device_idx]):
                    event = torch.cuda.Event()
                    event.record()
                with self.dispatch_lock:
                    self.bucket_events[bucket_idx][device_idx] = event
                    self._queue_reduction(bucket_idx)

        return distributed_data_parallel_hook

    def _queue_reduction(self, bucket_idx):
        dev_buckets = self.buckets[bucket_idx]
        dev_events = self.bucket_events[bucket_idx]

        # Check if it's ready
        if any(evt is None for evt in dev_events):
            return

        # Queue the reduction and make sure backward waits for it
        event = threading.Event()
        self._reduction_queues[bucket_idx].put((dev_buckets, dev_events, event))
        Variable._execution_engine.queue_callback(lambda: event.wait())

        # Reset bucket state
        self.buckets[bucket_idx] = [[] for _ in range(len(self.device_ids))]
        self.bucket_events[bucket_idx] = [None] * len(self.device_ids)
        self.reduced[bucket_idx] = True
        if all(self.reduced):
            self.reduced = [False] * len(self.bucket_sizes)

            def sync_reduction_streams():
                # We only have to sync with the first one, but it's safer to do it this way
                # in case we change the way in which we paralellize work
                r_streams = zip(*self._reduction_streams)
                for dev_id, default_stream, dev_r_streams in zip(self.device_ids, self._default_streams, r_streams):
                    with torch.cuda.device(dev_id):
                        for reduction_stream in dev_r_streams:
                            default_stream.wait_stream(reduction_stream)
            Variable._execution_engine.queue_callback(sync_reduction_streams)

    def _start_reduction_threads(self):
        num_buckets = len(self.bucket_sizes)
        self._reduction_queues = [queue.Queue() for _ in range(num_buckets)]
        self._reduction_threads = []
        self._reduction_streams = [[] for _ in range(num_buckets)]
        self._nccl_streams = []
        self._default_streams = []
        for dev_id in self.device_ids:
            with torch.cuda.device(dev_id):
                # TODO: don't assume we're on a default stream
                self._default_streams.append(torch.cuda.current_stream())
                self._nccl_streams.append(torch.cuda.Stream())
        for reduction_queue, reduction_streams in zip(self._reduction_queues, self._reduction_streams):
            for dev_id in self.device_ids:
                with torch.cuda.device(dev_id):
                    reduction_streams.append(torch.cuda.Stream())
            # We only use the first device for distributed reductions
            dist._register_stream(reduction_streams[0])

            group_id = dist.new_group()

            self._reduction_threads.append(threading.Thread(
                target=self._reduction_thread_fn,
                args=(reduction_queue, group_id, self.device_ids, reduction_streams, self._nccl_streams)))
            self._reduction_threads[-1].daemon = True
            self._reduction_threads[-1].start()

    @staticmethod
    def _reduction_thread_fn(queue, group_id, device_ids, reduction_streams, nccl_streams):

        def _process_batch():
            dev_grad_batch, dev_events, job_event = queue.get()
            dev_coalesced = []
            # Coalesce the tensors on all devices and start a local reduction
            for dev_id, grad_batch, event, stream in zip(device_ids, dev_grad_batch, dev_events, reduction_streams):
                with torch.cuda.device(dev_id), torch.cuda.stream(stream):
                    stream.wait_event(event)
                    coalesced = _flatten_dense_tensors(grad_batch)
                    dev_coalesced.append(coalesced)
            # Wait for all copies to complete before starting the NCCL kernel
            for stream in reduction_streams:
                stream.synchronize()
            nccl.reduce(dev_coalesced, root=0, streams=nccl_streams)

            # From now on we're only going to work on the first device (from device_ids)
            grad_batch = dev_grad_batch[0]
            coalesced = dev_coalesced[0]
            reduce_stream = reduction_streams[0]
            with torch.cuda.stream(reduce_stream):
                reduce_stream.wait_stream(nccl_streams[0])
                coalesced /= dist.get_world_size()
                dist.all_reduce(coalesced, group=group_id)
                for grad, reduced in zip(grad_batch, _unflatten_dense_tensors(coalesced, grad_batch)):
                    grad.copy_(reduced)
            job_event.set()

        with torch.cuda.device(device_ids[0]):
            while True:
                _process_batch()  # just to have a clear scope
