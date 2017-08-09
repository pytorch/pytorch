import sys
import math
import threading

import torch
from torch.autograd import Variable
from torch._utils import _flatten_tensors, _unflatten_tensors
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
    """Implements distributed data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. The module is replicated on each machine and each device, and
    each such replica handles a portion of the input. During the backwards
    pass, gradients from each node are averaged.

    The batch size should be larger than the number of GPUs used locally. It
    should also be an integer multiple of the number of GPUs so that each chunk
    is the same size (so that each GPU processes the same number of samples).

    See also: :ref:`cuda-nn-dataparallel-instead`. The same constraints on input
    as in :class:`torch.nn.DataParallel` apply.

    Creation of this class requires the distributed package to be already
    initialized in the process group mode
    (see :func:`torch.distributed.init_process_group`).

    .. warning::
        This module works only with the ``gloo`` backend.

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
        This module doesn't work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast form the module in process of rank
        0, to all other replicas in the system in every iteration.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        >>> torch.distributed.init_process_group(world_size=4, init_method='...')
        >>> net = torch.nn.DistributedDataParallel(model)
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DistributedDataParallel, self).__init__()

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device

        # Sync params and buffers
        for p in self.module.state_dict().values():
            dist.broadcast(p, 0)

        if len(device_ids) > 1:
            # TODO: we don't need to replicate params in here. they're always going to
            # be broadcasted using larger blocks in broadcast_coalesce, so it might be
            # better to not pollute the caches with these small blocks
            self._module_copies = replicate(self.module, self.device_ids)
            self._module_copies[0] = self.module
            for module_copy in self._module_copies[1:]:
                for param, copy_param in zip(self.module.parameters(), module_copy.parameters()):
                    copy_param.detach_()
                    copy_param.requires_grad = param.requires_grad
        else:
            self._module_copies = [self.module]

        # Split parameters into buckets that will coalesce reductions
        # TODO: different types need different buckets
        t = None
        for p in self.module.parameters():
            tp = type(p.data)
            if t is not None and t is not tp:
                raise ValueError("DistributedDataParallel requires all parameters' data to be of the same type")
            t = tp

        self.bucket_sizes = []
        self.bucket_map = {}
        MB = 1024 * 1024
        self.broadcast_bucket_size = 10 * MB  # used for param sync before forward
        bucket_bytes_cap = 1 * MB
        bucket_bytes = bucket_bytes_cap  # to init the first bucket immediately
        for param_tuple in zip(*map(lambda m: m.parameters(), self._module_copies)):
            if bucket_bytes >= bucket_bytes_cap:
                self.bucket_sizes.append(0)
                bucket_bytes = 0
            self.bucket_sizes[-1] += 1
            for p in param_tuple:
                self.bucket_map[p] = len(self.bucket_sizes) - 1
            bucket_bytes += p.numel() * p.element_size()

        self.buckets = [[[] for _ in range(len(self.device_ids))] for _ in range(len(self.bucket_sizes))]
        self.bucket_events = [[None] * len(self.device_ids) for _ in range(len(self.bucket_sizes))]
        self.reduced = [False] * len(self.bucket_sizes)

        self._register_grad_hooks()

        self.dispatch_lock = threading.Lock()
        self._start_reduction_threads()

    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        del attrs['_grad_accs'], attrs['_reduction_queues'], attrs['_reduction_streams'], \
            attrs['_reduction_threads'], attrs['_nccl_streams'], attrs['_default_streams']

    def __setstate__(self, state):
        super(DistributedDataParallel, self).__setstate__(state)
        self._register_grad_hooks()
        self._start_reduction_threads()

    def forward(self, *inputs, **kwargs):
        if len(self.device_ids) == 1:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        self._sync_params()
        outputs = self.parallel_apply(self._module_copies, inputs, kwargs)
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

    def _sync_params(self):
        params = [p.data for p in self.module.parameters()]
        result = broadcast_coalesced(params, self.device_ids, self.broadcast_bucket_size)
        for tensors, module in zip(result[1:], self._module_copies[1:]):
            for tensor, param in zip(tensors, module.parameters()):
                param.data.set_(tensor)

        # cross-node buffer sync
        buffers = list(self.module._all_buffers())
        flat_buffers = _flatten_tensors(buffers)
        dist.broadcast(flat_buffers, 0)
        for buf, synced in zip(buffers, _unflatten_tensors(flat_buffers, buffers)):
            buf.copy_(synced)

        # intra-node buffer sync
        result = broadcast_coalesced(buffers, self.device_ids, self.broadcast_bucket_size)
        for tensors, module in zip(result[1:], self._module_copies[1:]):
            for tensor, buf in zip(tensors, module._all_buffers()):
                buf.set_(tensor)

    def _register_grad_hooks(self):
        self._grad_accs = []  # need to keep them in scope
        for device_idx, module in enumerate(self._module_copies):
            for p in module.parameters():
                # TODO: no-op for these that don't require grad
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(p, device_idx))
                self._grad_accs.append(grad_acc)

    def _make_param_hook(self, param, device_idx):
        bucket_idx = self.bucket_map[param]

        def distributed_data_parallel_hook(*unused):
            if not param.grad.volatile:
                raise RuntimeError("DistributedDataParallel only works with volatile gradients")
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
                    coalesced = _flatten_tensors(grad_batch)
                    dev_coalesced.append(coalesced)
            # Wait for all copies to complete before starting the NCCL kernel
            for stream in reduction_streams:
                stream.synchronize()
            nccl.reduce(dev_coalesced, root=device_ids[0], streams=nccl_streams)

            # From now on we're only going to work on the first device (from device_ids)
            grad_batch = dev_grad_batch[0]
            coalesced = dev_coalesced[0]
            reduce_stream = reduction_streams[0]
            with torch.cuda.stream(reduce_stream):
                reduce_stream.wait_stream(nccl_streams[0])
                coalesced /= dist.get_world_size()
                dist.all_reduce(coalesced, group=group_id)
                for grad, reduced in zip(grad_batch, _unflatten_tensors(coalesced, grad_batch)):
                    grad.copy_(reduced)
            job_event.set()

        with torch.cuda.device(device_ids[0]):
            while True:
                _process_batch()  # just to have a clear scope
