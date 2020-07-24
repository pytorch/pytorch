from contextlib import contextmanager
import copy
import itertools
import os
import inspect

import torch

from . import comm
import torch.distributed as dist

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group

from ..modules import Module
from .replicate import replicate
from .scatter_gather import scatter_kwargs, gather
from .parallel_apply import parallel_apply
from torch._utils import _get_device_index, _get_all_device_indices


def _find_tensors(obj):
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []

def _dump_DDP_relevant_env_vars():
    relevant_env_vars = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_PORT",
        "MASTER_ADDR",
        "CUDA_VISIBLE_DEVICES",
        "GLOO_SOCKET_IFNAME",
        "GLOO_DEVICE_TRANSPORT",
        "NCCL_SOCKET_IFNAME",
        "NCCL_BLOCKING_WAIT",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_IB_DISABLE",
        # More NCCL env vars:
        "NCCL_P2P_DISABLE",
        "NCCL_P2P_LEVEL",
        "NCCL_SHM_DISABLE",
        "NCCL_SOCKET_NTHREADS",
        "NCCL_NSOCKS_PERTHREAD",
        "NCCL_BUFFSIZE",
        "NCCL_NTHREADS",
        "NCCL_RINGS",
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_CHECKS_DISABLE",
        "NCCL_CHECK_POINTERS",
        "NCCL_LAUNCH_MODE",
        "NCCL_IB_HCA",
        "NCCL_IB_TIMEOUT",
        "NCCL_IB_RETRY_CNT",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_SL",
        "NCCL_IB_TC",
        "NCCL_IB_AR_THRESHOLD",
        "NCCL_IB_CUDA_SUPPORT",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_NET_GDR_READ",
        "NCCL_SINGLE_RING_THRESHOLD",
        "NCCL_LL_THRESHOLD",
        "NCCL_TREE_THRESHOLD",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_IGNORE_CPU_AFFINITY",
        "NCCL_DEBUG_FILE",
        "NCCL_COLLNET_ENABLE",
        "NCCL_TOPO_FILE",
        "NCCL_TOPO_DUMP_FILE",
    ]
    formatted_output = ""
    for var in relevant_env_vars:
        value = os.environ[var] if var in os.environ else "N/A"
        formatted_output += "env:%s=%s\n" % (var, value)
    print(formatted_output)


class DistributedDataParallel(Module):
    r"""Implements distributed data parallelism that is based on
    ``torch.distributed`` package at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. The module is replicated on each machine and each device, and
    each such replica handles a portion of the input. During the backwards
    pass, gradients from each node are averaged.

    The batch size should be larger than the number of GPUs used locally.

    See also: :ref:`distributed-basics` and :ref:`cuda-nn-ddp-instead`.
    The same constraints on input as in :class:`torch.nn.DataParallel` apply.

    Creation of this class requires that ``torch.distributed`` to be already
    initialized, by calling :func:`torch.distributed.init_process_group`.

    ``DistributedDataParallel`` is proven to be significantly faster than
    :class:`torch.nn.DataParallel` for single-node multi-GPU data
    parallel training.

    Here is how to use it: on each host with N GPUs, you should spawn up N
    processes, while ensuring that each process individually works on a single GPU
    from 0 to N-1. Therefore, it is your job to ensure that your training script
    operates on a single given GPU by calling:

        >>> torch.cuda.set_device(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``

    .. note ::
        Please refer to `PyTorch Distributed Overview <https://pytorch.org/tutorials/beginner/dist_overview.html>`__
        for a brief introduction to all features related to distributed training.

    .. note:: ``nccl`` backend is currently the fastest and
        highly recommended backend to be used with Multi-Process Single-GPU
        distributed training and this applies to both single-node and multi-node
        distributed training

    .. note:: This module also supports mixed-precision distributed training.
        This means that your model can have different types of parameters such
        as mixed types of fp16 and fp32, the gradient reduction on these
        mixed types of parameters will just work fine.
        Also note that ``nccl`` backend is currently the fastest and highly
        recommended backend for fp16/fp32 mixed-precision training.

    .. note:: If you use ``torch.save`` on one process to checkpoint the module,
        and ``torch.load`` on some other processes to recover it, make sure that
        ``map_location`` is configured properly for every process. Without
        ``map_location``, ``torch.load`` would recover the module to devices
        where the module was saved from.

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
        registered parameters of the model. In other words, it is users'
        responsibility to ensure that each distributed process has the exact
        same model and thus the exact same parameter registration order.

    .. warning::
        This module allows parameters with non-rowmajor-contiguous strides.
        For example, your model may contain some parameters whose
        :class:`torch.memory_format` is ``torch.contiguous_format``
        and others whose format is ``torch.channels_last``.  However,
        corresponding parameters in different processes must have the
        same strides.

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

    .. note::
        If you are using DistributedDataParallel in conjunction with the
        :ref:`distributed-rpc-framework`, you should always use
        :meth:`torch.distributed.autograd.backward` to compute gradients and
        :class:`torch.distributed.optim.DistributedOptimizer` for optimizing
        parameters.

    Example::
        >>> import torch.distributed.autograd as dist_autograd
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> from torch import optim
        >>> from torch.distributed.optim import DistributedOptimizer
        >>> from torch.distributed.rpc import RRef
        >>>
        >>> t1 = torch.rand((3, 3), requires_grad=True)
        >>> t2 = torch.rand((3, 3), requires_grad=True)
        >>> rref = rpc.remote("worker1", torch.add, args=(t1, t2))
        >>> ddp_model = DDP(my_model)
        >>>
        >>> # Setup optimizer
        >>> optimizer_params = [rref]
        >>> for param in ddp_model.parameters():
        >>>     optimizer_params.append(RRef(param))
        >>>
        >>> dist_optim = DistributedOptimizer(
        >>>     optim.SGD,
        >>>     optimizer_params,
        >>>     lr=0.05,
        >>> )
        >>>
        >>> with dist_autograd.context() as context_id:
        >>>     pred = ddp_model(rref.to_here())
        >>>     loss = loss_func(pred, loss)
        >>>     dist_autograd.backward(context_id, loss)
        >>>     dist_optim.step()

    .. warning::
        Using DistributedDataParallel in conjuction with the
        :ref:`distributed-rpc-framework` is experimental and subject to change.

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices. This should
                   only be provided when the input module resides on a single
                   CUDA device. For single-device modules, the ``i``th
                   :attr:`module` replica is placed on ``device_ids[i]``. For
                   multi-device modules and CPU modules, device_ids must be None
                   or an empty list, and input data for the forward pass must be
                   placed on the correct device. (default: all devices for
                   single-device modules)
        output_device (int or torch.device): device location of output for
                      single-device CUDA modules. For multi-device modules and
                      CPU modules, it must be None, and the module itself
                      dictates the output location. (default: device_ids[0] for
                      single-device modules)
        broadcast_buffers (bool): flag that enables syncing (broadcasting) buffers of
                          the module at beginning of the forward function.
                          (default: ``True``)
        process_group: the process group to be used for distributed data
                       all-reduction. If ``None``, the default process group, which
                       is created by ```torch.distributed.init_process_group```,
                       will be used. (default: ``None``)
        bucket_cap_mb: DistributedDataParallel will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       :attr:`bucket_cap_mb` controls the bucket size in MegaBytes (MB)
                       (default: 25)
        find_unused_parameters (bool): Traverse the autograd graph of all tensors
                                       contained in the return value of the wrapped
                                       module's ``forward`` function.
                                       Parameters that don't receive gradients as
                                       part of this graph are preemptively marked
                                       as being ready to be reduced. Note that all
                                       ``forward`` outputs that are derived from
                                       module parameters must participate in
                                       calculating loss and later the gradient
                                       computation. If they don't, this wrapper will
                                       hang waiting for autograd to produce gradients
                                       for those parameters. Any outputs derived from
                                       module parameters that are otherwise unused can
                                       be detached from the autograd graph using
                                       ``torch.Tensor.detach``. (default: ``False``)
        check_reduction: when setting to ``True``, it enables DistributedDataParallel
                         to automatically check if the previous iteration's
                         backward reductions were successfully issued at the
                         beginning of every iteration's forward function.
                         You normally don't need this option enabled unless you
                         are observing weird behaviors such as different ranks
                         are getting different gradients, which should not
                         happen if DistributedDataParallel is correctly used.
                         (default: ``False``)

    Attributes:
        module (Module): the module to be parallelized

    Example::

        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = torch.nn.DistributedDataParallel(model, pg)
    """
    def __init__(self, module, device_ids=None,
                 output_device=None, dim=0, broadcast_buffers=True,
                 process_group=None,
                 bucket_cap_mb=25,
                 find_unused_parameters=False,
                 check_reduction=False):

        super(DistributedDataParallel, self).__init__()

        assert any((p.requires_grad for p in module.parameters())), (
            "DistributedDataParallel is not needed when a module "
            "doesn't have any parameter that requires a gradient."
        )

        self.is_multi_device_module = len({p.device for p in module.parameters()}) > 1
        distinct_device_types = {p.device.type for p in module.parameters()}
        assert len(distinct_device_types) == 1, (
            "DistributedDataParallel's input module must be on "
            "the same type of devices, but input module parameters locate in {}."
        ).format(distinct_device_types)
        self.device_type = list(distinct_device_types)[0]

        if self.device_type == "cpu" or self.is_multi_device_module:
            assert not device_ids and not output_device, (
                "DistributedDataParallel device_ids and output_device arguments "
                "only work with single-device GPU modules, but got "
                "device_ids {}, output_device {}, and module parameters {}."
            ).format(device_ids, output_device, {p.device for p in module.parameters()})

            self.device_ids = None
            self.output_device = None
        else:
            # Use all devices by default for single-device GPU modules
            if device_ids is None:
                device_ids = _get_all_device_indices()

            self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))

            if output_device is None:
                output_device = device_ids[0]

            self.output_device = _get_device_index(output_device, True)

        if process_group is None:
            self.process_group = _get_default_group()
        else:
            self.process_group = process_group

        self.dim = dim
        self.module = module
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True

        if check_reduction:
            # This argument is no longer used since the reducer
            # will ensure reduction completes even if some parameters
            # do not receive gradients.
            pass

        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = int(250 * 1024 * 1024)

        # reduction bucket size
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

        # Sync params and buffers
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._distributed_broadcast_coalesced(
                module_states,
                self.broadcast_bucket_size)

        self._ddp_init_helper()

    def _ddp_init_helper(self):
        """
        Initialization helper function that does the following:

        (1) replicating the module from device[0] to the other devices
        (2) bucketing the parameters for reductions
        (3) resetting the bucketing states
        (4) registering the grad hooks
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """

        def parameters(m, recurse=True):
            def model_parameters(m):
                ps = m._former_parameters.values() \
                    if hasattr(m, "_former_parameters") \
                    else m.parameters(recurse=False)
                for p in ps:
                    yield p

            for m in m.modules() if recurse else [m]:
                for p in model_parameters(m):
                    yield p

        if self.device_ids and len(self.device_ids) > 1:

            import warnings
            warnings.warn(
                "Single-Process Multi-GPU is not the recommended mode for "
                "DDP. In this mode, each DDP instance operates on multiple "
                "devices and creates multiple module replicas within one "
                "process. The overhead of scatter/gather and GIL contention "
                "in every forward pass can slow down training. "
                "Please consider using one DDP instance per device or per "
                "module replica by explicitly setting device_ids or "
                "CUDA_VISIBLE_DEVICES. "
            )

            # only create replicas for single-device CUDA modules
            #
            # TODO: we don't need to replicate params in here. they're always going to
            # be broadcasted using larger blocks in broadcast_coalesced, so it might be
            # better to not pollute the caches with these small blocks
            self._module_copies = replicate(self.module, self.device_ids, detach=True)
            self._module_copies[0] = self.module

            for module_copy in self._module_copies[1:]:
                for param, copy_param in zip(self.module.parameters(), parameters(module_copy)):
                    # Reducer requires param copies have the same strides across replicas.
                    # Fixes up copy_param strides in case replicate didn't match param strides.
                    if param.layout is torch.strided and param.stride() != copy_param.stride():
                        with torch.no_grad():
                            copy_param.set_(copy_param.clone()
                                                      .as_strided(param.size(), param.stride())
                                                      .copy_(copy_param))
                    copy_param.requires_grad = param.requires_grad

        else:
            self._module_copies = [self.module]

        self.modules_params = [list(parameters(m)) for m in self._module_copies]
        self.modules_buffers = [list(m.buffers()) for m in self._module_copies]

        # Build tuple of (module, parameter) for all parameters that require grads.
        modules_and_parameters = [
            [
                (module, parameter)
                for module in replica.modules()
                for parameter in filter(
                    lambda parameter: parameter.requires_grad,
                    parameters(module, recurse=False))
            ] for replica in self._module_copies]

        # Build list of parameters.
        parameters = [
            list(parameter for _, parameter in replica)
            for replica in modules_and_parameters]

        # Checks if a module will produce a sparse gradient.
        def produces_sparse_gradient(module):
            if isinstance(module, torch.nn.Embedding):
                return module.sparse
            if isinstance(module, torch.nn.EmbeddingBag):
                return module.sparse
            return False

        # Build list of booleans indicating whether or not to expect sparse
        # gradients for the corresponding parameters.
        expect_sparse_gradient = [
            list(produces_sparse_gradient(module) for module, _ in replica)
            for replica in modules_and_parameters]

        # The bucket size limit is specified in the constructor.
        # Additionally, we allow for a single small bucket for parameters
        # that are defined first, such that their gradients don't spill into
        # a much larger bucket, adding unnecessary latency after gradient
        # computation finishes. Experiments showed 1MB is a reasonable value.
        bucket_indices = dist._compute_bucket_assignment_by_size(
            parameters[0],
            [dist._DEFAULT_FIRST_BUCKET_BYTES, self.bucket_bytes_cap],
            expect_sparse_gradient[0])

        # Note: reverse list of buckets because we want to approximate the
        # order in which their gradients are produced, and assume they
        # are used in the forward pass in the order they are defined.
        self.reducer = dist.Reducer(
            parameters,
            list(reversed(bucket_indices)),
            self.process_group,
            expect_sparse_gradient,
            self.bucket_bytes_cap,
            self.find_unused_parameters)

        # passing a handle to torch.nn.SyncBatchNorm layer
        self._passing_sync_batchnorm_handle(self._module_copies)

    def __getstate__(self):
        self._check_default_group()
        attrs = copy.copy(self.__dict__)
        del attrs['process_group']
        del attrs['reducer']
        return attrs

    def __setstate__(self, state):
        # If serializable, then the process group should be the default one
        self.process_group = _get_default_group()
        super(DistributedDataParallel, self).__setstate__(state)
        self.__dict__.setdefault('require_forward_param_sync', True)
        self.__dict__.setdefault('require_backward_grad_sync', True)
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

    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> ddp = torch.nn.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            ...   for input in inputs:
            ...     ddp(input).backward()  # no synchronization, accumulate grads
            ... ddp(another_input).backward()  # synchronize grads
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync

    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_params()

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module(*inputs, **kwargs)

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

        return output

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

    def _register_comm_hook(self, state: object, hook: callable):
        r"""
        Register a communication hook which is an enhancement that provides a
        flexible hook to users where they can specify how DDP aggregates gradients
        across multiple workers.

        This hook would be very useful for researchers to try out new ideas. For
        example, this hook can be used to implement several algorithms like GossipGrad
        and gradient compression which involve different communication strategies for
        parameter syncs while running Distributed DataParallel training.

        Arguments:
            state (object): state is passed to the hook and can be used to maintain
                            and update any state information that users would like to
                            maintain as part of the training process. Examples: error
                            feedback in gradient compression, peers to communicate with
                            next in GossipGrad etc.
            hook (callable): is defined as:
                             hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future:

                             This function is called once the bucket is ready. The
                             hook can perform whatever processing is needed and return
                             a Future indicating completion of any async work (ex: allreduce).
                             If the hook doesn't perform any communication, it can also
                             just return a completed Future. The Future should hold the
                             new value of grad bucket's tensors. Once a bucket is ready,
                             c10d reducer would call this hook and use the tensors returned
                             by the Future and copy grads to individual parameters.

                             We also provide an API called "get_future" to retrieve a future
                             associated with the completion of c10d.ProcessGroupNCCL.work.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        .. warning ::
            The torch.futures.Future object that hook returns should contain a result that
            has the same shape with the tensors inside GradBucket bucket.

        .. warning ::
            DDP communication hook is experimental and subject to change.

        Example::
            Below is an example of a noop hook that returns back the same tensors:

            >>> ddp._register_comm_hook(state = None, hook = noop)

            >>> def noop(state: object, bucket: dist.GradBucket): -> torch.futures.Future
            >>>     fut = torch.futures.Future()
            >>>     fut.set_result(bucket.get_tensors())
            >>>     return fut

        Example::
            Below is an example of a simple allreduce hook.

            >>> ddp._register_comm_hook(state = None, hook = allreduce)

            >>> def allreduce(state: object, bucket: dist.GradBucket): -> torch.futures.Future
            >>>     work = dist.allreduce(bucket.get_tensors())
            >>>     return work.get_future()

        Example::
            Below is an example of a Parallel SGD algorithm where gradients are encoded before
            allreduce, and then decoded after allreduce.

            >>> ddp._register_comm_hook(state = None, hook = encode_and_decode)

            >>> def encode_and_decode(state: object, bucket: dist.GradBucket): -> torch.futures.Future
            >>>     encoded_tensors = encode(bucket.get_tensors()) # encode gradients
            >>>     fut = process_group.allreduce(encoded_tensors).get_future()
            >>>     # Define the then callback to decode.
            >>>     def decode(fut):
            >>>         decoded_tensors = decode(fut.wait()) # decode gradients
            >>>         return decoded_tensors
            >>>     return fut.then(decode)

        """
        self._check_comm_hook(hook)
        dist._register_comm_hook(self.reducer, state, hook)

    def _distributed_broadcast_coalesced(self, tensors, buffer_size):
        dist._broadcast_coalesced(self.process_group, tensors, buffer_size)

    def _sync_params(self):
        with torch.no_grad():
            # only do intra-node parameters sync for replicated single-device
            # CUDA modules
            if self.device_ids and len(self.device_ids) > 1:
                # intra-node parameter sync
                result = comm.broadcast_coalesced(
                    self.modules_params[0],
                    self.device_ids,
                    self.broadcast_bucket_size)
                for tensors, module_params in zip(result[1:],
                                                  self.modules_params[1:]):
                    for tensor, param in zip(tensors, module_params):
                        # Formerly, this spot used param.set_(tensor) to steal tensor's
                        # data without a deep copy.  Unfortunately, that wiped out the
                        # allreduce hook attached to param's AccumulateGrad function,
                        # likely causing https://github.com/pytorch/pytorch/issues/37079.
                        # TODO:  If set_ becomes safe to use here, use set_.
                        # Otherwise, find another way to steal tensor's data.
                        param.copy_(tensor)
                        # Assume we have just run the optimizer and zeroed the
                        # grads of the parameters on the root model. We need
                        # to zero the grads on all model replicas as well.
                        # This snippet is copied from torch.optim.Optimizer.
                        if param.grad is not None:
                            param.grad.detach_()
                            param.grad.zero_()

            # module buffer sync
            if self.broadcast_buffers and len(self.modules_buffers[0]) > 0:
                # Synchronize buffers across processes.
                # The process with rank 0 is considered the authoritative copy.
                self._distributed_broadcast_coalesced(
                    self.modules_buffers[0],
                    self.broadcast_bucket_size)
                # only do intra-node buffer sync for replicated single-device
                # CUDA modules
                if self.device_ids and len(self.device_ids) > 1:
                    # intra-node buffer sync
                    result = comm.broadcast_coalesced(
                        self.modules_buffers[0],
                        self.device_ids,
                        self.broadcast_bucket_size)
                    for tensors, module_buffers in zip(result[1:],
                                                       self.modules_buffers[1:]):
                        for tensor, buffer in zip(tensors, module_buffers):
                            buffer.set_(tensor)

    def _passing_sync_batchnorm_handle(self, module_copies):
        for dev_idx, module in enumerate(module_copies):
            for layer in module.modules():
                if isinstance(layer, torch.nn.modules.SyncBatchNorm):
                    assert self.device_type != 'cpu', "SyncBatchNorm layers only work with GPU modules"
                    layer._specify_ddp_gpu_num(
                        len(self.device_ids) if self.device_ids else 1)

    def _check_comm_hook(self, hook):
        if not callable(hook):
            raise TypeError("Communication hook must be callable.")

        sig = inspect.signature(hook)
        if (sig.parameters['bucket'].annotation != inspect._empty and
                sig.parameters['bucket'].annotation != dist.GradBucket):
            raise ValueError("Communication hook: bucket annotation is not dist.GradBucket.")

        if (sig.return_annotation != inspect._empty and
                sig.return_annotation != torch.futures.Future):
            raise ValueError("Communication hook: return annotation is not torch.futures.Future.")
