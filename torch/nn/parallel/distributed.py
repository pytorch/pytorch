from contextlib import contextmanager
import copy
import itertools
import os
import inspect
import logging
import warnings
from typing import NamedTuple

import torch

from . import comm
import torch.distributed as dist

RPC_AVAILABLE = False
if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group
    from torch.distributed.distributed_c10d import ReduceOp
if torch.distributed.rpc.is_available():
    RPC_AVAILABLE = True
    from torch.distributed.rpc import RRef
from ..modules import Module
from .replicate import replicate
from .scatter_gather import scatter_kwargs, gather, is_namedtuple
from .parallel_apply import parallel_apply
from torch._utils import _get_device_index, _get_all_device_indices
from ._functions import _get_stream


def _find_tensors(obj):
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if RPC_AVAILABLE and isinstance(obj, RRef):
        # If the current node is the owner of the RRef, unwrap it and try to
        # find Tensors.
        # TODO: Expand to remote RRefs.
        if obj.is_owner():
            return _find_tensors(obj.local_value())
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



class _DDPUnevenInputsConfig(NamedTuple):
    ddp_join_enabled: bool
    ddp_join_divide_by_initial_world_size: bool


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

    To use ``DistributedDataParallel`` on a host with N GPUs, you should spawn
    up ``N`` processes, ensuring that each process exclusively works on a single
    GPU from 0 to N-1. This can be done by either setting
    ``CUDA_VISIBLE_DEVICES`` for every process or by calling:

        >>> torch.cuda.set_device(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> torch.distributed.init_process_group(
        >>>     backend='nccl', world_size=N, init_method='...'
        >>> )
        >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``.

    .. note ::
        Please refer to `PyTorch Distributed Overview <https://pytorch.org/tutorials/beginner/dist_overview.html>`__
        for a brief introduction to all features related to distributed training.

    .. note:: ``nccl`` backend is currently the fastest and highly recommended
        backend when using GPUs. This applies to both single-node and
        multi-node distributed training.

    .. note:: This module also supports mixed-precision distributed training.
        This means that your model can have different types of parameters such
        as mixed types of ``fp16`` and ``fp32``, the gradient reduction on these
        mixed types of parameters will just work fine.

    .. note:: If you use ``torch.save`` on one process to checkpoint the module,
        and ``torch.load`` on some other processes to recover it, make sure that
        ``map_location`` is configured properly for every process. Without
        ``map_location``, ``torch.load`` would recover the module to devices
        where the module was saved from.

    .. note:: When a model is trained on ``M`` nodes with ``batch=N``, the
        gradient will be ``M`` times smaller when compared to the same model
        trained on a single node with ``batch=M*N`` if the loss is summed (NOT
        averaged as usual) across instances in a batch (because the gradients
        between different nodes are averaged). You should take this into
        consideration when you want to obtain a mathematically equivalent
        training process compared to the local training counterpart. But in most
        cases, you can just treat a DistributedDataParallel wrapped model, a
        DataParallel wrapped model and an ordinary model on a single GPU as the
        same (E.g. using the same learning rate for equivalent batch size).

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
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) are distributed synchronization
        points. Take that into account in case different processes might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.
        Same applies to buffers.

    .. warning::
        This module assumes all parameters are registered in the model of each
        distributed processes are in the same order. The module itself will
        conduct gradient ``allreduce`` following the reverse order of the
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
        up your model with ``DistributedDataParallel``. Because, when
        wrapping up your model with ``DistributedDataParallel``, the constructor
        of ``DistributedDataParallel`` will register the additional gradient
        reduction functions on all the parameters of the model itself at the
        time of construction. If you change the model's parameters afterwards,
        gradient redunction functions no longer match the correct set of
        parameters.

    .. warning::
        Using ``DistributedDataParallel`` in conjunction with the
        :ref:`distributed-rpc-framework` is experimental and subject to change.

    .. warning::
        The ``gradient_as_bucket_view`` mode  does not yet work with Automatic
        Mixed Precision (AMP). AMP maintains stashed gradients that are used for
        unscaling gradients. With ``gradient_as_bucket_view=True``, these
        stashed gradients will point to communication buckets in the first
        iteration. In the next iteration, the communication buckets are mutated
        and thus these stashed gradients will be unexpectedly mutated as well,
        which might lead to wrong results.

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices. This should
                   only be provided when the input module resides on a single
                   CUDA device. For single-device modules, the i'th
                   :attr:`module` replica is placed on ``device_ids[i]``. For
                   multi-device modules and CPU modules, ``device_ids`` must be
                   ``None`` or an empty list, and input data for the forward
                   pass must be placed on the correct device. (default: all
                   visible devices for single-device modules)
        output_device (int or torch.device): Device location of output for
                      single-device CUDA modules. For multi-device modules and
                      CPU modules, it must be ``None``, and the module itself
                      dictates the output location. (default: ``device_ids[0]``
                      for single-device modules)
        broadcast_buffers (bool): Flag that enables syncing (broadcasting)
                          buffers of the module at beginning of the ``forward``
                          function. (default: ``True``)
        process_group: The process group to be used for distributed data
                       all-reduction. If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)
        bucket_cap_mb: ``DistributedDataParallel`` will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       :attr:`bucket_cap_mb` controls the bucket size in
                       MegaBytes (MB). (default: 25)
        find_unused_parameters (bool): Traverse the autograd graph from all
                               tensors contained in the return value of the
                               wrapped module's ``forward`` function. Parameters
                               that don't receive gradients as part of this
                               graph are preemptively marked as being ready to
                               be reduced. Note that all ``forward`` outputs
                               that are derived from module parameters must
                               participate in calculating loss and later the
                               gradient computation. If they don't, this wrapper
                               will hang waiting for autograd to produce
                               gradients for those parameters. Any outputs
                               derived from module parameters that are otherwise
                               unused can be detached from the autograd graph
                               using ``torch.Tensor.detach``. (default: ``False``)
        check_reduction: This argument is deprecated.
        gradient_as_bucket_view (bool): This is a prototype feature and subject
                      to changes. When set to ``True``, gradients will be views
                      pointing to different offsets of ``allreduce`` communication
                      buckets. This can reduce peak memory usage, where the
                      saved memory size will be equal to the total gradients
                      size. Moreover, it avoids the overhead of copying between
                      gradients and ``allreduce`` communication buckets. When
                      gradients are views, ``detach_()`` cannot be called on the
                      gradients. If hitting such errors, please fix it by
                      referring to the :meth:`~torch.optim.Optimizer.zero_grad`
                      function in ``torch/optim/optimizer.py`` as a solution.


    Attributes:
        module (Module): the module to be parallelized.

    Example::

        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = torch.nn.parallel.DistributedDataParallel(model, pg)
    """
    def __init__(self, module, device_ids=None,
                 output_device=None, dim=0, broadcast_buffers=True,
                 process_group=None,
                 bucket_cap_mb=25,
                 find_unused_parameters=False,
                 check_reduction=False,
                 gradient_as_bucket_view=False):

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

            self.device_ids = [_get_device_index(x, True) for x in device_ids]

            if output_device is None:
                output_device = device_ids[0]

            self.output_device = _get_device_index(output_device, True)

        if process_group is None:
            self.process_group = _get_default_group()
        else:
            self.process_group = process_group

        self.dim = dim
        self.module = module
        self.device = list(self.module.parameters())[0].device
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.ddp_uneven_inputs_config = _DDPUnevenInputsConfig(
            ddp_join_enabled=False, ddp_join_divide_by_initial_world_size=False
        )
        self.gradient_as_bucket_view = gradient_as_bucket_view
        if hasattr(module, '_ddp_params_and_buffers_to_ignore'):
            self.parameters_to_ignore = module._ddp_params_and_buffers_to_ignore
        else:
            self.parameters_to_ignore = []

        if check_reduction:
            # This argument is no longer used since the reducer
            # will ensure reduction completes even if some parameters
            # do not receive gradients.
            warnings.warn(
                "The `check_reduction` argument in `DistributedDataParallel` "
                "module is deprecated. Please avoid using it."
            )
            pass

        # Check that a module does not have Uninitialized parameters
        for param in module.parameters():
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                raise RuntimeError(
                    'Modules with uninitialized parameters can\'t be used with `DistributedDataParallel`. '
                    'Run a dummy forward pass to correctly initialize the modules')
        # used for intra-node param sync and inter-node sync as wel
        self.broadcast_bucket_size = int(250 * 1024 * 1024)

        # reduction bucket size
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)
        # Whether to perform input tensor CPU to GPU copies on a side-stream
        self.use_side_stream_for_tensor_copies = os.environ.get("PYTORCH_DDP_USE_SIDE_STREAM", "1") == "1"

        # Sync params and buffers
        self._sync_params_and_buffers(authoritative_rank=0)

        self._ddp_init_helper()

    def _sync_params_and_buffers(self, authoritative_rank=0):
        module_states = []
        for name, param in self.module.state_dict().items():
            if name not in self.parameters_to_ignore:
                module_states.append(param)

        if len(module_states) > 0:
            self._distributed_broadcast_coalesced(
                module_states,
                self.broadcast_bucket_size,
                authoritative_rank)

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
        # Collect buffers for modules, filtering out buffers that should be ignored.
        named_module_buffers = [
            [(buffer, buffer_name) for buffer_name, buffer in m.named_buffers()]
            for m in self._module_copies
        ]
        self.modules_buffers = [
            [
                buffer
                for (buffer, buffer_name) in module_buffers
                if buffer_name not in self.parameters_to_ignore
            ]
            for module_buffers in named_module_buffers
        ]
        # Build tuple of (module, parameter) for all parameters that require grads.
        if self.device_ids and len(self.device_ids) > 1:
            # Single-process multi-device mode,does not support self.parameters_to_ignore.
            if self.parameters_to_ignore:
                raise ValueError(
                    "Single-Process multi-device mode does not "
                    "support ignoring parameters upfront. Please consider "
                    "using one DDP instance per device."
                )

            modules_and_parameters = [
                [
                    (module, parameter)
                    for module in replica.modules()
                    for parameter in filter(
                        lambda parameter: parameter.requires_grad,
                        parameters(module, recurse=False))
                ] for replica in self._module_copies]
        else:
            modules_and_parameters = [
                [
                    (module, parameter)
                    for module_name, module in replica.named_modules()
                    for parameter in [
                        param
                        # Note that we access module.named_parameters instead of
                        # parameters(module). parameters(module) is only needed in the
                        # single-process multi device case, where it accesses replicated
                        # parameters through _former_parameters.
                        for param_name, param in module.named_parameters(recurse=False)
                        if param.requires_grad
                        and f"{module_name}.{param_name}" not in self.parameters_to_ignore
                    ]
                ]
                for replica in self._module_copies
            ]

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
            self.find_unused_parameters,
            self.gradient_as_bucket_view)

        # Set logging data that can be got during construction time.
        dist._set_construction_logging_data(
            self.reducer,
            self.module.__class__.__name__,
            [] if self.device_ids is None else self.device_ids,
            -1 if self.output_device is None else self.output_device,
            self.broadcast_buffers)

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

            >>> ddp = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            >>>   for input in inputs:
            >>>     ddp(input).backward()  # no synchronization, accumulate grads
            >>> ddp(another_input).backward()  # synchronize grads
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync

    def forward(self, *inputs, **kwargs):
        if self.ddp_uneven_inputs_config.ddp_join_enabled:
            ones = torch.ones(
                1, device=self.device
            )
            work = dist.all_reduce(ones, group=self.process_group, async_op=True)
            self.reducer._set_forward_pass_work_handle(
                work, self.ddp_uneven_inputs_config.ddp_join_divide_by_initial_world_size
            )

        # Calling _rebuild_buckets before forward compuation,
        # It may allocate new buckets before deallocating old buckets
        # inside _rebuild_buckets. To save peak memory usage,
        # call _rebuild_buckets before the peak memory usage increases
        # during forward computation.
        # This should be called only once during whole training period.
        if self.reducer._rebuild_buckets():
            logging.info("Reducer buckets have been rebuilt in this iteration.")

        if self.require_forward_param_sync:
            self._sync_params()

        if self.ddp_uneven_inputs_config.ddp_join_enabled:
            # Notify joined ranks whether they should sync in backwards pass or not.
            self._check_global_requires_backward_grad_sync(is_joined_rank=False)

        if self.device_ids:
            if len(self.device_ids) == 1:
                inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
                output = self.module(*inputs[0], **kwargs[0])
            else:
                inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
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

    def _recursive_to(self, inputs, target_gpu):
        r"""
        Recursively moves input to the target_gpu.
        """
        def to_map(obj):
            if isinstance(obj, torch.Tensor):
                if not self.use_side_stream_for_tensor_copies:
                    return (obj.to(target_gpu), )
                else:
                    # Perform CPU -> GPU copies in a background stream. This code is
                    # motivated from similar logic in torch/nn/parallel/_functions.py
                    stream = _get_stream(target_gpu)
                    with torch.cuda.stream(stream):
                        output = obj.to(target_gpu)
                    # synchronize with the copy stream
                    with torch.cuda.device(target_gpu):
                        current_stream = torch.cuda.current_stream()
                        # Sync the current stream with the copy stream
                        current_stream.wait_stream(stream)
                        # Ensure tensor memory is not reused until work on
                        # main stream is complete
                        output.record_stream(current_stream)
                    return (output, )
            if is_namedtuple(obj):
                return [type(obj)(*args) for args in zip(*map(to_map, obj))]
            if isinstance(obj, tuple) and len(obj) > 0:
                return list(zip(*map(to_map, obj)))
            if isinstance(obj, list) and len(obj) > 0:
                return [list(i) for i in zip(*map(to_map, obj))]
            if isinstance(obj, dict) and len(obj) > 0:
                return [type(obj)(i) for i in zip(*map(to_map, obj.items()))]
            return [obj]

        # Avoid reference cycle
        try:
            res = to_map(inputs)
        finally:
            to_map = None
        return res

    def to_kwargs(self, inputs, kwargs, device_id):
        inputs = self._recursive_to(inputs, device_id) if inputs else []
        kwargs = self._recursive_to(kwargs, device_id) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def train(self, mode=True):
        super(DistributedDataParallel, self).train(mode)
        for module in self._module_copies[1:]:
            module.train(mode)
        return self

    def get_ddp_logging_data(self):
        return dist._get_ddp_logging_data(self.reducer)

    # When running in join mode, schedules an allreduce to match the one in the
    # forward pass to determine the no. of currently active processes and whether
    # all processes have joined.
    def _schedule_shadow_all_reduce_for_fwd_pass(self):
        all_active_procs = torch.zeros(
            1, device=self.device
        )
        dist.all_reduce(all_active_procs, group=self.process_group)
        return all_active_procs.item()

    # When running in join mode, schedules an allreduce to notify joined ranks
    # of whether backwards pass synchronization will run this iteraton or not.
    def _check_global_requires_backward_grad_sync(self, is_joined_rank):
        if not is_joined_rank and self.require_backward_grad_sync:
            requires_sync_tensor = torch.ones(1, device=self.device)
        else:
            requires_sync_tensor = torch.zeros(1, device=self.device)

        work = dist.all_reduce(
            requires_sync_tensor, group=self.process_group, async_op=True
        )
        return work, requires_sync_tensor

    # When running in join mode, checks and performs sync of module buffers if
    # the models have buffers that should be synchronized in the forward pass.
    def _check_and_sync_module_buffers(self):
        if self.will_sync_module_buffers():
            my_rank = dist.get_rank(self.process_group)
            authoritative_rank = self._find_common_rank(my_rank, False)
            self._distributed_broadcast_coalesced(
                self.modules_buffers[0], self.broadcast_bucket_size, authoritative_rank
            )

    # When running in join model, agrees upon a common rank and broadcast model
    # parameters to all other ranks.
    def _sync_final_model(self, is_last_joiner):
        # Agree upon the process that will be the authoritative model copy.
        # The current rank is a candidate for being the authoritative copy if
        # is_last_joiner=True. We break ties via picking the larger rank.
        my_rank = dist.get_rank(self.process_group)
        self._authoritative_rank = self._find_common_rank(my_rank, is_last_joiner)
        self._sync_params_and_buffers(authoritative_rank=self._authoritative_rank)

    # Schedule allreduce ops to match those scheduled in the reducer's backward
    # pass.
    def _match_all_reduce_for_bwd_pass(self):
        allreduce_work = []
        # Schedule allreduce in the same order as Reducer schedules them, i.e.
        # the order of the buckets. Retrieving the bucket order from the reducer
        # ensures that we keep the same order in join mode, such as when bucket
        # order is rebuilt dynamically.
        all_bucket_tensors = self.reducer.get_bucket_tensors()
        for bucket_tensors in all_bucket_tensors:
            # Joined processes contribute zero gradient. In the case that
            # divide_by_initial_world_size=True, we divide grads by the static
            # world size, if not, the dividing factor is reduced by the number
            # of joined processes.
            zero_tensors = [
                torch.zeros_like(t) for t in bucket_tensors
            ]
            work = self.process_group.allreduce(zero_tensors)
            allreduce_work.append(work)
        for work in allreduce_work:
            work.wait()

    # Allreduces the used parameter mapping across ranks.
    def _match_unused_params_allreduce(self):
        locally_used_param_maps = self.reducer._get_local_used_maps()
        self.process_group.allreduce(locally_used_param_maps)

    @contextmanager
    def join(self, divide_by_initial_world_size=True, enable=True):
        r"""
        A context manager to be used in conjunction with an instance of
        :class:`torch.nn.parallel.DistributedDataParallel` to be
        able to train with uneven inputs across participating processes.

        This context manager will keep track of already-joined DDP processes,
        and "shadow" the forward and backward passes by inserting collective
        communication operations to match with the ones created by non-joined
        DDP processes. This will ensure each collective call has a corresponding
        call by already-joined DDP processes, preventing hangs or errors that
        would otherwise happen when training with uneven inputs across
        processes.

        Once all DDP processes have joined, the context manager will broadcast
        the model corresponding to the last joined process to all processes to
        ensure the model is the same across all processes
        (which is guaranteed by DDP).

        To use this to enable training with uneven inputs across processes,
        simply wrap this context manager around your training loop. No further
        modifications to the model or data loading is required.

        .. warning::
            This module works only with the multi-process, single-device usage
            of :class:`torch.nn.parallel.DistributedDataParallel`,
            which means that a single process works on a single GPU.

        .. warning::
            This module currently does not support custom distributed collective
            operations in the forward pass, such as ``SyncBatchNorm`` or other
            custom defined collectives in the model's forward pass.

        Args:
            divide_by_initial_world_size (bool): If ``True``, will divide
                gradients by the initial ``world_size`` DDP training was launched
                with. If ``False``, will compute the effective world size
                (number of ranks that have not depleted their inputs yet) and
                divide gradients by that during allreduce. Set
                ``divide_by_initial_world_size=True`` to ensure every input
                sample including the uneven inputs have equal weight in terms of
                how much they contribute to the global gradient. This is
                achieved by always dividing the gradient by the initial
                ``world_size`` even when we encounter uneven inputs. If you set
                this to ``False``, we divide the gradient by the remaining
                number of nodes. This ensures parity with training on a smaller
                ``world_size`` although it also means the uneven inputs would
                contribute more towards the global gradient. Typically, you
                would want to set this to ``True`` for cases where the last few
                inputs of your training job are uneven. In extreme cases, where
                there is a large discrepancy in the number of inputs, setting
                this to ``False`` might provide better results.
            enable (bool): Whether to enable uneven input detection or not. Pass
                in ``enable=False`` to disable in cases where you know that
                inputs are even across participating processes. Default is
                ``True``.


        Example::

          >>>  import torch
          >>>  import torch.distributed as dist
          >>>  import os
          >>>  import torch.multiprocessing as mp
          >>>  import torch.nn as nn
          >>>  # On each spawned worker
          >>>  def worker(rank):
          >>>      dist.init_process_group("nccl", rank=rank, world_size=2)
          >>>      torch.cuda.set_device(rank)
          >>>      model = nn.Linear(1, 1, bias=False).to(rank)
          >>>      model = torch.nn.parallel.DistributedDataParallel(
          >>>          model, device_ids=[rank], output_device=rank
          >>>      )
          >>>      # Rank 1 gets one more input than rank 0.
          >>>      inputs = [torch.tensor([1]).float() for _ in range(10 + rank)]
          >>>      with model.join():
          >>>          for _ in range(5):
          >>>              for inp in inputs:
          >>>                  loss = model(inp).sum()
          >>>                  loss.backward()
          >>>  # Without the join() API, the below synchronization will hang
          >>>  # blocking for rank 1's allreduce to complete.
          >>>  torch.cuda.synchronize(device=rank)
        """
        try:
            if self.device_ids and len(self.device_ids) > 1:
                raise ValueError(
                    """DDP join() API does not support Single-Process Multi-GPU
                    mode training. The recommended approach for DDP training is
                    to spawn a single process that works on a single GPU."""
                )
            has_error = False
            self.ddp_uneven_inputs_config = _DDPUnevenInputsConfig(
                ddp_join_enabled=enable,
                ddp_join_divide_by_initial_world_size=divide_by_initial_world_size,
            )
            yield
        except Exception as e:
            # Set to skip any processing in the finally block.
            has_error = True
            raise e
        finally:
            # Skip any processing to let the exception immediately be raised if
            # there was one.
            if enable and not has_error:
                all_procs_joined = False
                is_last_joiner = True
                i = 0
                WARN_THRESHOLD = 1000
                warnings.simplefilter("once")
                while not all_procs_joined:
                    if i > WARN_THRESHOLD:
                        my_rank = dist.get_rank(self.process_group)
                        warnings.warn(
                            "Detected uneven input skew of greater "
                            f"than {WARN_THRESHOLD}. This means that rank {my_rank} "
                            f"has at least {WARN_THRESHOLD} fewer inputs than "
                            "other currently active ranks. This level of skew could "
                            "lead to performance degradation during training."
                        )
                    # Schedules allreduce to match fwd pass allreduce in non-joined procs
                    num_active_procs = self._schedule_shadow_all_reduce_for_fwd_pass()
                    if num_active_procs == 0:
                        all_procs_joined = True
                    else:
                        # Some DDP process still needs to be joined.
                        if is_last_joiner:
                            is_last_joiner = False
                        # It will rebuild buckets only once during training period
                        self.reducer._rebuild_buckets()
                        # Schedule a corresponding broadcast if we are syncing module
                        # buffers in the forward pass.
                        self._check_and_sync_module_buffers()

                        (
                            work,
                            should_sync_backwards_tensor,
                        ) = self._check_global_requires_backward_grad_sync(
                            is_joined_rank=True
                        )
                        work.wait()
                        # If nonzero, then we should sync in the bwd pass.
                        should_sync_backwards = should_sync_backwards_tensor.item() != 0
                        # Forward param sync is disabled in the next iteration
                        # if we are skipping grad sync this iteration. Hence, we
                        # set require_forward_param_sync appropriately here.
                        self.require_forward_param_sync = should_sync_backwards
                        if not should_sync_backwards:
                            continue
                        # Schedules one allreduce per gradient bucket to match
                        # the backwards pass allreduce.
                        self._match_all_reduce_for_bwd_pass()
                        # Check if we need to allreduce locally unused params.
                        if self.find_unused_parameters:
                            self._match_unused_params_allreduce()
                        # It will push rebuilt params only once during training period
                        self.reducer._push_all_rebuilt_params()
                        i += 1

                # All procs joined. Agree on authoritative rank and broadcast the model.
                self._sync_final_model(is_last_joiner)

    def register_comm_hook(self, state: object, hook: callable):
        r"""
        Registers a communication hook which is an enhancement that provides a
        flexible hook to users where they can specify how DDP aggregates gradients
        across multiple workers.

        This hook would be very useful for researchers to try out new ideas. For
        example, this hook can be used to implement several algorithms like GossipGrad
        and gradient compression which involve different communication strategies for
        parameter syncs while running Distributed DataParallel training.

        Args:
            state (object): Passed to the hook to maintain any state information during the training process.
                            Examples include error feedback in gradient compression,
                            peers to communicate with next in GossipGrad, etc.

                            It is locally stored by each worker
                            and shared by all the gradient tensors on the worker.
            hook (callable): Averages gradient tensors across workers and defined as:
                             ``hook(state: object, bucket: dist._GradBucket) -> torch.futures.Future``:

                             This function is called once the bucket is ready. The
                             hook can perform whatever processing is needed and return
                             a Future indicating completion of any async work (ex: allreduce).
                             If the hook doesn't perform any communication, it can also
                             just return a completed Future. The Future should hold the
                             new value of grad bucket's tensors. Once a bucket is ready,
                             c10d reducer would call this hook and use the tensors returned
                             by the Future and copy grads to individual parameters.

                             We also provide an API called ``get_future`` to retrieve a
                             Future associated with the completion of ``c10d.ProcessGroup.work``.

        .. warning ::
            Grad bucket's tensors will not be predivided by world_size. User is responsible
            to divide by the world_size in case of operations like allreduce.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        .. warning ::
            The Future object that hook returns should contain a result that has the same
            shape with the tensors inside grad bucket.

        .. warning ::
            DDP communication hook does not support single-process multiple-device mode.
            Gradbucket tensors should consist of only a single tensor.

        .. warning ::
            ``get_future`` API supports only NCCL backend and will return a ``torch._C.Future``
            which is an internal type and should be used with caution. It can still be used by
            ``register_comm_hook`` API, but it is subject to some subtle differences compared
            to ``torch.futures.Future``.

        .. warning ::
            DDP communication hook is experimental and subject to change.

        Example::
            Below is an example of a noop hook that returns the same tensors.

            >>> def noop(state: object, bucket: dist._GradBucket): -> torch.futures.Future
            >>>     fut = torch.futures.Future()
            >>>     fut.set_result(bucket.get_tensors())
            >>>     return fut

            >>> ddp.register_comm_hook(state = None, hook = noop)

        Example::
            Below is an example of a Parallel SGD algorithm where gradients are encoded before
            allreduce, and then decoded after allreduce.

            >>> def encode_and_decode(state: object, bucket: dist._GradBucket): -> torch.futures.Future
            >>>     tensors = [t / process_group.world_size for t in bucket.get_tensors()]
            >>>     encoded_tensors = encode(tensors) # encode gradients
            >>>     fut = process_group.allreduce(encoded_tensors).get_future()
            >>>     # Define the then callback to decode.
            >>>     def decode(fut):
            >>>         decoded_tensors = decode(fut.value()) # decode gradients
            >>>         return decoded_tensors
            >>>     return fut.then(decode)

            >>> ddp.register_comm_hook(state = None, hook = encode_and_decode)
        """
        self._check_comm_hook(hook)
        dist._register_comm_hook(self.reducer, state, hook)

    def _register_builtin_comm_hook(
        self, comm_hook_type
    ):
        r"""
        Registers a built-in communication hook that specifies how DDP
        aggregates gradients across multiple workers.
        The built-in hooks aim to provide efficient C++ implementations for certain hooks,
        which might not be as efficient if implemented in Python using a Python communication hook.

        Args:
            comm_hook_type (dist.BuiltinCommHookType): type of communication hook, such as
            ALLREDUCE, FP16_COMPRESS, etc.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        .. warning ::
            DDP communication hook does not support single-process multiple-device mode.
            Gradbucket tensors should consist of only a single tensor.

        .. warning ::
            DDP communication hook is experimental and subject to change.

        Example::
            Below is an example of a FP16 compression where gradients are
            compressed into 16-bit floating-point numbers before allreduce, and
            then decompressed after allreduce.

            >>> ddp._register_builtin_comm_hook(dist.BuiltinCommHookType.FP16_COMPRESS)

        """
        dist._register_builtin_comm_hook(self.reducer, comm_hook_type)

    def _distributed_broadcast_coalesced(
        self, tensors, buffer_size, authoritative_rank=0
    ):
        dist._broadcast_coalesced(
            self.process_group, tensors, buffer_size, authoritative_rank
        )

    def will_sync_module_buffers(self):
        return (
            self.require_forward_param_sync
            and self.broadcast_buffers
            and len(self.modules_buffers[0]) > 0
        )

    def _find_common_rank(self, input_rank, rank_cond):
        # -1 indicates that this rank is not under consideration to be the
        # common_rank
        rank_to_use = torch.tensor(
            [input_rank if rank_cond else -1],
            device=self.device,
        )
        dist.all_reduce(rank_to_use, op=ReduceOp.MAX, group=self.process_group)
        if rank_to_use.item() == -1:
            raise ValueError(
                "BUG! Expected rank_cond to be true for at least one process."
            )
        return rank_to_use.item()

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
                            if param.grad.grad_fn is not None:
                                param.grad.detach_()
                            else:
                                param.grad.requires_grad_(False)
                            param.grad.zero_()

            # module buffer sync
            if self.will_sync_module_buffers():
                # Synchronize buffers across processes.
                # If we are running DDP with the join manager, we have to agree
                # upon a rank to sync module buffers from, since rank 0 may
                # already have been joined and have stale module buffers.
                if self.ddp_uneven_inputs_config.ddp_join_enabled:
                    authoritative_rank = self._find_common_rank(dist.get_rank(), True)
                else:
                    # The process with rank 0 is considered the authoritative copy.
                    authoritative_rank = 0
                self._distributed_broadcast_coalesced(
                    self.modules_buffers[0],
                    self.broadcast_bucket_size,
                    authoritative_rank,
                )
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
        if (
            sig.parameters["bucket"].annotation != inspect._empty
            and sig.parameters["bucket"].annotation != dist._GradBucket
        ):
            raise ValueError(
                "Communication hook: bucket annotation should be dist._GradBucket."
            )

        if sig.return_annotation != inspect._empty and (
            sig.return_annotation != torch.futures.Future
            and sig.return_annotation != torch._C.Future
        ):
            raise ValueError(
                "Communication hook: return annotation should be torch.futures.Future or torch._C.Future."
            )

    @staticmethod
    def _set_params_and_buffers_to_ignore_for_model(
        module, params_and_buffers_to_ignore
    ):
        # This is a workaround to set parameters and buffers DDP should ignore
        # during synchronization. It will be removed when the API is finalized
        # as part of addressing https://github.com/pytorch/pytorch/issues/43690.
        module._ddp_params_and_buffers_to_ignore = params_and_buffers_to_ignore
