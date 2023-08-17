import copy
import functools
import inspect
import itertools
import logging
import os
import sys
import warnings
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import auto, Enum
from typing import Any, Callable, List, Optional, Type

import torch
import torch.distributed as dist
from torch.autograd import Function, Variable
from torch.distributed.algorithms.join import Join, Joinable, JoinHook

from torch.utils._pytree import tree_flatten, tree_unflatten

RPC_AVAILABLE = False
if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
    from torch.distributed.utils import (
        _alloc_storage,
        _cast_forward_inputs,
        _free_storage,
        _sync_module_states,
        _to_kwargs,
        _verify_param_shape_across_processes,
    )
if torch.distributed.rpc.is_available():
    RPC_AVAILABLE = True
    from torch.distributed.rpc import RRef

from torch._utils import _get_device_index

from ..modules import Module
from .scatter_gather import gather, scatter_kwargs  # noqa: F401

__all__ = ["DistributedDataParallel"]

logger = logging.getLogger(__name__)


@dataclass
class _MixedPrecision:
    """
    This configures DDP-native mixed precision training.

    Attributes:
        param_dtype (torch.dtype): This specifies the dtype for model
            parameters, inputs (when ``cast_forward_inputs`` is set to
            ``True``), and therefore the dtype for computation.
            However, outside the forward and backward passes, parameters are in
            full precision. Model checkpointing always happens in full
            precision.
        reduce_dtype (torch.dtype): This specifies the dtype for gradient
            reduction, which is permitted to differ from ``param_dtype``.
        buffer_dtype (torch.dtype): This specifies the dtype for buffers.

    .. note:: This API is experimental and subject to change.

    .. note:: Only floating point tensors are cast to their specified dtypes.

    .. note:: ``state_dict`` checkpoints parameters and buffers in full
        precision.

    .. note:: Each low precision dtype must be specified explicitly. For
        example, ``_MixedPrecision(reduce_dtype=torch.float16)`` only specifies
        the reduction dtype to be low precision, and DDP will not cast
        parameters or buffers.

    .. note:: If a ``reduce_dtype`` is not specified, then gradient reduction
        happens in ``param_dtype`` if specified or the original parameter dtype
        otherwise. For example, ``_MixedPrecision(param_dtype=torch.float16)``
        would result in communication occurring in fp16.
    """

    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    buffer_dtype: Optional[torch.dtype] = None
    # TODO (rohan-varma): keep_low_precision_grads: bool = False
    # TODO (rohan-varma): APIs to allow users to run batchnorm and layernorm
    # in full precision. For DDP, this can be implemented by not performing the
    # parameter cast for BN and LN units.


def _cast_buffers(mixed_precision_config, root_module):
    """
    Casts buffers to the given ``buffer_dtype``.
    """
    for buf in root_module.buffers():
        if hasattr(buf, "_ddp_ignored") and buf._ddp_ignored:
            continue

        buf.data = buf.to(dtype=mixed_precision_config.buffer_dtype)


def _setup_mixed_precision_params(mixed_precision_config, root_module):
    """
    Creates and frees storage for the mixed precision parameters.
    """
    for param in root_module.parameters():
        # Do not setup mixed precision for DDP ignored parameters.
        if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
            continue

        if not hasattr(param, "_mp_param"):
            param._mp_param = torch.zeros_like(
                param,
                device=param.device,
                dtype=mixed_precision_config.param_dtype,
                requires_grad=param.requires_grad,
            )
            _free_storage(param._mp_param)
            # _fp_param will point to the full precision param so it can be switched
            # back to at the end of forward / backward.
            param._fp_param = param.data


def _tree_flatten_with_rref(output):
    output_is_rref = RPC_AVAILABLE and isinstance(output, RRef)
    if output_is_rref:
        output_tensor_list, treespec = tree_flatten(output.local_value())
    else:
        output_tensor_list, treespec = tree_flatten(output)
    # Need to return flattened tensors, spec to re-pack them, as well
    # as if the return type was actually an RRef to reconstruct.
    return output_tensor_list, treespec, output_is_rref


def _tree_unflatten_with_rref(output, treespec, output_is_rref):
    output = tree_unflatten(output, treespec)
    if output_is_rref:
        output = RRef(output)
    return output


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
    if is_dataclass(obj):
        return itertools.chain(
            *map(_find_tensors, (getattr(obj, f.name) for f in fields(obj)))
        )

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
        "NCCL_ASYNC_ERROR_HANDLING",
    ]
    formatted_output = ""
    for var in relevant_env_vars:
        value = os.environ[var] if var in os.environ else "N/A"
        formatted_output += f"env:{var}={value}\n"
    print(formatted_output)


class _BufferCommHookLocation(Enum):
    PRE_FORWARD = auto()
    POST_FORWARD = auto()


@dataclass
class _BufferCommHook:
    buffer_comm_hook: Callable
    buffer_comm_hook_state: Any
    buffer_comm_hook_location: _BufferCommHookLocation


# Add a DDPSink to run various functions when backwards starts, such as
# queueing call back of out-most backward/graph task,
# this helps call back is fired after all gradients' calculation
# is completed.
class _DDPSink(Function):
    @staticmethod
    def forward(ctx, ddp_weakref, *inputs):
        # set_materialize_grads(False) will ensure that None gradients stay as
        # None and are not filled with zeros.
        ctx.set_materialize_grads(False)
        ctx.ddp_weakref = ddp_weakref
        ret = tuple(
            inp.clone() if isinstance(inp, torch.Tensor) else inp for inp in inputs
        )
        return ret

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Enqueue delay allreduce for static graph training on the first
        # iteration.
        ddp_weakref = ctx.ddp_weakref()
        reducer = ddp_weakref.reducer
        static_graph = ddp_weakref.static_graph
        delay_ar_enqueued = (
            static_graph and ddp_weakref._static_graph_delay_allreduce_enqueued
        )
        if static_graph and not delay_ar_enqueued:
            Variable._execution_engine.queue_callback(  # type: ignore[call-arg,misc]
                reducer._delay_all_reduce
            )
            ddp_weakref._static_graph_delay_allreduce_enqueued = True

        return (None, *grad_outputs)


class _DDPJoinHook(JoinHook):
    def __init__(self, ddp, divide_by_initial_world_size):
        """
        Sets config variables for internal usage.
        """
        assert isinstance(ddp, DistributedDataParallel), (
            "DDP join hook requires passing in a DistributedDataParallel "
            "instance as the state"
        )
        assert ddp.logger is not None
        ddp.logger._set_uneven_input_join()
        self.ddp = ddp
        self.ddp._divide_by_initial_world_size = divide_by_initial_world_size
        super().__init__()

    def main_hook(self):
        """
        Shadows the DDP collective communication operations in the forward and
        backward passes.
        """
        ddp = self.ddp
        # Buckets are rebuilt only once during a training period
        ddp.reducer._rebuild_buckets()

        # Schedule a broadcast if we are syncing module buffers in the
        # forward pass
        # TODO: make DDP uneven inputs context manager support buffer
        # comm hook (https://github.com/pytorch/pytorch/issues/65436)
        ddp._check_and_sync_module_buffers()

        # Check if need to sync in the backward pass
        work = ddp._check_global_requires_backward_grad_sync(is_joined_rank=True)
        work.wait()
        should_sync_backwards = work.result()[0].item() != 0
        # Forward parameter sync is disabled in the next iteration if we
        # are skipping gradient sync this iteration, so set
        # `require_forward_param_sync` accordingly
        ddp.require_forward_param_sync = should_sync_backwards
        if not should_sync_backwards:
            return

        # Schedule one allreduce per gradient bucket to match the backward
        # pass allreduce
        ddp._match_all_reduce_for_bwd_pass()

        # Check if we need to allreduce locally unused parameters
        if ddp.find_unused_parameters:
            ddp._match_unused_params_allreduce()

        # Rebuilt parameters are pushed only once during a training period
        ddp.reducer._push_all_rebuilt_params()

    def post_hook(self, is_last_joiner: bool):
        """
        Syncs the final model to ensure that the model is the same across all
        processes.
        """
        self.ddp._sync_final_model(is_last_joiner)


class DistributedDataParallel(Module, Joinable):
    r"""Implements distributed data parallelism that is based on
    ``torch.distributed`` package at the module level.

    This container provides data parallelism by synchronizing gradients
    across each model replica. The devices to synchronize across are
    specified by the input ``process_group``, which is the entire world
    by default. Note that ``DistributedDataParallel`` does not chunk or
    otherwise shard the input across participating GPUs; the user is
    responsible for defining how to do so, for example through the use
    of a :class:`DistributedSampler`.

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

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.cuda.set_device(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.distributed.init_process_group(
        >>>     backend='nccl', world_size=N, init_method='...'
        >>> )
        >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``.

    .. note::
        Please refer to `PyTorch Distributed Overview <https://pytorch.org/tutorials/beginner/dist_overview.html>`__
        for a brief introduction to all features related to distributed training.

    .. note::
        ``DistributedDataParallel`` can be used in conjunction with
        :class:`torch.distributed.optim.ZeroRedundancyOptimizer` to reduce
        per-rank optimizer states memory footprint. Please refer to
        `ZeroRedundancyOptimizer recipe <https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html>`__
        for more details.

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

            >>> # xdoctest: +SKIP("undefined variables")
            >>> import torch.distributed.autograd as dist_autograd
            >>> from torch.nn.parallel import DistributedDataParallel as DDP
            >>> import torch
            >>> from torch import optim
            >>> from torch.distributed.optim import DistributedOptimizer
            >>> import torch.distributed.rpc as rpc
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
            >>>     loss = loss_func(pred, target)
            >>>     dist_autograd.backward(context_id, [loss])
            >>>     dist_optim.step(context_id)

    .. note::
        DistributedDataParallel currently offers limited support for gradient
        checkpointing with :meth:`torch.utils.checkpoint`. DDP will work as
        expected when there are no unused parameters in the model and each layer
        is checkpointed at most once (make sure you are not passing
        `find_unused_parameters=True` to DDP). We currently do not support the
        case where a layer is checkpointed multiple times, or when there unused
        parameters in the checkpointed model.

    .. note::
        To let a non-DDP model load a state dict from a DDP model,
        :meth:`~torch.nn.modules.utils.consume_prefix_in_state_dict_if_present`
        needs to be applied to strip the prefix "module." in the DDP state dict before loading.

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
        You should never try to change your model's parameters after wrapping
        up your model with ``DistributedDataParallel``. Because, when
        wrapping up your model with ``DistributedDataParallel``, the constructor
        of ``DistributedDataParallel`` will register the additional gradient
        reduction functions on all the parameters of the model itself at the
        time of construction. If you change the model's parameters afterwards,
        gradient reduction functions no longer match the correct set of
        parameters.

    .. warning::
        Using ``DistributedDataParallel`` in conjunction with the
        :ref:`distributed-rpc-framework` is experimental and subject to change.

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices.
                   1) For single-device modules, ``device_ids`` can
                   contain exactly one device id, which represents the only
                   CUDA device where the input module corresponding to this process resides.
                   Alternatively, ``device_ids`` can also be ``None``.
                   2) For multi-device modules and CPU modules,
                   ``device_ids`` must be ``None``.

                   When ``device_ids`` is ``None`` for both cases,
                   both the input data for the forward pass and the actual module
                   must be placed on the correct device.
                   (default: ``None``)
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
                               be reduced. In addition, parameters that may have
                               been used in the wrapped module's ``forward``
                               function but were not part of loss computation and
                               thus would also not receive gradients are
                               preemptively marked as ready to be reduced.
                               (default: ``False``)
        check_reduction: This argument is deprecated.
        gradient_as_bucket_view (bool): When set to ``True``, gradients will be views
                      pointing to different offsets of ``allreduce`` communication
                      buckets. This can reduce peak memory usage, where the
                      saved memory size will be equal to the total gradients
                      size. Moreover, it avoids the overhead of copying between
                      gradients and ``allreduce`` communication buckets. When
                      gradients are views, ``detach_()`` cannot be called on the
                      gradients. If hitting such errors, please fix it by
                      referring to the :meth:`~torch.optim.Optimizer.zero_grad`
                      function in ``torch/optim/optimizer.py`` as a solution.
                      Note that gradients will be views after first iteration, so
                      the peak memory saving should be checked after first iteration.
        static_graph (bool): When set to ``True``, DDP knows the trained graph is
                     static. Static graph means 1) The set of used and unused
                     parameters will not change during the whole training loop; in
                     this case, it does not matter whether users set
                     ``find_unused_parameters = True`` or not. 2) How the graph is trained
                     will not change during the whole training loop (meaning there is
                     no control flow depending on iterations).
                     When static_graph is set to be ``True``, DDP will support cases that
                     can not be supported in the past:
                     1) Reentrant backwards.
                     2) Activation checkpointing multiple times.
                     3) Activation checkpointing when model has unused parameters.
                     4) There are model parameters that are outside of forward function.
                     5) Potentially improve performance when there are unused parameters,
                     as DDP will not search graph in each iteration to detect unused
                     parameters when static_graph is set to be ``True``.
                     To check whether you can set static_graph to be ``True``, one way is to
                     check ddp logging data at the end of your previous model training,
                     if ``ddp_logging_data.get("can_set_static_graph") == True``, mostly you
                     can set ``static_graph = True`` as well.

                     Example::
                         >>> # xdoctest: +SKIP("undefined variables")
                         >>> model_DDP = torch.nn.parallel.DistributedDataParallel(model)
                         >>> # Training loop
                         >>> ...
                         >>> ddp_logging_data = model_DDP._get_ddp_logging_data()
                         >>> static_graph = ddp_logging_data.get("can_set_static_graph")
        delay_all_reduce_named_params (list of tuple of str and torch.nn.Parameter): a list
                    of named parameters whose all reduce will be delayed when the gradient of
                    the parameter specified in ``param_to_hook_all_reduce`` is ready. Other
                    arguments of DDP do not apply to named params specified in this argument
                    as these named params will be ignored by DDP reducer.
        param_to_hook_all_reduce (torch.nn.Parameter): a parameter to hook delayed all reduce
                    of parameters specified in ``delay_all_reduce_named_params``.


    Attributes:
        module (Module): the module to be parallelized.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = torch.nn.parallel.DistributedDataParallel(model)
    """

    # used to track whether the given thread is inside ddp forward for torchdynamo purposes
    _active_ddp_module = None

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        static_graph=False,
        delay_all_reduce_named_params=None,
        param_to_hook_all_reduce=None,
        mixed_precision: Optional[_MixedPrecision] = None,
    ):
        super().__init__()
        Joinable.__init__(self)
        self.logger = None
        if bool(delay_all_reduce_named_params is not None) != bool(
            param_to_hook_all_reduce is not None
        ):
            self._log_and_throw(
                ValueError,
                "delay_all_reduce_named_params and param_to_hook_all_reduce "
                "need to be set at the same time.",
            )

        self._delay_all_reduce_params = []
        if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = set(module._ddp_params_and_buffers_to_ignore)
        else:
            self.parameters_to_ignore = set()
        if delay_all_reduce_named_params is not None:
            for name, param in delay_all_reduce_named_params:
                self.parameters_to_ignore.add(name)
                self._delay_all_reduce_params.append(param)

        self._module_parameters = [
            p
            for n, p in module.named_parameters()
            if n not in self.parameters_to_ignore
        ]
        if not any(p.requires_grad for p in self._module_parameters):
            if len(self._delay_all_reduce_params):
                logger.info("Delay the AllReduce of all parameters.")
            else:
                self._log_and_throw(
                    RuntimeError,
                    "DistributedDataParallel is not needed when a module "
                    "doesn't have any parameter that requires a gradient.",
                )

        if device_ids is not None and len(device_ids) > 1:
            self._log_and_throw(
                ValueError,
                "device_ids can only be None or contain a single element.",
            )

        self.is_multi_device_module = (
            len({p.device for p in self._module_parameters}) > 1
        )
        distinct_device_types = {
            p.device.type for p in self._module_parameters if p.device is not None
        }
        if len(distinct_device_types) != 1:
            self._log_and_throw(
                ValueError,
                "DistributedDataParallel's input module must be on "
                "the same type of devices, but input module parameters locate in {}.".format(
                    distinct_device_types
                ),
            )

        self.device_type = list(distinct_device_types)[0]

        if (
            device_ids is None
            or len(device_ids) == 0  # For backward compatibility.
            or self.device_type == "cpu"
            or self.is_multi_device_module
        ):
            if device_ids or output_device:
                self._log_and_throw(
                    ValueError,
                    "DistributedDataParallel device_ids and output_device arguments "
                    "only work with single-device/multiple-device GPU modules or CPU modules, "
                    "but got device_ids {}, output_device {}, and module parameters {}.".format(
                        device_ids,
                        output_device,
                        {p.device for p in self._module_parameters},
                    ),
                )

            self.device_ids = None
            self.output_device = None
        else:
            self.device_ids = [_get_device_index(x, True) for x in device_ids]

            if output_device is None:
                output_device = device_ids[0]

            self.output_device = _get_device_index(output_device, True)

        if process_group is None:
            self.process_group = _get_default_group()
        else:
            self.process_group = process_group

        self.static_graph = False
        self.dim = dim
        self.module = module
        self.device = list(self._module_parameters)[0].device
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.mixed_precision = mixed_precision
        if self.mixed_precision is not None:
            logger.warning("Received mixed precision config %s", self.mixed_precision)

        if check_reduction:
            # This argument is no longer used since the reducer
            # will ensure reduction completes even if some parameters
            # do not receive gradients.
            warnings.warn(
                "The `check_reduction` argument in `DistributedDataParallel` "
                "module is deprecated. Please avoid using it.",
                stacklevel=2,
            )

        # Check that a module does not have Uninitialized parameters
        for param in self._module_parameters:
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                self._log_and_throw(
                    RuntimeError,
                    "Modules with uninitialized parameters can't be used with `DistributedDataParallel`. "
                    "Run a dummy forward pass to correctly initialize the modules",
                )
        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = int(250 * 1024 * 1024)

        # reduction bucket size
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)
        # Whether to perform input tensor CPU to GPU copies on a side-stream
        self.use_side_stream_for_tensor_copies = (
            os.environ.get("PYTORCH_DDP_USE_SIDE_STREAM", "1") == "1"
        )

        # Initialize gradient buffers and register all reduce hook
        self._delay_grad_buffer = None
        self._delay_grad_views: List[torch.Tensor] = []
        self._delay_all_reduce_all_params = False
        if len(self._delay_all_reduce_params) != 0:
            self._register_delay_all_reduce_hook(
                bucket_cap_mb=bucket_cap_mb,
                process_group=self.process_group,
                param_to_hook_all_reduce=param_to_hook_all_reduce,
                device_ids=device_ids,
            )
            if self._delay_all_reduce_all_params:
                return

        # Build parameters for reducer.
        parameters, expect_sparse_gradient = self._build_params_for_reducer()
        # Verify model equivalence.
        _verify_param_shape_across_processes(self.process_group, parameters)
        # Sync params and buffers. Ensures all DDP models start off at the same value.
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=0,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
            broadcast_buffers=self.broadcast_buffers,
        )
        # In debug mode, build a mapping of parameter index -> parameter.
        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)

        # Builds reducer.
        self._ddp_init_helper(
            parameters,
            expect_sparse_gradient,
            param_to_name_mapping,
            static_graph,
        )
        if self.mixed_precision is not None:
            _setup_mixed_precision_params(self.mixed_precision, self.module)
            _cast_buffers(self.mixed_precision, self.module)
            # Stream used for async low precision copies.
            self._mp_stream = torch.cuda.Stream()
            self._submodule_to_event = defaultdict(deque)  # type: ignore[var-annotated]
            # Add forward pre-hook to root module to kick off copies to lower
            # precision.
            self.module.register_forward_pre_hook(
                self._root_copy_hook, prepend=False, with_kwargs=True
            )
            # Add forward pre hook to all submodules to wait for copy events
            # before running computation.
            for module in self.module.modules():
                module.register_forward_pre_hook(
                    self._module_wait_for_copy_hook,
                    prepend=False,
                    with_kwargs=True,
                )
            # Set up callbacks in backward to upcast and use full precision
            # params. TODO (rohan-varma): Make this compose with general
            # comm hooks and apply_optimizer_in_backward. Importing inline to
            # avoid circular import issue.
            from torch.distributed.algorithms.ddp_comm_hooks.mixed_precision_hooks import (
                _AllreduceUpcastHookState,
                _reducer_allreduce_and_upcast_hook,
            )

            upcast_hook_state = _AllreduceUpcastHookState(
                ddp_weakref=weakref.ref(self),
                upcast_stream=torch.cuda.Stream(),
            )
            self.register_comm_hook(
                upcast_hook_state,
                _reducer_allreduce_and_upcast_hook,
            )
            # Inform reducer of reduced precision param dtype for correctness
            # of type checks between gradient and bucket.
            self.reducer._set_mixed_precision_param_dtype(  # type: ignore[attr-defined]
                self.mixed_precision.param_dtype
            )

        self._has_rebuilt_buckets = False

        if static_graph:
            self._set_static_graph()

        self._lazy_init_ran = False

    def _register_delay_all_reduce_hook(
        self,
        bucket_cap_mb,
        process_group,
        param_to_hook_all_reduce,
        device_ids,
    ):
        # 1. Create gradient buffer
        device = torch.device("cpu") if device_ids is None else device_ids[0]
        self._delay_grad_buffer = torch.zeros(
            sum([p.numel() for p in self._delay_all_reduce_params]),
            device=device,
        )

        # 2. Broadcast the parameters
        detached_params = [p.detach() for p in self._delay_all_reduce_params]
        dist._broadcast_coalesced(process_group, detached_params, bucket_cap_mb, 0)

        # 3. Hook all reduce to the specified parameter
        world_size = dist.get_world_size(process_group)

        def _delayed_all_reduce(grad):
            self._delay_grad_buffer.div_(world_size)  # type: ignore[union-attr]
            _ = dist.all_reduce(
                self._delay_grad_buffer, group=process_group, async_op=True
            )
            return grad

        param_to_hook_all_reduce.register_hook(_delayed_all_reduce)

        # 4. Build tensor views for gradients
        offset = 0
        for param in self._delay_all_reduce_params:
            grad_view = self._delay_grad_buffer[offset : (offset + param.numel())].view(
                param.shape
            )
            self._delay_grad_views.append(grad_view)
            offset = offset + param.numel()

        # 5. Check whether the all reduce of all params requiring grad is delayed.
        for module_name, module in self.module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    full_name = f"{module_name}.{param_name}"
                    if full_name not in self.parameters_to_ignore:
                        # There is at least a param whose all reduce will not be delayed.
                        # In this case, we should not set self._delay_all_reduce_all_params
                        # to True.
                        return
        self._delay_all_reduce_all_params = True

    def _setup_in_backward_optimizers(self):
        # Check if user has used apply_optim_in_backward to overlap optimizer
        # step + DDP backward. Current constraints:
        # 1. Only allreduce is supported at the moment, no custom communication.
        # 2. For DDP-managed parameters that have their optimizer run in
        # backward, their gradients are set to ``None``. If your use case
        # requires DDP parameters grad not to be set to ``None`` after their
        # in-backward optimizer runs, please ping
        # https://github.com/pytorch/pytorch/issues/90052.
        # NOTE: we use self._module_parameters instead of .parameters() since
        # the former excludes ignored (non-DDP managed) parameters.
        if any(hasattr(p, "_in_backward_optimizers") for p in self._module_parameters):
            # Remove hooks that apply_optim_in_backward had registered because
            # DDP customizes how optimizer is overlapped with backward due to
            # the allreduce.
            param_to_handle_map = (
                dist.optim.apply_optimizer_in_backward.param_to_optim_hook_handle_map
            )
            for p in self._module_parameters:
                for handle in param_to_handle_map.get(p, []):
                    handle.remove()

            # Need a weakref to DDP instance to run all_reduce (from reducer)
            # and get managed DDP parameters.
            ddp_weakref = weakref.ref(self)
            # Note: importing in function, otherwise this will cause a circular
            # import.
            from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
                _apply_optim_in_backward_hook,
            )

            self.register_comm_hook(
                ddp_weakref,
                _apply_optim_in_backward_hook(
                    gradient_is_bucket_view=self.gradient_as_bucket_view
                ),
            )

            self.reducer._set_optimizer_in_backward()  # type: ignore[attr-defined]

    def _fire_reducer_autograd_hook(self, idx, *unused):
        """
        Fires the reducer's autograd hook to allreduce params in a Reducer bucket.
        Note that this is only used during mixed precision training as the
        Reducer's hooks installed during construction time would not be called
        as we're working in the low precision parameter setting.
        """
        self.reducer._autograd_hook(idx)  # type: ignore[attr-defined]

    def _root_copy_hook(self, *args: Any, **kwargs: Any) -> None:
        """
        When training with DDP mixed precision, this root pre-forward hook kicks
        off low precision copies on a separate stream and creates respective
        events to wait for them.
        """
        # Clear out previous iteration submodule to event. This is because we
        # may have populated some events for modules that didn't end up being
        # used.
        self._submodule_to_event = defaultdict(deque)  # type: ignore[var-annotated]
        with torch.cuda.stream(self._mp_stream):
            for submodule in self.module.modules():
                for param in submodule.parameters(recurse=False):
                    # Do not cast DDP ignored parameters.
                    if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
                        continue
                    _alloc_storage(param._mp_param, param.size())
                    # copy() implicitly casts to low precision
                    with torch.no_grad():
                        param._mp_param.copy_(param.data)
                        # TODO: when zero_grad(set_to_none=False) or in grad
                        # accumulation case, accumulated grads can be in fp32
                        # which can cause errors when running DDP backwards due
                        # to mismatched incoming and accumulated gradient types.
                        # So we manually cast the accumulated grad down for now,
                        # in the future we may shift to FSDP style gradient
                        # accumulation management where the accumulated gradient
                        # is saved and .grad field is set to None, bypassing
                        # this issue.
                        if param.grad is not None:
                            param.grad.data = param.grad.to(
                                self.mixed_precision.param_dtype  # type: ignore[union-attr]
                            )
                    param.data = param._mp_param
                copy_event = torch.cuda.Event()
                copy_event.record()
                self._submodule_to_event[submodule].append(copy_event)

    def _module_wait_for_copy_hook(
        self,
        module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Before carrying out computation, wait on the appropriate event to ensure
        low precision copies have finished.
        """
        try:
            event = self._submodule_to_event[module].popleft()
        except IndexError:
            # copy event has already been waited on
            return

        event.wait(stream=torch.cuda.current_stream())
        for p in module.parameters(recurse=False):
            # Don't register hooks if param does not require grad
            if not p.requires_grad or (hasattr(p, "_ddp_ignored") and p._ddp_ignored):
                continue
            # We need to register autograd hook here instead of DDP's ctor
            # since we're working with the low precision param. Register them
            # via obtaining the gradient accumulator.
            tmp = p.expand_as(p)
            grad_acc = tmp.grad_fn.next_functions[0][0]

            hook = grad_acc.register_hook(
                functools.partial(self._fire_reducer_autograd_hook, p._idx)
            )
            p._ddp_mp_hook_state = (grad_acc, hook)

    def _log_and_throw(self, err_type, err_msg):
        if self.logger is not None:
            self.logger.set_error_and_log(f"{str(err_type)}: {err_msg}")
        raise err_type(err_msg)

    def _ddp_init_helper(
        self,
        parameters,
        expect_sparse_gradient,
        param_to_name_mapping,
        static_graph,
    ):
        """
        Initialization helper function that does the following:
        (1) bucketing the parameters for reductions
        (2) resetting the bucketing states
        (3) registering the grad hooks
        (4) Logging construction-time DDP logging data
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """
        # Notice, the parameters order is not in the order in which they are used,
        # especially in models with control flow.
        #
        # Alongside parameters are not presented in the real execution order,
        # if a certain model happens to also
        #   1) have other collectives comm ops in its backward graph.
        #   2) have unused parameter in subset ranks of the whole world.
        # bucketing could insert ALL-REDUCE comm op too early on the rank with unused parameter,
        # matching up with other collectives comm ops on other ranks unexpectedly.
        #
        # In order to handle this corner case, when the parameters are not in the real execution order,
        # we don't do bucketing, thus only one ALL-REDUCE is inserted after all the gradients
        # of the whole graph are computed.
        #
        # Notice, here we only disable bucketing for the first iteration.
        # After the first iteration, it's OK to rebuild buckets,
        # because "bucket rebuild" bucketizes parameters based on its real execution order in backward graph.

        # Can remove this branching once #73732 is landed.
        if static_graph is True or self.find_unused_parameters is False:
            bucket_size_limits = [sys.maxsize]
        else:
            bucket_size_limits = [
                dist._DEFAULT_FIRST_BUCKET_BYTES,
                self.bucket_bytes_cap,
            ]
        (
            bucket_indices,
            per_bucket_size_limits,
        ) = dist._compute_bucket_assignment_by_size(
            parameters,
            bucket_size_limits,
            expect_sparse_gradient,
        )

        # Remember index for parameters if we are in mixed precision, as we
        # need to pass in index to Reducer's autograd hook via python.
        if self.mixed_precision is not None:
            for i, p in enumerate(parameters):
                p._idx = i

        # Note: reverse list of buckets because we want to approximate the
        # order in which their gradients are produced, and assume they
        # are used in the forward pass in the order they are defined.
        self.reducer = dist.Reducer(
            parameters,
            list(reversed(bucket_indices)),
            list(reversed(per_bucket_size_limits)),
            self.process_group,
            expect_sparse_gradient,
            # The bucket size limit is specified in the constructor.
            # Additionally, we allow for a single small bucket for parameters
            # that are defined first, such that their gradients don't spill into
            # a much larger bucket, adding unnecessary latency after gradient
            # computation finishes. Experiments showed 1MB is a reasonable value.
            self.bucket_bytes_cap,
            self.find_unused_parameters,
            self.gradient_as_bucket_view,
            param_to_name_mapping,
            # User can set dist._DEFAULT_FIRST_BUCKET_BYTES to tune DDP first
            # bucket.
            dist._DEFAULT_FIRST_BUCKET_BYTES,
        )

        self.logger = dist.Logger(self.reducer)
        # Set as a weak reference to avoid reference cycle between
        # logger and reducer.
        self.reducer.set_logger(self.logger)

        has_sync_bn = False
        for submodule in self.module.modules():
            if isinstance(submodule, torch.nn.SyncBatchNorm):
                has_sync_bn = True
                break

        # Set logging data that can be got during construction time.
        self.logger.set_construction_data_and_log(
            self.module.__class__.__name__,
            [] if self.device_ids is None else self.device_ids,
            -1 if self.output_device is None else self.output_device,
            self.broadcast_buffers,
            has_sync_bn,
            static_graph,
        )

        # passing a handle to torch.nn.SyncBatchNorm layer
        self._passing_sync_batchnorm_handle(self.module)

    def __getstate__(self):
        self._check_default_group()
        attrs = copy.copy(self.__dict__)
        del attrs["process_group"]
        del attrs["reducer"]
        del attrs["logger"]
        return attrs

    def __setstate__(self, state):
        # If serializable, then the process group should be the default one
        self.process_group = _get_default_group()
        super().__setstate__(state)
        self.__dict__.setdefault("require_forward_param_sync", True)
        self.__dict__.setdefault("require_backward_grad_sync", True)
        parameters, expect_sparse_gradient = self._build_params_for_reducer()
        # In debug mode, build a mapping of parameter index -> parameter.
        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)
        # Builds reducer.
        self._ddp_init_helper(
            parameters,
            expect_sparse_gradient,
            param_to_name_mapping,
            self.static_graph,
        )
        if self.static_graph:
            self.reducer._set_static_graph()
            assert self.logger is not None
            self.logger._set_static_graph()

    def _build_params_for_reducer(self):
        # Build tuple of (module, parameter) for all parameters that require grads.
        modules_and_parameters = [
            (module, parameter)
            for module_name, module in self.module.named_modules()
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

        # Deduplicate any parameters that might be shared across child modules.
        memo = set()
        modules_and_parameters = [
            # "p not in memo" is the deduplication check.
            # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
            (m, p)
            for m, p in modules_and_parameters
            if p not in memo and not memo.add(p)  # type: ignore[func-returns-value]
        ]

        # Build list of parameters.
        parameters = [parameter for _, parameter in modules_and_parameters]

        # Checks if a module will produce a sparse gradient.
        def produces_sparse_gradient(module):
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                return module.sparse
            return False

        # Build list of booleans indicating whether or not to expect sparse
        # gradients for the corresponding parameters.
        expect_sparse_gradient = [
            produces_sparse_gradient(module) for module, _ in modules_and_parameters
        ]

        self._assign_modules_buffers()

        return parameters, expect_sparse_gradient

    def _assign_modules_buffers(self):
        """
        Assigns module buffers to self.modules_buffers which are then used to
        broadcast across ranks when broadcast_buffers=True. Note that this
        must be called every time buffers need to be synced because buffers can
        be reassigned by user module,
        see https://github.com/pytorch/pytorch/issues/63916.
        """
        # Collect buffers for modules, filtering out buffers that should be ignored.
        named_module_buffers = [
            (buffer, buffer_name)
            for buffer_name, buffer in self.module.named_buffers()
            if buffer_name not in self.parameters_to_ignore
        ]
        self.modules_buffers = [
            buffer for (buffer, buffer_name) in named_module_buffers
        ]
        # Dict[str, tensor] representing module buffers not ignored by DDP.
        self.named_module_buffers = {
            buffer_name: buffer for (buffer, buffer_name) in named_module_buffers
        }

    def _build_debug_param_to_name_mapping(self, parameters):
        param_to_param_index = {parameters[i]: i for i in range(len(parameters))}
        param_set = set(parameters)
        param_index_to_param_fqn = {}
        for module_name, module in self.module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fqn = f"{module_name}.{param_name}"
                # Bypass ignored parameters since those are not reduced by DDP
                # to begin with.
                if fqn not in self.parameters_to_ignore and param.requires_grad:
                    if param not in param_set:
                        self._log_and_throw(
                            ValueError,
                            f"Param with name {fqn} found in module parameters, but not DDP parameters."
                            " This indicates a bug in DDP, please report an issue to PyTorch.",
                        )
                    param_index = param_to_param_index[param]
                    param_index_to_param_fqn[param_index] = fqn

        # Ensure we covered all parameters
        if len(param_set) != len(param_index_to_param_fqn):
            self._log_and_throw(
                ValueError,
                (
                    "Expected param to name mapping to cover all parameters, but"
                    f" got conflicting lengths: {len(param_set)} vs "
                    f"{len(param_index_to_param_fqn)}. This indicates a bug in DDP"
                    ", please report an issue to PyTorch."
                ),
            )

        return param_index_to_param_fqn

    def _get_parameters(self, m, recurse=True):
        """
        Returns a generator of module parameters
        """

        def model_parameters(m):
            ps = (
                m._former_parameters.values()
                if hasattr(m, "_former_parameters")
                else m.parameters(recurse=False)
            )
            yield from ps

        for m in m.modules() if recurse else [m]:
            yield from model_parameters(m)

    def _check_default_group(self):
        pickle_not_supported = False
        try:
            if self.process_group != _get_default_group():
                pickle_not_supported = True
        except RuntimeError:
            pickle_not_supported = True

        if pickle_not_supported:
            self._log_and_throw(
                RuntimeError,
                "DDP Pickling/Unpickling are only supported "
                "when using DDP with the default process "
                "group. That is, when you have called "
                "init_process_group and have not passed "
                "process_group argument to DDP constructor",
            )

    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> ddp = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            >>>     for input in inputs:
            >>>         ddp(input).backward()  # no synchronization, accumulate grads
            >>> ddp(another_input).backward()  # synchronize grads

        .. warning::
            The forward pass should be included inside the context manager, or
            else gradients will still be synchronized.
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync

    @classmethod
    def _get_active_ddp_module(cls):
        """
        TorchDynamo needs to know whether DDP is currently active, and access the DDP module in order to cooperatively optimize it.
        """
        return cls._active_ddp_module

    # note, this ctxmgr function is marked 'skip' in torchdynamo, so dynamo only kicks in
    # for the 'module_to_run' underneath
    # see torch._dynamo/eval_frame.py TorchPatcher.patch for more details
    @contextmanager
    @torch._disable_dynamo(recursive=False)
    def _inside_ddp_forward(self):
        DistributedDataParallel._active_ddp_module = self
        try:
            yield
        finally:
            DistributedDataParallel._active_ddp_module = None

    def _run_ddp_forward(self, *inputs, **kwargs):
        with self._inside_ddp_forward():
            return self.module(*inputs, **kwargs)  # type: ignore[index]

    def _clear_grad_buffer(self):
        # Making param.grad points to the grad buffers before backward is based on the
        # assumption that the grad accumulation is done in place in autograd engine,
        # for some edge cases, if the grad accumulation in autograd engine is not in
        # place, then the param.grad and grad buffers are detached.
        if self._delay_grad_buffer is not None:
            # We batch zero_grad for all params by resetting the whole grad
            # buffer when the grad of all params is set to None.
            all_param_grad_none = all(
                param.grad is None for param in self._delay_all_reduce_params
            )

            for index, param in enumerate(self._delay_all_reduce_params):
                if param.grad is None:
                    param.grad = self._delay_grad_views[index]
                    if not all_param_grad_none:
                        param.grad.zero_()

            if all_param_grad_none:
                self._delay_grad_buffer.zero_()

    def _lazy_init(self):
        # Initialization for DDP that occurs after construction, but lazily
        # before the first forward pass.
        self._setup_in_backward_optimizers()
        self._lazy_init_ran = True

    def _pre_forward(self, *inputs, **kwargs):
        if not self._lazy_init_ran:
            self._lazy_init()
        if self._delay_all_reduce_all_params:
            return inputs, kwargs

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            assert self.logger is not None
            self.logger.set_runtime_stats_and_log()
            self.reducer.prepare_for_forward()

        # Notify the join context that this process has not joined, if
        # needed
        work = Join.notify_join_context(self)
        if work:
            self.reducer._set_forward_pass_work_handle(
                work, self._divide_by_initial_world_size  # type: ignore[arg-type]
            )

        # Calling _rebuild_buckets before forward computation,
        # It may allocate new buckets before deallocating old buckets
        # inside _rebuild_buckets. To save peak memory usage,
        # call _rebuild_buckets before the peak memory usage increases
        # during forward computation.
        # This should be called only once during whole training period.
        if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
            logger.info("Reducer buckets have been rebuilt in this iteration.")
            self._has_rebuilt_buckets = True

        # sync params according to location (before/after forward) user
        # specified as part of hook, if hook was specified.
        if self._check_sync_bufs_pre_fwd():
            self._sync_buffers()

        if self._join_config.enable:
            # Notify joined ranks whether they should sync in backwards pass or not.
            self._check_global_requires_backward_grad_sync(is_joined_rank=False)

        if self.device_ids:
            moved_inputs, moved_kwargs = _to_kwargs(
                inputs,
                kwargs,
                torch.device(self.device_type, self.device_ids[0]),
                self.use_side_stream_for_tensor_copies,
            )
            args, kwargs = moved_inputs[0], moved_kwargs[0]
            # Cast inputs to reduced precision if needed.
            if self.mixed_precision is not None:
                args, kwargs = _cast_forward_inputs(
                    self.mixed_precision.param_dtype,
                    *args,
                    **kwargs,
                )
            return args, kwargs
        else:
            # Cast inputs to reduced precision if needed.
            # TODO (rohan-varma) test this codepath.
            if self.mixed_precision is not None:
                inputs, kwargs = _cast_forward_inputs(
                    self.mixed_precision.param_dtype,
                    *inputs,
                    **kwargs,
                )
            return inputs, kwargs

    def _post_forward(self, output):
        if self._delay_all_reduce_all_params:
            self._clear_grad_buffer()
            return

        # sync params according to location (before/after forward) user
        # specified as part of hook, if hook was specified.
        if self._check_sync_bufs_post_fwd():
            self._sync_buffers()

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters and not self.static_graph:
                # Do not need to populate this for static graph.
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

        # TODO: DDPSink is currently enabled for unused parameter detection and
        # static graph training for first iteration.
        if (self.find_unused_parameters and not self.static_graph) or (
            self.static_graph and not self._static_graph_delay_allreduce_enqueued
        ):
            (
                output_tensor_list,
                treespec,
                output_is_rref,
            ) = _tree_flatten_with_rref(output)
            output_placeholders = [None for _ in range(len(output_tensor_list))]
            # Do not touch tensors that have no grad_fn, which can cause issues
            # such as https://github.com/pytorch/pytorch/issues/60733
            for i, output in enumerate(output_tensor_list):
                if torch.is_tensor(output) and output.grad_fn is None:
                    output_placeholders[i] = output

            # When find_unused_parameters=True, makes tensors which require grad
            # run through the DDPSink backward pass. When not all outputs are
            # used in loss, this makes those corresponding tensors receive
            # undefined gradient which the reducer then handles to ensure
            # param.grad field is not touched and we don't error out.
            passthrough_tensor_list = _DDPSink.apply(
                weakref.ref(self),
                *output_tensor_list,
            )
            for i in range(len(output_placeholders)):
                if output_placeholders[i] is None:
                    output_placeholders[i] = passthrough_tensor_list[i]

            # Reconstruct output data structure.
            output = _tree_unflatten_with_rref(
                output_placeholders, treespec, output_is_rref
            )

        # At the end of the forward pass, reset the grad buffer and grad views
        self._clear_grad_buffer()
        return output

    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
            inputs, kwargs = self._pre_forward(*inputs, **kwargs)
            output = (
                self.module.forward(*inputs, **kwargs)
                if self._delay_all_reduce_all_params
                else self._run_ddp_forward(*inputs, **kwargs)
            )
            return self._post_forward(output)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def to_kwargs(self, inputs, kwargs, device_id):
        # Kept for BC
        return _to_kwargs(
            inputs,
            kwargs,
            torch.device(self.device_type, device_id),
            self.use_side_stream_for_tensor_copies,
        )

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def train(self, mode=True):
        super().train(mode)
        return self

    # When running in join mode, schedules an allreduce to notify joined ranks
    # of whether backwards pass synchronization will run this iteration or not.
    def _check_global_requires_backward_grad_sync(self, is_joined_rank):
        if not is_joined_rank and self.require_backward_grad_sync:
            requires_sync_tensor = torch.ones(1, device=self.device)
        else:
            requires_sync_tensor = torch.zeros(1, device=self.device)

        work = dist.all_reduce(
            requires_sync_tensor, group=self.process_group, async_op=True
        )
        return work

    # When running in join mode, checks and performs sync of module buffers if
    # the models have buffers that should be synchronized in the forward pass.
    def _check_and_sync_module_buffers(self):
        if self._check_sync_bufs_pre_fwd():
            authoritative_rank = self._find_common_rank(self._distributed_rank, False)
            self._sync_module_buffers(authoritative_rank)

    # When running in join model, agrees upon a common rank and broadcast model
    # parameters to all other ranks.
    def _sync_final_model(self, is_last_joiner):
        # Agree upon the process that will be the authoritative model copy.
        # The current rank is a candidate for being the authoritative copy if
        # is_last_joiner=True. We break ties via picking the larger rank.
        self._authoritative_rank = self._find_common_rank(
            self._distributed_rank, is_last_joiner
        )
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=self._authoritative_rank,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
            broadcast_buffers=self.broadcast_buffers,
        )

    # Schedule comm ops to match those scheduled in the reducer's backward
    # pass.
    def _match_all_reduce_for_bwd_pass(self):
        comm_work = []
        # Schedule comm in the same order as Reducer schedules them, i.e.
        # the order of the buckets. Retrieving the bucket order from the reducer
        # ensures that we keep the same order in join mode, such as when bucket
        # order is rebuilt dynamically.

        # Returns grad_buckets in order, but real tensors are substituted with
        # zero tensors of the same shape.
        grad_buckets = self.reducer._get_zeros_like_grad_buckets()
        for grad_bucket in grad_buckets:
            # Joined processes contribute zero gradient. In the case that
            # divide_by_initial_world_size=True, we divide grads by the static
            # world size, if not, the dividing factor is reduced by the number
            # of joined processes.
            work = self.reducer._run_comm_hook(grad_bucket)
            comm_work.append(work)
        for work in comm_work:
            work.wait()

    # Allreduces the used parameter mapping across ranks.
    def _match_unused_params_allreduce(self):
        locally_used_param_map = self.reducer._get_local_used_map()
        self.process_group.allreduce(locally_used_param_map)

    def join(
        self,
        divide_by_initial_world_size: bool = True,
        enable: bool = True,
        throw_on_early_termination: bool = False,
    ):
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
        processes. Alternatively, if the flag ``throw_on_early_termination`` is
        specified to be ``True``, all trainers will throw an error once one rank
        runs out of inputs, allowing these errors to be caught and handled
        according to application logic.

        Once all DDP processes have joined, the context manager will broadcast
        the model corresponding to the last joined process to all processes to
        ensure the model is the same across all processes
        (which is guaranteed by DDP).

        To use this to enable training with uneven inputs across processes,
        simply wrap this context manager around your training loop. No further
        modifications to the model or data loading is required.

        .. warning::
            If the model or training loop this context manager is wrapped around
            has additional distributed collective operations, such as
            ``SyncBatchNorm`` in the model's forward pass, then the flag
            ``throw_on_early_termination`` must be enabled. This is because this
            context manager is not aware of non-DDP collective communication.
            This flag will cause all ranks to throw when any one rank
            exhausts inputs, allowing these errors to be caught and recovered
            from across all ranks.

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
            throw_on_early_termination (bool): Whether to throw an error
                or continue training when at least one rank has exhausted
                inputs. If ``True``, will throw upon the first rank reaching end
                of data. If ``False``, will continue training with a smaller
                effective world size until all ranks are joined. Note that if
                this flag is specified, then the flag
                ``divide_by_initial_world_size`` would be ignored. Default
                is ``False``.


        Example::

            >>> # xdoctest: +SKIP("Distributed")
            >>> import torch
            >>> import torch.distributed as dist
            >>> import os
            >>> import torch.multiprocessing as mp
            >>> import torch.nn as nn
            >>> # On each spawned worker
            >>> def worker(rank):
            >>>     dist.init_process_group("nccl", rank=rank, world_size=2)
            >>>     torch.cuda.set_device(rank)
            >>>     model = nn.Linear(1, 1, bias=False).to(rank)
            >>>     model = torch.nn.parallel.DistributedDataParallel(
            >>>         model, device_ids=[rank], output_device=rank
            >>>     )
            >>>     # Rank 1 gets one more input than rank 0.
            >>>     inputs = [torch.tensor([1]).float() for _ in range(10 + rank)]
            >>>     with model.join():
            >>>         for _ in range(5):
            >>>             for inp in inputs:
            >>>                 loss = model(inp).sum()
            >>>                 loss.backward()
            >>>     # Without the join() API, the below synchronization will hang
            >>>     # blocking for rank 1's allreduce to complete.
            >>>     torch.cuda.synchronize(device=rank)
        """
        return Join(
            [self],
            enable,
            throw_on_early_termination,
            divide_by_initial_world_size=divide_by_initial_world_size,
        )

    def join_hook(
        self,
        **kwargs,
    ):
        r"""
        Returns the DDP join hook, which enables training on uneven inputs by
        shadowing the collective communications in the forward and backward
        passes.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.

        The hook supports the following keyword arguments:
            divide_by_initial_world_size (bool, optional):
                If ``True``, then gradients are divided by the initial world
                size that DDP was launched with.
                If ``False``, then gradients are divided by the effective world
                size (i.e. the number of non-joined processes), meaning that
                the uneven inputs contribute more toward the global gradient.
                Typically, this should be set to ``True`` if the degree of
                unevenness is small but can be set to ``False`` in extreme
                cases for possibly better results.
                Default is ``True``.
        """
        divide_by_initial_world_size = kwargs.get("divide_by_initial_world_size", True)
        return _DDPJoinHook(
            self, divide_by_initial_world_size=divide_by_initial_world_size
        )

    @property
    def join_device(self):
        return self.device

    @property
    def join_process_group(self):
        return self.process_group

    def _register_buffer_comm_hook(
        self,
        state,
        hook: Callable,
        comm_hook_location=_BufferCommHookLocation.POST_FORWARD,
    ):
        r"""
        Allows custom registration of hooks that define how buffer are
        synchronized across ranks. The hook takes in an optional state
        and is passed in a Dict[str, Tensor] corresponding to buffer names
        and the buffers, and can run arbitrary reductions on buffers as
        opposed to DDP's default broadcast from rank 0. This is useful for
        example if a counter needs to be summed or averaged across ranks
        every iteration.

        Args:
            state (Any): Optional state that is passed to the hook.
            hook (Callable): Callable with the following signature:
                         ``hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]``
            comm_hook_location (_BufferCommHookLocation): Enum value indicating
                            where to run the hook.
                            _BufferCommHookLocation.PRE_FORWARD means that the
                            hook will run _before_ the forward pass, and
                            _BufferCommHookLocation.POST_FORWARD means that the
                            hook will run _after_ the forward pass.

            NOTE: To maximize performance, users can return a
                List[torch.futures.Future] from their hook, and DDP will
                install and await these hooks appropriately at the end of
                the backward pass. This will ensure all buffers are
                synchronized by the end of the backward pass. If this
                setting is used, it is recommended to pass
                comm_hook_location=_BufferCommHookLocation.POST_FORWARD,
                which will trigger the hook after the forward pass.
                If _BufferCommHookLocation.PRE_FORWARD is used, users must
                ensure appropriate synchronization when manipulating GPU
                buffers in the forward pass.
        """
        assert callable(hook)
        self.buffer_hook = _BufferCommHook(
            buffer_comm_hook=hook,
            buffer_comm_hook_state=state,
            buffer_comm_hook_location=comm_hook_location,
        )

    def register_comm_hook(self, state: object, hook: Callable):
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
            hook (Callable): Callable with the following signature:
                             ``hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]``:

                             This function is called once the bucket is ready. The
                             hook can perform whatever processing is needed and return
                             a Future indicating completion of any async work (ex: allreduce).
                             If the hook doesn't perform any communication, it still
                             must return a completed Future. The Future should hold the
                             new value of grad bucket's tensors. Once a bucket is ready,
                             c10d reducer would call this hook and use the tensors returned
                             by the Future and copy grads to individual parameters.
                             Note that the future's return type must be a single tensor.

                             We also provide an API called ``get_future`` to retrieve a
                             Future associated with the completion of ``c10d.ProcessGroup.Work``.
                             ``get_future`` is currently supported for NCCL and also supported for most
                             operations on GLOO and MPI, except for peer to peer operations (send/recv).

        .. warning ::
            Grad bucket's tensors will not be predivided by world_size. User is responsible
            to divide by the world_size in case of operations like allreduce.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        .. warning ::
            The Future object that hook returns should contain a single tensor
            that has the same shape with the tensors inside grad bucket.

        .. warning ::
            ``get_future`` API supports NCCL, and partially GLOO and MPI backends (no support
            for peer-to-peer operations like send/recv) and will return a ``torch.futures.Future``.

        Example::
            Below is an example of a noop hook that returns the same tensor.

            >>> # xdoctest: +SKIP('undefined name')
            >>> def noop(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            >>>     fut = torch.futures.Future()
            >>>     fut.set_result(bucket.buffer())
            >>>     return fut
            >>> ddp.register_comm_hook(state=None, hook=noop)

        Example::
            Below is an example of a Parallel SGD algorithm where gradients are encoded before
            allreduce, and then decoded after allreduce.

            >>> # xdoctest: +SKIP('undefined name')
            >>> def encode_and_decode(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            >>>     encoded_tensor = encode(bucket.buffer())  # encode gradients
            >>>     fut = torch.distributed.all_reduce(encoded_tensor).get_future()
            >>>     # Define the then callback to decode.
            >>>     def decode(fut):
            >>>         decoded_tensor = decode(fut.value()[0])  # decode gradients
            >>>         return decoded_tensor
            >>>     return fut.then(decode)
            >>> ddp.register_comm_hook(state=None, hook=encode_and_decode)
        """
        self._check_comm_hook(hook)
        assert self.logger is not None
        self.logger._set_comm_hook_name(hook.__qualname__)
        dist._register_comm_hook(self.reducer, state, hook)

    def _register_builtin_comm_hook(self, comm_hook_type):
        r"""
        Registers a built-in communication hook that specifies how DDP
        aggregates gradients across multiple workers.
        The built-in hooks aim to provide efficient C++ implementations for certain hooks,
        which might not be as efficient if implemented in Python using a Python communication hook.

        Args:
            comm_hook_type (dist.BuiltinCommHookType): type of communication hook, such as ALLREDUCE, FP16_COMPRESS, etc.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        Example::
            Below is an example of a FP16 compression where gradients are
            compressed into 16-bit floating-point numbers before allreduce, and
            then decompressed after allreduce.

            >>> # xdoctest: +SKIP('undefined name')
            >>> ddp._register_builtin_comm_hook(dist.BuiltinCommHookType.FP16_COMPRESS)

        """
        assert self.logger is not None
        self.logger._set_comm_hook_name(str(comm_hook_type))
        dist._register_builtin_comm_hook(self.reducer, comm_hook_type)

    def _register_fused_optim(self, optim: Type, *args, optim_params=None, **kwargs):
        r"""
        Registers an optimizer with DDP such that the optimization for a
        parameter will run immediately when that parameter's gradient is
        finished with reduction, instead of waiting for all parameters'
        gradients to finish reduction. This can result in a training speedup
        depending on your workload since the optimizer can run while gradient
        reduction for other parameters are still ongoing. In addition, this has
        the potential to reduce peak memory consumption during training, as it
        only needs to load the per-parameter optimizer states of a single
        parameter at a time, instead of loading all per-parameter optimizer
        states at once.

        Args:
            optim (Type): a ``torch.optim.Optimizer`` class to be registered
            as a fused optimizer.
            *args (Sequence[Any]): Arguments to forward to `optim`.
            optim_params (Optional[Iterable[torch.Tensor]]): Set of parameters
            to optimize, similar to `params` argument of traditional `torch.optim`
            Optimizers. If this is omitted, all DDP model parameters will be
            optimized.
            **kwargs: (Dict[str, Any]): Keyword arguments to forward to `optim`.

        .. warning ::
            _register_fused_optim should only be called once on a DDP instance,
            and registering multiple fused optimizers for the same DDP model
            is not currently supported. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        .. warning ::
            _register_fused_optim and register_comm_hook currently do not
            compose together, meaning that custom DDP communication hooks are
            not supported with overlapped optimizers. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        .. warning ::
            Gradient accumulation and DDP `no_sync` are currently not supported
            with overlapped optimizer. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        Example::

            >>> # xdoctest: +SKIP("No rendezvous handler")
            >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
            >>> net = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> lr = 1e-2
            >>> betas = (0.9, 0.99)
            >>> eps = 1e-6
            >>> net._register_fused_optim(torch.optim.Adam, lr, betas=betas, eps=eps)
            >>> # Example with subset of parameters
            >>> params_to_opt = [list(net.parameters())[0]]
            >>> net._register_fused_optim(
            ...   torch.optim.Adam, lr, optim_params=params_to_opt,  betas=betas, eps=eps
            ... )
        """
        # Note: importing in function, otherwise this will cause a circular
        # import as optimizer_overlap module needs to import DistributedDataParallel.
        from torch.distributed.algorithms._optimizer_overlap import _as_overlapped_optim

        overlapped_optim = _as_overlapped_optim(optim, optim_params, *args, **kwargs)
        try:
            overlapped_optim.register_ddp(self)
        except NotImplementedError as e:
            raise RuntimeError(
                f"{optim} does not support overlapped DDP. Please file an issue to PyTorch or the respective owner of {optim}."
            ) from e

    def _distributed_broadcast_coalesced(
        self, tensors, buffer_size, authoritative_rank=0
    ):
        dist._broadcast_coalesced(
            self.process_group, tensors, buffer_size, authoritative_rank
        )

    def _check_sync_bufs_post_fwd(self):
        return (
            self.will_sync_module_buffers()
            and hasattr(self, "buffer_hook")
            and self.buffer_hook.buffer_comm_hook_location
            == _BufferCommHookLocation.POST_FORWARD
        )

    def _check_sync_bufs_pre_fwd(self):
        return self.will_sync_module_buffers() and (
            not hasattr(self, "buffer_hook")
            or self.buffer_hook.buffer_comm_hook_location
            == _BufferCommHookLocation.PRE_FORWARD
        )

    def will_sync_module_buffers(self):
        return (
            self.require_forward_param_sync
            and self.broadcast_buffers
            and len(self.modules_buffers) > 0
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
            self._log_and_throw(
                ValueError,
                "BUG! Expected rank_cond to be true for at least one process."
                " This indicates a bug in PyTorch, please report an issue.",
            )
        return rank_to_use.item()

    def _sync_buffers(self):
        with torch.no_grad():
            # module buffer sync
            # Synchronize buffers across processes.
            # If we are running DDP with the join manager, we have to agree
            # upon a rank to sync module buffers from, since rank 0 may
            # already have been joined and have stale module buffers.
            if self._join_config.enable:
                authoritative_rank = self._find_common_rank(
                    self._distributed_rank, True
                )
            else:
                # The process with rank 0 is considered the authoritative copy.
                authoritative_rank = 0
            # Update self.modules_buffers incase any buffers were
            # reassigned.
            self._assign_modules_buffers()
            self._sync_module_buffers(authoritative_rank)

    def _sync_module_buffers(self, authoritative_rank):
        if not hasattr(self, "buffer_hook"):
            self._default_broadcast_coalesced(authoritative_rank=authoritative_rank)
        else:
            hook = self.buffer_hook.buffer_comm_hook
            state = self.buffer_hook.buffer_comm_hook_state
            futs = hook(state, self.named_module_buffers)
            if futs is not None:
                self.reducer._install_post_backward_futures(futs)

    def _default_broadcast_coalesced(
        self, bufs=None, bucket_size=None, authoritative_rank=0
    ):
        """
        Broadcasts buffers from rank 0 to rest of workers. If bufs, bucket_size
        are None, default values self.modules_buffers and
        self.broadcast_bucket_size are used instead.
        """
        if bufs is None:
            bufs = self.modules_buffers
        if bucket_size is None:
            bucket_size = self.broadcast_bucket_size

        self._distributed_broadcast_coalesced(bufs, bucket_size, authoritative_rank)

    def _passing_sync_batchnorm_handle(self, module):
        for layer in module.modules():
            if isinstance(layer, torch.nn.modules.SyncBatchNorm):
                if self.device_type == "cpu":
                    self._log_and_throw(
                        ValueError,
                        "SyncBatchNorm layers only work with GPU modules",
                    )

    def _check_comm_hook(self, hook):
        if not callable(hook):
            self._log_and_throw(TypeError, "Communication hook must be callable.")

        sig = inspect.signature(hook)
        if (
            sig.parameters["bucket"].annotation != inspect._empty
            and sig.parameters["bucket"].annotation != dist.GradBucket
        ):
            self._log_and_throw(
                ValueError,
                "Communication hook: bucket annotation should be dist.GradBucket.",
            )

        if (
            sig.return_annotation != inspect._empty
            and sig.return_annotation != torch.futures.Future[torch.Tensor]
        ):
            self._log_and_throw(
                ValueError,
                "Communication hook: return annotation should be torch.futures.Future[torch.Tensor].",
            )

        if hook.__name__ in [
            "bf16_compress_hook",
            "bf16_compress_wrapper_hook",
        ] and (
            (torch.version.cuda is None and torch.version.hip is None)
            or (
                torch.version.cuda is not None
                and int(torch.version.cuda.split(".")[0]) < 11
            )
            or not dist.is_available()
            or not dist.is_nccl_available()
            or torch.cuda.nccl.version() < (2, 10)
        ):
            self._log_and_throw(
                TypeError,
                "BF16 all reduce communication hook required CUDA 11+ and NCCL 2.10+.",
            )

    @property
    def _distributed_rank(self):
        return dist.get_rank(self.process_group)

    @staticmethod
    def _get_data_parallel_params(module, named_params=False):
        """
        Returns a generator of parameters managed by a given DDP unit.
        """
        for param in (
            module.parameters() if not named_params else module.named_parameters()
        ):
            if not hasattr(param, "_ddp_ignored"):
                yield param

    @staticmethod
    def _set_params_and_buffers_to_ignore_for_model(
        module, params_and_buffers_to_ignore
    ):
        """
        Sets parameters and buffers to be ignored by DDP. Expected format for
        parameters is the fully qualified name: {module_name}.{param_name}, and
        similarly, {module_name}.{buffer_name} for buffers. For example:
        params_to_ignore = []
        # NB: model here is vanilla PyTorch module, not yet wrapped with DDP.
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if should_ignore(param):
                    # Create expected format
                    fqn = f"{module_name}.{param_name}"
                    params_to_ignore.append(fqn)
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            model,
            params_to_ignore
        )
        """
        # This is a workaround to set parameters and buffers DDP should ignore
        # during synchronization. It will be removed when the API is finalized
        # as part of addressing https://github.com/pytorch/pytorch/issues/43690.
        module._ddp_params_and_buffers_to_ignore = params_and_buffers_to_ignore
        for name, param in module.named_parameters():
            if name in params_and_buffers_to_ignore:
                param._ddp_ignored = True
        for name, buffer in module.named_buffers():
            if name in params_and_buffers_to_ignore:
                buffer._ddp_ignored = True

    def _get_ddp_logging_data(self):
        r"""
        This interface can be called after DistributedDataParallel() is
        constructed. It returns a dictionary of logging data. It could help
        for debugging and analysis. The logging data includes DistributedDataParallel
        constructor input parameters, some internal states of DistributedDataParallel
        and performance metrics. Simply print the dictionary and see what
        these metrics are.
        This is a prototype interface and subject to change in the future.
        """
        assert self.logger is not None
        ddp_logging_data = self.logger._get_ddp_logging_data()
        return {**ddp_logging_data.strs_map, **ddp_logging_data.ints_map}

    def _set_ddp_runtime_logging_sample_rate(self, sample_rate):
        r"""
        This interface allows users to set sample_rate of collecting
        runtime stats. The runtime stats will be recorded for the
        first 10 iterations, after 10 iterations runtime stats will be
        recorded once every "sample_rate" training iterations. In
        default, runtime stats are recorded for the first 10 iterations,
        after 10 iterations runtime stats are recorded once every
        "kDDPRuntimeLoggingSampleRate=100" training iterations.
        This is a prototype interface and subject to change in the future.
        """
        if sample_rate < 1:
            self._log_and_throw(
                ValueError,
                "DDP runtime logging sample rate should be equal or greater than 1",
            )
        self.reducer._set_ddp_runtime_logging_sample_rate(sample_rate)

    def _set_static_graph(self):
        """
        It is recommended to set static graph in the DDP constructor, which will
        call this private API internally.
        """
        # If self.static_graph has been set, no need to set it again
        if self.static_graph:
            warnings.warn(
                "You've set static_graph to be True, no need to set it again.",
                stacklevel=2,
            )
            return
        self.static_graph = True
        self._static_graph_delay_allreduce_enqueued = False
        self.reducer._set_static_graph()
        assert self.logger is not None
        self.logger._set_static_graph()
        if self.find_unused_parameters:
            warnings.warn(
                "You passed find_unused_parameters=true to DistributedDataParallel, "
                "`_set_static_graph` will detect unused parameters automatically, so "
                "you do not need to set find_unused_parameters=true, just be sure these "
                "unused parameters will not change during training loop while calling "
                "`_set_static_graph`.",
                stacklevel=2,
            )

    def _remove_autograd_hooks(self):
        """
        Removes autograd hooks registered by the reducer on the model parameters.
        """
        self.reducer._remove_autograd_hooks()

    def _check_reducer_finalized(self):
        """
        Checks if the reducer has processed all buckets and finalized the backward
        appropriately.

        It is useful to call this method after calling .backward() in your training loop
        in order to avoid subsequent hard to debug errors down the road due to the
        reducer not finalizing backward.
        """
        self.reducer._check_reducer_finalized()

    def _set_sparse_metadata(self, global_unique_ids):
        self.reducer._set_sparse_metadata(global_unique_ids)
