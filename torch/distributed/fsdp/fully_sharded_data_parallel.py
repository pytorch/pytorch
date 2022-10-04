import collections
import contextlib
import copy
import functools
import itertools
import math
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
import torch.distributed as dist
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributed import ProcessGroup
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    init_from_local_shards,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
)
from torch.distributed.algorithms._comm_hooks import (
    LOW_PRECISION_HOOKS,
    default_hooks,
)
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.utils import (
    _replace_by_prefix,
    _sync_params_and_buffers,
    _to_kwargs,
)
from torch.nn.parameter import Parameter

from ._optim_utils import (
    _broadcast_pos_dim_tensor_states,
    _broadcast_processed_optim_state_dict,
    _flatten_optim_state_dict,
    _get_param_id_to_param,
    _get_param_id_to_param_from_optim_input,
    _get_param_to_param_id,
    _get_param_to_param_id_from_optim_input,
    _optim_state_dict,
    _process_pos_dim_tensor_state,
    _rekey_sharded_optim_state_dict,
)
from ._fsdp_extensions import _ext_chunk_tensor, _ext_pre_load_state_dict_transform
from ._utils import (
    _apply_to_modules,
    _apply_to_tensors,
    _contains_batchnorm,
    _free_storage,
    _is_fsdp_flattened,
    _override_batchnorm_mixed_precision,
    p_assert,
)
from .flat_param import (
    FlatParameter,
    FlatParamHandle,
    HandleConfig,
    HandleShardingStrategy,
    HandleTrainingState,
)
from .flatten_params_wrapper import (
    FLAT_PARAM,
    FPW_MODULE,
    FlattenParamsWrapper,
)
from .wrap import (
    ParamExecOrderWrapPolicy,
    _or_policy,
    _recursive_wrap,
    _wrap_batchnorm_individually,
)

_TORCHDISTX_AVAIL = True
try:
    from torchdistx import deferred_init, fake
except ImportError:
    _TORCHDISTX_AVAIL = False

_TORCH_FX_AVAIL = True
if not hasattr(torch, "fx"):
    _TORCH_FX_AVAIL = False
if _TORCH_FX_AVAIL:
    from ._symbolic_trace import (
        TracingConfig,
        _init_execution_info,
        _patch_tracer,
    )


__all__ = [
    "FullyShardedDataParallel", "ShardingStrategy", "MixedPrecision",
    "CPUOffload", "BackwardPrefetch", "StateDictType", "StateDictConfig",
    "FullStateDictConfig", "LocalStateDictConfig", "ShardedStateDictConfig",
    "OptimStateKeyType", "TrainingState_", "clean_tensor_name",
]


FSDP_WRAPPED_MODULE = "_fsdp_wrapped_module"
FSDP_PREFIX = FSDP_WRAPPED_MODULE + "." + FPW_MODULE + "."

_PARAM_BROADCAST_BUCKET_SIZE = int(250 * 1024 * 1024)


class ShardingStrategy(Enum):
    """
    This specifies the sharding strategy to be used for distributed training by
    :class:`FullyShardedDataParallel`.
    FULL_SHARD: Parameters, gradients, and optimizer states are sharded. For
                the parameters, this algorithm all-gathers before the forward,
                reshards after the forward, all-gathers before the backward
                computation, and reshards after the backward computation. The
                gradients are synchronized and sharded via reduce-scatter after
                the backward computation. The sharded optimizer states are
                updated locally.
    SHARD_GRAD_OP: Gradients and optimizer states are sharded during
                   computation, and additionally parameters are sharded outside
                   computation. For the parameters, this algorithm all-gathers
                   before the forward, does not reshard after the forward, and
                   only reshards after the backward computation. The gradients
                   are synchronized and sharded via reduce-scatter after the
                   backward computation. The sharded optimizer states are
                   updated locally. Inside ``no_sync()``, the parameters are
                   not resharded after the backward computation.
    NO_SHARD: Parameters, gradients, and optimizer states are not sharded but
              instead replicated across ranks, similar to PyTorch's
              ``DistributedDataParallel`` API. The gradients are synchronized
              via all-reduce after the backward computation. The unsharded
              optimizer states are updated locally.
    HYBRID_SHARD(future support): Apply ``FULL_SHARD`` intra-node and
                                  ``NO_SHARD`` inter-node.

    """
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    # TODO
    # HYBRID_SHARD = auto()


@dataclass
class MixedPrecision:
    """
    A config to enable mixed precision training with FullyShardedDataParallel.
    This class can be constructed with several flags:
        ``param_dtype`` controls the precision of model parameters, inputs, and
        therefore the precision under which computation happens. After forward
        and backward passes, FSDP parameters point to full precision shards
        that are kept in memory. Full precision parameters are always
        checkpointed.
        ``reduce_dtype`` controls the precision under which gradient reduction
        would occur, which can potentially be different than ``param_dtype``
        for use cases such as communication efficiency.
        ``buffer_dtype`` controls the precision that buffers are cast to. Note
        that buffers are unsharded and are cast in the first forward pass, and
        remain in their reduced precision state even after forward/backward
        passes. However, when taking checkpoints with ``state_dict``, buffers
        are checkpointed in their full precision (and then restored back to
        to their reduced precision) as expected. Note that this checkpoint
        support is currently limited to ``StateDictType.FULL_STATE_DICT``.
        ``keep_low_precision_grads``: Whether to upcast gradients back to the
        full parameter precision after backwards or not. This can be disabled
        to keep the gradients in the lower precision, which can potentially
        save memory if custom Optimizers are able to perform parameter updates
        effectively with lower precision grads.

    .. note:: In ``summon_full_params``, parameters are summoned in full
        precision but buffers are not.

    .. note:: Parameters and buffers are checkpointed in full precision. For
        buffers, this is only guaranteed to work for ``StateDictType.FULL_STATE_DICT``.

    .. note:: This API is experimental and subject to change.

    .. note:: Specification of reduced precision types must be explicit, in that
        if, for example, ``param_dtype`` is not specified, it will not be cast by
        FSDP. Thus, a config such as ``MixedPrecision(reduce_dtype=torch.float16)``
        will not cast buffers or parameters. Note that if a ``MixedPrecision``
        config is specified without a ``reduce_dtype``, gradient communication
        would occur in the `param_dtype` precision, if given, otherwise, in the
        original parameter precision.
    """
    # maintain a tensor of this dtype that the fp32 param shard will be cast to.
    # Will control the precision of model params, inputs, and thus compute as
    # well.
    param_dtype: Optional[torch.dtype] = None
    # Gradient communication precision.
    reduce_dtype: Optional[torch.dtype] = None
    # Buffer precision.
    # TODO: buffer + param are usually of the same type, if user specifies
    # param but not buffer, should we automatically make buffer be the same?
    buffer_dtype: Optional[torch.dtype] = None
    keep_low_precision_grads: Optional[bool] = False


@dataclass
class CPUOffload:
    """
    CPU offloading config. Currently, only parameter and gradient CPU
    offload are supported.
    offload_params: Offloading parameters to CPUs when these parameters are
                    not used for computation on GPUs. This implicitly enables
                    gradient offloading to CPUs in order for parameters and
                    gradients to be on the same device to work with optimizer.
    """

    offload_params: bool = False


class BackwardPrefetch(Enum):
    """
    Specify where to prefetch next layer's full parameters
    during backward pass.
    BACKWARD_PRE: prefetch right before current layer's backward computation
                  starts, this approach will increase backward communication
                  and computation overalpping and potentialy improve training
                  performance, but it may increase the peak memory usage as
                  the prefetched full parameters will be kept in the GPU memory
                  until next layer's backward computation is done.
    BACKWARD_POST: prefetch right after current layer's backward computation finishes,
                   this approach will not increase peak memory as prefetching happens
                   after current layer's full parameters are freed.
                   It could potentially improve backward communication and computation
                   overlapping as it avoids all_gather and reduce_scatter are blocked
                   each other in the single NCCL stream. However, based on our experiments,
                   for some models, the backward post backward hook fire order is not always
                   the reversed forward computation order, so this
                   approach may prefetch full parameters for layers ahead of next layer,
                   this 'ahead' all_gather could delay next layer's all_gather in the
                   single NCCL stream and cause the next layer's computation delay. So it may
                   cause some performance regession for some models.
    """

    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    # TODO, BACKWARD_PRE_CPU, prefetch full parameters and keep them in the CPU memory


class TrainingState_(Enum):
    """
    Simple enum to indicate what state FSDP is in. Used for asserting
    to make sure APIs are called in the correct state.
    ..note::
        ``BACKWARD_PRE`` and ``BACKWARD_POST`` states are used to ensure we
        receives backward hooks in the correct order. It is used to catch
        unexpected order of hooks being called (likely due to our
        hook registration logic or autograd engine logic changes).
    """

    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


class StateDictType(Enum):
    """
    This enum indicates that which type of ``state_dict`` the FSDP module is
    currently processing (returning or loading).
    The default value is FULL_STATE_DICT to comply the PyTorch convention.
    ..note::
        FSDP currently supports three types of ``state_dict``:
            1. ``state_dict/load_state_dict`: this pair of APIs return and load
               the non-sharded, unflattened parameters. The semantics is the
               same as using DDP.
            2. ``_local_state_dict/_load_local_state_dict``: this pair of APIs return
               and load local sharded, flattened parameters. The values returned
               by ``_local_state_dict`` can be directly used by FSDP and is only
               meaningful to FSDP (because parameters are flattened). Note that
               these APIs are meant for use via the :func:`state_dict_type`
               context manager as follows:
                   >>> # xdoctest: +SKIP("undefined variables")
                   >>> with fsdp.state_dict_type(StateDictType.LOCAL_STATE_DICT):
                   ...     state = fsdp.state_dict()  # loads local state dict
            3. ``_sharded_state_dict/_load_sharded_state_dict``: this pair of APIs
               return and load sharded, unflattened parameters. The ``state_dict``
               return by ``sharded_state_dict`` can be used by all other parallel
               schemes (resharding may be required).
    """

    FULL_STATE_DICT = auto()
    LOCAL_STATE_DICT = auto()
    SHARDED_STATE_DICT = auto()

@dataclass
class StateDictConfig:
    """
    ``StateDictConfig`` is the base class for all state_dict configuration classes.
    Users should instantiate a child version (i.e. ``FullStateDictConfig``) in
    order to configure settings for the particular type of ``state_dict``
    implementation FSDP will use.
    """
    pass

@dataclass
class FullStateDictConfig(StateDictConfig):
    """
    ``FullStateDictConfig`` is a config class meant to be used with
    ``StateDictType.FULL_STATE_DICT``. Currently, it accepts two parameters,
    ``offload_to_cpu`` and ``rank0_only`` which can be configured to offload
    the full ``state_dict`` to CPU and to materialize the ``state_dict`` on
    rank 0 only. When used, it is recommended to enable both of these flags
    together to optimize memory savings when taking checkpoints. Note that
    this config class is meant for user via the :func:`state_dict_type`
    context manager as follows:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> fsdp = FSDP(model, auto_wrap_policy=...)
        >>> cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        >>> with FullyShardedDataParallel.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, cfg):
        >>>     state = fsdp.state_dict()
        >>>     # state will be empty on non rank 0 and contain CPU tensors on rank 0.
        >>> # To reload checkpoint for inference, finetuning, transfer learning, etc:
        >>> model = model_fn() # Initialize model on CPU in preparation for wrapping with FSDP
        >>> if dist.get_rank() == 0:
        >>>     # Load checkpoint only on rank 0 to avoid memory redundancy
        >>>     state_dict = torch.load("my_checkpoint.pt")
        >>>     model.load_state_dict(state_dict)
        >>> # All ranks initialize FSDP module as usual. ``sync_module_states`` argument
        >>> # communicates loaded checkpoint states from rank 0 to rest of the world.
        >>> fsdp = FSDP(model, device_id=torch.cuda.current_device(), auto_wrap_policy=..., sync_module_states=True)
        >>> # After this point, all ranks have FSDP model with loaded checkpoint.
    """
    offload_to_cpu: bool = False
    rank0_only: bool = False

@dataclass
class LocalStateDictConfig(StateDictConfig):
    pass

@dataclass
class ShardedStateDictConfig(StateDictConfig):
    pass

_state_dict_type_to_config = {
    StateDictType.FULL_STATE_DICT: FullStateDictConfig,
    StateDictType.LOCAL_STATE_DICT: LocalStateDictConfig,
    StateDictType.SHARDED_STATE_DICT: ShardedStateDictConfig,
}

class OptimStateKeyType(Enum):
    PARAM_NAME = auto()
    PARAM_ID = auto()


# A handles key represents the group of `FlatParamHandle`s involved in a given
# module's forward. These will be all-gathered together in the pre-forward and
# pre-backward.
_HandlesKey = Tuple[FlatParamHandle, ...]


class _ExecOrderWarnStatus(Enum):
    """Used internally for execution order validation."""
    NONE = auto()     # no deviation yet
    WARNING = auto()  # deviated this iteration; currently issuing warnings
    WARNED = auto()   # deviated in a previous iteration


class _ExecOrderData:
    """
    This contains the data structures to track the execution order. We track
    the pre-forward order on the *first* iteration for forward prefetching
    (which thus assumes static graph) and the post-forward order on *every*
    iteration for backward prefetching (which thus does not assume static
    graph but may be provide an incorrect order).
    """

    def __init__(
        self,
        debug_level: dist.DebugLevel,
        backward_prefetch_limit: int,
        forward_prefetch_limit: int,
    ) -> None:
        # Tracks the (static) pre-forward order for execution order validation
        # and forward prefetching
        self.handles_pre_forward_order: List[int] = []
        # Maps each handles key to its index in `handles_pre_forward_order`
        self.handles_to_pre_forward_order_index: Dict[_HandlesKey, int] = {}
        # Tracks the post-forward order for pre-backward prefetching
        self.handles_post_forward_order: List[int] = []
        # Maps each handles key to its index in `handles_post_forward_order`
        self.handles_to_post_forward_order_index: Dict[_HandlesKey, int] = {}
        self.is_first_iter = True

        # Gives the max number of backward/forward prefetched all-gathers by a
        # single module
        self._backward_prefetch_limit = backward_prefetch_limit
        self._forward_prefetch_limit = forward_prefetch_limit

        # Data structures for execution order validation
        self._checking_order: bool = (
            debug_level in [dist.DebugLevel.INFO, dist.DebugLevel.DETAIL]
        )
        self.process_group: Optional[dist.ProcessGroup] = None
        self.world_size: Optional[int] = None
        self.all_handles: List[FlatParamHandle] = []
        # Maps each handle to its index in `all_handles`, which must be the
        # same across ranks for the execution order validation to work
        self.handle_to_handle_index: Dict[FlatParamHandle, int] = {}
        # Names are prefixed from the root module
        self.flat_param_to_prefixed_param_names: Dict[FlatParameter, List[str]] = {}
        # Current index in the pre-forward execution order
        self.current_order_index = 0
        self.warn_status = _ExecOrderWarnStatus.NONE

    def init(
        self,
        fsdp_root: "FullyShardedDataParallel",
        process_group: dist.ProcessGroup,
    ) -> None:
        """
        Initializes the data structures needed for checking the forward order.
        This should be called after a root FSDP instance has been set during
        lazy initialization.
        """
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        # Fix an order over the handles, which should be the same across ranks
        for fsdp_module in fsdp_root.fsdp_modules(fsdp_root):
            for handle in fsdp_module._handles:
                index = len(self.all_handles)
                self.all_handles.append(handle)
                self.handle_to_handle_index[handle] = index
        self.flat_param_to_prefixed_param_names = cast(
            Dict[FlatParameter, List[str]],
            _get_param_to_unflat_param_names(fsdp_root),
        )
        # TODO (awgu): We can broadcast the metadata of rank 0's `all_handles`
        # to check that all ranks have the same handles in the same order.
        # https://github.com/pytorch/pytorch/issues/79620

    def get_handles_to_backward_prefetch(
        self,
        current_handles_key: _HandlesKey,
    ) -> List[_HandlesKey]:
        """
        Returns a :class:`list` of the handles keys of the handles to backward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = self.handles_to_post_forward_order_index.get(current_handles_key, None)
        if current_index is None:
            return None
        target_index = current_index - 1
        target_handles_keys: List[_HandlesKey] = []
        for _ in range(self._backward_prefetch_limit):
            if target_index < 0:
                break
            target_handles_keys.append(
                self.handles_post_forward_order[target_index]
            )
            target_index -= 1
        return target_handles_keys

    def get_handles_to_forward_prefetch(
        self,
        current_handles_key: _HandlesKey,
    ) -> List[_HandlesKey]:
        """
        Returns a :class:`list` of the handles keys of the handles to forward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = self.handles_to_pre_forward_order_index.get(current_handles_key, None)
        if current_index is None:
            return None
        target_index = current_index + 1
        target_handles_keys: List[_HandlesKey] = []
        for _ in range(self._forward_prefetch_limit):
            if target_index >= len(self.handles_pre_forward_order):
                break
            target_handles_keys.append(
                self.handles_pre_forward_order[target_index]
            )
            target_index += 1
        return target_handles_keys

    def record_post_forward(self, handles: List[FlatParamHandle]) -> None:
        """
        Records ``handles`` in the post-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        Unlike :meth:`record_pre_forward`, this records the order *every*
        iteration with the expectation that the recorded order is reset in
        :meth:`next_iter`.
        """
        if not handles:
            return
        handles_key = tuple(handles)
        # Only record the first usage of a handles key
        if handles_key in self.handles_to_post_forward_order_index:
            return
        index = len(self.handles_post_forward_order)
        self.handles_to_post_forward_order_index[handles_key] = index
        self.handles_post_forward_order.append(handles_key)

    def record_pre_forward(self, handles: List[FlatParamHandle], is_training: bool) -> None:
        """
        Records ``handles`` in the pre-forward order on the first iteration,
        where ``handles`` should be a group of handles used in the same
        module's forward. If ``handles`` is empty, then it is omitted.

        On the first iteration, this checks the execution order across ranks.
        See :meth:`_check_order` for details.
        """
        if not handles:
            return
        handles_key = tuple(handles)
        self._check_order(handles_key, is_training)
        # Fix the order after the first iteration and only record the first
        # usage of a handles key
        if (
            not self.is_first_iter
            or handles_key in self.handles_to_pre_forward_order_index
        ):
            return
        index = len(self.handles_pre_forward_order)
        self.handles_to_pre_forward_order_index[handles_key] = index
        self.handles_pre_forward_order.append(handles_key)

    def _check_order(self, handles_key: _HandlesKey, is_training: bool) -> None:
        """
        Checks the forward execution order as long as ``is_training`` is
        ``True`` since checking in eval mode is not supported.

        - On the first iteration, this uses all-gathers to check that all ranks
        are all-gathering the same handles and hence ``FlatParameter`` s,
        raising an error if not.
        - On subsequent iterations, if the distributed debug level is at least
        INFO, then this checks that each rank is locally consistent with its
        own forward order from the first iteration, issuing a warning if not.
        This issues a warning on the first deviating iteration and stops
        warning thereafter.
        """
        # Do not check order in eval mode since the post-backward callback does
        # not run so it cannot be used to mark the end of an iteration
        if not is_training:
            return
        if self.is_first_iter:
            msg_prefix = "Forward order differs across ranks:"
            local_indices: Optional[Tuple[int, ...]] = self._get_handle_indices(
                handles_key
            )
            device = handles_key[0].device  # guaranteed to be non-CPU
            num_valid_indices = sum((index is not None) for index in local_indices)
            tensor_kwargs = {"dtype": torch.int32, "device": device}
            world_num_valid_indices = torch.zeros(self.world_size, **tensor_kwargs)
            local_num_valid_indices = torch.tensor([num_valid_indices], **tensor_kwargs)
            dist._all_gather_base(
                world_num_valid_indices,
                local_num_valid_indices,
                group=self.process_group,
            )
            # Check that all ranks plan to all-gather the same number of
            # parameters
            # TODO (awgu): Since every module has at most one handle in the
            # current implementation, this should never raise the error.
            for (r1, n1), (r2, n2) in itertools.combinations(
                (
                    (rank, world_num_valid_indices[rank])
                    for rank in range(self.world_size)
                ),
                2,
            ):
                if n1 != n2:
                    raise RuntimeError(
                        f"{msg_prefix} rank {r1} is all-gathering {n1} parameters "
                        f"while rank {r2} is all-gathering {n2} parameters"
                    )
            world_indices = torch.zeros(
                self.world_size * num_valid_indices, **tensor_kwargs
            )
            local_indices = torch.tensor(local_indices, **tensor_kwargs)
            dist._all_gather_base(
                world_indices, local_indices, group=self.process_group
            )
            # Check that all ranks plan to all-gather the same index parameters
            for (r1, i1), (r2, i2) in itertools.combinations(
                (
                    (
                        rank,
                        world_indices[
                            rank * num_valid_indices : (rank + 1) * num_valid_indices
                        ],
                    )
                    for rank in range(self.world_size)
                ),
                2,
            ):
                if i1 != i2:
                    r1_param_names = self._get_names_from_handle_indices(i1)
                    r2_param_names = self._get_names_from_handle_indices(i2)
                    raise RuntimeError(
                        f"{msg_prefix} rank {r1} is all-gathering parameters "
                        f"for {r1_param_names} while rank {r2} is all-gathering "
                        f"parameters for {r2_param_names}"
                    )
        elif self._checking_order:
            # Only issue warnings on the first deviating iteration and stop
            # checking thereafter to avoid flooding the console
            if self.warn_status == _ExecOrderWarnStatus.WARNED:
                return
            msg_prefix = None  # non-`None` means we should warn
            if self.current_order_index >= len(self.handles_pre_forward_order):
                # This iteration sees extra all-gather(s) compared to the first
                msg_prefix = (
                    "Expected to not all-gather any more parameters in the "
                    "forward but trying to all-gather parameters for "
                )
            else:
                expected_handles_key = self.handles_pre_forward_order[
                    self.current_order_index
                ]
                if expected_handles_key != handles_key:
                    expected_param_names = self._get_names_from_handles(
                        expected_handles_key
                    )
                    msg_prefix = (
                        f"Expected to all-gather for {expected_param_names} "
                        "but trying to all-gather parameters for "
                    )
            if msg_prefix is not None:
                param_names = self._get_names_from_handles(handles_key)
                msg_suffix = (
                    f"{param_names}"
                    if param_names
                    else "a newly-added parameter since construction time"
                )
                warnings.warn(
                    "Forward order differs from that of the first iteration "
                    f"on rank {self.rank}. Collectives are unchecked and may "
                    f"give incorrect results or hang.\n{msg_prefix}{msg_suffix}"
                )
                self.warn_status = _ExecOrderWarnStatus.WARNING
            self.current_order_index += 1

    def _get_handle_indices(
        self,
        handles_key: _HandlesKey,
    ) -> Tuple[Optional[int], ...]:
        """
        Returns the handle indices (i.e. indices into ``self.all_handles``)
        corresponding to the handles in ``handles_key``. An entry in the
        returned tuple is ``None`` if the handle is invalid.
        """
        indices: List[int] = []
        for handle in handles_key:
            if handle not in self.handle_to_handle_index:
                indices.append(None)
            else:
                indices.append(self.handle_to_handle_index[handle])
        return tuple(indices)

    def _get_names_from_handle_indices(
        self,
        handle_indices: Tuple[int, ...],
    ) -> List[List[str]]:
        """
        Returns a list of prefixed parameter names for each handle in
        ``handle_indices``. If a handle index is invalid, then its prefixed
        parameter names are omitted from the returned list.
        """
        prefixed_param_names: List[List[str]] = []
        for index in handle_indices:
            if index is None or index < 0 or index >= len(self.all_handles):
                continue
            handle = self.all_handles[index]
            flat_param = handle.flat_param
            prefixed_param_names.append(self.flat_param_to_prefixed_param_names[flat_param])
        return prefixed_param_names

    def _get_names_from_handles(
        self,
        handles_key: _HandlesKey,
    ) -> List[List[str]]:
        """
        Returns a list of prefixed parameter names for each handle in
        ``handles_key``. If a handle is invalid, then its prefixed parameter
        names are omitted from the returned list.
        """
        prefixed_param_names: List[List[str]] = []
        for handle in handles_key:
            flat_param = handle.flat_param
            if flat_param not in self.flat_param_to_prefixed_param_names:
                continue
            prefixed_param_names.append(self.flat_param_to_prefixed_param_names[flat_param])
        return prefixed_param_names

    def next_iter(self):
        """
        Advances the internal data structures per iteration. This should be
        called in the post-backward callback since that marks the true end of
        an iteration.
        """
        self.is_first_iter = False
        self.handles_to_post_forward_order_index.clear()
        self.handles_post_forward_order.clear()
        if self._checking_order:
            self.current_order_index = 0
            if self.warn_status == _ExecOrderWarnStatus.WARNING:
                self.warn_status = _ExecOrderWarnStatus.WARNED


class _FreeEventQueue:
    """
    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """

    def __init__(self) -> None:
        self._queue: Deque[torch.cuda.Event] = collections.deque()
        self._max_num_inflight_all_gathers = 2  # empirically chosen

    def enqueue(self, free_event: torch.cuda.Event) -> None:
        """Enqueues a free event."""
        self._queue.append(free_event)

    def dequeue_if_needed(self) -> Optional[torch.cuda.Event]:
        """Dequeues a single event if the limit is reached."""
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self._dequeue()
        return None

    def _dequeue(self) -> Optional[torch.cuda.Event]:
        """Dequeues a free event if possible."""
        if self._queue:
            event = self._queue.popleft()
            return event
        return None


# TODO (awgu): Refactor this later
sharding_strategy_map = {
    ShardingStrategy.NO_SHARD: HandleShardingStrategy.NO_SHARD,
    ShardingStrategy.FULL_SHARD: HandleShardingStrategy.FULL_SHARD,
    ShardingStrategy.SHARD_GRAD_OP: HandleShardingStrategy.SHARD_GRAD_OP,
}


class FullyShardedDataParallel(nn.Module):
    """
    A wrapper for sharding Module parameters across data parallel workers. This
    is inspired by `Xu et al.`_ as well as the ZeRO Stage 3 from DeepSpeed_.
    FullyShardedDataParallel is commonly shortened to FSDP.

    .. _`Xu et al.`: https://arxiv.org/abs/2004.13336
    .. _DeepSpeed: https://www.deepspeed.ai/

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> torch.cuda.set_device(device_id)
        >>> sharded_module = FSDP(my_module)
        >>> optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        >>> x = sharded_module(x, y=3, z=torch.Tensor([1]))
        >>> loss = x.sum()
        >>> loss.backward()
        >>> optim.step()

    .. warning::
        The optimizer must be initialized *after* the module has been wrapped,
        since FSDP will shard parameters in-place and this will break any
        previously initialized optimizers.

    .. warning::
        If the destination CUDA device has ID ``dev_id``, either (1)
        ``module`` should already be placed on that device, (2) the device
        should be set using ``torch.cuda.set_device(dev_id)``, or (3)
        ``dev_id`` should be passed into the ``device_id`` constructor
        argument. This FSDP instance's compute device will be that destination
        device. For (1) and (3), the FSDP initialization always occurs on GPU.
        For (2), the FSDP initialization happens on ``module`` 's current
        device, which may be CPU.

    .. warning::
        FSDP currently does not support gradient accumulation outside
        ``no_sync()`` when using CPU offloading. Trying to do so yields
        incorrect results since FSDP will use the newly-reduced gradient
        instead of accumulating with any existing gradient.

    .. warning::
        Changing the original parameter variable names after construction will
        lead to undefined behavior.

    .. warning::
        Passing in `sync_module_states=True` flag requires module to be put
        on GPU, or to use ``device_id`` argument to specify a CUDA device that
        FSDP will move module to. This is because ``sync_module_states=True``
        requires GPU communication.

    .. warning::
        As of PyTorch 1.12, FSDP only offers limited support for shared parameters
        (for example, setting one ``Linear`` layer's weight to another's). In
        particular, modules that share parameters must be wrapped as part of the
        same FSDP unit. If enhanced shared parameter support is needed for your
        use case, please ping https://github.com/pytorch/pytorch/issues/77724

    .. note::
        Inputs into FSDP ``forward`` function will be moved to compute device
        (same device FSDP module is on) before running ``forward``, so user does
        not have to manually move inputs from CPU -> GPU.

    Args:
        module (nn.Module):
            module to be wrapped with FSDP.
        process_group (Optional[ProcessGroup]):
            process group for sharding
        sharding_strategy (Optional[ShardingStrategy]):
            Config sharding algorithm, different sharding algorithm has trade
            off between memory saving and communication overhead. ``FULL_SHARD``
            will be chosen if sharding_strategy is not specified.
        cpu_offload (Optional[CPUOffload]):
            CPU offloading config. Currently, only parameter and gradient CPU
            offload is supported. It can be enabled via passing in
            ``cpu_offload=CPUOffload(offload_params=True)``. Note that this
            currently implicitly enables gradient offloading to CPU in order for
            params and grads to be on same device to work with optimizer. This
            API is subject to change. Default is ``None`` in which case there
            will be no offloading.
        auto_wrap_policy (Optional[Callable[[nn.Module, bool, int], bool]]):
            A callable specifying a policy to recursively wrap layers with FSDP.
            Note that this policy currently will only apply to child modules of
            the passed in module. The remainder modules are always wrapped in
            the returned FSDP root instance.
            ``size_based_auto_wrap_policy`` written in ``torch.distributed.fsdp.wrap`` is
            an example of ``auto_wrap_policy`` callable, this policy wraps layers
            with the number of parameters larger than 100M. ``transformer_auto_wrap_policy``
            written in ``torch.distributed.fsdp.wrap`` is an example of ``auto_wrap_policy``
            callable for transformer-like model architectures. Users can supply the customized
            ``auto_wrap_policy`` callable that should accept following arguments:
            ``module: nn.Module``, ``recurse: bool``, ``unwrapped_params: int``, and return
            a ``bool`` specifying whether the passed in ``module``` should be wrapped
            (if ``recurse=False``) or whether we should recurse down the subgraph of ``module``
            children (if ``recurse=True``). Extra customized arguments could be added to
            the customized ``auto_wrap_policy`` callable as well. It is a good practice to
            print out the sharded model and check whether the sharded model is what
            the application wants and then adjust accordingly.

            Example::

                >>> def custom_auto_wrap_policy(
                >>>     module: nn.Module,
                >>>     recurse: bool,
                >>>     unwrapped_params: int,
                >>>     # These are customizable for this policy function.
                >>>     min_num_params: int = int(1e8),
                >>> ) -> bool:
                >>>     return unwrapped_params >= min_num_params
                >>> # Configure a custom min_num_params
                >>> my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=1e5)

        backward_prefetch (Optional[BackwardPrefetch]):
            This is an experimental feature that is subject to change in the
            the near future. It allows users to enable two different backward_prefetch
            algorithms to help backward communication and computation overlapping.
            Pros and cons of each algorithm is explained in the class ``BackwardPrefetch``.
        mixed_precision (Optional[MixedPrecision]): A ``MixedPrecision`` instance
            describing the mixed precision training config to be used. ``MixedPrecision``
            supports configuring parameter, buffer, and gradient communication dtype. Note
            that only floating point data is cast to the reduced precision. This allows
            users potential memory saving and training speedup while trading off
            accuracy during model training. If ``None``, no mixed precision is applied.
            Note that if ``mixed_precision`` is enabled for FSDP model that
            contains ``BatchNorm`` with ``auto_wrap_policy``, FSDP will take
            care to disable mixed precision for ``BatchNorm`` units by wrapping
            them separately in their own FSDP unit with ``mixed_precision=None``.
            This is done because several ``BatchNorm`` kernels do not implement
            reduced type support at the moment. If individually wrapping the model,
            users must take care to set ``mixed_precision=None`` for
            ``BatchNorm`` units.
            (Default: ``None``)
        ignored_modules (Optional[Iterable[torch.nn.Module]]): Modules whose
            own parameters and child modules' parameters and buffers are
            ignored by this instance. None of the modules directly in
            ``ignored_modules`` should be :class:`FullyShardedDataParallel`
            instances, and any child modules that are already-constructed
            :class:`FullyShardedDataParallel` instances will not be ignored if
            they are nested under this instance. This argument may be used to
            avoid sharding specific parameters at module granularity when using an
            ``auto_wrap_policy`` or if parameters' sharding is not managed by
            FSDP. (Default: ``None``)
        param_init_fn (Optional[Callable[[nn.Module], None]]):
            A ``Callable[torch.nn.Module] -> None`` that
            specifies how modules that are currently on the meta device should be initialized
            onto an actual device. Note that as of v1.12, we detect modules on the meta
            device via ``is_meta`` check and apply a default initialization that calls
            ``reset_parameters`` method on the passed in ``nn.Module`` if ``param_init_fn``
            is not specified, otherwise we run ``param_init_fn`` to initialize the passed
            in ``nn.Module``. In particular, this means that if ``is_meta=True`` for any
            module parameters for modules that will be wrapped with FSDP and ``param_init_fn``
            is not specified, we assume your module properly implements a ``reset_paramters()``
            and will throw errors if not. Note that additionally, we offer support for modules
            initialized with torchdistX's (https://github.com/pytorch/torchdistX)
            ``deferred_init`` API. In this case, deferred modules would be initialized
            by a default initialization function that calls torchdistX's
            ``materialize_module``, or the passed in ``param_init_fn``, if it is not
            ``None``. The same ``Callable`` is applied to initialize all meta modules.
            Note that this initialization function is applied before doing any FSDP sharding
            logic.

            Example::

                >>> # xdoctest: +SKIP("undefined variables")
                >>> module = MyModule(device="meta")
                >>> def my_init_fn(module):
                >>>     # responsible for initializing a module, such as with reset_parameters
                >>>     ...
                >>> fsdp_model = FSDP(module, param_init_fn=my_init_fn, auto_wrap_policy=size_based_auto_wrap_policy)
                >>> print(next(fsdp_model.parameters()).device) # current CUDA device
                >>> # With torchdistX
                >>> module = deferred_init.deferred_init(MyModule, device="cuda")
                >>> # Will initialize via deferred_init.materialize_module().
                >>> fsdp_model = FSDP(module, auto_wrap_policy=size_based_auto_wrap_policy)

        device_id (Optional[Union[int, torch.device]]): An ``int`` or ``torch.device``
            describing the CUDA device the FSDP module should be moved to determining where
            initialization such as sharding takes place. If this argument is not specified
            and ``module`` is on CPU, we issue a warning mentioning that this argument can
            be specified for faster initialization. If specified, resulting FSDP instances
            will reside on this device, including moving ignored modules' parameters if
            needed. Note that if ``device_id`` is specified but ``module`` is already on a
            different CUDA device, an error will be thrown. (Default: ``None``)
        sync_module_states (bool): If ``True``, each individually wrapped FSDP unit will broadcast
            module parameters from rank 0 to ensure they are the same across all ranks after
            initialization. This helps ensure model parameters are the same across ranks
            before starting training, but adds communication overhead to ``__init__``, as at least
            one broadcast is triggered per individually wrapped FSDP unit.
            This can also help load checkpoints taken by ``state_dict`` and to be loaded by
            ``load_state_dict`` in a memory efficient way. See documentation for
            :class:`FullStateDictConfig` for an example of this. (Default: ``False``)
        forward_prefetch (bool): If ``True``, then FSDP *explicitly* prefetches
            the next upcoming all-gather while executing in the forward pass.
            This may improve communication and computation overlap for CPU
            bound workloads. This should only be used for static graph models
            since the forward order is fixed based on the first iteration's
            execution. (Default: ``False``)
        limit_all_gathers (bool): If ``False``, then FSDP allows the CPU
            thread to schedule all-gathers without any extra synchronization.
            If ``True``, then FSDP explicitly synchronizes the CPU thread to
            prevent too many in-flight all-gathers. This ``bool`` only affects
            the sharded strategies that schedule all-gathers. Enabling this can
            help lower the number of CUDA malloc retries.
    """
    def __init__(
        self,
        module: nn.Module,
        process_group: Optional[ProcessGroup] = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        cpu_offload: Optional[CPUOffload] = None,
        auto_wrap_policy: Optional[Callable] = None,
        backward_prefetch: Optional[BackwardPrefetch] = None,
        mixed_precision: Optional[MixedPrecision] = None,
        ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
        param_init_fn: Optional[Callable[[nn.Module], None]] = None,
        device_id: Optional[Union[int, torch.device]] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = False,
    ):
        if isinstance(auto_wrap_policy, ParamExecOrderWrapPolicy):
            self._init_param_exec_order_wrap_policy(
                module=module,
                process_group=process_group,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                auto_wrap_policy=auto_wrap_policy,
                backward_prefetch=backward_prefetch,
                mixed_precision=mixed_precision,
                ignored_modules=ignored_modules,
                param_init_fn=param_init_fn,
                device_id=device_id,
                sync_module_states=sync_module_states,
                forward_prefetch=forward_prefetch,
                limit_all_gathers=limit_all_gathers,
            )
            return

        torch._C._log_api_usage_once("torch.distributed.fsdp")
        super().__init__()

        self._ignored_modules = self._get_ignored_modules(module, ignored_modules)
        ignored_params, self._ignored_param_names = self._get_ignored_params(
            module, self._ignored_modules
        )
        self._buffer_names = self._get_buffer_names(module)
        if auto_wrap_policy is not None:
            auto_wrap_kwargs = {
                "module": module,
                "auto_wrap_policy": auto_wrap_policy,
                "wrapper_cls": FullyShardedDataParallel,
                "ignored_modules": self._ignored_modules,
                "ignored_params": ignored_params,
                "only_wrap_children": True,  # avoid double wrapping the root
            }
            fsdp_kwargs = {
                "process_group": process_group,
                "sharding_strategy": sharding_strategy,
                "cpu_offload": cpu_offload,
                "backward_prefetch": backward_prefetch,
                "mixed_precision": mixed_precision,
                "param_init_fn": param_init_fn,
                "device_id": device_id,
                "sync_module_states": sync_module_states,
                "forward_prefetch": forward_prefetch,
                "limit_all_gathers": limit_all_gathers,
            }
            self._auto_wrap(auto_wrap_kwargs, fsdp_kwargs)

        self.process_group = process_group or _get_default_group()
        self.rank = self.process_group.rank()
        self.world_size = self.process_group.size()
        self.training_state = TrainingState_.IDLE
        self.cpu_offload = cpu_offload or CPUOffload()
        self.backward_prefetch = backward_prefetch
        self.forward_prefetch = forward_prefetch
        self.limit_all_gathers = limit_all_gathers
        backward_prefetch_limit = 1
        forward_prefetch_limit = 1
        # We clamp the strategy to `NO_SHARD` for world size of 1 since they
        # are currently functionally equivalent. This may change if/when we
        # integrate FSDP with MoE.
        if self.world_size == 1:
            sharding_strategy = ShardingStrategy.NO_SHARD
        self.sharding_strategy = sharding_strategy or ShardingStrategy.FULL_SHARD
        self.mixed_precision = mixed_precision or MixedPrecision()
        # Save a mapping from fully prefixed buffer name to its original dtype
        # since for mixed precision, buffers are restored to their original
        # dtype for model checkpointing
        self._buffer_name_to_orig_dtype: Dict[str, torch.dtype] = {}

        self._check_single_device_module(module, ignored_params)
        device_from_device_id: Optional[torch.device] = self._get_device_from_device_id(device_id)
        self._materialize_module(module, param_init_fn, device_from_device_id)
        self._move_module_to_device(module, ignored_params, device_from_device_id)
        self.compute_device = self._get_compute_device(module, ignored_params, device_from_device_id)
        params_to_flatten = list(self._get_orig_params(module, ignored_params))
        if sync_module_states:
            self._sync_module_states(module, params_to_flatten)

        # This FSDP instance's handles should inherit the same process group,
        # compute device, CPU offload, and mixed precision settings. However,
        # different sharding strategies are allowed.
        config = HandleConfig(
            sharding_strategy_map[self.sharding_strategy],
            self.cpu_offload.offload_params,
            self.mixed_precision.param_dtype,
            self.mixed_precision.reduce_dtype,
            self.mixed_precision.keep_low_precision_grads,
        )
        self._fsdp_wrapped_module = FlattenParamsWrapper(
            module,
            params_to_flatten,
            self.compute_device,
            config,
        )
        self._check_orig_params_flattened(ignored_params)
        # Invariant: `self.params` contains exactly the `FlatParameter`s of the
        # handles in `self._handles`
        self._handles: List[FlatParamHandle] = []
        self.params: List[FlatParameter] = []
        if self._fsdp_wrapped_module.has_params:
            handle = self._fsdp_wrapped_module.handle
            self.params.append(handle.flat_param)
            self._register_param_handle(handle)
            handle.shard(self.process_group)
            if self.cpu_offload.offload_params and handle.flat_param.device != torch.device("cpu"):
                with torch.no_grad():
                    handle._flat_param_to(torch.device("cpu"))

        self._sync_gradients = True
        self._communication_hook = self._get_default_comm_hook()
        self._communication_hook_state = self._get_default_comm_hook_state()
        self._hook_registered = False

        # Used to prevent running the pre-backward hook multiple times
        self._ran_pre_backward_hook: Dict[_HandlesKey, bool] = {}
        self._is_root: Optional[bool] = None  # `None` indicates not yet set
        # The following attributes are owned by the root FSDP instance and
        # shared with non-root FSDP instances
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._free_event_queue = _FreeEventQueue()
        self._debug_level = dist.get_debug_level()
        self._exec_order_data = _ExecOrderData(
            self._debug_level,
            backward_prefetch_limit,
            forward_prefetch_limit,
        )
        self._handles_prefetched: Dict[_HandlesKey, bool] = {}
        # Used for guarding against mistargeted backward prefetches
        self._needs_pre_backward_unshard: Dict[_HandlesKey, bool] = {}
        # Used for guarding against mistargeted forward prefetches
        self._needs_pre_forward_unshard: Dict[_HandlesKey, bool] = {}
        # The data structures use tuples of handles to generalize over the case
        # where a module's forward involves multiple handles.

        # `_state_dict_type` controls the `state_dict()` behavior, which is
        # implemented using post-save and pre-load hooks
        self._state_dict_type = StateDictType.FULL_STATE_DICT
        self._state_dict_config = FullStateDictConfig()
        self._register_state_dict_hook(self._post_state_dict_hook)
        self._post_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: self._full_post_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: self._local_post_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: self._sharded_post_state_dict_hook,
        }
        self._register_load_state_dict_pre_hook(
            self._pre_load_state_dict_hook, with_module=True
        )
        self._pre_load_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: self._full_pre_load_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: self._local_pre_load_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: self._sharded_pre_load_state_dict_hook,
        }
        self.register_load_state_dict_post_hook(
            self._post_load_state_dict_hook
        )
        self._post_load_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: self._full_post_load_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: self._local_post_load_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: self._sharded_post_load_state_dict_hook,
        }

    def _get_ignored_modules(
        self,
        root_module: nn.Module,
        _ignored_modules: Optional[Iterable[torch.nn.Module]],
    ) -> Set[nn.Module]:
        """
        Checks that ``_ignored_modules`` is an iterable of ``nn.Module`` s
        without any FSDP instances, and returns the modules contained in their
        module subtrees as a :class:`set`. Nested FSDP instances are excluded,
        but their already-computed ignored modules are included.
        """
        if _ignored_modules is None:
            return set()
        msg_prefix = "`ignored_modules` should be an iterable of `torch.nn.Module`s "
        try:
            ignored_root_modules = set(_ignored_modules)
        except TypeError:
            raise TypeError(msg_prefix + f"but got {type(_ignored_modules)}")
        for module in ignored_root_modules:
            if not isinstance(module, torch.nn.Module):
                raise TypeError(msg_prefix + f"but got an iterable with {type(module)}")
            if isinstance(module, FullyShardedDataParallel):
                raise ValueError("`ignored_modules` should not include FSDP modules")
        # Include child modules and exclude nested FSDP modules themselves
        ignored_modules = set(
            child
            for module in ignored_root_modules
            for child in module.modules()
            if not isinstance(child, (FullyShardedDataParallel, FlattenParamsWrapper))
        )
        if root_module in ignored_modules:
            warnings.warn(
                "Trying to ignore the top-level module passed into the FSDP "
                "constructor itself will result in all parameters being "
                f"ignored and is not well-supported: {module}"
            )
        # Include nested FSDP modules' ignored modules
        for submodule in root_module.modules():
            if isinstance(submodule, FullyShardedDataParallel):
                assert hasattr(submodule, "_ignored_modules")
                ignored_modules.update(submodule._ignored_modules)
        return ignored_modules

    def _get_ignored_params(
        self,
        root_module: torch.nn.Module,
        ignored_modules: Set[torch.nn.Module],
    ) -> Tuple[Set[torch.nn.Parameter], Set[str]]:
        """
        Returns the parameters of the modules in ``ignored_modules``,
        excluding any :class:`FlatParameter` s, and their fully prefixed names,
        both as :class:`set` s.
        """
        ignored_params = set(
            p
            for m in ignored_modules
            for p in m.parameters()
            if not _is_fsdp_flattened(p)
        )
        # Conservatively include all shared parameters' names
        param_to_unflat_param_names = _get_param_to_unflat_param_names(
            root_module,
            dedup_shared_params=False,
        )
        ignored_param_names = set()
        for param in ignored_params:
            unflat_param_names = param_to_unflat_param_names[param]
            clean_names = []
            for k in unflat_param_names:
                # Clean any module wrapper prefixes in case of nested wrapping
                clean_names.append(clean_tensor_name(k))
            ignored_param_names.update(clean_names)
        return ignored_params, ignored_param_names

    def _get_buffer_names(self, root_module: nn.Module) -> Set[str]:
        """
        Returns the fully prefixed names of all buffers in the module hierarchy
        rooted at ``root_module`` as a class:`set`.
        """

        def module_fn(module: nn.Module, prefix: str, buffer_names: Set[str]):
            # For FSDP modules, only add the entry when considering the
            # contained `FlattenParamsWrapper` to avoid duplication
            if not isinstance(module, FullyShardedDataParallel):
                for buffer_name, _ in module.named_buffers(recurse=False):
                    # Clean module wrapper prefixes in case of nested wrapping
                    prefixed_buffer_name = clean_tensor_name(prefix + buffer_name)
                    buffer_names.add(prefixed_buffer_name)

        def return_fn(buffer_names: Set[str], *args):
            return buffer_names

        buffer_names: Set[str] = set()
        return _apply_to_modules(
            root_module,
            module_fn,
            return_fn,
            buffer_names,
        )

    def _auto_wrap(
        self,
        auto_wrap_kwargs: Dict[str, Any],
        fsdp_kwargs: Dict[str, Any],
    ) -> None:
        """
        Recursively auto wraps the root module given by the key "module" in
        ``auto_wrap_kwargs`` with the arguments in ``auto_wrap_kwargs`` and
        ``fsdp_kwargs``.

        Precondition: ``auto_wrap_policy`` contains the arguments expected by
        ``_recursive_wrap()``, where ``auto_wrap_policy`` is not ``None``.
        ``fsdp_kwargs`` contains all FSDP arguments except ``module``.
        """
        auto_wrap_policy = auto_wrap_kwargs["auto_wrap_policy"]
        root_module = auto_wrap_kwargs["module"]
        assert auto_wrap_policy is not None
        # For auto wrapping, submodules should not already be wrapped with FSDP
        # since double wrapping is not supported
        for module_name, module in root_module.named_modules():
            if isinstance(module, FullyShardedDataParallel):
                raise ValueError(
                    f"Expected {module_name} to NOT be FullyShardedDataParallel "
                    "if using an `auto_wrap_policy`"
                )
        mixed_precision = fsdp_kwargs["mixed_precision"]
        if mixed_precision is not None and _contains_batchnorm(root_module):
            _override_batchnorm_mixed_precision(root_module)
            auto_wrap_policy = functools.partial(
                _or_policy, policies=[_wrap_batchnorm_individually, auto_wrap_policy]
            )
            warnings.warn(
                "Both mixed precision and an `auto_wrap_policy` were specified "
                "for FSDP, where the wrapped module has batch norm submodules. "
                "The batch norm submodules will be wrapped as separate FSDP "
                "instances with mixed precision disabled since some batch norm "
                "kernels do not support low precision."
            )
            auto_wrap_kwargs["auto_wrap_policy"] = auto_wrap_policy
        _recursive_wrap(**auto_wrap_kwargs, **fsdp_kwargs)

    def _check_single_device_module(
        self,
        module: nn.Module,
        ignored_params: Set[nn.Parameter],
    ) -> None:
        """
        Raises an error if ``module`` has original parameters on multiple
        devices, ignoring the parameters in ``ignored_params``. Thus, after
        this method, the module must be either fully on the CPU or fully on a
        non-CPU device.
        """
        devices = set(
            param.device for param in self._get_orig_params(module, ignored_params)
        )
        if len(devices) > 1:
            raise RuntimeError(
                f"FSDP only supports single device modules but got params on {devices}"
            )

    def _get_device_from_device_id(
        self,
        device_id: Optional[Union[int, torch.device]],
    ) -> Optional[torch.device]:
        """
        """
        if device_id is None:
            return None
        device = (
            device_id
            if isinstance(device_id, torch.device)
            else torch.device(device_id)
        )
        if device == torch.device("cuda"):
            warnings.warn(
                f"FSDP got the argument `device_id` {device_id} on rank "
                f"{self.rank}, which does not have an explicit index. "
                f"FSDP will use the current device {torch.cuda.current_device()}. "
                "If this is incorrect, please explicitly call `torch.cuda.set_device()` "
                "before FSDP initialization or pass in the explicit device "
                "index as the `device_id` argument."
            )
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    def _materialize_module(
        self,
        module: nn.Module,
        param_init_fn: Optional[Callable[[nn.Module], None]],
        device_from_device_id: Optional[torch.device],
    ) -> None:
        """
        Materializes the wrapped module ``module`` in place if needed: either
        if the module has parameters that use meta device or are torchdistX
        fake tensors.

        This method uses ``param_init_fn`` to materialize the module if the
        function is not ``None`` and falls back to default behavior otherwise.
        For meta device, this moves the module to ``device_from_device_id`` if
        it is not ``None`` or the current device otherwise and calls
        ``reset_parameters()``, and for torchdistX fake tensors, this calls
        ``deferred_init.materialize_module()``.
        """
        is_meta_module = any(p.is_meta for p in module.parameters())
        is_torchdistX_deferred_init = (
            not is_meta_module
            and _TORCHDISTX_AVAIL
            and any(fake.is_fake(p) for p in module.parameters())
        )
        if (
            is_meta_module or is_torchdistX_deferred_init
        ) and param_init_fn is not None:
            if not callable(param_init_fn):
                raise ValueError(
                    f"Expected {param_init_fn} to be callable but got {type(param_init_fn)}"
                )
            param_init_fn(module)
        elif is_meta_module:
            # Run default meta device initialization
            materialization_device = device_from_device_id or torch.cuda.current_device()
            module.to_empty(device=materialization_device)
            try:
                with torch.no_grad():
                    module.reset_parameters()
            except BaseException as e:
                warnings.warn(
                    "Unable to call `reset_parameters()` for module on meta "
                    f"device with error {str(e)}. Please ensure your "
                    "module implements a `reset_parameters()` method."
                )
                raise e
        elif is_torchdistX_deferred_init:
            # Run default torchdistX initialization
            deferred_init.materialize_module(
                module,
                check_fn=lambda k: not isinstance(k, FullyShardedDataParallel),
            )

    def _move_module_to_device(
        self,
        module: nn.Module,
        ignored_params: Set[nn.Parameter],
        device_from_device_id: Optional[torch.device],
    ):
        """
        Moves ``module`` depending on ``device_from_device_id`` and its current
        device. This includes moving ignored modules' parameters.

        - If ``device_from_device_id`` is not ``None``, then this moves
        ``module`` to the device.
        - If ``device_from_device_id`` is ``None``, then this does not move
        ``module`` but warns the user if it is on CPU.

        Precondition: ``_check_single_device_module()``.
        """
        cpu_device = torch.device("cpu")
        param = next(self._get_orig_params(module, ignored_params), None)
        if param is None:
            return  # no original parameters to manage
        if device_from_device_id is not None:
            if param.device == cpu_device:
                # NOTE: This includes moving ignored modules' parameters.
                module = module.to(device_from_device_id)
                # TODO: This is a temporary fix to move already- constructed
                # `FlatParameter`s back to CPU if needed. This is needed to
                # make CPU offload work with `device_id`.
                for submodule in module.modules():
                    if (
                        isinstance(submodule, FullyShardedDataParallel)
                        and submodule.cpu_offload.offload_params
                    ):
                        with torch.no_grad():
                            for handle in submodule._handles:
                                handle._flat_param_to(torch.device("cpu"))
        elif param.device == cpu_device:
            warnings.warn(
                "Module is put on CPU and will thus have flattening and sharding"
                " run on CPU, which is less efficient than on GPU. We recommend passing in "
                "`device_id` argument which will enable FSDP to put module on GPU device,"
                " module must also be on GPU device to work with `sync_module_states=True` flag"
                " which requires GPU communication."
            )

    def _get_compute_device(
        self,
        module: nn.Module,
        ignored_params: Set[nn.Parameter],
        device_from_device_id: Optional[torch.device],
    ) -> torch.device:
        """
        Determines and returns this FSDP instance's compute device. If the
        module is already on a non-CPU device, then the compute device is that
        non-CPU device. If the module is on CPU, then the compute device is the
        current device.

        Since this method should be called after materializing the module, any
        non-CPU device should not be meta device. For now, the compute device
        is always a CUDA GPU device with its explicit index.

        Precondition: ``_check_single_device_module()`` and
        ``_move_module_to_device()``.
        """
        # If the module is on GPU already, then that GPU device has priority
        # over the current device
        param = next(self._get_orig_params(module, ignored_params), None)
        if param is not None and param.device.type == "cuda":
            compute_device = param.device
        else:
            compute_device = torch.device("cuda", torch.cuda.current_device())
        if (
            device_from_device_id is not None
            and compute_device != device_from_device_id
        ):
            raise ValueError(
                "Inconsistent compute device and `device_id` on rank "
                f"{self.rank}: {compute_device} vs {device_from_device_id}"
            )
        return compute_device

    def _sync_module_states(
        self, module: nn.Module, params: List[nn.Parameter]
    ) -> None:
        """
        Synchronizes module states (i.e. parameters ``params`` and all
        not-yet-synced buffers) by broadcasting from rank 0 to all ranks.

        Precondition: ``sync_module_states == True`` and ``self.process_group``
        has been set.
        """
        if params and any(param.device == torch.device("cpu") for param in params):
            raise ValueError(
                "Module has CPU parameters, but sync_module_states=True is specified."
                "This only works for GPU module, please specify `device_id` argument or move"
                " module to GPU before init."
            )
        module_states: List[torch.Tensor] = []
        # TODO (awgu): When exposing the original parameters, we need to also
        # use this attribute to prevent re-synchronizing parameters.
        for buffer in module.buffers():
            # Avoid re-synchronizing buffers in case of nested wrapping
            if not getattr(buffer, "_fsdp_synced", False):
                buffer._fsdp_synced = True
                module_states.append(buffer.detach())
        module_states.extend(param.detach() for param in params)
        _sync_params_and_buffers(
            self.process_group, module_states, _PARAM_BROADCAST_BUCKET_SIZE, src=0,
        )

    def _get_orig_params(
        self,
        module: nn.Module,
        ignored_params: Set[nn.Parameter],
    ) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over the original parameters in ``module``,
        ignoring the parameters in ``ignored_params`` and any ``FlatParameter``
        s (which may be present due to nested FSDP wrapping).
        """
        param_gen = module.parameters()
        try:
            while True:
                param = next(param_gen)
                if param not in ignored_params and not _is_fsdp_flattened(param):
                    yield param
        except StopIteration:
            pass

    def _check_orig_params_flattened(self, ignored_params: Set[nn.Parameter]) -> None:
        """
        Checks that all original parameters have been flattened and hence made
        invisible to ``named_parameters()``. This should be called as a sanity
        check after flattening the wrapped module's parameters.
        """
        for param_name, param in self.named_parameters():
            if param not in ignored_params and not _is_fsdp_flattened(param):
                raise RuntimeError(
                    f"Found an unflattened parameter: {param_name}; "
                    f"{param.size()} {param.__class__}"
                )

    def _register_param_handle(self, handle: FlatParamHandle) -> None:
        """Registers the parameter handle to this FSDP instance."""
        if handle not in self._handles:
            self._handles.append(handle)

    @torch.no_grad()
    def _unshard(
        self,
        handles: List[FlatParamHandle],
    ) -> None:
        """
        Unshards the handles in ``handles``. If the handles are in
        :meth:`summon_full_params` and are using mixed precision, then they are
        forced to full precision.

        Postcondition: Each handle's ``FlatParameter`` 's data is the padded
        unsharded flattened parameter on the compute device.
        """
        if not handles:
            return
        if self.limit_all_gathers:
            event = self._free_event_queue.dequeue_if_needed()
            if event:
                event.synchronize()
        any_ran_pre_unshard = False
        with torch.cuda.stream(self._streams["pre_all_gather"]):
            for handle in handles:
                ran_pre_unshard = handle.pre_unshard()
                any_ran_pre_unshard = any_ran_pre_unshard or ran_pre_unshard
        if any_ran_pre_unshard:
            self._streams["all_gather"].wait_stream(self._streams["pre_all_gather"])
        with torch.cuda.stream(self._streams["all_gather"]):
            for handle in handles:
                handle.unshard()
                handle.post_unshard()

    def _reshard(
        self,  # unused
        handles: List[FlatParamHandle],
        free_unsharded_flat_params: List[bool],
    ) -> None:
        """
        Reshards the handles in ``handles``. ``free_unsharded_flat_params``
        should have the same length as ``handles``, and each element should
        give whether the corresponding handle should free its padded unsharded
        flattened parameter.
        """
        if not handles:
            return
        p_assert(
            len(handles) == len(free_unsharded_flat_params),
            "Expects both lists to have equal length but got "
            f"{len(handles)} and {len(free_unsharded_flat_params)}"
        )
        for handle, free_unsharded_flat_param in zip(
            handles,
            free_unsharded_flat_params,
        ):
            handle.reshard(free_unsharded_flat_param)
            if self.limit_all_gathers and free_unsharded_flat_param:
                free_event = torch.cuda.Event()
                free_event.record()
                self._free_event_queue.enqueue(free_event)
            handle.post_reshard()
        # Since we prefetch entire handles keys at a time, conservatively mark
        # the entire key as no longer prefetched once we free at least one
        handles_key = tuple(handles)
        if any(free_unsharded_flat_params):
            self._handles_prefetched.pop(handles_key, None)

    @property
    def module(self) -> nn.Module:
        """
        Returns the wrapped module (like :class:`DistributedDataParallel`).
        """
        assert isinstance(self._fsdp_wrapped_module, FlattenParamsWrapper)
        return self._fsdp_wrapped_module.module

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._fsdp_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self._fsdp_wrapped_module.__getitem__(key)  # type: ignore[operator]

    def check_is_root(self) -> bool:
        self._lazy_init()
        assert self._is_root is not None
        return self._is_root

    @staticmethod
    def fsdp_modules(
        module: nn.Module,
        root_only: bool = False,
    ) -> List["FullyShardedDataParallel"]:
        """
        Returns all nested FSDP instances, possibly including ``module`` itself
        and only including FSDP root modules if ``root_only=True``.

        Args:
            module (torch.nn.Module): Root module, which may or may not be an
                ``FSDP`` module.
            root_only (bool): Whether to return only FSDP root modules.
                (Default: ``False``)

        Returns:
            List[FullyShardedDataParallel]: FSDP modules that are nested in
            the input ``module``.
        """
        return [
            submodule for submodule in module.modules()
            if isinstance(submodule, FullyShardedDataParallel) and
            (not root_only or submodule.check_is_root())
        ]

    def apply(self, fn: Callable[[nn.Module], None]) -> "FullyShardedDataParallel":
        r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Compared to ``torch.nn.Module.apply``, this version additionally gathers
        the full parameters before applying ``fn``. It should not be called from
        within another ``summon_full_params`` context.

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self
        """
        uninitialized = self._is_root is None
        self._assert_state(TrainingState_.IDLE)
        with self._summon_full_params(recurse=False, writeback=True):
            ret = super().apply(fn)

        # Reset lazy init that might be called by _summon_full_params, since
        # it could have set is_root incorrectly for non-root FSDP instances.
        if uninitialized and self._is_root:
            for module in self.fsdp_modules(self):
                module._reset_lazy_init()

        return ret

    def _mixed_precision_enabled_for_params(self) -> bool:
        """
        Whether user explicitly enabled mixed precision for
        parameters or not.
        """
        return self.mixed_precision.param_dtype is not None

    def _mixed_precision_enabled_for_buffers(self) -> bool:
        """
        Whether user explicitly enabled mixed precision for
        buffers or not.
        """
        return self.mixed_precision.buffer_dtype is not None

    def _mixed_precision_enabled_for_reduce(self) -> bool:
        """
        Whether user explicitly enabled mixed precision for
        gradient reduction or not.
        """
        return self.mixed_precision.reduce_dtype is not None

    def _mixed_precision_keep_low_precision_grads(self) -> bool:
        return (
            self.mixed_precision is not None
            and self.mixed_precision.keep_low_precision_grads
        )

    def _low_precision_hook_enabled(self) -> bool:
        """
        Wether a low precision hook is registered or not.
        """
        return (
            self._communication_hook is not None
            and self._communication_hook in LOW_PRECISION_HOOKS
        )

    def _cast_fp_inputs_to_dtype(
        self, dtype: torch.dtype, *args: Any, **kwargs: Any
    ) -> Tuple[Any, Any]:
        """
        Casts floating point tensors in ``args`` and ``kwargs`` to the
        precision given by ``dtype``, while respecting the existing
        ``requires_grad`` on the tensors.
        """
        def cast_fn(x: torch.Tensor) -> torch.Tensor:
            if not torch.is_floating_point(x):
                return x
            y = x.to(dtype)
            # Explicitly copy over `requires_grad` since this runs inside
            # `torch.no_grad()`
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y

        with torch.no_grad():
            return (
                _apply_to_tensors(cast_fn, args),
                _apply_to_tensors(cast_fn, kwargs)
            )

    def _cast_buffers(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[Dict[str, torch.dtype]] = None,
        memo: Optional[Set] = None,
        recurse: bool = True,
    ) -> None:
        """Move all buffers to the given *device* and *dtype*.
        If *device* is not given, then it will default to
        ``self.compute_device``, otherwise buffer will be moved to ``device``.
        In the case of nested FSDP instances, we will respect the child instance's
        ``compute_device`` configuration.
        If *dtype* is given, it must be a mapping of buffer name to buffer dtype,
            and this argument is currently only given to restore back to original
            buffer types during checkpoint. If *dtype* is not given, and we are
            in mixed precision training, the buffer will be cast to buffer_dtype,
            otherwise the buffer will not be cast.
        Args:
            device (torch.device, Optional):
                device to cast buffers to (defaults to compute_device)
            dtype: (Dict[str, torch.dtype], Optional):
                Mapping of buffer name to their dtype to cast to.
            memo (Set, Optional):
                set of modules that have already been processed
            recurse (bool, Optional):
                Whether to call _cast_buffers recursively on nested FSDP
                instances (default is True).
        """
        if memo is None:
            memo = set()
        for module in self.modules():
            if module is not self and isinstance(module, FullyShardedDataParallel) and recurse:
                # Allow any child FSDP instances to handle their own buffers.
                module._cast_buffers(device=device, dtype=dtype, memo=memo, recurse=recurse)
            elif module not in memo:
                memo.add(module)
                for name, buf in module.named_buffers(recurse=False):
                    if buf is None:
                        continue
                    buf = buf.to(device=device or self.compute_device)
                    if name not in self._buffer_name_to_orig_dtype:
                        self._buffer_name_to_orig_dtype[name] = buf.dtype
                    # If given, cast buffer to the given dtype. This is used to
                    # suppport mixed precision for buffers
                    # (given by self.mixed_precision.buffer_dtype) and also used
                    # to restore the buffer dtype to the original precision for
                    # state_dict() calls.
                    # Note that non-floating point buffers are not casted.
                    if torch.is_floating_point(buf):
                        # We are restoring the original buffer type in
                        # preparation for checkpoint.
                        if dtype:
                            buf = buf.to(dtype=dtype[name])
                        # Note that we don't pass in self.mixed_precision.buffer_dtype
                        # recursively into _cast_buffers, as we want to respect
                        # mp config for child FSDP instances.
                        elif self._mixed_precision_enabled_for_buffers():
                            buf = buf.to(self.mixed_precision.buffer_dtype)

                    setattr(module, name, buf)

    def _reset_lazy_init(self) -> None:
        """
        Reset instance so :func:`_lazy_init` will run on the next forward.
        """
        self._is_root: Optional[bool] = None
        for p in self.params:
            if hasattr(p, "_local_shard"):
                # We only need to `del` `_local_shard` because
                # `_init_param_attributes()` gates the logic based on its
                # existence (and not any of the other attributes).
                del p._local_shard  # type: ignore[attr-defined]

    def _lazy_init(self) -> None:
        """
        Performs initialization lazily, typically right before the first
        forward pass. The laziness is needed to ensure that the parameter
        device/dtype and the FSDP hierarchy have finalized.

        This method's actual logic only runs on the root FSDP instance, which
        performs initialization for all non-root FSDP instances to avoid
        partial initialization.
        """
        if self._is_root is not None:
            return  # no-op: already initialized
        if not torch.cuda.is_available():
            # Allow the FSDP constructor to run even with CUDA but check this
            # once we start real execution
            raise RuntimeError("FSDP does not support CPU only execution")
        # The following logic is only run on the root FSDP instance since it
        # will set `_is_root=False` for the non-root instances
        self._is_root = True
        self._assert_state(TrainingState_.IDLE)
        self._init_streams()
        self._cast_buffers(recurse=True)
        for handle in self._handles:
            self._init_param_attributes(handle)
        self._exec_order_data.init(self, self.process_group)
        # Initialize non-root FSDP instances and share attributes from the root
        # to non-root instances
        inconsistent_limit_all_gathers = False
        for fsdp_module in self.fsdp_modules(self):
            if fsdp_module is not self:
                # Relax the assert for non-root FSDP instances in case the
                # nested initialized module is wrapped again in FSDP later (e.g.
                # after training to run inference)
                assert fsdp_module._is_root is None or not fsdp_module._is_root, (
                    "Non-root FSDP instance's `_is_root` should not have been "
                    "set yet or should have been set to `False`"
                )
                fsdp_module._is_root = False
                fsdp_module._streams = self._streams
                fsdp_module._exec_order_data = self._exec_order_data
                if fsdp_module.limit_all_gathers != self.limit_all_gathers:
                    # Prefer the root's value
                    inconsistent_limit_all_gathers = True
                    fsdp_module.limit_all_gathers = self.limit_all_gathers
                fsdp_module._free_event_queue = self._free_event_queue
                fsdp_module._handles_prefetched = self._handles_prefetched
                fsdp_module._needs_pre_backward_unshard = self._needs_pre_backward_unshard
                for handle in fsdp_module._handles:
                    fsdp_module._init_param_attributes(handle)
        if inconsistent_limit_all_gathers:
            warnings.warn(
                "Found inconsistent `limit_all_gathers` values across FSDP "
                f"instances on rank {self.rank}. Using the root FSDP's value "
                f"of {self.limit_all_gathers} for all instances."
            )

    # TODO (awgu): Move this to the `FlatParamHandle` class later
    @torch.no_grad()
    def _init_param_attributes(self, handle: FlatParamHandle) -> None:
        """
        We manage several attributes on each Parameter instance.
        A few attributes are set here:
            ``_local_shard``: a single shard of the parameter. This is needed to
                recover the shard after rebuilding full parameter in forward
                and backward.
            ``_full_param_padded``: the full weight (padded to be evenly
                divisible by ``world_size``), used for computation in the
                forward and backward pass. It is initialized with the
                appropriate size and then has its storage freed. This will be
                resized in place and only materialized (via all-gather) as needed.
        Another attribute is set by :func:`_register_post_backward_hooks`:
            ``_post_backward_hook_state``: it holds the parameter's AccumulateGrad object
                and the registered post hook handle.
        """
        p = handle.flat_param
        # If _local_shard has been set in the first lazy init and
        # current parameter is pointed to _local_shard, no need to
        # set the _local_shard again.
        if hasattr(p, "_local_shard"):
            # If CPU offloading, p._local_shard should have been placed on CPU
            # during its first lazy construction.
            if self.cpu_offload.offload_params:
                assert p._local_shard.device == torch.device(  # type: ignore[attr-defined]
                    "cpu"
                ), (
                    "Expected p._local_shard to be on CPU, "  # type: ignore[attr-defined]
                    f"but it's on {p._local_shard.device}"  # type: ignore[attr-defined]
                )
            return

        # A single shard of the parameters. Also makes p._local_shard to be on
        # CPU if we are CPU offloading, since p.data would be on CPU during
        # init.
        if self.cpu_offload.offload_params:
            assert p.device == torch.device("cpu"), (
                "Expected param to be on CPU when cpu_offloading is enabled. "
                "If CPU offloading is enabled correctly, you may be "
                "accidentally moving the model to CUDA after FSDP initialization."
            )
        p._local_shard = p.data  # type: ignore[attr-defined]
        # If CPU offloading, pin the memory to enable faster CPU -> GPU device
        # transfer.
        if self.cpu_offload.offload_params:
            assert p._local_shard.device == torch.device("cpu")  # type: ignore[attr-defined]
            p._local_shard = p._local_shard.pin_memory()  # type: ignore[attr-defined]
            # When offloading parameters, also move the grad shard to CPU during
            # backward pass. In this case, it's important to pre-allocate the
            # CPU grad shard in pinned memory so that we can do a non-blocking
            # transfer.
            p._cpu_grad = torch.zeros_like(  # type: ignore[attr-defined]
                p, device=torch.device("cpu")
            ).pin_memory()

        # If mixed_precision, maintain reduced precision param shard on
        # compute_device for computation in fwd/bwd. We resize storage to 0 here
        # and rematerialize before building the full param when needed. After
        # fwd/bwd, it is freed and we only hold on to the full precision shard.
        # As a result, this reduced precision shard is not allocated if we are
        # not in the forward/backward pass.
        if (
            self._mixed_precision_enabled_for_params()
        ):
            p._mp_shard = torch.zeros_like(
                p._local_shard,
                device=self.compute_device,
                dtype=self.mixed_precision.param_dtype
            )
            _free_storage(p._mp_shard)

        # We also maintain a full-sized parameter of type self.compute_dtype.
        # We resize the storage to size 0 at init (here) and only materialize
        # as needed. The storage may contain padding elements so that it is
        # evenly divisible by world_size, although these padding elements will
        # be removed before the relevant computation.
        if handle.uses_sharded_strategy:  # type: ignore[attr-defined]
            # We set p._full_param_padded's dtype to the desired parameter dtype
            # in the case of mixed precision. This is so that when we all_gather
            # into full_param_padded it can occur without issues and result in
            # full_param_padded having the expected param_dtype.
            full_param_dtype = (
                p.dtype if not self._mixed_precision_enabled_for_params()
                else self.mixed_precision.param_dtype
            )
            p._full_param_padded = torch.zeros(  # type: ignore[attr-defined]
                p.numel() * self.world_size,
                device=self.compute_device,
                dtype=full_param_dtype,
            )
            p._padded_unsharded_size = p._full_param_padded.size()  # type: ignore[attr-defined]
            _free_storage(p._full_param_padded)  # type: ignore[attr-defined]

            if self._mixed_precision_enabled_for_params():
                p._full_prec_full_param_padded = torch.zeros(  # type: ignore[attr-defined]
                    p.numel() * self.world_size,
                    device=self.compute_device,
                    dtype=p.dtype,  # full precision
                )
                _free_storage(p._full_prec_full_param_padded)

        # Track whether the `FlatParameter`'s post-backward hook has been
        # called for validation in `_wait_for_post_backward()`
        p._post_backward_called = False

    def _init_streams(self) -> None:
        """Initializes CUDA streams for overlapping data transfer and
        computation. This should only be called on the root FSDP instance."""
        assert self._is_root
        assert torch.cuda.is_available()
        # Stream for all-gathering parameters.
        self._streams["all_gather"] = torch.cuda.Stream()
        # Stream for overlapping grad reduction with the backward pass.
        self._streams["post_backward"] = torch.cuda.Stream()
        # Stream for pre-all-gather copies (e.g. H2D or precision cast).
        self._streams["pre_all_gather"] = torch.cuda.Stream()

    def _wait_for_previous_optim_step(self) -> None:
        """
        The root :class:`FullyShardedDataParallel` instance needs to
        synchronize with the default stream to ensure that the previous
        optimizer step is done.
        """
        if not self._is_root:
            return
        current_stream = torch.cuda.current_stream()
        self._streams["all_gather"].wait_stream(current_stream)
        # Having the pre-all-gather stream wait for the current stream even if
        # we do not leverage the pre-all-gather stream is tolerable since this
        # only runs once per iteration
        self._streams["pre_all_gather"].wait_stream(current_stream)

    def _prefetch_handles(
        self,
        current_handles_key: _HandlesKey,
    ) -> None:
        """
        Prefetches the next handles if needed (without synchronization). An
        empty handles key cannot prefetch.
        """
        if not current_handles_key:
            return
        handles_to_prefetch = self._get_handles_to_prefetch(current_handles_key)
        for handles_key in handles_to_prefetch:
            # Prefetch the next set of handles without synchronizing to allow
            # the sync to happen as late as possible to maximize overlap
            self._unshard(handles_key)
            self._handles_prefetched[handles_key] = True

    def _get_handles_to_prefetch(
        self,
        current_handles_key: _HandlesKey,
    ) -> List[_HandlesKey]:
        """
        Returns a :class:`list` of the handles keys to prefetch for the next
        module(s), where ``current_handles_key`` represents the current module.

        "Prefetching" refers to running the unshard logic early (without
        synchronization), and the "next" modules depend on the recorded
        execution order and the current training state.
        """
        training_state = self._get_training_state(current_handles_key)
        valid_training_states = (
            HandleTrainingState.BACKWARD_PRE,
            HandleTrainingState.BACKWARD_POST,
            HandleTrainingState.FORWARD,
        )
        p_assert(
            training_state in valid_training_states,
            f"Prefetching is only supported in {valid_training_states} but "
            f"currently in {training_state}"
        )
        eod = self._exec_order_data
        target_handles_keys: List[_HandlesKey] = []
        if (
            (
                training_state == HandleTrainingState.BACKWARD_PRE
                and self.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
            )
            or (
                training_state == HandleTrainingState.BACKWARD_POST
                and self.backward_prefetch == BackwardPrefetch.BACKWARD_POST
            )
        ):
            target_handles_keys = [
                target_handles_key for target_handles_key in
                eod.get_handles_to_backward_prefetch(current_handles_key)
                if self._needs_pre_backward_unshard.get(target_handles_key, False)
                and not self._handles_prefetched.get(target_handles_key, False)
            ]
        elif (
            training_state == HandleTrainingState.FORWARD
            and self.forward_prefetch
        ):
            target_handles_keys = [
                target_handles_key for target_handles_key in
                eod.get_handles_to_forward_prefetch(current_handles_key)
                if self._needs_pre_forward_unshard.get(target_handles_key, False)
                and not self._handles_prefetched.get(target_handles_key, False)
            ]
        return target_handles_keys

    def _get_training_state(
        self,
        handles_key: _HandlesKey,
    ) -> HandleTrainingState:
        """Returns the training state of the handles in ``handles_key``."""
        p_assert(len(handles_key) > 0, "Expects a non-empty handles key")
        training_states = set(handle._training_state for handle in handles_key)
        p_assert(
            len(training_states) == 1,
            f"Expects uniform training state but got {training_states}"
        )
        return next(iter(training_states))

    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: Optional[StateDictConfig] = None,
    ) -> Generator:
        """
        A context manager to set the ``state_dict_type`` of all the descendant
        FSDP modules of the target module. The target module does not have to
        be a FSDP module. If the target module is a FSDP module, its
        ``state_dict_type`` will also be changed.

        .. note:: This API should be called for only the top-level (root)
            module.

        .. note:: This API enables users to transparently use the conventional
            ``state_dict`` API to take model checkpoints in cases where the
            root FSDP module is wrapped by another ``nn.Module``. For example,
            the following will ensure ``state_dict``  is called on all non-FSDP
            instances, while dispatching into `local_state_dict` implementation
            for FSDP:

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = DDP(FSDP(...))
            >>> with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            >>>     checkpoint = model.state_dict()

        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired ``state_dict_type`` to set.
        """
        prev_state_dict_type = None
        prev_state_dict_config = None
        # Use default config a state_dict config is not set.
        if state_dict_config is None:
            state_dict_config = _state_dict_type_to_config[state_dict_type]()
        for submodule in FullyShardedDataParallel.fsdp_modules(module):
            if prev_state_dict_type is None:
                prev_state_dict_type = submodule._state_dict_type
            if prev_state_dict_config is None:
                prev_state_dict_config = submodule._state_dict_config
            if prev_state_dict_type != submodule._state_dict_type:
                raise RuntimeError("All FSDP module should the same state_dict_type.")
            if type(prev_state_dict_config) != type(submodule._state_dict_config):
                raise RuntimeError(
                    "All FSDP modules should have the same type of state_dict_config."
                )

            expected_state_dict_config_type = _state_dict_type_to_config[state_dict_type]
            if expected_state_dict_config_type != type(state_dict_config):
                raise RuntimeError(
                    f"Expected state_dict_config of type {expected_state_dict_config_type} but got {type(state_dict_config)}"
                )
            submodule._state_dict_type = state_dict_type
            submodule._state_dict_config = state_dict_config
        try:
            yield
        finally:
            assert prev_state_dict_type is not None  # Avoid mypy warning
            assert prev_state_dict_config is not None  # Avoid mypy warning
            for submodule in FullyShardedDataParallel.fsdp_modules(module):
                submodule._state_dict_type = prev_state_dict_type
                submodule._state_dict_config = prev_state_dict_config

    def _convert_to_wrapped_module_name(self, module_name: str) -> str:
        module_name = module_name.replace(f"{FPW_MODULE}.", "")
        module_name = module_name.replace(f"{FPW_MODULE}", "")
        if module_name:
            module_name = f"{module_name}."
        # Activation checkpoint adds a prefix that has to be
        # removed as well.
        module_name = module_name.replace(
            f"{checkpoint_wrapper._CHECKPOINT_PREFIX}.", ""
        )
        return module_name

    @property
    def _param_fqns(self) -> Iterator[Tuple[str, str, str]]:
        for param_name, module_name in (
            self._fsdp_wrapped_module.handle.parameter_module_names()
        ):
            module_name = self._convert_to_wrapped_module_name(module_name)
            fqn = f"{module_name}{param_name}"
            yield fqn, param_name, module_name

    def _full_post_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
    ) -> Dict[str, Any]:
        """
        Hook that runs after model.state_dict() is called before returning result to
        user. For FSDP, we may have to clone the tensors in state_dict as params go
        back to sharded version after _summon_full_params ends, and also remove
        "_fsdp_wrapped_module" prefix.
        """
        _replace_by_prefix(state_dict, prefix + f"{FSDP_WRAPPED_MODULE}.", prefix)
        self._assert_state([TrainingState_.SUMMON_FULL_PARAMS])
        # Return early for trivial cases
        if not state_dict or not self._fsdp_wrapped_module.has_params:
            return state_dict

        # If the `FlatParameter` is registered, then this rank only needed to
        # participate in the all-gather but does not actually save the state
        # dict (e.g. when `rank0_only=True` and `self.rank != 0`)
        if hasattr(self._fsdp_wrapped_module, "flat_param"):
            return state_dict

        offload_to_cpu = self._state_dict_config.offload_to_cpu
        cpu_device = torch.device("cpu")

        # Loop only the parameters saved in self._fsdp_wrapped_module to avoid
        # processing buffers.
        for fqn, param_name, module_name in self._param_fqns:
            fqn = f"{prefix}{fqn}"
            clean_key = fqn
            clean_prefix = clean_tensor_name(prefix)
            # Strip prefix out of key if needed as buffer names and param names
            # do not have prefix considered as they are not computed in `state_dict`
            # call.
            if clean_key.startswith(clean_prefix):
                clean_key = clean_key[len(clean_prefix):]

            # Clone non-ignored parameters before exiting the
            # `_summon_full_params()` context
            assert fqn in state_dict, (
                f"FSDP assumes {fqn} is in the state_dict but the state_dict "
                f"only has {state_dict.keys()}. prefix={prefix}, "
                f"module_name={module_name} param_name={param_name} rank={self.rank}."
            )
            if clean_key not in self._ignored_param_names and \
                    not getattr(state_dict[fqn], "_has_been_cloned", False):
                try:
                    state_dict[fqn] = state_dict[fqn].clone().detach()
                    state_dict[fqn]._has_been_cloned = True  # type: ignore[attr-defined]
                except BaseException as e:
                    warnings.warn(
                        f"Failed to clone() tensor with name {fqn}. This may mean "
                        "that this state_dict entry could point to invalid memory "
                        "regions after returning from state_dict() call if this "
                        "parameter is managed by FSDP. Please check clone "
                        f"implementation of {fqn}. Error: {str(e)}"
                    )

        # Offload the buffer to CPU if needed -- we do not do this in
        # `_summon_full_params()` since without care, that would free
        # the original buffer's GPU memory and require reallocating
        # that memory later; this only affects the state dict's buffer
        # variable and leaves the original buffer's GPU memory intact
        if offload_to_cpu:
            for clean_key in self._buffer_names:
                # This is a hack to support activation checkpoint.
                clean_key = clean_key.replace(
                    f"{checkpoint_wrapper._CHECKPOINT_PREFIX}.", ""
                )
                fqn = f"{prefix}{clean_key}"
                if fqn not in state_dict:
                    # A buffer can be registered as non-persistent.
                    continue
                if state_dict[fqn].device != cpu_device:
                    state_dict[fqn] = state_dict[fqn].to(cpu_device)
        return state_dict

    def _local_post_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
    ) -> Dict[str, Any]:
        """
        This hook create a ShardedTensor from the local flat_param and replace
        the state_dict[f"{prefix}{FLAT_PARAM}] with the ShardedTensor. No copy
        will happen. The underlying storage is the same.
        """
        _replace_by_prefix(state_dict, f"{prefix}{FSDP_WRAPPED_MODULE}.", prefix)
        if not self._fsdp_wrapped_module.has_params:
            return state_dict

        # state_dict[f"{prefix}{FLAT_PARAM}"] exists and has the same tensor
        # value as the flat_param but it is a pure Tensor because
        # nn.Module.state_dict() will detach the parameter. Therefore, we need
        # to get flat_param from the FlattenParamsWrapper to get the metadata.
        flat_param = getattr(self._fsdp_wrapped_module, FLAT_PARAM, None)
        assert flat_param is not None
        # Construct a ShardedTensor from the flat_param.
        full_numel = flat_param._unpadded_unsharded_size.numel()  # type: ignore[attr-defined]
        shard_offset = flat_param.numel() * self.rank
        valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
        if valid_data_size > 0 and flat_param._shard_numel_padded > 0:
            flat_param = flat_param.narrow(0, 0, valid_data_size)
        local_shards = [
            Shard.from_tensor_and_offsets(flat_param, [shard_offset], self.rank)
        ]
        state_dict[f"{prefix}{FLAT_PARAM}"] = init_from_local_shards(
            local_shards, full_numel, process_group=self.process_group
        )  # type: ignore[assignment]

        return state_dict

    @torch.no_grad()
    def _sharded_post_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
    ) -> Dict[str, Any]:
        """
        The hook replaces the unflattened, unsharded parameter in the state_dict
        with a unflattened, sharded parameter (a ShardedTensor).
        """
        _replace_by_prefix(state_dict, f"{prefix}{FSDP_WRAPPED_MODULE}.", prefix)
        if not self._fsdp_wrapped_module.has_params:
            return state_dict

        assert self.training_state != TrainingState_.SUMMON_FULL_PARAMS, (
            "Inside _sharded_post_load_state_dict_hook, the training_state must "
            "not be SUMMON_FULL_PARAMS."
        )
        with self._summon_full_params(recurse=False, writeback=False):
            for fqn, _, _ in self._param_fqns:
                # Create a ShardedTensor for the unflattened, non-sharded parameter.
                param = functools.reduce(getattr, fqn.split("."), self.module)
                state_dict[f"{prefix}{fqn}"] = _ext_chunk_tensor(
                    tensor=param,
                    rank=self.rank,
                    world_size=self.world_size,
                    num_devices_per_node=torch.cuda.device_count(),
                    pg=self.process_group
                )  # type: ignore[assignment]
        state_dict.pop(f"{prefix}{FLAT_PARAM}")
        return state_dict

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> Dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this
        FSDP module is executed. ``self._state_dict_type`` is used to decide
        what postprocessing will be done.
        """
        self = cast(FullyShardedDataParallel, module)
        processed_state_dict = self._post_state_dict_hook_fn[self._state_dict_type](state_dict, prefix)
        # Restore buffers, which currently are in their full precision type,
        # back to their mixed precision type. This is because buffers are cast
        # during lazy_init() and stay at their mixed precision type before/after
        # forward/backward. As a result state_dict() should maintain this.
        if (
            self._is_root
            and self._mixed_precision_enabled_for_buffers()
        ):
            self._cast_buffers(recurse=True)
        return processed_state_dict

    def state_dict(self, *args, **kwargs):
        """
        This is the entry point of all three FSDP ``state_dict`` APIs: full,
        local, and sharded. For the full state dict
        (``StateDictType.FULL_STATE_DICT``), FSDP attempts to unshard the model
        on all ranks, which may result in an OOM error if the full model cannot
        fit on a single GPU. In that case, users may pass in a
        :class:`FullStateDictConfig` to only save the checkpoint on rank 0 and/
        or to offload it to CPU memory layer by layer, enabling much larger
        checkpoints. If the full model cannot fit in CPU memory, then users may
        instead take a local state dict (``StateDictType.LOCAL_STATE_DICT``)
        that only saves the local shard of the model. The sharded state dict
        (``StateDictType.SHARDED_STATE_DICT``) saves the model parameters as
        ``ShardedTensor`` s. The ``state_dict`` type can be configured using
        the :meth:`state_dict_type` context manager.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> import torch
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> from torch.distributed.fsdp import StateDictType
            >>> torch.cuda.set_device(device_id)
            >>> my_module = nn.Linear(...)
            >>> sharded_module = FSDP(my_module)
            >>> full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            >>> with FSDP.state_dict_type(sharded_module, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            >>>     full_dict = sharded_module.state_dict()
            >>> full_dict.keys()
            >>> odict_keys(['weight', 'bias'])
            >>> # using local state dict
            >>> with FSDP.state_dict_type(sharded_module, StateDictType.LOCAL_STATE_DICT):
            >>>     local_dict = sharded_module.state_dict()
            >>> local_dict.keys()
            >>> odict_keys(['flat_param', 'inner.flat_param'])

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives may be used.
        """
        # TODO (rohan-varma): separate these out once a state_dict pre-hook
        # is available.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._lazy_init()
        if self._state_dict_type == StateDictType.FULL_STATE_DICT:
            # Get config args
            full_state_dict_config = (
                self._state_dict_config if self._state_dict_config is not None
                else FullStateDictConfig()
            )
            rank0_only = full_state_dict_config.rank0_only
            offload_to_cpu = full_state_dict_config.offload_to_cpu
            summon_ctx = (
                self._summon_full_params(
                    recurse=False, writeback=False, offload_to_cpu=offload_to_cpu, rank0_only=rank0_only
                )
                if self.training_state != TrainingState_.SUMMON_FULL_PARAMS else
                contextlib.suppress()
            )
            with summon_ctx:
                # Since buffers are not sharded and stay casted, restore them to their
                # original user module specified types for checkpoint. We take care to
                # recast in post_state_dict_hook for consistency with the fact that
                # buffers stay casted after forward/backward. We must have the
                # call here instead of above because _summon_full_params itself
                # calls _lazy_init() which would cast the buffers.
                if (
                    self._is_root
                    and self._mixed_precision_enabled_for_buffers()
                ):
                    self._cast_buffers(
                        dtype=self._buffer_name_to_orig_dtype, recurse=False
                    )
                state_dict = super().state_dict(*args, **kwargs)

            # TODO: support offload to CPU in post state dict hook.
            if not rank0_only or self.rank == 0:
                return state_dict
            else:
                return {}

        elif (
            self._state_dict_type == StateDictType.LOCAL_STATE_DICT or
            self._state_dict_type == StateDictType.SHARDED_STATE_DICT
        ):
            if (
                self._fsdp_wrapped_module.flat_param is not None and
                not self._fsdp_wrapped_module.handle.uses_sharded_strategy
            ):
                raise RuntimeError(
                    "sharded_state_dict/local_state_dict can only be called "
                    "when parameters are flatten and sharded."
                )
            return super().state_dict(*args, **kwargs)
        else:
            raise ValueError(f"Unknown StateDictType {self._state_dict_type}.")

    def _local_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Returns the local state of the module. Parameters are flattened and
        sharded, so the resulting state_dict can only be loaded after the module
        has been wrapped with FSDP.
        """
        with self.state_dict_type(self, StateDictType.LOCAL_STATE_DICT):
            return self.state_dict(*args, **kwargs)

    def _full_post_load_state_dict_hook(self, *args, **kwargs) -> None:
        # We should exit summon_full_params context.
        self._assert_state([TrainingState_.SUMMON_FULL_PARAMS])
        assert getattr(self, '_full_param_ctx', None) is not None
        self._full_param_ctx.__exit__(None, None, None)
        self._full_param_ctx = None

    def _sharded_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Returns the sharded states of the module. Parameters are unflattened and
        sharded, so the resulting state_dict can be used with any parallelism
        (e.g., DPP, model parallelism, and single trainer) after a valid
        resharding.
        """
        with self.set_state_dict_type(StateDictType.SHARDED_STATE_DICT):
            return self.state_dict(self, *args, **kwargs)

    def _full_pre_load_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
    ) -> None:
        # We do not expect to be calling pre-hooks twice without post-hook
        # call in between.
        assert getattr(self, '_full_param_ctx', None) is None
        # Note that it needs writeback=True to persist.
        self._full_param_ctx = self._summon_full_params(
            recurse=False, writeback=True
        )
        self._full_param_ctx.__enter__()
        _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_WRAPPED_MODULE}.")

    def _local_post_load_state_dict_hook(self, *args, **kwargs) -> None:
        pass

    def _local_pre_load_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
    ) -> None:
        """
        This hook finds the local flat_param for this FSDP module from the
        state_dict. The flat_param should be a ShardedTensor. This hook converts
        the ShardedTensor to a tensor. No copy happen unless padding is required.
        """
        _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP_WRAPPED_MODULE}.")
        fqn = f"{prefix}{FSDP_WRAPPED_MODULE}.{FLAT_PARAM}"
        if fqn not in state_dict:
            assert getattr(self._fsdp_wrapped_module, FLAT_PARAM, None) is None, (
                "No flat parameter in state_dict but self._fsdp_wrapped_module.flat_param is not None"
            )
            return
        load_tensor = state_dict[fqn]
        assert isinstance(
            load_tensor, ShardedTensor
        ), "Tensors in local_state_dict should be ShardedTensor."

        # Convert the ShardedTensor to a Tensor.
        shards = load_tensor.local_shards()
        assert len(shards), "load_local_state_dict assume one shard per ShardedTensor."
        load_tensor = cast(torch.Tensor, shards[0].tensor)

        # Get the metada of the flat_param to decide whether to pad the loaded
        # tensor.
        flat_param = self._fsdp_wrapped_module.flat_param
        assert flat_param is not None
        if flat_param._shard_numel_padded not in (0, flat_param.numel()):
            assert load_tensor.numel() < flat_param.numel(), (
                f"Local shard size = {flat_param.numel()} and the tensor in "
                f"the state_dict is {load_tensor.numel()}."
            )
            load_tensor = F.pad(load_tensor, [0, flat_param._shard_numel_padded])
        state_dict[fqn] = load_tensor

    def _sharded_post_load_state_dict_hook(self, *args, **kwargs) -> None:
        pass

    def _sharded_pre_load_state_dict_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
    ) -> None:
        """
        The hook combines the unflattened, sharded parameters (ShardedTensor) to
        a new FlatParameter and shards the new FlatParameter to the local chunk.
        """
        _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_WRAPPED_MODULE}.")
        if not self._fsdp_wrapped_module.has_params:
            return

        if not self._fsdp_wrapped_module.handle.uses_sharded_strategy:
            raise RuntimeError(
                "load_sharded_state_dict can only be called when parameters "
                "are flatten and sharded."
            )

        nonsharded_tensors = []
        # TODO: Reduce the communication by using only one _all_gather_base to
        # gather all the parameters in this layer. This can be achieved by
        # concatenated all the local shards and then append the padding.
        # https://github.com/pytorch/pytorch/issues/77461
        for (param_name, _, module_name) in self._fsdp_wrapped_module.handle.flat_param._param_infos:
            module_name = self._convert_to_wrapped_module_name(module_name)
            fqn = f"{prefix}{FSDP_WRAPPED_MODULE}.{module_name}{param_name}"
            param = state_dict.pop(fqn)

            # All-gather the param (ShardedTensor)
            param, shards = _ext_pre_load_state_dict_transform(param)
            assert len(shards) < 2, (
                f"Expects 0 or 1 shard per rank but got {len(shards)} shards on rank {self.rank}"
            )
            param_numel = param.size().numel()
            dim_0_size = param.size()[0]
            chunk_size = (
                math.ceil(dim_0_size / self.world_size) * param_numel // dim_0_size
            )
            if shards:
                local_tensor = cast(torch.Tensor, shards[0].tensor).flatten()
                if not local_tensor.is_cuda:
                    local_tensor = local_tensor.cuda()
                num_padding = chunk_size - local_tensor.numel()
                if num_padding > 0:
                    local_tensor = F.pad(local_tensor, [0, num_padding])
            else:
                local_tensor = torch.zeros(chunk_size, dtype=param.dtype).cuda()
            tensor = torch.empty(
                chunk_size * self.world_size, dtype=local_tensor.dtype
            ).cuda()
            dist._all_gather_base(tensor, local_tensor, group=self.process_group)
            tensor = tensor.narrow(0, 0, param_numel).reshape(param.size())
            nonsharded_tensors.append(tensor)

        # Create a new flat_param from the loaded, non-sharded tensors.
        flat_param = self._fsdp_wrapped_module.flat_param
        loaded_flat_param = FlatParamHandle.flatten_params(nonsharded_tensors, requires_grad=False)

        # Get the chunk from the loaded flat_param for the local rank.
        loaded_flat_param, num_to_pad = FlatParamHandle._get_shard(
            loaded_flat_param, self.rank, self.world_size,
        )
        loaded_flat_param.to(flat_param.device)
        assert flat_param.numel() == loaded_flat_param.numel(), (
            f"The loaded local chunk has different numel({flat_param.numel()}) "
            f"from the local chunk {flat_param.numel()}."
        )
        assert flat_param._shard_numel_padded == num_to_pad, (
            f"The loaded local chunk has different padding({num_to_pad}) "
            f"from the local chunk {flat_param._shard_numel_padded}."
        )
        state_dict[f"{prefix}_fsdp_wrapped_module.flat_param"] = loaded_flat_param

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()``
        is called. ``self._state_dict_type`` is used to decide what preprocessing
        will be done.
        """
        # Code that is common for all state_dict impls
        self = cast(FullyShardedDataParallel, module)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Dispatch into state_dict specific implementation of pre-hook.
        self._pre_load_state_dict_hook_fn[self._state_dict_type](state_dict, prefix)

    @staticmethod
    def _post_load_state_dict_hook(module: nn.Module, *args: Any) -> None:
        # Code that is common for all state_dict impls
        self = cast(FullyShardedDataParallel, module)
        # Dispatch into state_dict type specific implementation of post-hook for
        # loading state_dict.
        self._post_load_state_dict_hook_fn[self._state_dict_type]()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> NamedTuple:
        """
        The entry point of all three FSDP ``load_state_dict`` APIs. By default,
        calling ``load_state_dict`` on an FSDP module will result in FSDP
        attempting to load a "full" state_dict, i.e. a state_dict consisting of
        full, unsharded, unflattened original module parameters. This requires
        FSDP to load the full parameter context on each rank which could result
        in GPU OOM. As a result, :func:`state_dict_type` API is available to
        configure between ``load_state_dict`` implementations. User can thus use
        ``with self.state_dict_type(self, StateDictType.LOCAL_STATE_DICT)`` context
        manager to load a local state dict checkpoint that will restore only
        local shards of the module. Currently, the only supported
        implementations are ``StateDictType.LOCAL_STATE_DICT`` and
        ``StateDictType.FULL_STATE_DICT`` (default). Please see :func:`state_dict`
        for documentation around creating an FSDP checkpoint.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> import torch
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> from torch.distributed.fsdp import StateDictType
            >>> torch.cuda.set_device(device_id)
            >>> my_module = nn.Linear(...)
            >>> sharded_module = FSDP(my_module)
            >>> checkpoint = torch.load(PATH)
            >>> full_state_dict = checkpoint['full_state_dict']
            >>> with FSDP.state_dict_type(sharded_module, StateDictType.FULL_STATE_DICT):
            >>>     sharded_module.load_state_dict(full_state_dict)
            >>> full_dict.keys()
            >>> odict_keys(['weight', 'bias'])
            >>> # using local state dict
            >>> local_state_dict = checkpoint['local_state_dict']
            >>> with FSDP.state_dict_type(sharded_module, StateDictType.LOCAL_STATE_DICT):
            >>>     sharded_module.load_state_dict(local_state_dict)
            >>> local_dict.keys()
            >>> odict_keys(['flat_param', 'inner.flat_param'])

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives may be used.
        """
        return super().load_state_dict(state_dict, *args)

    def _load_local_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
    ) -> NamedTuple:
        """
        Load states from a flattened, sharded state dictionary.
        """
        with self.state_dict_type(self, StateDictType.LOCAL_STATE_DICT):
            return self.load_state_dict(state_dict, *args)

    def _load_sharded_state_dict(
        self,
        state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"],
        strict: bool = True,
    ) -> NamedTuple:
        """
        Load states from a unflattened, sharded state dictionary.
        """
        with self.set_state_dict_type(StateDictType.SHARDED_STATE_DICT):
            return self.load_state_dict(state_dict, strict)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the forward pass for the wrapped module, inserting FSDP-specific
        pre- and post-forward sharding logic.
        """
        with torch.autograd.profiler.record_function("FullyShardedDataParallel.forward"):
            self._lazy_init()
            args, kwargs = self._fsdp_root_pre_forward(*args, **kwargs)
            unused = None
            unshard_fn = functools.partial(self._pre_forward_unshard, handles=self._handles)
            # Do not free the root's parameters in the post-forward for
            # `FULL_SHARD` with the intention that they are immediately used
            # for backward computation (though this may not be true)
            free_unsharded_flat_params = [
                not self._is_root
                and handle._config.sharding_strategy == HandleShardingStrategy.FULL_SHARD
                for handle in self._handles
            ]
            reshard_fn = functools.partial(
                self._reshard,
                self._handles,
                free_unsharded_flat_params,
            )
            self._pre_forward(self._handles, unshard_fn, unused, unused)
            for handle in self._handles:
                p_assert(
                    handle.flat_param.device == self.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{self.compute_device} but got {handle.flat_param.device}"
                )
            output = self._fsdp_wrapped_module(*args, **kwargs)
            return self._post_forward(self._handles, reshard_fn, unused, unused, output)

    def _pre_forward(
        self,
        handles: List[FlatParamHandle],
        unshard_fn: Optional[Callable],
        module: nn.Module,
        input: Any,
    ):
        """
        Runs the pre-forward logic. This includes an opportunity to unshard
        currently sharded parameters such as those for the current forward and
        registering post-backward hooks for these current parameters.

        Args:
            handles (List[FlatParamHandle]): Handles giving the parameters
                used in the current forward.
            unshard_fn (Optional[Callable]): A callable to unshard any
                currently sharded parameters or ``None`` to not do any
                unsharding.
            module (nn.Module): Unused; expected by the hook signature.
            input (Any): Unused; expected by the hook signature.
        """
        self.training_state = TrainingState_.FORWARD
        self._exec_order_data.record_pre_forward(handles, self.training)
        for handle in handles:
            handle._training_state = HandleTrainingState.FORWARD
        if unshard_fn is not None:
            unshard_fn()
        # Register post-backward hooks to reshard the parameters and
        # reduce-scatter their gradients. They must be re-registered every
        # forward pass in case the `grad_fn` is mutated.
        self._register_post_backward_hooks(handles)

    def _pre_forward_unshard(
        self,
        handles: List[FlatParamHandle],
    ) -> None:
        """Unshards parameters in the pre-forward."""
        if handles:
            self._unshard(handles)
            handles_key = tuple(handles)
            self._needs_pre_forward_unshard[handles_key] = False
            torch.cuda.current_stream().wait_stream(self._streams["all_gather"])
            self._prefetch_handles(handles_key)

    def _post_forward(
        self,
        handles: List[FlatParamHandle],
        reshard_fn: Optional[Callable],
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> Any:
        """
        Runs the post-forward logic. This includes an opportunity to reshard
        currently unsharded parameters such as those used in the current
        forward and registering pre-backward hooks on the forward outputs.

        Args:
            handles (List[FlatParamHandle]): Handles giving the parameters
                used in the current forward.
            reshard_fn (Optional[Callable]): A callable to reshard any
                currently unsharded parameters (e.g. from the current forward)
                or ``None`` to not do any resharding.
            module (nn.Module): Unused; expected by the hook signature.
            input (Any): Unused; exepcted by the hook signature.
            output (Any): Forward pass output; pre-backward hooks are
                registered on the tensors that require gradients in this
                output.

        Postcondition: Each ``FlatParameter`` 's data points to the sharded
        flattened parameter.
        """
        self._exec_order_data.record_post_forward(handles)
        if reshard_fn is not None:
            reshard_fn()
        # Register pre-backward hooks to unshard the flattened parameters
        # for the gradient computation (if needed)
        output = self._register_pre_backward_hooks(output, handles)
        self.training_state = TrainingState_.IDLE
        for handle in handles:
            handle._training_state = HandleTrainingState.IDLE
        return output

    def _cast_forward_inputs(self, *args, **kwargs):
        """Moves the forward inputs to the compute device and casts them to the
        appropriate dtype if needed."""
        # TODO: Do not use the side stream for tensor copies for now;
        # investigate the perf with/without it
        # TODO: For mixed precision, move the inputs to the compute device and
        # cast to reduced-precision in a single `to()` call
        args, kwargs = _to_kwargs(args, kwargs, self.compute_device.index, False)
        args = args[0]
        kwargs = kwargs[0]
        if self._mixed_precision_enabled_for_params():
            input_dtype = self.mixed_precision.param_dtype
            args, kwargs = self._cast_fp_inputs_to_dtype(
                input_dtype, *args, **kwargs,
            )
        return args, kwargs

    def _fsdp_root_pre_forward(self, *args, **kwargs):
        """
        Runs pre-forward logic specific to the root FSDP instance, which should
        run before any individual module's pre-forward. This includes
        synchronizing with the previous iteration and casting the forward
        inputs appropriately. If this is called on a non-root FSDP instance,
        then the forward inputs are returned directly.
        """
        p_assert(self._is_root is not None, "Expects a root FSDP to have been set")
        if not self._is_root:
            return args, kwargs
        if self.forward_prefetch:
            for fsdp_module in self.fsdp_modules(self):
                handles_key = tuple(fsdp_module._handles)
                if handles_key:
                    self._needs_pre_forward_unshard[handles_key] = True
        self._wait_for_previous_optim_step()
        args, kwargs = self._cast_forward_inputs(*args, **kwargs)
        return args, kwargs

    @staticmethod
    @contextlib.contextmanager
    def summon_full_params(
        module,
        recurse: bool = True,
        writeback: bool = True,
        rank0_only: bool = False,
        offload_to_cpu: bool = False,
    ) -> Generator:
        r""" A context manager to expose full params for FSDP instances.
        Can be useful *after* forward/backward for a model to get
        the params for additional processing or checking. It can take a non-FSDP
        module and will summon full params for all contained FSDP modules as
        well as their children, depending on the ``recurse`` argument.

        .. note:: This can be used on inner FSDPs.
        .. note:: This can *not* be used within a forward or backward pass. Nor
            can forward and backward be started from within this context.
        .. note:: Parameters will revert to their local shards after the context
            manager exits, storage behavior is the same as forward.
        .. note:: The full parameters can be modified, but only the portion
            corresponding to the local param shard will persist after the
            context manager exits (unless ``writeback=False``, in which case
            changes will be discarded). In the case where FSDP does not shard
            the parameters, currently only when ``world_size == 1``, or ``NO_SHARD``
            config, the modification is persisted regardless of ``writeback``.
        .. note:: This method works on modules which are not FSDP themselves but
            may contain multiple independent FSDP units. In that case, the given
            arguments will apply to all contained FSDP units.

        .. warning:: Note that ``rank0_only=True`` in conjunction with
            ``writeback=True`` is not currently supported and will raise an
            error. This is because model parameter shapes would be different
            across ranks within the context, and writing to them can lead to
            inconsistency across ranks when the context is exited.

        .. warning:: Note that ``offload_to_cpu`` and ``rank0_only=False`` will
            result in full parameters being redundantly copied to CPU memory for
            GPUs that reside on the same machine, which may incur the risk of
            CPU OOM. It is recommended to use ``offload_to_cpu`` with
            ``rank0_only=True``.

        Args:
            recurse (bool, Optional): recursively summon all params for nested
                FSDP instances (default: True).
            writeback (bool, Optional): if ``False``, modifications to params are
                discarded after the context manager exits;
                disabling this can be slightly more efficient (default: True)
            rank0_only (bool, Optional): if ``True``, full parameters are
                materialized on only global rank 0. This means that within the
                context, only rank 0 will have full parameters and the other
                ranks will have sharded parameters. Note that setting
                ``rank0_only=True`` with ``writeback=True`` is not supported,
                as model parameter shapes will be different across ranks
                within the context, and writing to them can lead to
                inconsistency across ranks when the context is exited.
            offload_to_cpu (bool, Optional): If ``True``, full parameters are
                offloaded to CPU. Note that this offloading currently only
                occurs if the parameter is sharded (which is only not the case
                for world_size = 1 or ``NO_SHARD`` config). It is recommended
                to use ``offload_to_cpu`` with ``rank0_only=True`` to avoid
                redundant copies of model parameters being offloaded to the same CPU memory.
        """
        # Note that we specify root_only as FSDP roots will handle summoning
        # child FSDP instances based on recurse argument.
        root_fsdp_modules = FullyShardedDataParallel.fsdp_modules(
            module, root_only=True
        )
        # Summon all params for all FSDP instances
        with contextlib.ExitStack() as stack:
            for module in root_fsdp_modules:
                stack.enter_context(
                    module._summon_full_params(
                        recurse=recurse,
                        writeback=writeback,
                        rank0_only=rank0_only,
                        offload_to_cpu=offload_to_cpu,
                    )
                )
            # Yield to the caller, with full params in all FSDP instances.
            yield
        # Exiting from the ExitStack will reshard all params.
        return

    @contextlib.contextmanager
    def _summon_full_params(
        self,
        recurse: bool = True,
        writeback: bool = True,
        rank0_only: bool = False,
        offload_to_cpu: bool = False,
    ):
        if writeback and rank0_only:
            raise ValueError(
                "writeback=True and rank0_only=True is not supported, as model "
                "parameter shapes will be different across ranks, and writing "
                "to them can lead to inconsistencies across ranks when the "
                "context is exited."
            )
        if offload_to_cpu and not rank0_only:
            warnings.warn(
                "offload_to_cpu and rank0_only=False will result in "
                "full parameters being redundantly copied to CPU memory for "
                "GPUs that reside on the same machine, which may incur the risk of "
                "CPU OOM. It is recommended to use ``offload_to_cpu`` with "
                "rank0_only=True."
            )

        if recurse:
            with contextlib.ExitStack() as stack:
                for module in self.fsdp_modules(self):
                    stack.enter_context(
                        module._summon_full_params(
                            recurse=False,
                            writeback=writeback,
                            rank0_only=rank0_only,
                            offload_to_cpu=offload_to_cpu,
                        )
                    )
                yield
            return

        torch.cuda.synchronize()
        self._lazy_init()
        self._assert_state([TrainingState_.IDLE])
        for handle in self._handles:
            assert handle._training_state == HandleTrainingState.IDLE
        self.training_state = TrainingState_.SUMMON_FULL_PARAMS
        for handle in self._handles:
            handle._training_state = HandleTrainingState.SUMMON_FULL_PARAMS

        free_unsharded_flat_params = [handle.needs_unshard() for handle in self._handles]
        self._unshard(self._handles)
        torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

        if rank0_only and self.rank != 0:
            # Free the unsharded flattened parameter early
            self._reshard(self._handles, free_unsharded_flat_params)
            try:
                yield
            finally:
                self.training_state = TrainingState_.IDLE
                for handle in self._handles:
                    handle._training_state = HandleTrainingState.IDLE
        else:
            # Unflatten the unsharded flattened parameters
            with contextlib.ExitStack() as stack:
                # Invariant: rank == 0 or !rank0_only
                for handle in self._handles:
                    if offload_to_cpu and handle.uses_sharded_strategy:
                        stack.enter_context(handle.to_cpu())
                # TODO (awgu): This FPW call assumes 1 `FlatParameter`
                stack.enter_context(self._fsdp_wrapped_module.unflatten_as_params())
                try:
                    yield
                finally:
                    stack.close()
                    if writeback:
                        self._write_back_to_local_shard(self._handles)
                    self._reshard(self._handles, free_unsharded_flat_params)
                    self.training_state = TrainingState_.IDLE
                    for handle in self._handles:
                        handle._training_state = HandleTrainingState.IDLE

    @torch.no_grad()
    def _write_back_to_local_shard(self, handles: List[FlatParamHandle]):
        """
        For each handle, writes back the this rank's shard of the unsharded
        flattened parameter to the sharded flattened parameter.

        Precondition: Each handle's ``FlatParameter`` 's data points to the
        padded unsharded flattened parameter.
        """
        for handle in handles:
            # For `NO_SHARD`, `_local_shard` is the unsharded flattened
            # parameter as well
            if not handle.uses_sharded_strategy:
                continue
            assert (
                handle.flat_param.ndim == 1
            ), f"Expects `flat_param` to be flattened but got {handle.flat_param.shape}"
            # Get the unpadded shard instead of the padded shard to persist
            # user changes to the padding (though FSDP does not explicitly
            # support this)
            shard, _ = FlatParamHandle._get_unpadded_shard(handle.flat_param, handle.rank, handle.world_size)
            handle.flat_param._local_shard[:shard.numel()].copy_(shard)

    def named_buffers(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Overrides :meth:`named_buffers()` to intercept buffer names and
        remove all occurrences of the FSDP-specific flattened buffer prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        in_summon_full_params = self.training_state == TrainingState_.SUMMON_FULL_PARAMS
        for buffer_name, buffer in super().named_buffers(*args, **kwargs):
            if in_summon_full_params:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                buffer_name = buffer_name.replace(FSDP_PREFIX, "")
            yield (buffer_name, buffer)

    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Overrides :meth:`named_parameters()` to intercept parameter names and
        remove all occurrences of the FSDP-specific flattened parameter prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        # Determine which logic to use based on the context at call time
        in_summon_full_params = self.training_state == TrainingState_.SUMMON_FULL_PARAMS
        for param_name, param in super().named_parameters(*args, **kwargs):
            if in_summon_full_params:
                # Remove any instances of the FSDP-specific prefix; there can
                # be multiple in the case of nested FSDP modules
                param_name = param_name.replace(FSDP_PREFIX, "")
            yield (param_name, param)

    def _register_pre_backward_hooks(
        self,
        outputs: Any,
        handles: List[FlatParamHandle],
    ) -> Any:
        """
        Registers pre-backward hooks on the tensors that require gradients in
        the forward pass outputs ``outputs``, which were computed using the
        ``FlatParameter`` s of ``handles``.

        Returns:
            Forward pass outputs with pre-backward hooks registered to tensors
            that require gradients.
        """
        # If there is no gradient computation, then there is no need for
        # pre-backward logic
        if not torch.is_grad_enabled():
            return outputs

        if self._is_root:
            self._post_backward_callback_queued = False  # only defined on the root

        handles_key = tuple(handles)
        if handles_key:
            # Since these handles' `FlatParameter`s participated in a forward,
            # we conservatively assume that they will be used in the backward
            self._needs_pre_backward_unshard[handles_key] = False
            self._ran_pre_backward_hook[handles_key] = False

        def _pre_backward_hook(_handles: List[FlatParamHandle], *unused: Any) -> None:
            """Prepares ``_handles`` 's ``FlatParameter`` s for gradient
            computation."""
            _handles_key = tuple(_handles)  # avoid shadowing `handles_key`
            # Only run the pre-backward hook once per group of handles involved
            # in the same module forward computation
            if _handles_key and self._ran_pre_backward_hook.get(_handles_key, False):
                return

            with torch.autograd.profiler.record_function(
                "FullyShardedDataParallel._pre_backward_hook"
            ):
                # Queue the post-backward callback once for the root FSDP
                # instance to attach it to the outermost backward graph task so
                # that it is called after all backward calls complete
                if self._is_root and not self._post_backward_callback_queued:
                    self._queue_wait_for_post_backward()
                elif _handles_key:
                    self._assert_state([TrainingState_.IDLE])
                self.training_state = TrainingState_.BACKWARD_PRE
                # Queueing the post-backward callback is the only logic that is
                # not per-handle in the pre-backward hook, so we can return
                # early here if there are no handles.
                if not _handles_key:
                    return
                for handle in _handles:
                    handle._training_state = HandleTrainingState.BACKWARD_PRE

                # If the handles have been prefetched, this `_unshard()` simply
                # switches to using the unsharded parameter
                self._unshard(_handles)
                torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

                # Set this to `False` to ensure that a mistargeted prefetch
                # does not actually unshard these handles
                self._needs_pre_backward_unshard[_handles_key] = False
                self._prefetch_handles(_handles_key)
                for handle in _handles:
                    handle.prepare_gradient()
                self._ran_pre_backward_hook[_handles_key] = True

        def _register_hook(t: torch.Tensor) -> torch.Tensor:
            if t.requires_grad:
                t.register_hook(functools.partial(_pre_backward_hook, handles))
                self._needs_pre_backward_unshard[handles_key] = True
            return t

        return _apply_to_tensors(_register_hook, outputs)

    def _register_post_backward_hooks(
        self,
        handles: List[FlatParamHandle],
    ) -> None:
        """
        Registers post-backward hooks on the ``FlatParameter`` s'
        ``AccumulateGrad`` objects to reshard and to reduce-scatter gradients.

        The ``AccumulateGrad`` object represents the last function that
        finalizes the ``FlatParameter`` 's gradient, so it only runs after its
        entire gradient computation has finished.

        We register the post-backward hook only once in the *first* forward
        that a ``FlatParameter`` participates in. This relies on the
        ``AccumulateGrad`` object being preserved through multiple forwards.
        """
        # If there is no gradient computation, then there is no need for
        # post-backward logic
        if not torch.is_grad_enabled():
            return
        for handle in handles:
            flat_param = handle.flat_param
            already_registered = hasattr(flat_param, "_post_backward_hook_state")
            if already_registered or not flat_param.requires_grad:
                continue
            # Get the `AccumulateGrad` object
            temp_flat_param = flat_param.expand_as(flat_param)
            p_assert(
                temp_flat_param.grad_fn is not None,
                "The `grad_fn` is needed to access the `AccumulateGrad` and "
                "register the post-backward hook"
            )
            acc_grad = temp_flat_param.grad_fn.next_functions[0][0]
            hook_handle = acc_grad.register_hook(
                functools.partial(self._post_backward_hook, handle)
            )
            flat_param._post_backward_hook_state = (acc_grad, hook_handle)  # type: ignore[attr-defined]

    @torch.no_grad()
    def _post_backward_hook(
        self,
        handle: FlatParamHandle,
        *unused: Any,
    ) -> None:
        """
        Reduce-scatters the gradient of ``handle`` 's ``FlatParameter``.

        Precondition: The ``FlatParameter`` 's ``.grad`` attribute contains the
        unsharded gradient for the local batch.

        Postcondition:
        - If using ``NO_SHARD``, then the ``.grad`` attribute is the reduced
        unsharded gradient.
        - Otherwise, the ``_saved_grad_shard`` attribute is the reduced sharded
        gradient (accumulating with any existing gradient).
        """
        param = handle.flat_param
        param._post_backward_called = True
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel._post_backward_hook"
        ):
            # First hook callback will see PRE state. If we have multiple params,
            # then subsequent hook callbacks will see POST state.
            self._assert_state([TrainingState_.BACKWARD_PRE, TrainingState_.BACKWARD_POST])
            self.training_state = TrainingState_.BACKWARD_POST
            handle._training_state = HandleTrainingState.BACKWARD_POST

            if self._use_param_exec_order_policy() and self._param_exec_order_prep_stage:
                # In self._fsdp_params_exec_order, the parameters are ordered based on
                # the execution order in the backward pass in the first iteration.
                self._fsdp_params_exec_order.append(param)

            if param.grad is None:
                return
            if param.grad.requires_grad:
                raise RuntimeError(
                    "FSDP only works with gradients that don't require gradients"
                )

            free_unsharded_flat_param = self._should_free_unsharded_flat_param(handle)
            self._reshard([handle], [free_unsharded_flat_param])

            # TODO (awgu): Post-backward prefetching does not support the
            # multiple handles per module case (which was why we keyed by
            # *tuple*). The post-backward hook runs per handle, not per group
            # of handles. To generalize this, we may need a 2-level mapping,
            # where we map each individual handle to its groups of handles and
            # then from the groups of handles to their indices in the order.
            handles_key = (handle,)
            self._prefetch_handles(handles_key)

            if not self._sync_gradients:
                return

            # Wait for all ops in the current stream (e.g. gradient
            # computation) to finish before reduce-scattering the gradient
            self._streams["post_backward"].wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(self._streams["post_backward"]):
                orig_grad_data = param.grad.data
                if (
                    self._mixed_precision_enabled_for_reduce()
                    and not self._low_precision_hook_enabled()
                ):
                    # Cast gradient to precision in which it should be communicated.
                    # If a low precision hook is registered and reduce_dtype is specified
                    # in `MixedPrecision`, communication hook will take care of
                    # casting to lower precision and back.
                    # TODO: Make this a communication hook when communication hooks
                    # are implemented for FSDP. Note that this is a noop if the
                    # reduce_dtype matches the param dtype.
                    param.grad.data = param.grad.data.to(self.mixed_precision.reduce_dtype)

                if self._exec_order_data.is_first_iter:
                    # For all sharding strategies communication is performed through `_communication_hook`:
                    # default comm hooks are: `reduce_scatter` for sharded strategies and
                    # `all_reduce` for non-sharded strategies. This checks asserts that `_communication_hook`
                    # and `_communication_hook_state`, required for communication not `None`.`
                    p_assert(
                        self._communication_hook is not None,
                        "Communication hook should not be None"
                    )
                    p_assert(
                        self._communication_hook_state is not None,
                        "Communication hook state should not be None"
                    )
                grad = param.grad.data
                if handle.uses_sharded_strategy:
                    # We clear `param.grad` to permit repeated gradient
                    # computations when this FSDP module is called multiple times.
                    # This is to avoid a race among multiple re-entrant backward
                    # passes. For example, the second backward pass computation
                    # precedes ahead of the first backward pass reduction, which is
                    # possible since the reduction is in a different stream and is
                    # async. Then, the first backward pass may be incorrectly
                    # reducing the second backward pass's `param.grad`.
                    # The reduced gradients are accumulated in
                    # `param._saved_grad_shard`, and the gradient reductions can
                    # happen in arbitrary order, though we tolerate this due to the
                    # (approximate) commutativity of floating-point addition.
                    param.grad = None
                    grad_flatten = torch.flatten(grad)
                    chunks = list(grad_flatten.chunk(self.world_size))
                    num_pad = self.world_size * chunks[0].numel() - grad.numel()
                    input_flattened = F.pad(grad_flatten, [0, num_pad])
                    output = torch.zeros_like(chunks[0])
                    self._communication_hook(self._communication_hook_state, input_flattened, output)

                    self._cast_grad_to_param_dtype(output, param)

                    # To support gradient accumulation outside `no_sync()`, we save
                    # the gradient data to `param._saved_grad_shard` before the
                    # backward pass, accumulate gradients into it here, and set
                    # `param.grad` with the accumulated value at the end of the
                    # backward pass in preparation for the optimizer step.
                    accumulate_grad = hasattr(param, "_saved_grad_shard")
                    if accumulate_grad:
                        p_assert(
                            param._saved_grad_shard.shape == output.shape,  # type: ignore[attr-defined]
                            "Shape mismatch when accumulating gradients: "  # type: ignore[attr-defined]
                            f"existing grad shape={param._saved_grad_shard.shape} "
                            f"new grad shape={output.shape}"  # type: ignore[attr-defined]
                        )
                        p_assert(
                            param._saved_grad_shard.device == output.device,  # type: ignore[attr-defined]
                            "Device mismatch when accumulating gradients: "  # type: ignore[attr-defined]
                            f"existing grad device={param._saved_grad_shard.device} "
                            f"new grad device={output.device}"  # type: ignore[attr-defined]
                        )
                        param._saved_grad_shard += output  # type: ignore[attr-defined]
                    else:
                        param._saved_grad_shard = output  # type: ignore[attr-defined]
                    grad = param._saved_grad_shard  # type: ignore[attr-defined]
                else:
                    if self.sharding_strategy == ShardingStrategy.NO_SHARD:
                        self._communication_hook(self._communication_hook_state, param.grad)

                    # For NO_SHARD keeping grads in the reduced precision, we
                    # can simply omit the cast as needed, we can't do this for
                    # other sharding strategies because grad field is assigned
                    # in _finalize_params. TODO (rvarm1) this divergence in
                    # logic is not ideal.
                    if not self._mixed_precision_keep_low_precision_grads():
                        self._cast_grad_to_param_dtype(param.grad, param)

                # Regardless of sharding or not, offload the grad to CPU if we are
                # offloading params. This is so param and grad reside on same device
                # which is needed for the optimizer step.
                if handle._config.offload_params:
                    # We specify non_blocking=True
                    # and ensure the appropriate synchronization is done by waiting
                    # streams in _wait_for_post_backward.
                    param._cpu_grad.copy_(  # type: ignore[attr-defined]
                        grad.detach(), non_blocking=True
                    )
                    # Don't let this memory get reused until after the transfer.
                    grad.data.record_stream(torch.cuda.current_stream())

                # After _post_backward_hook returns, orig_grad_data will eventually
                # go out of scope, at which point it could otherwise be freed for
                # further reuse by the main stream while the div/reduce_scatter/copy
                # are underway in the post_backward stream. See:
                # github.com/NVIDIA/apex/blob/master/apex/parallel/distributed.py
                orig_grad_data.record_stream(self._streams["post_backward"])

    def _cast_grad_to_param_dtype(
        self,
        grad: torch.Tensor,
        param: FlatParameter,
    ):
        """
        Casts gradient ``grad`` back to the full parameter dtype so that the
        optimizer step runs with that dtype. This performs an actual cast if
        1. parameters were in reduced precision during the forward since then
        gradients would be in that reduced precision, or
        2. parameters were not in reduced precision but gradients were in
        reduced precision for communication.
        However, if a low precision communication hook is registered, then this
        dtype cast happens in the hook instead.
        """
        self._assert_state(TrainingState_.BACKWARD_POST)
        if (
            not self._low_precision_hook_enabled()
            and (
                self._mixed_precision_enabled_for_params()
                or self._mixed_precision_enabled_for_reduce()
            )
        ):
            low_prec_grad_data = grad.data
            grad.data = grad.data.to(dtype=param.dtype)
            # Do not let the low precision gradient memory get reused until
            # the cast to full parameter precision completes
            low_prec_grad_data.record_stream(torch.cuda.current_stream())

    def _should_free_unsharded_flat_param(self, handle: FlatParamHandle):
        return (
            (self._sync_gradients and handle.uses_sharded_strategy)
            or handle._config.sharding_strategy == HandleShardingStrategy.FULL_SHARD
        )

    def _queue_wait_for_post_backward(self) -> None:
        """
        Queues a post-backward callback from the root FSDP instance, which
        should happen at the beginning of its pre-backward.
        """
        p_assert(
            self._is_root,
            "`_queue_wait_for_post_backward()` should be called on the root FSDP instance"
        )
        if self._post_backward_callback_queued:
            return
        self._assert_state([TrainingState_.IDLE])
        self._post_backward_callback_queued = True
        Variable._execution_engine.queue_callback(self._wait_for_post_backward)

    @torch.no_grad()
    def _wait_for_post_backward(self) -> None:
        """Wait for post-backward to finish. Only called on root instance."""
        assert self._is_root, "_wait_for_post_backward can only be called on root."
        # Root's training state might be backward_pre or backward_post depending on
        # if root parameter's post backward hook was called. The post-backward hook
        # may not have been called if gradient was not computed for this param/FSDP
        # module.

        if self._sync_gradients:
            torch.cuda.current_stream().wait_stream(self._streams["post_backward"])
            if self.cpu_offload.offload_params:
                # We need to wait for the non-blocking GPU ->
                # CPU grad transfers to finish. We need to do this for GPU -> CPU
                # copies because when grad is on CPU, it won't wait for any CUDA
                # stream to finish GPU -> CPU copies unless we explicitly block the
                # host-side with synchronize().
                torch.cuda.current_stream().synchronize()
        self._exec_order_data.next_iter()

        # A backward pass is done, clean up below.
        def _catch_all_reshard(fsdp_module: FullyShardedDataParallel) -> None:
            """
            Reshards full parameters that may have not been resharded in
            post_backward_hook. This can happen when an FSDP module's output
            is used in forward so its pre-backward fires unsharding the param,
            but post-backward does not fire since the output was not ultimately
            used in loss computation so FSDP parameter did not get a gradient.
            """
            # Note that we wrap resharding logic in a try-catch as a defensive
            # approach, as if an error is thrown, we are in the backwards pass,
            # and autograd would not print out much useful info about the actual
            # error hit.
            try:
                free_unsharded_flat_params: List[bool] = []
                handles_to_reshard: List[FlatParamHandle] = []
                for handle in fsdp_module._handles:
                    # TODO: This already-resharded check is brittle:
                    # https://github.com/pytorch/pytorch/issues/83956
                    already_resharded = (
                        handle.flat_param.data_ptr() == handle.flat_param._local_shard.data_ptr()
                    )
                    if already_resharded:
                        continue
                    free_unsharded_flat_params.append(self._should_free_unsharded_flat_param(handle))
                    handles_to_reshard.append(handle)
                self._reshard(handles_to_reshard, free_unsharded_flat_params)
            except Exception as e:
                p_assert(
                    False,
                    f"Got exception while resharding module {fsdp_module}: {str(e)}",
                    raise_assertion_error=False
                )
                raise e

        def _finalize_params(fsdp_module: FullyShardedDataParallel) -> None:
            """Helper used below on all fsdp modules."""
            for handle in fsdp_module._handles:
                p = handle.flat_param
                if p.requires_grad:
                    if hasattr(p, "_post_backward_hook_state"):
                        p_assert(
                            len(p._post_backward_hook_state) == 2,  # type: ignore[attr-defined]
                            "p._post_backward_hook_state fields are not valid."
                        )
                        p._post_backward_hook_state[1].remove()  # type: ignore[attr-defined]
                        delattr(p, "_post_backward_hook_state")
                    # Preserve the gradient accumulation state if not
                    # synchronizing: `p.grad` remains the unsharded gradient
                    # accumulated from prior `no_sync()` iterations, and
                    # `p._saved_grad_shard` remains the sharded gradient from
                    # the last synchronized iteration
                    if not self._sync_gradients:
                        continue
                    # Set `p.grad` as needed to ensure optimizer correctness
                    # since optimizers operate on the `grad` attribute
                    if hasattr(p, "_cpu_grad"):
                        p_assert(
                            p.device == torch.device("cpu"),
                            f"Device mismatch: p={p.device} "  # type: ignore[attr-defined]
                            f"p._cpu_grad={p._cpu_grad}"
                        )
                        p.grad = p._cpu_grad  # type: ignore[attr-defined]
                    elif hasattr(p, "_saved_grad_shard"):
                        p_assert(
                            p.device == p._saved_grad_shard.device,  # type: ignore[attr-defined]
                            f"Device mismatch: p={p.device} "  # type: ignore[attr-defined]
                            f"p._saved_grad_shard={p._saved_grad_shard.device}"
                        )
                        # Check if post-backward was called for this param (FSDP unit).
                        # TODO: This logic will have to be revisited when non-recursive wrapping
                        # lands. If it was not called, there is no new gradient to accumulate
                        if p._post_backward_called:
                            p.grad = p._saved_grad_shard
                            if fsdp_module._mixed_precision_keep_low_precision_grads():
                                p.grad.data = p.grad.to(
                                    fsdp_module.mixed_precision.param_dtype
                                )
                    else:
                        p_assert(
                            not handle.uses_sharded_strategy or not p._post_backward_called,
                            "All sharded parameters that received a gradient "
                            "should use `_saved_grad_shard`"
                        )
                    if hasattr(p, "_saved_grad_shard"):
                        delattr(p, "_saved_grad_shard")

                    p_assert(
                        hasattr(p, '_post_backward_called'),
                        "Expected flag _post_backward_called to be set on param."
                    )
                    # Reset _post_backward_called in preparation for the next iteration.
                    p._post_backward_called = False

        # Update root and nested FSDP's hooks and flags.
        for m in self.fsdp_modules(self):  # includes self
            _finalize_params(m)
            _catch_all_reshard(m)
            m._ran_pre_backward_hook.clear()
            m.training_state = TrainingState_.IDLE
            for handle in m._handles:
                handle._training_state = HandleTrainingState.IDLE
            m._handles_prefetched.clear()
            if m._is_root:
                # reset this flag for cases like "one forward pass + multiple backward passes"
                self._post_backward_callback_queued = False

        if self._use_param_exec_order_policy() and self._param_exec_order_prep_stage:
            self._param_exec_order_policy_second_iter_init()

    def _param_exec_order_policy_second_iter_init(self) -> None:
        self._param_exec_order_prep_stage = False
        # Let the parameters in self._fsdp_params_exec_order ordered based on
        # the execution order in the forward pass.
        self._fsdp_params_exec_order.reverse()
        for m in self.modules():
            if m is not self and isinstance(m, FullyShardedDataParallel):
                assert hasattr(
                    m, "_param_exec_order_policy"
                ), "Non-root FSDP modules should also have _param_exec_order_policy attribute"
                assert hasattr(
                    m, "_param_exec_order_prep_stage"
                ), "Non-root FSDP modules should also have _param_exec_order_prep_stage attribute"
                m._param_exec_order_prep_stage = False
        # TODO (linjianma): Construct a fsdp_wrap_map whose keys are all children modules with a FSDP wrap,
        # and values are its FSDP wraps. These children FSDP wraps will be detached from the root FSDP module
        # and will be used to schedule the parameters (rebuild_full_params and reshard).
        # TODO (linjianma): Remove all internal FSDP wraps from the root FSDP module.
        # TODO (linjianma): Based on self._fsdp_params_exec_order, get the information
        # needed to patch the forward() function of each key in the fsdp_wrap_map. The rules are as follows:
        # 1: Before each forward(), rebuild_full_params of all parameters that are currently sharded and
        # will be used in the forward, and reshard all parameters that are currently full and will not be
        # used in the next forward()
        # 2: After each forward(), reshard all parameters just used in the forward, and rebuild_full_params of
        # all parameters that will be used next.
        # TODO (linjianma): Patch the forward of each model in the keys
        # of fsdp_wrap_map based on the information above.

    def _assert_state(self, state: Union[TrainingState_, List[TrainingState_]]) -> None:
        """Assert we are in the given state."""
        # Since assert can be turned off and this error checking
        # is really important, we use explicit error checking
        # and raise a ValueError if needed.
        if isinstance(state, TrainingState_):
            state = [state]
        if self.training_state not in state:
            msg = (
                f"expected to be in states {state} but current state "
                f"is {self.training_state}"
            )
            # In case we are failing in the context of autograd hook, asserting
            # may not generate useful msg. So, let's print it to be sure.
            if self.rank == 0:
                print(f"Asserting FSDP instance is: {self}")
                print(f"ERROR: {msg}")
                traceback.print_stack()
            raise ValueError(msg)

    @contextmanager
    def no_sync(self) -> Generator:
        """
        A context manager to disable gradient synchronizations across FSDP
        instances. Within this context, gradients will be accumulated in module
        variables, which will later be synchronized in the first
        forward-backward pass after exiting the context. This should only be
        used on the root FSDP instance and will recursively apply to all
        children FSDP instances.

        .. note:: This likely results in higher memory usage because FSDP will
            accumulate the full model gradients (instead of gradient shards)
            until the eventual sync.

        .. note:: When used with CPU offloading, the gradients will not be
            offloaded to CPU when inside the context manager. Instead, they
            will only be offloaded right after the eventual sync.
        """
        self._lazy_init()
        assert self._is_root, "`no_sync()` on inner FSDP instances is not supported"
        self._assert_state(TrainingState_.IDLE)
        old_flags = []
        for m in self.modules():
            if isinstance(m, FullyShardedDataParallel):
                old_flags.append((m, m._sync_gradients))
                m._sync_gradients = False
        try:
            yield
        finally:
            for m, old_flag in old_flags:
                assert not m._sync_gradients, (
                    "`_sync_gradients` was incorrectly set to "
                    "`True` while in the `no_sync()` context manager"
                )
                m._sync_gradients = old_flag

    @property
    def params_with_grad(self) -> List[Parameter]:
        """
        Recursively returns a list of all module parameters that have a gradient.
        """
        return [p for p in self.parameters() if p.grad is not None]

    @torch.no_grad()
    def clip_grad_norm_(
        self, max_norm: Union[float, int], norm_type: Union[float, int] = 2.0
    ) -> None:
        """
        Clip all gradients at this point in time. The norm is computed over all
        gradients together, as if they were concatenated into a single vector.
        Gradients are modified in-place.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'``
                for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).

        .. note:: This is analogous to ``torch.nn.utils.clip_grad_norm_`` but
            handles the partitioning and multiple devices per rank under the
            hood. The default torch util is not applicable here, because each
            rank only has a partial view of all the grads in the model, so
            calling it for FSDP models would lead to different scaling being
            applied per subset of model parameters.

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives will be used.
        """
        self._lazy_init()
        self._wait_for_previous_optim_step()
        assert self._is_root, "clip_grad_norm should only be called on the root (parent) instance"
        self._assert_state(TrainingState_.IDLE)

        max_norm = float(max_norm)
        norm_type = float(norm_type)
        # Computes the max norm for this shard's gradients and sync's across workers
        local_norm = _calc_grad_norm(self.params_with_grad, norm_type).cuda()  # type: ignore[arg-type]
        if norm_type == math.inf:
            total_norm = local_norm
            dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=self.process_group)
        else:
            total_norm = local_norm ** norm_type
            dist.all_reduce(total_norm, group=self.process_group)
            total_norm = total_norm ** (1.0 / norm_type)

        if self.cpu_offload:
            total_norm = total_norm.cpu()

        clip_coef = torch.tensor(max_norm, dtype=total_norm.dtype, device=total_norm.device) / (total_norm + 1e-6)
        if clip_coef < 1:
            # multiply by clip_coef, aka, (max_norm/total_norm).
            for p in self.params_with_grad:
                assert p.grad is not None
                p.grad.detach().mul_(clip_coef.to(p.grad.device))

    @staticmethod
    def _warn_optim_input(optim_input):
        if optim_input is not None:
            warnings.warn(
                "The `optim_input` argument is deprecated and will be removed after PyTorch 1.13. You may remove it "
                "from your code without changing its functionality."
            )

    @staticmethod
    def _is_using_optim_input(optim_input, optim) -> bool:
        if optim_input is None and optim is None:
            # Use the default behavior of `optim_input``
            return True
        if optim_input is not None:
            # Use the `optim_input` code path
            return True
        # Use the `optim` code path
        return False

    @staticmethod
    def full_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_input: Optional[Union[
            List[Dict[str, Any]], Iterable[torch.nn.Parameter],
        ]] = None,
        rank0_only: bool = True,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, Any]:
        """
        Consolidates the full optimizer state on rank 0 and returns it
        as a :class:`dict` following the convention of
        :meth:`torch.optim.Optimizer.state_dict`, i.e. with keys ``"state"``
        and ``"param_groups"``. The flattened parameters in ``FSDP`` modules
        contained in ``model`` are mapped back to their unflattened parameters.

        .. warning:: This needs to be called on all ranks since synchronization
            primitives are used. However, if ``rank0_only=True``, then the
            state dict is only populated on rank 0, and all other ranks return
            an empty :class:`dict`.

        .. warning:: Unlike ``torch.optim.Optimizer.state_dict()``, this method
            uses full parameter names as keys instead of parameter IDs.

        .. note:: Like in :meth:`torch.optim.Optimizer.state_dict`, the tensors
            contained in the optimizer state dict are not cloned, so there may
            be aliasing surprises. For best practices, consider saving the
            returned optimizer state dict immediately, e.g. using
            ``torch.save()``.

        Args:
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                were passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer ``optim`` representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            rank0_only (bool): If ``True``, saves the populated :class:`dict`
                only on rank 0; if ``False``, saves it on all ranks. (Default:
                ``True``)
            group (dist.ProcessGroup): Model's process group or ``None`` if using
                the default process group. (Default: ``None``)

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            ``model`` 's original unflattened parameters and including keys
            "state" and "param_groups" following the convention of
            :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=True``,
            then nonzero ranks return an empty :class:`dict`.
        """
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input, optim,
        )
        return _optim_state_dict(
            model=model,
            optim=optim,
            optim_input=optim_input,
            rank0_only=rank0_only,
            shard_state=False,
            group=group,
            using_optim_input=using_optim_input,
        )

    @staticmethod
    def sharded_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]], Iterable[torch.nn.Parameter],
            ]
        ] = None,
        group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, Any]:
        """
        The API is similar to :meth:`full_optim_state_dict` but this API chunks
        all non-zero-dimension states to :class:`ShardedTensor` to save memory.
        This API should only be used when the model ``state_dict`` is derived
        with the context manager ``with state_dict_type(SHARDED_STATE_DICT):``.

        For the detailed usage, refer to :meth:`full_optim_state_dict`.

        .. warning:: The returned state dict contains ``ShardedTensor`` and
            cannot be directly used by the regular ``optim.load_state_dict``.
        """
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input, optim,
        )
        # TODO: The ultimate goal of the optimizer state APIs should be the same
        # as state_dict/load_state_dict -- using one API to get optimizer states
        # and one API to load optimizer states. ``state_dict_type`` will be used
        # to decide which optimizer states should be returned.
        # There are currently two APIs to load a full optimizer state. So the
        # first step of the unification is to merge the two full optimizer state
        # loading APIs.
        # Task: https://github.com/pytorch/pytorch/issues/82232
        return _optim_state_dict(
            model=model,
            optim=optim,
            optim_input=optim_input,
            rank0_only=False,
            shard_state=True,
            group=group,
            using_optim_input=using_optim_input,
        )

    @staticmethod
    def shard_full_optim_state_dict(
        full_optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]], Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Shards the full optimizer state dict ``full_optim_state_dict`` by
        remapping the state to flattened parameters instead of unflattened
        parameters and restricting to only this rank's part of the optimizer
        state. The first argument should be the return value of
        :meth:`full_optim_state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> model, optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(model, optim)
            >>> torch.save(full_osd, PATH)
            >>> # Define new model with possibly different world size
            >>> new_model, new_optim = ...
            >>> full_osd = torch.load(PATH)
            >>> sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, new_model)
            >>> new_optim.load_state_dict(sharded_osd)

        .. note:: Both :meth:`shard_full_optim_state_dict` and
            :meth:`scatter_full_optim_state_dict` may be used to get the
            sharded optimizer state dict to load. Assuming that the full
            optimizer state dict resides in CPU memory, the former requires
            each rank to have the full dict in CPU memory, where each rank
            individually shards the dict without any communication, while the
            latter requires only rank 0 to have the full dict in CPU memory,
            where rank 0 moves each shard to GPU memory (for NCCL) and
            communicates it to ranks appropriately. Hence, the former has
            higher aggregate CPU memory cost, while the latter has higher
            communication cost.

        Args:
            full_optim_state_dict (Dict[str, Any]): Optimizer state dict
                corresponding to the unflattened parameters and holding the
                full non-sharded optimizer state.
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                correspond to the optimizer state in ``full_optim_state_dict``.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            optim (Optional[torch.optim.Optimizer]): Optimizer that will load
                the state dict returned by this method. This is the preferred
                argument to use over ``optim_input``. (Default: ``None``)

        Returns:
            Dict[str, Any]: The full optimizer state dict now remapped to
            flattened parameters instead of unflattened parameters and
            restricted to only include this rank's part of the optimizer state.
        """
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input, optim,
        )
        sharded_osd = _flatten_optim_state_dict(
            full_optim_state_dict, model, True,
        )
        return _rekey_sharded_optim_state_dict(
            sharded_osd, model, optim, optim_input, using_optim_input,
        )

    @staticmethod
    def flatten_sharded_optim_state_dict(
        sharded_optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]], Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        The API is similar to :meth:`shard_full_optim_state_dict`. The only
        difference is that the input ``sharded_optim_state_dict`` should be
        returned from :meth:`sharded_optim_state_dict`. Therefore, there will
        be all-gather calls on each rank to gather ``ShardedTensor`` s.

        Args:
            sharded_optim_state_dict (Dict[str, Any]): Optimizer state dict
                corresponding to the unflattened parameters and holding the
                sharded optimizer state.
            model (torch.nn.Module):
                Refer to :meth:``shard_full_optim_state_dict``.

        Returns:
            Refer to :meth:`shard_full_optim_state_dict`.
        """
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input, optim,
        )
        # TODO: The implementation is the same as ``shard_full_optim_state_dict``.
        # See the TODO in ``shard_full_optim_state_dict`` for the future
        # unification plan.
        flattened_osd = _flatten_optim_state_dict(
            sharded_optim_state_dict,
            model=model,
            shard_state=True,
        )
        return _rekey_sharded_optim_state_dict(
            flattened_osd, model, optim, optim_input, using_optim_input,
        )

    @staticmethod
    def scatter_full_optim_state_dict(
        full_optim_state_dict: Optional[Dict[str, Any]],
        model: torch.nn.Module,
        optim_input: Optional[Union[
            List[Dict[str, Any]], Iterable[torch.nn.Parameter],
        ]] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        group: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Scatters the full optimizer state dict from rank 0 to all other ranks,
        returning the sharded optimizer state dict on each rank. The return
        value is the same as :meth:`shard_full_optim_state_dict`, and on rank
        0, the first argument should be the return value of
        :meth:`full_optim_state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> model, optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(model, optim)  # only non-empty on rank 0
            >>> # Define new model with possibly different world size
            >>> new_model, new_optim, new_group = ...
            >>> sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, new_model, group=new_group)
            >>> new_optim.load_state_dict(sharded_osd)

        .. note:: Both :meth:`shard_full_optim_state_dict` and
            :meth:`scatter_full_optim_state_dict` may be used to get the
            sharded optimizer state dict to load. Assuming that the full
            optimizer state dict resides in CPU memory, the former requires
            each rank to have the full dict in CPU memory, where each rank
            individually shards the dict without any communication, while the
            latter requires only rank 0 to have the full dict in CPU memory,
            where rank 0 moves each shard to GPU memory (for NCCL) and
            communicates it to ranks appropriately. Hence, the former has
            higher aggregate CPU memory cost, while the latter has higher
            communication cost.

        Args:
            full_optim_state_dict (Optional[Dict[str, Any]]): Optimizer state
                dict corresponding to the unflattened parameters and holding
                the full non-sharded optimizer state if on rank 0; the argument
                is ignored on nonzero ranks.
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                correspond to the optimizer state in ``full_optim_state_dict``.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            optim (Optional[torch.optim.Optimizer]): Optimizer that will load
                the state dict returned by this method. This is the preferred
                argument to use over ``optim_input``. (Default: ``None``)
            group (dist.ProcessGroup): Model's process group or ``None`` if
                using the default process group. (Default: ``None``)

        Returns:
            Dict[str, Any]: The full optimizer state dict now remapped to
            flattened parameters instead of unflattened parameters and
            restricted to only include this rank's part of the optimizer state.
        """
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input, optim,
        )
        # Try to use the passed-in process group, the model's process group,
        # or the default process group (i.e. `None`) in that priority order
        if group is None and hasattr(model, "process_group"):
            group = model.process_group
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        # Check for a valid broadcast device, preferring GPU when available
        using_nccl = dist.distributed_c10d._check_for_nccl_backend(group)
        broadcast_device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        if using_nccl and not torch.cuda.is_available():
            raise RuntimeError("NCCL requires a GPU for collectives")
        # Flatten the optimizer state dict and construct a copy with the
        # positive-dimension tensors' shapes in place of the tensors themselves
        # since those tensors will be broadcast separately to avoid copying
        if rank == 0:
            if full_optim_state_dict is None:
                raise ValueError("Rank 0 must pass in the full optimizer state dict")
            flat_osd = _flatten_optim_state_dict(
                full_optim_state_dict,
                model=model,
                shard_state=False,
            )
            processed_osd = _process_pos_dim_tensor_state(flat_osd, world_size)
        # Broadcast the optim state dict without positive-dimension tensor
        # state and the FSDP parameter IDs from rank 0 to all ranks
        processed_osd = _broadcast_processed_optim_state_dict(
            processed_osd if rank == 0 else None, rank, group,
        )
        # Broadcast positive-dimension tensor state (both sharded tensors for
        # FSDP parameters and unsharded tensors for non-FSDP parameters)
        sharded_osd = _broadcast_pos_dim_tensor_states(
            processed_osd, flat_osd if rank == 0 else None, rank, world_size,
            group, broadcast_device,
        )
        # Rekey the optimizer state dict to use parameter IDs according to this
        # rank's `optim`
        sharded_osd = _rekey_sharded_optim_state_dict(
            sharded_osd, model, optim, optim_input, using_optim_input,
        )
        return sharded_osd

    @staticmethod
    def rekey_optim_state_dict(
        optim_state_dict: Dict[str, Any],
        optim_state_key_type: OptimStateKeyType,
        model: torch.nn.Module,
        optim_input: Optional[Union[
            List[Dict[str, Any]], Iterable[torch.nn.Parameter],
        ]] = None,
        optim: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Re-keys the optimizer state dict ``optim_state_dict`` to use the key
        type ``optim_state_key_type``. This can be used to achieve
        compatibility between optimizer state dicts from models with FSDP
        instances and ones without.

        To re-key an FSDP full optimizer state dict (i.e. from
        :meth:`full_optim_state_dict`) to use parameter IDs and be loadable to
        a non-wrapped model::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> wrapped_model, wrapped_optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(wrapped_model, wrapped_optim)
            >>> nonwrapped_model, nonwrapped_optim = ...
            >>> rekeyed_osd = FSDP.rekey_optim_state_dict(full_osd, OptimStateKeyType.PARAM_ID, nonwrapped_model)
            >>> nonwrapped_optim.load_state_dict(rekeyed_osd)

        To re-key a normal optimizer state dict from a non-wrapped model to be
        loadable to a wrapped model::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> nonwrapped_model, nonwrapped_optim = ...
            >>> osd = nonwrapped_optim.state_dict()
            >>> rekeyed_osd = FSDP.rekey_optim_state_dict(osd, OptimStateKeyType.PARAM_NAME, nonwrapped_model)
            >>> wrapped_model, wrapped_optim = ...
            >>> sharded_osd = FSDP.shard_full_optim_state_dict(rekeyed_osd, wrapped_model)
            >>> wrapped_optim.load_state_dict(sharded_osd)

        Returns:
            Dict[str, Any]: The optimizer state dict re-keyed using the
            parameter keys specified by ``optim_state_key_type``.
        """
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input, optim,
        )
        assert optim_state_key_type in (
            OptimStateKeyType.PARAM_NAME, OptimStateKeyType.PARAM_ID,
        )
        osd = optim_state_dict  # alias
        # Validate that the existing parameter keys are uniformly typed
        uses_param_name_mask = [
            type(param_key) is str for param_key in osd["state"]
        ]
        uses_param_id_mask = [
            type(param_key) is int for param_key in osd["state"]
        ]
        if (
            (any(uses_param_name_mask) and not all(uses_param_name_mask))
            or (any(uses_param_id_mask) and not all(uses_param_id_mask))
        ):
            error_msg = f"Invalid parameter keys: {osd['state'].keys()}"
            raise ValueError(error_msg)
        # Return directly if the existing key type matches the target key type
        if (optim_state_key_type == OptimStateKeyType.PARAM_NAME and
            all(uses_param_name_mask)) or \
            (optim_state_key_type == OptimStateKeyType.PARAM_ID and
                all(uses_param_id_mask)):
            return osd
        # Otherwise, actually perform the re-keying
        new_osd = {}
        if optim_state_key_type == OptimStateKeyType.PARAM_NAME:  # ID -> name
            param_id_to_param = (
                _get_param_id_to_param_from_optim_input(model, optim_input)
                if using_optim_input
                else _get_param_id_to_param(optim)
            )
            param_to_param_name = _get_param_to_param_name(model)
            param_id_to_param_name: List[str] = [
                param_to_param_name[param] for param in param_id_to_param
            ]
            new_osd["state"] = {
                param_id_to_param_name[param_id]: param_state
                for param_id, param_state in osd["state"].items()
            }
            new_osd["param_groups"] = copy.deepcopy(osd["param_groups"])
            for param_group in new_osd["param_groups"]:
                param_group["params"] = sorted([
                    param_id_to_param_name[param_id]
                    for param_id in param_group["params"]
                ])
            return new_osd
        elif optim_state_key_type == OptimStateKeyType.PARAM_ID:  # name -> ID
            param_name_to_param = _get_param_name_to_param(model)
            param_to_param_id = (
                _get_param_to_param_id_from_optim_input(model, optim_input)
                if using_optim_input
                else _get_param_to_param_id(optim)
            )
            # Because not all model parameters may be passed as the optimizer
            # input, we may need to drop some parameters from this mapping
            param_name_to_param_id = {
                param_name: param_to_param_id[param]
                for param_name, param in param_name_to_param.items()
                if param in param_to_param_id
            }
            new_osd["state"] = {
                param_name_to_param_id[param_name]: param_state
                for param_name, param_state in osd["state"].items()
            }
            new_osd["param_groups"] = copy.deepcopy(osd["param_groups"])
            for param_group in new_osd["param_groups"]:
                param_group["params"] = sorted([
                    param_name_to_param_id[param_name]
                    for param_name in param_group["params"]
                ])
            return new_osd
        return new_osd  # should never reach here

    def _get_default_comm_hook(self) -> Any:
        r"""
        Returns a default communication hook based on a sharding strategy.
        """
        if self.sharding_strategy != ShardingStrategy.NO_SHARD:
            return default_hooks.reduce_scatter_hook
        else:
            return default_hooks.allreduce_hook

    def _get_default_comm_hook_state(self) -> Any:
        r"""
        Returns a default communication hook state based on a sharding strategy.
        """
        return default_hooks.DefaultState(process_group=self.process_group)

    def register_comm_hook(self, state: object, hook: callable):
        """
        Registers a communication hook which is an enhancement that provides a
        flexible hook to users where they can specify how FSDP aggregates gradients
        across multiple workers.
        This hook can be used to implement several algorithms like
        `GossipGrad <https://arxiv.org/abs/1803.05880>`_ and gradient compression
        which involve different communication strategies for
        parameter syncs while training with :class:`FullyShardedDataParallel`.

        .. warning ::
            FSDP communication hook should be registered before running an initial forward pass
            and only once.

        Args:
            state (object): Passed to the hook to maintain any state information during the training process.
                            Examples include error feedback in gradient compression,
                            peers to communicate with next in `GossipGrad <https://arxiv.org/abs/1803.05880>`_, etc.
                            It is locally stored by each worker
                            and shared by all the gradient tensors on the worker.
            hook (Callable): Callable, which has one of the following signatures:
                            1) ``hook: Callable[torch.Tensor] -> None``:
                            This function takes in a Python tensor, which represents
                            the full, flattened, unsharded gradient with respect to all variables
                            corresponding to the model this FSDP unit is wrapping
                            (that are not wrapped by other FSDP sub-units).
                            It then performs all necessary processing and returns ``None``;
                            2) ``hook: Callable[torch.Tensor, torch.Tensor] -> None``:
                            This function takes in two Python tensors, the first one represents
                            the full, flattened, unsharded gradient with respect to all variables
                            corresponding to the model this FSDP unit is wrapping
                            (that are not wrapped by other FSDP sub-units). The latter
                            represents a pre-sized tensor to store a chunk of a sharded gradient after
                            reduction.
                            In both cases, callable performs all necessary processing and returns ``None``.
                            Callables with signature 1 are expected to handle gradient communication for a `NO_SHARD` case.
                            Callables with signature 2 are expected to handle gradient communication for sharded cases.

        """
        if not self.check_is_root():
            raise AssertionError("register_comm_hook can only be called on a root instance.")
        for submodule in self.fsdp_modules(self):
            assert not submodule._hook_registered, "communication hook can be only registered once"
            submodule._hook_registered = True
            assert submodule._communication_hook == self._get_default_comm_hook(),\
                f"communication hook should be default, but it is {submodule._communication_hook.__name__} instead"
            submodule._communication_hook_state = state
            submodule._communication_hook = hook


    def _init_param_exec_order_wrap_policy(self, *args, **kwargs) -> None:
        auto_wrap_policy = kwargs["auto_wrap_policy"]
        module = kwargs["module"]
        assert hasattr(auto_wrap_policy, "tracing_config")
        if not _TORCH_FX_AVAIL:
            assert (
                auto_wrap_policy.tracing_config is None
            ), "tracing_config should be None when torch.fx is not enabled"
        elif isinstance(
            auto_wrap_policy.tracing_config,
            TracingConfig
        ):
            tracer = auto_wrap_policy.tracing_config.tracer
            execution_info = _init_execution_info(module)

            for m in module.modules():
                assert not isinstance(
                    m, FullyShardedDataParallel
                ), "The input module of _patch_tracer should not contain FSDP modules"

            with _patch_tracer(
                tracer=tracer,
                root_module=module,
                execution_info=execution_info,
            ):
                try:
                    tracer.trace(module, auto_wrap_policy.tracing_config.concrete_args)
                except BaseException as e:
                    raise RuntimeError(
                        "tracer.trace failed inside _init_param_exec_order_wrap_policy"
                        f" with the error: {e}."
                    )
        else:
            assert (
                auto_wrap_policy.tracing_config is None
            ), "tracing_config should either be an instance of TracingConfig or be None"
        # The initial FSDP wrapping is done with auto_wrap_policy.init_policy
        kwargs["auto_wrap_policy"] = auto_wrap_policy.init_policy
        self.__init__(*args, **kwargs)
        self._param_exec_order_policy: bool = True
        # self._param_exec_order_prep_stage is set to True before we get the execution order
        self._param_exec_order_prep_stage: bool = True
        # A list that stores the flatten parameters and its name based on the parameter execution order
        self._fsdp_params_exec_order: List[FlatParameter] = []
        if _TORCH_FX_AVAIL and isinstance(
            auto_wrap_policy.tracing_config,
            TracingConfig
        ):
            # Initialize a dict that maps each module to its parent FSDP wrap
            module_to_fsdp: Dict[nn.Module, FullyShardedDataParallel] = dict()
            for wrap in self.fsdp_modules(self):
                module_to_fsdp[wrap.module] = wrap
            # Set self._fsdp_params_exec_order based on execution_info.module_forward_order.
            # TODO (linjianma): self._fsdp_params_exec_order will be set based on
            # the parameter execution order rather than module_forward_order,
            # once the non-recursive wrapping policy is fully implemented.
            for m in execution_info.module_forward_order:
                if m in module_to_fsdp:
                    for flat_param in module_to_fsdp[m].params:
                        self._fsdp_params_exec_order.append(flat_param)
            self._param_exec_order_prep_stage = False

        for m in self.modules():
            if m is not self and isinstance(m, FullyShardedDataParallel):
                # Assignment by reference, so each children FSDP wrap has access to
                # the _fsdp_params_exec_order of the root module
                m._fsdp_params_exec_order = self._fsdp_params_exec_order
                m._param_exec_order_policy = self._param_exec_order_policy
                m._param_exec_order_prep_stage = self._param_exec_order_prep_stage

    def _use_param_exec_order_policy(self) -> bool:
        return (
            hasattr(self, "_param_exec_order_policy")
            and self._param_exec_order_policy
        )

    def _is_param_exec_order_prep_stage(self) -> bool:
        is_prep_stage = (
            hasattr(self, "_param_exec_order_prep_stage")
            and self._param_exec_order_prep_stage
        )
        if not is_prep_stage:
            for p in self.parameters():
                assert (
                    not hasattr(p, "_params_exec_order_hook_handle")
                ), "When not in execution order prep stage, all _params_exec_order_hook_handle should be removed."
        return is_prep_stage


def _calc_grad_norm(parameters: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    r"""Calculate gradient norm of an iterable of parameters.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.0)
    if p == math.inf:
        local_norm = torch.tensor(max(par.grad.detach().abs().max() for par in parameters))
    else:
        # Compute the norm in full precision no matter what
        local_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(par.grad.detach(), p, dtype=torch.float32)
                    for par in parameters
                ]
            ),
            p,
        )
    local_norm.to(dtype=parameters[0].dtype)
    return local_norm


def _get_param_to_unflat_param_names(
    model: torch.nn.Module,
    dedup_shared_params: bool = True,
) -> Dict[torch.nn.Parameter, List[str]]:
    """
    Constructs a mapping from flattened parameter (including non-FSDP-module
    parameters) to its unflattened parameter names. For non-FSDP-module
    parameters, these mapped-to lists always contain a single element. The
    unflattened parameter names should match the keys of the model state dict.

    For shared parameters, only the first parameter name is included (following
    the ``torch.nn.Module.parameters()`` order).

    Args:
        model (torch.nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance).
        dedup_shared_params (bool): If ``True``, only includes the first
            list of unflattened parameter names corresponding to a parameter
            in the module walk order; if ``False``, then includes all of the
            unflattened parameter names.
    """
    def module_fn(module, prefix, param_to_unflat_param_names):
        # For FSDP modules, only add the entry when considering the contained
        # `FlattenParamsWrapper` to avoid duplication
        if not isinstance(module, FullyShardedDataParallel):
            for param_name, param in module.named_parameters(recurse=False):
                module_prefixed_param_names = (
                    param._prefixed_param_names if type(param) is FlatParameter
                    else [param_name]
                )  # prefixed from `module`
                fully_prefixed_param_names = [
                    clean_tensor_name(prefix + name)
                    for name in module_prefixed_param_names
                ]  # fully prefixed from the top level including `prefix`
                # If this parameter has already been visited, then it is a
                # shared parameter; then, only take the first parameter name
                is_shared_param = param in param_to_unflat_param_names
                if not is_shared_param:
                    param_to_unflat_param_names[param] = fully_prefixed_param_names
                elif not dedup_shared_params:
                    param_to_unflat_param_names[param].extend(fully_prefixed_param_names)

    def return_fn(param_to_unflat_param_names):
        return param_to_unflat_param_names

    param_to_unflat_param_names: Dict[torch.nn.Parameter, List[str]] = {}
    return _apply_to_modules(
        model, module_fn, return_fn, param_to_unflat_param_names,
    )


def _get_param_to_param_name(
    model: torch.nn.Module,
) -> Dict[torch.nn.Parameter, str]:
    """
    Constructs a mapping from parameters to their parameter names. ``model``
    should not contain any :class:`FullyShardedDataParallel` instances, which
    means that none of the parameters should be ``FlatParameter`` s. As a
    result, compared to :meth:`_get_param_to_unflat_param_names`, the mapped
    values may be flattened from singleton :class:`list` s to the contained
    names themselves.

    Args:
        model (torch.nn.Module): Root module, which should not contain any
            :class:`FullyShardedDataParallel` instances.
    """
    param_to_param_names = _get_param_to_unflat_param_names(model)
    for param_names in param_to_param_names.values():
        assert len(param_names) > 0, "`_get_param_to_unflat_param_names()` " \
            "should not construct empty lists"
        if len(param_names) > 1:
            raise RuntimeError(
                "Each parameter should only map to one parameter name but got "
                f"{len(param_names)}: {param_names}"
            )
    param_to_param_name = {
        param: param_names[0]
        for param, param_names in param_to_param_names.items()
    }
    return param_to_param_name


def _get_param_name_to_param(
    model: torch.nn.Module,
) -> Dict[str, torch.nn.Parameter]:
    """Constructs the inverse mapping of :meth:`_get_param_to_param_name`."""
    param_to_param_name = _get_param_to_param_name(model)
    return dict(zip(param_to_param_name.values(), param_to_param_name.keys()))


def clean_tensor_name(tensor_name: str) -> str:
    """Cleans the parameter or buffer name by removing any module wrapper
    prefixes."""
    # Call `replace()` twice separately since the name may not have both
    tensor_name = tensor_name.replace(FSDP_WRAPPED_MODULE + ".", "")
    tensor_name = tensor_name.replace(FPW_MODULE + ".", "")
    # TODO: Explicitly replacing checkpoint_wrapper prefix is not ideal,
    # as it increases coupling between CheckpointWrapper and FSDP. This is also not
    # scalable for additional wrapped modules, we should come up with a general solution
    # for this issue.
    tensor_name = tensor_name.replace(_CHECKPOINT_PREFIX + ".", "")
    return tensor_name
