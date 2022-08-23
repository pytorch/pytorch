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
    _get_param_to_param_id,
    _optim_state_dict,
    _process_pos_dim_tensor_state,
    _rekey_sharded_optim_state_dict,
)
from ._shard_utils import _create_chunk_sharded_tensor
from ._utils import (
    _apply_to_modules,
    _apply_to_tensors,
    _contains_batchnorm,
    _override_batchnorm_mixed_precision,
)
from .flat_param import FlatParameter, FlatParamHandle
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
    "OptimStateKeyType", "TrainingState_", "p_assert", "clean_tensor_name",
]


FSDP_WRAPPED_MODULE = "_fsdp_wrapped_module"
FSDP_PREFIX = FSDP_WRAPPED_MODULE + "." + FPW_MODULE + "."

_PARAM_BROADCAST_BUCKET_SIZE = int(250 * 1024 * 1024)

def _default_meta_device_init_fn(module):
    """
    Default initializer for modules initialized on the meta device.
    """
    # TODO: move module to device_id here once device_id is available.
    module.to_empty(device=torch.cuda.current_device())
    try:
        with torch.no_grad():
            module.reset_parameters()
    except BaseException as e:
        warnings.warn(
            f"Unable to call reset_parameters() for module on meta device with error {str(e)}. "
            "Please ensure your module implements a ``reset_parameters`` function."
        )
        raise e


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
    This class can be constructed with three flags:
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
        FSDP currently supports two types of ``state_dict``:
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


class _ExecOrderWarnStatus(Enum):
    """Used internally for execution order validation."""
    NONE = auto()     # no deviation yet
    WARNING = auto()  # deviated this iteration; currently issuing warnings
    WARNED = auto()   # deviated in a previous iteration


class _ExecOrderData():
    """
    This contains the data used for validating execution order across ranks.

    Attributes:
        _all_flat_params (List[FlatParameter]): A :class:`list` of all
            flattened parameters contained in the FSDP module hierarchy with
            the list index implicitly giving a unique parameter index.
        _param_to_unflat_param_names (Dict[FlatParameter, List[str]]): A
            mapping from flattened parameter to the comprising unflattened
            parameters' names.
        is_first_iter (bool): Whether executing in the first iteration or not.
        param_order (List[int]): Order that parameters participate in the
            forward pass; constructed on the first iteration and validated
            against in subsequent iterations.
        index (int): Index tracking the position in ``param_order``
            when validating the forward pass execution order in subsequent
            iterations.
        warn_status (_ExecOrderWarnStatus): To avoid flooding the console, we
            only issue warnings throughout the first deviating iteration and no
            longer check thereafter; this tracks the warning status.
    """
    def __init__(self) -> None:
        self._all_flat_params: List[FlatParameter] = []
        self._param_to_unflat_param_names: Dict[FlatParameter, List[str]] = []
        # Modified in the first iteration:
        self.is_first_iter: bool = True
        self.param_order: List[int] = []
        # Modified in the subsequent iterations:
        self.index: int = 0
        self.warn_status: _ExecOrderWarnStatus = _ExecOrderWarnStatus.NONE

    def init(self, root_module: "FullyShardedDataParallel"):
        assert root_module._is_root, "This data structure should only be " \
            "initialized on an FSDP root module"
        # Save all `FlatParameter`s in `root_module`'s hierarchy to
        # `_all_flat_params` instead of re-materializing each time to avoid the
        # result depending on the calling context (e.g. when some parameters
        # have been rebuilt)
        self._all_flat_params = [
            param for param in root_module.parameters()
            if isinstance(param, FlatParameter)
        ]
        self._param_to_unflat_param_names = cast(
            Dict[FlatParameter, List[str]],
            _get_param_to_unflat_param_names(root_module)
        )

    def get_param_index(self, param: FlatParameter) -> int:
        """Returns a unique non-negative parameter index for ``param`` if it is
        valid or -1 otherwise. Critically, this index assignment must be the
        same across ranks."""
        assert isinstance(param, FlatParameter), \
            f"Expects `param` is a `FlatParameter` but got {type(param)}"
        for i, p in enumerate(self._all_flat_params):
            if p is param:
                return i
        return -1

    def get_param(self, param_index: int) -> Optional[FlatParameter]:
        """Returns the parameter corresponding to ``param_index`` or ``None``
        if the index is invalid."""
        for i, p in enumerate(self._all_flat_params):
            if i == param_index:
                return p
        return None

    def get_unflat_param_names(self, param_index: int) -> List[str]:
        """Returns a :class:`list` of unflattened parameter names comprising
        the flattened parameter with index ``param_index`` or an empty
        :class:`list` if ``param_index`` is invalid."""
        param = self.get_param(param_index)
        if param is None:
            return []
        assert param in self._param_to_unflat_param_names, \
            "Internal data structures out of sync; check `init()`"
        return self._param_to_unflat_param_names[param]

    def reset(self):
        """Called in :meth:`_wait_for_post_backward` to reset data for the next
        iteration."""
        self.is_first_iter = False
        self.index = 0
        # `reset()` marks the end of an iteration, so transition if needed
        if self.warn_status == _ExecOrderWarnStatus.WARNING:
            self.warn_status = _ExecOrderWarnStatus.WARNED


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
            and ``module`` is on CPU, we will move ``module`` to current CUDA device for faster
            initialization and move ``module`` back to CPU before returning.
            If specified, resulting FSDP instances will reside on this device.
            Note that if ``device_id`` is specified but ``module`` is already
            on a different CUDA device, an error will be thrown. (Default: ``None``)

        sync_module_states (bool): If ``True``, each individually wrapped FSDP unit will broadcast
            module parameters from rank 0 to ensure they are the same across all ranks after
            initialization. This helps ensure model parameters are the same across ranks
            before starting training, but adds communication overhead to ``__init__``, as at least
            one broadcast is triggered per individually wrapped FSDP unit.
            This can also help load checkpoints taken by ``state_dict`` and to be loaded by
            ``load_state_dict`` in a memory efficient way. See documentation for
            :class:`FullStateDictConfig` for an example of this. (Default: ``False``)

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
            )
            return

        torch._C._log_api_usage_once("torch.distributed.fsdp")
        super().__init__()
        self._handles: List[FlatParamHandle] = []
        # Validate the ignored modules and derive the ignored parameters/buffers
        ignored_modules = self._get_ignored_modules(module, ignored_modules)
        self._ignored_modules = ignored_modules
        ignored_params, ignored_param_names = \
            self._get_ignored_params(module, ignored_modules)
        buffer_names = self._get_buffer_names(module)
        # Compute the names to ignore for full state dict cloning (i.e. those
        # of the ignored modules' parameters and of all modules' buffers)
        self._ignored_param_names = ignored_param_names
        self._buffer_names = buffer_names
        # NOTE: Since the names are computed at construction time, if the user
        # changes them later, then FSDP will not properly ignore them. However,
        # the `FlatParameter` implementation already relies on this assumption.
        # We do this at construction time since we want the fully prefixed
        # parameter names matching the keys in the model state dict (namely,
        # including the wrapped module's name in the prefix), which may be done
        # most non-intrusively here before flattening.

        # if auto_wrap_policy is specified, submodules should not be
        # already wrapped, otherwise we'd attempt to double wrap them resulting
        # in errors.
        if auto_wrap_policy is not None:
            self._check_wrapped(
                module,
                check_fn=lambda mod: not isinstance(mod, FullyShardedDataParallel),
                err_fn=lambda mod: f"Expected {mod} to NOT be FullyShardedDataParallel if auto_wrap is enabled.",
            )
            if mixed_precision is not None and _contains_batchnorm(module):
                _override_batchnorm_mixed_precision(module)
                policy_to_use = functools.partial(
                    _or_policy,
                    policies=[_wrap_batchnorm_individually, auto_wrap_policy]
                )
                warnings.warn(
                    "Mixed precision was specified for FSDP module with"
                    " batchnorm submodules wrapped via ``auto_wrap_policy``."
                    " BatchNorm units will be wrapped as a separate FSDP unit,"
                    " with mixed_precision disabled (i.e. set to ``None``)"
                    " as several BatchNorm kernels would raise errors when"
                    " operating on reduced precision inputs."
                )
            else:
                policy_to_use = auto_wrap_policy
            _recursive_wrap(
                module,
                auto_wrap_policy=policy_to_use,
                wrapper_cls=FullyShardedDataParallel,
                ignored_modules=ignored_modules,
                ignored_params=ignored_params,
                # Note that we have the recursive_wrap skip wrapping for
                # the outermost (this) module otherwise it will result in a
                # double-wrap causing issues.
                only_wrap_children=True,
                # FSDP arguments follow.
                process_group=process_group,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                backward_prefetch=backward_prefetch,
                forward_prefetch=forward_prefetch,
                mixed_precision=mixed_precision,
                param_init_fn=param_init_fn,
                device_id=device_id,
                sync_module_states=sync_module_states,
            )

        self.process_group = process_group or _get_default_group()
        self.rank = self.process_group.rank()
        self.world_size = self.process_group.size()
        if device_id is not None:
            self.device_id = (
                device_id if isinstance(device_id, torch.device)
                else torch.device(device_id)
            )
            # If user passed in something like torch.device("cuda"),
            # device index of current device is unclear, make it explicit.
            if self.device_id == torch.device("cuda"):
                warnings.warn(
                    f"Passed in {self.device_id} does not have explicit index, "
                    f"setting it to current index: {torch.cuda.current_device()}. "
                    "If this is not correct, please explicitly call torch.cuda.set_device()"
                    "before FSDP initialization or pass in explicit device index as device_id argument."
                )
                self.device_id = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device_id = None


        is_meta_module = any(p.is_meta for p in module.parameters())
        is_torchdistX_deferred_init = (
            not is_meta_module and _TORCHDISTX_AVAIL
            and any(fake.is_fake(p) for p in module.parameters())
        )

        def _run_param_init_fn():
            # Call user-specified initialization function.
            if not callable(param_init_fn):
                raise ValueError(
                    f"Expected {param_init_fn} to be callable, but got {type(param_init_fn)}"
                )
            param_init_fn(module)

        if is_meta_module:
            if param_init_fn is not None:
                _run_param_init_fn()
            else:
                # Call default initialization function that is dependent on
                # reset_parameters.
                _default_meta_device_init_fn(module)
        elif is_torchdistX_deferred_init:
            assert _TORCHDISTX_AVAIL, "Got torchdistX initialized module but torchdistX lib is not available."
            if param_init_fn is not None:
                _run_param_init_fn()
            else:
                # Call default torchdistX initialization function. Omit re-initialization of FSDP submodules
                # which is unnecessary.
                check_fn = lambda k: not isinstance(k, FullyShardedDataParallel)  # noqa: E731
                deferred_init.materialize_module(module, check_fn=check_fn)

        # Check that module was placed onto a single device.
        module_devices = set(
            p.device for p in module.parameters() if p not in ignored_params and not isinstance(p, FlatParameter)
        )

        if len(module_devices) > 1:
            raise RuntimeError(
                f"FSDP only supports single device modules, but got params on {module_devices}"
            )

        # Move module appropriately depending on device_id and whether module is on CPU.
        self._move_module_if_needed(module)

        # device for computation, if module is on GPU, use module.device;
        # if module is on CPU, use current device;
        self.compute_device = _get_default_cuda_device(module)

        # if device_id is specified, ensure it is the same
        assert (
            self.device_id is None or self.compute_device == self.device_id
        ), f"Inconsistent compute_device and device_id: {self.compute_device} vs {self.device_id}"

        # Enum to indicate if we're in the forward/backward pass, idle, etc.
        self.training_state = TrainingState_.IDLE
        self.cpu_offload = cpu_offload or CPUOffload()
        self.backward_prefetch = backward_prefetch
        self.forward_prefetch = forward_prefetch
        self.sharding_strategy = sharding_strategy or ShardingStrategy.FULL_SHARD
        self.mixed_precision = mixed_precision
        # Original buffer type (mapping since all buffers may not be of same type). In
        # the case of mixed precision training, this is used to restore buffers
        # to their original type (which may not be the same as that of the
        # parameters in the model) when checkpointing.
        self._orig_buffer_dtypes: Dict[str, torch.dtype] = {}

        # Only handle params which are not already sharded. This enables
        # sharding individual layers of a Module, with an outer wrapper to
        # shard any leftover parameters.
        params = [
            p for p in module.parameters()
            if p not in ignored_params and not isinstance(p, FlatParameter)
        ]

        if sync_module_states:
            if params != [] and params[0].device == torch.device("cpu"):
                raise ValueError(
                    "Module has CPU parameters, but sync_module_states=True is specified."
                    "This only works for GPU module, please specify `device_id` argument or move"
                    " module to GPU before init."
                )
            # Collect buffers we have to synchronize, avoiding buffers that have already
            # been synchronized to avoid redundant synchronization.
            bufs_to_sync = []
            for buf in module.buffers():
                if not getattr(buf, '_fsdp_has_been_sync', False):
                    buf._fsdp_has_been_sync = True
                    bufs_to_sync.append(buf.detach())

            states_to_sync = [param.detach() for param in params]
            states_to_sync.extend(bufs_to_sync)
            _sync_params_and_buffers(
                process_group=self.process_group,
                module_states=states_to_sync,
                # Same bucket size as DDP
                broadcast_bucket_size=_PARAM_BROADCAST_BUCKET_SIZE,
                src=0,
            )

        self._fsdp_wrapped_module = FlattenParamsWrapper(module, params)
        assert getattr(self, FSDP_WRAPPED_MODULE) is self._fsdp_wrapped_module
        self.params = []
        if self._fsdp_wrapped_module.has_params:
            self.params.append(self._fsdp_wrapped_module.flat_param)
            self._register_param_handle(self._fsdp_wrapped_module.handle)

        # Shard module parameters in place
        self._shard_parameters()

        # Check that the sharding logic was applied to all parameters by
        # checking that the original module parameters have been replaced by
        # `Tensor` views and are no longer `nn.Parameter`s
        for n, p in self.named_parameters():
            if p not in ignored_params and not isinstance(p, FlatParameter):
                raise RuntimeError(
                    f"found unflattened parameter: {n} ; {p.size()} {p.__class__}"
                )
        self._reset_lazy_init()

        # Flag indicating if we require gradient reduction in the backward
        # pass (set to `False` in the `no_sync()` context manager)
        self._require_backward_grad_sync: bool = True

        self._state_dict_type = StateDictType.FULL_STATE_DICT
        self._state_dict_config = FullStateDictConfig()

        # FSDP currently provides three different state_dicts. The actual
        # state_dict that will be saved/loaded is decided by
        # self._state_dict_type. And the main logic of each state_dict is
        # implemented in the hook. Therefore, for each hook (post-save and
        # pre-load), there is a dispatcher dictionary to dispatch the execution
        # flow to the correct implementation.
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

        # Flag to guard against preparing gradients multiple times per backward pass.
        self._pre_backward_hook_has_run = False
        # Used for prefetching all gather full params in post backward hook
        self._need_rebuild_full_params = False

        # If specified, offload parameter shard to CPU.
        if self.cpu_offload.offload_params:
            for p in self.params:
                self._offload_to_cpu(p)

        # For validating execution order across ranks
        self._exec_order_data = _ExecOrderData()

        # Setting communication hook to a default:
        # ``reduce_scatter`` for shareded strategies and
        # ``all_reduce`` for ``NO_SHARD``
        self._communication_hook = self._get_default_comm_hook()
        self._communication_hook_state = self._get_default_comm_hook_state()
        self._hook_registered = False

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

    def _move_module_if_needed(self, module) -> None:
        """
        Moves module if module is on CPU and device_id is specified.
        If device_id is not specified and module is on CPU, we log a
        warning to user mentioning to use ``device_id`` argument to speed
        up initialization performance.
        """
        # Move module to device specified. Note that this is done prior to
        # setting compute_device to ensure that they align.
        if self.device_id is not None:
            param = None
            try:
                # Get the next unflat param
                param_gen = module.parameters()
                while True:
                    param = next(param_gen)
                    if not isinstance(param, FlatParameter):
                        break

                if param.device == torch.device("cpu"):
                    module = module.to(self.device_id)
            except StopIteration:
                # this FSDP instance manages no parameters.
                pass

            # For GPU modules, module device should match device_id.
            if (
                param is not None
                and not isinstance(param, FlatParameter)
                and param.device != self.device_id
            ):
                raise RuntimeError(
                    f"Module on rank {self.rank} is given device_id argument "
                    f"{self.device_id}, but is on {param.device}. "
                    " Either move module before FSDP init or omit device_id argument."
                )
        else:
            # device_id argument is not specified
            # If module is on CPU, log a warning asking user to use `device_id` for faster
            # GPU init.
            try:
                # Get the next unflat param
                param_gen = module.parameters()
                while True:
                    param = next(param_gen)
                    if not isinstance(param, FlatParameter):
                        break

                if param.device == torch.device("cpu"):
                    warnings.warn(
                        "Module is put on CPU and will thus have flattening and sharding"
                        " run on CPU, which is less efficient than on GPU. We recommend passing in "
                        "`device_id` argument which will enable FSDP to put module on GPU device,"
                        " module must also be on GPU device to work with `sync_module_states=True` flag"
                        " which requires GPU communication."
                    )
            except StopIteration:
                # this FSDP instance manages no parameters
                pass

    def _init_reshard_after_forward(self):
        if self.sharding_strategy == ShardingStrategy.FULL_SHARD:
            # Free full params and keep shard only after forward
            self.reshard_after_forward = True
        elif self.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP:
            # Keep full params in the GPU memory until backward
            # computation is done
            self.reshard_after_forward = False
        elif self.sharding_strategy == ShardingStrategy.NO_SHARD:
            # self.reshard_after_forward is not used when NO_SHARD
            # is set, just setting it as False here
            self.reshard_after_forward = False
        else:
            raise RuntimeError(
                "sharding_strategy only supports FULL_SHARD, SHARD_GRAD_OP and NO_SHARD right now."
            )

    def _get_ignored_modules(
        self,
        root_module: torch.nn.Module,
        _ignored_modules: Any,
    ) -> Set[torch.nn.Module]:
        """
        Checks that ``_ignored_modules`` (1) is an iterable of
        ``torch.nn.Module`` s without any :class:`FullyShardedDataParallel`
        instances and does not contain the top-level ``module`` itself, and
        then returns them and their children as a :class:`set`, excluding
        nested :class:`FullyShardedDataParallel` instances.

        We include the child modules of modules in ``_ignored_modules`` to be
        more intuitive since ignoring a module should ignore its child modules
        as well, and we exclude :class:`FullyShardedDataParallel` instances
        since ``self`` may be the intended root instance that manages them.
        """
        if _ignored_modules is None:
            return set()
        msg_prefix = "`ignored_modules` should be an iterable of " \
            "`torch.nn.Module`s "
        try:
            ignored_root_modules = set(_ignored_modules)
        except TypeError:
            raise TypeError(msg_prefix + f"but got {type(_ignored_modules)}")
        for module in ignored_root_modules:
            if not isinstance(module, torch.nn.Module):
                raise TypeError(
                    msg_prefix + f"but got an iterable with {type(module)}"
                )
            if isinstance(module, FullyShardedDataParallel):
                raise ValueError(
                    "`ignored_modules` should not include FSDP modules"
                )
        # Include child modules and exclude nested FSDP modules
        ignored_modules = set(
            child for module in ignored_root_modules
            for child in module.modules()
            if not isinstance(child, FullyShardedDataParallel) and
            not isinstance(child, FlattenParamsWrapper)
        )
        if root_module in ignored_modules:
            warnings.warn(
                "Trying to ignore the top-level module passed into the FSDP "
                "constructor itself will result in all parameters being "
                f"ignored and is not supported: {module}"
            )
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
        excluding any :class:`FlatParameter` s and their fully prefixed names,
        both as :class:`set` s.

        Args:
            root_module (torch.nn.Module): Top-level module passed into the
                FSDP constructor from which to derive the fully prefixed names.
            ignored_modules (Set[torch.nn.Module]): Modules to ignore.
        """
        ignored_params = set(
            p for m in ignored_modules for p in m.parameters()
            if not isinstance(p, FlatParameter)
        )
        param_to_unflat_param_names = _get_param_to_unflat_param_names(
            root_module, dedup_shared_params=False,
        )
        ignored_param_names = set()
        for param in ignored_params:
            unflat_param_names = param_to_unflat_param_names[param]
            clean_names = []
            for k in unflat_param_names:
                clean_names.append(clean_tensor_name(k))
            ignored_param_names.update(clean_names)
        return ignored_params, ignored_param_names

    def _get_buffer_names(self, root_module: torch.nn.Module) -> Set[str]:
        """
        Returns the fully prefixed names of all buffers in the module hierarchy
        rooted at ``root_module`` as a class:`set`.

        Args:
            root_module (torch.nn.Module): Top-level module passed into the
                FSDP constructor from which to derive the fully prefixed names.
        """
        def module_fn(module, prefix, buffer_names):
            # For FSDP modules, only add the entry when considering the
            # contained `FlattenParamsWrapper` to avoid duplication
            if not isinstance(module, FullyShardedDataParallel):
                for buffer_name, _ in module.named_buffers(recurse=False):
                    prefixed_buffer_name = clean_tensor_name(prefix + buffer_name)
                    buffer_names.add(prefixed_buffer_name)

        def return_fn(buffer_names, *args):
            return buffer_names

        buffer_names: Set[str] = set()
        return _apply_to_modules(
            root_module, module_fn, return_fn, buffer_names,
        )

    @classmethod
    def _check_wrapped(cls, begin_module, check_fn, err_fn):
        for _, mod in begin_module.named_modules():
            if not check_fn(mod):
                raise ValueError(err_fn(mod))

    def _register_param_handle(self, handle: FlatParamHandle) -> None:
        """Registers the parameter handle to this FSDP instance."""
        if handle not in self._handles:
            self._handles.append(handle)

    @property
    def module(self) -> nn.Module:
        """Make model.module accessible, just like DDP. Return the
        underlying module without the flatten_params_wrapper
        """
        assert isinstance(self._fsdp_wrapped_module, FlattenParamsWrapper)
        return self._fsdp_wrapped_module.module

    def check_is_root(self) -> bool:
        self._lazy_init()
        assert self._is_root is not None
        return self._is_root

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

    def _offload_to_cpu(self, p):
        """
        Offloads parameter to CPU from self.compute_device. If the parameter is
        already on CPU then this is a noop.
        """
        cpu_device = torch.device("cpu")
        if p.device == cpu_device:
            return
        with torch.no_grad():
            p.data = p.to(cpu_device)

    def _mixed_precision_enabled_for_params(self) -> bool:
        """
        Whether user explicitly enabled mixed precision for
        parameters or not.
        """
        return (
            self.mixed_precision is not None
            and self.mixed_precision.param_dtype is not None
        )

    def _mixed_precision_enabled_for_buffers(self) -> bool:
        """
        Whether user explicitly enabled mixed precision for
        buffers or not.
        """
        return (
            self.mixed_precision is not None
            and self.mixed_precision.buffer_dtype is not None
        )

    def _mixed_precision_enabled_for_reduce(self) -> bool:
        """
        Whether user explicitly enabled mixed precision for
        gradient reduction or not.
        """
        return (
            self.mixed_precision is not None
            and self.mixed_precision.reduce_dtype is not None
        )

    def _low_precision_hook_enabled(self) -> bool:
        """
        Wether a low precision hook is registered or not.
        """
        return (
            self._communication_hook is not None
            and self._communication_hook in LOW_PRECISION_HOOKS
        )

    def _cast_fp_inputs_to_precision(
        self, dtype: torch.dtype, *args: Any, **kwargs: Any
    ) -> Tuple[Any, Any]:
        """
        Casts floating point tensors in args and kwargs to precision given by dtype.
        requires_grad field is respected.
        """
        def cast_fn(x: torch.Tensor) -> torch.Tensor:
            if not torch.is_floating_point(x):
                return x
            y = x.to(dtype)
            # Explicitly copy over requires_grad context since this is happening
            # within torch.no_grad.
            if x.is_leaf:
                y.requires_grad = x.requires_grad
            return y

        with torch.no_grad():
            return (
                _apply_to_tensors(cast_fn, args),
                _apply_to_tensors(cast_fn, kwargs)
            )

    @torch.no_grad()
    def _cast_param_shards_to_dtype(self):
        """
        Allocates a mixed precision paramter shard and casts parameter shards to
        reduced precision by copying into this mixed precision shard. Note that
        if we are CPU offloading, this also implicitly loads the parameter shard
        back to GPU.
        """
        assert (
            self._mixed_precision_enabled_for_params()
        ), "Expected to only be called when mixed precision for parameters is enabled."
        with torch.cuda.stream(self._streams["mixed_precision_params"]):
            for p in self.params:
                assert p._mp_shard is not None
                _alloc_storage(data=p._mp_shard, size=p._local_shard.size())
                # Cast is done by copy
                p._mp_shard.copy_(
                    # no-op if not CPU offloading, otherwise nonblocking because
                    # p._local_shard is pinned in _init_param_attributes.
                    p._local_shard.to(p._mp_shard.device, non_blocking=True)
                )
                # Point p to the mp shard
                p.data = p._mp_shard
        # Block current stream on this copy work.
        torch.cuda.current_stream().wait_stream(self._streams["mixed_precision_params"])

    @torch.no_grad()
    def _free_mp_shard(self, params: List[FlatParameter]):
        """
        Deallocate storage for parameter's mixed precision shard.
        """
        assert (
            self._mixed_precision_enabled_for_params()
        ), "Expected to only be called when mixed precision for parameters is enabled."
        current_stream = torch.cuda.current_stream()
        for p in params:
            # mp_shard should always be allocated.
            assert p._mp_shard is not None
            # Shard is allocated in "mixed_precision_stream" and then we block
            # current stream on this stream, so don't free it until work in the
            # current stream is completed.
            p._mp_shard.record_stream(current_stream)
            _free_storage(p._mp_shard)

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
                    if name not in self._orig_buffer_dtypes:
                        self._orig_buffer_dtypes[name] = buf.dtype
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

    @torch.no_grad()
    def _shard_parameters(self) -> None:
        """
        At initialization we wrap a module with full parameters and shard the
        parameters in-place. Sharding is implemented by viewing each parameter
        as a 1D Tensor and retaining only a single slice, where the slice size
        is determined by the number of data parallel workers.
        After this initial sharding is complete, the user can initialize a
        ``torch.optim.Optimizer`` in the usual way, i.e.::
        .. code-block:: python
            optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        The optimizer will see only a single slice of parameters and will thus
        allocate less memory for optimizer state, avoiding redundancy across
        data parallel workers.
        """
        for handle in self._handles:
            p = handle.flat_param
            assert not p._is_sharded, "Param should have not been sharded yet."
            assert (
                p.is_floating_point()
            ), "Autograd does not support operations for integer type."

            # Sharding is done only when world_size is larger than 1 and
            # sharding_strategy!=NO_SHARD.
            p._is_sharded = (  # type: ignore[attr-defined]
                self.world_size > 1
                and self.sharding_strategy != ShardingStrategy.NO_SHARD
            )

            if not p._is_sharded:  # type: ignore[attr-defined]
                continue

            # Save the original storage and free it later on.
            # Since we're modifying the tensor's storage directly,
            # make sure the tensor is the sole occupant of the storage.
            assert (
                p.storage_offset() == 0
            ), "The tensor is not the sole occupant of the storage."
            orig_storage = p.storage()

            # Replace p with the relevant shard.
            local_shard, numel_padded = FlatParamHandle._get_shard(p, self.rank, self.world_size)
            p.set_(local_shard)  # type: ignore[call-overload]
            handle.init_shard_metadata(local_shard.numel(), numel_padded, self.rank)

            # Free storage that contains the original full data.
            if orig_storage.size() > 0:
                orig_storage.resize_(0)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._fsdp_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self._fsdp_wrapped_module.__getitem__(key)  # type: ignore[operator]

    def _reset_lazy_init(self) -> None:
        """
        Reset instance so :func:`_lazy_init` will run on the next forward.
        """
        self._is_root: Optional[bool] = None
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._fsdp_graph_order: List[nn.Module] = []
        self._my_fsdp_idx_in_graph: Optional[int] = None
        self._pre_backward_hook_full_params_prefetched: bool = False
        self._forward_full_params_prefetched: bool = False

        for p in self.params:
            if hasattr(p, "_local_shard"):
                # reset attributes that are added in _init_param_attributes, as
                # part of _lazy_init
                del p._local_shard  # type: ignore[attr-defined]
        # set 'self.reshard_after_forward' flag based on self.sharding_strategy
        self._init_reshard_after_forward()

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
        # The following logic is only run on the root FSDP instance
        self._is_root = True
        self._assert_state(TrainingState_.IDLE)
        self._init_streams()
        self._cast_buffers(recurse=True)
        for param in self.params:
            self._init_param_attributes(param)
        # Do not reshard the root's parameters at the end of the forward pass
        # with the intention that they are immediately used in the backward
        # pass gradient computation (though this may not be true)
        self.reshard_after_forward = False
        self._exec_order_data.init(self)
        # Initialize non-root FSDP instances and share attributes from the root
        # to non-root instances (e.g. streams for overlapping)
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
                fsdp_module._fsdp_graph_order = self._fsdp_graph_order
                fsdp_module._exec_order_data = self._exec_order_data
                for param in fsdp_module.params:
                    fsdp_module._init_param_attributes(param)

    @torch.no_grad()
    def _init_param_attributes(self, p: FlatParameter) -> None:
        """
        We manage several attributes on each Parameter instance. The first is
        set by :func:`_shard_parameters`:
            ``_is_sharded``: ``True`` if the Parameter is sharded or ``False``
                if the Parameter is intentionally not sharded (in which case we
                will all-reduce grads for this param). Currently the way
                `_is_sharded = False` is if world_size = 1 or sharding strategy
                is NO_SHARD.
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
            ``_shard_bwd_hook``: it holds the parameter's AccumulateGrad object
                and the registered post hook handle.
        """
        assert hasattr(p, "_is_sharded"), "Parameters should have been sharded during construction."
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
            p._local_shard.pin_memory()  # type: ignore[attr-defined]
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
        if p._is_sharded:  # type: ignore[attr-defined]
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
            _free_storage(p._full_param_padded)  # type: ignore[attr-defined]

        # Track whether the `FlatParameter`'s post-backward hook has been
        # called for validation in `_wait_for_post_backward()`
        p._post_backward_called = False

    def _init_streams(self) -> None:
        """Initializes CUDA streams for overlapping data transfer and
        computation. This should only be called on the root FSDP instance."""
        assert self._is_root
        if torch.cuda.is_available():
            # Stream for all-gathering parameters.
            self._streams["all_gather"] = torch.cuda.Stream()
            # Stream for overlapping grad reduction with the backward pass.
            self._streams["post_backward"] = torch.cuda.Stream()
            # Stream to move main params to self.mixed_precision.param_dtype
            # for forward pass.
            if self._mixed_precision_enabled_for_params():
                self._streams["mixed_precision_params"] = torch.cuda.Stream()

    def _wait_for_previous_optim_step(self) -> None:
        """
        The root :class:`FullyShardedDataParallel` instance needs to
        synchronize with the default stream to ensure that the previous
        optimizer step is done.
        """
        if not torch.cuda.is_available() or not self._is_root:
            return
        if self._mixed_precision_enabled_for_params():
            self._streams["mixed_precision_params"].wait_stream(
                torch.cuda.current_stream()
            )
        self._streams["all_gather"].wait_stream(torch.cuda.current_stream())

    def _need_prefetch_full_params(self, state: TrainingState_) -> bool:
        allowed_states = (
            TrainingState_.FORWARD, TrainingState_.BACKWARD_PRE, TrainingState_.BACKWARD_POST
        )
        assert state in allowed_states, f"state needs to be in the set of {allowed_states}"
        valid_fsdp_graph_and_index = (
            self._fsdp_graph_order is not None
            and self._my_fsdp_idx_in_graph is not None
        )
        if state == TrainingState_.FORWARD:
            return (
                self.forward_prefetch
                and valid_fsdp_graph_and_index
                and self._my_fsdp_idx_in_graph < len(self._fsdp_graph_order) - 1
                and self._fsdp_graph_order[self._my_fsdp_idx_in_graph + 1].training_state
                != TrainingState_.FORWARD
            )
        elif state == TrainingState_.BACKWARD_PRE:
            return (
                self.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
                and valid_fsdp_graph_and_index
                and self._my_fsdp_idx_in_graph > 0
                and self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1].training_state
                != TrainingState_.BACKWARD_POST
            )
        else:
            return (
                self.backward_prefetch == BackwardPrefetch.BACKWARD_POST
                and valid_fsdp_graph_and_index
                and self._my_fsdp_idx_in_graph > 0
                and self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1].training_state
                != TrainingState_.BACKWARD_POST
                and self._fsdp_graph_order[
                    self._my_fsdp_idx_in_graph - 1
                ]._need_rebuild_full_params
            )

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

    @property
    def _param_fqns(self) -> Iterator[Tuple[str, str, str]]:
        for param_name, module_name in (
            self._fsdp_wrapped_module.handle.parameter_module_names()
        ):
            module_name = module_name.replace(f"{FPW_MODULE}.", "")
            module_name = module_name.replace(f"{FPW_MODULE}", "")
            if module_name:
                module_name = f"{module_name}."
            # Activation checkpoint adds a prefix that has to be
            # removed as well.
            module_name = module_name.replace(
                f"{checkpoint_wrapper._CHECKPOINT_PREFIX}.", ""
            )
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
        full_numel = flat_param._unsharded_size.numel()
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
                state_dict[f"{prefix}{fqn}"] = _create_chunk_sharded_tensor(
                    tensor=param,
                    rank=self.rank,
                    world_size=self.world_size,
                    device_per_node=torch.cuda.device_count(),
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
                        dtype=self._orig_buffer_dtypes, recurse=False
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
                not self._fsdp_wrapped_module.flat_param._is_sharded
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

        if not self._fsdp_wrapped_module.flat_param._is_sharded:
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
            module_name = module_name.replace(f"{FPW_MODULE}.", "")
            module_name = module_name.replace(f"{FPW_MODULE}", "")
            if module_name:
                module_name = f"{module_name}."
            fqn = f"{prefix}{FSDP_WRAPPED_MODULE}.{module_name}{param_name}"
            param = state_dict.pop(fqn)

            # All-gather the param (ShardedTensor)
            shards = param.local_shards()
            local_tensor = cast(torch.Tensor, shards[0].tensor).flatten()
            dim_0_size = param.size()[0]
            param_numel = param.size().numel()
            chunk_size = (
                math.ceil(dim_0_size / self.world_size) * param_numel // dim_0_size
            )
            num_padding = chunk_size - local_tensor.numel()
            if num_padding > 0:
                local_tensor = F.pad(local_tensor, [0, num_padding])
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
        with torch.autograd.profiler.record_function("FullyShardedDataParallel.forward"):
            self._lazy_init()
            self._wait_for_previous_optim_step()

            # Start of a forward pass.
            self.training_state = TrainingState_.FORWARD
            if self._is_root:
                # TODO: disabling side stream for tensor copies for now, investigate
                # perf with it on / off.
                # Place inputs on compute_device. This is a noop if inputs are already
                # on compute_device. Note that when device_id is specified,
                # device_id == self.compute_device is guaranteed.
                # TODO: for mixed precision, move inputs to right device + cast might
                # be done in one go for performance.
                args, kwargs = _to_kwargs(args, kwargs, self.compute_device.index, False)
                args = args[0]
                kwargs = kwargs[0]

            # Cast inputs to their mixed precision type.
            if (
                self._is_root
                and self._mixed_precision_enabled_for_params()
            ):
                input_dtype = self.mixed_precision.param_dtype
                args, kwargs = self._cast_fp_inputs_to_precision(
                    input_dtype, *args, **kwargs
                )

            # Only rebuilding full params when the params are not prefetched in previous layers
            if not self._forward_full_params_prefetched:
                self._rebuild_full_params()
            self._forward_full_params_prefetched = False
            # Wait for all_gather full parameters to finish before computation
            torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

            # Prefetch next layer's full params in forward pass
            if self._need_prefetch_full_params(self.training_state):
                # This guarantees that pre-fetching is initialized only after all
                # previous computations are finished. Therefore, all gather next layer's
                # parameters will only overlap with this layer's computation. This
                # prevents over-prefetching, where multiple layer's parameters are prefetched
                # before the computation.
                self._streams["all_gather"].wait_stream(torch.cuda.current_stream())
                self._fsdp_graph_order[self._my_fsdp_idx_in_graph + 1]._rebuild_full_params()
                self._fsdp_graph_order[self._my_fsdp_idx_in_graph + 1]._forward_full_params_prefetched = True

            # Register backward hooks to reshard params and reduce-scatter grads.
            # These need to be re-registered every forward pass in some cases where grad_fn
            # is mutated.
            self._register_post_backward_hooks()
            outputs = self._fsdp_wrapped_module(*args, **kwargs)

            if self not in self._fsdp_graph_order:
                self._my_fsdp_idx_in_graph = len(self._fsdp_graph_order)
                self._fsdp_graph_order.append(self)

            if self.reshard_after_forward:
                self._free_full_params()
                if (
                    self._mixed_precision_enabled_for_params()
                ):
                    self._free_mp_shard(self.params)
            # Switch to original local shards of params. We maintain this invariant throughout
            # the code, i.e., ``p.data == p._local_shard`` after each function. This
            # also ensures that after the first forward, the optimizer state will be
            # initialized with the correct dtype and (sharded) size, since optimizer
            # state is typically initialized lazily in ``optim.step()``. Note that
            # when CPU offload is enabled, _use_param_local_shard implicitly
            # offloads the local shard to CPU by making p.data point to
            # p._local_shard, which would reside on CPU.
            self._use_param_local_shard()

            # Register pre-backward hooks to all-gather the params for the backward
            # pass (if output's grad was needed). This won't register anything if
            # we are in eval mode.
            outputs = self._register_pre_backward_hooks(outputs)

            # Done with a forward pass.
            self.training_state = TrainingState_.IDLE

        return outputs

    @torch.no_grad()
    def _write_back_current_shard(self, full_params):
        """
        Writes back full_params into self.params.
        """
        for p, (full_param, _) in zip(self.params, full_params):
            if not p._is_sharded:  # type: ignore[attr-defined]
                continue  # Already copied because no sharding.

            # TODO: Might be able to refactor to use _get_shard.
            chunks = full_param.chunk(self.world_size)  # type: ignore[attr-defined]
            assert len(chunks) > self.rank
            chunk = chunks[self.rank]
            p._local_shard.copy_(chunk)  # type: ignore[attr-defined]

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

        def _free_full_params_and_use_local_shard(params_to_free):
            # We may not always be able to free the full param, for example in
            # the case where world_size == 1 and the shard actually points to
            # the full parameter.
            for (param, can_free) in params_to_free:
                if can_free:
                    current_stream = torch.cuda.current_stream()
                    # Don't let PyTorch reuse this memory until all work in the
                    # current stream is complete
                    param.record_stream(current_stream)
                    _free_storage(param)

            # when CPU offload is enabled, _use_param_local_shard implicitly
            # offloads the local shard to CPU by making p.data point to
            # p._local_shard, which would reside on CPU.
            self._use_param_local_shard()

        if recurse:
            with contextlib.ExitStack() as stack:
                # Summon all params for any nested FSDP instances.
                for module in self.fsdp_modules(self):
                    stack.enter_context(
                        module._summon_full_params(
                            recurse=False,
                            writeback=writeback,
                            rank0_only=rank0_only,
                            offload_to_cpu=offload_to_cpu,
                        )
                    )
                # Yield to the caller, with full params in all nested instances.
                yield
            # Exiting from the ExitStack will re-shard params.
            return
        else:
            torch.cuda.synchronize()
            self._lazy_init()
            self._assert_state([TrainingState_.IDLE])
            # Set the state so that we assert when trying to go into
            # forward/backward.
            self.training_state = TrainingState_.SUMMON_FULL_PARAMS

            # Even if rank0_only = True, we need to materialize all params here
            # and free them right after as full param materialization requires
            # collective comm.
            currently_local_params = self._rebuild_full_params()
            # Wait for all_gather to finish before computation
            torch.cuda.current_stream().wait_stream(self._streams["all_gather"])
            my_rank = dist.get_rank(self.process_group)
            if offload_to_cpu and (not rank0_only or my_rank == 0):
                for p in self.params:
                    if p._is_sharded:
                        with torch.no_grad():
                            # Note that we avoid using p._full_param_padded
                            # directly here as we may not be using that param
                            # as the full_param from _rebuild_full_params (i.e.)
                            # in mixed precision.
                            for p, (full_param, _) in zip(
                                self.params, currently_local_params
                            ):
                                full_param = full_param.to(torch.device("cpu"))
                                self._update_p_data(p, output_tensor=full_param)

            if rank0_only and my_rank != 0:
                _free_full_params_and_use_local_shard(currently_local_params)
                try:
                    yield
                finally:
                    self.training_state = TrainingState_.IDLE
            else:
                # FSDP now has the full flattened parameter. Unflatten it to get the
                # full parameters.
                with contextlib.ExitStack() as stack:
                    # Invariant: rank == 0 or !rank0_only
                    stack.enter_context(self._fsdp_wrapped_module.unflatten_as_params())
                    try:
                        yield
                    finally:
                        if offload_to_cpu and (not rank0_only or my_rank == 0):
                            for p in self.params:
                                if p._is_sharded:
                                    with torch.no_grad():
                                        # Note that we avoid using
                                        # p._full_param_padded directly here as
                                        # we may not be using that param
                                        # as the full_param from
                                        # _rebuild_full_params (i.e. in mixed
                                        # precision.
                                        for p, (full_param, _) in zip(
                                            self.params, currently_local_params
                                        ):
                                            full_param = full_param.to(self.compute_device)
                                            self._update_p_data(
                                                p, output_tensor=full_param,
                                            )

                        if writeback:
                            self._write_back_current_shard(currently_local_params)
                        stack.close()
                        _free_full_params_and_use_local_shard(currently_local_params)
                        self.training_state = TrainingState_.IDLE

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
                discarded after the context manager exists;
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
        fsdp_modules = FullyShardedDataParallel.fsdp_modules(
            module, root_only=True
        )
        # Summon all params for all FSDP instances
        with contextlib.ExitStack() as stack:
            for module in fsdp_modules:
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

    def _register_pre_backward_hooks(self, outputs: Any) -> Any:
        """Register pre-backward hook to run before the wrapped module's
        backward. Hooks should be attached to all outputs from the forward.
        Returns:
            outputs: new outputs with hooks registered if they requires gradient.
        """
        # Reset before each backward pass
        self._need_rebuild_full_params = False

        if not torch.is_grad_enabled():
            return outputs  # don't register hooks if grad isn't enabled

        if self._is_root:
            # This actually means that only root instance has
            # _post_backward_callback_queued defined. Accidentally accessing this field
            # will assert on all other instances, giving us a nice bug checker.
            self._post_backward_callback_queued = False

        # Reset before each backward pass
        self._pre_backward_hook_has_run = False

        def _pre_backward_hook(*unused: Any) -> None:
            # Run ``_pre_backward_hook`` only once per backward pass
            if self._pre_backward_hook_has_run:
                return

            with torch.autograd.profiler.record_function("FullyShardedDataParallel._pre_backward_hook"):
                # try to queue final backward callback only once for root, so
                # that final backward callback is attached to the outer most
                # backward graph task and called after all the backward
                # calls are completed.
                if self._is_root:
                    self._queue_wait_for_post_backward()

                if self._pre_backward_hook_full_params_prefetched:
                    # Always wait for all_gather before rebuilding full params, just
                    # in case full params have already been prefetched in previous layer's
                    # pre-backward hook.
                    torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

                # Start of a backward pass for the first time in an backward pass.
                self._assert_state([TrainingState_.IDLE])
                self.training_state = TrainingState_.BACKWARD_PRE

                # All-gather full parameters, moving them to compute device if
                # necessary.
                self._rebuild_full_params()
                self._pre_backward_hook_full_params_prefetched = False
                # Wait for all_gather to finish before computation
                torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

                # Prefetch next layer's full params in backward pass,
                # since it is prefetching, no need to wait for all_gather stream.
                if self._need_prefetch_full_params(self.training_state):
                    self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1]._rebuild_full_params()  # type: ignore[operator]
                    self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1]._pre_backward_hook_full_params_prefetched = True

                self._pre_backward_hook_has_run = True
                # Prepare p.grad so that it is in the right shape, device, accumulated values, etc.
                self._prep_grads_for_backward()

        def _register_hook(t: torch.Tensor) -> torch.Tensor:
            if t.requires_grad:
                t.register_hook(_pre_backward_hook)
                self._need_rebuild_full_params = True
            return t

        # Attach hooks to Tensor outputs.
        outputs = _apply_to_tensors(_register_hook, outputs)

        return outputs

    def _register_post_backward_hooks(self) -> None:
        """
        Register backward hooks to reshard params and reduce-scatter grads.
        This is called during forward pass. The goal is to attach a hook
        on each of the parameter's gradient generating function (``grad_acc``
        below) so that the hook is called *after* all gradients for that
        param are computed.
        Goals:
        1. We want the hook to fire once and only once *after* all gradients
        are accumulated for a param.
        2. If it fires more than once, we end up incorrectly shard the grad
        multiple times. (could lead to dimension too small)
        3. If it fires once but too early or doesn't fire, we leave gradients
        unsharded. (could lead to dimension too large)
        Due to multiple-pass forward, this function can be called on
        the same parameter multiple times in a single forward pass. If we register
        the hook multiple time, we end up getting called multiple times. We
        could try to get a new hook every time and delete the previous one
        registered. However, due to *unknown reason* (I have debugged it for
        a long time!), in mixed precision mode, we get two different ``grad_acc``
        objects below during different calls of this function (in the same
        forward pass). If we keep the last one, the hook end up firing too
        early. In full precision mode, we luckily get the *same* ``grad_acc``
        object, so deleting and re-registering still ensured the hook fire
        once after all gradients are generated.
        Empirically, keep the first hook register per forward pass seems to
        work the best. We do need to remove the hook at the end of the
        backward pass. Otherwise, the next forward pass will not register
        a new hook, which is needed for a new forward pass.
        """
        if not torch.is_grad_enabled():
            return  # don't register grad hooks if grad isn't enabled
        for p in self.params:
            if p.requires_grad:
                if hasattr(p, "_shard_bwd_hook"):
                    continue
                # Register a hook on the first call, empirically, autograd
                # fires it at the end for this param, which makes sense.
                p_tmp = p.expand_as(p)  # Get a grad_fn on p_tmp.
                assert (
                    p_tmp.grad_fn is not None
                ), "p_tmp grad_fn should not be None, it is used to access \
                    p's AccumulateGrad object and register post hook on it."
                grad_acc = p_tmp.grad_fn.next_functions[0][
                    0
                ]  # Gets its AccumulateGrad object.
                handle = grad_acc.register_hook(
                    functools.partial(self._post_backward_hook, p)
                )
                p._shard_bwd_hook = (grad_acc, handle)  # type: ignore[attr-defined]

    @torch.no_grad()
    def _post_backward_hook(self, param: Parameter, *unused: Any) -> None:
        """
        At the start of :func:`_post_backward_hook`, ``param.grad`` contains the
        full gradient for the local batch. The reduce-scatter op will replace
        ``param.grad`` with a single shard of the summed gradient across all
        GPUs. This shard will align with the current GPU rank. For example::
            before reduce_scatter:
                param.grad (GPU #0): [1, 2, 3, 4]
                param.grad (GPU #1): [5, 6, 7, 8]
            after reduce_scatter:
                param.grad (GPU #0): [6, 8]    # 1+5, 2+6
                param.grad (GPU #1): [10, 12]  # 3+7, 4+8
        The local GPU's ``optim.step`` is responsible for updating a single
        shard of params, also corresponding to the current GPU's rank. This
        alignment is created by :func:`_shard_parameters`, which ensures that
        the local optimizer only sees the relevant parameter shard.
        """
        p_assert(
            hasattr(param, '_post_backward_called'),
            "Expected flag _post_backward_called to exist on param."
        )
        param._post_backward_called = True
        with torch.autograd.profiler.record_function("FullyShardedDataParallel._post_backward_hook"):
            # First hook callback will see PRE state. If we have multiple params,
            # then subsequent hook callbacks will see POST state.
            self._assert_state([TrainingState_.BACKWARD_PRE, TrainingState_.BACKWARD_POST])
            self.training_state = TrainingState_.BACKWARD_POST

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

            if (
                self._require_backward_grad_sync
                or self.sharding_strategy == ShardingStrategy.FULL_SHARD
            ):
                self._free_full_params(cast(List[FlatParameter], [param]))

            if self._mixed_precision_enabled_for_params():
                # Noop if reshard_after_forward=True because we'd free the param
                # shard when rebuilding the full params in the pre_beckward_hook.
                self._free_mp_shard(cast(List[FlatParameter], [param]))

            # Switch to local shard after backward. Note that
            # when CPU offload is enabled, _use_param_local_shard implicitly
            # offloads the local shard to CPU by making p.data point to
            # p._local_shard, which would reside on CPU.
            self._use_param_local_shard(cast(List[FlatParameter], [param]))

            # Prefetch previous layer's full params in backward pass post backward hook,
            # If next layer's backward computation is done and full params are freed,
            # no need to prefetch the full params again.
            # Only prefetch full params if any of the next layer's outputs requires grad
            if self._need_prefetch_full_params(self.training_state):
                self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1]._rebuild_full_params()  # type: ignore[operator]
                # Next layer's computation will start right after this all_gather,
                # Wait for all_gather to finish before computation.
                torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

            if not self._require_backward_grad_sync:
                return

            # Wait for all work in the current stream to finish, then start the
            # reductions in post_backward stream.
            self._streams["post_backward"].wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(self._streams["post_backward"]):
                orig_grad_data = param.grad.data
                if (
                    self._mixed_precision_enabled_for_reduce() and not self._low_precision_hook_enabled()
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
                if param._is_sharded:  # type: ignore[attr-defined]
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
                    # Currently the way for _is_sharded to be False is if
                    # world_size == 1 or sharding_strategy is NO_SHARD.
                    assert (
                        self.world_size == 1 or self.sharding_strategy == ShardingStrategy.NO_SHARD
                    ), "Currently the way for _is_sharded to be False is \
                        world_size == 1 or sharding_stratagy is set to be NO_SHARD"
                    if self.sharding_strategy == ShardingStrategy.NO_SHARD:
                        self._communication_hook(self._communication_hook_state, param.grad)

                    self._cast_grad_to_param_dtype(param.grad, param)

                # Regardless of sharding or not, offload the grad to CPU if we are
                # offloading params. This is so param and grad reside on same device
                # which is needed for the optimizer step.
                if self.cpu_offload.offload_params:
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

    def _queue_wait_for_post_backward(self) -> None:
        """Try to queue a `wait_for_post_backward` callback.
        Only called on root and only queue one callback at the beginning of
        outer most backward.
        """
        assert (
            self._is_root
        ), "_queue_wait_for_post_backward can only be called on root."
        if not self._post_backward_callback_queued:
            self._assert_state([TrainingState_.IDLE])
            self._post_backward_callback_queued = True
            Variable._execution_engine.queue_callback(self._wait_for_post_backward)

    @torch.no_grad()
    def _wait_for_post_backward(self) -> None:
        """Wait for post-backward to finish. Only called on root instance."""
        assert self._is_root, "_wait_for_post_backward can only be called on root."
        # Check if the root module has params and if any of them has
        # the `requires_grad` field set. If `requires_grad=False` for
        # all the params, the post_backward hook will not fire and the
        # state will remain in `TrainingState_.BACKWARD_PRE`.
        if any([p.requires_grad for p in self.params]):
            self._assert_state(TrainingState_.BACKWARD_POST)
        else:
            self._assert_state(TrainingState_.BACKWARD_PRE)

        if self._require_backward_grad_sync:
            torch.cuda.current_stream().wait_stream(self._streams["post_backward"])
            if self.cpu_offload.offload_params:
                # We need to wait for the non-blocking GPU ->
                # CPU grad transfers to finish. We need to do this for GPU -> CPU
                # copies because when grad is on CPU, it won't wait for any CUDA
                # stream to finish GPU -> CPU copies unless we explicitly block the
                # host-side with synchronize().
                torch.cuda.current_stream().synchronize()

        # A backward pass is done, clean up below.
        self._exec_order_data.reset()

        def _finalize_params(fsdp_module: FullyShardedDataParallel) -> None:
            """Helper used below on all fsdp modules."""
            for p in fsdp_module.params:
                if p.requires_grad:
                    if hasattr(p, "_shard_bwd_hook"):
                        assert len(p._shard_bwd_hook) == 2 and len(  # type: ignore[attr-defined]
                            p._shard_bwd_hook  # type: ignore[attr-defined]
                        ), (  # type: ignore[attr-defined]
                            "p._shard_bwd_hook fields are not valid."
                        )
                        p._shard_bwd_hook[1].remove()  # type: ignore[attr-defined]
                        delattr(p, "_shard_bwd_hook")
                    # Preserve the gradient accumulation state if not
                    # synchronizing: `p.grad` remains the unsharded gradient
                    # accumulated from prior `no_sync()` iterations, and
                    # `p._saved_grad_shard` remains the sharded gradient from
                    # the last synchronized iteration
                    if not self._require_backward_grad_sync:
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
                        p.grad = p._saved_grad_shard  # type: ignore[attr-defined]
                    else:
                        p_assert(
                            not p._is_sharded or not p._post_backward_called,
                            "All sharded parameters that received gradient should "
                            "use `_saved_grad_shard`"
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
            m._pre_backward_hook_has_run = False
            m.training_state = TrainingState_.IDLE

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

    def _update_p_data(self, p, output_tensor: torch.Tensor) -> None:
        """
        Helper function to update p.data pointer.
        Args:
            output_tensor (torch.Tensor): this tensor contains the data we just gathered.
        """
        p.data = output_tensor
        # Trim any padding and reshape to match original size.
        p.data = p.data[:p._unsharded_size.numel()].view(p._unsharded_size)  # type: ignore[attr-defined]

    @torch.no_grad()
    def _rebuild_full_params(self) -> List[Tuple[torch.Tensor, bool]]:
        """
        Gather all shards of params.
        """
        # _summon_full_params must do a full precision rebuild even under mixed
        # precision, because it is used for e.g. checkpoint where we'd like to
        # checkpoint in full precision.
        force_full_precision = (self.training_state == TrainingState_.SUMMON_FULL_PARAMS)
        # full param output tensors and a flag indicating whether
        # _summon_full_params can free them or not. It is possible that we can't
        # free the full param, which currently occurs when the returned
        # parameter points to the unsharded param when world_size == 1, or when
        # we're returning the full parameter and reshard_after_forward=False
        # (because we need to ensure p._full_param_padded stays intact)
        output_tensors: List[Tuple[torch.Tensor, bool]] = []
        with torch.cuda.stream(self._streams["all_gather"]):
            for p in self.params:
                mixed_precision_cast_ran = (
                    self._mixed_precision_enabled_for_params()
                    and not force_full_precision
                )
                if mixed_precision_cast_ran:
                    self._cast_param_shards_to_dtype()
                    # TODO: remove below
                    for p in self.params:
                        assert p.dtype == self.mixed_precision.param_dtype
                # We can skip moving params to GPU if mixed precision, as p.data
                # would then be pointing to p._mp_shard which is already on
                # self.compute_device.
                if self.cpu_offload.offload_params and not mixed_precision_cast_ran:
                    # Move params to GPU if needed. Note that we don't use
                    # self._full_param_padded.device here because the attr is
                    # not set always, i.e. when world_size=1 and
                    # p._is_sharded = False. However when it is set, the
                    # device is always self.compute_device.
                    p.data = p.data.to(self.compute_device, non_blocking=True)
                # Check the validity of this `_rebuild_full_params()` call in
                # terms of execution order (regardless of if FSDP actually
                # needs to all-gather or not)
                self._check_rebuild_full_params(p)
                # e.g., when world_size == 1
                if not p._is_sharded:  # type: ignore[attr-defined]
                    if mixed_precision_cast_ran:
                        # p.data should be the same type as p._mp_shard, and it
                        # is safe to free.
                        assert p.data.dtype == p._mp_shard.dtype
                        # Safe to free because p.data points to the mp shard.
                        output_tensors.append((p.data, True))
                    else:
                        # p.data points to the unsharded parameter, so not safe to
                        # free.
                        output_tensors.append((p.data, False))
                    continue
                # If full param has been rebuilt or has not been freed, no need to call all gather
                elif (
                    p._full_param_padded.storage().size()  # type: ignore[attr-defined]
                    == p._full_param_padded.size().numel()  # type: ignore[attr-defined]
                ):
                    # Check that the full param is in the expected precision, if
                    # training with mixed precision
                    if mixed_precision_cast_ran:
                        if p._full_param_padded.dtype != self.mixed_precision.param_dtype:
                            raise ValueError(
                                "_rebuild_full_params: Expected full param to be "
                                f"of type {self.mixed_precision.param_dtype}, "
                                f"but got {p._full_param_padded.dtype}!"
                            )
                    # output is full_param_padded which can be freed depending
                    # on reshard_after_forward (this path is exercised by tests
                    # in test_fsdp_summon_full_params).
                    output_tensors.append((p._full_param_padded, self.reshard_after_forward))

                    self._update_p_data(p, output_tensor=p._full_param_padded)  # type: ignore[attr-defined]
                    continue
                else:
                    # If full param has not been rebuilt or has been freed, call all gather
                    p_data = p.data  # type: ignore[attr-defined]
                    p_full_size = p._full_param_padded.size()  # type: ignore[attr-defined]
                    assert (
                        p_full_size.numel() == p_data.numel() * self.world_size
                    ), "Param full size should be equal to its shard size multiply world_size."
                    assert (
                        p._full_param_padded.storage().size() == 0  # type: ignore[attr-defined]
                    ), "Full param's storage should have been freed before if all gather is needed."  # type: ignore[attr-defined]
                    if (
                        self._mixed_precision_enabled_for_params()
                        and force_full_precision
                    ):
                        # p._full_param_padded has the reduced precision type,
                        # but we need full precision rebuild as we're in
                        # _summon_full_params. Note that this is why
                        # _summon_full_params collects locally used params from
                        # _rebuild_full_params instead of relying on
                        # p._full_param_padded, as it may not always be
                        # allocated such as during mixed precision.
                        output_tensor = p_data.new_zeros(p_full_size)
                    else:
                        # Allocate based on full size from all shards.
                        _alloc_storage(p._full_param_padded, size=p_full_size)  # type: ignore[attr-defined]
                        output_tensor = p._full_param_padded  # type: ignore[attr-defined]
                    # Fill output_tensor with (p.data for each shard in self.world_size)
                    dist._all_gather_base(
                        output_tensor, p_data, group=self.process_group
                    )

                    # The full parameter, which can be freed. Note that we
                    # append here before update_p_data so as to not saved the
                    # tensor with padding trimmed, which causes issues with
                    # writeback in _summon_full_params.
                    output_tensors.append((output_tensor, True))
                    # Set p.data = output_tensor (with padding trimmed)
                    self._update_p_data(p, output_tensor=output_tensor)
                    # We can free the reduced precision shard as we have the
                    # full precision parameter.
                    if (
                        self._mixed_precision_enabled_for_params()
                    ):
                        self._free_mp_shard(cast(List[FlatParameter], [p]))
        return output_tensors

    def _check_rebuild_full_params(self, param: FlatParameter):
        """
        Checks the validity of a call to :meth:`_rebuild_full_params` in terms
        of the execution order. If on the first iteration, this uses an
        all-gather to check that all ranks are running ``forward()`` with the
        same parameter, erroring if not, and on subsequent iterations, if the
        forward order differs from that of the first iteration (meaning that we
        can no longer guarantee correct execution since all-gathers may be
        mismatched), then we issue a warning to the user. This only issues
        warnings on the first deviating iteration and stops checking
        thereafter.

        Only the :meth:`_rebuild_full_params` calls in the forward pass are
        checked since a correct forward order should imply a correct
        pre-backward order for typical cases.

        Executing in ``no_sync()`` does not affect this check for
        ``FULL_SHARD`` and ``SHARD_GRAD_OP``: (1) Being in ``no_sync()`` in the
        first iteration does not yield a different forward
        :meth:`_rebuild_full_params()` sequence, and (2) being in ``no_sync()``
        in a later iteration does not give false positive warnings since the
        forward :meth:`_rebuild_full_params()` sequence still matches the first
        iteration sequence (for ``FULL_SHARD``) or the first iteration
        sequence's prefix (for ``SHARD_GRAD_OP``).
        """
        # Only check when rebuilding the full parameters in the forward pass,
        # and skip the check (1) when in eval mode since then there is not a
        # safe point at which to reset the execution order data and (2) if
        # world size is 1 since then there is no chance of desynchronization
        if self.training_state != TrainingState_.FORWARD or \
                not self.training or self.world_size == 1:
            return
        eod = self._exec_order_data
        param_index = eod.get_param_index(param)
        if not eod.is_first_iter:
            # Only issue warnings on the first deviating iteration and stop
            # checking thereafter to avoid flooding the console
            if eod.warn_status == _ExecOrderWarnStatus.WARNED:
                return
            # However, we may issue multiple warnings on the first deviating
            # iteration to help debugging, where either:
            # 1. This iteration sees an extra `_rebuild_full_params()` in
            # `forward()` compared to the first iteration
            msg_prefix = curr_param_order = None  # non-`None` means we warn
            if eod.index >= len(eod.param_order):
                msg_prefix = "Expected to not rebuild any more parameters " \
                    "in `forward()` for this module but trying to rebuild " \
                    "parameters for "
                curr_param_order = eod.param_order + [param_index]
            else:
                expected_param_index = eod.param_order[eod.index]
                # 2. This iteration sees the same number of
                # `_rebuild_full_params()` (so far) but the current parameter
                # differs
                if param_index != expected_param_index:
                    expected_param_names = eod.get_unflat_param_names(expected_param_index)
                    assert len(expected_param_names) > 0, \
                        "Expected parameter should always be valid"
                    msg_prefix = "Expected to rebuild parameters in " \
                        f"`forward()` for {expected_param_names} but " \
                        "instead trying to rebuild parameters for "
                    curr_param_order = eod.param_order[:eod.index - 1] + [param_index]
            to_issue_warning = msg_prefix is not None
            if to_issue_warning:
                assert curr_param_order is not None
                param_names = eod.get_unflat_param_names(param_index)
                is_added_param = len(param_names) == 0
                if is_added_param:
                    msg_suffix = "a newly-added parameter since construction time"
                else:
                    msg_suffix = f"{param_names}"
                sub_msg = msg_prefix + msg_suffix
                first_iter_param_names = [
                    eod.get_unflat_param_names(index) for index in eod.param_order
                ]
                curr_iter_param_names = [
                    eod.get_unflat_param_names(index) for index in curr_param_order
                ]
                warnings.warn(
                    "Forward order differs from that of the first iteration "
                    f"on rank {self.rank} -- collectives are unchecked and may "
                    "give incorrect results or hang\n" + sub_msg + "\n" +
                    f"First iteration's forward order: {first_iter_param_names}"
                    "\nThis iteration's forward order (so far): "
                    f"{curr_iter_param_names}"
                )
                eod.warn_status = _ExecOrderWarnStatus.WARNING
            eod.index += 1
        else:
            # Use `compute_device` instead of the parameter's device in case it
            # is offloaded on CPU and we are using NCCL backend, which requires
            # communicated tensors be on GPU
            device = self.compute_device
            indices = torch.zeros(self.world_size, dtype=torch.int32, device=device)
            index = torch.tensor([param_index], dtype=torch.int32, device=device)
            dist._all_gather_base(indices, index, group=self.process_group)
            # Check that all ranks plan to all-gather the same parameter index
            for (r1, i1), (r2, i2) in itertools.combinations(
                ((rank, indices[rank]) for rank in range(self.world_size)), 2,
            ):
                if not torch.equal(i1, i2):
                    r1_param_names = eod.get_unflat_param_names(i1)
                    r2_param_names = eod.get_unflat_param_names(i2)
                    raise RuntimeError(
                        f"Forward order differs across ranks: rank {r1} is "
                        "rebuilding full parameters in `forward()` for "
                        f"{r1_param_names} while rank {r2} is rebuilding full "
                        f"parameters in `forward()` for {r2_param_names}"
                    )
            eod.param_order.append(param_index)

    @torch.no_grad()
    def _prep_grads_for_backward(self) -> None:
        """Make sure p.grad has the correct size/device, otherwise set it to None."""
        for p in self.params:
            if p.grad is not None and (
                p.grad.size() != p._unsharded_size  # type: ignore[attr-defined]
                or p.grad.device != p.device
            ):
                offloaded: bool = p.grad.device != p.device
                if offloaded:
                    assert self.cpu_offload.offload_params, \
                        "`p.grad.device` and `p.device` should be the same " \
                        "if not offloading parameters to CPU"
                prev_iter_outside_no_sync: bool = \
                    p.grad.size() == p._local_shard.shape  # type: ignore[attr-defined]
                # As long as the previous iteration was outside `no_sync()`,
                # then we must save the gradient in `_saved_grad_shard`, even
                # if the current iteration is inside `no_sync()`. This is to
                # prepare for the next iteration outside `no_sync()`, which may
                # try to accumulate gradients. FSDP accumulates gradients in
                # the separate variable `p._saved_grad_shard` to leave `p.grad`
                # for the per-iteration gradient.
                if prev_iter_outside_no_sync:
                    # FSDP currently does not support gradient accumulation
                    # outside `no_sync()` when using CPU offloading (see the
                    # warning in the class's docstring).
                    if not offloaded:
                        p._saved_grad_shard = p.grad.data  # type: ignore[attr-defined]
                p.grad = None

    @torch.no_grad()
    def _free_full_params(self, params: Optional[List[FlatParameter]] = None) -> None:
        """
        Free up storage for full parameters.
        """
        if params is None:
            params = self.params
        current_stream = torch.cuda.current_stream()
        for p in params:
            # e.g., world_size == 1 or self.sharding_strategy = NO_SHARD
            if not p._is_sharded:  # type: ignore[attr-defined]
                if (
                    self._mixed_precision_enabled_for_params()
                ):
                    self._free_mp_shard(cast(List[FlatParameter], [p]))
                continue
            # Don't let PyTorch reuse this memory until all work in the current
            # stream is complete.
            p._full_param_padded.record_stream(current_stream)  # type: ignore[attr-defined]
            # There may be external references to the Tensor Storage that we
            # can't modify, such as references that are created by
            # ctx.save_for_backward in the forward pass. Thus when we
            # unshard parameters, we should reuse the original Tensor
            # Storage object and unshard it in-place. For now, just resize
            # the Storage to 0 to save memory.
            _free_storage(p._full_param_padded)  # type: ignore[attr-defined]

    @torch.no_grad()
    def _use_param_local_shard(
        self, params: Optional[List[FlatParameter]] = None
    ) -> None:
        """Use local shard for a list of params. Also implicitly offloads
        parameters back to CPU if we are CPU offloading."""
        if params is None:
            params = self.params
        for p in params:
            if self.cpu_offload.offload_params:
                # Ensure local_shard resides in CPU if we are offloading params.
                assert p._local_shard.device == torch.device(  # type: ignore[attr-defined]
                    "cpu"
                ), "Expected p._local_shard to be on CPU"
            p.data = p._local_shard  # type: ignore[attr-defined]

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
                old_flags.append((m, m._require_backward_grad_sync))
                m._require_backward_grad_sync = False
        try:
            yield
        finally:
            for m, old_flag in old_flags:
                assert not m._require_backward_grad_sync, (
                    "`_require_backward_grad_sync` was incorrectly set to "
                    "`True` while in the `no_sync()` context manager"
                )
                m._require_backward_grad_sync = old_flag

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

        .. warning:: If you do not pass ``model.parameters()`` as the first
            argument to the optimizer, then you should pass that same value to
            this method as ``optim_input``.

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
                ``model.parameters()``. (Default: ``None``)
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
        return _optim_state_dict(
            model=model,
            optim=optim,
            optim_input=optim_input,
            rank0_only=rank0_only,
            shard_state=False,
            group=group,
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
        The API is similar to :meth:``full_optim_state_dict`` but this API
        chunks all non-zero-dimension states to ShardedTensor to save memory.
        This API should only be used when the model state_dict is derived with
        the context manager ``with state_dict_type(SHARDED_STATE_DICT):``.

        For the detail usages, refer to the :meth:``full_optim_state_dict`` doc.

        .. warning:: The returned state dict contains ShardedTensor and cannot be
            directly used by the regular ``optim.load_state_dict``.
        """

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
        )

    @staticmethod
    def shard_full_optim_state_dict(
        full_optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim_input: Optional[Union[
            List[Dict[str, Any]], Iterable[torch.nn.Parameter],
        ]] = None,
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

        .. warning:: If you do not pass ``model.parameters()`` as the first
            argument to the optimizer, then you should pass that same value to
            this method as ``optim_input``.

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
                ``model.parameters()``. (Default: ``None``)

        Returns:
            Dict[str, Any]: The full optimizer state dict now remapped to
            flattened parameters instead of unflattened parameters and
            restricted to only include this rank's part of the optimizer state.
        """
        sharded_osd = _flatten_optim_state_dict(
            full_optim_state_dict, model, True,
        )
        return _rekey_sharded_optim_state_dict(sharded_osd, model, optim_input)

    @staticmethod
    def flatten_sharded_optim_state_dict(
        sharded_optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]], Iterable[torch.nn.Parameter],
            ]
        ] = None,
    ) -> Dict[str, Any]:
        """
        The API is similar to :meth:``shard_full_optim_state_dict``. The only
        difference is that the input ``sharded_optim_state_dict`` should be
        returned from :meth:`sharded_optim_state_dict`. Therefore, there will be
        allgather calls on each rank to gather ShardedTensor.

        Args:
            sharded_optim_state_dict (Dict[str, Any]): Optimizer state dict
                corresponding to the unflattened parameters and holding the
                sharded optimizer state.
            model (torch.nn.Module):
                Refer to :meth:``shard_full_optim_state_dict``.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Refer to :meth:``shard_full_optim_state_dict``.

        Returns:
            Refer to :meth:``shard_full_optim_state_dict``.
        """

        # TODO: The implementation is the same as ``shard_full_optim_state_dict``.
        # See the TODO in ``shard_full_optim_state_dict`` for the future
        # unification plan.
        flattened_osd = _flatten_optim_state_dict(
            sharded_optim_state_dict,
            model=model,
            shard_state=True,
        )
        return _rekey_sharded_optim_state_dict(flattened_osd, model, optim_input)

    @staticmethod
    def scatter_full_optim_state_dict(
        full_optim_state_dict: Optional[Dict[str, Any]],
        model: torch.nn.Module,
        optim_input: Optional[Union[
            List[Dict[str, Any]], Iterable[torch.nn.Parameter],
        ]] = None,
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
                ``model.parameters()``. (Default: ``None``)
            group (dist.ProcessGroup): Model's process group or ``None`` if
                using the default process group. (Default: ``None``)

        Returns:
            Dict[str, Any]: The full optimizer state dict now remapped to
            flattened parameters instead of unflattened parameters and
            restricted to only include this rank's part of the optimizer state.
        """
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
        # rank's `optim_input`
        sharded_osd = _rekey_sharded_optim_state_dict(sharded_osd, model, optim_input)
        return sharded_osd

    @staticmethod
    def rekey_optim_state_dict(
        optim_state_dict: Dict[str, Any],
        optim_state_key_type: OptimStateKeyType,
        model: torch.nn.Module,
        optim_input: Optional[Union[
            List[Dict[str, Any]], Iterable[torch.nn.Parameter],
        ]] = None,
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
        assert optim_state_key_type in \
            (OptimStateKeyType.PARAM_NAME, OptimStateKeyType.PARAM_ID)
        osd = optim_state_dict  # alias
        # Validate that the existing parameter keys are uniformly typed
        uses_param_name_mask = [
            type(param_key) is str for param_key in osd["state"]
        ]
        uses_param_id_mask = [
            type(param_key) is int for param_key in osd["state"]
        ]
        if (any(uses_param_name_mask) and not all(uses_param_name_mask)) or \
                (any(uses_param_id_mask) and not all(uses_param_id_mask)):
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
            param_id_to_param = _get_param_id_to_param(model, optim_input)
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
            param_to_param_id = _get_param_to_param_id(model, optim_input)
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


def _get_default_cuda_device(module: nn.Module) -> torch.device:
    """Try to infer CUDA device from module parameters."""
    try:
        compute_device = next(module.parameters()).device
        if compute_device.type == "cuda":
            return compute_device
    # e.g., if module does not have parameters, it will throw StopIteration,
    # in this case, instead of raising exception, return cuda device.
    except StopIteration:
        pass
    # Fall back to current CUDA device
    return torch.device("cuda", torch.cuda.current_device())


def _free_storage(data: torch.Tensor) -> None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert (
            data.storage_offset() == 0
        ), "The tensor is not the sole occupant of the storage."
        data.storage().resize_(0)  # type: ignore[attr-defined]


@torch.no_grad()
def _alloc_storage(data: torch.Tensor, size: torch.Size) -> None:
    """Allocate storage for a tensor."""
    if data.storage().size() == size.numel():  # no need to reallocate
        return
    assert (
        data.storage().size() == 0
    ), "Then tensor storage should have been resized to be 0."
    data.storage().resize_(size.numel())  # type: ignore[attr-defined]

def p_assert(cond: Any, s: Any) -> None:
    """This is used as an alternate to ``assert`` when in the backward context
    to print the error message ``s`` since otherwise, it is swallowed."""
    if not cond:
        print(s)
        traceback.print_stack()
        raise AssertionError

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
                    param._prefixed_param_names if isinstance(param, FlatParameter)
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
    """Cleans the parameter or buffer name by removing any FSDP-related
    prefixes."""
    # FSDP full tensor names may not have both (i.e. `FSDP_PREFIX`), so we
    # call `replace()` twice separately
    tensor_name = tensor_name.replace(FSDP_WRAPPED_MODULE + ".", "")
    tensor_name = tensor_name.replace(FPW_MODULE + ".", "")
    # TODO: Explicitly replacing checkpoint_wrapper prefix is not ideal,
    # as it increases coupling between CheckpointWrapper and FSDP. This is also not
    # scalable for additional wrapped modules, we should come up with a general solution
    # for this issue.
    tensor_name = tensor_name.replace(_CHECKPOINT_PREFIX + ".", "")
    return tensor_name
