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
    TYPE_CHECKING,
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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributed import ProcessGroup
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    init_from_local_shards,
)
from torch.distributed.distributed_c10d import _get_default_group
from torch.nn.parameter import Parameter

from ._optim_utils import (
    _broadcast_pos_dim_tensor_states,
    _broadcast_processed_optim_state_dict,
    _flatten_full_optim_state_dict,
    _get_flat_param_to_fsdp_module,
    _get_param_id_to_param,
    _get_param_to_param_id,
    _process_pos_dim_tensor_state,
    _unflatten_optim_state,
)
from ._utils import (
    _apply_to_modules, _apply_to_tensors, _replace_by_prefix,
    _override_batchnorm_mixed_precision, _contains_batchnorm
)
from .flatten_params_wrapper import (
    FLAT_PARAM,
    FPW_MODULE,
    FlatParameter,
    FlattenParamsWrapper,
)
from .wrap import _recursive_wrap, _wrap_batchnorm_individually, _or_policy

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: F401

_TORCHDISTX_AVAIL = True
try:
    from torchdistx import deferred_init, fake
except ImportError:
    _TORCHDISTX_AVAIL = False


FSDP_WRAPPED_MODULE = "_fsdp_wrapped_module"
FSDP_PREFIX = FSDP_WRAPPED_MODULE + "." + FPW_MODULE + "."


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
    Specify which sharding strategy will be used for the distributed training.
    FULL_SHARD: if Shard parameters, gradients and optimizer states, this algorithm
                inserts all_gather before forward and backward computation to gather
                parameters, also inserts reduce_scatter after backward computation for
                synchronizing and sharding gradients. Sharded optimizer states are
                updated locally.
    SHARD_GRAD_OP: Shard optimizer states and gradients, this algorithm inserts all_gather
                   before forward computation and keeps the full parameters in
                   GPU memory until backward computation is done. It inserts reduce_scater
                   after backward computation for synchronizing and sharding gradients.
                   Sharded optimizer states are updated locally.
    NO_SHARD: This is similar to PyTorch ``DistributedDataParallel`` API. Parameters, gradients
              and optimizer states are replicated among ranks, all_reduce is inserted after
              backward computation is done for synchronizing gradients. Full optimizer states
              are updated in each rank.
    HYBRID_SHARD(future support): apply FULL_SHARD algorithm in the intra node and
                                  apply NO_SHARD algorithm in the inter nodes.

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
    # TODO: state dict offloading
    # https://github.com/pytorch/pytorch/issues/67224


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
                   >>> with fsdp.state_dict_type(StateDictType.LOCAL_STATE_DICT):
                   >>>     state = fsdp.state_dict()  # loads local state dict
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
        >>> cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        >>> with fsdp.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, cfg):
        >>>     state = fsdp.state_dict()
        >>>     # state will be empty on non rank 0 and contain CPU tensors on rank 0.
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
        # Save `root_modules.parameters()` to `_all_flat_params` instead of
        # re-materializing each time to avoid the result depending on the
        # calling context (e.g. when some parameters have been rebuilt)
        self._all_flat_params = list(root_module.parameters())
        self._param_to_unflat_param_names = cast(
            Dict[FlatParameter, List[str]],
            _get_param_to_unflat_param_names(root_module)
        )  # `root_module.parameters()` should only contain `FlatParameter`s

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
        Module should be already placed on the destination device or
        device is set properly using ``torch.cuda.set_device(device_id)``.
        FSDP will get compute device from module first, if module device
        is CPU, FSDP will then get compute device from current device.

    .. warning::
        FSDP currently does not support gradient accumulation outside
        `no_sync()` when using CPU offloading. Trying to do so yields incorrect
        results since FSDP will use the newly-reduced gradient instead of
        accumulating with any existing gradient.

    .. warning::
        Changing the original parameter variable names after construction will
        lead to undefined behavior.

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
        auto_wrap_policy (Optional[Callable]):
            A callable specifying a policy to recursively wrap layers with FSDP.
            Note that this policy currently will only apply to child modules of
            the passed in module. The remainder modules are always wrapped in
            the returned FSDP root instance.
            ``size_based_auto_wrap_policy`` written in ``torch.distributed.fsdp.wrap`` is
            an example of ``auto_wrap_policy`` callable, this policy wraps layers
            with the number of parameters larger than 100M. ``transformer_auto_wrap_policy``
            written in ``torch.distributed.fsdp.wrap`` is an example of ``auto_wrap_policy``
            callable for tranformer-like model architectures. Users can supply the customized
            ``auto_wrap_policy`` callable that should accept following arguments:
            ``module: nn.Module``, ``recurse: bool``, ``unwrapped_params: int``,
            extra customized arguments could be added to the customized
            ``auto_wrap_policy`` callable as well. It is a good practice to print out
            the sharded model and check whether the sharded model is what
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

        backward_prefetch (Optional[BackwardPrefetch]):
            This is an experimental feature that is subject to change in the
            the near future. It allows users to enable two different backward_prefetch
            algorithms to help backward communication and computation overlapping.
            Pros and cons of each algorithm is explained in the class ``BackwardPrefetch``.
        mixed_precision: (Optional[MixedPrecision]): A ``MixedPrecision`` instance
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
            avoid sharding specific parameters when using an
            ``auto_wrap_policy`` or if parameters' sharding is not managed by
            FSDP. (Default: ``None``)
        param_init_fn: (Optional[Callable[[nn.Module], None]]):
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

                >>> module = MyModule(device="meta")
                >>> def my_init_fn(module):
                >>>     # responsible for initializing a module, such as with reset_parameters
                >>> fsdp_model = FSDP(module, param_init_fn=my_init_fn, auto_wrap_policy=size_based_auto_wrap_policy)
                >>> print(next(fsdp_model.parameters()).device) # current CUDA device
                >>> # With torchdistX
                >>> module = deferred_init.deferred_init(MyModule, device="cuda")
                >>> # Will initialize via deferred_init.materialize_module().
                >>> fsdp_model = FSDP(module, auto_wrap_policy=size_based_auto_wrap_policy)

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
    ):
        torch._C._log_api_usage_once("torch.distributed.fsdp")
        super().__init__()
        # Validate the ignored modules and derive the ignored parameters/buffers
        ignored_modules = self._get_ignored_modules(module, ignored_modules)
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
                mixed_precision=mixed_precision,
                param_init_fn=param_init_fn,
            )

        self.process_group = process_group or _get_default_group()
        self.rank = self.process_group.rank()
        self.world_size = self.process_group.size()

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

        # device for computation, if module is on GPU, use module.device;
        # if module is on CPU, use current device;
        self.compute_device = _get_default_cuda_device(module)

        # Enum to indicate if we're in the forward/backward pass, idle, etc.
        self.training_state = TrainingState_.IDLE

        # setting two factors to avoid underflow and overflow
        self.gradient_predivide_factor: float = self._get_gradient_predivide_factor(
            self.world_size
        )
        self.gradient_postdivide_factor: float = (
            self.world_size / self.gradient_predivide_factor
        )

        self.numel_padded_per_param: List[int] = []
        self.cpu_offload = cpu_offload or CPUOffload()
        self.backward_prefetch = backward_prefetch
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

        self._fsdp_wrapped_module: FlattenParamsWrapper = FlattenParamsWrapper(
            module, param_list=params
        )
        assert getattr(self, FSDP_WRAPPED_MODULE) is self._fsdp_wrapped_module
        del module  # free original module in case it helps garbage collection
        if self._fsdp_wrapped_module.flat_param is not None:
            self.params = [self._fsdp_wrapped_module.flat_param]
        else:
            self.params = []

        # Shard module parameters in place
        self._shard_parameters()

        # Make sure all parameters are sharded.
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
            raise ValueError(
                "Trying to ignore the top-level module passed into the FSDP "
                "constructor itself will result in all parameters being "
                f"ignored and is not supported: {module}"
            )
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
            ignored_param_names.update(unflat_param_names)
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

    @property
    def module(self) -> FlattenParamsWrapper:
        """make model.module accessible, just like DDP."""
        assert isinstance(self._fsdp_wrapped_module, FlattenParamsWrapper)
        return self._fsdp_wrapped_module

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
        within another ``_summon_full_params`` context.

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

    # setting two factors 'self.gradient_predivide_factor'
    # and 'self.gradient_postdivide_factor' to avoid underflow and overflow
    def _get_gradient_predivide_factor(self, world_size: int) -> float:
        factor: int = 1
        while world_size % factor == 0 and world_size / factor > factor:
            factor *= 2
        return float(factor)

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
        for p in self.params:
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
            p._orig_size = p.size()  # type: ignore[attr-defined]

            if not p._is_sharded:  # type: ignore[attr-defined]
                self.numel_padded_per_param.append(0)
                continue

            # Save the original storage and free it later on.
            # Since we're modifying the tensor's storage directly,
            # make sure the tensor is the sole occupant of the storage.
            assert (
                p.storage_offset() == 0
            ), "The tensor is not the sole occupant of the storage."
            orig_storage = p.storage()

            # Replace p with the relevant shard.
            local_shard, num_padded = self._get_shard(p)
            p.set_(local_shard)  # type: ignore[call-overload]
            p.shard_by_offsets(
                self.rank * local_shard.numel(),
                (self.rank + 1) * local_shard.numel() - 1,
                num_padded,
            )
            self.numel_padded_per_param.append(num_padded)

            # Free storage that contains the original full data.
            if orig_storage.size() > 0:
                orig_storage.resize_(0)  # type: ignore[attr-defined]

        assert len(self.numel_padded_per_param) == len(
            self.params
        ), "numel_padded_per_param is not populated correctly."

    @staticmethod
    def _get_chunk(
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> Tuple[torch.Tensor, int]:
        """Returns the unpadded chunk as a view and the number of padding
        elements of a full tensor for the given rank and world size."""
        # Shard using `torch.chunk()` to match all-gather/reduce-scatter.
        chunks = torch.flatten(tensor).chunk(world_size)
        if len(chunks) < (rank + 1):
            # If there are not enough chunks to shard across ranks, create an
            # empty chunk that will just be padded with zeros to be the
            # appropriate size.
            chunk = chunks[0].new_empty(0)
        else:
            chunk = chunks[rank]
        # Determine number of padding elements.
        num_to_pad = chunks[0].numel() - chunk.numel()
        assert num_to_pad >= 0, \
            "Chunk's size should at most the first chunk's size"
        return chunk, num_to_pad

    @staticmethod
    def _get_shard_functional(
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> Tuple[torch.Tensor, int]:
        """Functional version of :meth:`_get_shard`."""
        chunk, num_to_pad = FullyShardedDataParallel._get_chunk(
            tensor, rank, world_size,
        )
        # We always need to clone here regardless of the padding and even
        # though `chunk` is a view of `tensor` because `tensor` may be
        # deallocated after this method returns
        shard = chunk.clone()
        if num_to_pad > 0:
            shard = F.pad(shard, [0, num_to_pad])
        return shard, num_to_pad

    def _get_shard(
        self,
        tensor: torch.Tensor,
        rank: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Returns the local shard and the number of padding elements of a full
        tensor for the calling rank if ``rank=None`` or for the rank ``rank``
        if not ``None``."""
        rank = self.rank if rank is None else rank
        return FullyShardedDataParallel._get_shard_functional(
            tensor, rank, self.world_size,
        )

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self.module.__getitem__(key)  # type: ignore[operator]

    def _reset_lazy_init(self) -> None:
        """
        Reset instance so :func:`_lazy_init` will run on the next forward.
        Currently this is only called in __init__
        """
        self._is_root: Optional[bool] = None
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._fsdp_graph_order: List[nn.Module] = []
        self._my_fsdp_idx_in_graph: Optional[int] = None
        for p in self.params:
            if hasattr(p, "_local_shard"):
                # reset attributes that are added in _init_param_attributes, as
                # part of _lazy_init
                del p._local_shard  # type: ignore[attr-defined]
        # set 'self.reshard_after_forward' flag based on self.sharding_strategy
        self._init_reshard_after_forward()

    def _lazy_init(self) -> None:
        """Initialization steps that should happen lazily, typically right
        before the first forward pass.
        """
        # Initialize param attributes lazily, in case the param's dtype or
        # device changes after __init__.
        for p in self.params:
            self._init_param_attributes(p)

        # Initialize _is_root and setup streams. These steps would ideally
        # happen in __init__, but _is_root can only be determined after the
        # entire model hierarchy is setup, thus we run it lazily.
        if self._is_root is None:
            # _is_root means that we are in the outermost module's forward.
            self._set_is_root()
            self._setup_streams()

        if self._is_root:
            # Buffers stay on GPU, and don't get sharded. Since _cast_buffers
            # applies recursively, we only call this from the root instance.
            self._cast_buffers(recurse=True)

            # Don't free the full params for the outer-most (root) instance,
            # In most cases, root instance contains params in the last layers
            # or has no params. In these cases, those params will be needed
            # immediately after for the backward pass. Note that this only
            # applies currently when freeing parameters at end of layer's
            # forward pass.
            self.reshard_after_forward = False

            # Due to the use of streams, we need to make sure the previous
            # ``optim.step()`` is done before we all-gather parameters.
            self._wait_for_previous_optim_step()

    @torch.no_grad()
    def _init_param_attributes(self, p: Parameter) -> None:
        """
        We manage several attributes on each Parameter instance. The first two
        are set by :func:`_shard_parameters`:
            ``_is_sharded``: ``True`` if the Parameter is sharded or ``False``
                if the Parameter is intentionally not sharded (in which case we
                will all-reduce grads for this param). Currently the way
                `_is_sharded = False` is if world_size = 1 or sharding strategy
                is NO_SHARD.
            ``_orig_size``: the size of the original Parameter (before sharding)
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
        assert hasattr(p, "_is_sharded") and hasattr(
            p, "_orig_size"
        ), "Parameters should have been sharded during construction."
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

    def _set_is_root(self) -> None:
        """If ``True``, implies that no other :class:`FullyShardedDataParallel`
        instance wraps this one. Called once by :func:`_lazy_init`.
        """
        if self._is_root is not None:
            return
        # No FSDP instance wraps this, else _is_root would be set to False.
        self._is_root = True
        self._exec_order_data.init(self)
        # If final backward callback is never been queued, state should be IDLE.
        # If final backward callback is queued, the callback should be finished
        # and the state was reset to be IDLE.
        # This should be asserted at the beginning of forward pass in the root instance only.
        # For children instances, if they are checkpointed, state will not be reset to
        # IDLE after each inner forward/backward.
        self._assert_state(TrainingState_.IDLE)
        for m in self.modules():
            if m is not self and isinstance(m, FullyShardedDataParallel):
                # We relax the assert for non-root instance, when the nested initialized module is wrapped
                # again in FSDP later, for example after training to run inference.
                assert (
                    m._is_root is None or not m._is_root
                ), "Non-root instance's _is_root flag should have not been set yet" \
                    "or has already been set as False."
                if m._is_root is None:
                    m._is_root = False

    def _setup_streams(self) -> None:
        """Create streams to overlap data transfer and computation."""
        if len(self._streams) > 0 or not self._is_root:
            return

        if torch.cuda.is_available():
            # Stream for all-gathering parameters.
            self._streams["all_gather"] = torch.cuda.Stream()
            # Stream for overlapping grad reduction with the backward pass.
            self._streams["post_backward"] = torch.cuda.Stream()
            # Stream to move main params to self.mixed_precision.param_dtype
            # for forward pass.
            if self._mixed_precision_enabled_for_params():
                self._streams["mixed_precision_params"] = torch.cuda.Stream()

        # We share streams with all children instances, which allows them to
        # overlap transfers across the forward pass without synchronizing with
        # the default stream.
        for m in self.modules():
            if m is not self and isinstance(m, FullyShardedDataParallel):
                m._streams = self._streams
                m._fsdp_graph_order = self._fsdp_graph_order
                # Give each non-root FSDP module an alias to the root's
                # execution order data structure and the root's ignored
                # parameters and all buffer names since only the root's names
                # are fully prefixed like the state dict keys
                m._exec_order_data = self._exec_order_data
                m._ignored_param_names = self._ignored_param_names
                m._buffer_names = self._buffer_names

    def _wait_for_previous_optim_step(self) -> None:
        """
        The outer-most :class:`FullyShardedDataParallel` instance (i.e., the root
        instance) needs to synchronize with the default stream to ensure the
        previous optimizer step is done.
        """
        if not torch.cuda.is_available():
            return

        if self._mixed_precision_enabled_for_params():
            self._streams["mixed_precision_params"].wait_stream(
                torch.cuda.current_stream()
            )

        self._streams["all_gather"].wait_stream(torch.cuda.current_stream())

    def _need_prefetch_pre_backward_hook(self) -> bool:
        if (
            self.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
            and self._fsdp_graph_order is not None
            and self._my_fsdp_idx_in_graph is not None
            and self._my_fsdp_idx_in_graph > 0
            and self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1].training_state
            != TrainingState_.BACKWARD_POST
        ):
            return True
        else:
            return False

    def _need_prefetch_post_backward_hook(self) -> bool:
        if (
            self.backward_prefetch == BackwardPrefetch.BACKWARD_POST
            and self._fsdp_graph_order is not None
            and self._my_fsdp_idx_in_graph is not None
            and self._my_fsdp_idx_in_graph > 0
            and self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1].training_state
            != TrainingState_.BACKWARD_POST
            and self._fsdp_graph_order[
                self._my_fsdp_idx_in_graph - 1
            ]._need_rebuild_full_params
        ):
            return True
        else:
            return False

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

        >>> model = DDP(FSDP(...))
        >>> fsdp_root = model.module
        >>> with FSDP.state_dict_type(fsdp_root, StateDictType.LOCAL_STATE_DICT):
        >>>     checkpoint = model.state_dict()

        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired state_dict_type to set.
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
        self._assert_state([TrainingState_.SUMMON_FULL_PARAMS])
        # state_dict is empty for nonzero ranks if `rank0_only` was enabled.
        if not state_dict:
            return state_dict

        offload_to_cpu = self._state_dict_config.offload_to_cpu
        cpu_device = torch.device("cpu")
        for key in state_dict:
            clean_key = clean_tensor_name(key)
            # Do not need to clone buffers since they are not sharded
            if clean_key in self._buffer_names:
                # Offload the buffer to CPU if needed -- we do not do this in
                # `_summon_full_params()` since without care, that would free
                # the original buffer's GPU memory and require reallocating
                # that memory later; this only affects the state dict's buffer
                # variable and leaves the original buffer's GPU memory intact
                if offload_to_cpu and state_dict[key].device != cpu_device:
                    state_dict[key] = state_dict[key].to(cpu_device)
                continue
            # Clone non-ignored parameters before exiting the
            # `_summon_full_params()` context
            if clean_key not in self._ignored_param_names and \
                    not getattr(state_dict[key], "_has_been_cloned", False):
                try:
                    state_dict[key] = state_dict[key].clone().detach()
                    state_dict[key]._has_been_cloned = True  # type: ignore[attr-defined]
                except BaseException as e:
                    warnings.warn(
                        f"Failed to clone() tensor with name {key}. This may mean "
                        "that this state_dict entry could point to invalid memory "
                        "regions after returning from state_dict() call if this "
                        "parameter is managed by FSDP. Please check clone "
                        f"implementation of {key}. Error: {str(e)}"
                    )

        _replace_by_prefix(state_dict, prefix + f"{FSDP_WRAPPED_MODULE}.", prefix)
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
        if self.module.no_params:
            return state_dict

        # state_dict[f"{prefix}{FLAT_PARAM}"] exists and has the same tensor
        # value as the flat_param but it is a pure Tensor because
        # nn.Module.state_dict() will detach the parameter. Therefore, we need
        # to get flat_param from the FlattenParamsWrapper to get the metadata.
        flat_param = getattr(self.module, FLAT_PARAM, None)
        # Construct a ShardedTensor from the flat_param.
        full_numel = flat_param.full_numel
        shard_offset = flat_param.numel() * self.rank
        valid_data_size = flat_param.numel() - flat_param.num_padded
        if valid_data_size > 0 and flat_param.num_padded > 0:
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
        if self.module.no_params:
            return state_dict

        for module_name, _, param_name in self.module.orig_flat_param[0].param_info:
            module_name = module_name.replace(f"{FPW_MODULE}.", "")
            module_name = module_name.replace(f"{FPW_MODULE}", "")
            if module_name:
                module_name = f"{module_name}."
            fqn = f"{prefix}{module_name}{param_name}"

            # Create a ShardedTensor for the unflattened, non-sharded parameter.
            param = state_dict[fqn]
            local_shard = param.chunk(self.world_size)[self.rank].clone()
            offsets = [0 for _ in param.size()]
            offsets[0] = math.ceil(param.size()[0] / self.world_size) * self.rank
            local_shards = [
                Shard.from_tensor_and_offsets(local_shard, offsets, self.rank)
            ]
            state_dict[fqn] = init_from_local_shards(
                local_shards, param.size(), process_group=self.process_group
            )  # type: ignore[assignment]
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
        The entry point of all three FSDP state_dict APIs. By default, calling
        ``state_dict`` on an FSDP module will result in FSDP attempting to bring
        the entire (nested) model into memory and taking the local model's
        ``state_dict`` on every rank, which could result in OOM if the model
        cannot fit on a single GPU. As a result, :func:`state_dict_type` API is
        available to configure between `state_dict` implementations. User can
        thus use `with self.state_dict_type(self, StateDictType.LOCAL_STATE_DICT)`
        context manager to perform a local checkpoint that will store only local
        shards of the module. Currently, the only supported implementations are
        ``StateDictType.LOCAL_STATE_DICT`` and ``StateDictType.FULL_STATE_DICT``
        (default).

        Example::

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

        elif self._state_dict_type == StateDictType.LOCAL_STATE_DICT:
            if (
                self.module.flat_param is not None and
                not self.module.flat_param._is_sharded
            ):
                raise RuntimeError(
                    "local_state_dict can only be called "
                    "when parameters are flatten and sharded."
                )
            return super().state_dict(*args, **kwargs)
        elif self._state_dict_type == StateDictType.SHARDED_STATE_DICT:
            summon_ctx = (
                self._summon_full_params(recurse=False, writeback=False)
                if self.training_state != TrainingState_.SUMMON_FULL_PARAMS else
                contextlib.suppress()
            )
            with summon_ctx:
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
        _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_WRAPPED_MODULE}.")

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
            assert getattr(self.module, FLAT_PARAM, None) is None, (
                "No flat parameter in state_dict but self.module.flat_param is not None"
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
        flat_param = self.module.flat_param
        assert flat_param is not None
        if flat_param.num_padded not in (0, flat_param.numel()):
            assert load_tensor.numel() < flat_param.numel(), (
                f"Local shard size = {flat_param.numel()} and the tensor in "
                f"the state_dict is {load_tensor.numel()}."
            )
            load_tensor = F.pad(load_tensor, [0, flat_param.num_padded])
        state_dict[fqn] = load_tensor

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
        if self.module.no_params:
            return

        if not self.module.flat_param._is_sharded:
            raise RuntimeError(
                "load_sharded_state_dict can only be called when parameters "
                "are flatten and sharded."
            )

        nonsharded_tensors = []
        # TODO: Reduce the communication by using only one _all_gather_base to
        # gather all the parameters in this layer. This can be achieved by
        # concatenated all the local shards and then append the padding.
        # https://github.com/pytorch/pytorch/issues/77461
        for module_name, _, param_name in self.module.flat_param._param_infos:
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
        flat_param = self.module.flat_param
        loaded_flat_param = FlatParameter(nonsharded_tensors, requires_grad=False)

        # Get the chunk from the loaded flat_param for the local rank.
        loaded_flat_param, num_to_pad = self._get_shard(loaded_flat_param)
        assert flat_param.numel() == loaded_flat_param.numel(), (
            f"The loaded local chunk has different numel({flat_param.numel()}) "
            f"from the local chunk {flat_param.numel()}."
        )
        assert flat_param.num_padded == num_to_pad, (
            f"The loaded local chunk has different padding({num_to_pad}) "
            f"from the local chunk {flat_param.num_padded}."
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

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
    ) -> NamedTuple:
        """
        The entry point of all three FSDP load_state_dict APIs. By default,
        calling ``load_state_dict`` on an FSDP module will result in FSDP
        attempting to load a "full" state_dict, i.e. a state_dict consisting of
        full, unsharded, unflattened original module parameters. This requires
        FSDP to load the full parameter context on each rank which could result
        in GPU OOM. As a result, :func:`state_dict_type` API is available to
        configure between `load_state_dict` implementations. User can thus use
        ``with self.state_dict_type(self, StateDictType.LOCAL_STATE_DICT)`` context
        manager to load a local state dict checkpoint that will restore only
        local shards of the module. Currently, the only supported
        implementations are ``StateDictType.LOCAL_STATE_DICT`` and
        ``StateDictType.FULL_STATE_DICT`` (default). Please see :func:`state_dict`
        for documentation around creating an FSDP checkpoint.

        Example::

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
        >>> local_state_dict = checkpoint['local_state_dict]
        >>> with FSDP.state_dict_type(sharded_module, StateDictType.LOCAL_STATE_DICT):
        >>>     sharded_module.load_state_dict(local_state_dict)
        >>> local_dict.keys()
        >>> odict_keys(['flat_param', 'inner.flat_param'])

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives may be used.
        """
        if self._state_dict_type == StateDictType.FULL_STATE_DICT:
            # Note that it needs writeback=True to persist
            with self._summon_full_params(writeback=True):
                return super().load_state_dict(state_dict, *args)
        elif self._state_dict_type == StateDictType.LOCAL_STATE_DICT:
            return super().load_state_dict(state_dict, *args)
        elif self._state_dict_type == StateDictType.SHARDED_STATE_DICT:
            return super().load_state_dict(state_dict, *args)
        else:
            raise ValueError(f"Unknown StateDictType {self._state_dict_type}.")

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

            # Start of a forward pass.
            self.training_state = TrainingState_.FORWARD

            # Cast inputs to their mixed precision type.
            if (
                self._is_root
                and self._mixed_precision_enabled_for_params()
            ):
                input_dtype = self.mixed_precision.param_dtype
                args, kwargs = self._cast_fp_inputs_to_precision(
                    input_dtype, *args, **kwargs
                )

            # All-gather full parameters, moving them to compute_device if
            # necessary.
            self._rebuild_full_params()
            # Wait for all_gather full parameters to finish before computation
            torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

            # Register backward hooks to reshard params and reduce-scatter grads.
            # These need to be re-registered every forward pass in some cases where grad_fn
            # is mutated.
            self._register_post_backward_hooks()
            outputs = self.module(*args, **kwargs)

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
                    stack.enter_context(self.module.unflatten_params())
                    try:
                        yield
                    finally:
                        if offload_to_cpu:
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
            the parameters, currently only when ``world_size == 1``, the
            modification is persisted regardless of ``writeback``.
        .. note:: This method works on modules which are not FSDP themselves but
            may contain multiple independent FSDP units. In that case, the given
            arguments will apply to all contained FSDP units.

        .. warning:: Note that ``rank0_only=True`` in conjunction with
            ``writeback=True`` is not currently supported and will raise an
            error. This is because model parameter shapes would be different
            across ranks within the context, and writing to them can lead to
            inconsistency across ranks when the context is exited.

        ..warning:: Note that ``offload_to_cpu`` and ``rank0_only=False`` will
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
            offload_to_cpu (bool, optional): If ``True``, full parameters are
                offloaded to CPU. Note that this offloading currently only
                occurs if the parameter is sharded (which is only not the case
                for world_size = 1). It is recommended to use ``offload_to_cpu``
                with ``rank0_only=True`` to avoid redundant copies of model
                parameters being offloaded to the same CPU memory.
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
        when inside the :meth:`_summon_full_params` context manager.
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
        when inside the :meth:`_summon_full_params` context manager.
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
            # try to queue final backward callback only once for root, so
            # that final backward callback is attached to the outer most
            # backward graph task and called after all the backward
            # calls are completed.
            if self._is_root:
                self._queue_wait_for_post_backward()

            if self._need_prefetch_pre_backward_hook():
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
            # Wait for all_gather to finish before computation
            torch.cuda.current_stream().wait_stream(self._streams["all_gather"])

            # Prefetch next layer's full params in backward pass,
            # since it is prefetching, no need to wait for all_gather stream.
            if self._need_prefetch_pre_backward_hook():
                self._fsdp_graph_order[self._my_fsdp_idx_in_graph - 1]._rebuild_full_params()  # type: ignore[operator]

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
        # First hook callback will see PRE state. If we have multiple params,
        # then subsequent hook callbacks will see POST state.
        self._assert_state([TrainingState_.BACKWARD_PRE, TrainingState_.BACKWARD_POST])
        self.training_state = TrainingState_.BACKWARD_POST
        if param.grad is None:
            return

        if param.grad.requires_grad:
            raise RuntimeError(
                "FSDP only works with gradients that don't require gradients"
            )

        if self._require_backward_grad_sync or \
                self.sharding_strategy == ShardingStrategy.FULL_SHARD:
            # We free full parameters unless we are in `no_sync()` (i.e. when
            # `_require_backward_grad_sync=False`) and not using the
            # `FULL_SHARD` strategy. If we are not using the `FULL_SHARD`
            # strategy (e.g. instead using `SHARD_GRAD_OP`), then we keep the
            # full parameters in memory and save network overhead.
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
        if self._need_prefetch_post_backward_hook():
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
                self._mixed_precision_enabled_for_reduce()
            ):
                # Cast gradient to precision in which it should be communicated.
                # TODO: Make this a communication hook when communication hooks
                # are implemented for FSDP. Note that this is a noop if the
                # reduce_dtype matches the param dtype.
                param.grad.data = param.grad.data.to(self.mixed_precision.reduce_dtype)

            if self.gradient_predivide_factor > 1:
                # Average grad by world_size for consistency with PyTorch DDP.
                param.grad.div_(self.gradient_predivide_factor)

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
                dist._reduce_scatter_base(
                    output, input_flattened, group=self.process_group
                )
                if self.gradient_postdivide_factor > 1:
                    # Average grad by world_size for consistency with PyTorch DDP.
                    output.div_(self.gradient_postdivide_factor)

                # Note that we need to cast grads back to the full precision if
                # 1) parameters were in reduced precision during fwd, as grads
                # would thus be in this reduced precision, or
                # 2) parameters did not have precision reduced, but grads
                # had reduced precision for communication.
                if (
                    self._mixed_precision_enabled_for_params() or self._mixed_precision_enabled_for_reduce()
                ):
                    # Cast gradients back to the full parameter precision so that
                    # optimizer.step() happens in full precision.
                    orig_param_grad_data = output
                    output.data = output.data.to(dtype=param.data.dtype)
                    # Don't let this memory get reused until after the transfer.
                    orig_param_grad_data.record_stream(torch.cuda.current_stream())

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
                    dist.all_reduce(param.grad, group=self.process_group)
                    if self.gradient_postdivide_factor > 1:
                        # Average grad by world_size for consistency with PyTorch DDP.
                        param.grad.div_(self.gradient_postdivide_factor)
                # Note that we need to cast grads back to the full precision if
                # 1) parameters were in reduced precision during fwd, as grads
                # would thus be in this reduced precision, or
                # 2) parameters did not have precision reduced, but grads
                # had reduced precision for communication.
                if (
                    self._mixed_precision_enabled_for_params() or self._mixed_precision_enabled_for_reduce()
                ):
                    # Cast gradients back to the full parameter precision so that
                    # optimizer.step() happens in full precision.
                    orig_param_grad_data = param.grad.data
                    param.grad.data = param.grad.data.to(dtype=param.data.dtype)
                    # Don't let this memory get reused until after the transfer.
                    orig_param_grad_data.record_stream(torch.cuda.current_stream())

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
                            not p._is_sharded, "All sharded parameters should "
                            "use `_saved_grad_shard`"
                        )
                    if hasattr(p, "_saved_grad_shard"):
                        delattr(p, "_saved_grad_shard")

        # Update root and nested FSDP's hooks and flags.
        for m in self.modules():  # includes self
            if isinstance(m, FullyShardedDataParallel):
                _finalize_params(m)
                m._pre_backward_hook_has_run = False
                if any(p.requires_grad for p in m.parameters()):
                    # Check if the module has params and if any of them has
                    # the `requires_grad` field set. If `requires_grad=False` for
                    # all the params, the post_backward hook will not fire and the
                    # state will remain in `TrainingState_.BACKWARD_PRE`.
                    if any([p.requires_grad for p in m.params]):
                        m._assert_state(TrainingState_.BACKWARD_POST)
                    else:
                        m._assert_state(TrainingState_.BACKWARD_PRE)
                else:
                    # When `m` and its children have no non-ignored params or
                    # have non-ignored params but none with `requires_grad==True`,
                    # there are two cases:
                    # 1. output tensors are `requires_grad==True`. In this case,
                    # pre-backward hook is still registered, so it is in BACKWARD_PRE state.
                    # 2. output tensors are `requires_grad==False`. In this case,
                    # pre-backward hook is not registered, so it is in IDLE state.
                    m._assert_state([TrainingState_.BACKWARD_PRE, TrainingState_.IDLE])
                m.training_state = TrainingState_.IDLE

                if m._is_root:
                    # reset this flag for cases like "one forward pass + multiple backward passes"
                    self._post_backward_callback_queued = False

    def _update_p_data(self, p, output_tensor: torch.Tensor) -> None:
        """
        Helper function to update p.data pointer.
        Args:
            output_tensor (torch.Tensor): this tensor contains the data we just gathered.
        """
        p.data = output_tensor
        # Trim any padding and reshape to match original size.
        p.data = p.data[: p._orig_size.numel()].view(p._orig_size)  # type: ignore[attr-defined]

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
                print(first_iter_param_names, type(first_iter_param_names))
                print(curr_iter_param_names, type(curr_iter_param_names))
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
                p.grad.size() != p._orig_size  # type: ignore[attr-defined]
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

        .. note:: This is analogous to `torch.nn.utils.clip_grad_norm_` but
            handles the partitioning and multiple devices per rank under the
            hood. The default torch util is not applicable here, because each
            rank only has a partial view of all the grads in the model, so
            calling it for FSDP models would lead to different scaling being
            applied per subset of model parameters.

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives will be used.
        """
        # Call `_lazy_init` to ensure the stream synchronization is done appropriately.
        self._lazy_init()
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

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            ``model`` 's original unflattened parameters and including keys
            "state" and "param_groups" following the convention of
            :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=False``,
            then nonzero ranks return an empty :class:`dict`.
        """
        osd = optim.state_dict()
        osd_state, osd_param_groups = osd["state"], osd["param_groups"]  # alias

        group = model.process_group if hasattr(model, "process_group") \
            else None  # not all `torch.nn.Module`s have `process_group`
        rank = dist.get_rank(group)
        to_save = not rank0_only or rank == 0
        full_osd: Dict = {"state": {}, "param_groups": []} if to_save else {}
        full_osd_state = full_osd["state"] if to_save else None  # alias

        # Handle the "state" part of the optimizer state dict
        param_to_unflat_param_names = _get_param_to_unflat_param_names(model)
        flat_param_id_to_param = _get_param_id_to_param(model, optim_input)
        flat_param_to_fsdp_module = _get_flat_param_to_fsdp_module(model)
        for flat_param_id, param in enumerate(flat_param_id_to_param):  # type: ignore[assignment]
            # Do not include parameters without state to avoid empty mappings
            if flat_param_id not in osd_state:
                continue
            assert param in param_to_unflat_param_names, \
                "Check the `param_to_unflat_params` construction\n" \
                f"param: {param}"
            unflat_param_names = param_to_unflat_param_names[param]
            # For FSDP parameters, we need to unflatten
            if isinstance(param, FlatParameter):
                assert param in flat_param_to_fsdp_module, \
                    "Check the `flat_param_to_fsdp_module` construction\n" \
                    f"param: {param}"
                unflat_state = _unflatten_optim_state(
                    flat_param_to_fsdp_module[param], param,
                    osd_state[flat_param_id], to_save,
                )
                if to_save:
                    assert len(unflat_state) == len(unflat_param_names) and \
                        len(unflat_state) == param._num_unflattened_params, \
                        f"{len(unflat_state)} {len(unflat_param_names)} " \
                        f"{param._num_unflattened_params}"
                    for unflat_param_name, unflat_param_state in zip(
                        unflat_param_names, unflat_state,
                    ):
                        full_osd_state[unflat_param_name] = unflat_param_state
            # For parameters from non-FSDP modules, we do not need to unflatten
            elif to_save:
                assert len(unflat_param_names) == 1
                unflat_param_name = unflat_param_names[0]
                # Do not `deepcopy()` to avoid unnecessarily duplicating
                # tensor storage
                full_osd_state[unflat_param_name] = \
                    copy.copy(osd_state[flat_param_id])
                # Move all tensor state to CPU
                param_state = full_osd_state[unflat_param_name]
                for state_name, value in param_state.items():
                    if torch.is_tensor(value):
                        param_state[state_name] = value.cpu()

        # Non-target ranks may return since there is no more communication
        if not to_save:
            return full_osd

        # Handle the "param_groups" part of the optimizer state dict
        full_osd_param_groups = full_osd["param_groups"]  # alias
        for flat_param_group in osd_param_groups:
            unflat_param_group = copy.deepcopy(flat_param_group)
            param_group_params = [
                flat_param_id_to_param[flat_param_id]
                for flat_param_id in flat_param_group["params"]
            ]
            nested_unflat_param_names = [
                param_to_unflat_param_names[param]
                for param in param_group_params
            ]
            unflat_param_group["params"] = [
                unflat_param_name
                for unflat_param_names in nested_unflat_param_names
                for unflat_param_name in unflat_param_names
            ]  # flatten the list of lists
            full_osd_param_groups.append(unflat_param_group)
        return full_osd

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
        return _flatten_full_optim_state_dict(
            full_optim_state_dict, model, True, optim_input,
        )[0]

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
                ``model.parameters()``; the argument is ignored on nonzero
                ranks. (Default: ``None``)
            group (Optional[Any]): Model's process group or ``None`` if using
                the default process group. (Default: ``None``)

        Returns:
            Dict[str, Any]: The full optimizer state dict now remapped to
            flattened parameters instead of unflattened parameters and
            restricted to only include this rank's part of the optimizer state.
        """
        # Try to use the passed-in process group, the model's process group,
        # or the default process group (i.e. ``None``) in that priority order
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
            flat_osd, fsdp_flat_param_ids = _flatten_full_optim_state_dict(
                full_optim_state_dict, model, False, optim_input,
            )
            processed_osd = _process_pos_dim_tensor_state(
                flat_osd, fsdp_flat_param_ids, world_size,
            )
        # Broadcast the optim state dict without positive-dimension tensor
        # state and the FSDP parameter IDs from rank 0 to all ranks
        processed_osd, fsdp_flat_param_ids = \
            _broadcast_processed_optim_state_dict(
                processed_osd if rank == 0 else None,
                fsdp_flat_param_ids if rank == 0 else None, rank, group,
                broadcast_device,
            )
        # Broadcast positive-dimension tensor state (both sharded tensors for
        # FSDP parameters and unsharded tensors for non-FSDP parameters)
        sharded_osd = _broadcast_pos_dim_tensor_states(
            processed_osd, fsdp_flat_param_ids,
            flat_osd if rank == 0 else None, rank, world_size, group,
            broadcast_device,
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
    ) -> Dict[str, Any]:
        """
        Re-keys the optimizer state dict ``optim_state_dict`` to use the key
        type ``optim_state_key_type``. This can be used to achieve
        compatibility between optimizer state dicts from models with FSDP
        instances and ones without.

        To re-key an FSDP full optimizer state dict (i.e. from
        :meth:`full_optim_state_dict`) to use parameter IDs and be loadable to
        a non-wrapped model::

            >>> wrapped_model, wrapped_optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(wrapped_model, wrapped_optim)
            >>> nonwrapped_model, nonwrapped_optim = ...
            >>> rekeyed_osd = FSDP.rekey_optim_state_dict(full_osd, OptimStateKeyType.PARAM_ID, nonwrapped_model)
            >>> nonwrapped_optim.load_state_dict(rekeyed_osd)

        To re-key a normal optimizer state dict from a non-wrapped model to be
        loadable to a wrapped model::

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
    return torch.device("cuda")


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
        local_norm = torch.linalg.norm(
            torch.stack(
                [
                    torch.linalg.norm(par.grad.detach(), p, dtype=torch.float32)
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
    def _clean_param_name(prefix, param_info):
        """This replicates the parameter name cleaning logic in model state
        dict but avoids gathering any parameters."""
        name = clean_tensor_name(
            prefix + param_info.module_name + "." + param_info.param_name
        )
        return name

    def module_fn(module, prefix, param_to_unflat_param_names):
        # For FSDP modules, only add the entry when considering the contained
        # `FlattenParamsWrapper` to avoid duplication
        if not isinstance(module, FullyShardedDataParallel):
            for param_name, param in module.named_parameters(recurse=False):
                prefixed_param_names = [
                    _clean_param_name(prefix, param_info)
                    for param_info in param._param_infos
                ] if isinstance(param, FlatParameter) else [prefix + param_name]
                # If this parameter has already been visited, then it is a
                # shared parameter; then, only take the first parameter name
                is_shared_param = param in param_to_unflat_param_names
                if not is_shared_param:
                    param_to_unflat_param_names[param] = prefixed_param_names
                elif not dedup_shared_params:
                    param_to_unflat_param_names[param].extend(prefixed_param_names)

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
    return tensor_name
