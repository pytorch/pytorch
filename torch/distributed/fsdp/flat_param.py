import contextlib
import functools
import logging
import os
import warnings
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    Iterator,
    List,
    NamedTuple,
    no_type_check,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp._common_utils import (
    _FSDPDeviceHandle,
    _named_parameters_with_duplicates,
    _set_fsdp_flattened,
    HandleTrainingState,
)
from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]

from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
from ._utils import _no_dispatch_record_stream, _same_storage_as_data_ptr

__all__ = [
    "FlatParameter",
    "FlatParamHandle",
    "FlatParamShardMetadata",
    "ParamInfo",
    "SharedParamInfo",
    "HandleShardingStrategy",
]

log = logging.getLogger(__name__)


"""
[Note: Fully Sharded Module]
We define the "fully sharded module" to be the original ``nn.Module`` that owns
a ``FlatParamHandle``. It is the *single* module logically responsible for the
*single* unshard/reshard pair for the handle's ``FlatParameter`` for a given
forward or backward pass. The fully sharded module should be passed to the
``FlatParamHandle`` constructor.

For the wrapper code path:
- The ``FullyShardedDataParallel`` module wrapping the fully sharded module
runs the unshard/reshard on behalf of the fully sharded module by overriding
``nn.Module.forward``.
- The fully sharded module is exactly the module passed to the
``FullyShardedDataParallel`` constructor's ``module`` argument.

For the non-wrapper code path:
- Hooks registered on the fully sharded module run the unshard/reshard.
- The fully sharded module may either be the direct argument to ``fully_shard``
or a submodule chosen by the provided wrapping policy.
"""

# Environment variable toggling whether to use unsafe `setattr()` for view
# setting in `_use_sharded_views()` and `_use_unsharded_views()`
# We should use 'safe' by default since it respects method overrides, but for
# special cases such as for high CPU overhead or for intentionally bypassing
# checks in the overrides, we may use 'unsafe'.
_FSDP_USE_UNSAFE_SETATTR = "FSDP_USE_UNSAFE_SETATTR"
# Environment variable toggling whether to check for parameter/gradient
# writeback in case their storages change after FSDP initialization
# We should check by default since it prevents silent correctness errors, but
# since such changes are atypical, we may want to skip the check to save CPU
# overhead, especially since the check happens in the pre-forward and
# pre-backward each iteration.
_FSDP_SKIP_WRITEBACK_CHECK = "FSDP_SKIP_WRITEBACK_CHECK"

# Env var toggling whether when model is in .eval() mode, should we run in fp32
# or the reduced precision.
_FSDP_USE_FULL_PREC_IN_EVAL = "FSDP_USE_FULL_PREC_IN_EVAL"

# Some value to set padding in tensors to for debuggability
_FLAT_PARAM_PADDING_VALUE = 42


# TODO: Define this for now to avoid circular imports. See if we can remove.
class HandleShardingStrategy(Enum):
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()
    _HYBRID_SHARD_ZERO2 = auto()


RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.FULL_SHARD,
    HandleShardingStrategy.HYBRID_SHARD,
)
NO_RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.SHARD_GRAD_OP,
    HandleShardingStrategy._HYBRID_SHARD_ZERO2,
)


class ParamInfo(NamedTuple):
    """Information for an original parameter."""

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str


class SharedParamInfo(NamedTuple):
    """
    Additional information for a shared parameter.

    For each shared parameter, we designate one module and its parameter
    variable to be the primary owner, determined as the first one encountered
    in the parameter walk. These are prefixed with "prim". The primary module
    and parameter do not have their own :class:`SharedParamInfo` instance.
    """

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str
    prim_param_name: str  # unprefixed
    prim_module: nn.Module
    prim_module_name: str


class _ShardParamInfo(NamedTuple):
    """Shard-related information for an original parameter."""

    in_shard: bool
    # Use to index into the sharded flat parameter, e.g.
    # `flat_param[offset_in_shard : offset_in_shard + numel_in_shard]`
    offset_in_shard: Optional[int]
    numel_in_shard: Optional[int]
    # Use to get part of the parameter in the local shard from a flattened
    # version of the unsharded parameter, e.g.
    # `param.flatten()[intra_param_start_idx : intra_param_end_idx + 1]`
    intra_param_start_idx: Optional[int]
    intra_param_end_idx: Optional[int]  # inclusive


class FlatParamShardMetadata(NamedTuple):
    """
    This holds metadata specific to this rank's shard of the flat parameter.

    Attributes:
        param_names (Tuple[str, ...]): Prefixed parameter names of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_shapes (Tuple[torch.Size, ...]): Parameter shapes of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_numels (Tuple[int, ...]): Parameter numels of this rank's shard
            of the parameters; see :class:`FlatParameter`.
        param_offsets (Tuple[Tuple[int, int], ...]): [start, end] offsets (in
            units of numels) giving this rank's part of each flattened
            original parameter.
    """

    param_names: Tuple[str, ...]
    param_shapes: Tuple[torch.Size, ...]
    param_numels: Tuple[int, ...]
    param_offsets: Tuple[Tuple[int, int], ...]


class _FlatParameterMeta(_ParameterMeta):
    # Make `isinstance(t, FlatParameter)` return True for custom tensor
    # instances that have the _is_flat_param flag for BC
    def __instancecheck__(self, instance):
        # NB: do NOT test the super implementation
        return isinstance(instance, torch.Tensor) and getattr(
            instance, "_is_flat_param", False
        )


class FlatParameter(nn.Parameter, metaclass=_FlatParameterMeta):
    """
    This is the flat parameter used by :class:`FullyShardedDataParallel`. It is
    comprised of one or more original parameters, which are flattened and
    concatenated to construct the flat parameter.

    Under the current design, this parameter logically represents both the
    unsharded and sharded flat parameter, and its data changes storages
    dynamically.
        - In the :class:`FullyShardedDataParallel` constructor, the parameter
        is initialized as unsharded and then sharded in-place.
        - At runtime, the parameter is lazily (re)-initialized. The sharded
        parameter data is saved in ``self._local_shard``, and a new ``Tensor``
        ``self._full_param_padded`` is created, which is the all-gather
        destination and owns the unsharded parameter storage thereafter. (See
        :meth:`FlatParamHandle.init_flat_param_attributes`.)
        - Throughout runtime, the parameter data changes storages as needed,
        e.g. to the sharded flat parameter, low precision sharded flat
        parameter, or the unsharded flat parameter.

    NOTE: Since ``use_orig_params=True`` supports intra-``FlatParameter``
    padding, we have two versions of the per-parameter numels, one that
    includes the padding (``_numels_with_padding``) and one that does not
    (``_numels``). The former may have length longer than the other data
    structures, while the latter has the same length as the number of actual
    original parameters like the other per-parameter data structures.

    NOTE: This is not a real class; instead, you will always get a Parameter
    back out if you try to create one of these.  This is similar to the trick
    we implemented for Parameter to get it to work with subclasses; this
    is primarily so that FlatParameter supports combination with FakeTensor.

    Attributes:
        _unpadded_unsharded_size (torch.Size): Unsharded flat parameter's size
            without right-hand-side padding for divisibility by the world size.
            For ``use_orig_params=True``, this includes alignment padding.
        _padded_unsharded_size (torch.Size): Unsharded flat parameter's size
            with right-hand-side padding for divisibility by the world size.
            For ``use_orig_params=True``, this includes alignment padding. This
            is only set for sharded strategies since they require padding for
            the all-gather.
        _sharded_size (torch.Size): Sharded flat parameter's size with padding.
            This is also set for ``NO_SHARD``, in which case it is the same as
            the unsharded sizes. (We omit "padded" because there is no
            analogous unpadded one.)

        _num_params (int): Number of original parameters flattened into this
            flat parameter. This is the length of the per-parameter data
            structures.
        _param_infos (Tuple[ParamInfo, ...]): Each parameter's parameter info
            entry; see :class:`ParamInfo` for details.
        _shapes (Tuple[torch.Size, ...]): Each parameter's original shape.
        _fqns (Tuple[str, ...]): Each parameter's fully-qualified name (FQN)
            prefixed from the ``_fully_sharded_module``. The names are
            guaranteed to be unique in the subtree rooted at that module.
        _param_extensions (Tuple[Optional[Any], ...]): Each parameter's
            extension (i.e. some per-parameter state) used to customize
            pre-flatten and post-unflatten behavior or ``None``. This is
            experimental, and users should not depend on its existence in the
            future.
        _numels_with_padding (Tuple[int, ...]): Each parameter's numel
            including entries for the padding. This is used to construct views
            into the flat parameter via ``torch.split()``. This may have length
            longer than ``_num_params``.
        _numels (Tuple[int, ...]): Each parameter's numel excluding entries for
            padding. This has length equal to ``_num_params``.
        _shard_param_infos (Tuple[_ShardParamInfo, ...]): Each parameter's
            shard parameter info; see :class:`_ShardParamInfo` for details.
        _shared_param_infos (Tuple[SharedParamInfo, ...]): Shared parameter
            info entries; see :class:`SharedParamInfo` for details.
        _modules (Set[nn.Module]): Modules that contain some original parameter
            that is flattened into the flat parameter.

        _shard_numel_padded (int): Numel padded for this rank's sharded flat
            parameter.
        _local_shard (Tensor): Sharded flat parameter with padding if using a
            sharded strategy. If using ``NO_SHARD``, then this is the unpadded
            unsharded flat parameter, and there is no notion of a sharded flat
            parameter or padded unsharded flat parameter.
        _full_param_padded (Tensor): Unsharded flat parameter with padding.
            This is not defined for ``NO_SHARD``. When using mixed precision
            for parameters, this has the low precision.
        _full_prec_full_param_padded (Tensor): Full precision unsharded flat
            parameter with padding. This is used for unsharding outside of
            computation when using mixed precision for parameters. This is
            never defined for ``NO_SHARD``.
        _post_backward_hook_state (Tuple[AccumulateGrad, RemovableHandle]):
            Flat parameter's :class:`AccumulateGrad` object and post-backward
            hook handle.
        _mp_shard (Tensor): Low precision sharded flat parameter with padding.
            This is only defined when parameter mixed precision is enabled. For
            ``NO_SHARD``, this is used for computation.
        _cpu_grad (Tensor): Sharded gradient with padding stored on CPU.
            This is only defined when offloading parameters is enabled.
        _saved_grad_shard (Tensor): Sharded gradient with padding from previous
            iterations for gradient accumulation without :meth:`no_sync`.

        _params (Optional[List[nn.Parameter]]): If ``use_orig_params=True``,
            then each original parameter variable; otherwise, ``None``. This
            does not include any padding tensors.
        _shared_params (Optional[List[nn.Parameter]]): The original shared
            parameter variables if ``use_orig_params=True`` and ``None``
            otherwise.
        _tensors (Optional[List[Optional[Tensor]]]): This saves the ``Tensor``
            views created in the forward and tracked by autograd when
            ``use_orig_params=True`` and is ``None`` otherwise. This is to
            preserve those ``Tensor`` variables for the backward to ensure that
            the ``FlatParameter`` 's ``AccumulateGrad`` object does not change
            in which case the post-backward hook does not run. This is relevant
            for cases like reentrant activation checkpointing.
        _is_grad_none_mask (Optional[List[bool]]): If ``use_orig_params=True``,
            a mask over the original parameters' gradients indicating if it is
            logically ``None`` or not; otherwise, ``None``. This does not
            include entries for padding. This mask is needed because only some
            of the parameters may have ``None`` gradient, in which case the
            flat gradient must be non-``None`` and must use zeros to
            approximate those original ``None`` gradients. This mask informs
            FSDP to set the original parameter gradients to ``None`` (instead
            of zeros) as needed.
    """

    _unpadded_unsharded_size: torch.Size
    _padded_unsharded_size: torch.Size
    _sharded_size: torch.Size
    _num_params: int
    _param_infos: Tuple[ParamInfo, ...]
    _shapes: Tuple[torch.Size, ...]
    _fqns: Tuple[str, ...]
    _param_extensions: Tuple[Optional[Any], ...]
    _numels_with_padding: Tuple[int, ...]
    _numels: Tuple[int, ...]
    _shard_param_infos: Tuple[_ShardParamInfo, ...]
    _shared_param_infos: Tuple[SharedParamInfo, ...]
    _modules: Set[nn.Module]
    _shard_numel_padded: int
    _local_shard: Tensor
    _full_param_padded: Tensor
    _full_prec_full_param_padded: Tensor
    _post_backward_hook_state: Tuple[Any, Any]
    _mp_shard: Tensor
    _cpu_grad: Tensor
    _saved_grad_shard: Tensor
    _params: Optional[List[nn.Parameter]]
    _shared_params: Optional[List[nn.Parameter]]
    _tensors: Optional[List[Optional[Tensor]]]
    _is_grad_none_mask: Optional[List[bool]]

    _is_padding_mask: List[bool]

    def __new__(cls, data=None, requires_grad=True):
        assert cls is FlatParameter, "subclasses FlatParameter not supported"
        r = nn.Parameter.__new__(nn.Parameter, data, requires_grad)  # type: ignore[call-arg]
        r._is_flat_param = True  # type: ignore[attr-defined]
        return r

    # NB: This is not a regular method, because FlatParameter are not actually
    # instances of this class (see __new__ above).  So you must indirectly
    # call this directly through the classmethod.
    @classmethod
    def _init_metadata(
        cls,
        self,
        param_infos: List[ParamInfo],
        numels: List[int],
        shapes: List[torch.Size],
        fqns: List[str],
        shared_param_infos: List[SharedParamInfo],
        param_extensions: List[Optional[Any]],
        params: Optional[List[nn.Parameter]],
        shared_params: Optional[List[nn.Parameter]],
        is_padding_mask: List[bool],
    ) -> None:
        """
        Initializes attributes holding metadata about the original parameters
        comprising the flat parameter.

        We expose this method separate from the constructor to keep the
        constructor only responsible for the flat parameter's tensor data. This
        method should only be called once per model, while the constructor may
        be called multiple times, e.g. when reloading from a checkpoint, in
        which case only the tensor data needs to be passed to the constructor.
        Since :meth:`load_state_dict` is implemented via :meth:`copy_`, the
        metadata is correctly assumed to be unchanged.

        Args:
            See the Attributes in the class docstring.
        """
        assert len(param_infos) == len(shapes)
        assert len(param_infos) == len(fqns)
        assert len(param_infos) == len(param_extensions)
        self._num_params = len(param_infos)
        self._param_infos = param_infos
        self._shapes = shapes
        self._fqns = fqns
        self._param_extensions = param_extensions
        self._is_padding_mask = is_padding_mask

        numels_without_padding: List[int] = []
        for numel, is_padding in zip(numels, is_padding_mask):
            if not is_padding:
                numels_without_padding.append(numel)
        self._numels = tuple(numels_without_padding)
        self._numels_with_padding = tuple(numels)
        assert len(self._numels) == self._num_params

        self._shared_param_infos = tuple(shared_param_infos)
        self._modules = {pi.module for pi in self._param_infos}.union(
            {spi.module for spi in self._shared_param_infos}
        )
        assert (params is None) == (shared_params is None)
        if params is not None:
            assert shared_params is not None and len(shared_params) == len(
                shared_param_infos
            )
            self._params = []
            for param, is_padding in zip(params, is_padding_mask):
                if not is_padding:
                    self._params.append(param)
            self._shared_params = shared_params
            # Mark the original parameters to avoid flattening them into
            # another `FlatParameter` during recursive construction
            for param in chain(self._params, self._shared_params):
                _set_fsdp_flattened(param)
            self._is_grad_none_mask = [False for _ in range(self._num_params)]
            self._tensors = [None for _ in range(self._num_params)]
        else:
            self._params = None
            self._shared_params = None
            self._is_grad_none_mask = None
            self._tensors = None
        self._unpadded_unsharded_size = self.size()
        _set_fsdp_flattened(self)
        # Tracks whether the `FlatParameter`'s post-backward hook has been
        # called to modify the behavior of the post-backward callback
        self._post_backward_called = False


class FlatParamHandle:
    """
    This handle manages a flat parameter (:class:`FlatParameter`). This
    includes sharding and view management.

    Args:
        params (Sequence[nn.Parameter]): The parameters to flatten into the
            flat parameter.
        fully_sharded_module (nn.Module): See [Note: Fully Sharded Module].
        device (torch.device): The compute and communication device, which
            should be a non-CPU device. We refer to it as the compute device.
        sharding_strategy (ShardingStrategy): Sharding strategy to apply to
            this handle's ``FlatParameter``.
        offload_params (bool): Whether to offload the handle's
            ``FlatParameter`` to CPU.
        mp_param_dtype (Optional[torch.dtype]): Parameter mixed precision
            setting passed to the FSDP constructor.
        mp_reduce_dtype (Optional[torch.dtype]): Gradient reduction mixed
            precision setting passed to the FSDP constructor.
        keep_low_precision_grads (bool): Whether to keep gradients in low
            precision.
        use_orig_params (bool): If ``True``, then FSDP preserves the original
            parameter variables and returns them from ``named_parameters()``
            (e.g. to support different optimizer hyperparameters within one
            :class:`FlatParameter`). If ``False``, then FSDP reconstructs the
            parameters every iteration and returns the :class:`FlatParameter` s
            from ``named_parameters()``.
    """

    ##################
    # INITIALIZATION #
    ##################
    def __init__(
        self,
        params: Sequence[Union[nn.Parameter, Tensor]],
        fully_sharded_module: nn.Module,
        device: torch.device,
        sharding_strategy: HandleShardingStrategy,
        offload_params: bool,
        mp_param_dtype: Optional[torch.dtype],
        mp_reduce_dtype: Optional[torch.dtype],
        keep_low_precision_grads: bool,
        process_group: dist.ProcessGroup,
        use_orig_params: bool,
    ):
        super().__init__()
        params = list(params)
        if len(params) == 0:
            raise ValueError(
                f"Cannot construct a {self.__class__.__name__} with an empty parameter list"
            )
        self._init_setattr_fns()
        self._skip_writeback_check = (
            os.environ.get(_FSDP_SKIP_WRITEBACK_CHECK, "") == "1"
        )
        self._use_full_prec_in_eval = (
            os.environ.get(_FSDP_USE_FULL_PREC_IN_EVAL, "") == "1"
        )
        if self._skip_writeback_check:
            _warn_skip_writeback_check(
                log,
                f"Since {_FSDP_SKIP_WRITEBACK_CHECK}=1, FSDP will not check "
                "for parameter or gradient writeback. Changing parameter or "
                "gradient storages may lead to silent correctness errors.",
            )
        # Only align addresses for `use_orig_params=True` (for now)
        align_addresses = use_orig_params
        self._init_get_unflat_views_fn(align_addresses)
        self.device = device
        self._device_handle = _FSDPDeviceHandle.from_device(self.device)
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        self._sharding_strategy = sharding_strategy
        self._offload_params = offload_params
        self._use_orig_params = use_orig_params
        self._keep_low_precision_grads = keep_low_precision_grads
        self._training_state = HandleTrainingState.IDLE
        self._debug_level = dist.get_debug_level()
        self._fully_sharded_module = fully_sharded_module
        # For strategies that do not free after forward, we skip using sharded
        # views after forward since the unsharded data exists. We still switch
        # `self.flat_param` to point to the sharded flat parameter since what
        # it points to parameterizes behavior. We use the following attribute
        # to track which tensor data the parameters are unsharded views into.
        self._unsharded_flat_param_for_skipped_views: Optional[Tensor] = None
        # Optimistically assume a valid input `params` and set dtype attributes
        # before `_init_flat_param()`, which performs the actual validation
        self._orig_param_dtype = params[0].dtype
        self._init_param_reduce_dtypes(mp_param_dtype, mp_reduce_dtype)
        assert self._fwd_bwd_param_dtype is not None  # mypy
        self._aligned_numel = (
            _get_aligned_numel(unsharded_dtype=self._fwd_bwd_param_dtype)
            if align_addresses
            else 0
        )
        self._init_flat_param_and_metadata(
            params, fully_sharded_module, self._aligned_numel, use_orig_params  # type: ignore[arg-type]
        )
        self._use_unsharded_views(as_params=False)

    def _init_setattr_fns(self):
        use_unsafe_setattr = os.environ.get(_FSDP_USE_UNSAFE_SETATTR, "") == "1"
        self._setattr_tensor: Callable[[nn.Module, str, Tensor], None]
        self._setattr_param: Callable[[nn.Module, str, nn.Parameter], None]
        if use_unsafe_setattr:
            self._setattr_tensor = _unsafe_setattr_tensor
            self._setattr_param = _unsafe_setattr_param
        else:
            self._setattr_tensor = _safe_setattr_tensor_or_param
            self._setattr_param = _safe_setattr_tensor_or_param

    def _init_get_unflat_views_fn(self, align_addresses: bool):
        self._get_unflat_views = (
            self._get_unflat_views_aligned
            if align_addresses
            else self._get_unflat_views_unaligned
        )

    def _init_flat_param_and_metadata(
        self,
        params: List[Union[Tensor, nn.Parameter]],
        module: nn.Module,
        aligned_numel: int,
        use_orig_params: bool,
    ) -> None:
        """
        NOTE: This should only be called once at construction time, after which
        the ``FlatParameter`` metadata is assumed to be static.

        NOTE: The elements of ``params`` should only be ``Tensor`` s when
        composing with ``DTensor`` -based tensor parallelism, in which case the
        elements may be ``DTensor`` local shards.
        """
        if len(params) == 0:
            raise ValueError("Expects non-empty `params`")
        if aligned_numel < 0:
            raise ValueError(
                f"Expects non-negative `aligned_numel` but got {aligned_numel}"
            )
        (
            dtype,
            flat_param_requires_grad,
            device,
        ) = self._validate_tensors_to_flatten(params)
        params_set = set(params)
        # For alignment padding, only `numels` gets strictly non-`None`
        # elements, and all other lists get `None` elements for padding.
        param_infos: List[ParamInfo] = []
        numels: List[int] = []
        shapes: List[torch.Size] = []
        fqns: List[str] = []
        shared_param_infos: List[SharedParamInfo] = []
        shared_param_memo: Dict[
            Union[Tensor, nn.Parameter], Tuple[nn.Module, str, str]
        ] = {}
        params_to_flatten: List[Union[Tensor, nn.Parameter]] = []
        shared_params: List[Union[Tensor, nn.Parameter]] = []
        param_extensions: List[Any] = []
        is_padding_mask: List[bool] = []
        total_numel = total_numel_without_padding = 0
        for submodule_name, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param not in params_set:
                    continue
                if param in shared_param_memo:  # shared reference
                    prim_module, prim_module_name, prim_param_name = shared_param_memo[
                        param
                    ]
                    shared_params.append(param)
                    shared_param_infos.append(
                        SharedParamInfo(
                            param_name,
                            submodule,
                            submodule_name,
                            prim_param_name,
                            prim_module,
                            prim_module_name,
                        )
                    )
                else:
                    if aligned_numel > 0:
                        numel_to_pad = aligned_numel - (total_numel % aligned_numel)
                        if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                            padding_tensor = _construct_padding_tensor(
                                numel_to_pad, dtype, False, device
                            )
                            params_to_flatten.append(padding_tensor)
                            is_padding_mask.append(True)
                            numels.append(numel_to_pad)
                            total_numel += numel_to_pad
                    transform_t, extension = _ext_pre_flatten_transform(param)
                    param = cast(nn.Parameter, transform_t)
                    param_extensions.append(extension)
                    shared_param_memo[param] = (submodule, submodule_name, param_name)
                    params_to_flatten.append(param)
                    is_padding_mask.append(False)
                    param_infos.append(ParamInfo(param_name, submodule, submodule_name))
                    numels.append(param.numel())
                    shapes.append(param.shape)
                    fqn = (
                        submodule_name + "." + param_name
                        if submodule_name
                        else param_name
                    )
                    fqns.append(fqn)
                    total_numel += param.numel()
                    total_numel_without_padding += param.numel()
        if len(params_to_flatten) == 0:
            raise ValueError(
                f"`params` were not found in `module`'s tree"
                f"params: {params}\nmodule: {module}"
            )
        if (
            self.rank == 0
            and aligned_numel > 0
            and total_numel != total_numel_without_padding
        ):
            log.info(
                "FSDP FlatParameter address alignment created "
                "%s numel of padding (%s vs. %s)",
                total_numel - total_numel_without_padding,
                total_numel,
                total_numel_without_padding,
            )
        if aligned_numel > 0:
            # Pad to be divisible by world size to avoid a copy for the
            # post-backward reduce-scatter
            numel_to_pad = self.world_size - (total_numel % self.world_size)
            if numel_to_pad > 0 and numel_to_pad < self.world_size:
                if self.rank == 0:
                    log.info(
                        "FSDP FlatParameter world size divisibility created "
                        "%s numel of padding",
                        numel_to_pad,
                    )
                padding_tensor = _construct_padding_tensor(
                    numel_to_pad, dtype, False, device
                )
                params_to_flatten.append(padding_tensor)
                is_padding_mask.append(True)
                numels.append(numel_to_pad)
                total_numel += numel_to_pad
        # Pass `aligned_numel=0` since we already included padding tensors
        self.flat_param: FlatParameter = self.flatten_tensors_into_flat_param(
            params_to_flatten,
            aligned_numel=0,
            requires_grad=flat_param_requires_grad,
        )
        FlatParameter._init_metadata(
            self.flat_param,
            param_infos,
            numels,
            shapes,
            fqns,
            shared_param_infos,
            param_extensions,
            _convert_to_params(params_to_flatten) if use_orig_params else None,
            _convert_to_params(shared_params) if use_orig_params else None,
            is_padding_mask,
        )

    def _validate_tensors_to_flatten(
        self, tensors: List[Union[Tensor, nn.Parameter]]
    ) -> Tuple:
        """
        Validates the tensors to flatten and returns any necessary metadata.
        """
        dtype: Optional[torch.dtype] = None
        # Return as the logical OR over each tensor's value
        flat_param_requires_grad: Optional[bool] = None
        device: Optional[torch.device] = None
        # For `use_orig_params=True`, permit non-uniform `requires_grad`
        for tensor in tensors:
            if isinstance(tensor, FlatParameter):
                raise ValueError("Cannot flatten a `FlatParameter`")
            if dtype is None and not tensor.is_floating_point():
                raise ValueError("Cannot flatten integer dtype tensors")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(
                    f"Must flatten tensors with uniform dtype but got {dtype} "
                    f"and {tensor.dtype}"
                )
            if (
                not self._use_orig_params
                and flat_param_requires_grad is not None
                and tensor.requires_grad != flat_param_requires_grad
            ):
                raise ValueError(
                    "Must flatten tensors with uniform `requires_grad` when "
                    "`use_orig_params=False`"
                )
            if device is not None and tensor.device != device:
                raise ValueError(
                    "Must flatten tensors on the same device but got both "
                    f"{device} and {tensor.device}"
                )
            dtype = tensor.dtype
            flat_param_requires_grad = flat_param_requires_grad or tensor.requires_grad
            device = tensor.device
        assert flat_param_requires_grad is not None, "Requires non-empty `tensors` list"
        return dtype, flat_param_requires_grad, device

    def flatten_tensors(
        self,
        tensors: List[Tensor],
        aligned_numel: int,
    ) -> Tensor:
        """
        Flattens ``tensors`` into a single flat tensor optionally including
        padding if ``aligned_numel`` is greater than 0, where ``aligned_numel``
        gives the numel required to have address alignment.

        NOTE: The padding alignment algorithm must be kept in sync with
        :meth:`_init_flat_param_metadata`. We separate the two methods because
        the initialization happens once, whereas this method may be called
        multiple times throughout training (e.g. for checkpointing).
        """
        if len(tensors) == 0:
            raise ValueError("Expects non-empty `tensors`")
        if aligned_numel < 0:
            raise ValueError(
                f"Expects non-negative `aligned_numel` but got {aligned_numel}"
            )
        dtype, _, device = self._validate_tensors_to_flatten(tensors)
        flat_tensors: List[Tensor] = []
        if aligned_numel > 0:
            total_numel = 0
            for tensor in tensors:
                numel_to_pad = aligned_numel - (total_numel % aligned_numel)
                if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                    padding_tensor = _construct_padding_tensor(
                        numel_to_pad, dtype, False, device
                    )
                    flat_tensors.append(padding_tensor)
                    total_numel += numel_to_pad
                flat_tensors.append(torch.flatten(_detach_if_needed(tensor)))
                total_numel += tensor.numel()
            numel_to_pad = self.world_size - (total_numel % self.world_size)
            if numel_to_pad > 0 and numel_to_pad < self.world_size:
                padding_tensor = _construct_padding_tensor(
                    numel_to_pad, dtype, False, device
                )
                flat_tensors.append(padding_tensor)
                total_numel += numel_to_pad
        else:
            flat_tensors = [
                torch.flatten(_detach_if_needed(tensor)) for tensor in tensors
            ]
        return torch.cat(flat_tensors, dim=0)

    def flatten_tensors_into_flat_param(
        self,
        tensors: List[Tensor],
        aligned_numel: int,
        requires_grad: bool,
    ) -> FlatParameter:
        flat_param_data = self.flatten_tensors(tensors, aligned_numel)
        return FlatParameter(flat_param_data, requires_grad=requires_grad)

    def _init_param_reduce_dtypes(
        self,
        mp_param_dtype: Optional[torch.dtype],
        mp_reduce_dtype: Optional[torch.dtype],
    ) -> None:
        """
        Precondition: ``self.flat_param`` is set. This ensures that this
        handle's parameters have a single dtype.

        Postcondition: This sets ``self._fwd_bwd_param_dtype`` and
        ``self._reduce_dtype``. If ``mp_param_dtype`` or ``mp_reduce_dtype``
        is ``None``, then we assume the original parameter dtype. One special
        case is if ``mp_param_dtype`` is not ``None`` and ``mp_reduce_dtype``
        is ``None``, in which case we assume the gradient reduction dtype
        matches the forward/backward parameter dtype.
        """
        # Save whether these dtypes were specified so that we permit the
        # parameter dtype to change up until the lazy initialization
        self._low_prec_param_dtype_specified = mp_param_dtype is not None
        self._low_prec_reduce_dtype_specified = mp_reduce_dtype is not None
        if (
            self._low_prec_param_dtype_specified
            and not self._low_prec_reduce_dtype_specified
        ):
            # Special case: infer gradient reduction mixed precision
            self._fwd_bwd_param_dtype = mp_param_dtype
            self._reduce_dtype = self._fwd_bwd_param_dtype
        else:
            self._fwd_bwd_param_dtype = mp_param_dtype or self._orig_param_dtype
            self._reduce_dtype = mp_reduce_dtype or self._orig_param_dtype
        assert self._fwd_bwd_param_dtype is not None
        assert self._reduce_dtype is not None

    ###################################
    # SHARD INITIALIZATION & METADATA #
    ###################################
    @torch.no_grad()
    def shard(self):
        """
        Shards the handle's ``FlatParameter``. This allocates new memory for
        the sharded flat parameter and frees the unsharded flat parameter's
        storage.

        Postcondition: ``self.flat_param`` is the sharded flat parameter. Shard
        metadata attributes are set for all sharding strategies.
        """
        flat_param = self.flat_param
        if not self.uses_sharded_strategy:
            self._init_shard_metadata(0, 0, flat_param.numel() - 1)
        else:
            _p_assert(
                flat_param.storage_offset() == 0,
                "The `FlatParameter` is not the sole occupant of its storage",
            )
            orig_storage = flat_param._typed_storage()
            sharded_flat_param, numel_padded = FlatParamHandle._get_shard(
                flat_param, self.rank, self.world_size
            )
            flat_param.set_(sharded_flat_param)  # type: ignore[call-overload]
            start_idx = sharded_flat_param.numel() * self.rank
            end_idx = sharded_flat_param.numel() * (self.rank + 1) - 1  # inclusive
            self._init_shard_metadata(numel_padded, start_idx, end_idx)
            if orig_storage._size() > 0:
                orig_storage._resize_(0)
        if self._use_orig_params:
            self._use_sharded_views()

    def _init_shard_metadata(
        self,
        numel_padded: int,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> None:
        """
        Initializes shard-related metadata for this rank's shard of the flat
        parameter: ``_sharded_size``, ``_shard_param_infos``, and
        ``_shard_numel_padded``.

        Args:
            numel_padded (int): Numel padded for this rank's sharded flat
                parameter.
            unsharded_start_idx (int): Start index in the unsharded flat
            parameter assigned to this rank.
            unsharded_end_idx (int): End index (inclusive) in the unsharded
                flat parameter assigned to this rank.

        Precondition: ``self.flat_param`` 's data is the sharded flat
        parameter.
        """
        flat_param = self.flat_param
        flat_param._sharded_size = flat_param.size()  # type: ignore[attr-defined]
        sharded_flat_param_numel = flat_param.numel()  # includes `numel_padded`
        _p_assert(
            unsharded_start_idx >= 0 and unsharded_start_idx <= unsharded_end_idx,
            f"unsharded_start_idx: {unsharded_start_idx} unsharded_end_idx: {unsharded_end_idx}",
        )
        _p_assert(
            numel_padded <= sharded_flat_param_numel,
            f"numel_padded: {numel_padded} "
            f"sharded_flat_param_numel: {sharded_flat_param_numel}",
        )
        shard_param_infos = self._get_shard_metadata(
            unsharded_start_idx, unsharded_end_idx
        )
        assert (
            len(shard_param_infos) == flat_param._num_params
        ), f"Expects length {flat_param._num_params} but got {len(shard_param_infos)}"
        flat_param._shard_param_infos = shard_param_infos  # type: ignore[attr-defined]
        flat_param._shard_numel_padded = numel_padded  # type: ignore[attr-defined]

    def _get_shard_metadata(
        self,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> Tuple[_ShardParamInfo, ...]:
        """
        Computes the shard metadata based on ``unsharded_start_idx`` and
        ``unsharded_end_idx`` (inclusive), which give the interval of the
        unsharded flat parameter specifying the shard.
        """
        flat_param_offsets = self._get_flat_param_offsets()
        assert len(flat_param_offsets) == len(
            self.flat_param._numels_with_padding
        ), f"Expected {len(self.flat_param._numels_with_padding)} but got {len(flat_param_offsets)}"
        shard_param_infos: List[_ShardParamInfo] = []
        sharded_flat_param_numel = unsharded_end_idx - unsharded_start_idx + 1
        # `unsharded_param_start_idx` and `unsharded_param_end_idx` are indices
        # into the unsharded flat parameter (inclusive) of the given parameter
        for i, (
            (unsharded_param_start_idx, unsharded_param_end_idx),
            is_padding,
        ) in enumerate(zip(flat_param_offsets, self.flat_param._is_padding_mask)):
            if is_padding:
                continue
            in_sharded_flat_param = (
                unsharded_start_idx <= unsharded_param_end_idx
                and unsharded_end_idx >= unsharded_param_start_idx
            )
            if not in_sharded_flat_param:
                shard_param_info = _ShardParamInfo(False, None, None, None, None)
            else:
                if unsharded_start_idx <= unsharded_param_start_idx:
                    # This branch can only happen once since the rank's
                    # unsharded start index can only intersect one parameter
                    intra_param_start_idx = 0
                    offset_in_shard = unsharded_param_start_idx - unsharded_start_idx
                else:
                    intra_param_start_idx = (
                        unsharded_start_idx - unsharded_param_start_idx
                    )
                    offset_in_shard = 0
                assert (
                    offset_in_shard >= 0 and offset_in_shard < sharded_flat_param_numel
                ), (
                    f"Invalid `offset_in_shard` of {offset_in_shard} for "
                    f"sharded flat parameter with {sharded_flat_param_numel} numel"
                )
                intra_param_end_idx = (
                    min(unsharded_param_end_idx, unsharded_end_idx)
                    - unsharded_param_start_idx
                )
                numel_in_shard = intra_param_end_idx - intra_param_start_idx + 1
                shard_param_info = _ShardParamInfo(
                    True,
                    offset_in_shard,
                    numel_in_shard,
                    intra_param_start_idx,
                    intra_param_end_idx,
                )
            shard_param_infos.append(shard_param_info)
        return tuple(shard_param_infos)

    @staticmethod
    def _get_unpadded_shard(
        tensor: Tensor,
        rank: int,
        world_size: int,
    ) -> Tuple[Tensor, int]:
        """
        Returns the shard of ``tensor`` without any padding for the given
        ``rank`` and ``world_size`` and the numel to pad for that shard.

        If ``tensor`` is already flattened or may be viewed in the flattened
        shape (which is true in the expected usage), then this method does not
        allocate any new tensor memory.
        """
        chunks = torch.flatten(tensor).chunk(world_size)
        if len(chunks) < (rank + 1):
            # This rank gets an empty chunk fully padded with zeros since there
            # are not enough chunks across ranks
            chunk = chunks[0].new_empty(0)
        else:
            chunk = chunks[rank]
        numel_to_pad = chunks[0].numel() - chunk.numel()
        assert (
            numel_to_pad >= 0
        ), "Chunk's size should be at most the first chunk's size"
        return chunk, numel_to_pad

    @staticmethod
    def _get_shard(
        tensor: Tensor,
        rank: int,
        world_size: int,
    ) -> Tuple[Tensor, int]:
        """
        Returns the shard of ``tensor`` with padding for the given ``rank`` and
        ``world_size`` and the numel padded for that shard.

        This method allocates new memory (via :meth:`clone`) since the
        unsharded ``tensor`` may be deallocated after this method returns.
        """
        chunk, numel_to_pad = FlatParamHandle._get_unpadded_shard(
            tensor, rank, world_size
        )
        shard = chunk.clone()
        if numel_to_pad > 0:
            shard = F.pad(shard, [0, numel_to_pad])
        return shard, numel_to_pad

    @staticmethod
    def _get_sharded_size(tensor: Tensor, rank: int, world_size: int) -> torch.Size:
        """
        Returns the shape of ``tensor`` after sharding including padding. This
        requires ``tensor`` to have 1D shape and ensures that the returned
        shape is 1D.
        """
        assert len(tensor.shape) == 1, f"{tensor.shape}"
        unpadded_sharded_tensor, numel_to_pad = FlatParamHandle._get_unpadded_shard(
            tensor, rank, world_size
        )
        unpadded_sharded_size = unpadded_sharded_tensor.size()
        assert len(unpadded_sharded_size) == 1, f"{unpadded_sharded_size}"
        return torch.Size([unpadded_sharded_size[0] + numel_to_pad])

    def _get_flat_param_offsets(self) -> List[Tuple[int, int]]:
        """
        Returns [start, end] offsets of each original parameter's flattened
        data in the unsharded flat parameter (without padding).
        NOTE: The returned list includes elements for alignment padding.
        """
        cumulative_sum = list(accumulate(self.flat_param._numels_with_padding))
        starts = [0] + cumulative_sum[:-1]
        ends = [end - 1 for end in cumulative_sum]  # inclusive
        param_offsets = list(zip(starts, ends))
        return param_offsets

    @no_type_check
    def shard_metadata(
        self,
    ) -> FlatParamShardMetadata:
        """
        Returns shard-related metadata specific to this rank's shard of the
        flat parameter.
        NOTE: The returned tuple does not include elements for alignment
        padding but does account for the padding.
        """
        fqns_list = []
        shapes_list = []
        numels_list = []
        shard_param_offsets = []
        for fqn, shape, numel, shard_param_info in zip(
            self.flat_param._fqns,
            self.flat_param._shapes,
            self.flat_param._numels,
            self.flat_param._shard_param_infos,
        ):
            if not shard_param_info.in_shard:
                continue
            fqns_list.append(fqn)
            shapes_list.append(shape)
            numels_list.append(numel)
            shard_param_offsets.append(
                (
                    shard_param_info.intra_param_start_idx,
                    shard_param_info.intra_param_end_idx,
                )
            )
        return FlatParamShardMetadata(
            tuple(fqns_list),
            tuple(shapes_list),
            tuple(numels_list),
            shard_param_offsets,
        )

    @no_type_check
    @torch.no_grad()
    def init_flat_param_attributes(self) -> None:
        """
        This initializes some attributes on the handle's ``FlatParameter``.
        This should be called during lazy initialization since it requires the
        parameter to be on the compute device if not offloading to CPU and we
        want to give users the chance to move the parameter appropriately after
        the FSDP constructor.

        For each tensor attribute on the ``FlatParameter``, see the unshard and
        reshard methods in this class for the allocation and free pattern.
        """
        flat_param = self.flat_param
        if flat_param.dtype != self._orig_param_dtype:
            # Entering this branch means that the user changed the parameter
            # dtype after FSDP initialization, in which case we may need to
            # refresh some saved dtype attributes (dtypes specified as a part
            # of mixed precision take precedence).
            if not self._low_prec_param_dtype_specified:
                self._fwd_bwd_param_dtype = flat_param.dtype
            # For `reduce_dtype`, require `param_dtype` was not specified since
            # then we infer the `reduce_dtype` from the specified `param_dtype`
            if (
                not self._low_prec_reduce_dtype_specified
                and not self._low_prec_param_dtype_specified
            ):
                self._reduce_dtype = flat_param.dtype
            self._orig_param_dtype = flat_param.dtype
        cpu_device = torch.device("cpu")
        if self._offload_params:
            _p_assert(
                flat_param.device == cpu_device,
                f"Expects the `FlatParameter` to be on CPU when parameter CPU "
                f"offloading is enabled, not {flat_param.device}",
            )
        else:
            self._check_on_compute_device(self.flat_param)
        flat_param._local_shard = flat_param.data
        if self._offload_params:
            # Pin the memory for faster H2D transfer
            flat_param._local_shard = flat_param._local_shard.pin_memory()
            # Pre-allocate the sharded gradient on CPU to enable non-blocking
            # D2H transfer during the backward pass
            flat_param._cpu_grad = torch.zeros_like(
                flat_param._local_shard, device=cpu_device
            ).pin_memory()
        if self._uses_param_mixed_precision:
            # For parameter mixed precision, we maintain a low precision
            # sharded tensor on the compute device to be all-gathered (for
            # sharded strategies) or directly used (for `NO_SHARD`) for
            # computation.
            flat_param._mp_shard = torch.zeros_like(
                flat_param._local_shard,
                device=self.device,
                dtype=self._fwd_bwd_param_dtype,
            )
            _free_storage(flat_param._mp_shard)
        if self.uses_sharded_strategy:
            # We maintain a padded unsharded tensor that serves as the
            # all-gather destination and owns the original parameter storages.
            unsharded_param_dtype = (
                self._fwd_bwd_param_dtype
                if self._uses_param_mixed_precision
                else flat_param.dtype
            )  # use low precision if parameter mixed precision is enabled
            padded_unsharded_numel = flat_param.numel() * self.world_size
            flat_param._full_param_padded = torch.zeros(
                padded_unsharded_numel,
                device=self.device,
                dtype=unsharded_param_dtype,
            )
            flat_param._padded_unsharded_size = flat_param._full_param_padded.size()
            _free_storage(flat_param._full_param_padded)

            if self._uses_param_mixed_precision:
                # For parameter mixed precision, we maintain a full precision
                # padded unsharded tensor for when we force full precision.
                flat_param._full_prec_full_param_padded = torch.zeros(
                    padded_unsharded_numel,
                    device=self.device,
                    dtype=flat_param.dtype,  # full precision
                )
                _free_storage(flat_param._full_prec_full_param_padded)

    ###################
    # UNSHARD/RESHARD #
    ###################
    def pre_unshard(self) -> bool:
        """
        Returns: ``False`` if this is a no-op and ``True`` otherwise.

        Postcondition: ``self.flat_param`` 's data is on the device for
        communication and is what should be all-gathered. This means that it
        matches the dtype of the expected unsharded parameter.
        """
        if (
            self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS
            and self._skipped_use_sharded_views
        ):
            # Since this path imposes special semantics for the unsharded flat
            # parameter (e.g. forcing full precision), use sharded views to
            # reuse the existing logic for that special handling
            self._use_sharded_views()
        ret = False
        if self._use_orig_params and not self._skip_writeback_check:
            ret = self._writeback_orig_params()
        if (
            self.uses_sharded_strategy
            and not self._offload_params
            and not self.needs_unshard()
        ):
            pass  # no-op
        elif self._uses_param_mixed_precision and not self._force_full_precision:
            self._use_low_precision_shard()
            ret = True
        elif self._offload_params and self.flat_param.device != self.device:
            # NOTE: This creates a new tensor distinct from any attributes.
            self.flat_param_to(self.device, non_blocking=True)
            ret = True
        self._check_on_compute_device(self.flat_param)
        return ret

    def _use_low_precision_shard(self):
        """
        Allocates the low precision shard directly on the compute device and
        switches to using the low precision sharded flat parameter.
        """
        self._check_low_precision_shard()
        flat_param = self.flat_param
        _alloc_storage(
            flat_param._mp_shard, flat_param._local_shard.size()  # type: ignore[attr-defined]
        )
        # `copy_()` implicitly casts to the low precision
        flat_param._mp_shard.copy_(  # type: ignore[attr-defined]
            flat_param._local_shard.to(  # type: ignore[attr-defined]
                self.device, non_blocking=True
            )
        )
        # Invariant: `_mp_shard` is always on the compute device.
        flat_param.data = flat_param._mp_shard  # type: ignore[attr-defined]

    def unshard(self):
        """
        Runs the unshard logic. This includes all-gathering the flat parameter
        and switching to using the unsharded flat parameter. If the handle does
        not need unsharding, then this only switches to using the unsharded
        flat parameter. For ``NO_SHARD``, this is a no-op.

        If FSDP is in :meth:`summon_full_params` and the handle uses parameter
        mixed precision, then the parameter is forced to full precision.
        """
        if not self.needs_unshard():
            # Even when not needing an unshard, we should switch to using
            # the unsharded flat parameter
            unsharded_flat_param = (
                self._get_padded_unsharded_flat_param()
                if self.uses_sharded_strategy
                else self.flat_param
            )
            self._use_unsharded_flat_param(unsharded_flat_param)
            return
        unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
        padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)
        self._use_unsharded_flat_param(padded_unsharded_flat_param)

    def needs_unshard(self) -> bool:
        """Returns if the handle's flat parameter needs to be unsharded."""
        if not self.uses_sharded_strategy:
            return False
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        already_unsharded = (
            unsharded_flat_param._typed_storage()._size()
            == unsharded_flat_param.numel()
        )
        return not already_unsharded

    def _alloc_padded_unsharded_flat_param(self):
        """
        Allocates the *padded* unsharded flat parameter. The unpadded unsharded
        flat parameter is always a view into the padded one. This padded
        parameter is saved to a different attribute on the ``FlatParameter``
        depending on if we force full precision.
        """
        self._check_sharded_strategy()
        flat_param = self.flat_param
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        self._check_storage_freed(unsharded_flat_param)
        _alloc_storage(unsharded_flat_param, flat_param._padded_unsharded_size)  # type: ignore[attr-defined]
        return unsharded_flat_param

    def _get_padded_unsharded_flat_param(self) -> torch.Tensor:
        """
        Returns a reference to the padded unsharded flat parameter depending on
        the calling context. This should only be called if using a sharded
        strategy.
        """
        self._check_sharded_strategy()
        flat_param = self.flat_param
        if self._force_full_precision and self._uses_param_mixed_precision:
            # When parameter mixed precision is enabled, we use a different
            # tensor as the all-gather destination to preserve the invariant
            # that  `_full_param_padded` is in the low precision
            unsharded_flat_param = flat_param._full_prec_full_param_padded  # type: ignore[attr-defined]
            _p_assert(
                unsharded_flat_param.dtype != self._fwd_bwd_param_dtype,
                f"Expects full precision but got {self._fwd_bwd_param_dtype}",
            )
        else:
            unsharded_flat_param = flat_param._full_param_padded  # type: ignore[attr-defined]
        return unsharded_flat_param

    def _all_gather_flat_param(
        self,
        padded_unsharded_flat_param: Tensor,
    ) -> Tensor:
        """
        All-gathers the handle's flat parameter to the destination
        ``padded_unsharded_flat_param``, and switches to using the all-gathered
        tensor.
        """
        _p_assert(
            hasattr(self, "process_group") and hasattr(self, "world_size"),
            "Expects a process group and world size to have been set via `shard()`",
        )
        sharded_flat_param = self.flat_param.data
        expected_numel = sharded_flat_param.numel() * self.world_size
        _p_assert(
            padded_unsharded_flat_param.numel() == expected_numel,
            f"Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}",
        )

        # HACK this should be handled by C10D
        if sharded_flat_param.is_cpu:  # type: ignore[attr-defined]
            tensor_list = list(
                torch.chunk(
                    padded_unsharded_flat_param, dist.get_world_size(self.process_group)
                )
            )
            work = dist.all_gather(
                tensor_list, sharded_flat_param, group=self.process_group
            )
        else:
            dist.all_gather_into_tensor(
                padded_unsharded_flat_param,
                sharded_flat_param,
                self.process_group,
            )
        return padded_unsharded_flat_param

    def _use_unsharded_flat_param(
        self,
        padded_unsharded_flat_param: torch.Tensor,
    ) -> None:
        """
        Switches to using the *unpadded* unsharded flat parameter, which is a
        view into the *padded* unsharded flat parameter.
        """
        unsharded_size = self.flat_param._unpadded_unsharded_size
        self.flat_param.data = padded_unsharded_flat_param[
            : unsharded_size.numel()
        ].view(
            unsharded_size
        )  # this `.view()` is not autograd visible
        in_forward = self._training_state == HandleTrainingState.FORWARD
        in_pre_backward = self._training_state == HandleTrainingState.BACKWARD_PRE
        if self._use_orig_params:
            if self._skipped_use_sharded_views and in_pre_backward:
                # This call corresponds to the complementary pre-backward
                # `_use_unsharded_views()` to the skipped pre-forward
                # `_use_sharded_views()`, so we should skip this one too.
                return
            # We use `Tensor` views in the forward so that they are tracked by
            # autograd. We use them in the pre-backward as well to support
            # reentrant activation checkpointing, which needs the views to be
            # tracked by autograd in the backward pass's recomputed forward.
            self._use_unsharded_views(
                as_params=(not in_forward and not in_pre_backward)
            )
        elif in_forward:
            self._use_unsharded_views(as_params=False)

    def post_unshard(self):
        """
        Runs the post-unshard logic. This includes freeing the low precision
        shard if needed.
        """
        if self._uses_param_mixed_precision and self.uses_sharded_strategy:
            self._free_low_precision_sharded_param()
        self._check_on_compute_device(self.flat_param)

    def _free_low_precision_sharded_param(self):
        """Frees the low precision sharded flat parameter."""
        self._check_low_precision_shard()
        # `_mp_shard` is allocated in the pre-unshard stream, consumed in the
        # unshard stream for sharded strategies, and consumed in both the
        # unshard and default streams for `NO_SHARD`. For sharded strategies,
        # the current stream here is the unshard stream, and for `NO_SHARD`,
        # it is the default stream. For `NO_SHARD`, only recording for the
        # default stream suffices since the default stream waits for the
        # unshard stream.
        _no_dispatch_record_stream(
            self.flat_param._mp_shard, self._device_handle.current_stream()  # type: ignore[attr-defined]
        )
        _free_storage(self.flat_param._mp_shard)  # type: ignore[attr-defined]

    @torch.no_grad()
    def unshard_grad(self):
        """
        Unshards the handle's ``FlatParameter`` 's gradient. If all ranks have
        ``None`` gradient, then all original parameters will as well. This
        method performs an all-reduce and an all-gather. The additional
        all-reduce is tolerable since this method is not meant to be used on
        the computation critical path.

        Postcondition: ``_saved_grad_shard`` is defined and contains the value
        to set ``flat_param.grad`` after gradients are resharded.
        """
        if not self.uses_sharded_strategy:
            self._use_unsharded_grad_views()
            return
        flat_param = self.flat_param
        self._check_unsharded(flat_param)

        # Check if all ranks have a `None` gradient
        num_grad_none = torch.zeros(1, dtype=torch.int32, device=self.device)
        num_grad_none[0] = flat_param.grad is None
        dist.all_reduce(num_grad_none, group=self.process_group)
        if num_grad_none[0] == self.world_size:
            flat_param._saved_grad_shard = None  # type: ignore[assignment]
            self._use_unsharded_grad_views()
            return

        padded_unsharded_grad = torch.empty(
            flat_param._padded_unsharded_size,  # type: ignore[attr-defined]
            device=self.device,
        )
        if flat_param.grad is None:
            # In the case that only some ranks have `None` gradient, we use
            # zeros to approximate as a best effort attempt
            if self._debug_level == dist.DebugLevel.INFO:
                warnings.warn(
                    f"[Rank {self.rank}] Only some but not all ranks have a "
                    "`None` `FlatParameter` gradient, so FSDP is using zeros to "
                    "approximate those ranks' sharded gradients being `None`"
                )
            flat_param._saved_grad_shard = None  # type: ignore[assignment]
            sharded_grad = torch.zeros(flat_param._sharded_size, device=self.device)  # type: ignore[attr-defined]
        else:
            self._check_sharded(flat_param.grad)
            flat_param._saved_grad_shard = flat_param.grad  # type: ignore[attr-defined]
            sharded_grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
        dist.all_gather_into_tensor(
            padded_unsharded_grad, sharded_grad, self.process_group
        )
        unsharded_size = self.flat_param._unpadded_unsharded_size
        flat_param.grad = padded_unsharded_grad[: unsharded_size.numel()].view(
            unsharded_size
        )
        self._use_unsharded_grad_views()

    def reshard_grad(self):
        if self._use_orig_params:
            self._use_sharded_grad_views()
        if not self.uses_sharded_strategy:
            return
        self.flat_param.grad = self.flat_param._saved_grad_shard  # type: ignore[attr-defined]
        delattr(self.flat_param, "_saved_grad_shard")

    def prepare_gradient_for_backward(self):
        """
        Prepares the gradient for the backward computation by saving and
        clearing any existing sharded gradient in ``.grad`` to enable computing
        a new unsharded gradient.
        """
        _p_assert(
            self._training_state
            in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.IDLE),
            "Expects to be in `BACKWARD_PRE` or `IDLE` (if prefetching)",
        )
        flat_param = self.flat_param
        if flat_param.grad is not None and (
            flat_param.grad.size() != flat_param._unpadded_unsharded_size
            or flat_param.grad.device != flat_param.device  # grad on CPU
        ):
            self._check_on_compute_device(self.flat_param)
            grad_offloaded = flat_param.grad.device != self.device
            _p_assert(
                not grad_offloaded or self._offload_params,
                f"Expects the sharded gradient to be on {self.device} "
                f"but got {flat_param.grad.device}",
            )
            prev_iter_synced_gradients = (
                flat_param.grad.size()
                == flat_param._local_shard.size()  # type: ignore[attr-defined]
            )
            if prev_iter_synced_gradients:
                # TODO (awgu): Gradient accumulation outside `no_sync()`
                # does not work with CPU offloading. The issue should be
                # that, in the post-backward hook, we cannot do an addition
                # between a CPU tensor (the existing sharded gradient) and
                # a GPU tensor (the new sharded gradient).
                if not grad_offloaded:
                    flat_param._saved_grad_shard = flat_param.grad.data  # type: ignore[attr-defined]
                    sharded_grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
                else:
                    _p_assert(
                        hasattr(flat_param, "_cpu_grad"),
                        "`_cpu_grad` should be defined if the gradient is on CPU",
                    )
                    sharded_grad = flat_param._cpu_grad  # type: ignore[attr-defined]
                # If user specified to keep the gradient in low precision, then
                # the gradient may still be of the low precision dtype if the
                # user did not set the gradient to `None` after the previous
                # backward, in which case FSDP should cast back to the full
                # precision dtype so that FSDP can accumulate in that dtype in
                # the post-backward hook and assign to `.grad` in that dtype in
                # the post-backward callback.
                local_shard_dtype = flat_param._local_shard.dtype  # type: ignore[attr-defined]
                if (
                    self._keep_low_precision_grads
                    and sharded_grad.dtype != local_shard_dtype
                ):
                    sharded_grad.data = sharded_grad.to(local_shard_dtype)
            else:
                padded_unsharded_size = flat_param._padded_unsharded_size  # type: ignore[attr-defined]
                _p_assert(
                    flat_param.grad.size() == padded_unsharded_size,
                    "Expects `.grad` to be the unsharded gradient in "
                    f"`no_sync()` with size {padded_unsharded_size} "
                    f"but got size {flat_param.grad.size()}",
                )
            flat_param.grad = None

    def prepare_gradient_for_optim(self):
        """
        Prepares the gradient for optimizer computation by moving the sharded
        gradient to the ``.grad`` attribute.
        """

        def cast_grad_to_param_dtype_if_needed(flat_param):
            # TODO (rohan-varma): test for full precision with keep_low_precision_grads
            if not self._force_full_precision and self._keep_low_precision_grads:
                _p_assert(flat_param.grad is not None, "Unexpected None grad!")
                if flat_param.grad.dtype != self._fwd_bwd_param_dtype:
                    flat_param.grad.data = flat_param.grad.to(self._fwd_bwd_param_dtype)
                    if self._use_orig_params:
                        self._use_sharded_grad_views()

        flat_param = self.flat_param
        # TODO (awgu): We should replace these conditional checks to encode
        # the logical intention more directly.
        if hasattr(flat_param, "_cpu_grad"):
            # NOTE: This branch includes `NO_SHARD`.
            self._check_sharded(flat_param)
            self._check_on_cpu(flat_param)
            flat_param.grad = flat_param._cpu_grad  # type: ignore[attr-defined]
            cast_grad_to_param_dtype_if_needed(flat_param)
        elif hasattr(flat_param, "_saved_grad_shard"):
            self._check_sharded(flat_param)
            self._check_on_compute_device(flat_param)
            if flat_param._saved_grad_shard is not None:
                self._check_on_compute_device(flat_param._saved_grad_shard)  # type: ignore[attr-defined]
            # If no sharded gradient was computed this iteration, then there is
            # no need to forward `_saved_grad_shard` to `grad`
            if flat_param._post_backward_called:  # type: ignore[attr-defined]
                flat_param.grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
                if flat_param.grad is not None:
                    cast_grad_to_param_dtype_if_needed(flat_param)
        else:
            _p_assert(
                not self.uses_sharded_strategy
                or not flat_param._post_backward_called,  # type: ignore[attr-defined]
                "All sharded parameters that received a gradient in the "
                "post-backward should use `_saved_grad_shard`",
            )
        # Delete `_saved_grad_shard` since its existence indicates a previous
        # gradient to accumulate with in the post-backward hook
        if hasattr(flat_param, "_saved_grad_shard"):
            delattr(flat_param, "_saved_grad_shard")

    @contextlib.contextmanager
    def to_cpu(self):
        """
        Moves the unpadded unsharded flat parameter to CPU while in the context
        and moves it back to the previous device upon exit. For now, this
        assumes the ``FlatParameter`` is the unpadded unsharded flat parameter
        since (1) there is no reason to include the padding in the copy and (2)
        there is no use case for the sharded flat parameter.

        Precondition: ``self.flat_param`` 's data is the unpadded unsharded
        flat parameter on the compute device, and the handle uses a sharded
        strategy.
        Postcondition: Same as the precondition.
        """
        self._check_sharded_strategy()
        _p_assert(
            self.flat_param.size() == self.flat_param._unpadded_unsharded_size,
            f"Expects size {self.flat_param._unpadded_unsharded_size} but got {self.flat_param.size()}",
        )
        self._check_on_compute_device(self.flat_param)
        # Check that the unpadded unsharded flat parameter is a view into the
        # padded unsharded flat parameter as expected
        # NOTE: This check is not strictly needed for correctness but is a
        # useful sanity check since the tensor should only be used internally.
        unpadded_storage_ptr = self.flat_param._typed_storage()._data_ptr()
        padded_storage_ptr = (
            self._get_padded_unsharded_flat_param()._typed_storage()._data_ptr()
        )
        _p_assert(
            unpadded_storage_ptr == padded_storage_ptr,
            "Expects the unpadded parameter to be a view into the padded parameter",
        )
        self.flat_param_to(torch.device("cpu"))
        self._free_unsharded_flat_param()
        try:
            yield
        finally:
            _p_assert(
                self.flat_param.size() == self.flat_param._unpadded_unsharded_size,
                f"Expects size {self.flat_param._unpadded_unsharded_size} but got {self.flat_param.size()}",
            )
            padded_unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
            # Copy from CPU to the compute device
            padded_unsharded_flat_param[: self.flat_param.numel()].copy_(
                self.flat_param
            )
            self._use_unsharded_flat_param(padded_unsharded_flat_param)

    def reshard(self, free_unsharded_flat_param: bool):
        """
        Runs the reshard logic. This includes freeing the unsharded flat
        parameter if ``free_unsharded_flat_param`` and switching to using the
        sharded flat parameter. Note that this also implicitly offloads
        the sharded flat parameter (if CPU offload is enabled) by pointing
        it to the ``_local_shard`` attribute which resides on CPU.
        """
        # Switch to the sharded `FlatParameter` before freeing to prevent
        # "use-after-free"-type bugs with external profiling tools, where for
        # `use_orig_params=True`, the `param` does not point to valid memory
        # when setting `param.data = ...` in `_use_sharded_views()`.
        self._use_sharded_flat_param()
        if free_unsharded_flat_param:
            self._free_unsharded_flat_param()

    def post_reshard(self):
        """
        Runs the post-reshard logic. This includes freeing any memory that
        can now be freed given that the ``FlatParameter`` points to the full
        precision sharded flat parameter.

        Precondition: ``self.flat_param`` 's data points to the full precision
        sharded flat parameter.
        """
        # For `NO_SHARD`, `_mp_shard` is not freed in the post-unshard since it
        # is also the low precision *unsharded* flat parameter. Hence, we delay
        # the free until the reshard.
        if (
            self._uses_param_mixed_precision
            and not self.uses_sharded_strategy
            and not self._force_full_precision  # did not use the low precision shard
        ):
            self._free_low_precision_sharded_param()

    def _free_unsharded_flat_param(self):
        """
        Frees the padded unsharded flat parameter. The tensor to free depends
        on the calling context since the unshard may have forced full
        precision, in which case a different tensor is used.
        """
        self._check_sharded_strategy()
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        self._check_storage_allocated(unsharded_flat_param)
        self._check_on_compute_device(unsharded_flat_param)
        # Do not free the memory until all ops in the current stream finish
        _no_dispatch_record_stream(
            unsharded_flat_param, self._device_handle.current_stream()
        )
        _free_storage(unsharded_flat_param)

    def _use_sharded_flat_param(self) -> None:
        """Switches to using the sharded flat parameter."""
        flat_param = self.flat_param
        if self._use_orig_params:
            in_forward = self._training_state == HandleTrainingState.FORWARD
            skip_use_sharded_views = (
                in_forward
                and self._sharding_strategy
                in NO_RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
            )
            # Only incur the extra `.data` call if needed
            if skip_use_sharded_views:
                unsharded_flat_param = flat_param.data
        if self._offload_params:
            device = flat_param._local_shard.device  # type: ignore[attr-defined]
            _p_assert(
                device == torch.device("cpu"),
                f"Expects the local shard to be on CPU but got {device}",
            )
        flat_param.data = flat_param._local_shard  # type: ignore[attr-defined]
        if self._use_orig_params:
            if skip_use_sharded_views:
                self._unsharded_flat_param_for_skipped_views = unsharded_flat_param
            else:
                self._use_sharded_views()
            # For the post-forward reshard, we may try to use sharded gradient
            # views (or unsharded gradient views if a gradient was accumulated
            # in `no_sync()`), but for the post-backward reshard, we delay the
            # call to after the reduce-scatter.
            if (
                in_forward
                # Skip using gradient views if skipped using sharded views
                # since exposing unsharded parameters with sharded gradients
                # may be confusing to the user
                and not self._skipped_use_sharded_views
            ):
                # TODO: Change `_unpadded_unsharded_size` if we change the
                # gradient to be computed directly with padding.
                accumulated_grad_in_no_sync = (
                    flat_param.grad is not None
                    and self.uses_sharded_strategy
                    and flat_param.grad.shape == flat_param._unpadded_unsharded_size
                )
                if accumulated_grad_in_no_sync:
                    self._use_unsharded_grad_views()
                else:
                    self._use_sharded_grad_views()

    #########
    # VIEWS #
    #########
    @no_type_check
    def _get_unflat_views_unaligned(
        self,
        tensor: Optional[torch.Tensor] = None,
    ) -> Iterator[Tensor]:
        """
        Returns unflattened ``Tensor`` views into ``tensor`` if it is not
        ``None`` or ``flat_param`` otherwise, where the unflattening is based
        on ``flat_param`` 's metadata.

        Examples for ``tensor`` include ``flat_param.grad`` or unsharded
        tensor optimizer state.
        """
        flat_param = self.flat_param
        if tensor is None:
            tensor = flat_param
        views = (
            _ext_post_unflatten_transform(subtensor.view(shape), param_extension)
            for (subtensor, shape, param_extension) in zip(
                torch.split(tensor, flat_param._numels, dim=0),
                flat_param._shapes,
                flat_param._param_extensions,
            )
        )
        return views

    @no_type_check
    def _get_unflat_views_aligned(
        self,
        tensor: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """
        This has the same contract as :meth:`_get_unflat_views_unaligned`
        except it checks for ``None`` placeholders representing padding for
        alignment, which may incur slightly more CPU overhead.
        """
        flat_param = self.flat_param
        if tensor is None:
            tensor = flat_param
        splits: List[Tensor] = torch.split(
            tensor, flat_param._numels_with_padding, dim=0
        )
        idx = 0
        views: List[Tensor] = []
        for split, is_padding in zip(splits, flat_param._is_padding_mask):
            if is_padding:
                continue
            views.append(
                _ext_post_unflatten_transform(
                    split.view(flat_param._shapes[idx]),
                    flat_param._param_extensions[idx],
                )
            )
            idx += 1
        return views

    @no_type_check
    def _use_unsharded_views(self, as_params: bool) -> None:
        """
        Unflattens the unsharded flat parameter by setting the original
        parameter variables to be views into it.

        Args:
            as_params (bool): If ``True``, then registers the original
                parameters as ``nn.Parameter`` s; if ``False``, then registers
                the original parameters only as ``Tensor`` s. ``False`` should
                be used during forward/backward computation and when hiding the
                original parameters from :meth:`nn.Module.named_parameters`.
        """
        flat_param = self.flat_param
        self._check_unsharded(flat_param)
        views = self._get_unflat_views()
        for i, (view, (param_name, module, _)) in enumerate(
            zip(views, flat_param._param_infos)
        ):
            if self._use_orig_params and as_params:
                if type(view) is DTensor:
                    # A `DTensor` `view` is not compatible with assigning
                    # `param.data = view`, so we cannot preserve the parameter
                    # variable.
                    self._setattr_param(module, param_name, nn.Parameter(view))
                    continue
                param = self.flat_param._params[i]
                self._setattr_param(module, param_name, param)
                param.data = view
            elif as_params:
                self._setattr_param(module, param_name, nn.Parameter(view))
            else:  # `as_params=False`
                param_var: Tensor = view
                if self._use_orig_params:
                    if self._training_state == HandleTrainingState.FORWARD:
                        # Save the `Tensor` for the pre-backward
                        self.flat_param._tensors[i] = view  # save for pre-backward
                    elif self._training_state == HandleTrainingState.BACKWARD_PRE:
                        # Use the saved `Tensor` variable from the forward to
                        # preserve the autograd graph so that the post-backward
                        # hook fires (e.g. for reentrant AC)
                        tensor = self.flat_param._tensors[i]
                        tensor.data = view
                        param_var = tensor
                self._setattr_tensor(module, param_name, param_var)
                if (
                    self._use_orig_params
                    and self._training_state == HandleTrainingState.FORWARD
                ):
                    module._parameters[param_name] = param_var
        for i, (
            param_name,
            module,
            _,
            prim_param_name,
            prim_module,
            _,
        ) in enumerate(self.flat_param._shared_param_infos):
            prim_param: Union[Tensor, nn.Parameter] = getattr(
                prim_module, prim_param_name
            )
            _p_assert(
                not as_params or isinstance(prim_param, nn.Parameter),
                f"as_params={as_params} type(prim_param)={type(prim_param)}",
            )
            if self._use_orig_params and as_params:
                shared_param = self.flat_param._shared_params[i]
                self._setattr_param(module, param_name, shared_param)
                shared_param.data = prim_param
            elif as_params:
                self._setattr_param(module, param_name, prim_param)
            else:
                self._setattr_tensor(module, param_name, prim_param)
                if (
                    self._use_orig_params
                    and self._training_state == HandleTrainingState.FORWARD
                ):
                    module._parameters[param_name] = prim_param

    @no_type_check
    def _use_unsharded_grad_views(self) -> None:
        """
        Unflattens the unsharded flat parameter's gradient by setting the
        original parameter variables' gradients to be views into it.
        """
        # Expects the gradient to be in `flat_param.grad`
        if self.flat_param.grad is None:
            for param in chain(self.flat_param._params, self.flat_param._shared_params):
                param.grad = None
            return
        self._check_unsharded(self.flat_param.grad)
        views = self._get_unflat_views(self.flat_param.grad)
        for i, (view, (param_name, module, _)) in enumerate(
            zip(views, self.flat_param._param_infos)
        ):
            _p_assert(
                hasattr(module, param_name),
                f"{self.flat_param._fqns[i]} is missing",
            )
            param = getattr(module, param_name)
            if (
                param.shape != view.shape
                or param.dtype != view.dtype
                or param.device != view.device
            ):
                # NOTE: This is a hack using `.data` to side step the check
                # that parameter/gradient sizes/dtypes/devices match. From
                # calling `reshard()`, `param` has the sharded size, has the
                # full precision dtype, and if CPU offloading is enabled, is on
                # CPU. Thus, one or more of the following cases can hold when
                # in `no_sync()`, where `view` is the original parameter's
                # gradient:
                # 1. `view` can have the unsharded size.
                # 2. `view` can have the parameter low precision dtype.
                # 3. `view` can be on GPU.
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.data = view
            else:
                param.grad = view
        for i, (
            param_name,
            module,
            module_name,
            prim_param_name,
            prim_module,
            _,
        ) in enumerate(self.flat_param._shared_param_infos):
            _p_assert(
                hasattr(module, param_name),
                f"{module_name + '.' + param_name if module_name else param_name} is missing",
            )  # did not save FQN info in `_shared_param_infos`
            param = getattr(module, param_name)
            prim_param = getattr(prim_module, prim_param_name)
            if (
                param.shape != prim_param.grad.shape
                or param.dtype != prim_param.grad.dtype
                or param.device != prim_param.grad.device
            ):
                # NOTE: This is the same hack to use `.data` to side step the
                # size check.
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.data = prim_param.grad
            else:
                param.grad = prim_param.grad

    @contextlib.contextmanager
    def unflatten_as_params(self) -> Generator:
        """
        Assumes the flat parameter is unsharded. When in the context,
        unflattens the original parameters as ``nn.Parameter`` views into the
        flat parameter, and after the context, restores the original parameters
        as ``Tensor`` views into the flat parameter.
        """
        self._use_unsharded_views(as_params=True)
        try:
            yield
        finally:
            self._use_unsharded_views(as_params=False)

    @no_type_check
    @torch.no_grad()
    def _use_sharded_views(self) -> None:
        """
        Sets the original parameter variables' data to be flattened views into
        the sharded flat parameter.

        The views are kept as flattened to simplify the case where a parameter
        is sharded across ranks. Parameters whose data is not present in the
        sharded flat parameter have their data set to a size-0 empty tensor. We
        do not delete them to ensure to preserve expected behaviors like model
        printability. Parameters whose data is present must preserve their
        variables to be passable to an optimizer.
        """
        self._unsharded_flat_param_for_skipped_views = None
        if not self.uses_sharded_strategy:
            # For `NO_SHARD`, use the *unflattened* unsharded views since we
            # have the unsharded parameter
            self._use_unsharded_views(as_params=True)
            return
        flat_param = self.flat_param
        self._check_sharded(flat_param)
        # Construct once and reuse for all parameters not in the local shard
        size_0_empty_tensor = torch.empty(
            0,
            dtype=self.flat_param.dtype,  # in case `flat_param` changed dtype
            device=self.flat_param.device,
            requires_grad=False,
        )
        for param, shard_param_info, (param_name, module, _) in zip(
            flat_param._params, flat_param._shard_param_infos, flat_param._param_infos
        ):
            self._setattr_param(module, param_name, param)
            if not shard_param_info.in_shard:
                # Allow the original data to be freed via garbage collection
                param.data = size_0_empty_tensor
            else:
                offset = shard_param_info.offset_in_shard
                numel_in_shard = shard_param_info.numel_in_shard
                param.data = flat_param[offset : offset + numel_in_shard]
        assert self.flat_param._shared_params is not None
        for i, (
            param,
            (param_name, module, _, prim_param_name, prim_module, _),
        ) in enumerate(
            zip(self.flat_param._shared_params, self.flat_param._shared_param_infos)
        ):
            self._setattr_param(module, param_name, param)
            prim_param = getattr(prim_module, prim_param_name)
            param.data = prim_param  # could be both empty and non-empty
        if self._training_state == HandleTrainingState.BACKWARD_POST:
            # Clear the saved `Tensor`s since they are unneeded now
            for i in range(len(self.flat_param._tensors)):
                self.flat_param._tensors[i] = None

    @no_type_check
    @torch.no_grad()
    def _use_sharded_grad_views(self) -> None:
        """
        Sets the original parameter variables' gradients to be flattened
        views into the sharded flat parameter's gradient. This is a no-op if
        there is no gradient.

        Parameters whose data is not present in the sharded flat parameter and
        parameters with ``requires_grad=False`` have their gradients set to
        ``None``. Since the gradient variables do not need to be preserved,
        this method does not manipulate existing ``Tensor`` data directly and
        creates new ``Tensor`` variables instead.
        """
        flat_param = self.flat_param
        self._check_sharded(flat_param)
        grad = self.sharded_grad
        if grad is None:
            for param in chain(flat_param._params, flat_param._shared_params):
                param.grad = None
            return
        self._check_sharded(grad)
        for param, shard_param_info, is_grad_none in zip(
            flat_param._params,
            flat_param._shard_param_infos,
            flat_param._is_grad_none_mask,
        ):
            if not shard_param_info.in_shard:
                param.grad = None
            else:
                numel_in_shard = shard_param_info.numel_in_shard
                if param.requires_grad and not is_grad_none:
                    offset = shard_param_info.offset_in_shard
                    if self._keep_low_precision_grads or param.dtype != grad.dtype:
                        # NOTE: This is a hack using `.data` to side step the
                        # check that parameter/gradient dtypes match. Here,
                        # `param` has full precision; `grad` has low precision.
                        if param.grad is None:
                            # `.grad` must have the same shape as `param`
                            param.grad = torch.empty_like(param)
                        param.grad.data = grad[
                            offset : offset + numel_in_shard
                        ].reshape(param.shape)
                    else:
                        param.grad = grad[offset : offset + numel_in_shard].reshape(
                            param.shape
                        )
                else:
                    param.grad = None
        assert flat_param._shared_params is not None
        for i, (param, (_, _, _, prim_param_name, prim_module, _)) in enumerate(
            zip(flat_param._shared_params, flat_param._shared_param_infos)
        ):
            in_sharded_flat_param = hasattr(prim_module, prim_param_name)
            if in_sharded_flat_param and param.requires_grad:
                prim_param = getattr(prim_module, prim_param_name)
                param.grad = prim_param.grad  # share the same reference
            else:
                param.grad = None

    @no_type_check
    @torch.no_grad()
    def _writeback_orig_params(self) -> bool:
        """
        Iterates over the original parameters and writes back any parameters
        that changed storages (due to a non-inplace operator) to the handle's
        ``FlatParameter``. This method preserves the ``FlatParameter` 's
        device even if an original parameter's device changes.

        Raises:
            RuntimeError: If an original parameter or gradient changes storages
            but no longer has the expected flattened shape.
        Returns: ``True`` if some writeback happened, and ``False`` otherwise.
        """
        if (
            self.uses_sharded_strategy
            and not self.is_sharded(self.flat_param)
            and not self._skipped_use_sharded_views
        ):
            # For `NO_SHARD`, we may still need to writeback
            return False
        flat_param = self.flat_param
        wroteback = False
        if self._skipped_use_sharded_views and self.uses_sharded_strategy:
            # NOTE: We must use the unsharded flat parameter from which the
            # unsharded views were computed, not the one from the current
            # calling context (`_get_padded_unsharded_flat_param()`) since that
            # may be different (e.g. the model changed from train to eval).
            flat_param_data_ptr = (
                self._unsharded_flat_param_for_skipped_views.untyped_storage().data_ptr()
            )
            _p_assert(
                flat_param_data_ptr > 0,
                "If skipped using sharded views, the unsharded flat parameter "
                "should be allocated",
            )
        else:
            flat_param_data_ptr = flat_param.untyped_storage().data_ptr()
        # NOTE: Since this method is called in the pre-unshard, which is only
        # called during computation in the pre-forward or pre-backward, the
        # sharded gradient should be guaranteed to be in `.grad`, not in
        # `._saved_grad_shard`.
        flat_param_grad = (
            flat_param.grad
            if self.uses_sharded_strategy or not self._offload_params
            else flat_param._cpu_grad
        )
        flat_param_grad_data_ptr = (
            None
            if flat_param_grad is None
            else flat_param_grad.untyped_storage().data_ptr()
        )
        for i, (
            param,
            (in_shard, offset_in_shard, numel_in_shard, _, _),
            (param_name, module, _),
        ) in enumerate(
            zip(
                flat_param._params,
                flat_param._shard_param_infos,
                flat_param._param_infos,
            )
        ):
            if not in_shard:
                continue
            if not hasattr(module, param_name):
                # Do not writeback if original parameters are deregistered
                # (e.g. during model checkpointing)
                continue

            # Check for parameter writeback
            if self._skipped_use_sharded_views:
                param = flat_param._tensors[i]
                _p_assert(
                    param is not None,
                    f"Expects to have saved tensor for {flat_param._fqns[i]}",
                )
            param_changed = getattr(module, param_name) is not param
            needs_param_writeback = (
                param_changed  # changed parameter variable itself
                or not _same_storage_as_data_ptr(
                    param, flat_param_data_ptr
                )  # changed `.data`
            )
            if self._skipped_use_sharded_views and (
                param_changed or needs_param_writeback
            ):
                raise AssertionError(
                    "FSDP does not support changing the parameters between "
                    f"forward and backward for {self._sharding_strategy}"
                )
            if param_changed:
                # NOTE: The gradient is not preserved after a parameter change.
                param = getattr(module, param_name)
                flat_param._params[i] = param
            if needs_param_writeback:
                expected_shape = torch.Size([numel_in_shard])
                self._writeback_tensor(
                    param, flat_param, i, expected_shape, offset_in_shard, True
                )
                wroteback = True

            # Check for gradient writeback
            if self._skipped_use_sharded_views:
                # Skip the writeback check because we do not expose gradients
                # when we skipped using sharded views
                continue
            if param.grad is None and flat_param.grad is not None:
                expected_shape = torch.Size([numel_in_shard])
                self._writeback_tensor(
                    None, flat_param.grad, i, expected_shape, offset_in_shard, False
                )
            elif param.grad is not None:
                # For `NO_SHARD` + CPU offloading, `_cpu_grad` is always in
                # memory and owns the gradient storage, so it will never
                # require gradient writeback.
                if not self.uses_sharded_strategy and self._offload_params:
                    # Explicitly continue to handle the case of `no_sync()`,
                    # where `param.grad` is a view into the GPU gradient
                    # referenced by `flat_param.grad`, while `flat_param_grad`
                    # is `flat_param._cpu_grad`, which is on CPU
                    continue
                needs_grad_writeback = (
                    flat_param_grad is None
                    or not _same_storage_as_data_ptr(
                        param.grad, flat_param_grad_data_ptr
                    )
                )
                if needs_grad_writeback:
                    if flat_param_grad is None:
                        flat_param_grad = torch.zeros_like(flat_param)
                    expected_shape = torch.Size([numel_in_shard])
                    self._writeback_tensor(
                        param.grad,
                        flat_param_grad,
                        i,
                        expected_shape,
                        offset_in_shard,
                        False,
                    )
                    flat_param.grad = flat_param_grad
                    flat_param_grad = flat_param.grad
                    flat_param_grad_data_ptr = (
                        flat_param_grad.untyped_storage().data_ptr()
                    )
        # TODO: If we want to handle shared parameters, we need to re-generate
        # the shared parameter data structures in case sharedness changed.
        for i, (
            param_name,
            module,
            _,
            prim_param_name,
            prim_module,
            _,
        ) in enumerate(flat_param._shared_param_infos):
            if getattr(module, param_name) is not getattr(prim_module, prim_param_name):
                raise NotImplementedError(
                    "Changing shared parameters is not supported yet"
                )
        return wroteback

    def _writeback_tensor(
        self,
        src_tensor: Optional[Tensor],
        dst_tensor: Tensor,
        tensor_index: int,
        expected_shape: torch.Size,
        offset: int,
        is_param: bool,  # else gradient
    ) -> None:
        """
        Writes back ``src_tensor`` to ``dst_tensor`` at offset ``offset``,
        where ``src_tensor`` should have shape ``expected_shape``. ``is_param``
        indicates if the tensor is the parameter (if ``True``) or gradient (if
        ``False``). If ``src_tensor`` is ``None``, then the effect is zeroing
        instead of copying. ``tensor_index`` gives the index of ``src_tensor``
        in the metadata structures.

        Raises:
            RuntimeError: If the ``src_tensor`` does not have the expected
            shape.
        """
        _p_assert(
            len(expected_shape) == 1,
            f"Expects a 1D expected shape but got {expected_shape}",
        )
        if self._debug_level == dist.DebugLevel.INFO:
            rank = self.rank if hasattr(self, "rank") else dist.get_rank()
            src_shape = src_tensor.shape if src_tensor is not None else None
            src_device = src_tensor.device if src_tensor is not None else None
            warnings.warn(
                f"[Rank {rank}] {'Parameter' if is_param else 'Gradient'} needs "
                f"writeback in {self._training_state}\n"
                f"expected shape={expected_shape} shape={src_shape} "
                f"expected device={dst_tensor.device} device={src_device}"
            )
        if src_tensor is not None and src_tensor.shape != expected_shape:
            # NOTE: Gradient shape mismatch is not possible in practice since
            # the gradient shape is enforced to match that of the parameter and
            # we already check for parameter shape mismatch.
            raise RuntimeError(
                f"Cannot writeback when the {'parameter' if is_param else 'gradient'} "
                f"shape changes\nExpects {expected_shape} but got {src_tensor.shape}"
            )
        if src_tensor is not None:
            dst_tensor[offset : offset + expected_shape.numel()].copy_(src_tensor)
        else:
            dst_tensor[offset : offset + expected_shape.numel()].zero_()
            assert self.flat_param._is_grad_none_mask is not None
            self.flat_param._is_grad_none_mask[tensor_index] = True

    def _reset_flat_param_grad_info_if_needed(self):
        """
        When ``use_orig_params=True``:
        (1) sets the underlying ``flat_param.grad`` to ``None`` if *all* of the
        original parameters' ``.grad`` are ``None``, and
        (2) sets ``flat_param.requires_grad=False`` if *none* of the original
        parameters require gradient.
        For (1), this is targeting ``optim.zero_grad(set_to_none=True)``, in
        which case we want to free the gradients as soon after the
        ``zero_grad()`` call as possible.
        """
        if not self._use_orig_params:
            return
        flat_param = self.flat_param
        assert flat_param._params is not None  # mypy
        all_grad_none = True
        requires_grad = False
        for param in flat_param._params:
            all_grad_none &= param.grad is None
            requires_grad |= param.requires_grad
        if all_grad_none:
            flat_param.grad = None
        # As long as one parameter requires gradient, then the flat parameter
        # must require gradient
        flat_param.requires_grad = requires_grad

    def _deregister_orig_params(self):
        for param_info in self.flat_param._param_infos:
            param_name, module, _ = param_info
            if hasattr(module, param_name):
                delattr(module, param_name)
        for param_name, module, _, _, _, _ in self.flat_param._shared_param_infos:
            if hasattr(module, param_name):
                delattr(module, param_name)

    ###########
    # HELPERS #
    ###########
    def flat_param_to(self, *args, **kwargs):
        """Wraps an in-place call to ``.to()`` for ``self.flat_param``."""
        self.flat_param.data = self.flat_param.to(*args, **kwargs)
        if self._use_orig_params:
            # Refresh the views because their storage may have changed
            if self.is_sharded(self.flat_param):
                self._use_sharded_views()
            else:
                self._use_unsharded_views(as_params=True)

    def _get_modules(self) -> Set[nn.Module]:
        """
        Returns a :class:`set` of the modules whose parameters are included
        in this handle's flat parameter.
        """
        return {pi.module for pi in self.flat_param._param_infos}.union(
            {spi.module for spi in self.flat_param._shared_param_infos}
        )

    def is_sharded(self, tensor: Tensor) -> bool:
        """
        Returns if ``tensor`` is *currently* sharded. For ``NO_SHARD``, we
        choose to have this always return ``False`` for clarity.
        """
        if (
            not hasattr(self.flat_param, "_sharded_size")
            or not self.uses_sharded_strategy
        ):
            # `_sharded_size` is defined iff `handle.shard()` has been called
            return False
        sharded_size = self.flat_param._sharded_size  # type: ignore[attr-defined]
        return tensor.size() == sharded_size

    def param_module_names(self) -> Iterator[Tuple[str, str]]:
        shared_param_infos = [
            ParamInfo(param_name, module, module_name)
            for (
                param_name,
                module,
                module_name,
                _,
                _,
                _,
            ) in self.flat_param._shared_param_infos
        ]
        for param_info in chain(self.flat_param._param_infos, shared_param_infos):
            param_name, _, module_name = param_info  # type: ignore[misc]
            yield (param_name, module_name)

    def shared_param_module_names(self) -> Iterator[Tuple[str, str]]:
        for param_name, _, module_name in [
            ParamInfo(param_name, module, module_name)
            for (
                param_name,
                module,
                module_name,
                _,
                _,
                _,
            ) in self.flat_param._shared_param_infos
        ]:
            yield (param_name, module_name)

    @property
    def _fqns_in_shard(self) -> List[str]:
        """Returns the FQNs of the parameters present in this rank's shard."""
        fqns_in_shard: List[str] = []
        for fqn, shard_param_info in zip(
            self.flat_param._fqns, self.flat_param._shard_param_infos  # type: ignore[attr-defined]
        ):
            if shard_param_info.in_shard:
                fqns_in_shard.append(fqn)
        return fqns_in_shard

    @property
    def sharded_grad(self) -> Optional[Tensor]:
        """Returns the handle's sharded gradient."""
        flat_param = self.flat_param
        # Priority for non-`None`: `_cpu_grad` > `_saved_grad_shard` > `grad`
        # - CPU offloading: `_cpu_grad`
        # - No CPU offloading + sharded strategies: `_saved_grad_shard`
        # - No CPU offloading + `NO_SHARD`: `grad`
        grad: Optional[Tensor]
        if hasattr(flat_param, "_cpu_grad"):
            grad = flat_param._cpu_grad  # type: ignore[attr-defined]
        elif hasattr(flat_param, "_saved_grad_shard"):
            # In the post-backward hook, the sharded gradient is still in
            # `_saved_grad_shard`.
            grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
        else:
            # If in IDLE or in FORWARD states, then there may be an
            # (accumulated) gradient. If accessed in IDLE, then this should
            # be due to re-registering the original parameters (e.g. in state
            # dict load).
            _p_assert(
                flat_param.grad is None
                or not self.uses_sharded_strategy
                or self._training_state
                in (HandleTrainingState.FORWARD, HandleTrainingState.IDLE),
                "Sharded strategies should use `_cpu_grad` or `_saved_grad_shard` "
                "unless in IDLE or FORWARD",
            )
            grad = flat_param.grad
        return grad

    def _reset_is_grad_none(self) -> None:
        """
        Resets ``_is_grad_none_mask`` as needed. This method should only be
        called in the post-backward after gradient computation, in which case
        if a parameter requires gradient, then it will surely receive a
        gradient and we may reset its mask entry to ``False``.
        """
        if not self._use_orig_params:
            return
        _p_assert(
            self._training_state == HandleTrainingState.BACKWARD_POST,
            "Expects to only be called in the post-backward after gradient computation",
        )
        flat_param = self.flat_param
        assert flat_param._params is not None  # mypy
        for i, param in enumerate(flat_param._params):  # type: ignore[arg-type]
            # As long as the parameter requires gradient, it should receive a
            # meaningful gradient (even if the gradient happens to be zeros)
            if param.requires_grad:
                assert flat_param._is_grad_none_mask is not None  # mypy
                flat_param._is_grad_none_mask[i] = False

    #######################
    # CHECKS & INVARIANTS #
    #######################
    def _check_sharded_strategy(self):
        _p_assert(self.uses_sharded_strategy, "Expects sharded strategy")

    def _check_on_compute_device(self, tensor: Tensor):
        _p_assert(
            tensor.device == self.device,
            f"Expects tensor to be on the compute device {self.device}",
        )

    def _check_on_cpu(self, tensor: Tensor):
        _p_assert(
            tensor.device == torch.device("cpu"),
            f"Expects tensor to be on CPU but got {tensor.device}",
        )

    @staticmethod
    def _check_storage_freed(tensor: Tensor):
        storage_size: int = tensor._typed_storage()._size()
        _p_assert(
            storage_size == 0,
            f"Expects storage to be freed but got storage with size {storage_size}",
        )

    @staticmethod
    def _check_storage_allocated(tensor: Tensor):
        storage_size: int = tensor._typed_storage()._size()
        _p_assert(storage_size > 0, "Expects storage to be allocated")

    def _check_low_precision_shard(self):
        _p_assert(
            self._uses_param_mixed_precision,
            "Not using low precision for parameters",
        )
        _p_assert(
            getattr(self.flat_param, "_mp_shard", None) is not None,
            "Expects `_mp_shard` to exist",
        )
        device = self.flat_param._mp_shard.device  # type: ignore[attr-defined]
        _p_assert(
            device == self.device,
            f"Expects the low precision shard to be on {self.device} but got {device}",
        )

    def _check_unsharded(self, tensor: Tensor):
        msg_prefix = "Expects tensor to be unsharded "
        _p_assert(tensor is not None, msg_prefix + "but got `None`")
        unsharded_size = self.flat_param._unpadded_unsharded_size
        _p_assert(
            tensor.size() == unsharded_size,
            msg_prefix + f"with size {unsharded_size} but got {tensor.size()}",
        )

    def _check_sharded(self, tensor: Tensor):
        msg_prefix = "Expects tensor to be sharded "
        _p_assert(tensor is not None, msg_prefix + "but got `None`")
        sharded_size = self.flat_param._sharded_size  # type: ignore[attr-defined]
        _p_assert(
            tensor.size() == sharded_size,
            msg_prefix + f"with size {sharded_size} but got {tensor.size()}",
        )

    ##############
    # PROPERTIES #
    ##############
    @property
    def uses_sharded_strategy(self) -> bool:
        return self._sharding_strategy != HandleShardingStrategy.NO_SHARD

    @property
    def _uses_param_mixed_precision(self) -> bool:
        return self._fwd_bwd_param_dtype != self._orig_param_dtype

    @property
    def _uses_reduce_mixed_precision(self) -> bool:
        return self._reduce_dtype != self._orig_param_dtype

    @property
    def _force_full_precision(self) -> bool:
        return (
            self._uses_param_mixed_precision or self._uses_reduce_mixed_precision
        ) and (
            self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS
            or
            # Also disable mixed precision in model eval mode, if configured
            (not self._fully_sharded_module.training and self._use_full_prec_in_eval)
        )

    @property
    def _skipped_use_sharded_views(self) -> bool:
        """
        This property is used for sharding strategies that do not free after
        forward with ``use_orig_params=True``. This returns if this handle is
        currently in a state where it has skipped using sharded views, in which
        case it can restore view invariants via ``_use_sharded_views()``.
        """
        return self._unsharded_flat_param_for_skipped_views is not None


# NOTE: These are hacks to bypass `nn.Module.__setattr__` checks.
def _unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None:
    module._parameters[param_name] = param
    # This bypasses any overrides in case `module` is an instance of an
    # `nn.Module` subclass
    super(nn.Module, module).__setattr__(param_name, param)


def _unsafe_setattr_tensor(module: nn.Module, param_name: str, tensor: Tensor) -> None:
    module._parameters.pop(param_name, None)
    # This bypasses any overrides in case `module` is an instance of an
    # `nn.Module` subclass
    super(nn.Module, module).__setattr__(param_name, tensor)


def _safe_setattr_tensor_or_param(
    module: nn.Module, param_name: str, tensor_or_param: Union[Tensor, nn.Parameter]
):
    # Call `delattr()` and `setattr()` to go through `nn.Module` checks
    if hasattr(module, param_name):
        delattr(module, param_name)
    setattr(module, param_name, tensor_or_param)


def _convert_to_params(
    tensors: List[Union[torch.Tensor, nn.Parameter]]
) -> List[nn.Parameter]:
    return [t if isinstance(t, nn.Parameter) else nn.Parameter(t) for t in tensors]


def _detach_if_needed(param_or_tensor: Union[nn.Parameter, Tensor]) -> Tensor:
    return (
        param_or_tensor.detach()
        if isinstance(param_or_tensor, nn.Parameter)
        else param_or_tensor
    )


def _get_aligned_numel(unsharded_dtype: torch.dtype):
    # NOTE: This alignment constraint comes from TorchInductor.
    ALIGNMENT = 16  # bytes
    unsharded_dtype_size = _get_dtype_size(unsharded_dtype)
    aligned_numel = ALIGNMENT // unsharded_dtype_size
    return aligned_numel


@functools.lru_cache(8)
def _get_dtype_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def _construct_padding_tensor(
    padding_numel: int, dtype: torch.dtype, requires_grad: bool, device: torch.device
):
    # NOTE: Set the padding value as a magic number for debuggability. The
    # value itself should never be used in any user-facing computation.
    return (
        torch.ones(
            (padding_numel,), dtype=dtype, requires_grad=requires_grad, device=device
        )
        * _FLAT_PARAM_PADDING_VALUE
    )


# Use `lru_cache(1)` to only log the warning once (assuming the fixed warning
# messasge is passed in)
@functools.lru_cache(1)
def _warn_skip_writeback_check(log: logging.Logger, warning: str):
    log.warning(warning)
