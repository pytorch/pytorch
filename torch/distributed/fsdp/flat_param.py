import contextlib
from dataclasses import dataclass
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
    Any,
    cast,
    Dict,
    Generator,
    Iterator,
    List,
    NamedTuple,
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

from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
from ._utils import _alloc_storage, _free_storage, _set_fsdp_flattened, p_assert

__all__ = [
    "FlatParameter",
    "FlatParamHandle",
    "FlatParamShardMetadata",
    "ParamInfo",
    "SharedParamInfo",
    "HandleConfig",
    "HandleShardingStrategy",
    "HandleTrainingState",
]


class ParamInfo(NamedTuple):
    """Information for an original module parameter."""

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


class FlatParamShardMetadata(NamedTuple):
    """
    This holds metadata specific to this rank's shard of the flattened
    parameter.

    Attributes:
        param_names (Tuple[str, ...]): Prefixed parameter names of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_shapes (Tuple[torch.Size, ...]): Parameter shapes of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_numels (Tuple[int, ...]): Parameter numels of this rank's shard
            of the parameters; see :class:`FlatParameter`.
        param_offsets (Tuple[Tuple[int, int], ...]): [start, end] offsets (in
            units of numels) giving this rank's part of each flattened
            original module parameter.
    """

    param_names: Tuple[str, ...]
    param_shapes: Tuple[torch.Size, ...]
    param_numels: Tuple[int, ...]
    param_offsets: Tuple[Tuple[int, int], ...]


# TODO (awgu): Prefix these with "Handle" for now to avoid circular imports and
# inadvertent misuses; coalesce with those in fully_sharded_data_parallel.py
# later
class HandleShardingStrategy(Enum):
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()


class HandleTrainingState(Enum):
    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


@dataclass
class HandleConfig:
    sharding_strategy: HandleShardingStrategy
    offload_params: bool
    param_dtype: Optional[torch.dtype]
    reduce_dtype: Optional[torch.dtype]
    keep_low_precision_grads: Optional[bool] = False


class FlatParameter(nn.Parameter):
    """
    This is the flattened parameter used by :class:`FullyShardedDataParallel`.
    It is comprised of one or more original parameters, which are flattened
    and concatenated to construct the flattened parameter.

    Under the current design, this parameter logically represents both the
    unsharded and sharded flattened parameter, and its data changes storages
    dynamically.
        - In the :class:`FullyShardedDataParallel` constructor, the parameter
        is initialized as unsharded and then sharded in-place.
        - At runtime, the parameter is lazily (re)-initialized. The sharded
        parameter data is saved in ``self._local_shard``, and a new ``Tensor``
        ``self._full_param_padded`` is created, which is the all-gather
        destination and owns the unsharded parameter storage thereafter. (See
        :meth:`FullyShardedDataParallel._init_param_attributes`.)
        - Throughout runtime, the parameter data changes storages as needed,
        e.g. to the sharded flattened parameter, reduced-precision sharded
        flattened parameter, or the unsharded flattened parameter.

    Attributes:
        _unpadded_unsharded_size (torch.Size): Unsharded flattened parameter's
            size without padding.
        _padded_unsharded_size (torch.Size): Unsharded flattened parameter's
            size with padding. This is only set for sharded strategies since
            they require padding for the all-gather.

        _param_infos (Tuple[ParamInfo, ...]): Each parameter's parameter info
            entry; see :class:`ParamInfo`.
        _numels (Tuple[int, ...]): Each parameter's numel.
        _shapes (Tuple[torch.Size, ...]): Each parameter's shape.
        _prefixed_param_names (Tuple[str, ...]): Each parameter's name prefixed
            with the parent module names starting from the module passed to
            construct this flattened parameter via :class:`FlatParamHandle`;
            the prefixed names are guaranteed to be unique within the subtree
            rooted in that module.
        _num_params (int): Number of original parameters flattened into this
            flattened parameter; this is the length of ``_param_infos``,
            ``_numels``, ``_shapes``, and ``_prefixed_param_names``.
        _shared_param_infos (Tuple[SharedParamInfo, ...]): Shared parameter
            info entries; see :class:`SharedParamInfo`.
        _param_extensions (Tuple[Optional[Any], ...]): Parameter extensions
            (i.e. some per-parameter state) used to customize pre-flatten and
            post-unflatten behavior. This is experimental, and users should not
            depend on its existence in the future.

        _shard_param_offsets (List[Tuple[int, int])): [start, end] offsets (in
            units of numel) giving this rank's part of each flattened original
            module parameter; for any parameter ``p`` that is not sharded
            across ranks, this will be [0, ``p.numel()``-1].
        _shard_indices (Tuple[int, int]): [start, end] indices (in units of
            parameters) for this rank's shard of the original model parameters,
            where the parameters follow the order in which they were originally
            flattened; this indexes appropriately into any data structure that
            follows the flattening order (e.g. ``_param_infos``, ``_numels``,
            etc.).
        _shard_numel_padded (int): Numel padded for this rank's sharded
            flattened parameter.

        _local_shard (Tensor): Sharded flattened parameter with padding if
            using a sharded strategy. If using ``NO_SHARD``, then this is the
            unpadded unsharded flattened parameter, and there is no notion of a
            sharded flattened parameter or padded unsharded flattened
            parameter.
        _full_param_padded (Tensor): Unsharded flattened parameter with
            padding. This is not defined for ``NO_SHARD``. When using mixed
            precision for parameters, this has the low precision.
        _full_prec_full_param_padded (Tensor): Full precision unsharded
            flattened parameter with padding. This is used for unsharding
            outside of computation when using mixed precision for parameters.
            This is never defined for ``NO_SHARD``.
        _post_backward_hook_state (Tuple[AccumulateGrad, RemovableHandle]):
            Flattened parameter's :class:`AccumulateGrad` object and
            post-backward hook handle.
        _mp_shard (Tensor): Low precision sharded flattened parameter with
            padding. This is only defined when parameter mixed precision is
            enabled. For ``NO_SHARD``, this is used for computation.
        _cpu_grad (Tensor): Sharded gradient with padding stored on CPU.
            This is only defined when offloading parameters is enabled.
        _saved_grad_shard (Tensor): Sharded gradient with padding from previous
            iterations for gradient accumulation without :meth:`no_sync`.
    """

    def _init_metadata(
        self,
        param_infos: List[ParamInfo],
        numels: List[int],
        shapes: List[torch.Size],
        prefixed_param_names: List[str],
        shared_param_infos: List[SharedParamInfo],
        param_extensions: List[Any],
    ) -> None:
        """
        Initializes attributes holding metadata about the original parameters
        comprising the flattened parameter.

        We expose this method separate from the constructor to keep the
        constructor only responsible for the flattened parameter's tensor data.
        This method should only be called once per model, while the constructor
        may be called multiple times, e.g. when reloading from a checkpoint, in
        which case only the tensor data needs to be passed to the constructor.
        Since :meth:`load_state_dict` is implemented via :meth:`copy_`, the
        metadata is correctly assumed to be unchanged.

        Args:
            See the Attributes in the class docstring.
        """
        assert len(param_infos) == len(numels)
        assert len(param_infos) == len(shapes)
        assert len(param_infos) == len(prefixed_param_names)
        assert len(param_infos) == len(param_extensions)
        self._num_params = len(param_infos)
        self._param_infos = tuple(param_infos)
        self._numels = tuple(numels)
        self._shapes = tuple(shapes)
        self._prefixed_param_names = tuple(prefixed_param_names)
        self._shared_param_infos = tuple(shared_param_infos)
        self._param_extensions = tuple(param_extensions)
        self._unpadded_unsharded_size = self.size()
        _set_fsdp_flattened(self)


class FlatParamHandle:
    """
    This handle manages a flattened parameter (:class:`FlatParameter`). This
    includes sharding and view management.

    Args:
        params (Sequence[nn.Parameter]): The parameters to use for the
            flattened parameter.
        module (nn.Module): A module that is the root of the subtree containing
            all parameters in ``params``; for non-recursive wrapping, this must
            be the top-level module, while for recursive wrapping, this may not
            necessarily be the top-level module.
        device (torch.device): The compute and communication device, which
            should be a non-CPU device. We refer to it as the compute device.
        config (HandleConfig): A config customizing the handle based on FSDP's
            available features.
    """

    ##################
    # INITIALIZATION #
    ##################
    def __init__(
        self,
        params: Sequence[nn.Parameter],
        module: nn.Module,
        device: torch.device,
        config: HandleConfig,
    ) -> None:
        super().__init__()
        self.device = device
        self._config = config
        self._training_state = HandleTrainingState.IDLE
        self._init_flat_param(params, module)
        self._unflatten(as_params=False)

    def _init_flat_param(
        self,
        params: Sequence[Optional[nn.Parameter]],
        module: nn.Module,
    ) -> None:
        """
        Initializes the flattened parameter ``self.flat_param`` by flattening
        the parameters in ``params`` into a single :class:`FlatParameter` and
        saves relevant metadata. Shared parameters are only included in the
        flattened parameter once.

        This checks that all comprising parameters have the same dtype and
        ``requires_grad`` and does not support nested construction of
        :class:`FlatParameter` s.

        Args:
            See the Args in the class docstring.
        """
        params_set = set(params)
        params_set.discard(None)
        assert (
            len(params_set) > 0
        ), "Cannot initialize a `FlatParameter` from an empty parameter list"
        param_infos: List[ParamInfo] = []
        numels: List[int] = []
        shapes: List[torch.Size] = []
        prefixed_param_names: List[str] = []
        shared_param_infos: List[SharedParamInfo] = []
        shared_param_memo: Dict[nn.Parameter, Tuple[nn.Module, str, str]] = {}
        params_to_flatten: List[nn.Parameter] = []
        param_extensions: List[Any] = []
        dtype: Optional[torch.dtype] = None
        requires_grad: Optional[bool] = None
        for submodule_name, submodule in module.named_modules():
            for param_name, param in submodule.named_parameters(recurse=False):
                if param not in params_set:
                    continue
                if param in shared_param_memo:
                    prim_module, prim_module_name, prim_param_name = shared_param_memo[
                        param
                    ]
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
                    if type(param) is FlatParameter:
                        raise ValueError("`FlatParameter` does not support nesting")
                    if dtype is not None and param.dtype != dtype:
                        raise ValueError(
                            "`FlatParameter` requires uniform dtype but got "
                            f"{dtype} and {param.dtype}"
                        )
                    if dtype is None and not param.is_floating_point():
                        raise ValueError("Integer parameters are unsupported")
                    if (
                        requires_grad is not None
                        and param.requires_grad != requires_grad
                    ):
                        raise ValueError(
                            "`FlatParameter` requires uniform `requires_grad`"
                        )
                    param, extension = _ext_pre_flatten_transform(param)
                    param_extensions.append(extension)
                    dtype = param.dtype
                    requires_grad = param.requires_grad
                    shared_param_memo[param] = (submodule, submodule_name, param_name)
                    params_to_flatten.append(param)
                    param_infos.append(ParamInfo(param_name, submodule, submodule_name))
                    numels.append(param.numel())
                    shapes.append(param.shape)
                    prefixed_param_name = (
                        submodule_name + "." + param_name
                        if submodule_name
                        else param_name
                    )
                    prefixed_param_names.append(prefixed_param_name)
        assert requires_grad is not None
        self.flat_param = FlatParamHandle.flatten_params(
            params_to_flatten, requires_grad
        )
        self.flat_param._init_metadata(
            param_infos,
            numels,
            shapes,
            prefixed_param_names,
            shared_param_infos,
            param_extensions,
        )

    @staticmethod
    def flatten_params(
        params: Sequence[torch.Tensor],
        requires_grad: bool,
    ) -> FlatParameter:
        """
        Flattens the parameters in ``params`` into a single
        :class:`FlatParameter`. This should be the only way used to construct
        :class:`FlatParameter` s.

        We expose this factory method for checkpointing (e.g. sharded state
        dict). The flattened parameter's metadata should only be initialized
        once (see :meth:`_init_metadata`), but its tensor data may be reloaded.
        """
        with torch.no_grad():
            flat_params = [
                p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1)
                for p in params
            ]
            flat_param_data = torch.cat(flat_params, dim=0)
        flat_param = FlatParameter(flat_param_data, requires_grad=requires_grad)
        return flat_param

    ###################################
    # SHARD INITIALIZATION & METADATA #
    ###################################
    @torch.no_grad()
    def shard(self, process_group: dist.ProcessGroup):
        """
        Shards the handle's ``FlatParameter``. In terms of memory, this
        allocates new memory for the sharded flattened parameter and frees the
        unsharded flattened parameter's storage.

        Postcondition: ``self.flat_param`` is the sharded flattened parameter.
        ``process_group``, ``rank``, and ``world_size`` attributes are set.

        TODO (awgu): Once we retire ``FlattenParamsWrapper``, we should pass
        the process group directly to the ``FlatParamHandle`` constructor. For
        now, we decouple ``FlattenParamsWrapper` from a process group, but this
        makes the process-group-related attributes not necessarily defined.
        """
        if not self.uses_sharded_strategy:
            return
        flat_param = self.flat_param
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        assert (
            flat_param.storage_offset() == 0
        ), "The `FlatParameter` is not the sole occupant of its storage"
        orig_storage = flat_param.storage()
        local_shard, numel_padded = FlatParamHandle._get_shard(
            flat_param, self.rank, self.world_size
        )
        flat_param.set_(local_shard)  # type: ignore[call-overload]
        self._init_shard_metadata(local_shard.numel(), numel_padded, self.rank)
        if orig_storage.size() > 0:
            orig_storage.resize_(0)

    def _init_shard_metadata(
        self,
        sharded_flat_param_numel: int,
        numel_padded: int,
        rank: int,
    ) -> None:
        """
        Initializes shard-related metadata for this rank's shard of the
        flattened parameter: ``_shard_param_offsets``, ``_shard_indices``, and
        ``_shard_numel_padded``.

        Args:
            sharded_flat_param_numel (int): Numel of each rank's sharded
                flattened parameter with padding (i.e. including
                ``numel_padded``).
            numel_padded (int): Numel padded for this rank's sharded flattened
                parameter.
            rank (int): Caller's rank.
        """
        if numel_padded > sharded_flat_param_numel:
            raise ValueError(
                f"Sharded flattened parameter with {sharded_flat_param_numel} "
                f"numel cannot have {numel_padded} numel padded"
            )
        start = sharded_flat_param_numel * rank
        end = sharded_flat_param_numel * (rank + 1) - 1  # inclusive
        (
            self.flat_param._shard_param_offsets,  # type: ignore[attr-defined]
            self.flat_param._shard_indices,  # type: ignore[attr-defined]
        ) = self._get_shard_metadata(start, end)
        self.flat_param._shard_numel_padded = numel_padded  # type: ignore[attr-defined]

    def _get_shard_metadata(
        self,
        start: int,
        end: int,
    ) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[int, int]]:
        """
        Computes the shard metadata based on ``start`` and ``end``, which give
        the closed interval of the unsharded flattened parameter specifying the
        shard.

        Args:
            start (int): Start index (in units of numel) of this rank's shard
                of the flattened parameter.
            end (int): End index (in units of numel and inclusive) of this
                rank's shard of the flattened parameter.

        Return:
            Tuple[Tuple[Tuple[int, int], ...], Tuple[int, int]]: See
            ``_shard_param_offsets`` and ``_shard_indices`` in
            :class:`FlatParameter` 's docstring.
        """
        flat_param_offsets = self._get_flat_param_offsets()
        # Indices of the original parameters in this rank's sharded flattened
        # parameter
        shard_param_indices_range = []  # elements will be consecutive
        # [start, end] offsets giving this rank's part of the flattened
        # original module parameter (which will be [0, `p.numel()`-1] for any
        # parameter that is not sharded across ranks)
        shard_param_offsets = []
        for i, (param_start, param_end) in enumerate(flat_param_offsets):
            if start > param_end or end < param_start:
                continue
            if start <= param_start:
                intra_param_start = 0
            else:
                intra_param_start = start - param_start
            intra_param_end = min(param_end, end) - param_start
            shard_param_indices_range.append(i)
            shard_param_offsets.append(
                (intra_param_start, intra_param_end)
            )  # both inclusive
        if len(shard_param_indices_range) == 0:
            shard_param_indices = (0, 0)
            assert len(shard_param_offsets) == 0
        else:
            shard_param_indices = (
                shard_param_indices_range[0],
                shard_param_indices_range[-1],
            )
            assert (
                len(shard_param_offsets)
                == shard_param_indices[-1] - shard_param_indices[0] + 1
            )
        return tuple(shard_param_offsets), shard_param_indices

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
        """Returns [start, end] offsets of each original parameter's flattened
        data in the unsharded flattened parameter (without padding)."""
        cumulative_sum = list(accumulate(self.flat_param._numels))
        starts = [0] + cumulative_sum[:-1]
        ends = [end - 1 for end in cumulative_sum]  # inclusive
        param_offsets = list(zip(starts, ends))
        return param_offsets

    def shard_metadata(
        self,
    ) -> FlatParamShardMetadata:
        """Returns shard-related metadata specific to this rank's shard of the
        flattened parameter."""
        assert hasattr(self.flat_param, "_shard_indices") and hasattr(
            self.flat_param, "_shard_param_offsets"
        ), "Shard metadata has not been initialized"
        shard_param_start_index = self.flat_param._shard_indices[0]  # type: ignore[attr-defined]
        shard_param_end_index = self.flat_param._shard_indices[1]  # type: ignore[attr-defined]
        sl = (
            slice(shard_param_start_index, shard_param_end_index + 1)
            if shard_param_start_index <= shard_param_end_index
            else slice(0, 0)
        )
        return FlatParamShardMetadata(
            self.flat_param._prefixed_param_names[sl],
            self.flat_param._shapes[sl],
            self.flat_param._numels[sl],
            self.flat_param._shard_param_offsets[:],  # type: ignore[attr-defined]
        )

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
        ret = False
        if (
            self.uses_sharded_strategy
            and not self._config.offload_params
            and not self.needs_unshard()
        ):
            pass  # no-op
        elif self._uses_param_mixed_precision and not self._force_full_precision:
            self._use_low_precision_shard()
            ret = True
        elif self._config.offload_params and self.flat_param.device != self.device:
            # NOTE: This creates a new tensor distinct from any attributes.
            self._flat_param_to(self.device, non_blocking=True)
            ret = True
        self._check_on_compute_device(self.flat_param)
        return ret

    def _use_low_precision_shard(self):
        """
        Allocates the low precision shard directly on the compute device and
        switches to using the low precision sharded flattened parameter.
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
        Runs the unshard logic. This includes all-gathering the flattened
        parameter and switching to using the unsharded flattened parameter. If
        the handle does not need unsharding, then this only switches to using
        the unsharded flattened parameter. For ``NO_SHARD``, this is a no-op.

        If FSDP is in :meth:`summon_full_params` and the handle uses parameter
        mixed precision, then the parameter is forced to full precision.
        """
        if not self.needs_unshard():
            if self.uses_sharded_strategy:
                # The handle may have been resharded without freeing the padded
                # unsharded flattened parameter, in which case we need to
                # switch to using the unsharded parameter
                unsharded_flat_param = self._get_padded_unsharded_flat_param()
                self._use_unsharded_flat_param(unsharded_flat_param)
            return
        unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
        self._all_gather_flat_param(unsharded_flat_param)

    def needs_unshard(self) -> bool:
        """Returns if the handle's flattened parameter needs to be unsharded."""
        if not self.uses_sharded_strategy:
            return False
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        already_unsharded = (
            unsharded_flat_param.storage().size() == unsharded_flat_param.numel()
        )
        return not already_unsharded

    def _alloc_padded_unsharded_flat_param(self):
        """
        Allocates the *padded* unsharded flattened parameter. The unpadded
        unsharded flattened parameter is always a view into the padded one.
        This padded parameter is saved to a different attribute on the
        ``FlatParameter`` depending on if we force full precision.
        """
        self._check_sharded_strategy()
        flat_param = self.flat_param
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        self._check_storage_freed(unsharded_flat_param)
        _alloc_storage(unsharded_flat_param, flat_param._padded_unsharded_size)  # type: ignore[attr-defined]
        return unsharded_flat_param

    def _get_padded_unsharded_flat_param(self) -> torch.Tensor:
        """
        Returns a reference to the padded unsharded flattened parameter
        depending on the calling context. This should only be called if using a
        sharded strategy.
        """
        self._check_sharded_strategy()
        flat_param = self.flat_param
        if self._force_full_precision:
            # When parameter mixed precision is enabled, we use a different
            # tensor as the all-gather destination to preserve the invariant
            # that  `_full_param_padded` is in the low precision
            unsharded_flat_param = flat_param._full_prec_full_param_padded  # type: ignore[attr-defined]
            p_assert(
                unsharded_flat_param.dtype != self._config.param_dtype,
                f"Expects full precision but got {self._config.param_dtype}",
            )
        else:
            unsharded_flat_param = flat_param._full_param_padded  # type: ignore[attr-defined]
        return unsharded_flat_param

    def _all_gather_flat_param(
        self,
        padded_unsharded_flat_param: Tensor,
    ) -> None:
        """
        All-gathers the handle's flattened parameter to the destination
        ``padded_unsharded_flat_param``, and switches to using the all-gathered
        tensor.
        """
        p_assert(
            hasattr(self, "process_group") and hasattr(self, "world_size"),
            "Expects a process group and world size to have been set via `shard()`",
        )
        sharded_flat_param = self.flat_param.data
        expected_numel = sharded_flat_param.numel() * self.world_size
        p_assert(
            padded_unsharded_flat_param.numel() == expected_numel,
            f"Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}",
        )
        dist._all_gather_base(
            padded_unsharded_flat_param,
            sharded_flat_param,
            self.process_group,
        )
        self._use_unsharded_flat_param(padded_unsharded_flat_param)

    def _use_unsharded_flat_param(
        self,
        padded_unsharded_flat_param: torch.Tensor,
    ) -> None:
        """
        Switches to using the *unpadded* unsharded flattened parameter, which
        is a view into the *padded* unsharded flattened parameter.
        """
        unsharded_size = self.flat_param._unpadded_unsharded_size
        self.flat_param.data = padded_unsharded_flat_param[
            : unsharded_size.numel()
        ].view(unsharded_size)

    def post_unshard(self):
        """
        Runs the post-unshard logic. This includes freeing the low precision
        shard if needed.
        """
        if self._uses_param_mixed_precision and self.uses_sharded_strategy:
            self._free_low_precision_sharded_param()
        self._check_on_compute_device(self.flat_param)

    def _free_low_precision_sharded_param(self):
        """Frees the low precision sharded flattened parameter."""
        self._check_low_precision_shard()
        _free_storage(self.flat_param._mp_shard)  # type: ignore[attr-defined]

    def prepare_gradient(self):
        """
        Prepares the gradient for the backward computation by saving and
        clearing any existing sharded gradient in ``.grad`` to enable computing
        a new unsharded gradient.
        """
        p_assert(
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
            p_assert(
                not grad_offloaded or self._config.offload_params,
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
                    # If we're using mixed precision with keeping grads
                    # casted, gradient here might still be of the reduced
                    # dtype if we didn't clear / set the gradients to None
                    # after previous backward. In that case, make sure
                    # p._saved_grad_shard is cast to the full precision type
                    # so that we can accumulate in full precision in
                    # _post_backward_hook and assign back in full precision
                    # in _wait_for_post_backward.
                    if (
                        self._config.keep_low_precision_grads
                        and flat_param._saved_grad_shard.dtype  # type: ignore[attr-defined]
                        != flat_param._local_shard.dtype  # type: ignore[attr-defined]
                    ):
                        flat_param._saved_grad_shard = flat_param._saved_grad_shard.to(  # type: ignore[attr-defined]
                            flat_param._local_shard.dtype  # type: ignore[attr-defined]
                        )
            else:
                padded_unsharded_size = flat_param._padded_unsharded_size  # type: ignore[attr-defined]
                p_assert(
                    flat_param.grad.size() == padded_unsharded_size,
                    "Expects `.grad` to be the unsharded gradient in "
                    f"`no_sync()` with size {padded_unsharded_size} "
                    f"but got size {flat_param.grad.size()}",
                )
            flat_param.grad = None

    @contextlib.contextmanager
    def to_cpu(self):
        """
        Moves the unpadded unsharded flattened parameter to CPU while in the
        context and moves it back to the previous device upon exit. For now,
        this assumes the ``FlatParameter`` is the unpadded unsharded flattened
        parameter since (1) there is no reason to include the padding in the
        copy and (2) there is no use case for the sharded flattened parameter.

        Precondition: ``self.flat_param`` 's data is the unpadded unsharded
        flattened parameter on the compute device, and the handle uses a
        sharded strategy.
        Postcondition: Same as the precondition.
        """
        self._check_sharded_strategy()
        p_assert(
            self.flat_param.size() == self.flat_param._unpadded_unsharded_size,
            f"Expects size {self.flat_param._unpadded_unsharded_size} but got {self.flat_param.size()}",
        )
        self._check_on_compute_device(self.flat_param)
        # Check that the unpadded unsharded flattened parameter is a view into
        # the padded unsharded flattened parameter as expected
        # NOTE: This check is not strictly needed for correctness but is a
        # useful sanity check since the tensor should only be used internally.
        unpadded_storage_ptr = self.flat_param.storage().data_ptr()
        padded_storage_ptr = (
            self._get_padded_unsharded_flat_param().storage().data_ptr()
        )
        p_assert(
            unpadded_storage_ptr == padded_storage_ptr,
            "Expects the unpadded parameter to be a view into the padded parameter",
        )
        self._flat_param_to(torch.device("cpu"))
        self._free_unsharded_flat_param()
        try:
            yield
        finally:
            p_assert(
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
        Runs the reshard logic. This includes freeing the unsharded flattened
        parameter if ``free_unsharded_flat_param`` and switching to using the
        sharded flattened parameter.
        """
        if free_unsharded_flat_param:
            self._free_unsharded_flat_param()
        self._use_sharded_flat_param()

    def post_reshard(self):
        """
        Runs the post-reshard logic. This includes freeing any memory that
        can now be freed given that the ``FlatParameter`` points to the full
        precision sharded flattened parameter.

        Precondition: ``self.flat_param`` 's data points to the full precision
        sharded flattened parameter.
        """
        # For `NO_SHARD`, `_mp_shard` is not freed in the post-unshard since
        # it is also the low precision *unsharded* flattened parameter. Hence,
        # we delay the free until the reshard.
        if (
            self._uses_param_mixed_precision
            and not self.uses_sharded_strategy
            and not self._force_full_precision  # did not use the low precision shard
        ):
            self._free_low_precision_sharded_param()

    def _free_unsharded_flat_param(self):
        """
        Frees the padded unsharded flattened parameter. The tensor to free
        depends on the calling context since the unshard may have forced full
        precision, in which case a different tensor is used.
        """
        self._check_sharded_strategy()
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        self._check_storage_allocated(unsharded_flat_param)
        self._check_on_compute_device(unsharded_flat_param)
        # Do not free the memory until all ops in the current stream finish
        unsharded_flat_param.record_stream(
            cast(torch._C.Stream, torch.cuda.current_stream())
        )
        _free_storage(unsharded_flat_param)

    def _use_sharded_flat_param(self) -> None:
        """Switches to using the sharded flattened parameter."""
        flat_param = self.flat_param
        if self._config.offload_params:
            device = flat_param._local_shard.device  # type: ignore[attr-defined]
            p_assert(
                device == torch.device("cpu"),
                f"Expects the local shard to be on CPU but got {device}",
            )
        flat_param.data = flat_param._local_shard  # type: ignore[attr-defined]

    #########
    # VIEWS #
    #########
    @staticmethod
    def _get_unflat_views(
        flat_param: FlatParameter,
        tensor: Optional[torch.Tensor] = None,
    ) -> Iterator[Tensor]:
        """
        Returns unflattened ``Tensor`` views into ``tensor`` if it is not
        ``None`` or ``flat_param`` otherwise, where the unflattening is based
        on ``flat_param`` 's metadata.

        In other words, to get views into the unsharded flattened parameter,
        pass ``tensor`` as ``None``, but to get views into tensor optimizer
        state, pass ``tensor`` as the optimizer state tensor.
        """
        if tensor is None:
            tensor = flat_param
        p_assert(
            tensor.numel() == flat_param._unpadded_unsharded_size.numel(),
            f"Expects {flat_param._unpadded_unsharded_size.numel()} numel but got "
            f"{tensor.numel()} numel",
        )
        views = (
            _ext_post_unflatten_transform(subtensor.view(shape), param_extension)
            for (subtensor, shape, param_extension) in zip(
                torch.split(tensor, flat_param._numels, dim=0),  # type: ignore[arg-type]
                flat_param._shapes, flat_param._param_extensions,
            )
        )
        return views

    def _unflatten(self, as_params: bool) -> None:
        """
        Unflattens the unsharded flattened parameter by setting the original
        module parameter variables to be views into it.

        Args:
            as_params (bool): If ``True``, then registers the original
                parameters as ``nn.Parameter`` s; if ``False``, then registers
                the original parameters only as ``Tensor`` s. ``False`` should
                be used during forward/backward computation and when hiding the
                original parameters from :meth:`nn.Module.named_parameters`.
        """
        views = self._get_unflat_views(self.flat_param)
        for view, (param_name, module, _) in zip(views, self.flat_param._param_infos):
            if hasattr(module, param_name):
                delattr(module, param_name)
            if as_params:
                module.register_parameter(param_name, nn.Parameter(view))
            else:
                setattr(module, param_name, view)
        for (
            param_name,
            module,
            _,
            prim_param_name,
            prim_module,
            _,
        ) in self.flat_param._shared_param_infos:
            if hasattr(module, param_name):
                delattr(module, param_name)
            assert hasattr(prim_module, prim_param_name)
            param: Union[Tensor, nn.Parameter] = getattr(prim_module, prim_param_name)
            if as_params:
                assert isinstance(param, nn.Parameter)
                module.register_parameter(param_name, param)
            else:
                setattr(module, param_name, param)

    @contextlib.contextmanager
    def unflatten_as_params(self) -> Generator:
        """
        Assumes the flattened parameter is unsharded. When in the context,
        unflattens the original parameters as ``nn.Parameter`` views into the
        flattened parameter, and after the context, restores the original
        parameters as ``Tensor`` views into the flattened parameter.
        """
        self._unflatten(as_params=True)
        try:
            yield
        finally:
            self._unflatten(as_params=False)

    ###########
    # HELPERS #
    ###########
    def _flat_param_to(self, *args, **kwargs):
        """Wraps an in-place call to ``.to()`` for ``self.flat_param``."""
        self.flat_param.data = self.flat_param.to(*args, **kwargs)

    def _get_modules(self) -> Set[nn.Module]:
        """Returns a :class:`set` of the modules whose parameters are included
        in this handle's flattened parameter."""
        return set(pi.module for pi in self.flat_param._param_infos).union(
            set(spi.module for spi in self.flat_param._shared_param_infos)
        )

    def parameter_module_names(self) -> Iterator[Tuple[str, str]]:
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
        for param_name, _, module_name in chain(
            self.flat_param._param_infos, shared_param_infos
        ):
            yield (param_name, module_name)

    #######################
    # CHECKS & INVARIANTS #
    #######################
    def _check_sharded_strategy(self):
        p_assert(self.uses_sharded_strategy, "Expects sharded strategy")

    def _check_on_compute_device(self, tensor: Tensor):
        p_assert(
            tensor.device == self.device,
            f"Expects tensor to be on the compute device {self.device}",
        )

    @staticmethod
    def _check_storage_freed(tensor: Tensor):
        storage_size: int = tensor.storage().size()
        p_assert(
            storage_size == 0,
            f"Expects storage to be freed but got storage with size {storage_size}",
        )

    @staticmethod
    def _check_storage_allocated(tensor: Tensor):
        storage_size: int = tensor.storage().size()
        p_assert(storage_size > 0, "Expects storage to be allocated")

    def _check_low_precision_shard(self):
        p_assert(
            self._uses_param_mixed_precision,
            "Not using low precision for parameters",
        )
        p_assert(
            getattr(self.flat_param, "_mp_shard", None) is not None,
            "Expects `_mp_shard` to exist",
        )
        device = self.flat_param._mp_shard.device  # type: ignore[attr-defined]
        p_assert(
            device == self.device,
            f"Expects the low precision shard to be on {self.device} but got {device}",
        )

    ##############
    # PROPERTIES #
    ##############
    @property
    def uses_sharded_strategy(self) -> bool:
        return self._config.sharding_strategy != HandleShardingStrategy.NO_SHARD

    @property
    def _uses_param_mixed_precision(self) -> bool:
        return self._config.param_dtype is not None

    @property
    def _force_full_precision(self) -> bool:
        return (
            self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS
            and self._uses_param_mixed_precision
        )
