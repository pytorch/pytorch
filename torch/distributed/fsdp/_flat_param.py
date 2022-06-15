import contextlib
from itertools import accumulate
from typing import (
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
import torch.nn as nn
from torch import Tensor

from ._utils import _get_param_to_param_name


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
        _is_sharded (bool): Whether the flattened parameter is *ever* sharded
            across ranks (not whether it is *currently* sharded).
        _unsharded_size (torch.Size): Unsharded flattened parameter's size.
        _flat_param_name (str): Uniquely-identifying name for the flattened
            parameter.

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

        _local_shard (Tensor): Sharded flattened parameter with padding.
        _full_param_padded (Tensor): Unsharded flattened parameter with
            padding.
        _shard_bwd_hook (Tuple[AccumulateGrad, RemovableHandle]): Flattened
            parameter's :class:`AccumulateGrad` object and post-backward hook
            handle.
        _mp_shard (Tensor): Reduced-precision flattened parameter with padding.
        _cpu_grad (Tensor): Sharded gradient with padding stored on CPU.
        _saved_grad_shard (Tensor): Sharded gradient with padding from previous
            iterations for gradient accumulation without :meth:`no_sync`.
    """

    def init_metadata(
        self,
        param_infos: List[ParamInfo],
        numels: List[int],
        shapes: List[torch.Size],
        prefixed_param_names: List[str],
        shared_param_infos: List[SharedParamInfo],
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
        self._num_params = len(param_infos)
        self._param_infos = tuple(param_infos)
        self._numels = tuple(numels)
        self._shapes = tuple(shapes)
        self._prefixed_param_names = tuple(prefixed_param_names)
        self._shared_param_infos = tuple(shared_param_infos)
        self._is_sharded = False
        self._unsharded_size = self.size()

    @property
    def _flat_param_name(self) -> str:
        """
        Returns a name for the flattened parameter that uniquely identifies
        it in a module hierarchy wrapped by a top-level FSDP instance. The name
        is constructed from the fully-prefixed names of the original parameters
        comprising the flattened parameter with "." replaced with "_" since
        parameter names cannot contain ".".
        """
        return ",".join(self._prefixed_param_names).replace(".", "_")


class FlatParamHandle:
    """
    This handle manages a flattened parameter (:class:`FlatParameter`).

    Args:
        params (Sequence[nn.Parameter]): The parameters to use for the
            flattened parameter.
        module (nn.Module): A module that is the root of the subtree containing
            all parameters in ``params``; for non-recursive wrapping, this must
            be the top-level module, while for recursive wrapping, this may not
            necessarily be the top-level module.
    """
    def __init__(
        self,
        params: Sequence[nn.Parameter],
        module: nn.Module,
    ) -> None:
        super().__init__()
        self._init_flat_param(module, params)
        self._unflatten(as_params=False)

    def _init_flat_param(
        self,
        module: nn.Module,
        params: Sequence[Optional[nn.Parameter]],
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
        assert len(params_set) > 0
        param_infos: List[ParamInfo] = []
        numels: List[int] = []
        shapes: List[torch.Size] = []
        prefixed_param_names: List[str] = []
        shared_param_infos: List[SharedParamInfo] = []
        shared_param_memo: Dict[nn.Parameter, Tuple[nn.Module, str, str]] = {}
        params_to_flatten: List[nn.Parameter] = []
        dtype: Optional[torch.dtype] = None
        requires_grad: Optional[bool] = None
        param_to_param_name = _get_param_to_param_name(module)
        for submodule_name, submodule in module.named_modules():
            for param_name, param in submodule.named_parameters(recurse=False):
                if param not in params_set:
                    continue
                if param in shared_param_memo:
                    prim_module, prim_module_name, prim_param_name = shared_param_memo[param]
                    shared_param_infos.append(SharedParamInfo(
                        param_name, submodule, submodule_name, prim_param_name,
                        prim_module, prim_module_name,
                    ))
                else:
                    if isinstance(param, FlatParameter):
                        raise ValueError("`FlatParameter` does not support nesting")
                    if dtype is not None and param.dtype != dtype:
                        raise ValueError(
                            "`FlatParameter` requires uniform dtype but got "
                            f"{dtype} and {param.dtype}"
                        )
                    if requires_grad is not None and param.requires_grad != requires_grad:
                        raise ValueError("`FlatParameter` requires uniform `requires_grad`")
                    dtype = param.dtype
                    requires_grad = param.requires_grad
                    shared_param_memo[param] = (submodule, submodule_name, param_name)
                    params_to_flatten.append(param)
                    param_infos.append(ParamInfo(param_name, submodule, submodule_name))
                    numels.append(param.numel())
                    shapes.append(param.shape)
                    prefixed_param_names.append(param_to_param_name[param])
        assert requires_grad is not None
        self.flat_param = FlatParamHandle.flatten_params(params_to_flatten, requires_grad)
        self.flat_param.init_metadata(
            param_infos, numels, shapes, prefixed_param_names, shared_param_infos,
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
        once (see :meth:`init_metadata`), but its tensor data may be reloaded.
        """
        with torch.no_grad():
            flat_params = [
                p.detach().reshape(-1) if isinstance(p, nn.Parameter)
                else p.reshape(-1) for p in params
            ]
            flat_param_data = torch.cat(flat_params, dim=0)
        flat_param = FlatParameter(flat_param_data, requires_grad=requires_grad)
        return flat_param

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
        assert tensor.numel() == flat_param._unsharded_size.numel(), \
            f"Expects {flat_param._unsharded_size.numel()} numel but got " \
            f"{tensor.numel()} numel"
        views = (
            tensor.view(shape) for (tensor, shape) in
            zip(torch.split(tensor, flat_param._numels, dim=0), flat_param._shapes)  # type: ignore[arg-type]
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
        for (param_name, module, _, prim_param_name, prim_module, _) in self.flat_param._shared_param_infos:
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

    def init_shard_metadata(
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
            sharded_flat_param_numel (int): Numel of this rank's sharded
                flattened parameter.
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
        self.flat_param._shard_param_offsets, self.flat_param._shard_indices = (  # type: ignore[attr-defined]
            self._get_shard_metadata(start, end)
        )
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
            shard_param_offsets.append((intra_param_start, intra_param_end))  # both inclusive
        if len(shard_param_indices_range) == 0:
            shard_param_indices = (0, 0)
            assert len(shard_param_offsets) == 0
        else:
            shard_param_indices = (
                shard_param_indices_range[0], shard_param_indices_range[-1],
            )
            assert len(shard_param_offsets) == \
                shard_param_indices[-1] - shard_param_indices[0] + 1
        return tuple(shard_param_offsets), shard_param_indices

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
        assert hasattr(self.flat_param, "_shard_indices") and \
            hasattr(self.flat_param, "_shard_param_offsets"), \
            "Shard metadata has not been initialized"
        shard_param_start_index = self.flat_param._shard_indices[0]  # type: ignore[attr-defined]
        shard_param_end_index = self.flat_param._shard_indices[1]  # type: ignore[attr-defined]
        sl = slice(shard_param_start_index, shard_param_end_index + 1) \
            if shard_param_start_index <= shard_param_end_index else slice(0, 0)
        return FlatParamShardMetadata(
            self.flat_param._prefixed_param_names[sl],
            self.flat_param._shapes[sl],
            self.flat_param._numels[sl],
            self.flat_param._shard_param_offsets[:],  # type: ignore[attr-defined]
        )

    def _get_modules(self) -> Set[nn.Module]:
        """Returns a :class:`set` of the modules whose parameters are included
        in this handle's flattened parameter."""
        return set(pi.module for pi in self.flat_param._param_infos).union(
            set(spi.module for spi in self.flat_param._shared_param_infos)
        )
