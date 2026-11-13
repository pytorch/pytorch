# mypy: allow-untyped-defs
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import auto, Enum
from typing import Any, cast, TYPE_CHECKING


if TYPE_CHECKING:
    from ._fsdp_api import DataParallelMeshDims
    from ._fsdp_engine import FullyShardBucket

import torch
import torch.distributed as dist
import torch.nn as nn


if dist._is_spmd_types_available():
    import spmd_types as spmd
    from spmd_types._mesh_axis import flatten_axes
    from spmd_types.runtime import get_partition_spec
    from spmd_types.types import partition_spec_to_shard_types

from torch._prims_common import make_contiguous_strides_for
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp._fully_shard._fsdp_common import DDPMeshInfo
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.placement_types import _StridedShard, Placement

from .._flex_shard.placements.shard import Shard as EngineShard
from ._fsdp_api import MixedPrecisionPolicy
from ._fsdp_common import (
    _from_local_no_grad,
    _raise_assert_with_print,
    DataParallelMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
    resolve_shard_placement,
    ShardPlacementFnResult,
)


_orig_param_uid_counter = itertools.count()


def _get_orig_param_uid(param: nn.Parameter) -> int:
    if not hasattr(param, "_fsdp_orig_uid"):
        uid = next(_orig_param_uid_counter)
        param._fsdp_orig_uid = uid  # pyrefly: ignore[missing-attribute]
    return param._fsdp_orig_uid  # pyrefly: ignore[missing-attribute]


def _spans_same_mesh(lhs_axes: tuple[Any, ...], rhs_axes: tuple[Any, ...]) -> bool:
    """Return whether two axis sets span the same normalized rank mesh."""
    lhs_mesh = spmd.normalize_mesh(frozenset(lhs_axes))
    rhs_mesh = spmd.normalize_mesh(frozenset(rhs_axes))
    if not lhs_mesh or not rhs_mesh:
        return lhs_mesh == rhs_mesh
    return flatten_axes(tuple(lhs_mesh)) == flatten_axes(tuple(rhs_mesh))


"""
[Note: FSDP tensors]
FSDP considers the following tensors:
- Original parameter: parameter passed to :class:`FSDPParam`, i.e. the one
  on the module when applying FSDP
- Sharded parameter: sharding the original parameter on dim-0 (or a
  user-specified dim) as a DTensor over the main mesh
- All-gather inputs: the ``torch.Tensor`` or ``Tensor`` s passed to all-gather,
  derived from the sharded parameter
- All-gather output: the ``torch.Tensor`` or ``Tensor`` s resulting from
  all-gathering the all-gather inputs
- Unsharded parameter: parameter used for forward/backward computation, derived
  from the all-gather output; autograd leaf

We define these tensors to describe the general framework that can accommodate
extensions, where:
- all-gather-inputs = pre-all-gather-transform(sharded-parameter)
- unsharded-parameter = post-all-gather-transform(all-gather-outputs)

For the default ``torch.Tensor`` case, there is only one all-gather input, and
it shares the same underlying tensor data as the sharded parameter, meaning
that they can be thought of as the same tensors. The same applies for the
all-gather output and unsharded parameter. For non-``torch.Tensor`` extensions,
these equivalences may no longer hold due to the pre/post-all-gather
transforms, and some may have multiple all-gather inputs/outputs (e.g.
quantized data and scales).

[Note: FSDP and autograd]
FSDP dynamically frees and allocates the unsharded parameter. Since autograd
can pack a reference to it or a view to save for backward, we use storage
resizing to implement the freeing/allocation since that preserves the aliasing.
This implies that we construct the unsharded parameter object once and write to
it in-place thereafter. For the default ``torch.Tensor` original parameter
case, the all-gather output and unsharded parameter share the same
data, so we use storage resizing on the all-gather output.

[Note: FSDP and spmd_types]
Refer to https://github.com/meta-pytorch/spmd_types/blob/main/docs/local_spmd_types.md
"""


class ShardedState(Enum):
    """
    - ``SHARDED``: The sharded parameter is registered to the module. It is the
      only contributor to parameter memory.
    - ``SHARDED_POST_FORWARD``: The unsharded parameter is resharded to a
      smaller world size. Since this data should not be used for computation,
      we do not register it to the module. Users should reshard the module
      before any in-place modifications. Both it and the sharded parameter
      contribute to parameter memory.
    - ``UNSHARDED``: The unsharded parameter is registered to the module. Both
      it and the sharded parameter contribute to parameter memory.
    """

    SHARDED = auto()
    SHARDED_POST_FORWARD = auto()
    UNSHARDED = auto()


@dataclass
class ParamModuleInfo:
    """
    For a parameter, this stores the module and the parameter name to be able
    to do a parameter swap via ``setattr(module, param_name, ...)`` or to get
    the parameter via ``getattr(module, param_name)``. We additionally save
    shared modules and shared parameter names to update them accordingly.
    """

    # Parameter names are unprefixed, e.g. "weight", not "lin.weight"
    module: nn.Module
    param_name: str
    shared_modules: list[nn.Module] = field(default_factory=list)
    shared_param_names: list[str] = field(default_factory=list)


class FSDPParam:
    """
    The DTensor view of one parameter of a ``fully_shard`` group: the spec of
    its sharded DTensor over the data-parallel mesh, composed with any TP/EP or
    SPMD sharding it had, and the conversions between that DTensor and the
    local data that the group's engine bucket stores, gathers and reduces.
    """

    _orig_size: torch.Size  # ND
    sharded_size: torch.Size  # ND
    _sharding_spec: DTensorSpec
    # Sharding specs for sharded grads, whose dtype can differ from the param's.
    # Cached per dtype so each backward doesn't build a DTensorSpec per param,
    # which costs CPU time with many params. Reset when _sharding_spec changes.
    _sharding_spec_by_dtype: dict[torch.dtype, DTensorSpec]
    _unsharded_dtensor_spec: (
        DTensorSpec | None
    )  # set for DTensor params (SPMD or TP/EP)
    _orig_param_uid: int

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: DataParallelMeshInfo,
        post_forward_mesh_info: FSDPMeshInfo | None,
        device: torch.device,
        shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None,
        mp_policy: MixedPrecisionPolicy,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self.mp_policy = mp_policy
        self._param_fqn: str | None = None  # prefixed from root module
        self.sharded_state = ShardedState.SHARDED
        # The group whose engine bucket stores this parameter, and its index
        # there, set by the group
        self._group: FullyShardBucket | None = None
        self._index = -1
        self._init_sharded_param(param, device, shard_placement_fn, mesh_info)

    @torch.no_grad()
    def _init_sharded_param(
        self,
        param: nn.Parameter,
        device: torch.device,
        shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None,
        mesh_info: DataParallelMeshInfo,
    ) -> None:
        shard_placement_result = None
        if callable(shard_placement_fn):
            shard_placement_result = resolve_shard_placement(
                shard_placement_fn(param),
                cast(FSDPMeshInfo, mesh_info),
            )
            self.mesh_info = shard_placement_result.mesh_info
            fsdp_placement = shard_placement_result.placement
        else:
            self.mesh_info = mesh_info  # pyrefly: ignore[bad-assignment]
            fsdp_placement = None
        self._shard_mesh = self._init_shard_mesh()
        if param.device != device and param.device.type != "meta":
            raise AssertionError(
                f"Expects the parameter to already be moved to device {device} but got {param.device}"
            )
        if not param.is_contiguous():
            raise NotImplementedError(
                f"FSDP does not support non-contiguous parameters yet: {param.shape=} {param.stride()=}"
            )
        # Capture the policy before parameter rewrites (e.g. spmd_types -> DTensor)
        self._init_grad_dtype_override = param._has_grad_dtype_override
        self._init_grad_dtype = param.grad_dtype
        if fsdp_placement is None:
            fsdp_placement = Shard(0)
        elif fsdp_placement.dim < 0:
            fsdp_placement = Shard(fsdp_placement.dim + param.ndim)
        if not isinstance(fsdp_placement, Shard):
            raise AssertionError(
                f"Expected Shard, got {type(fsdp_placement)}: {fsdp_placement}"
            )
        self.fsdp_placement = fsdp_placement
        shard_dim = fsdp_placement.dim
        # TODO: Replace the sharded DTensor parameter construction logic with
        # `distribute_tensor` after https://github.com/pytorch/pytorch/issues/116101
        # TODO: Simplify the following sharded parameter padding logic after
        # https://github.com/pytorch/pytorch/issues/113045
        # Tracks that this parameter contained FSDP-init-time spmd_types annotations;
        # this remains true even if spmd typechecking is disabled at runtime.
        self.is_spmd_types = (
            dist._is_spmd_types_available()
            and bool(spmd_local_type := spmd.get_local_type(param))
            and not isinstance(param, DTensor)
        )
        if self.is_spmd_types:
            param = self._resolve_spmd_types_for_storage(
                param, get_partition_spec(param), spmd_local_type, self.mesh_info
            )
        self.is_dtensor = isinstance(param, DTensor)
        self._orig_param_uid = _get_orig_param_uid(param)
        param_data = self._init_sharding_spec(param, fsdp_placement, shard_dim)
        self._sharding_spec_by_dtype = {}
        if not param_data.is_contiguous():
            raise AssertionError(
                f"Expected contiguous tensor, got {param_data.shape=} {param_data.stride()=}"
            )
        shard_dim = fsdp_placement.dim
        if shard_dim >= param_data.ndim:
            raise AssertionError(
                f"Shard dim {shard_dim} is invalid for {param_data.ndim}D tensor: {param.shape}"
            )
        self._orig_size = param_data.size()
        self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)
        if isinstance(self.mesh_info, FSDPMeshInfo):  # FSDP or HSDP
            shard_rank = self.mesh_info.shard_mesh_rank
            shard_world_size = self.mesh_info.shard_mesh_size
        else:  # DDP
            shard_rank = 0
            shard_world_size = 1

        self.sharded_size = EngineShard(shard_dim).compute_local_shape(
            param_data.size(), shard_rank, shard_world_size
        )
        # The tensor whose chunks the group's engine bucket stores: the
        # parameter, or the local tensor of a DTensor parameter
        self.shard_source = param_data
        self._requires_grad = param.requires_grad

    def _resolve_spmd_types_for_storage(
        self,
        param: nn.Parameter,
        partition_spec: "spmd.PartitionSpec | None",
        init_local_type: "spmd.LocalSpmdType",
        mesh_info: DataParallelMeshInfo,
    ) -> nn.Parameter:
        """
        Resolve spmd_types metadata and translate it into FSDP storage.

        The original parameter is a plain tensor annotated for module compute
        on the typechecking mesh. FSDP stores parameters as DTensors on
        ``mesh_info.spmd_mesh``, so init time only needs enough type information
        to translate into DTensor placements on the storage mesh axes. Axes
        named by ``dp_mesh_dims`` are FSDP-managed DP axes and are treated as
        ``spmd.R``; non-FSDP storage axes must have a direct spmd_types
        annotation.

        The original compute annotations and PartitionSpec are saved. At
        compute time, FSDP all-gathers the DTensor storage, exposes a plain
        tensor parameter, and restores spmd_types metadata on the init-time
        compute mesh. Missing compute axes are filled in as ``spmd.R``.

        Gradient storage placements are derived from the backward spmd_type.
        FSDP-managed "DP" axes require forward ``spmd.R`` since FSDP owns
        their gradient reduction. Consequently, the backward type is ``spmd.P``,
        so the incoming plain grad is wrapped as a ``Partial()`` DTensor on those
        axes, pending the gradient all-reduce/reduce-scatter to storage-time
        placement.

        For non-FSDP axes (e.g. TP), annotations are flexible. For a parameter
        annotated with ``spmd.I`` on TP, the grad is wrapped as ``Replicate()`` and no
        reduction is pending. ``spmd.S(dim)`` stays ``Shard(dim)``.

        This follows tied FWD/BWD typing for spmd_types: ``R`` -> ``P``,
        ``P`` -> ``R``, ``I`` -> ``I``, and ``S(dim)`` -> ``S(dim)``.
        """
        storage_mesh = mesh_info.spmd_mesh
        if storage_mesh is None or storage_mesh.mesh_dim_names is None:
            raise ValueError(
                "spmd_types parameters require fully_shard() to be called with both "
                "a named full DeviceMesh for FSDP storage and "
                "dp_mesh_dims=DataParallelMeshDims(shard=..., replicate=...), "
                "where shard names the mesh axis or axes FSDP shards parameters "
                "across and replicate names the HSDP/DDP mesh axis or axes FSDP "
                "replicates across."
            )

        storage_mesh_dim_names = storage_mesh.mesh_dim_names
        storage_mesh_axes = tuple(
            (name, spmd.MeshAxis.of(storage_mesh.get_group(name)))
            for name in storage_mesh_dim_names
        )
        storage_axes = tuple(axis for _, axis in storage_mesh_axes)
        local_type: spmd.PerMeshAxisSpmdTypes = dict(init_local_type)
        if partition_spec is not None:
            local_type.update(  # pyrefly: ignore [no-matching-overload]
                partition_spec_to_shard_types(partition_spec)
            )
        annotated_axes = tuple(local_type.keys())

        # resolve typechecking mesh on which parameters are annotated
        if _spans_same_mesh(annotated_axes, storage_axes):
            restore_mesh = annotated_axes
        elif (current_mesh := spmd.current_mesh()) is not None:
            restore_mesh = tuple(current_mesh)
        else:
            raise ValueError(
                f"Parameter '{self._module_info.param_name}' has partial "
                "spmd_types annotations that cannot be restored from its FSDP "
                "storage mesh. Fully annotate the parameter on its storage "
                "mesh, or wrap fully_shard() in spmd.set_current_mesh(...) "
                "when all parameters in the FSDP unit share one typechecking "
                "mesh. Mixed per-parameter typechecking meshes with partial "
                "annotations are not supported. "
                f"Annotated axes: {annotated_axes}. "
                f"Storage mesh axes: {storage_axes}."
            )

        # Validate the typechecking mesh with FSDP storage mesh.
        unknown_axes = tuple(
            axis for axis in init_local_type if axis not in restore_mesh
        )
        if unknown_axes:
            raise ValueError(
                f"FSDP found spmd_types annotations on parameter "
                f"'{self._module_info.param_name}' and will restore runtime "
                f"annotations on compute mesh axes {restore_mesh}, but the "
                f"parameter is annotated on axes {unknown_axes} that are not in "
                "that mesh. Annotate only axes in the compute mesh."
            )
        if not _spans_same_mesh(restore_mesh, storage_axes):
            raise ValueError(
                f"Parameter '{self._module_info.param_name}' uses spmd_types "
                "annotations on a typechecking mesh that does not span the "
                "same ranks as its FSDP storage mesh. FSDP can fill omitted "
                "FSDP-managed axes only when these meshes span the same "
                f"rank set. Typechecking mesh axes: {restore_mesh}. "
                f"Storage mesh axes: {storage_axes}."
            )
        dp_names = set(
            itertools.chain(
                mesh_info.dp_mesh_dims.shard_names,  # pyrefly: ignore [missing-attribute]
                mesh_info.dp_mesh_dims.replicate_names,  # pyrefly: ignore [missing-attribute]
            )
        )
        fsdp_axis = flatten_axes(
            tuple(axis for name, axis in storage_mesh_axes if name in dp_names)
        )
        non_fsdp_storage_mesh_axes = {
            axis for name, axis in storage_mesh_axes if name not in dp_names
        }
        storage_axis_types: spmd.PerMeshAxisSpmdTypes = {
            axis: spmd.R for name, axis in storage_mesh_axes if name in dp_names
        }
        # Param annotations are either optional spmd.R on FSDP axes or explicit
        # annotations on non-FSDP storage axes.
        for axis, axis_type in local_type.items():
            # MeshAxis ordering implements the FSDP-submesh containment check.
            if axis <= fsdp_axis:  # pyrefly: ignore [unsupported-operation]
                if axis_type is not spmd.R:
                    raise ValueError(
                        f"Expected spmd.R on FSDP DP axis {axis} for parameter "
                        f"'{self._module_info.param_name}' but got {axis_type}. "
                        "FSDP requires DP parameters to be R since it handles "
                        "the DP gradient reduction."
                    )
            else:
                if axis not in non_fsdp_storage_mesh_axes:
                    raise ValueError(
                        f"Parameter '{self._module_info.param_name}' has spmd_types "
                        f"annotation on axis {axis}, which is neither a non-FSDP "
                        "storage-mesh axis nor contained in the FSDP DP mesh."
                    )
                storage_axis_types[axis] = axis_type

        if set(storage_axes) != set(storage_axis_types.keys()):
            raise ValueError(
                f"Parameter '{self._module_info.param_name}' has incomplete "
                "spmd_types annotations for FSDP storage. "
                f"Annotated axes: {tuple(local_type.keys())}. "
                f"Storage mesh axes: {storage_axes}. "
                f"FSDP mesh dims: {mesh_info.dp_mesh_dims}. "
                "Missing non-FSDP storage axes: "
                f"{tuple(axis for _, axis in storage_mesh_axes if axis not in storage_axis_types)}."
            )

        # Translate to DTensor FWD/BWD placements; R@FSDP axes are optional.
        restore_type: spmd.PerMeshAxisSpmdTypes = {
            axis: init_local_type.get(axis, spmd.R) for axis in restore_mesh
        }
        placements = []
        grad_placements = []
        for axis_type in storage_axis_types.values():
            placements.append(spmd.spmd_type_to_dtensor_placement(axis_type))
            grad_placements.append(
                spmd.spmd_type_to_dtensor_placement(axis_type.backward_type())
            )
        dtensor_param = nn.Parameter(
            DTensor.from_local(param.data, storage_mesh, placements, run_check=False),
            requires_grad=param.requires_grad,
        )

        # Assign spmd_types state after validation succeeds.
        self._spmd_partition_spec: spmd.PartitionSpec | None = partition_spec
        self._spmd_init_local_type: spmd.LocalSpmdType = init_local_type
        self._spmd_restore_mesh: tuple[Any, ...] = restore_mesh
        self._spmd_restore_type: spmd.PerMeshAxisSpmdTypes = restore_type
        self._spmd_grad_placements: tuple[Placement, ...] = tuple(grad_placements)
        return dtensor_param

    def _restore_spmd_types(self, tensor: torch.Tensor) -> None:
        """Restore the saved spmd_types annotation onto a tensor."""
        if not self.is_spmd_types or not spmd.is_type_checking():
            return
        if self._spmd_restore_type:
            spmd.assert_type(
                tensor,
                self._spmd_restore_type,
                partition_spec=self._spmd_partition_spec,
            )

    def _init_sharding_spec(
        self,
        param: nn.Parameter,
        fsdp_placement: Shard,
        shard_dim: int,
    ) -> torch.Tensor:
        """
        Build ``_sharding_spec``, ``_spmd_mesh``, and ``_spmd_placements`` and
        return the local tensor data to be sharded.
        """
        self._unsharded_dtensor_spec = None
        if self.mesh_info.is_spmd_mesh and not self.is_dtensor:
            raise ValueError(
                "When dp_mesh_dims is provided, all parameters must be "
                "DTensors on the full SPMD mesh (e.g. via distribute_module). "
                f"Got plain tensor for parameter '{self._module_info.param_name}'."
            )
        if self.is_dtensor and self.mesh_info.is_spmd_mesh:
            return self._init_sharding_spec_spmd(param, fsdp_placement, shard_dim)
        if self.is_dtensor:
            return self._init_sharding_spec_tp(param, fsdp_placement, shard_dim)
        return self._init_sharding_spec_plain(param, fsdp_placement)

    def _init_sharding_spec_spmd(
        self,
        param: nn.Parameter,
        fsdp_placement: Shard,
        shard_dim: int,
    ) -> torch.Tensor:
        """SPMD path: param is a DTensor on the full SPMD mesh."""
        self._unsharded_dtensor_spec = cast(DTensor, param)._spec
        spmd_mesh = self._unsharded_dtensor_spec.mesh
        dp_dim_names = self.mesh_info.dp_mesh_dims
        if dp_dim_names is None:
            raise AssertionError("dp_dim_names must not be None for SPMD mesh")
        if spmd_mesh.mesh_dim_names is None:
            raise AssertionError("spmd_mesh.mesh_dim_names must not be None")
        if (
            self.mesh_info.spmd_mesh is not None
            and spmd_mesh is not self.mesh_info.spmd_mesh
        ):
            raise ValueError(
                "Expected param's DTensor mesh to be the same mesh passed "
                "to fully_shard, but got different mesh objects"
            )

        dp_shard_indices = [
            spmd_mesh.mesh_dim_names.index(n) for n in dp_dim_names.shard_names
        ]

        orig_placements = self._unsharded_dtensor_spec.placements
        for idx in dp_shard_indices:
            if not isinstance(orig_placements[idx], Replicate):
                raise ValueError(
                    f"Expected Replicate() on DP shard dim "
                    f"'{spmd_mesh.mesh_dim_names[idx]}' (index {idx}) "
                    f"but got {orig_placements[idx]}"
                )
        dp_replicate_indices = []
        for rep_name in dp_dim_names.replicate_names:
            rep_idx = spmd_mesh.mesh_dim_names.index(rep_name)
            dp_replicate_indices.append(rep_idx)
            if not isinstance(orig_placements[rep_idx], Replicate):
                raise ValueError(
                    f"Expected Replicate() on DP replicate dim "
                    f"'{spmd_mesh.mesh_dim_names[rep_idx]}' (index {rep_idx}) "
                    f"but got {orig_placements[rep_idx]}"
                )

        # Cache DP dim indices so _get_grad_inner_tensor can skip
        # redistribution on DP dims and let FSDP's reduce-scatter handle them.
        self._dp_dim_indices: frozenset[int] = frozenset(
            dp_shard_indices + dp_replicate_indices
        )

        new_placements = list(orig_placements)
        for dp_idx in dp_shard_indices:
            # split_factor = number of non-DP shards on shard_dim from
            # mesh dims with higher index (the "right-side" dims that
            # _StridedShard needs to interleave with)
            sf = 1
            for j in range(dp_idx + 1, spmd_mesh.ndim):
                p = orig_placements[j]
                if isinstance(p, (Shard, _StridedShard)) and p.dim == shard_dim:
                    sf *= spmd_mesh.size(j)
            new_placements[dp_idx] = (
                _StridedShard(shard_dim, split_factor=sf) if sf > 1 else fsdp_placement
            )

        self._spmd_mesh = spmd_mesh
        self._spmd_placements: tuple[Placement, ...] = tuple(new_placements)
        self._sharding_spec = self._build_spmd_sharding_spec(
            dp_dim_names,
            dp_shard_indices,
            fsdp_placement,
        )
        return cast(DTensor, param)._local_tensor

    def _build_spmd_sharding_spec(
        self,
        dp_dim_names: "DataParallelMeshDims",
        dp_shard_indices: list[int],
        fsdp_placement: Shard,
    ) -> DTensorSpec:
        """Build the DTensorSpec for the sharded parameter/gradient.

        When multiple DP shard dims exist (e.g. dp_shard + cp), flatten
        them into one axis so the parameter and gradient DTensor have one
        Shard axis for DP.
        """
        spmd_mesh = self._spmd_mesh
        spmd_placements = self._spmd_placements
        if self._unsharded_dtensor_spec is None:
            raise AssertionError("_unsharded_dtensor_spec cannot be None")
        tensor_meta = self._unsharded_dtensor_spec.tensor_meta

        if len(dp_shard_indices) <= 1:
            return DTensorSpec(spmd_mesh, spmd_placements, tensor_meta=tensor_meta)

        shard_names_set = set(dp_dim_names.shard_names)
        replicate_names_set = set(dp_dim_names.replicate_names)

        # Walk spmd_mesh.mesh_dim_names in order. Replace consecutive DP
        # shard dims with the flattened DP mesh; keep non-DP dims as-is.
        if spmd_mesh.mesh_dim_names is None:
            raise AssertionError("mesh_dim_names cannot be None")
        submeshes: list[DeviceMesh] = []
        spec_placements: list[Placement] = []
        skip = 0
        for i, name in enumerate(spmd_mesh.mesh_dim_names):
            if skip > 0:
                skip -= 1
                continue
            if name in shard_names_set:
                submeshes.append(self.mesh_info.mesh)
                if isinstance(self.mesh_info, HSDPMeshInfo):
                    spec_placements.append(Replicate())
                # Reuse the placement already computed for this DP shard dim
                # so that we don't lose _StridedShard.
                spec_placements.append(spmd_placements[i])
                skip = len(dp_dim_names.shard_names) - 1
            elif name in replicate_names_set and isinstance(
                self.mesh_info, HSDPMeshInfo
            ):
                # HSDP replicate is inserted by the shard branch above.
                continue
            else:
                submeshes.append(spmd_mesh[name])
                spec_placements.append(spmd_placements[i])

        spec_mesh = DeviceMesh._concatenate(submeshes)
        return DTensorSpec(spec_mesh, tuple(spec_placements), tensor_meta=tensor_meta)

    def _init_sharding_spec_tp(
        self,
        param: nn.Parameter,
        fsdp_placement: Shard,
        shard_dim: int,
    ) -> torch.Tensor:
        """TP/EP path: param is a DTensor, DP mesh is separate from TP mesh."""
        self._unsharded_dtensor_spec = cast(DTensor, param)._spec
        dp_mesh, tp_mesh = (self.mesh_info.mesh, self._unsharded_dtensor_spec.mesh)
        if dp_mesh is None or tp_mesh is None:
            raise AssertionError(
                "FSDP requires the DP and model parallel TP/EP mesh to be not None but got: \n"
                f"DP's mesh: {dp_mesh}\nTP/EP's mesh: {tp_mesh}"
            )
        self._spmd_mesh = DeviceMesh._concatenate([dp_mesh, tp_mesh])
        if len(self._unsharded_dtensor_spec.placements) > 2:
            raise NotImplementedError(
                f"FSDP only supports 1D TP/EP or 2D EP+TP, not {self._unsharded_dtensor_spec.placements}"
            )
        split_factor = self._unsharded_dtensor_spec.num_shards_map[shard_dim]
        if not (2 <= self._spmd_mesh.ndim <= 4):
            raise AssertionError(
                "_spmd_mesh.ndim can only be 2 (FSDP+TP/EP), 3 (FSDP+EP+TP, HSDP+TP/EP), "
                f"or 4 (HSDP+EP+TP) but got {self._spmd_mesh.ndim}."
            )
        if isinstance(self.mesh_info, FSDPMeshInfo):
            dp_shard_tp_placement = (
                (
                    _StridedShard(shard_dim, split_factor=split_factor)
                    if split_factor > 1
                    else fsdp_placement
                ),
                *self._unsharded_dtensor_spec.placements,
            )
        else:  # DDP
            dp_shard_tp_placement = (
                Replicate(),
                *self._unsharded_dtensor_spec.placements,
            )
        self._spmd_placements: tuple[Placement, ...]
        if isinstance(self.mesh_info, HSDPMeshInfo):
            if self.mesh_info.replicate_mesh_dim != 0:
                raise AssertionError(
                    f"Expected replicate_mesh_dim to be 0, got {self.mesh_info.replicate_mesh_dim}"
                )
            self._spmd_placements = (Replicate(),) + dp_shard_tp_placement
        else:
            self._spmd_placements = dp_shard_tp_placement

        self._sharding_spec = DTensorSpec(
            self._spmd_mesh,
            self._spmd_placements,
            tensor_meta=self._unsharded_dtensor_spec.tensor_meta,
        )
        return cast(DTensor, param)._local_tensor

    def _init_sharding_spec_plain(
        self,
        param: nn.Parameter,
        fsdp_placement: Shard,
    ) -> torch.Tensor:
        """Plain tensor path: param is not a DTensor."""
        self._spmd_mesh = self.mesh_info.mesh
        if isinstance(self.mesh_info, HSDPMeshInfo):
            self._spmd_placements = (Replicate(), fsdp_placement)
        elif isinstance(self.mesh_info, FSDPMeshInfo):
            self._spmd_placements = (fsdp_placement,)
        elif isinstance(self.mesh_info, DDPMeshInfo):
            self._spmd_placements = (Replicate(),)
        self._sharding_spec = DTensorSpec(
            self._spmd_mesh,
            self._spmd_placements,
            tensor_meta=TensorMeta(param.size(), param.stride(), param.dtype),
        )
        return param

    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """
        Converts a local tensor representing either the sharded parameter or
        sharded gradient to DTensor.
        """
        if tensor.shape != self.sharded_size:
            _raise_assert_with_print(
                f"Expects size {self.sharded_size} but got {tensor.shape}"
            )
        spec = self._sharding_spec_by_dtype.get(tensor.dtype)
        if spec is None:
            spec = _get_dtensor_spec_with_dtype(self._sharding_spec, tensor.dtype)
            self._sharding_spec_by_dtype[tensor.dtype] = spec
        return _from_local_no_grad(tensor, spec)

    def check_post_forward_reshard(self) -> None:
        """FSDP2's limit for resharding DTensor parameters to a smaller mesh
        after forward."""
        if self.is_dtensor:
            raise NotImplementedError(
                f"Resharding {self._display_name} to smaller mesh is not supported "
                "for DTensor parameters yet"
            )

    def refresh(self, sharded_param: nn.Parameter) -> None:
        """Re-read the spec of a sharded parameter that a module may have
        replaced (``nn.Module._apply``, ``load_state_dict(assign=True)``)."""
        if not isinstance(sharded_param, DTensor):
            raise AssertionError(f"Expected DTensor, got {type(sharded_param)}")
        self._sharding_spec = sharded_param._spec
        self._sharding_spec_by_dtype.clear()

    # The engine bucket's state of this parameter, which tools read #
    @property
    def _bucket_param(self):
        if self._group is None:
            raise AssertionError(f"{self._display_name} has no group yet")
        return self._group.bucket.bucket_params[self._index]

    @property
    def sharded_param(self) -> nn.Parameter:  # ND
        return self._bucket_param.sharded_param

    @property
    def _sharded_param_data(self) -> torch.Tensor:
        return cast("FullyShardBucket", self._group).homing.slots[self._index]

    @property
    def _unsharded_param(self) -> nn.Parameter:
        params = cast("FullyShardBucket", self._group).bucket.unsharded_params
        if params is None:
            raise AttributeError(f"{self._display_name} has not been unsharded yet")
        return params[self._index]

    @property
    def unsharded_param(self) -> nn.Parameter:  # ND
        return self._unsharded_param

    @property
    def unsharded_accumulated_grad(self) -> torch.Tensor | None:
        # The autograd leaf owns accumulation even while its parameter data is
        # resharded. Kept to avoid breaking potential external callers.
        params = cast("FullyShardBucket", self._group).bucket.unsharded_params
        return params[self._index].grad if params is not None else None

    @property
    def _display_name(self) -> str:
        # The FQN is set in lazy init
        return self._param_fqn or self._module_info.param_name

    @property
    def may_reduce_grad_outside_dp(self) -> bool:
        """Whether ``_get_grad_inner_tensor`` may reduce the gradient over a
        non-DP mesh dim, e.g. the TP all-reduce of a SequenceParallel norm
        weight's ``Partial`` gradient, which runs in the gradient's dtype."""
        spec = self._unsharded_dtensor_spec
        if spec is None:
            return False
        dp_dims = self._dp_dim_indices if self.mesh_info.is_spmd_mesh else ()
        if self.is_spmd_types:
            return any(
                placement.is_partial()
                for i, placement in enumerate(self._spmd_grad_placements)
                if i not in dp_dims
            )
        # A replicated parameter's gradient is only known at runtime, and it
        # can be Partial
        return any(
            isinstance(placement, Replicate)
            for i, placement in enumerate(spec.placements)
            if i not in dp_dims
        )

    def _get_grad_inner_tensor(self, grad: torch.Tensor) -> torch.Tensor:
        if self.is_spmd_types:
            if self._unsharded_dtensor_spec is None:
                raise AssertionError(
                    "Expected _unsharded_dtensor_spec for spmd_types param"
                )
            grad = DTensor.from_local(
                grad,
                self._unsharded_dtensor_spec.mesh,
                tuple(self._spmd_grad_placements),
                run_check=False,
            )
            if not self.is_dtensor:
                raise AssertionError(
                    "Expected spmd_types -> DTensor wrapping to use the DTensor gradient "
                    "path for correct gradient redistribution."
                )

        if self.is_dtensor:
            if isinstance(grad, AsyncCollectiveTensor):
                grad = grad.wait()
            if not isinstance(grad, DTensor):
                raise AssertionError(f"Expected DTensor, got {type(grad)}")
            if self._unsharded_dtensor_spec is None:
                raise AssertionError(
                    "Expected _unsharded_dtensor_spec for DTensor param"
                )
            placements = self._unsharded_dtensor_spec.placements
            if self.mesh_info.is_spmd_mesh:
                # Only redistribute non-DP dims; keep Partial on DP dims
                # so FSDP's reduce-scatter handles them directly, avoiding
                # a redundant all-reduce on the DP dimensions.
                target_placements = tuple(
                    grad.placements[i] if i in self._dp_dim_indices else placements[i]
                    for i in range(len(placements))
                )
                if target_placements != grad.placements:
                    if len(placements) != len(grad.placements):
                        raise AssertionError(
                            f"Expected same placement length: {placements=} {grad.placements=}"
                        )
                    grad = grad.redistribute(placements=target_placements)
            else:
                if placements != grad.placements:
                    if len(placements) != len(grad.placements):
                        raise AssertionError(
                            f"Expected same placement length: {placements=} {grad.placements=}"
                        )
                    grad = grad.redistribute(placements=placements)
            grad = grad._local_tensor
        return grad

    @property
    def _sharded_local_tensor(self) -> torch.Tensor:
        return cast(DTensor, self.sharded_param)._local_tensor

    def _init_shard_mesh(self) -> DeviceMesh:
        mesh = self.mesh_info.mesh
        if mesh.ndim == 1:
            return mesh
        if mesh.mesh_dim_names is None:
            raise AssertionError("Expected mesh_dim_names to not be None")
        return mesh[mesh.mesh_dim_names[-1]]

    @property
    def shard_mesh(self):
        return self._shard_mesh

    @property
    def shard_mesh_from_root(self):
        return self.shard_mesh

    def __repr__(self):
        return f"FSDPParam(fqn={self._param_fqn}, orig_size={self._orig_size})"


def _get_dtensor_spec_with_dtype(spec: DTensorSpec, dtype: torch.dtype) -> DTensorSpec:
    tensor_meta = spec.tensor_meta
    if tensor_meta is None or tensor_meta.dtype == dtype:
        return spec
    return replace(
        spec,
        tensor_meta=TensorMeta(tensor_meta.shape, tensor_meta.stride, dtype),
    )
