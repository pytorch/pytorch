"""Local Tensor with variance tracking for distributed operations."""

from typing import Callable, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.utils._pytree import tree_map

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate


_CUSTOM_VARIANCE_STRATEGY_MAP: dict[Callable, Callable] = {}


def register_variance_strategy(ops):
    """Register custom variance handler for collective operations."""
    def wrapper(func):
        for op in ops:
            _CUSTOM_VARIANCE_STRATEGY_MAP[op] = func
        return func
    return wrapper


def _get_axis_name_from_group(mesh: DeviceMesh, group_name: str) -> Optional[str]:
    """Get mesh axis name for a process group."""
    for axis_name in mesh.mesh_dim_names:
        if mesh.get_group(axis_name).group_name == group_name:
            return axis_name
    return None


@register_variance_strategy([
    torch.ops._c10d_functional.all_reduce,
    torch.ops._c10d_functional.all_gather_into_tensor
])
def _invariant_output_strategy(input_variant_axes, *args, **kwargs):
    """Collectives that make output invariant on the reduced axis (all_reduce, all_gather)."""
    input_tensor, _, group_name = args

    if not isinstance(input_tensor, LTensor):
        return input_variant_axes

    mesh = input_tensor._mesh
    axis_name = _get_axis_name_from_group(mesh, group_name)
    if axis_name is None:
        raise ValueError(
            f"Could not find mesh axis for group {group_name}. "
            f"Available axes: {mesh.mesh_dim_names}"
        )

    return input_variant_axes - {axis_name}


@register_variance_strategy([]) # [FIXME]: scatter ops
def _variant_output_strategy(input_variant_axes, *args, **kwargs):
    """Collectives that make output variant on the scatter axis (reduce_scatter)."""
    return input_variant_axes


class LTensor(torch.Tensor):
    """
    ``LTensor`` (Local Tensor) is a subclass of ``torch.Tensor`` that tracks variance metadata
    through DTensor -> Tensor -> DTensor transitions. It tracks mesh axes that a tensor varies along
    (i.e., has different values across ranks)

    States:
    * Variant on axis: Tensor has different values across ranks on this mesh axis
    * Invariant on axis: Tensor has identical values across ranks on this mesh axis (Replicate())

    When calling PyTorch operators, ``LTensor`` computes the union of variant axes from all inputs
    and automatically inserts gradient aggregation collectives where needed to ensure correctness
    during backward pass

    .. note:: The recommended way to use ``LTensor`` is within ``local_map`` or ``DTensor.to_local``
        which automatically infers variance from DTensor placements (non-Replicate placements become
        variant axes).
    """

    _local_tensor: torch.Tensor
    _variant_axes: set[str]
    _mesh: DeviceMesh

    def __new__(
        cls,
        local_tensor: torch.Tensor,
        variant_axes: set[str],
        mesh: DeviceMesh,
    ):
        if not isinstance(variant_axes, set):
            raise TypeError(
                f"variant_axes must be a set, got {type(variant_axes)}"
            )

        invalid_axes = variant_axes - set(mesh.mesh_dim_names)
        if invalid_axes:
            raise ValueError(
                f"Invalid variant axes {invalid_axes}. "
                f"Valid mesh axes: {mesh.mesh_dim_names}"
            )

        r = torch.Tensor._make_subclass(
            cls,
            local_tensor,
            require_grad=local_tensor.requires_grad
        )

        r._variant_axes = variant_axes
        r._mesh = mesh
        r._local_tensor = local_tensor

        return r

    @staticmethod
    def from_dtensor(dtensor: DTensor) -> "LTensor":
        """Convert DTensor to LTensor. Non-Replicate placements become variant_axes."""
        local_tensor = dtensor._local_tensor
        mesh = dtensor.device_mesh
        placements = dtensor.placements

        if mesh.mesh_dim_names is None:
            raise ValueError(
                "DTensor's mesh must have mesh_dim_names to convert to LTensor. "
                "LTensor requires named mesh axes for variance tracking."
            )

        variant_axes = set()
        for mesh_dim_idx, placement in enumerate(placements):
            if not isinstance(placement, Replicate):
                mesh_dim_name = mesh.mesh_dim_names[mesh_dim_idx]
                variant_axes.add(mesh_dim_name)

        return LTensor(local_tensor, variant_axes, mesh)

    @property
    def variant_axes(self) -> set[str]:
        return self._variant_axes

    @property
    def mesh(self) -> DeviceMesh:
        return self._mesh

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Dispatch hook: compute union variance, insert mark_varying, unwrap/wrap tensors."""
        if kwargs is None:
            kwargs = {}

        out_variant_axes: set[str] = set()
        meshes: set[DeviceMesh] = set()
        for arg in tree_map(lambda x: x, args):
            if isinstance(arg, LTensor):
                out_variant_axes |= arg._variant_axes
                meshes |= {arg._mesh}

        if len(meshes) > 1:
            raise RuntimeError(
                "Cannot mix LTensors from different meshes in one operation!"
            )
        mesh = meshes.pop()

        if func in _CUSTOM_VARIANCE_STRATEGY_MAP:
            out_variant_axes = _CUSTOM_VARIANCE_STRATEGY_MAP[func](out_variant_axes, *args, **kwargs)

        def unwrap_and_insert_mark_varying(t):
            if not isinstance(t, LTensor):
                return t

            local_tensor = t._local_tensor
            src_variant_axes = t._variant_axes
            missing_axes = out_variant_axes - src_variant_axes

            if not missing_axes:
                return local_tensor

            from . import _varying_collectives as vcols

            for axis_name in missing_axes:
                group_name = mesh.get_group(axis_name).group_name
                local_tensor = vcols.mark_varying(
                    local_tensor, group_name=group_name
                )

            return local_tensor

        unwrapped_args = tree_map(unwrap_and_insert_mark_varying, args)
        unwrapped_kwargs = tree_map(unwrap_and_insert_mark_varying, kwargs)

        result = func(*unwrapped_args, **unwrapped_kwargs)

        def wrap(t):
            if isinstance(t, torch.Tensor) and not isinstance(t, LTensor):
                return LTensor(t, out_variant_axes, mesh)
            else:
                return t

        return tree_map(wrap, result)

    def __tensor_flatten__(self):
        return (
            ["_local_tensor"],
            (self._variant_axes, self._mesh),
        )

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        variant_axes, mesh = metadata
        local_tensor = inner_tensors["_local_tensor"]
        return LTensor(local_tensor, variant_axes, mesh)

    def __repr__(self) -> str:
        return (
            f"LTensor(\n"
            f"  local_tensor={self._local_tensor},\n"
            f"  variant_axes={self.variant_axes},\n"
            f"  mesh_dims={self.mesh.mesh_dim_names}\n"
            f")"
        )
