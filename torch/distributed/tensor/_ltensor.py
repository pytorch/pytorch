"""Local Tensor with variance tracking for distributed operations."""

from collections.abc import Callable
from typing import Optional, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.distributed.tensor import DTensor

from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate
from torch.utils._pytree import tree_map
from . import _varying_collectives as vcols


_CUSTOM_VARIANCE_STRATEGY_MAP: dict[Callable, Callable] = {}

_CUSTOM_OPERATOR_HANDLER_MAP: dict[Callable, Callable] = {
    torch.ops._c10d_functional.all_reduce: vcols.all_reduce_invariant
}


def register_variance_strategy(ops):
    """Register custom variance strategy for calculating output variance."""

    def wrapper(func):
        for op in ops:
            _CUSTOM_VARIANCE_STRATEGY_MAP[op] = func
        return func

    return wrapper


def _get_axis_name_from_group(mesh: DeviceMesh, group_name: str) -> Optional[str]:
    """Get mesh axis name for a process group."""
    if mesh.mesh_dim_names:
        for axis_name in mesh.mesh_dim_names:
            if mesh.get_group(axis_name).group_name == group_name:
                return axis_name
    return None


# Note: Intentionally do not include all_gather in the invariant-output registration.
# Keeping all_gather's output as varying prevents mark_varying from being inserted in the forward,
# which would otherwise cause a redundant all_reduce to be introduced during the backward pass.
@register_variance_strategy(
    [
        torch.ops._c10d_functional.all_reduce,
    ]
)
def _invariant_output_strategy(input_variant_dims, *args, **kwargs):
    """Collectives that make output invariant on the reduced axis (all_reduce, all_gather)."""
    input_tensor, _, group_name = args

    if not isinstance(input_tensor, LTensor):
        return input_variant_dims

    mesh = input_tensor._mesh
    axis_name = _get_axis_name_from_group(mesh, group_name)
    if axis_name is None:
        raise ValueError(
            f"Could not find mesh axis for group {group_name}. "
            f"Available axes: {mesh.mesh_dim_names}"
        )

    return input_variant_dims - {axis_name}


@register_variance_strategy([])  # [FIXME]: scatter ops
def _variant_output_strategy(input_variant_dims, *args, **kwargs):
    """Collectives that make output variant on the scatter axis (reduce_scatter)."""
    return input_variant_dims


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
    _variant_dims: set[str]
    _mesh: DeviceMesh

    def __new__(
        cls,
        local_tensor: torch.Tensor,
        variant_dims: set[str],
        mesh: DeviceMesh,
    ):
        r = local_tensor.as_subclass(cls)
        r._variant_dims = variant_dims
        r._mesh = mesh
        r._local_tensor = local_tensor
        return r

    @staticmethod
    def compute_metadata_from_dtensor(dtensor: "DTensor"):
        mesh = dtensor.device_mesh
        placements = dtensor.placements

        if mesh.mesh_dim_names is None:
            raise ValueError(
                "DTensor's mesh must have mesh_dim_names to convert to LTensor. "
                "LTensor requires named mesh axes for variance tracking."
            )

        variant_dims = set()
        for mesh_dim_idx, placement in enumerate(placements):
            if not isinstance(placement, Replicate):
                mesh_dim_name = mesh.mesh_dim_names[mesh_dim_idx]
                variant_dims.add(mesh_dim_name)

        return {"mesh": dtensor.device_mesh, "variant_dims": variant_dims}

    @property
    def variant_dims(self) -> set[str]:
        return self._variant_dims

    @property
    def mesh(self) -> DeviceMesh:
        return self._mesh

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Dispatch hook: compute union variance, insert mark_varying, unwrap/wrap tensors."""
        if kwargs is None:
            kwargs = {}

        out_variant_dims: set[str] = set()
        meshes: set[DeviceMesh] = set()

        def _extract_metadata(t):
            if isinstance(t, LTensor):
                out_variant_dims.update(t._variant_dims)
                meshes.add(t._mesh)

        tree_map(_extract_metadata, args)

        if len(meshes) > 1:
            raise RuntimeError(
                "Cannot mix LTensors from different meshes in one operation!"
            )
        mesh = meshes.pop()

        if func in _CUSTOM_VARIANCE_STRATEGY_MAP:
            out_variant_dims = _CUSTOM_VARIANCE_STRATEGY_MAP[func](
                out_variant_dims, *args, **kwargs
            )

        func = _CUSTOM_OPERATOR_HANDLER_MAP.get(func, func)

        def unwrap_and_insert_mark_varying(t):
            assert not isinstance(t, AsyncCollectiveTensor), (
                f"AsyncCollectiveTensor: {t=}"
            )

            if not isinstance(t, LTensor):
                return t

            local_tensor = t._local_tensor
            src_variant_dims = t._variant_dims
            missing_axes = out_variant_dims - src_variant_dims

            if not missing_axes:
                return local_tensor

            for axis_name in missing_axes:
                group_name = mesh.get_group(axis_name).group_name
                local_tensor = vcols.mark_varying(local_tensor, group_name=group_name)

            return local_tensor

        unwrapped_args = tree_map(unwrap_and_insert_mark_varying, args)
        unwrapped_kwargs = tree_map(unwrap_and_insert_mark_varying, kwargs)

        result = func(*unwrapped_args, **unwrapped_kwargs)

        def wrap(t):
            assert not isinstance(t, AsyncCollectiveTensor), (
                f"AsyncCollectiveTensor: {t=}"
            )
            if isinstance(t, torch.Tensor) and not isinstance(t, LTensor):
                return LTensor(t, out_variant_dims, mesh)
            else:
                return t

        return tree_map(wrap, result)
