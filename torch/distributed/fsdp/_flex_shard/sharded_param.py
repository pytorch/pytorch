# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch
    from torch.distributed.device_mesh import DeviceMesh

    from .placement_contract import GlobalLayout, Placement


# Hidden attribute names for FlexShard metadata on local parameter tensors.
_PLACEMENTS_ATTR = "_placements"
_GLOBAL_SHAPE_ATTR = "_global_shape"
_GLOBAL_STRIDE_ATTR = "_global_stride"
_MESH_ATTR = "_mesh"
_OUTER_LAYOUT_ATTR = "_outer_layout"


def get_placements(tensor: torch.Tensor) -> tuple[Placement, ...] | None:
    """Get FlexShard placements from a tensor, or None if not annotated."""
    return getattr(tensor, _PLACEMENTS_ATTR, None)


def get_global_shape(tensor: torch.Tensor) -> torch.Size | None:
    """Get the global unsharded shape from a tensor, or None if not annotated."""
    return getattr(tensor, _GLOBAL_SHAPE_ATTR, None)


def get_mesh(tensor: torch.Tensor) -> DeviceMesh | None:
    """Get the mesh of the bucket holding a tensor, or None if not annotated."""
    return getattr(tensor, _MESH_ATTR, None)


def get_outer_layout(tensor: torch.Tensor) -> GlobalLayout | None:
    """Return the outer (e.g. TP/EP) layout a FlexShard parameter declared
    before ``flex_shard()``, or ``None``.

    The layout places the parameter's ``get_global_shape`` tensor in the full
    parameter.
    """
    return getattr(tensor, _OUTER_LAYOUT_ATTR, None)


def is_flex_shard_param(tensor: torch.Tensor) -> bool:
    """Return whether a tensor represents a FlexShard-managed parameter."""
    return hasattr(tensor, _PLACEMENTS_ATTR)


def set_sharding_info(
    tensor: torch.Tensor,
    placements: tuple[Placement, ...],
    global_shape: torch.Size,
    global_stride: tuple[int, ...],
    mesh: DeviceMesh,
    outer_layout: GlobalLayout | None = None,
) -> None:
    """Annotate a local parameter tensor with its global FlexShard metadata."""
    setattr(tensor, _PLACEMENTS_ATTR, placements)
    setattr(tensor, _GLOBAL_SHAPE_ATTR, global_shape)
    setattr(tensor, _GLOBAL_STRIDE_ATTR, global_stride)
    setattr(tensor, _MESH_ATTR, mesh)
    setattr(tensor, _OUTER_LAYOUT_ATTR, outer_layout)


__all__ = [
    "get_global_shape",
    "get_mesh",
    "get_outer_layout",
    "get_placements",
    "is_flex_shard_param",
    "set_sharding_info",
]
