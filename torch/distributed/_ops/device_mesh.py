# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
from torch import SymInt

from ..device_mesh import DeviceMesh


# Register custom operator
torch.library.define(
    "device_mesh::_runtime_compute_coordinate_on_dim",
    "(Tensor full_mesh, int index) -> SymInt",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.register_fake("device_mesh::_runtime_compute_coordinate_on_dim")
def _runtime_compute_coordinate_on_dim_fake(
    full_mesh: torch.Tensor, index: int
) -> SymInt:
    ctx = torch._custom_op.impl.get_ctx()
    sz = ctx.new_dynamic_size()
    # We may or may not actually end up returning/using this symbol - so
    # mark it as ignorable.
    ctx._shape_env.ignorable_fresh_unbacked_symbols.append(sz.node._expr)
    return sz


@torch.library.impl(
    "device_mesh::_runtime_compute_coordinate_on_dim", "CompositeExplicitAutograd"
)
def _runtime_compute_coordinate_on_dim_impl(full_mesh: torch.Tensor, index: int) -> int:
    rank = torch.distributed.get_rank()
    mesh = DeviceMesh._get_mesh_tensor_from_full_mesh(full_mesh)
    mesh_coords = DeviceMesh._compute_coordinates_from_mesh(mesh, rank)
    assert mesh_coords is not None
    return mesh_coords[index]
