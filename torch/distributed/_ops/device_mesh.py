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
    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

    ctx = torch._custom_op.impl.get_ctx()
    shape_env = ctx._shape_env

    # Bypass allow_dynamic_output_shape_ops check by directly creating the symint.
    # This is intentional - the coordinate is always valid and bounded.
    sz = shape_env.create_unbacked_symint()

    # Apply size constraints - coordinate is bounded by mesh size on the given dimension.
    # The full_mesh tensor has an extra batch dimension at the front, so the actual
    # mesh dimensions start at index 1. mesh.size(index) = full_mesh.size(index + 1)
    mesh_size = full_mesh.size(index + 1)
    _constrain_range_for_size(
        sz, min=0, max=mesh_size - 1 if isinstance(mesh_size, int) else None
    )

    try:
        # Check if we're currently tracing in dynamo (as opposed to AOT or export).
        in_dynamo = torch._dynamo.symbolic_convert.InstructionTranslator.current_tx()
    except AttributeError:
        in_dynamo = False

    if in_dynamo:
        # During dynamo tracing, distributed ops are treated as atomic - so the
        # rank SymInt may be computed but not traced into the graph (e.g., it
        # affects tensor values but not shapes). Mark it as ignorable here;
        # when we decompose these ops later (after dynamo), we'll create fresh
        # SymInts that do get traced.
        shape_env.ignorable_fresh_unbacked_symbols.append(sz.node._expr)

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
