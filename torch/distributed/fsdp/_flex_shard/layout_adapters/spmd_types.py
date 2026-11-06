# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import spmd_types as spmd

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from ..placement_contract import GlobalLayout, set_global_layout


def spmd_types_to_global_layout(
    module: nn.Module,
    layouts: Mapping[str, spmd.SpmdType],
    mesh: DeviceMesh,
) -> None:
    """Declare the layout of local parameters described by ``SpmdType`` layouts.

    Call before ``flex_shard()`` on a model whose parameters are already local
    shards under ``spmd_types``, e.g. after tensor parallelism. ``layouts`` maps
    parameter FQNs to their layouts; parameters without an entry are left
    unchanged. Call once per mesh when parameters live on different meshes.

    Layout axes are resolved against ``mesh``: string axes by mesh dim name,
    where a name absent from ``mesh`` counts as size 1, and ``MeshAxis`` or
    ``ProcessGroup`` axes by matching a mesh dim's group. Sharding is assumed
    to be even. Raises ``NotImplementedError`` for ``Partial`` axes, for
    ``Varying`` axes with no shard dim, and for a tensor dim sharded by
    several ``S(dim)`` axes without a ``PartitionSpec`` to order them.
    """
    for fqn, param in module.named_parameters(remove_duplicate=False):
        layout = layouts.get(fqn)
        if layout is None:
            continue
        set_global_layout(param, _spmd_global_layout(tuple(param.shape), layout, mesh))


def _spmd_global_layout(
    local_shape: tuple[int, ...], layout: spmd.SpmdType, mesh: DeviceMesh
) -> GlobalLayout:
    ndim = len(local_shape)
    global_shape = list(local_shape)
    offsets = [0] * ndim
    for dim, axes in enumerate(_shard_axes_per_dim(layout, ndim, mesh)):
        global_shape[dim] *= math.prod(size for size, _ in axes)
        chunk = global_shape[dim]
        # Earlier axes split the dim first, so each later axis indexes within
        # the previous axis's chunk.
        for size, coordinate in axes:
            chunk //= size
            offsets[dim] += coordinate * chunk
    if math.prod(local_shape) == 0:
        return GlobalLayout(tuple(global_shape), (), (), ())
    return GlobalLayout(
        global_shape=tuple(global_shape),
        global_offsets=(tuple(offsets),),
        local_offsets=((0,) * ndim,),
        local_sizes=(local_shape,),
    )


def _shard_axes_per_dim(
    layout: spmd.SpmdType, ndim: int, mesh: DeviceMesh
) -> list[list[tuple[int, int]]]:
    """Return ``(size, coordinate)`` of the axes sharding each tensor dim, in
    sharding order, skipping size-1 axes."""
    dim_axes: list[list[Any]] = [[] for _ in range(ndim)]
    spec_axes: list[Any] = []
    if layout.partition_spec is not None:
        if len(layout.partition_spec) != ndim:
            raise ValueError(
                f"PartitionSpec {layout.partition_spec!r} does not match a "
                f"{ndim}-dim parameter."
            )
        for dim, entry in enumerate(layout.partition_spec):
            if entry is not None:
                dim_axes[dim] = list(entry) if isinstance(entry, tuple) else [entry]
                spec_axes.extend(dim_axes[dim])
    for axis, typ in layout.local_type.items():
        if typ is spmd.P:
            raise NotImplementedError(f"axis {axis!r} is Partial.")
        if isinstance(typ, spmd.Shard):
            if not -ndim <= typ.dim < ndim:
                raise ValueError(f"{typ!r} is out of bounds for {ndim}-dim params.")
            dim_axes[typ.dim % ndim].append(axis)
        elif typ is spmd.V and axis not in spec_axes:
            raise NotImplementedError(f"axis {axis!r} is Varying with no shard dim.")

    result = []
    for dim, axes in enumerate(dim_axes):
        sized = [
            (size, coordinate)
            for size, coordinate in (_axis_size_and_coordinate(a, mesh) for a in axes)
            if size > 1
        ]
        if len(sized) > 1 and layout.partition_spec is None:
            raise NotImplementedError(
                f"tensor dim {dim} is sharded by several S(dim) axes; use a "
                "PartitionSpec to order them."
            )
        result.append(sized)
    return result


def _axis_size_and_coordinate(axis: Any, mesh: DeviceMesh) -> tuple[int, int]:
    names = mesh.mesh_dim_names or ()
    if isinstance(axis, str):
        if axis not in names:
            return 1, 0
        name = names[names.index(axis)]
    else:
        mesh_axis = spmd.normalize_axis(axis)
        if mesh_axis.size() == 1:
            return 1, 0
        matches = [n for n in names if spmd.MeshAxis.of(mesh.get_group(n)) == mesh_axis]
        if not matches:
            raise ValueError(f"Mesh axis {axis!r} is not a dim of {mesh}.")
        name = matches[0]
    return mesh.size(names.index(name)), mesh.get_local_rank(name)
