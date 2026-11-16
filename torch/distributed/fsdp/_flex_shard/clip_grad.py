# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gradient-norm clipping for ``flex_shard`` parameters, whose local tensors
hold this rank's parts of their gradients."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from .placement_contract import get_outer_shard_groups
from .sharded_param import get_mesh, get_outer_layout


if TYPE_CHECKING:
    from collections.abc import Iterable

    from .placement_contract import GlobalLayout


def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    r"""Clip the gradient norm of ``parameters`` as
    :func:`torch.nn.utils.clip_grad_norm_` does on their full gradients.

    A ``flex_shard`` parameter's local gradient is this rank's part of the full
    gradient, disjoint from the parts of the other ranks of its bucket's mesh,
    so its contribution to the norm is combined over that mesh. A parameter
    that is itself a part of a bigger one (tensor or expert parallelism, see
    ``set_global_layout``) is also combined over the groups that its outer
    sharding splits it over (``set_outer_shard_groups``; a layout that covers
    the full parameter needs none). Other parameters count once, as full
    gradients that every rank holds. Every rank must pass the same parameters,
    since the norm of each group's gradients is an all-reduce over it.

    Args:
        parameters: Tensors whose gradients to clip.
        max_norm: Max norm of the gradients.
        norm_type: Type of the p-norm, as in ``clip_grad_norm_``: ``inf`` for the
            infinity norm.
        error_if_nonfinite: Whether to raise if the total norm is ``nan``,
            ``inf`` or ``-inf``.
        foreach: Whether to use the foreach implementation; ``None`` picks it
            where it is fast.

    Returns:
        The total norm of the full gradients, viewed as a single vector.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    params = [p for p in parameters if p.grad is not None]
    total_norm = _total_norm(params, float(norm_type), foreach)
    if error_if_nonfinite and not torch.isfinite(total_norm):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from `parameters` "
            "is non-finite, so it cannot be clipped. To disable this error and "
            "scale the gradients by the non-finite norm anyway, set "
            "`error_if_nonfinite=False`"
        )
    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach)
    return total_norm


def _total_norm(
    params: list[torch.Tensor], norm_type: float, foreach: bool | None
) -> torch.Tensor:
    grads_by_groups: dict[tuple[dist.ProcessGroup, ...], list[torch.Tensor]] = {}
    for param in params:
        grads_by_groups.setdefault(_norm_groups(param), []).append(param.grad)  # type: ignore[arg-type]
    if not grads_by_groups:
        return torch.tensor(0.0)
    norms = []
    for groups, grads in grads_by_groups.items():
        # A rank may hold no part of a gradient (e.g. one it does not own),
        # which has no infinity norm; it still joins its groups' all-reduces
        nonempty = [grad for grad in grads if grad.numel() > 0]
        norm = (
            torch.nn.utils.get_total_norm(nonempty, norm_type, False, foreach)
            if nonempty
            else grads[0].new_zeros(())
        )
        if groups:
            norm = _all_reduce_norm(norm, norm_type, groups)
        norms.append(norm)
    device = norms[0].device
    return torch.linalg.vector_norm(
        torch.stack([norm.to(device) for norm in norms]), norm_type
    )


def _norm_groups(param: torch.Tensor) -> tuple[dist.ProcessGroup, ...]:
    """The groups whose ranks hold the other parts of ``param``'s full gradient:
    its bucket's mesh and the groups of its outer sharding."""
    mesh = get_mesh(param)
    if mesh is None:
        return ()
    groups = [mesh.get_group()]
    outer_layout = get_outer_layout(param)
    if outer_layout is not None and not _covers_full_parameter(outer_layout):
        outer_groups = get_outer_shard_groups(param)
        if not outer_groups:
            raise ValueError(
                "clip_grad_norm_ needs the groups over which a flex_shard parameter's "
                "outer (e.g. tensor- or expert-parallel) layout splits it: declare "
                "them with set_outer_shard_groups before flex_shard() "
                "(dtensor_to_global_layout does this for DTensor parameters)"
            )
        groups.extend(outer_groups)
    return tuple(group for group in groups if dist.get_world_size(group) > 1)


def _covers_full_parameter(layout: GlobalLayout) -> bool:
    return sum(math.prod(size) for size in layout.local_sizes) == math.prod(
        layout.global_shape
    )


def _all_reduce_norm(
    norm: torch.Tensor, norm_type: float, groups: tuple[dist.ProcessGroup, ...]
) -> torch.Tensor:
    """The norm of the disjoint parts that the ranks of ``groups`` hold, from
    each rank's norm of its part."""
    if math.isinf(norm_type):
        op = dist.ReduceOp.MAX if norm_type > 0 else dist.ReduceOp.MIN
        for group in groups:
            dist.all_reduce(norm, op=op, group=group)
        return norm
    if norm_type == 0:  # counts nonzero elements
        for group in groups:
            dist.all_reduce(norm, group=group)
        return norm
    norm = norm.pow(norm_type)
    for group in groups:
        dist.all_reduce(norm, group=group)
    return norm.pow(1.0 / norm_type)
