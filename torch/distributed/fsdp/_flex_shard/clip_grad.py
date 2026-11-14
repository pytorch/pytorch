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

from .sharded_param import get_mesh, get_outer_layout


if TYPE_CHECKING:
    from collections.abc import Iterable


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
    so its contribution to the norm is combined over that mesh. Other
    parameters count once, as full gradients that every rank holds. Every rank
    must pass the same parameters, since the norm of each mesh's gradients is
    an all-reduce over that mesh.

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
    grads_by_group: dict[dist.ProcessGroup | None, list[torch.Tensor]] = {}
    for param in params:
        if get_outer_layout(param) is not None:
            raise NotImplementedError(
                "clip_grad_norm_ cannot combine the norm of a parameter with an "
                "outer (e.g. tensor- or expert-parallel) layout over its outer ranks"
            )
        mesh = get_mesh(param)
        group = None if mesh is None else mesh.get_group()
        grads_by_group.setdefault(group, []).append(param.grad)  # type: ignore[arg-type]
    if not grads_by_group:
        return torch.tensor(0.0)
    norms = []
    for group, grads in grads_by_group.items():
        # A rank may hold no part of a gradient (e.g. one it does not own),
        # which has no infinity norm; it still joins its mesh's all-reduce
        nonempty = [grad for grad in grads if grad.numel() > 0]
        norm = (
            torch.nn.utils.get_total_norm(nonempty, norm_type, False, foreach)
            if nonempty
            else grads[0].new_zeros(())
        )
        if group is not None and dist.get_world_size(group) > 1:
            norm = _all_reduce_norm(norm, norm_type, group)
        norms.append(norm)
    device = norms[0].device
    return torch.linalg.vector_norm(
        torch.stack([norm.to(device) for norm in norms]), norm_type
    )


def _all_reduce_norm(
    norm: torch.Tensor, norm_type: float, group: dist.ProcessGroup
) -> torch.Tensor:
    """The norm of the disjoint parts that the ranks of ``group`` hold, from each
    rank's norm of its part."""
    if math.isinf(norm_type):
        op = dist.ReduceOp.MAX if norm_type > 0 else dist.ReduceOp.MIN
        dist.all_reduce(norm, op=op, group=group)
        return norm
    if norm_type == 0:  # counts nonzero elements
        dist.all_reduce(norm, group=group)
        return norm
    norm = norm.pow(norm_type)
    dist.all_reduce(norm, group=group)
    return norm.pow(1.0 / norm_type)
