# Copyright (c) Meta Platforms, Inc. and affiliates
# GDN context parallel: Megatron-style all-to-all, not Ring Attention.
#
# Sequence-parallel activations become head-parallel so the gated-delta
# recurrence (and the causal conv in ``GatedDeltaNet``) see the full
# timeline, then the inverse all-to-all restores the sequence shard.
#
# Collectives go through ``all_to_all_single_autograd`` so gradients flow
# back to in-proj / conv / A_log, not only through the residual.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn.attention.gated_delta as _gdn_mod
import torch.nn.functional as F
from torch.nn.attention.gated_delta import (
    _gated_delta_rule_impl as _gdn_local,
    gated_delta_rule as _gdn_public,
)


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


_cp_mesh: DeviceMesh | None = None


def current_cp_mesh() -> DeviceMesh | None:
    return _cp_mesh


def current_cp_group() -> dist.ProcessGroup | None:
    if _cp_mesh is None:
        return None
    return _cp_mesh.get_group()


def _cp_group(mesh: DeviceMesh) -> dist.ProcessGroup:
    return mesh.get_group()


def _all_to_all_single(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Dim-0 all-to-all that participates in autograd."""
    y = funcol.all_to_all_single_autograd(x.contiguous(), None, None, group)
    wait = getattr(y, "wait", None)
    if callable(wait):
        out = wait()
        if isinstance(out, torch.Tensor):
            return out
    if not isinstance(y, torch.Tensor):
        raise TypeError("all_to_all_single_autograd must return a Tensor")
    return y


def _plain(t: torch.Tensor) -> torch.Tensor:
    fn = getattr(t, "full_tensor", None)
    if not callable(fn):
        return t
    out = fn()
    return out if isinstance(out, torch.Tensor) else t


def slice_param_cp(
    param: torch.Tensor, dim: int, group: dist.ProcessGroup
) -> torch.Tensor:
    param = _plain(param)
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    size = param.size(dim)
    if size % world != 0:
        raise ValueError(f"param dim {size} not divisible by cp={world}")
    local = size // world
    slc = [slice(None)] * param.ndim
    slc[dim] = slice(rank * local, (rank + 1) * local)
    return param[tuple(slc)]


def slice_sections_cp(
    param: torch.Tensor, sections: list[int], dim: int, group: dist.ProcessGroup
) -> torch.Tensor:
    parts = torch.split(_plain(param), sections, dim=dim)
    return torch.cat([slice_param_cp(p, dim, group) for p in parts], dim=dim)


def a2a_seq_to_feat(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """``[B, S_local, F]`` → ``[B, S_full, F_local]``."""
    world = dist.get_world_size(group)
    if world == 1:
        return x
    if x.size(-1) % world != 0:
        raise ValueError(f"feat {x.size(-1)} not divisible by cp={world}")
    batch, seq_local, feat = x.shape
    feat_local = feat // world
    x = x.reshape(batch, seq_local, world, feat_local).permute(2, 1, 0, 3).contiguous()
    out = _all_to_all_single(x, group)
    return (
        out.permute(2, 0, 1, 3)
        .reshape(batch, world * seq_local, feat_local)
        .contiguous()
    )


def a2a_feat_to_seq(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """``[B, S_full, F_local]`` → ``[B, S_local, F]``."""
    world = dist.get_world_size(group)
    if world == 1:
        return x
    batch, seq_full, feat_local = x.shape
    if seq_full % world != 0:
        raise ValueError(f"seq {seq_full} not divisible by cp={world}")
    seq_local = seq_full // world
    x = x.reshape(batch, world, seq_local, feat_local).permute(1, 2, 0, 3).contiguous()
    out = _all_to_all_single(x, group)
    return (
        out.permute(2, 1, 0, 3)
        .reshape(batch, seq_local, world * feat_local)
        .contiguous()
    )


def a2a_seq_to_feat_sections(
    x: torch.Tensor, sections: list[int], group: dist.ProcessGroup
) -> torch.Tensor:
    parts = torch.split(x, sections, dim=-1)
    return torch.cat([a2a_seq_to_feat(p, group) for p in parts], dim=-1)


def all_to_all_seq_to_head(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """[B, T_local, H, D] → [B, T_global, H_local, D]."""
    world = dist.get_world_size(group)
    if world == 1:
        return x
    if x.size(2) % world != 0:
        raise ValueError(f"num_heads {x.size(2)} must be divisible by cp_size {world}")
    batch, t_local, n_heads, dim = x.shape
    h_local = n_heads // world
    x = (
        x.reshape(batch, t_local, world, h_local, dim)
        .permute(2, 1, 0, 3, 4)
        .contiguous()
    )
    out = _all_to_all_single(x, group)
    out = out.permute(2, 0, 1, 3, 4).reshape(batch, world * t_local, h_local, dim)
    return out.contiguous()


def all_to_all_head_to_seq(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """[B, T_global, H_local, D] → [B, T_local, H, D]."""
    world = dist.get_world_size(group)
    if world == 1:
        return x
    batch, t_global, h_local, dim = x.shape
    if t_global % world != 0:
        raise ValueError(f"seq {t_global} must be divisible by cp_size {world}")
    t_local = t_global // world
    x = (
        x.reshape(batch, world, t_local, h_local, dim)
        .permute(1, 2, 0, 3, 4)
        .contiguous()
    )
    out = _all_to_all_single(x, group)
    out = out.permute(2, 1, 0, 3, 4).reshape(batch, t_local, world * h_local, dim)
    return out.contiguous()


def gated_delta_rule_cp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    mesh: DeviceMesh,
    **kwargs: Any,
) -> torch.Tensor:
    from torch.distributed.tensor.experimental._context_parallel._attention import (
        _cp_options,
    )

    if _cp_options.enable_load_balance:
        raise RuntimeError(
            "GatedDeltaNet CP needs contiguous sequence shards; "
            "set _cp_options.enable_load_balance = False"
        )
    group = _cp_group(mesh)
    q = all_to_all_seq_to_head(query, group)
    k = all_to_all_seq_to_head(key, group)
    v = all_to_all_seq_to_head(value, group)
    decay_h = all_to_all_seq_to_head(decay.unsqueeze(-1), group).squeeze(-1)
    beta_h = all_to_all_seq_to_head(beta.unsqueeze(-1), group).squeeze(-1)
    out = _gdn_local(q, k, v, decay_h, beta_h, **kwargs)
    return all_to_all_head_to_seq(out, group)


def patch_gated_delta_rule(mesh: DeviceMesh) -> None:
    global _cp_mesh
    _cp_mesh = mesh

    def _wrapped(*args, **kwargs):
        return gated_delta_rule_cp(*args, mesh=mesh, **kwargs)

    F.gated_delta_rule = _wrapped  # type: ignore[attr-defined]
    _gdn_mod.gated_delta_rule = _wrapped  # type: ignore[misc]


def restore_gated_delta_rule() -> None:
    global _cp_mesh
    _cp_mesh = None
    F.gated_delta_rule = _gdn_public  # type: ignore[attr-defined]
    _gdn_mod.gated_delta_rule = _gdn_public  # type: ignore[misc]
