# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
from functools import partial
from typing import Any, Optional

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
)

__all__ = [
    "tp_checkpoint_wrapper",
]


class TPCheckpointWrapper(CheckpointWrapper):
    """
    An nn.Module that wraps another nn.Module with checkpointing.
    This is a wrapper created dedicated for Tensor Parallel so that
    replicate input can be sharded after forward to save memory.
    To shard and restore the input, we use
    `torch.autograd.graph.saved_tensors_hooks`.
    """

    def __init__(
        self,
        mod: torch.nn.Module,
        tp_device_mesh: DeviceMesh,
        input_reshard_dim: Optional[int] = None,
    ):
        super().__init__(mod, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        self.mesh = tp_device_mesh
        self.input_reshard_dim = input_reshard_dim

    def forward(self, *args, **kwargs):
        cx = (
            torch.autograd.graph.saved_tensors_hooks(
                partial(_pack_hook_tp, self.mesh, self.input_reshard_dim),
                partial(_unpack_hook_tp, self.mesh, self.input_reshard_dim),
            )
            if self.input_reshard_dim is not None
            else contextlib.suppress()
        )
        with cx:  # type: ignore[attr-defined]
            return super().forward(*args, **kwargs)


def tp_checkpoint_wrapper(
    module: torch.nn.Module,
    tp_device_mesh: DeviceMesh,
    input_reshard_dim: Optional[int] = None,
) -> torch.nn.Module:
    """
    Wrap an nn.Module with activation checkpointing so that we can shard
    per the given `tp_device_mesh` and `input_reshard_dim` and restore the
    input back when recomputing the activations in the backward. The reason
    why we can do this is that for Tensor Parallel(TP), the input are same
    across all TP ranks.

    Args:
        module (:class:`nn.Module`):
            Module to be wrapped with checkpointing.
        tp_device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for Tensor Parallel.
        input_reshard_dim (Optional[int]):
            The dimension of where we perform the sharding
            of input. If set None, there is no sharding of input.
            Default: None

    Return:
        A :class:`nn.Module` object wrapped with TP activation checkpointing.
    """
    return TPCheckpointWrapper(module, tp_device_mesh, input_reshard_dim)


def _pack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: torch.Tensor) -> Any:
    """
    Hook functions called after FWD to shard input.
    """

    if isinstance(x, DTensor) and all(p.is_replicate() for p in x._spec.placements):
        return x.redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, torch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(x, device_mesh=mesh)
            .redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
            .to_local()
        )
    else:
        return x


def _unpack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: Any) -> torch.Tensor:
    """
    Hook functions called before activation recomputing in BWD to restore input.
    """

    if (
        isinstance(x, DTensor)
        and len(x._spec.placements) == 1
        and x._spec.placements[0].is_shard()
    ):
        return x.redistribute(device_mesh=mesh, placements=[Replicate()])
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, torch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(
                x, device_mesh=mesh, placements=[Shard(input_reshard_dim)]
            )
            .redistribute(device_mesh=mesh, placements=[Replicate()])
            .to_local()
        )
    else:
        return x
