# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._flex_shard import BucketSpec, flex_shard
from torch.distributed.fsdp._flex_shard.bucket_storage import ParamInfo
from torch.distributed.fsdp._flex_shard.placement_contract import Placement
from torch.distributed.fsdp._flex_shard.placements.shard import per_param_placements
from torch.distributed.fsdp._flex_shard.utils import (  # noqa: F401
    alloc_grads_in_param_layout,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


@contextmanager
def single_rank_cpu_mesh() -> Iterator:
    """Create a single-rank CPU mesh for normal pytest unit tests."""
    created_pg = False
    # The file store can delete its file itself; a temporary directory's
    # cleanup tolerates that, but Python 3.11's NamedTemporaryFile raises.
    with TemporaryDirectory() as store_dir:
        if not dist.is_initialized():
            dist.init_process_group(
                "gloo",
                init_method=f"file://{store_dir}/store",
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=20),
            )
            created_pg = True
        try:
            yield init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
        finally:
            if created_pg and dist.is_initialized():
                dist.destroy_process_group()


@contextmanager
def single_rank_cuda_mesh() -> Iterator:
    """Create a single-rank CUDA mesh for FlexShard runtime tests."""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required for FlexShard runtime tests.")
    torch.cuda.set_device(0)
    created_pg = False
    with TemporaryDirectory() as store_dir:
        if not dist.is_initialized():
            dist.init_process_group(
                "nccl",
                init_method=f"file://{store_dir}/store",
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=20),
            )
            created_pg = True
        try:
            yield init_device_mesh("cuda", (1,), mesh_dim_names=("fsdp",))
        finally:
            if created_pg and dist.is_initialized():
                dist.destroy_process_group()


def flex_shard_cuda(
    model: nn.Module,
    mesh,
    buckets: list[BucketSpec] | None = None,
) -> nn.Module:
    """Apply FlexShard with single-rank CUDA eager settings."""
    if buckets is None:
        buckets = [
            BucketSpec(
                ["*"],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=False,
            )
        ]
    return flex_shard(
        model,
        buckets=buckets,
    )


def make_test_sgd(params, *, lr: float):
    return torch.optim.SGD(params, lr=lr)  # noqa: CITRINE(missing_for_each_optimizer)


def flex_shard_transformer_model(mesh) -> tuple[ModelArgs, Transformer]:
    """Return a small Transformer and args after applying FlexShard."""
    args, model = make_transformer_model()
    flex_shard_cuda(
        model,
        mesh,
        buckets=transformer_bucket_specs(
            args.n_layers, mesh, reshard_after_forward=False
        ),
    )
    return args, model


def make_transformer_model(
    *,
    device: torch.device | str | None = None,
    n_layers: int = 1,
    vocab_size: int = 16,
    max_seq_len: int = 8,
    dim: int = 8,
    n_heads: int = 2,
    weight_tying: bool = False,
) -> tuple[ModelArgs, Transformer]:
    """Build the small internal Transformer used across FlexShard tests."""
    args = ModelArgs(
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dim=dim,
        n_heads=n_heads,
        dropout_p=0.0,
        use_attn_mask=True,
        weight_tying=weight_tying,
        checkpoint_activations=False,
    )
    model = Transformer(args)
    if device is not None:
        model = model.to(device)
    return args, model


def transformer_inputs(
    args: ModelArgs,
    *,
    batch_size: int = 2,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create token inputs for the shared test Transformer."""
    return torch.randint(
        0,
        args.vocab_size,
        (batch_size, args.max_seq_len),
        device=device,
    )


def expected_shard(
    tensor: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """Return the Shard(dim) local tensor expected on one rank."""
    chunks = list(torch.chunk(tensor, world_size, dim=dim))
    empty_shape = list(tensor.shape)
    empty_shape[dim] = 0
    while len(chunks) < world_size:
        chunks.append(tensor.new_empty(empty_shape))
    return chunks[rank].contiguous()


def check_flex_shard_parity(
    cls,
    reference_module: nn.Module,
    flex_sharded_module: nn.Module,
    rank: int,
    world_size: int,
) -> None:
    """Check FlexShard local params/state_dict/grads against a reference module."""
    for (ref_name, ref_param), (name, param) in zip(
        reference_module.named_parameters(),
        flex_sharded_module.named_parameters(),
        strict=True,
    ):
        cls.assertEqual(ref_name, name)
        cls.assertEqual(
            param.detach(),
            expected_shard(ref_param.detach(), rank=rank, world_size=world_size),
        )
        if ref_param.grad is None:
            cls.assertIsNone(param.grad)
            continue
        cls.assertIsNotNone(param.grad)
        cls.assertEqual(
            param.grad.detach(),
            expected_shard(ref_param.grad.detach(), rank=rank, world_size=world_size),
        )

    ref_state_dict = reference_module.state_dict()
    state_dict = flex_sharded_module.state_dict()
    cls.assertEqual(list(ref_state_dict), list(state_dict))
    for key, value in state_dict.items():
        cls.assertEqual(
            value,
            expected_shard(ref_state_dict[key], rank=rank, world_size=world_size),
        )


def transformer_bucket_specs(
    num_layers: int,
    mesh,
    *,
    reshard_after_forward: bool = False,
) -> list[BucketSpec]:
    """Bucket the shared Transformer by top-level execution units."""
    return (
        [
            BucketSpec(
                ["tok_embeddings.*"],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
            ),
            BucketSpec(
                ["pos_embeddings.*"],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
            ),
        ]
        + [
            BucketSpec(
                [f"layers.{idx}.*"],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
            )
            for idx in range(num_layers)
        ]
        + [
            BucketSpec(
                ["norm.*"],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
            ),
            BucketSpec(
                ["output.*"],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
            ),
        ]
    )


class FakeMesh:
    """Minimal mesh for exercising bucket_storage_layout without a real PG."""

    def __init__(self, size: int, rank: int = 0) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank

    def get_group(self):
        return None


def make_param_infos(
    named_params: list[tuple[str, nn.Parameter]],
    placement: Placement,
    mesh: FakeMesh,
) -> list[ParamInfo]:
    """ParamInfos for params sharing one placement's bucket storage layout."""
    placements = {fqn: (placement,) for fqn, _ in named_params}
    layout = placement.bucket_storage_layout(
        named_params,
        placements,
        mesh,
    )
    return [
        ParamInfo(
            fqn=fqn,
            global_shape=param.shape,
            global_stride=tuple(param.stride()),
            dtype=param.dtype,
            requires_grad=param.requires_grad,
            placements=(placement,),
            local_shape=layout.param_layouts[fqn].local_shape,
            local_numel=layout.param_layouts[fqn].local_numel,
            byte_offset=layout.param_layouts[fqn].byte_offset,
            storage_nbytes=layout.param_layouts[fqn].storage_nbytes,
            global_numel=param.numel(),
            bucket_layout=layout.param_layouts[fqn].bucket_layout,
        )
        for fqn, param in named_params
    ]
