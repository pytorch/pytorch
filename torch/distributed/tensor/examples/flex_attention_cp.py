"""
To run the example, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=1 flex_attention_cp.py
"""

import os

from functools import lru_cache

import torch
import torch.nn.functional as F
import torch.distributed._functional_collectives as ft_c

from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    create_block_mask,
    flex_attention,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

def get_device_type() -> str:
    return "cuda"


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def flex_attn_example(world_size: int, rank: int) -> None:
    device_type = get_device_type()
    device_handle = getattr(torch, device_type, None)
    assert device_handle is not None, f"Unsupported device type: {device_type}"
    num_devices_per_host = device_handle.device_count()
    device_handle.set_device(rank % num_devices_per_host)
    torch._dynamo.config.cache_size_limit = 1000

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # Compile the flex_attention function
    compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

    torch.manual_seed(10)
    dtype = torch.float32
    B = 8
    H = 8
    S = 64
    D = 32

    q = torch.rand(
        (B, H, S, D),
        device=device_type,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.rand(
        (B, H, S, D),
        device=device_type,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.rand(
        (B, H, S, D),
        device=device_type,
        dtype=dtype,
        requires_grad=True,
    )

    block_mask = create_block_mask_cached(
        causal_mask,
        B=1,
        H=1,
        M=S,
        N=S,
        device=device_type,
    )

    out = compiled_flex_attention(q, k, v, score_mod=None, block_mask=block_mask)

    expect_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    torch.testing.assert_close(out, expect_out, atol=1e-1, rtol=1e-2)

    def rewrite_mask_mod_for_cp(
        mask_mod: _mask_mod_signature,
        rank: int,
        shard_size: int,
    ) -> _mask_mod_signature:
        # since we're sharding on `seq_dim`, global q_idx is mapped to q_idx % shard_size
        # on each rank which means q_idx = q_idx_on_rank + shard_size * rank
        return (
            lambda b, h, q_idx, kv_idx: mask_mod(b, h, q_idx + rank * shard_size, kv_idx)
        )

    # create qkv shards
    device_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(world_size,),
        mesh_dim_names=("cp",),
    )

    # input distribution
    seq_dim = 2
    qkv_dist = [
        distribute_tensor(t, device_mesh, [Shard(seq_dim)]) for t in (q, k ,v)
    ]

    # manually do context parallel on attention
    # the input hook of Context Parallel
    qkv_local = [ t.to_local() for t in qkv_dist ]
    # kv all-gather
    # NOTE: we don't consider load-balance for now
    # NOTE: wait() is immediately called in all_gather_tensor when gather_dim != 0
    kv_gathered = [
        ft_c.all_gather_tensor(t.contiguous(), gather_dim=seq_dim, group=device_mesh)
        for t in qkv_local[1:]
    ]
    # rewrite `block_mask`
    mask_mod: _mask_mod_signature = block_mask.mask_mod
    shard_size = S // world_size
    cp_mask_mod = rewrite_mask_mod_for_cp(mask_mod, rank, shard_size)
    cp_block_mask = create_block_mask_cached(
        cp_mask_mod, B=1, H=1, M=shard_size, N=S, device=device_type
    )
    partial_out = compiled_flex_attention(qkv_local[0], *kv_gathered, block_mask=cp_block_mask)

    # update partial output
    

    

if __name__ == "__main__":
    # this script is launched via torchrun which automatically manages ProcessGroup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # assert world_size == 4  # our example uses 4 worker ranks

    flex_attn_example(world_size, rank)
