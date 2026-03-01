"""
To run the example, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=4 flex_attention_cp.py
"""

import os
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Partial, Shard
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)


def get_device_type() -> str:
    return "cuda"


@lru_cache
def create_block_mask_cached(
    score_mod: _mask_mod_signature,
    B: int | None,
    H: int | None,
    M: int,
    N: int,
    device: str = "cuda",
) -> BlockMask:
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def flex_attn_example(world_size: int, rank: int) -> None:
    device_type = get_device_type()
    device_handle = getattr(torch, device_type, None)
    if device_handle is None:
        raise AssertionError(f"Unsupported device type: {device_type}")
    num_devices_per_host = device_handle.device_count()
    device_handle.set_device(rank % num_devices_per_host)
    torch._dynamo.config.cache_size_limit = 1000

    # init device mesh
    device_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(world_size,),
        mesh_dim_names=("cp",),
    )

    def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        return q_idx >= kv_idx

    # Compile the flex_attention function
    compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

    # init input
    torch.manual_seed(10)
    dtype = torch.float32
    B = 8
    H = 8
    S = 32 * world_size
    D = 32

    qkv = [
        torch.rand(
            (B, H, S, D),
            device=device_type,
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(3)
    ]

    # input distribution
    seq_dim = 2
    qkv_dist = [
        distribute_tensor(
            t.detach().clone().requires_grad_(), device_mesh, [Shard(seq_dim)]
        )
        for t in qkv
    ]

    # local forward pass
    block_mask = create_block_mask_cached(
        causal_mask,
        B=1,
        H=1,
        M=S,
        N=S,
        device=device_type,
    )

    q, k, v = qkv
    out = compiled_flex_attention(q, k, v, score_mod=None, block_mask=block_mask)
    if not isinstance(out, torch.Tensor):
        raise AssertionError
    expect_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, expect_out, atol=1e-1, rtol=1e-2)

    # context parallel forward pass
    def rewrite_mask_mod_for_cp(
        mask_mod: _mask_mod_signature,
        rank: int,
        shard_size: int,
    ) -> _mask_mod_signature:
        # since we're sharding on `seq_dim`, global q_idx is mapped to q_idx % shard_size
        # on each rank which means q_idx = q_idx_on_rank + shard_size * rank
        return lambda b, h, q_idx, kv_idx: mask_mod(
            b, h, q_idx + rank * shard_size, kv_idx
        )

    # manually do context parallel on attention
    # the input hook of Context Parallel
    q_local = qkv_dist[0].to_local()

    # kv all-gather
    # NOTE: we don't consider load-balance for now
    # NOTE: wait() is immediately called in all_gather_tensor when gather_dim != 0
    k_full, v_full = (t.full_tensor(grad_placements=[Partial()]) for t in qkv_dist[1:])

    # rewrite `block_mask`
    mask_mod: _mask_mod_signature = block_mask.mask_mod
    shard_size = S // world_size
    cp_mask_mod = rewrite_mask_mod_for_cp(mask_mod, rank, shard_size)
    cp_block_mask = create_block_mask_cached(
        cp_mask_mod, B=1, H=1, M=shard_size, N=S, device=device_type
    )

    # TODO: this doesn't address the return_lse=True case
    cp_out = compiled_flex_attention(
        q_local,
        k_full,
        v_full,
        score_mod=None,
        block_mask=cp_block_mask,
    )
    if not isinstance(cp_out, torch.Tensor):
        raise AssertionError

    # wrap the local output into a DTensor
    cp_out_dist = DTensor.from_local(cp_out, device_mesh, [Shard(seq_dim)])
    # compare with the flex_attention output
    torch.testing.assert_close(cp_out_dist.full_tensor(), out, atol=1e-1, rtol=1e-2)

    # local backward pass
    grad_out = torch.randn(
        (B, H, S, D),
        device=device_type,
        dtype=dtype,
    )
    grad_out_dist = distribute_tensor(
        grad_out.detach().clone().requires_grad_(), device_mesh, [Shard(seq_dim)]
    )

    out.backward(grad_out)
    grad1 = [t.grad for t in qkv]
    for t in qkv:
        t.grad = None

    expect_out.backward(grad_out)
    grad2 = [t.grad for t in qkv]
    for t in qkv:
        t.grad = None

    for flex_grad, expect_grad in zip(grad1, grad2):
        torch.testing.assert_close(flex_grad, expect_grad, atol=1e-1, rtol=1e-2)

    # context parallel backward pass
    cp_out.backward(grad_out_dist.to_local())

    for cp_flex_grad_dist, expect_grad in zip([t.grad for t in qkv_dist], grad2):
        if not isinstance(cp_flex_grad_dist, DTensor):
            raise AssertionError
        torch.testing.assert_close(
            cp_flex_grad_dist.full_tensor(), expect_grad, atol=1e-1, rtol=1e-2
        )


if __name__ == "__main__":
    # this script is launched via torchrun which automatically manages ProcessGroup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # assert world_size == 4  # our example uses 4 worker ranks

    try:
        flex_attn_example(world_size, rank)
    finally:
        dist.barrier()
        dist.destroy_process_group()
