"""
To run the example, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=4 context_parallel_example.py
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental._attention import context_parallel
from torch.nn.attention import sdpa_kernel, SDPBackend
from triton.testing import do_bench


def sdpa_example(world_size: int, rank: int) -> None:
    assert torch.cuda.is_available()
    torch.cuda.manual_seed(0)

    batch = 8
    nheads = 8
    qkv_len = 8192
    dim = 32
    backend = SDPBackend.FLASH_ATTENTION
    dtype = (
        torch.bfloat16
        if backend == SDPBackend.FLASH_ATTENTION
        or backend == SDPBackend.CUDNN_ATTENTION
        else torch.float32
    )

    torch.cuda.set_device(f"cuda:{rank}")
    qkv = [
        torch.rand(
            (batch, nheads, qkv_len, dim),
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(3)
    ]

    sdpa_func = lambda: F.scaled_dot_product_attention(*qkv, is_causal=True)
    with sdpa_kernel(backend):
        fwd_time = do_bench(sdpa_func)
        print(f"fwd_time: {fwd_time:.4f} ms")


def context_parallel_sdpa_example(world_size: int, rank: int) -> None:
    assert torch.cuda.is_available()
    assert dist.is_nccl_available()
    torch.cuda.manual_seed(0)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    device_mesh = init_device_mesh(
        device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("cp",)
    )

    batch = 8
    nheads = 8
    qkv_len = 8192
    dim = 32
    backend = SDPBackend.FLASH_ATTENTION
    dtype = (
        torch.bfloat16
        if backend == SDPBackend.FLASH_ATTENTION
        or backend == SDPBackend.CUDNN_ATTENTION
        else torch.float32
    )

    torch.cuda.set_device(f"cuda:{rank}")
    qkv = [
        torch.rand(
            (batch, nheads, qkv_len, dim),
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    cp_qkv = [t.detach().clone() for t in qkv]

    def cp_sdpa_func():
        with context_parallel(
            device_mesh, buffers=tuple(cp_qkv), buffer_seq_dims=(2, 2, 2)
        ):
            F.scaled_dot_product_attention(*qkv, is_causal=True)

    with sdpa_kernel(backend):
        fwd_time = do_bench(cp_sdpa_func)
        print(f"fwd_time: {fwd_time:.4f} ms")


if __name__ == "__main__":
    # this script is launched via torchrun which automatically manages ProcessGroup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # try:
    # sdpa_example(world_size, rank)
    context_parallel_sdpa_example(world_size, rank)
    # finally:
    # dist.barrier()
    # dist.destroy_process_group()
