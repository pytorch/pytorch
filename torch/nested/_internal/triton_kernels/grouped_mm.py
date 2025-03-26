# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import datetime
import itertools
import random
import sys

import triton
import triton.language as tl

import torch
from torch.nested._internal.triton_kernels.utils import do_bench


def gen_configs():
    products = itertools.product(
        [64, 128, 256],
        [64, 128, 256],
        [64, 128, 256],
        [4, 8],
        [4, 8],
        [4, 8],
    )
    configs = []
    for config in products:
        (
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            num_stages,
            num_warps,
            GROUP_SIZE_M,
        ) = config
        configs += [
            {
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "num_stages": num_stages,
                "num_warps": num_warps,
                "GROUP_SIZE_M": GROUP_SIZE_M,
            },
        ]
    return configs


@triton.jit
def grouped_matmul_kernel(
    b_ptr,
    group_size,
    gn,
    gk,
    ldb,
    a_offsets_ptr,
    a_ptr,
    lda,
    ldc,
    c_ptr,
    max_M,
    # # number of virtual SM
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(max_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(gn, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    tile_m_idx = pid_m
    tile_n_idx = pid_n

    # tile_m_idx = tl.program_id(0)
    # tile_n_idx = tl.program_id(1)

    batch_id = tl.program_id(1)

    # get the gemm size of the current problem
    a_offset_0 = tl.load(a_offsets_ptr + batch_id, eviction_policy="evict_last").to(
        tl.int32
    )
    a_offset_1 = tl.load(a_offsets_ptr + batch_id + 1, eviction_policy="evict_last").to(
        tl.int32
    )
    gm = a_offset_1 - a_offset_0
    num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
    if tile_m_idx < num_m_tiles:
        # pick up a tile from the current gemm problem
        # figure out tile coordinates

        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        offs_am = offs_am % gm
        offs_bn = offs_bn % gn

        # Rematerialize on each loop
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + a_offset_0 * lda + (offs_am[:, None]) * lda + offs_k[None, :]
        b_ptrs = b_ptr + batch_id * gk * gn + offs_k[:, None] * ldb + (offs_bn[None, :])
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for kk in range(0, tl.cdiv(gk, BLOCK_SIZE_K)):
            # # # hint to Triton compiler to do proper loop pipelining
            tl.multiple_of(a_ptrs, [16, 16])
            tl.multiple_of(b_ptrs, [16, 16])
            a = tl.load(
                a_ptrs, mask=offs_k[None, :] < gk - kk * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k[:, None] < gk - kk * BLOCK_SIZE_K, other=0.0
            )
            accumulator += tl.dot(a, b)  # , accumulator) #, "ieee")
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K * ldb
        c = accumulator.to(tl.float16)

        # offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        # offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Trying to save registers by recomputing indices
        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        a_offset_0 = tl.load(a_offsets_ptr + batch_id, eviction_policy="evict_last").to(
            tl.int32
        )
        a_offset_1 = tl.load(
            a_offsets_ptr + batch_id + 1, eviction_policy="evict_last"
        ).to(tl.int32)
        gm = a_offset_1 - a_offset_0

        c_ptrs = c_ptr + a_offset_0 * ldc + ldc * offs_am[:, None] + offs_bn[None, :]
        c_mask = (offs_am[:, None] < gm) & (offs_bn[None, :] < gn)

        # assumes full tile for now
        tl.store(c_ptrs, c, mask=c_mask)


def group_gemm_fn_kernel(a_values, a_offsets, max_M, tensor_b, c_values, config):
    B, K, N = tensor_b.shape
    group_size = a_offsets.size(0)
    grid = (
        triton.cdiv(max_M, config["BLOCK_SIZE_M"])
        * triton.cdiv(N, config["BLOCK_SIZE_N"]),
        group_size,
    )
    grouped_matmul_kernel[grid](
        tensor_b,
        group_size,
        N,
        K,
        tensor_b.stride(1),
        a_offsets,
        a_values,
        a_values.stride(0),
        c_values.stride(0),
        c_values,
        max_M,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        num_stages=config["num_stages"],
        num_warps=config["num_warps"],
    )

    return c_values


BEST_CONFIGS = {}
BEST_CONFIGS[
    (torch.Size([131072, 4096]), torch.Size([9]), 16384, torch.Size([8, 4096, 14336]))
] = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "num_stages": 4,
    "num_warps": 8,
    "GROUP_SIZE_M": 8,
}


def gen_config_key(a_values, a_offsets, max_M, tensor_b):
    return (a_values.size(), a_offsets.size(), max_M, tensor_b.size())


@torch.library.custom_op("triton_kernels::group_gemm_fn", mutates_args=())
def group_gemm_fn(
    a_values: torch.Tensor, a_offsets: torch.Tensor, max_M: int, tensor_b: torch.Tensor
) -> torch.Tensor:
    assert not tensor_b.is_nested
    group_size = a_offsets.size(0)

    assert tensor_b.is_contiguous()

    assert a_values.dim() == 2

    B, K, N = tensor_b.shape

    c_values = a_values.new_empty((a_values.size(0), N))
    config_key = gen_config_key(a_values, a_offsets, max_M, tensor_b)
    if config_key in BEST_CONFIGS:
        best_config = BEST_CONFIGS[config_key]
    else:
        current_timestamp = datetime.datetime.now()
        print_prefix = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(
            print_prefix,
            f"Don't have a config for config_key {config_key}. Need to run autotuning.",
        )
        best_ms, best_config = None, None
        all_configs = gen_configs()
        # Use a random order to increase chance of finding a good config early.
        random.shuffle(all_configs)
        for i, config in enumerate(all_configs):
            current_timestamp = datetime.datetime.now()
            print_prefix = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            print(print_prefix, end=" ")
            sys.stdout.flush()
            ms = do_bench(
                group_gemm_fn_kernel,
                [a_values, a_offsets, max_M, tensor_b, c_values, config],
                best_ms,
            )
            print(
                f"     ms: {ms} with config number {i} out of {len(all_configs)}: {config}"
            )
            if not isinstance(ms, str) and best_ms is None:
                best_ms, best_config = ms, config
                print(print_prefix, f"best_ms: {best_ms} with config: {best_config}")
            if not isinstance(ms, str) and best_ms > ms:
                best_ms, best_config = ms, config
                print(print_prefix, f"best_ms: {best_ms} with config: {best_config}")
        if best_config is None:
            raise ValueError("Could not find valid config.")
        BEST_CONFIGS[config_key] = best_config
        print(f"Found config {best_config} with ms {best_ms}.")

    return group_gemm_fn_kernel(
        a_values, a_offsets, max_M, tensor_b, c_values, best_config
    )


@group_gemm_fn.register_fake
def group_gemm_fn_meta(
    a_values: torch.Tensor, a_offsets: torch.Tensor, max_M: int, tensor_b: torch.Tensor
) -> torch.Tensor:
    B, K, N = tensor_b.shape
    c_values = a_values.new_empty((a_values.size(0), N))
    return c_values


def grouped_mm(tensor_a, tensor_b):
    assert tensor_a.is_nested
    assert not tensor_b.is_nested
    assert tensor_a.size(0) == tensor_b.size(0)
    group_size = tensor_a.size(0)

    assert tensor_b.is_contiguous()

    a_values = tensor_a.values()
    a_offsets = tensor_a.offsets()

    assert a_values.dim() == 2

    max_M = tensor_a._max_seqlen

    c_values = group_gemm_fn(a_values, a_offsets, max_M, tensor_b)

    return torch.nested.nested_tensor_from_jagged(c_values, offsets=a_offsets)
