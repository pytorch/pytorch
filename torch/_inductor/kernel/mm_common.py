# mypy: allow-untyped-defs
import functools
import itertools
import logging
from typing import cast, List, Tuple

import sympy

import torch
from torch._inductor.select_algorithm import realize_inputs
from torch._inductor.virtualized import V

from .. import config as inductor_config
from ..runtime.runtime_utils import next_power_of_2
from ..utils import ceildiv as cdiv

log = logging.getLogger(__name__)


def triton_config(num_stages, num_warps, **kwargs):
    from triton import Config

    return Config(kwargs, num_stages=num_stages, num_warps=num_warps)


def filtered_configs(
    m: int,
    n: int,
    k: int,
    configs: List[Tuple[int, int, int, int, int]],
    has_int8_tensor=False,
):
    """Heuristic to shrink configs when they are bigger than the input size"""

    min_block_size = 16
    # block_k=16 seems to be causing issues
    # see: https://github.com/triton-lang/triton/issues/2156#issuecomment-1695897424
    min_block_size_k = 32 if has_int8_tensor else 16
    m = max(
        next_power_of_2(
            V.graph.sizevars.size_hint(
                m, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
            )
        ),
        min_block_size,
    )
    n = max(
        next_power_of_2(
            V.graph.sizevars.size_hint(
                n, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
            )
        ),
        min_block_size,
    )
    k = max(
        next_power_of_2(
            V.graph.sizevars.size_hint(
                k, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
            )
        ),
        min_block_size_k,
    )
    used = set()
    for block_m, block_n, block_k, num_stages, num_warps in configs:
        # shrink configs for small sizes
        block_m = max(min(block_m, m), min_block_size)
        block_n = max(min(block_n, n), min_block_size)
        block_k = max(min(block_k, k), min_block_size_k)
        # each warp computes 16x16 tile = 256
        num_warps = min(num_warps, block_m * block_n // 256)
        if torch.version.hip:
            for matrix_instr_nonkdim in [0, 16]:
                if matrix_instr_nonkdim != 0 and (
                    block_m % matrix_instr_nonkdim != 0
                    or block_n % matrix_instr_nonkdim != 0
                ):
                    #  block_m and block_n must be a multiple of matrix_instr_nonkdim
                    continue
                if (
                    block_m,
                    block_n,
                    block_k,
                    num_stages,
                    num_warps,
                    matrix_instr_nonkdim,
                ) not in used:
                    used.add(
                        (
                            block_m,
                            block_n,
                            block_k,
                            num_stages,
                            num_warps,
                            matrix_instr_nonkdim,
                        )
                    )
                    yield triton_config(
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                        BLOCK_K=block_k,
                        num_stages=num_stages,
                        num_warps=num_warps,
                        matrix_instr_nonkdim=matrix_instr_nonkdim,
                    )
        else:
            if (block_m, block_n, block_k, num_stages, num_warps, 0) not in used:
                used.add((block_m, block_n, block_k, num_stages, num_warps, 0))
                yield triton_config(
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_stages=num_stages,
                    num_warps=num_warps,
                )


# List of dictionaries to store the kernel configs. Configs that evaluate to true
# will be utilised on the target platform. The configs are as follows:
# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
mm_kernel_configs = (
    [
        {"config": (32, 32, 16, 1, 2), "cond": True},
        {"config": (32, 32, 128, 2, 4), "cond": torch.version.hip is None},
        {"config": (32, 64, 32, 5, 8), "cond": True},
        {"config": (64, 32, 32, 5, 8), "cond": True},
        {"config": (64, 32, 128, 5, 4), "cond": True},
        {"config": (64, 64, 16, 2, 4), "cond": True},
        {"config": (64, 64, 32, 2, 4), "cond": True},
        {"config": (64, 64, 64, 3, 8), "cond": True},
        {"config": (64, 64, 128, 5, 4), "cond": True},
        {"config": (64, 128, 32, 3, 4), "cond": True},
        {"config": (64, 128, 32, 4, 8), "cond": True},
        {"config": (64, 128, 64, 3, 4), "cond": True},
        {"config": (64, 128, 128, 4, 4), "cond": True},
        {"config": (128, 64, 32, 3, 4), "cond": True},
        {"config": (128, 64, 32, 4, 8), "cond": True},
        {"config": (128, 128, 32, 2, 8), "cond": True},
        {"config": (128, 128, 32, 3, 4), "cond": True},
        {"config": (128, 128, 64, 3, 4), "cond": True},
        {"config": (128, 128, 64, 5, 8), "cond": True},
    ]
    if inductor_config.max_autotune_gemm_search_space != "EXHAUSTIVE"
    else [
        {"config": (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps), "cond": True}
        for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
            [16, 32, 64, 128, 256], repeat=3
        )
        for num_stages in [1, 2, 3, 4, 5]
        for num_warps in [2, 4, 8]
    ]
)

int8_mm_kernel_configs = [
    {"config": (64, 64, 32, 2, 4), "cond": True},
    {"config": (64, 128, 32, 3, 4), "cond": True},
    {"config": (128, 64, 32, 3, 4), "cond": True},
    {"config": (64, 128, 32, 4, 8), "cond": True},
    {"config": (128, 64, 32, 4, 8), "cond": True},
    {"config": (64, 32, 32, 5, 8), "cond": True},
    {"config": (32, 64, 32, 5, 8), "cond": True},
    {"config": (128, 128, 32, 2, 8), "cond": True},
    {"config": (64, 64, 64, 3, 8), "cond": True},
    # {"config": (32, 32, 128, 2, 4), "cond": True},
    # {"config": (64, 64, 16, 2, 4), "cond": True},
    # {"config": (32, 32, 16, 1, 2), "cond": True},
    {"config": (128, 256, 128, 3, 8), "cond": torch.version.hip is None},
    {"config": (256, 128, 128, 3, 8), "cond": torch.version.hip is None},
]

# Mixed precision kernel configs for small sizes of m for mm's like (16, 8192) x (8192, 8192).
mixed_mm_kernel_configs_small_m = [
    {"config": (16, 128, 256, 3, 4), "cond": True},
    {"config": (16, 128, 256, 5, 8), "cond": True},
]

mixed_mm_kernel_configs = (
    mm_kernel_configs + mixed_mm_kernel_configs_small_m
    if inductor_config.max_autotune_gemm_search_space != "EXHAUSTIVE"
    else mm_kernel_configs
)

# Create filtered list of configs based on cond evaluation


mm_platform_configs = tuple(
    cast(Tuple[int, int, int, int, int], config["config"])
    for config in mm_kernel_configs
    if config["cond"]
)
int8_platform_configs = tuple(
    cast(Tuple[int, int, int, int, int], config["config"])
    for config in int8_mm_kernel_configs
    if config["cond"]
)
mixed_mm_platform_configs = tuple(
    cast(Tuple[int, int, int, int, int], config["config"])
    for config in mixed_mm_kernel_configs
    if config["cond"]
)

# On ROCm convert num_stages to 0 to enable software pipelining
if torch.version.hip:
    mm_platform_configs = tuple(
        (config[0], config[1], config[2], 0, config[4])
        for config in mm_platform_configs
    )
    int8_platform_configs = tuple(
        (config[0], config[1], config[2], 0, config[4])
        for config in mm_platform_configs
    )
    mixed_mm_platform_configs = tuple(
        (config[0], config[1], config[2], 0, config[4])
        for config in mixed_mm_platform_configs
    )

mm_configs = functools.partial(
    filtered_configs,
    configs=mm_platform_configs,
)

int8_mm_configs = functools.partial(
    filtered_configs,
    configs=int8_platform_configs,
)

mixed_mm_configs = functools.partial(
    filtered_configs,
    configs=mixed_mm_platform_configs,
)


def mm_grid(m, n, meta):
    """
    The CUDA grid size for matmul triton templates.
    """
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), 1, 1)


def acc_type(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return "tl.float32"
    return f"tl.{dtype}".replace("torch.", "")


def mm_options(config, sym_m, sym_n, sym_k, layout, b_prologue_cast_type=None):
    """
    Common options to matmul triton templates.
    """
    even_k_symbolic = (
        # it isn't worth guarding on this
        sympy.gcd(sym_k, config.kwargs["BLOCK_K"])
        == config.kwargs["BLOCK_K"]
    )
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32 and (
        not inductor_config.force_same_precision
        or ((sym_m % 16) == 0 and (sym_n % 16) == 0 and (sym_k % 8) == 0)
    )
    return dict(
        GROUP_M=8,
        EVEN_K=even_k_symbolic,
        ALLOW_TF32=allow_tf32,
        ACC_TYPE=acc_type(layout.dtype),
        B_PROLOGUE_CAST_TYPE=b_prologue_cast_type,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        **config.kwargs,
    )


def mm_args(mat1, mat2, *others, layout=None, out_dtype=None, use_4x2_dim=False):
    """
    Common arg processing for mm,bmm,addmm,etc
    """
    mat1, mat2 = realize_inputs(mat1, mat2)
    *b1, m, k1 = mat1.get_size()
    *b2, k2, n = mat2.get_size()
    b = [V.graph.sizevars.guard_equals(a, b) for a, b in zip(b1, b2)]
    if use_4x2_dim:
        k2 = k2 * 2
    k = V.graph.sizevars.guard_equals(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()
        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            [*b, m, n],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."

    from ..lowering import expand

    others = [realize_inputs(expand(x, layout.size)) for x in others]

    return [m, n, k, layout, mat1, mat2, *others]


def addmm_epilogue(dtype, alpha, beta):
    def epilogue(acc, bias):
        if alpha != 1:
            acc = V.ops.mul(acc, V.ops.constant(alpha, dtype))
        if beta != 1:
            bias = V.ops.mul(bias, V.ops.constant(beta, dtype))
        return V.ops.add(acc, bias)

    return epilogue
