# mypy: allow-untyped-defs
"""Blackwell kernels used to form decompose-K partial products.

This module intentionally exposes only the partial template.  Choice wiring and
the final reduction remain separate so autotuning can compare complete plans.
"""

import dataclasses
import math
from typing import Any

import torch
from torch._inductor import inductor_prims, ir
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import SymbolicGridFn, TritonTemplate
from torch._inductor.utils import get_num_sms

from .mm_common import load_kernel_template


@SymbolicGridFn
def blackwell_decompose_k_partial_grid(
    flattened_m: int,
    n: int,
    meta: dict[str, Any],
    *,
    cdiv,
    min,
):
    """Launch one persistent MxN tile grid for every K partition."""
    m_pad = flattened_m // meta["BATCH_SIZE"]
    grid_m = cdiv(m_pad, meta["BLOCK_M"])
    grid_n = cdiv(n, meta["BLOCK_N"])
    grid_x = min(meta["NUM_SMS"], grid_m * grid_n)
    if meta["TWO_CTAS"]:
        grid_x = (grid_x // 2) * 2
    return (grid_x, meta["BATCH_SIZE"], 1)


blackwell_decompose_k_partial_template = TritonTemplate(
    name="blackwell_decompose_k_partial",
    grid=blackwell_decompose_k_partial_grid,
    source=load_kernel_template("triton_blackwell_ws_persistent_device_tma_bmm"),
    cache_codegen_enabled_for_template=True,
    prologue_loads_all_inputs=True,
)


@dataclasses.dataclass(frozen=True)
class BlackwellDecomposeKPartialConfig:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    epilogue_subtile: int
    two_ctas: bool


# Deliberately bounded proof set.  These are complete-plan winners from the
# production-shaped handwritten sweep, not a replacement for a future cost
# model.
BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS = (
    BlackwellDecomposeKPartialConfig(128, 128, 128, 3, 2, False),
    BlackwellDecomposeKPartialConfig(128, 128, 64, 4, 1, True),
    BlackwellDecomposeKPartialConfig(128, 256, 64, 6, 2, True),
)


def append_blackwell_decompose_k_partial_choice(
    choices: list[Any],
    input_nodes: tuple[ir.IRNode, ir.IRNode],
    layout: ir.Layout,
    *,
    k_split: int,
    config: BlackwellDecomposeKPartialConfig,
) -> None:
    """Append one standalone partial-GEMM choice over the original rank-2 inputs.

    The output layout is a flattened view of the physical contiguous
    ``[split, M_pad, N]`` FP32 workspace.  Keeping the descriptor rank two is
    intentional: current 2CTA descriptor-load/store transformation is rank-2.
    """
    mat1, mat2 = input_nodes
    m, k = map(int, mat1.get_size())
    k_b, n = map(int, mat2.get_size())
    if k != k_b:
        raise AssertionError(f"incompatible K dimensions: {k} and {k_b}")
    if layout.dtype != torch.float32:
        raise AssertionError("decompose-K partial workspace must be FP32")

    m_tiles = math.ceil(m / config.block_m)
    if config.two_ctas:
        m_tiles = math.ceil(m_tiles / 2) * 2
    m_pad = m_tiles * config.block_m
    expected_size = [k_split * m_pad, n]
    if list(map(int, layout.size)) != expected_size:
        raise AssertionError(
            f"expected flattened [split, M_pad, N] layout {expected_size}, "
            f"got {layout.size}"
        )

    k_part = math.ceil(math.ceil(k / k_split) / config.block_k) * config.block_k
    if (k_split - 1) * k_part >= k:
        raise NotImplementedError("aligned split leaves an empty final partition")

    kwargs: dict[str, Any] = {
        "BLOCK_M": config.block_m,
        "BLOCK_N": config.block_n,
        "BLOCK_K": config.block_k,
        "K_TILES": k_part // config.block_k,
        "GROUP_M": 8,
        "BATCH_SIZE": k_split,
        "LOGICAL_M": m,
        "LOGICAL_N": n,
        "DESCRIPTOR_K": k,
        "A_BATCH_STRIDE": 0,
        "B_BATCH_STRIDE": 0,
        "K_BATCH_OFFSET": k_part,
        "A_M_STRIDE": int(mat1.get_stride()[0]),
        "A_K_STRIDE": int(mat1.get_stride()[1]),
        "B_K_STRIDE": int(mat2.get_stride()[0]),
        "B_N_STRIDE": int(mat2.get_stride()[1]),
        "OUTPUT_BATCH_ROWS": m_pad,
        "NUM_SMS": min(get_num_sms(), m_tiles * math.ceil(n / config.block_n)),
        "A_ROW_MAJOR": mat1.get_stride()[1] == 1,
        "B_ROW_MAJOR": mat2.get_stride()[1] == 1,
        "ALLOW_TF32": False,
        "USE_META_WS": True,
        "WARP_SPECIALIZE": True,
        "FLATTEN": False,
        "DATA_PARTITION_FACTOR": 1,
        "SEPARATE_EPILOGUE_STORE": True,
        "EPILOGUE_SUBTILE": config.epilogue_subtile,
        "TWO_CTAS": config.two_ctas,
        "FLATTEN_OUTPUT": True,
        "tma_store": True,
        "transpose_discontiguous_tensor_descriptors_override": True,
    }
    if config.two_ctas:
        kwargs["ctas_per_cga"] = (2, 1, 1)

    error = blackwell_decompose_k_partial_template.maybe_append_choice(
        choices,
        input_nodes=input_nodes,
        layout=layout,
        call_sizes=layout.size,
        num_stages=config.num_stages,
        num_warps=8,
        **kwargs,
    )
    if error is not None:
        raise error


@register_lowering(
    inductor_prims.blackwell_decompose_k_partial,
    type_promotion_kind=None,
)
def lower_blackwell_decompose_k_partial(
    mat1,
    mat2,
    k_split: int,
    config_index: int,
):
    try:
        partial_config = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[int(config_index)]
    except IndexError as error:
        raise NotImplementedError(
            "unsupported Blackwell decompose-K partial config"
        ) from error
    except TypeError as error:
        raise NotImplementedError(
            "Blackwell decompose-K partial config must be static"
        ) from error
    m = int(mat1.get_size()[0])
    n = int(mat2.get_size()[1])
    m_tiles = math.ceil(m / partial_config.block_m)
    if partial_config.two_ctas:
        m_tiles = math.ceil(m_tiles / 2) * 2
    m_pad = m_tiles * partial_config.block_m
    layout = ir.FixedLayout(
        mat1.get_device(),
        torch.float32,
        [int(k_split) * m_pad, n],
        [n, 1],
    )
    choices: list[Any] = []
    append_blackwell_decompose_k_partial_choice(
        choices,
        (mat1, mat2),
        layout,
        k_split=int(k_split),
        config=partial_config,
    )
    return choices[0].output_node()
