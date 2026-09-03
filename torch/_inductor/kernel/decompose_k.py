# mypy: allow-untyped-defs
"""Blackwell partial-BMM lowering used by the decompose-K subgraph."""

import math
from typing import Any

import torch
from torch._inductor import inductor_prims, ir
from torch._inductor.lowering import register_lowering
from torch._inductor.utils import get_num_sms

from .bmm import blackwell_ws_persistent_tma_bmm_template, BlackwellBMMConfig


# TODO(@jananisriram): Refine the max-autotune search space.
BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS = (
    BlackwellBMMConfig(128, 128, 128, 3, 8, 2, 1, True, False),
    BlackwellBMMConfig(128, 128, 64, 4, 8, 1, 1, True, True),
    BlackwellBMMConfig(128, 256, 64, 6, 8, 2, 1, True, True),
    BlackwellBMMConfig(128, 128, 128, 3, 4, 2, 1, True, False),
    BlackwellBMMConfig(128, 128, 64, 4, 4, 1, 1, True, True),
    BlackwellBMMConfig(128, 256, 64, 6, 4, 2, 1, True, True),
)


def _blackwell_decompose_k_partial_kwargs(
    mat1,
    mat2,
    *,
    k_split: int,
    m_pad: int,
    k_part: int,
    config: BlackwellBMMConfig,
) -> dict[str, Any]:
    m, k = map(int, mat1.get_size())
    k_b, n = map(int, mat2.get_size())
    if k != k_b:
        raise AssertionError(f"incompatible K dimensions: {k} and {k_b}")
    if (k_split - 1) * k_part >= k:
        raise NotImplementedError("aligned split leaves an empty final partition")

    m_tiles = m_pad // config.block_m
    kwargs = {
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
        "NUM_SMS": min(
            get_num_sms(),
            m_tiles * math.ceil(n / config.block_n),
        ),
        "A_ROW_MAJOR": mat1.get_stride()[1] == 1,
        "B_ROW_MAJOR": mat2.get_stride()[1] == 1,
        "ALLOW_TF32": False,
        "USE_META_WS": True,
        "WARP_SPECIALIZE": True,
        "FLATTEN": False,
        "DATA_PARTITION_FACTOR": config.data_partition_factor,
        "SEPARATE_EPILOGUE_STORE": config.separate_epilogue_store,
        "EPILOGUE_SUBTILE": config.epilogue_subtile,
        "TWO_CTAS": config.two_ctas,
        "FLATTEN_OUTPUT": True,
        "tma_store": True,
        "transpose_discontiguous_tensor_descriptors_override": True,
    }
    if config.two_ctas:
        kwargs["ctas_per_cga"] = (2, 1, 1)
    return kwargs


@register_lowering(
    inductor_prims.blackwell_decompose_k_partial,
    type_promotion_kind=None,
)
def lower_blackwell_decompose_k_partial(
    mat1,
    mat2,
    k_split: int,
    config_index: int,
    m_pad: int,
    k_part: int,
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
    k = int(mat1.get_size()[1])
    n = int(mat2.get_size()[1])

    m_tiles = math.ceil(m / partial_config.block_m)
    if partial_config.two_ctas:
        m_tiles = math.ceil(m_tiles / 2) * 2

    expected_m_pad = m_tiles * partial_config.block_m
    expected_k_part = (
        math.ceil(math.ceil(k / int(k_split)) / partial_config.block_k)
        * partial_config.block_k
    )
    if int(m_pad) != expected_m_pad or int(k_part) != expected_k_part:
        raise AssertionError("decompose-K plan geometry does not match its config")

    layout = ir.FixedLayout(
        mat1.get_device(),
        torch.float32,
        [int(k_split) * int(m_pad), n],
        [n, 1],
    )

    template_kwargs = _blackwell_decompose_k_partial_kwargs(
        mat1,
        mat2,
        k_split=int(k_split),
        m_pad=int(m_pad),
        k_part=int(k_part),
        config=partial_config,
    )
    choice = blackwell_ws_persistent_tma_bmm_template.generate(
        input_nodes=(mat1, mat2),
        layout=layout,
        num_stages=partial_config.num_stages,
        num_warps=partial_config.num_warps,
        generate_with_caching=True,
        **template_kwargs,
    )
    if choice is None:
        raise NotImplementedError("Blackwell decompose-K partial choice is unavailable")
    return choice.output_node()


def blackwell_decompose_k_partial(a, b, k_split, config_index):
    """Produce aligned FP32 partials from the original rank-2 operands."""
    config = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[config_index]
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]

    m_tiles = (m + config.block_m - 1) // config.block_m
    if config.two_ctas:
        m_tiles = (m_tiles + 1) // 2 * 2

    m_pad = m_tiles * config.block_m
    k_part = (
        ((k + k_split - 1) // k_split + config.block_k - 1)
        // config.block_k
        * config.block_k
    )
    partial_flat = inductor_prims.blackwell_decompose_k_partial(
        a,
        b,
        k_split,
        config_index,
        m_pad,
        k_part,
    )

    return partial_flat.view(k_split, m_pad, n)[:, :m]
