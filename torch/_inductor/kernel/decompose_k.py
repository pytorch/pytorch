# mypy: allow-untyped-defs
"""Decompose-K subgraph and Blackwell partial-BMM lowering."""

import functools
import math
from typing import Any

import torch
from torch._inductor import inductor_prims, ir
from torch._inductor.autows_utils import meta_ws_enabled
from torch._inductor.lowering import register_lowering
from torch._inductor.utils import can_use_tma, get_num_sms
from torch.fx.experimental.proxy_tensor import make_fx

from ..codegen.subgraph import SubgraphChoiceCaller, SubgraphTemplate
from ..ir import Buffer, Layout
from ..virtualized import V
from .bmm import (
    blackwell_ws_persistent_tma_bmm_template,
    BlackwellBMMConfig,
    is_blackwell_bmm_2cta_compatible,
)


USE_META_WS = meta_ws_enabled()


# TODO(@jananisriram): Refine the max-autotune search space.
BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS = (
    BlackwellBMMConfig(128, 128, 128, 3, 8, 2, 1, True, False),
    BlackwellBMMConfig(128, 128, 64, 4, 8, 1, 1, True, True),
    BlackwellBMMConfig(128, 256, 64, 6, 8, 2, 1, True, True),
    BlackwellBMMConfig(128, 128, 128, 3, 4, 2, 1, True, False),
    BlackwellBMMConfig(128, 128, 64, 4, 4, 1, 1, True, True),
    BlackwellBMMConfig(128, 256, 64, 6, 4, 2, 1, True, True),
)


def get_cat2_fp32_prologue_sources(input_node) -> tuple[str, str] | None:
    """Return sources from an explicitly tagged FP32 cat-to-BF16 lowering."""
    node = input_node
    while isinstance(node, (ir.TensorBox, ir.StorageBox)):
        node = node.data
    if isinstance(node, ir.ComputedBuffer):
        node = node.data
    if not isinstance(node, ir.Pointwise):
        return None

    size = tuple(V.graph.sizevars.simplify(s) for s in node.get_size())
    if (
        len(size) != 2
        or node.get_dtype() != torch.bfloat16
        or not V.graph.sizevars.statically_known_equals(size[1], 128)
    ):
        return None

    source_names = node.annotations.get(ir.CAT2_FP32_TO_BF16_SOURCES)
    if not (
        isinstance(source_names, tuple)
        and len(source_names) == 2
        and all(isinstance(name, str) for name in source_names)
        and tuple(node.get_read_names()) == source_names
    ):
        return None
    for source_name in source_names:
        source = V.graph.try_get_buffer(source_name)
        if source is None:
            return None
        source_size = tuple(V.graph.sizevars.simplify(s) for s in source.get_size())
        source_stride = tuple(V.graph.sizevars.simplify(s) for s in source.get_stride())
        if (
            source.get_dtype() != torch.float32
            or len(source_size) != 2
            or not V.graph.sizevars.statically_known_equals(source_size[0], size[0])
            or not V.graph.sizevars.statically_known_equals(source_size[1], 64)
            or not V.graph.sizevars.statically_known_equals(source_stride[1], 1)
            or not V.graph.sizevars.statically_known_equals(
                source_stride[0], source_size[1]
            )
            or not V.graph.sizevars.statically_known_equals(
                source.get_layout().offset, 0
            )
        ):
            return None
    return source_names


def decomposeK(a, b, k_splits, bmm_backend="aten", bmm_config_index=-1):
    """Compute an MM as independent K partitions followed by a reduction."""
    m = a.shape[0]
    k = a.shape[1]

    if bmm_backend == "aten":
        n = b.shape[1]
        k_parts = k // k_splits
        a_reshaped = torch.permute(a.reshape(m, k_splits, k_parts), (1, 0, 2))
        b_reshaped = b.reshape(k_splits, k_parts, n)
        result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    elif bmm_backend == "triton":
        result = blackwell_decompose_k_partial(
            a,
            b,
            k_splits,
            bmm_config_index,
        )
    else:
        raise AssertionError(f"unsupported decompose-K BMM backend: {bmm_backend}")

    reduced_buf = torch.sum(result, 0)
    return reduced_buf.to(a.dtype)


class DecomposeKSubgraphTemplate(SubgraphTemplate):
    def __init__(self):
        super().__init__(name="decompose_k")

    def generate(  # type: ignore[override]
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        k_split: int,
        bmm_backend: str = "aten",
        bmm_config_index: int = -1,
    ) -> SubgraphChoiceCaller:
        from torch._dispatch.python import enable_python_dispatcher

        from ..decomposition import select_decomp_table

        name = f"decompose_k_mm_{k_split}_split_{bmm_backend}"
        if bmm_backend == "triton":
            name = f"{name}_config_{bmm_config_index}"
        description = f"{k_split=}, {bmm_backend=}, {bmm_config_index=}"

        with enable_python_dispatcher():
            decompositions = select_decomp_table()
            fn = make_fx(
                functools.partial(
                    decomposeK,
                    k_splits=k_split,
                    bmm_backend=bmm_backend,
                    bmm_config_index=bmm_config_index,
                ),
                decompositions,
            )
            return super().generate(
                name=name,
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=fn,
                description=description,
            )


decompose_k_subgraph_template = DecomposeKSubgraphTemplate()


def _blackwell_decompose_k_partial_kwargs(
    mat1,
    mat2,
    *,
    k_split: int,
    m_pad: int,
    k_part: int,
    config: BlackwellBMMConfig,
) -> dict[str, Any]:
    """Build launch kwargs for the partial-BMM template, not the outer graph."""
    m, k = map(int, mat1.get_size())
    k_b, n = map(int, mat2.get_size())
    if k != k_b:
        raise AssertionError(f"incompatible K dimensions: {k} and {k_b}")
    if (k_split - 1) * k_part >= k:
        raise NotImplementedError("aligned split leaves an empty final partition")

    use_meta_ws = meta_ws_enabled()
    two_ctas = use_meta_ws and config.two_ctas
    if two_ctas and not is_blackwell_bmm_2cta_compatible(
        output_batch_rows=m_pad,
        block_m=config.block_m,
        flatten_output=True,
        tma_store=True,
    ):
        raise NotImplementedError(
            "2CTA Blackwell decompose-K requires complete paired M tiles"
        )
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
        "A_BROADCAST_BATCH": False,
        "B_BROADCAST_BATCH": False,
        "VIRTUAL_BATCH": True,
        "OUTPUT_BATCH_ROWS": m_pad,
        "NUM_SMS": min(
            get_num_sms(),
            k_split * m_tiles * math.ceil(n / config.block_n),
        ),
        "A_ROW_MAJOR": mat1.get_stride()[1] == 1,
        "B_ROW_MAJOR": mat2.get_stride()[1] == 1,
        "ALLOW_TF32": False,
        "USE_META_WS": use_meta_ws,
        "WARP_SPECIALIZE": True,
        "FLATTEN": not use_meta_ws,
        "DATA_PARTITION_FACTOR": config.data_partition_factor,
        "SEPARATE_EPILOGUE_STORE": config.separate_epilogue_store,
        "EPILOGUE_SUBTILE": config.epilogue_subtile,
        "TWO_CTAS": two_ctas,
        "FLATTEN_OUTPUT": True,
        "tma_store": True,
        "transpose_discontiguous_tensor_descriptors_override": True,
    }
    if two_ctas:
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
    if meta_ws_enabled() and partial_config.two_ctas:
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
    if not can_use_tma(mat1, mat2, output_layout=layout):
        raise NotImplementedError(
            "Blackwell decompose-K requires TMA-compatible inputs and output"
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
    if USE_META_WS and config.two_ctas:
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
