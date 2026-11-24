# mypy: allow-untyped-defs
"""Decompose-K subgraph and Blackwell partial-BMM lowering."""

import functools
import hashlib
import math
from typing import Any

import torch
from torch._inductor import inductor_prims, ir
from torch._inductor.autows_utils import has_two_ctas, meta_ws_enabled
from torch._inductor.lowering import register_lowering
from torch._inductor.utils import can_use_tma, get_num_sms, sympy_product
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._ordered_set import OrderedSet

from .. import config as inductor_config
from ..codegen.subgraph import SubgraphChoiceCaller, SubgraphTemplate
from ..ir import Buffer, Layout
from ..virtualized import V
from .bmm import (
    blackwell_ws_persistent_tma_bmm_template,
    BlackwellBMMConfig,
    is_blackwell_bmm_2cta_compatible,
)


# TODO(@jananisriram): Refine the max-autotune search space.
BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS = (
    BlackwellBMMConfig(128, 128, 128, 3, 8, 2, 1, True, False),
    BlackwellBMMConfig(128, 128, 64, 4, 8, 1, 1, True, True),
    BlackwellBMMConfig(128, 256, 64, 6, 8, 2, 1, True, True),
    BlackwellBMMConfig(128, 128, 128, 3, 4, 2, 1, True, False),
    BlackwellBMMConfig(128, 128, 64, 4, 4, 1, 1, True, True),
    BlackwellBMMConfig(128, 256, 64, 6, 4, 2, 1, True, True),
)


def effective_two_ctas(config: BlackwellBMMConfig) -> bool:
    """Whether a partial config runs as 2CTA: needs Meta autoWS and tl.dot two_ctas."""
    return config.two_ctas and meta_ws_enabled() and has_two_ctas()


def _decompose_k_choice_name(
    input_nodes: list[Buffer],
    layout: Layout,
    k_split: int,
    bmm_backend: str,
    bmm_config_index: int,
) -> str:
    tensor_geometry = (
        tuple(
            (
                tuple(node.get_size()),
                tuple(node.get_stride()),
                node.get_dtype(),
                node.get_device(),
            )
            for node in input_nodes
        ),
        (
            tuple(layout.size),
            tuple(layout.stride),
            layout.dtype,
            layout.device,
        ),
    )
    geometry_hash = hashlib.sha256(repr(tensor_geometry).encode()).hexdigest()[:12]
    name = f"decompose_k_mm_{k_split}_split_{bmm_backend}_{geometry_hash}"
    if bmm_backend == "triton":
        name = f"{name}_config_{bmm_config_index}"
    return name


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

        name = _decompose_k_choice_name(
            input_nodes,
            layout,
            k_split,
            bmm_backend,
            bmm_config_index,
        )
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
            choice = super().generate(
                name=name,
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=fn,
                description=description,
            )
        from ..autotune_process import use_pipelined_autotuning

        # Pipelined autotuning returns no timings to apply a credit to.
        if (
            bmm_backend == "triton"
            and inductor_config.prologue_fusion
            and inductor_config.triton.decompose_k_prologue_fusion_credit
            and not use_pipelined_autotuning()
        ):
            credit_s = _partial_bmm_prologue_credit_s(
                input_nodes[0], input_nodes[1], k_split, bmm_config_index
            )
            # Inlining makes the partial BMM an ordinary template, so producers
            # of its operands can fuse into its descriptor loads.
            choice.inline_after_autotune = credit_s > 0
            choice.fusion_credit_ms = credit_s * 1e3
        return choice


decompose_k_subgraph_template = DecomposeKSubgraphTemplate()


def _pointwise_producer(node: Any) -> ir.ComputedBuffer | None:
    """The pointwise ComputedBuffer computing ``node`` in this graph, if any."""
    name = node.get_name()
    buf = V.graph.try_get_buffer(name)
    if (
        name in V.graph.graph_inputs
        or not isinstance(buf, ir.ComputedBuffer)
        or not isinstance(buf.data, ir.Pointwise)
    ):
        return None
    return buf


def _operand_has_single_use(index: int) -> bool:
    """Whether the mm being lowered reads operand ``index`` through single-use
    views of a single-use pointwise producer. Prologue fusion only fuses a
    producer whose one reader is the template, so a producer read elsewhere
    stays unfused and must not earn the decompose-K choice a credit."""
    aten = torch.ops.aten
    views = OrderedSet(
        [
            aten.view.default,
            aten.reshape.default,
            aten._unsafe_view.default,
            aten.permute.default,
            aten.t.default,
            aten.transpose.int,
            aten.slice.Tensor,
            aten.select.int,
            aten.squeeze.dim,
            aten.unsqueeze.default,
            aten.expand.default,
            aten.alias.default,
        ]
    )
    node = V.graph.current_node
    if node is None or node.target is not aten.mm.default:
        return False
    arg = node.args[index]
    while isinstance(arg, torch.fx.Node) and arg.op == "call_function":
        if len(arg.users) != 1:
            return False
        if arg.target not in views:
            return isinstance(arg.target, torch._ops.OpOverload) and (
                torch.Tag.pointwise in arg.target.tags
            )
        arg = arg.args[0]
    return False


def partial_bmm_prologue_reuse(
    is_a: bool, cfg: BlackwellBMMConfig, m: int, n: int, k: int, k_split: int
) -> tuple[int, int, int]:
    """(reuse, tiles, contraction length) of a partial-BMM operand, as the
    template records them for the prologue cost model (prologue_reuse_counts in
    triton_blackwell_ws_persistent_tma_bmm.py.jinja's VIRTUAL_BATCH loads)."""
    m_tiles = math.ceil(m / cfg.block_m)
    n_tiles = math.ceil(n / cfg.block_n)
    k_part = math.ceil(math.ceil(k / k_split) / cfg.block_k) * cfg.block_k
    if is_a:
        return n_tiles, k_split * m_tiles, k_part
    return m_tiles, k_split * n_tiles, k_part


def _partial_bmm_prologue_credit_s(
    mat1: Any, mat2: Any, k_split: int, config_index: int
) -> float:
    """Seconds saved by fusing the pointwise producers of ``mat1``/``mat2`` into
    the Blackwell partial BMM's descriptor loads, per the scheduler's prologue
    eligibility and cost model; 0 when neither operand has a producer that
    would fuse."""
    from ..scheduler import (
        descriptor_prologue_source_tile_bytes,
        descriptor_prologue_times,
    )
    from ..select_algorithm import PrologueReuse
    from ..utils import get_gpu_shared_memory

    cfg = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[config_index]
    two_ctas = effective_two_ctas(cfg)
    hint = V.graph.sizevars.optimization_hint
    m, k = (hint(s, fallback=1) for s in mat1.get_size())
    n = hint(mat2.get_size()[1], fallback=1)
    out_numel = k_split * math.ceil(m / cfg.block_m) * cfg.block_m * n
    # Bytes per stage of the operand rings; a fused operand's ring holds its
    # source tiles instead. The warp-specialization memory planner sizes each
    # ring separately and shrinks them to fit, so only one stage of every ring
    # must fit (a 2-stage bound rejects kernels that compile and run).
    a_tile, b_tile = cfg.block_m * cfg.block_k, cfg.block_k * cfg.block_n
    stage_bytes = (
        a_tile * mat1.get_dtype().itemsize + b_tile * mat2.get_dtype().itemsize
    )
    credit = 0.0
    for is_a, node in ((True, mat1), (False, mat2)):
        if not is_a and two_ctas:
            # The 2-CTA template loads B without prologue descriptors.
            continue
        producer = _pointwise_producer(node)
        if producer is None or not _operand_has_single_use(0 if is_a else 1):
            continue
        # The producer's read indices are in its own loop order. Only a 2-D
        # producer read as itself or transposed keeps them affine in the
        # operand's (row, col); behind a reshape the template gathers instead.
        sizes = [hint(s, fallback=-1) for s in node.get_size()]
        producer_sizes = [hint(s, fallback=-2) for s in producer.get_size()]
        if sizes not in (producer_sizes, producer_sizes[::-1]):
            continue
        tile_elems = a_tile if is_a else b_tile
        numel = hint(sympy_product(node.get_size()), fallback=0)
        source_tile_bytes = descriptor_prologue_source_tile_bytes(
            producer.get_read_writes().reads,
            OrderedSet([producer.get_name()]),
            tile_elems,
            numel,
        )
        if source_tile_bytes is None:
            continue
        fused_stage_bytes = (
            stage_bytes + source_tile_bytes - tile_elems * node.get_dtype().itemsize
        )
        if fused_stage_bytes > get_gpu_shared_memory():
            continue
        stage_bytes = fused_stage_bytes
        read_bytes = sum(
            hint(sympy_product(V.graph.get_buffer(r).get_size()), fallback=0)
            * V.graph.get_dtype(r).itemsize
            for r in producer.get_read_names()
        )
        reuse, tiles, k_part = partial_bmm_prologue_reuse(is_a, cfg, m, n, k, k_split)
        record = PrologueReuse(
            reuse=reuse,
            tiles=tiles,
            tile_elems=tile_elems,
            num_stages=cfg.num_stages,
            k=k_part,
        )
        unfused_s, fused_s = descriptor_prologue_times(
            node.get_device_or_error(),
            node.get_dtype(),
            record,
            read_bytes,
            numel * node.get_dtype().itemsize,
            numel,
            out_numel,
        )
        credit += max(0.0, unfused_s - fused_s)
    return credit


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
    # Imported here: the heuristics package imports this module's templates.
    from ..heuristics.template.triton import mm_allow_tf32

    m, k = map(int, mat1.get_size())
    k_b, n = map(int, mat2.get_size())
    if k != k_b:
        raise AssertionError(f"incompatible K dimensions: {k} and {k_b}")
    if (k_split - 1) * k_part >= k:
        raise NotImplementedError("aligned split leaves an empty final partition")

    use_meta_ws = meta_ws_enabled()
    two_ctas = effective_two_ctas(config)
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
        # Even under 2CTA: the kernel strides by NUM_SMS and pairs CTAs.
        "NUM_SMS": min(
            get_num_sms(two_ctas=two_ctas),
            k_split * m_tiles * math.ceil(n / config.block_n),
        ),
        "A_ROW_MAJOR": mat1.get_stride()[1] == 1,
        "B_ROW_MAJOR": mat2.get_stride()[1] == 1,
        # Gate TF32 on the full GEMM this plan replaces, as eager mm would.
        "ALLOW_TF32": mm_allow_tf32(m, n, k, mat1.get_device().type),
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

    # The kernel bakes operand strides and layouts into constexprs, so fix the
    # operand layouts before reading them.
    mat1, mat2 = (ir.ExternKernel.realize_input(node) for node in (mat1, mat2))
    for node in (mat1, mat2):
        if isinstance(node.get_layout(), ir.FlexibleLayout):
            node.freeze_layout()

    m = int(mat1.get_size()[0])
    k = int(mat1.get_size()[1])
    n = int(mat2.get_size()[1])

    m_tiles = math.ceil(m / partial_config.block_m)
    if effective_two_ctas(partial_config):
        m_tiles = math.ceil(m_tiles / 2) * 2

    expected_m_pad = m_tiles * partial_config.block_m
    expected_k_part = (
        math.ceil(math.ceil(k / int(k_split)) / partial_config.block_k)
        * partial_config.block_k
    )
    if int(m_pad) != expected_m_pad or int(k_part) != expected_k_part:
        raise AssertionError("decompose-K plan geometry does not match its config")

    layout = ir.FixedLayout(
        mat1.get_device_or_error(),
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
    if effective_two_ctas(config):
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
