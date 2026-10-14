# mypy: allow-untyped-defs
"""Triton Implementation of the flex_attention Kernel"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import sympy

import torch
from torch._inductor.virtualized import V
from torch._logging import warning_once
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from torch.nn.attention.flex_attention import _Backend, _LARGE_SPARSE_BLOCK_SIZE
from torch.utils._sympy.functions import FloorDiv

from ...ir import ComputedBuffer, ExternKernel, FixedLayout, TensorBox
from ...lowering import empty, empty_strided, lowerings, register_lowering, to_dtype
from ...runtime.runtime_utils import is_power_of_2
from ...select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
    TritonTemplate,
)
from ...utils import can_use_tma, use_flex_tdm_descriptor
from .common import (
    _flex_kernel_options_example,
    _flex_kernel_tuning_options,
    build_subgraph_buffer,
    can_skip_boundary_checks,
    create_indices_fake,
    create_num_blocks_fake_generator,
    create_placeholder,
    freeze_irnodes,
    get_fwd_subgraph_outputs,
    infer_dense_strides,
    is_tensor_ir_node,
    load_flex_template,
    maybe_realize,
    realize_captures_for_cutedsl,
    set_head_dim_values,
    SubgraphResults,
)
from .flex_cpu import lower_cpu
from .flex_decoding import _use_flex_decoding, create_flex_decoding_kernel
from .flex_flash_attention import (
    _use_flex_flash_attention,
    _use_flex_flash_attention_backward,
    create_flex_flash_attention_backward_kernel,
    create_flex_flash_attention_kernel,
    has_unsupported_cpu_scalar_tensor_captures,
    is_trivial_mask_graph,
    is_trivial_score_graph,
)


if TYPE_CHECKING:
    from ...heuristics.template.triton import FlexBwDConfig, FlexConfig


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims
Expr = sympy.Expr


@dataclass(frozen=True)
class _RocmBackwardRequests:
    fp32_balanced64_coverage_requested: Any = False
    fp32_balanced64_coverage_other_requests: Any = False
    fp16_coverage_requested: Any = False
    fp16_coverage_other_requests: Any = False
    dv_view_carry_split64_requested: Any = False
    dv_view_carry_split64_other_requests: Any = False
    reverse_dq_rows_requested: Any = False
    reverse_dq_rows_other_requests: Any = False
    early_probability_requested: Any = False
    d192_panel_phase_requested: Any = False
    d192_early_dq_dp_requested: Any = False
    d96_panel_phase_requested: Any = False
    d96_early_dq_dp_requested: Any = False
    d96_dq_unroll2_requested: Any = False
    d96_dq_unroll2_other_requests: Any = False


def _strip_rocm_query_tile_outer_options(options):
    for prefix in ("", "fwd_", "bwd_"):
        options.pop(prefix + "ROCM_D128_GQA_QUERY_TILE_OUTER", None)


def _normalize_rocm_backward_options(kernel_options):
    """Remove user aliases and retain backward requests before prefix normalization."""
    _strip_rocm_query_tile_outer_options(kernel_options)
    fp32_balanced64_coverage_requested = kernel_options.pop(
        "bwd_ROCM_D128_FP32_BALANCED64_COVERAGE", False
    )
    fp32_balanced64_coverage_other_requests = any(
        kernel_options.get(name, False) is not False
        for name in (
            "ROCM_D128_FP32_BALANCED64_COVERAGE",
            "fwd_ROCM_D128_FP32_BALANCED64_COVERAGE",
        )
    ) or any(
        kernel_options.get("bwd_" + name, kernel_options.get(name, False)) is not False
        for name in (
            "USE_TMA",
            "PRESCALE_QK",
            "ROCM_XCD_SNAKE",
            "ROCM_D256_PANEL_DOTS",
            "ROCM_D192_PANEL_DOTS",
            "ROCM_D96_PANEL_DOTS",
            "ROCM_D192_DQ_EARLY_DP",
            "ROCM_D96_DQ_EARLY_DP",
            "ROCM_D96_DQ_UNROLL2",
            "ROCM_EARLY_PROBABILITY_COMPACT",
            "ROCM_GQA_SPLIT_DKDV",
            "ROCM_D64_PAIRED_DQ_ROWS",
            "ROCM_D64_FOUR_DQ_ROWS",
            "ROCM_PACKED_STATISTICS",
            "ROCM_DQ_DELTA_PRODUCER",
            "ROCM_FUSED_DELTA",
            "ROCM_D128_FP16_COVERAGE",
            "ROCM_D128_FP32_COVERAGE",
        )
    )
    for fp32_balanced64_alias in (
        "ROCM_D128_FP32_BALANCED64_COVERAGE",
        "fwd_ROCM_D128_FP32_BALANCED64_COVERAGE",
    ):
        kernel_options.pop(fp32_balanced64_alias, None)
    fp16_coverage_requested = kernel_options.pop("bwd_ROCM_D128_FP16_COVERAGE", False)
    fp16_coverage_other_requests = any(
        kernel_options.get(name, False) is not False
        for name in ("ROCM_D128_FP16_COVERAGE", "fwd_ROCM_D128_FP16_COVERAGE")
    ) or any(
        kernel_options.get("bwd_" + name, kernel_options.get(name, False)) is not False
        for name in (
            "USE_TMA",
            "PRESCALE_QK",
            "ROCM_XCD_SNAKE",
            "ROCM_D256_PANEL_DOTS",
            "ROCM_D192_PANEL_DOTS",
            "ROCM_D96_PANEL_DOTS",
            "ROCM_D192_DQ_EARLY_DP",
            "ROCM_D96_DQ_EARLY_DP",
            "ROCM_D96_DQ_UNROLL2",
            "ROCM_EARLY_PROBABILITY_COMPACT",
            "ROCM_GQA_SPLIT_DKDV",
            "ROCM_D64_PAIRED_DQ_ROWS",
            "ROCM_D64_FOUR_DQ_ROWS",
            "ROCM_PACKED_STATISTICS",
            "ROCM_DQ_DELTA_PRODUCER",
            "ROCM_FUSED_DELTA",
        )
    )
    fp16_coverage_other_requests = (
        fp16_coverage_other_requests or fp32_balanced64_coverage_requested is not False
    )
    for fp16_alias in ("ROCM_D128_FP16_COVERAGE", "fwd_ROCM_D128_FP16_COVERAGE"):
        kernel_options.pop(fp16_alias, None)
    dv_view_carry_split64_requested = kernel_options.pop(
        "bwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64", False
    )
    dv_view_carry_split64_other_requests = any(
        kernel_options.get(option, False) is not False
        for option in (
            "ROCM_D128_DV_VIEW_CARRY_SPLIT64",
            "fwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64",
        )
    )
    kernel_options["ROCM_D128_DV_VIEW_CARRY_SPLIT64"] = False
    reverse_dq_rows_requested = kernel_options.pop(
        "bwd_ROCM_D128_REVERSE_DQ_ROWS", False
    )
    reverse_dq_rows_other_requests = any(
        kernel_options.get(prefix + option, False)
        for prefix in ("", "fwd_", "bwd_")
        for option in (
            "ROCM_D128_REVERSE_DQ_ROWS",
            "ROCM_D256_PANEL_DOTS",
            "ROCM_D192_PANEL_DOTS",
            "ROCM_D96_PANEL_DOTS",
            "ROCM_D192_DQ_EARLY_DP",
            "ROCM_D96_DQ_EARLY_DP",
            "ROCM_EARLY_PROBABILITY_COMPACT",
            "ROCM_GQA_SPLIT_DKDV",
            "ROCM_D64_PAIRED_DQ_ROWS",
            "ROCM_D64_FOUR_DQ_ROWS",
            "ROCM_PACKED_STATISTICS",
            "ROCM_FUSED_DELTA",
            "ROCM_DQ_DELTA_PRODUCER",
        )
    )
    kernel_options["ROCM_D128_REVERSE_DQ_ROWS"] = False
    kernel_options["ROCM_BWD_PHASE"] = 0
    kernel_options.pop("bwd_ROCM_BWD_PHASE", None)
    kernel_options["ROCM_GQA_SPLITS"] = 1
    kernel_options.pop("bwd_ROCM_GQA_SPLITS", None)
    # Only the backward-scoped option can request compaction.
    early_probability_requested = kernel_options.pop(
        "bwd_ROCM_EARLY_PROBABILITY_COMPACT", False
    )
    kernel_options["ROCM_EARLY_PROBABILITY_COMPACT"] = False
    d192_panel_phase_requested = kernel_options.pop("bwd_ROCM_D192_PANEL_DOTS", False)
    kernel_options["ROCM_D192_PANEL_DOTS"] = False
    d192_early_dq_dp_requested = kernel_options.pop("bwd_ROCM_D192_DQ_EARLY_DP", False)
    kernel_options["ROCM_D192_DQ_EARLY_DP"] = False
    d96_panel_phase_requested = kernel_options.pop("bwd_ROCM_D96_PANEL_DOTS", False)
    kernel_options["ROCM_D96_PANEL_DOTS"] = False
    d96_early_dq_dp_requested = kernel_options.pop("bwd_ROCM_D96_DQ_EARLY_DP", False)
    kernel_options["ROCM_D96_DQ_EARLY_DP"] = False
    d96_dq_unroll2_requested = kernel_options.pop("bwd_ROCM_D96_DQ_UNROLL2", False)
    d96_dq_unroll2_other_requests = any(
        kernel_options.get(option, False) is not False
        for option in ("ROCM_D96_DQ_UNROLL2", "fwd_ROCM_D96_DQ_UNROLL2")
    )
    kernel_options["ROCM_D96_DQ_UNROLL2"] = False
    return _RocmBackwardRequests(
        fp32_balanced64_coverage_requested=fp32_balanced64_coverage_requested,
        fp32_balanced64_coverage_other_requests=fp32_balanced64_coverage_other_requests,
        fp16_coverage_requested=fp16_coverage_requested,
        fp16_coverage_other_requests=fp16_coverage_other_requests,
        dv_view_carry_split64_requested=dv_view_carry_split64_requested,
        dv_view_carry_split64_other_requests=dv_view_carry_split64_other_requests,
        reverse_dq_rows_requested=reverse_dq_rows_requested,
        reverse_dq_rows_other_requests=reverse_dq_rows_other_requests,
        early_probability_requested=early_probability_requested,
        d192_panel_phase_requested=d192_panel_phase_requested,
        d192_early_dq_dp_requested=d192_early_dq_dp_requested,
        d96_panel_phase_requested=d96_panel_phase_requested,
        d96_early_dq_dp_requested=d96_early_dq_dp_requested,
        d96_dq_unroll2_requested=d96_dq_unroll2_requested,
        d96_dq_unroll2_other_requests=d96_dq_unroll2_other_requests,
    )


def _get_rocm_backward_config_options(options, *, is_gfx950):
    """Return a fresh config with backward prefixes and disabled phase controls."""
    cur_kernel_options = options.copy()
    cur_kernel_options["ROCM_BWD_PHASE"] = 0
    cur_kernel_options["ROCM_GQA_SPLITS"] = 1
    cur_kernel_options["ROCM_EARLY_PROBABILITY_COMPACT"] = False
    cur_kernel_options["ROCM_D192_PANEL_DOTS"] = False
    cur_kernel_options["ROCM_D192_DQ_EARLY_DP"] = False
    cur_kernel_options["ROCM_D96_PANEL_DOTS"] = False
    cur_kernel_options["ROCM_D96_DQ_EARLY_DP"] = False
    cur_kernel_options["ROCM_D96_DQ_UNROLL2"] = False
    cur_kernel_options["ROCM_D128_REVERSE_DQ_ROWS"] = False
    cur_kernel_options["ROCM_D128_DV_VIEW_CARRY_SPLIT64"] = False
    # Remove prefix for backward kernels options and delete forward kernel options.
    for k in list(cur_kernel_options.keys()):
        if k.startswith("bwd_"):
            v = cur_kernel_options.pop(k)
            cur_kernel_options[k[4:]] = v
        if k.startswith("fwd_"):
            cur_kernel_options.pop(k)
    for option in ("ROCM_EARLY_DP", "XCD_REMAP"):
        enabled = cur_kernel_options.get(option, False)
        cur_kernel_options[option] = is_gfx950 and enabled
    return cur_kernel_options


def _select_rocm_d128_query_tile_outer_options(
    input_nodes, dkdv_options, *, single_config, fp16_pilot, fp32_balanced64_pilot
):
    """Select query-first traversal only after the parent coverage guard succeeds."""
    if fp16_pilot is True and _use_rocm_d128_gqa_query_tile_outer(
        input_nodes, dkdv_options, single_config=single_config
    ):
        dkdv_options["ROCM_D128_GQA_QUERY_TILE_OUTER"] = True
    if fp32_balanced64_pilot is True and _use_rocm_d128_gqa_query_tile_outer(
        input_nodes, dkdv_options, single_config=single_config
    ):
        dkdv_options["ROCM_D128_GQA_QUERY_TILE_OUTER"] = True


def _get_rocm_backward_phase_options(
    options,
    dkdv_input_nodes,
    *,
    requests,
    single_config,
    is_gfx950,
    query,
    key,
    value,
    grad_out,
    use_separate_dq,
    use_d192_panel_phase,
    use_d96_panel_phase,
    fp16_pilot,
    fp32_balanced64_pilot,
    score_is_trivial,
    has_captures,
    mask_is_causal,
    mask_is_scalar_true,
    block_mask_is_default,
    grad_logsumexp_is_none,
    is_automatic=False,
):
    """Return dK/dV and dQ options without changing the fused config."""
    cur_kernel_options = options.copy()
    dkdv_options = dict(cur_kernel_options, ROCM_BWD_PHASE=1)
    dkdv_options["ROCM_D128_DV_VIEW_CARRY_SPLIT64"] = (
        single_config
        and _use_rocm_d128_dv_view_carry_split64(
            use_separate_dq,
            is_gfx950,
            query,
            key,
            value,
            grad_out,
            cur_kernel_options,
            score_is_trivial,
            has_captures,
            requested=requests.dv_view_carry_split64_requested,
            reverse_requested=requests.reverse_dq_rows_requested,
            mask_is_causal=mask_is_causal,
            grad_logsumexp_is_none=grad_logsumexp_is_none,
            other_requests=requests.dv_view_carry_split64_other_requests
            or requests.d96_dq_unroll2_requested is not False,
            reverse_other_requests=requests.reverse_dq_rows_other_requests,
            fp16_pilot=fp16_pilot,
            fp32_balanced64_pilot=fp32_balanced64_pilot,
        )
    )
    _select_rocm_d128_query_tile_outer_options(
        dkdv_input_nodes,
        dkdv_options,
        single_config=single_config,
        fp16_pilot=fp16_pilot,
        fp32_balanced64_pilot=fp32_balanced64_pilot,
    )
    dkdv_options["ROCM_D192_PANEL_DOTS"] = use_d192_panel_phase
    dkdv_options["ROCM_D96_PANEL_DOTS"] = use_d96_panel_phase
    dkdv_options["ROCM_EARLY_PROBABILITY_COMPACT"] = (
        _use_rocm_early_probability_compact(
            is_gfx950,
            query,
            key,
            value,
            cur_kernel_options,
            score_is_trivial,
            has_captures,
            requested=requests.early_probability_requested,
            mask_is_scalar_true=mask_is_scalar_true,
            block_mask_is_default=block_mask_is_default,
        )
    )
    reverse_dq_rows = single_config and _use_rocm_d128_reverse_causal_dq_rows(
        use_separate_dq,
        is_gfx950,
        query,
        key,
        value,
        grad_out,
        cur_kernel_options,
        score_is_trivial,
        has_captures,
        requested=requests.reverse_dq_rows_requested,
        mask_is_causal=mask_is_causal,
        grad_logsumexp_is_none=grad_logsumexp_is_none,
        other_requests=requests.reverse_dq_rows_other_requests,
        fp16_pilot=fp16_pilot,
        fp32_balanced64_pilot=fp32_balanced64_pilot,
    )
    cur_kernel_options["ROCM_D96_DQ_UNROLL2"] = (
        single_config
        and _use_rocm_d96_dq_unroll2(
            use_d96_panel_phase,
            cur_kernel_options,
            requested=requests.d96_dq_unroll2_requested,
            early_dq_dp_requested=requests.d96_early_dq_dp_requested,
            other_requests=requests.d96_dq_unroll2_other_requests,
            query=query,
            key=key,
        )
    )
    dq_options = dict(
        dict(
            cur_kernel_options,
            ROCM_BWD_PHASE=2,
            ROCM_D128_REVERSE_DQ_ROWS=reverse_dq_rows,
            ROCM_D192_DQ_EARLY_DP=_use_rocm_d192_dq_early_dp(
                use_d192_panel_phase,
                cur_kernel_options,
                requested=requests.d192_early_dq_dp_requested,
            ),
        ),
        ROCM_D96_DQ_EARLY_DP=_use_rocm_d96_dq_early_dp(
            use_d96_panel_phase,
            cur_kernel_options,
            requested=requests.d96_early_dq_dp_requested,
            query=query,
            key=key,
        ),
    )
    if (
        is_automatic is True
        and dkdv_options.get("ROCM_D256_PANEL_DOTS", False)
        and V.graph.sizevars.statically_known_equals(
            dkdv_options["GQA_SHARED_HEADS"], 4
        )
    ):
        dkdv_options["BLOCK_M1"] = 32
        dkdv_options["XCD_REMAP"] = True
    if (
        is_automatic is True
        and use_d96_panel_phase
        and V.graph.sizevars.statically_known_equals(
            dkdv_options["GQA_SHARED_HEADS"], 4
        )
    ):
        dkdv_options["ROCM_D96_PANEL_DOTS"] = False
        dkdv_tile = (
            32
            if V.graph.sizevars.statically_known_equals(query.get_size()[2], 2048)
            else 64
        )
        dkdv_options["BLOCK_M1"] = dkdv_tile
        dkdv_options["BLOCK_N1"] = dkdv_tile
        dkdv_options["num_stages"] = 1
    if (
        is_automatic is True
        and use_d192_panel_phase
        and all(
            V.graph.sizevars.statically_known_equals(actual, expected)
            for actual, expected in zip(query.get_size(), (4, 10, 4096, 192))
        )
        and all(
            V.graph.sizevars.statically_known_equals(actual, expected)
            for actual, expected in zip(key.get_size(), (4, 10, 4096, 192))
        )
    ):
        dkdv_options["matrix_instr_nonkdim"] = 16
        dq_options["matrix_instr_nonkdim"] = 16
    return dkdv_options, dq_options


def _get_rocm_backward_auto_profile(
    user_kernel_options,
    kernel_options,
    input_nodes,
    out,
    *,
    backend,
    is_gfx950,
    single_config,
    score_is_trivial,
    has_captures,
    mask_is_causal,
    mask_is_scalar_true,
    block_mask_is_default,
    grad_logsumexp_is_none,
):
    """Select measured gfx950 profiles without changing user or fused options."""
    if (
        is_gfx950 is not True
        or single_config is not True
        or backend not in ("AUTO", "TRITON")
        or score_is_trivial is not True
        or has_captures is not False
        or grad_logsumexp_is_none is not True
        or len(input_nodes) != 16
    ):
        return None
    for name in user_kernel_options:
        if name.startswith("fwd_"):
            continue
        name = name.removeprefix("bwd_")
        if name.startswith("ROCM_") or name in (
            "XCD_REMAP",
            "BLOCK_M1",
            "BLOCK_N1",
            "BLOCK_M2",
            "BLOCK_N2",
            "num_warps",
            "num_stages",
            "matrix_instr_nonkdim",
            "waves_per_eu",
            "kpack",
            "num_consumer_groups",
            "waves_per_simd",
            "llvm_fn_attrs",
        ):
            return None
    query, key, value = input_nodes[:3]
    if (
        query is None
        or key is None
        or len(query.get_size()) != 4
        or len(key.get_size()) != 4
    ):
        return None
    sizevars = V.graph.sizevars
    dtype = query.get_dtype()
    # Restrict automatic selection to measured stock comparisons.
    profiles = (
        (torch.float16, 1, 8, 2, 4096, 64),
        (torch.bfloat16, 4, 64, 8, 4096, 64),
        (torch.bfloat16, 4, 21, 21, 2048, 96),
        (torch.bfloat16, 4, 21, 21, 4096, 96),
        (torch.bfloat16, 2, 8, 2, 2048, 96),
        (torch.bfloat16, 2, 8, 2, 4096, 96),
        (torch.bfloat16, 4, 64, 8, 4096, 128),
        (torch.bfloat16, 4, 10, 10, 2048, 192),
        (torch.bfloat16, 4, 10, 10, 4096, 192),
        (torch.bfloat16, 2, 8, 2, 2048, 192),
        (torch.bfloat16, 4, 32, 32, 4096, 256),
        (torch.bfloat16, 4, 32, 8, 4096, 256),
        (torch.float16, 4, 64, 8, 2048, 128),
        (torch.float16, 4, 64, 8, 4096, 128),
        (torch.float16, 2, 64, 8, 2048, 128),
        (torch.float16, 2, 64, 8, 4096, 128),
        (torch.float16, 1, 64, 8, 4096, 128),
        (torch.float32, 4, 64, 8, 2048, 128),
        (torch.float32, 4, 64, 8, 4096, 128),
        (torch.float32, 2, 64, 8, 2048, 128),
        (torch.float32, 2, 64, 8, 4096, 128),
        (torch.float32, 1, 64, 8, 2048, 128),
        (torch.float32, 1, 64, 8, 4096, 128),
    )
    profile = next(
        (
            (batch, query_heads, kv_heads, length, dimension)
            for candidate_dtype, batch, query_heads, kv_heads, length, dimension in profiles
            if dtype == candidate_dtype
            and sizevars.statically_known_equals(key.get_size()[1], kv_heads)
            and all(
                type(actual) not in (bool, float, str)
                and sizevars.statically_known_equals(actual, expected)
                for actual, expected in zip(
                    query.get_size(), (batch, query_heads, length, dimension)
                )
            )
        ),
        None,
    )
    if profile is None:
        return None
    batch, query_heads, kv_heads, length, dimension = profile
    scale = kernel_options.get("SM_SCALE")
    if (
        type(scale) not in (int, float)
        or scale != 1 / math.sqrt(dimension)
        or kernel_options.get("PRESCALE_QK") is not False
        or kernel_options.get("USE_TMA") is not False
        or type(kernel_options.get("BLOCKS_ARE_CONTIGUOUS")) is not bool
    ):
        return None
    device = query.get_device()
    if getattr(device, "type", None) != "cuda":
        return None
    query_shape = (batch, query_heads, length, dimension)
    kv_shape = (batch, kv_heads, length, dimension)
    stats_shape = (batch, query_heads, length)
    operands: list[tuple[Any, tuple[int, ...], torch.dtype]] = [
        (query, query_shape, dtype),
        (key, kv_shape, dtype),
        (value, kv_shape, dtype),
        (input_nodes[3], stats_shape, torch.float32),
        (input_nodes[4], stats_shape, torch.float32),
        (input_nodes[5], query_shape, dtype),
        (out, query_shape, dtype),
    ]
    if dimension == 128:
        if (
            mask_is_causal is not True
            or kernel_options.get("HAS_FULL_BLOCKS") is not True
            or not all(
                sizevars.statically_known_equals(kernel_options.get(name), 128)
                for name in ("SPARSE_Q_BLOCK_SIZE", "SPARSE_KV_BLOCK_SIZE")
            )
        ):
            return None
        blocks = length // 128
        operands.extend(
            (
                tensor,
                (1, 1, blocks) if index % 2 == 0 else (1, 1, blocks, blocks),
                torch.int32,
            )
            for index, tensor in enumerate(input_nodes[8:])
        )
    elif mask_is_scalar_true is not True or block_mask_is_default is not True:
        return None
    if dimension == 64 and dtype == torch.float16:
        canonical_options = {
            "PRESCALE_QK": False,
            "ROWS_GUARANTEED_SAFE": False,
            "BLOCKS_ARE_CONTIGUOUS": False,
            "WRITE_DQ": True,
            "OUTPUT_LOGSUMEXP": True,
            "OUTPUT_MAX": False,
        }
        if (
            any(
                name != "BACKEND"
                and not name.startswith("fwd_")
                and (
                    name not in canonical_options
                    or value is not canonical_options[name]
                )
                for name, value in user_kernel_options.items()
            )
            or mask_is_causal is not False
            or kernel_options.get("BLOCKS_ARE_CONTIGUOUS") is not False
            or kernel_options.get("HAS_FULL_BLOCKS") is not False
            or kernel_options.get("IS_DIVISIBLE") is not True
            or kernel_options.get("SAFE_HEAD_DIM") is not True
            or not all(
                type(kernel_options.get(name)) not in (bool, float, str)
                and sizevars.statically_known_equals(kernel_options.get(name), expected)
                for name, expected in (
                    ("GQA_SHARED_HEADS", 4),
                    ("QK_HEAD_DIM", 64),
                    ("V_HEAD_DIM", 64),
                    ("QK_HEAD_DIM_ROUNDED", 64),
                    ("V_HEAD_DIM_ROUNDED", 64),
                    ("SPARSE_Q_BLOCK_SIZE", _LARGE_SPARSE_BLOCK_SIZE),
                    ("SPARSE_KV_BLOCK_SIZE", _LARGE_SPARSE_BLOCK_SIZE),
                )
            )
        ):
            return None
        operands.extend(
            (tensor, (1, 1, 1) if index % 2 == 0 else (1, 1, 1, 1), torch.int32)
            for index, tensor in enumerate(input_nodes[8:12])
        )
        operands.extend((tensor, (0,), torch.float32) for tensor in input_nodes[12:])
    for tensor, shape, expected_dtype in operands:
        if (
            tensor is None
            or tensor.get_dtype() != expected_dtype
            or tensor.get_device() != device
            or len(tensor.get_size()) != len(shape)
            or len(tensor.get_stride()) != len(shape)
            or not all(
                type(actual) not in (bool, float, str)
                and sizevars.statically_known_equals(actual, expected)
                for actual, expected in zip(tensor.get_size(), shape)
            )
        ):
            return None
        stride = 1
        for size, actual in zip(reversed(shape), reversed(tensor.get_stride())):
            if type(actual) in (
                bool,
                float,
                str,
            ) or not sizevars.statically_known_equals(actual, stride):
                return None
            stride *= size
    options = kernel_options.copy()
    options.update(num_warps=4, num_stages=1 if dtype == torch.float32 else 2)
    options.update(matrix_instr_nonkdim=0, waves_per_eu=0, kpack=1)
    tiles = (
        (16, 64, 128, 32)
        if dimension in (192, 256)
        else (16, 64, 64, 16)
        if dtype == torch.float32
        else (32, 128, 128, 32)
    )
    options.update(zip(("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"), tiles))
    if dimension == 64 and dtype == torch.float16:
        options.update(
            ROCM_GQA_SPLIT_DKDV=True,
            ROCM_SEPARATE_DQ=False,
            ROCM_EARLY_DP=False,
            XCD_REMAP=False,
        )
    elif dimension == 192:
        if user_kernel_options.get("bwd_BLOCKS_ARE_CONTIGUOUS", True) is not True:
            return None
        options["BLOCKS_ARE_CONTIGUOUS"] = True
        options.update(
            ROCM_SEPARATE_DQ=False,
            ROCM_EARLY_DP=False,
            XCD_REMAP=False,
            bwd_ROCM_D192_PANEL_DOTS=True,
            bwd_ROCM_D192_DQ_EARLY_DP=True,
        )
    elif dimension == 96:
        if user_kernel_options.get("bwd_BLOCKS_ARE_CONTIGUOUS", True) is not True:
            return None
        # A default dense mask contains one sparse block for the entire sequence.
        options["BLOCKS_ARE_CONTIGUOUS"] = True
        options.update(
            bwd_ROCM_D96_PANEL_DOTS=True,
            bwd_ROCM_D96_DQ_EARLY_DP=True,
            bwd_ROCM_D96_DQ_UNROLL2=True,
        )
    else:
        options.update(ROCM_SEPARATE_DQ=True, ROCM_EARLY_DP=True)
        options["XCD_REMAP"] = dimension == 128 or kv_heads == query_heads
        if dimension == 256:
            options["ROCM_D256_PANEL_DOTS"] = True
            options["bwd_ROCM_EARLY_PROBABILITY_COMPACT"] = True
        elif dimension == 128:
            options.update(
                bwd_ROCM_D128_REVERSE_DQ_ROWS=True,
                bwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64=True,
            )
            if dtype == torch.float16:
                options["bwd_ROCM_D128_FP16_COVERAGE"] = True
            elif dtype == torch.float32:
                options["bwd_ROCM_D128_FP32_BALANCED64_COVERAGE"] = True
    return options


def _append_flex_attention_backward_choices(
    choices,
    configs,
    input_nodes,
    subgraphs,
    layout,
    kernel_options,
    sparse_q_block_size,
    sparse_kv_block_size,
    *,
    mutated_inputs,
    split_available,
):
    """Preserve the fused hook ABI and prefer its custom pool over split execution."""
    default_choices = tuple(choices)
    choices = V.choices.append_flex_attention_backward_choices(
        choices,
        configs,
        list(input_nodes),
        list(subgraphs),
        layout,
        kernel_options,
        sparse_q_block_size,
        sparse_kv_block_size,
        mutated_inputs=list(mutated_inputs),
    )
    use_split = (
        split_available
        and len(choices) == len(default_choices)
        and all(actual is default for actual, default in zip(choices, default_choices))
    )
    return choices, use_split


def _use_rocm_d256_panel_dots(
    is_gfx950, dtype, qk_head_dim, v_head_dim, kernel_options
):
    return (
        kernel_options.get("ROCM_D256_PANEL_DOTS", False)
        and is_gfx950
        and dtype == torch.bfloat16
        and qk_head_dim == 256
        and v_head_dim == 256
        and not kernel_options["USE_TMA"]
    )


def _use_rocm_d96_panel_phase(
    is_gfx950,
    query,
    key,
    value,
    kernel_options,
    score_is_trivial,
    has_captures,
    *,
    requested=False,
    mask_is_scalar_true=False,
    block_mask_is_default=False,
    grad_logsumexp_is_none=False,
    grad_out=None,
):
    if (
        requested is not True
        or not is_gfx950
        or not score_is_trivial
        or has_captures
        or not mask_is_scalar_true
        or not block_mask_is_default
        or not grad_logsumexp_is_none
        or kernel_options.get("SAFE_HEAD_DIM") is not False
        or any(
            kernel_options.get(option, False)
            for option in (
                "USE_TMA",
                "PRESCALE_QK",
                "ROCM_D256_PANEL_DOTS",
                "ROCM_D192_PANEL_DOTS",
                "ROCM_EARLY_DP",
                "ROCM_EARLY_PROBABILITY_COMPACT",
                "XCD_REMAP",
                "ROCM_XCD_SNAKE",
                "ROCM_GQA_SPLIT_DKDV",
                "num_consumer_groups",
            )
        )
    ):
        return False
    if tuple(
        kernel_options.get(option)
        for option in (
            "QK_HEAD_DIM",
            "V_HEAD_DIM",
            "QK_HEAD_DIM_ROUNDED",
            "V_HEAD_DIM_ROUNDED",
            "BLOCK_M1",
            "BLOCK_N1",
            "BLOCK_M2",
            "BLOCK_N2",
            "num_warps",
            "num_stages",
            "matrix_instr_nonkdim",
            "waves_per_eu",
            "kpack",
        )
    ) != (96, 96, 128, 128, 32, 128, 128, 32, 4, 2, 0, 0, 1):
        return False
    scale = kernel_options.get("SM_SCALE")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale):
        return False
    sizevars = V.graph.sizevars
    if (
        any(len(tensor.get_size()) != 4 for tensor in (query, key, value))
        or grad_out is None
        or grad_out.get_dtype() != torch.bfloat16
        or len(grad_out.get_size()) != 4
        or not all(
            sizevars.statically_known_equals(actual, expected)
            for actual, expected in zip(grad_out.get_size(), query.get_size())
        )
    ):
        return False
    if not any(
        all(
            sizevars.statically_known_equals(actual, expected)
            for actual, expected in zip(
                (
                    query.get_size()[0],
                    query.get_size()[1],
                    key.get_size()[1],
                    query.get_size()[2],
                    key.get_size()[2],
                ),
                shape,
            )
        )
        for shape in (
            (4, 21, 21, 2048, 2048),
            (4, 21, 21, 4096, 4096),
            (2, 8, 2, 2048, 2048),
            (2, 8, 2, 4096, 4096),
        )
    ):
        return False
    if not all(
        tensor.get_dtype() == torch.bfloat16
        and sizevars.statically_known_equals(tensor.get_size()[-1], 96)
        and sizevars.statically_known_equals(tensor.get_size()[0], query.get_size()[0])
        and all(sizevars.statically_known_gt(size, 0) for size in tensor.get_size()[:3])
        and sizevars.statically_known_geq(4096, tensor.get_size()[2])
        for tensor in (query, key, value)
    ):
        return False
    gqa = kernel_options.get("GQA_SHARED_HEADS")
    if not any(
        sizevars.statically_known_equals(gqa, ratio) for ratio in (1, 4)
    ) or not sizevars.statically_known_equals(
        query.get_size()[1], key.get_size()[1] * gqa
    ):
        return False
    if not all(
        sizevars.statically_known_equals(k_size, v_size)
        for k_size, v_size in zip(key.get_size(), value.get_size())
    ):
        return False
    for tensor in (query, key, value, grad_out):
        stride = 1
        for size, actual in zip(
            reversed(tensor.get_size()), reversed(tensor.get_stride())
        ):
            if not sizevars.statically_known_equals(actual, stride):
                return False
            stride *= size
    return True


def _use_rocm_d96_dq_early_dp(
    use_d96_panel_phase, kernel_options, *, requested=False, query=None, key=None
):
    if (
        requested is not True
        or use_d96_panel_phase is not True
        or kernel_options.get("ROCM_EARLY_DP", False) is not False
        or kernel_options.get("BLOCKS_ARE_CONTIGUOUS") is not True
        or query is None
        or key is None
        or len(query.get_size()) != 4
        or len(key.get_size()) != 4
    ):
        return False
    sizevars = V.graph.sizevars
    return (
        sizevars.statically_known_equals(kernel_options.get("ROCM_BWD_PHASE"), 0)
        and sizevars.statically_known_equals(kernel_options.get("ROCM_GQA_SPLITS"), 1)
        and any(
            all(
                sizevars.statically_known_equals(actual, expected)
                for actual, expected in zip(
                    (
                        query.get_size()[0],
                        query.get_size()[1],
                        key.get_size()[1],
                        query.get_size()[2],
                        key.get_size()[2],
                    ),
                    shape,
                )
            )
            for shape in ((4, 21, 21, 2048, 2048), (4, 21, 21, 4096, 4096))
        )
    )


def _use_rocm_d96_dq_unroll2(
    use_d96_panel_phase,
    kernel_options,
    *,
    requested=False,
    early_dq_dp_requested=False,
    other_requests=False,
    query=None,
    key=None,
):
    if requested is not True or other_requests is not False:
        return False
    return _use_rocm_d96_dq_early_dp(
        use_d96_panel_phase,
        kernel_options,
        requested=early_dq_dp_requested,
        query=query,
        key=key,
    )


def _use_rocm_d192_dq_early_dp(
    use_d192_panel_phase, kernel_options, *, requested=False
):
    return (
        requested is True
        and use_d192_panel_phase is True
        and kernel_options.get("ROCM_BWD_PHASE") == 0
        and kernel_options.get("ROCM_GQA_SPLITS") == 1
        and kernel_options.get("ROCM_EARLY_DP", False) is False
    )


def _use_rocm_d192_panel_phase(
    is_gfx950,
    query,
    key,
    value,
    kernel_options,
    score_is_trivial,
    has_captures,
    *,
    requested=False,
    mask_is_scalar_true=False,
    block_mask_is_default=False,
    grad_logsumexp_is_none=False,
    grad_out=None,
):
    if (
        requested is not True
        or not is_gfx950
        or not score_is_trivial
        or has_captures
        or not mask_is_scalar_true
        or not block_mask_is_default
        or not grad_logsumexp_is_none
        or kernel_options.get("SAFE_HEAD_DIM") is not False
        or any(
            kernel_options.get(option, False)
            for option in (
                "USE_TMA",
                "PRESCALE_QK",
                "ROCM_D256_PANEL_DOTS",
                "ROCM_EARLY_DP",
                "ROCM_EARLY_PROBABILITY_COMPACT",
                "XCD_REMAP",
                "ROCM_XCD_SNAKE",
                "ROCM_GQA_SPLIT_DKDV",
                "num_consumer_groups",
            )
        )
    ):
        return False
    if tuple(
        kernel_options.get(option)
        for option in (
            "QK_HEAD_DIM",
            "V_HEAD_DIM",
            "QK_HEAD_DIM_ROUNDED",
            "V_HEAD_DIM_ROUNDED",
            "BLOCK_M1",
            "BLOCK_N1",
            "BLOCK_M2",
            "BLOCK_N2",
            "num_warps",
            "num_stages",
            "matrix_instr_nonkdim",
            "waves_per_eu",
            "kpack",
        )
    ) != (192, 192, 256, 256, 16, 64, 128, 32, 4, 2, 0, 0, 1):
        return False
    scale = kernel_options.get("SM_SCALE")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale):
        return False
    sizevars = V.graph.sizevars
    if (
        grad_out is None
        or grad_out.get_dtype() != torch.bfloat16
        or len(grad_out.get_size()) != 4
        or not all(
            sizevars.statically_known_equals(actual, expected)
            for actual, expected in zip(grad_out.get_size(), query.get_size())
        )
    ):
        return False
    tensors = (query, key, value, grad_out)
    if any(
        len(tensor.get_size()) != 4 or len(tensor.get_stride()) != 4
        for tensor in tensors
    ):
        return False
    if not any(
        sizevars.statically_known_equals(
            kernel_options.get("GQA_SHARED_HEADS"), q_heads // kv_heads
        )
        and all(
            sizevars.statically_known_equals(actual, expected)
            for tensor, shape in zip(
                tensors,
                (
                    (batch, q_heads, q_len, dim),
                    (batch, kv_heads, kv_len, dim),
                    (batch, kv_heads, kv_len, dim),
                    (batch, q_heads, q_len, dim),
                ),
                strict=True,
            )
            for actual, expected in zip(tensor.get_size(), shape, strict=True)
        )
        for batch, q_heads, kv_heads, q_len, kv_len, dim in (
            (4, 10, 10, 2048, 2048, 192),
            (4, 10, 10, 4096, 4096, 192),
            (2, 8, 2, 2048, 2048, 192),
        )
    ):
        return False
    if not all(
        tensor.get_dtype() == torch.bfloat16
        and sizevars.statically_known_equals(tensor.get_size()[-1], 192)
        and sizevars.statically_known_equals(tensor.get_size()[0], query.get_size()[0])
        and all(sizevars.statically_known_gt(size, 0) for size in tensor.get_size()[:3])
        and sizevars.statically_known_geq(4096, tensor.get_size()[2])
        for tensor in (query, key, value)
    ):
        return False
    for tensor in (query, key, value, grad_out):
        stride = 1
        for size, actual in zip(
            reversed(tensor.get_size()), reversed(tensor.get_stride())
        ):
            if not sizevars.statically_known_equals(actual, stride):
                return False
            stride *= size
    return True


def _use_rocm_d128_fp16_coverage(
    is_gfx950,
    query,
    key,
    value,
    grad_out,
    out,
    logsumexp,
    kernel_options,
    score_is_trivial,
    has_captures,
    *,
    requested=False,
    other_requests=False,
    mask_is_causal=False,
    grad_logsumexp_is_none=False,
):
    if (
        requested is not True
        or other_requests is not False
        or not is_gfx950
        or not score_is_trivial
        or has_captures
        or not mask_is_causal
        or not grad_logsumexp_is_none
        or any(
            kernel_options.get(name) is not True
            for name in (
                "ROCM_SEPARATE_DQ",
                "ROCM_EARLY_DP",
                "XCD_REMAP",
                "SAFE_HEAD_DIM",
                "IS_DIVISIBLE",
            )
        )
        or type(kernel_options.get("BLOCKS_ARE_CONTIGUOUS")) is not bool
        or type(kernel_options.get("HAS_FULL_BLOCKS")) is not bool
        or any(
            kernel_options.get(name, False) is not False
            for name in (
                "USE_TMA",
                "PRESCALE_QK",
                "ROCM_XCD_SNAKE",
                "ROCM_D256_PANEL_DOTS",
                "ROCM_D192_PANEL_DOTS",
                "ROCM_D96_PANEL_DOTS",
                "ROCM_D192_DQ_EARLY_DP",
                "ROCM_D96_DQ_EARLY_DP",
                "ROCM_D96_DQ_UNROLL2",
                "ROCM_EARLY_PROBABILITY_COMPACT",
                "ROCM_GQA_SPLIT_DKDV",
                "ROCM_D64_PAIRED_DQ_ROWS",
                "ROCM_D64_FOUR_DQ_ROWS",
                "ROCM_PACKED_STATISTICS",
                "ROCM_DQ_DELTA_PRODUCER",
                "ROCM_FUSED_DELTA",
            )
        )
        or any(
            kernel_options.get(name, 0)
            for name in (
                "num_consumer_groups",
                "waves_per_simd",
                "llvm_fn_attrs",
            )
        )
    ):
        return False
    scale = kernel_options.get("SM_SCALE")
    if type(scale) not in (int, float) or not math.isfinite(scale):
        return False
    sizevars = V.graph.sizevars
    if (
        not all(
            type(kernel_options.get(name)) not in (bool, float, str)
            and sizevars.statically_known_equals(kernel_options.get(name), expected)
            for name, expected in (
                ("ROCM_BWD_PHASE", 0),
                ("ROCM_GQA_SPLITS", 1),
                ("GQA_SHARED_HEADS", 8),
                ("QK_HEAD_DIM", 128),
                ("V_HEAD_DIM", 128),
                ("QK_HEAD_DIM_ROUNDED", 128),
                ("V_HEAD_DIM_ROUNDED", 128),
                ("BLOCK_M1", 32),
                ("BLOCK_N1", 128),
                ("BLOCK_M2", 128),
                ("BLOCK_N2", 32),
                ("num_warps", 4),
                ("num_stages", 2),
                ("matrix_instr_nonkdim", 0),
                ("waves_per_eu", 0),
                ("SPARSE_Q_BLOCK_SIZE", 128),
                ("SPARSE_KV_BLOCK_SIZE", 128),
            )
        )
        or type(kernel_options.get("kpack")) in (bool, float, str)
        or not any(
            sizevars.statically_known_equals(kernel_options.get("kpack"), value)
            for value in (1, 2)
        )
    ):
        return False
    operands = next(
        (
            (
                (query, (batch, 64, length, 128), torch.float16),
                (key, (batch, 8, length, 128), torch.float16),
                (value, (batch, 8, length, 128), torch.float16),
                (grad_out, (batch, 64, length, 128), torch.float16),
                (out, (batch, 64, length, 128), torch.float16),
                (logsumexp, (batch, 64, length), torch.float32),
            )
            for batch, length in ((4, 2048), (4, 4096), (2, 4096), (1, 4096), (2, 2048))
            if all(
                tensor is not None
                and len(tensor.get_size()) == len(shape)
                and all(
                    type(actual) not in (bool, float, str)
                    and sizevars.statically_known_equals(actual, expected)
                    for actual, expected in zip(tensor.get_size(), shape)
                )
                for tensor, shape in (
                    (query, (batch, 64, length, 128)),
                    (key, (batch, 8, length, 128)),
                    (value, (batch, 8, length, 128)),
                    (grad_out, (batch, 64, length, 128)),
                    (out, (batch, 64, length, 128)),
                    (logsumexp, (batch, 64, length)),
                )
            )
        ),
        None,
    )
    if operands is None:
        return False
    for tensor, shape, dtype in operands:
        if (
            tensor is None
            or tensor.get_dtype() != dtype
            or len(tensor.get_size()) != len(shape)
            or len(tensor.get_stride()) != len(shape)
            or tensor.get_device() != query.get_device()
        ):
            return False
        if not all(
            type(actual) not in (bool, float, str)
            and sizevars.statically_known_equals(actual, expected)
            for actual, expected in zip(tensor.get_size(), shape)
        ):
            return False
        stride = 1
        for size, actual in zip(
            reversed(tensor.get_size()), reversed(tensor.get_stride())
        ):
            if type(actual) in (
                bool,
                float,
                str,
            ) or not sizevars.statically_known_equals(actual, stride):
                return False
            stride *= size
    return True


def _use_rocm_d128_fp32_balanced64_coverage(
    is_gfx950,
    query,
    key,
    value,
    grad_out,
    out,
    logsumexp,
    kernel_options,
    score_is_trivial,
    has_captures,
    *,
    requested=False,
    other_requests=False,
    mask_is_causal=False,
    grad_logsumexp_is_none=False,
):
    if (
        requested is not True
        or other_requests is not False
        or query is None
        or getattr(query.get_device(), "type", None) != "cuda"
        or type(kernel_options.get("FLOAT32_PRECISION")) is not str
        or kernel_options["FLOAT32_PRECISION"] != "'ieee'"
        or is_gfx950 is not True
        or score_is_trivial is not True
        or has_captures is not False
        or mask_is_causal is not True
        or grad_logsumexp_is_none is not True
        or any(
            kernel_options.get(name) is not True
            for name in (
                "ROCM_SEPARATE_DQ",
                "ROCM_EARLY_DP",
                "XCD_REMAP",
                "SAFE_HEAD_DIM",
                "IS_DIVISIBLE",
            )
        )
        or type(kernel_options.get("BLOCKS_ARE_CONTIGUOUS")) is not bool
        or type(kernel_options.get("HAS_FULL_BLOCKS")) is not bool
        or any(
            kernel_options.get(name, False) is not False
            for name in (
                "USE_TMA",
                "PRESCALE_QK",
                "ROCM_XCD_SNAKE",
                "ROCM_D256_PANEL_DOTS",
                "ROCM_D192_PANEL_DOTS",
                "ROCM_D96_PANEL_DOTS",
                "ROCM_D192_DQ_EARLY_DP",
                "ROCM_D96_DQ_EARLY_DP",
                "ROCM_D96_DQ_UNROLL2",
                "ROCM_EARLY_PROBABILITY_COMPACT",
                "ROCM_GQA_SPLIT_DKDV",
                "ROCM_D64_PAIRED_DQ_ROWS",
                "ROCM_D64_FOUR_DQ_ROWS",
                "ROCM_PACKED_STATISTICS",
                "ROCM_DQ_DELTA_PRODUCER",
                "ROCM_FUSED_DELTA",
            )
        )
        or any(
            kernel_options.get(name, 0)
            for name in (
                "num_consumer_groups",
                "waves_per_simd",
                "llvm_fn_attrs",
            )
        )
    ):
        return False
    scale = kernel_options.get("SM_SCALE")
    if type(scale) not in (int, float) or not math.isfinite(scale):
        return False
    sizevars = V.graph.sizevars
    if (
        not all(
            type(kernel_options.get(name)) not in (bool, float, str)
            and sizevars.statically_known_equals(kernel_options.get(name), expected)
            for name, expected in (
                ("ROCM_BWD_PHASE", 0),
                ("ROCM_GQA_SPLITS", 1),
                ("GQA_SHARED_HEADS", 8),
                ("QK_HEAD_DIM", 128),
                ("V_HEAD_DIM", 128),
                ("QK_HEAD_DIM_ROUNDED", 128),
                ("V_HEAD_DIM_ROUNDED", 128),
                ("BLOCK_M1", 16),
                ("BLOCK_N1", 64),
                ("BLOCK_M2", 64),
                ("BLOCK_N2", 16),
                ("num_warps", 4),
                ("num_stages", 1),
                ("matrix_instr_nonkdim", 0),
                ("waves_per_eu", 0),
                ("SPARSE_Q_BLOCK_SIZE", 128),
                ("SPARSE_KV_BLOCK_SIZE", 128),
            )
        )
        or type(kernel_options.get("kpack")) in (bool, float, str)
        or not (sizevars.statically_known_equals(kernel_options.get("kpack"), 1))
        or any(
            type(kernel_options.get(name, 0)) in (bool, float, str)
            or not sizevars.statically_known_equals(kernel_options.get(name, 0), 0)
            for name in ("num_consumer_groups", "waves_per_simd", "llvm_fn_attrs")
        )
    ):
        return False
    operands = next(
        (
            (
                (query, (batch, 64, length, 128), torch.float32),
                (key, (batch, 8, length, 128), torch.float32),
                (value, (batch, 8, length, 128), torch.float32),
                (grad_out, (batch, 64, length, 128), torch.float32),
                (out, (batch, 64, length, 128), torch.float32),
                (logsumexp, (batch, 64, length), torch.float32),
            )
            for batch, length in (
                (4, 2048),
                (4, 4096),
                (2, 4096),
                (1, 4096),
                (1, 2048),
                (2, 2048),
            )
            if all(
                tensor is not None
                and len(tensor.get_size()) == len(shape)
                and all(
                    type(actual) not in (bool, float, str)
                    and sizevars.statically_known_equals(actual, expected)
                    for actual, expected in zip(tensor.get_size(), shape)
                )
                for tensor, shape in (
                    (query, (batch, 64, length, 128)),
                    (key, (batch, 8, length, 128)),
                    (value, (batch, 8, length, 128)),
                    (grad_out, (batch, 64, length, 128)),
                    (out, (batch, 64, length, 128)),
                    (logsumexp, (batch, 64, length)),
                )
            )
        ),
        None,
    )
    if operands is None:
        return False
    for tensor, shape, dtype in operands:
        if (
            tensor is None
            or tensor.get_dtype() != dtype
            or len(tensor.get_size()) != len(shape)
            or len(tensor.get_stride()) != len(shape)
            or tensor.get_device() != query.get_device()
        ):
            return False
        if not all(
            type(actual) not in (bool, float, str)
            and sizevars.statically_known_equals(actual, expected)
            for actual, expected in zip(tensor.get_size(), shape)
        ):
            return False
        stride = 1
        for size, actual in zip(
            reversed(tensor.get_size()), reversed(tensor.get_stride())
        ):
            if type(actual) in (
                bool,
                float,
                str,
            ) or not sizevars.statically_known_equals(actual, stride):
                return False
            stride *= size
    return True


def _use_rocm_d128_reverse_causal_dq_rows(
    use_separate_dq,
    is_gfx950,
    query,
    key,
    value,
    grad_out,
    kernel_options,
    score_is_trivial,
    has_captures,
    *,
    requested=False,
    mask_is_causal=False,
    grad_logsumexp_is_none=False,
    other_requests=False,
    fp16_pilot=False,
    fp32_balanced64_pilot=False,
):
    if fp32_balanced64_pilot is not False and (
        fp32_balanced64_pilot is not True
        or fp16_pilot is not False
        or any(
            type(kernel_options.get(option)) in (bool, float, str)
            for option in (
                "BLOCK_M1",
                "BLOCK_N1",
                "BLOCK_M2",
                "BLOCK_N2",
                "num_warps",
                "num_stages",
            )
        )
    ):
        return False
    if (
        requested is not True
        or not use_separate_dq
        or not is_gfx950
        or not score_is_trivial
        or has_captures
        or other_requests
        or not mask_is_causal
        or not grad_logsumexp_is_none
        or kernel_options.get("ROCM_SEPARATE_DQ") is not True
        or kernel_options.get("ROCM_EARLY_DP") is not True
        or kernel_options.get("XCD_REMAP") is not True
        or type(kernel_options.get("BLOCKS_ARE_CONTIGUOUS")) is not bool
        or kernel_options.get("SAFE_HEAD_DIM") is not True
        or kernel_options.get("IS_DIVISIBLE") is not True
        or type(kernel_options.get("HAS_FULL_BLOCKS")) is not bool
        or any(
            kernel_options.get(option, False)
            for option in (
                "USE_TMA",
                "PRESCALE_QK",
                "ROCM_XCD_SNAKE",
                "ROCM_D256_PANEL_DOTS",
                "ROCM_D192_PANEL_DOTS",
                "ROCM_D96_PANEL_DOTS",
                "ROCM_D192_DQ_EARLY_DP",
                "ROCM_D96_DQ_EARLY_DP",
                "ROCM_EARLY_PROBABILITY_COMPACT",
                "ROCM_GQA_SPLIT_DKDV",
                "ROCM_D64_PAIRED_DQ_ROWS",
                "ROCM_D64_FOUR_DQ_ROWS",
                "ROCM_PACKED_STATISTICS",
                "ROCM_DQ_DELTA_PRODUCER",
                "ROCM_FUSED_DELTA",
                "num_consumer_groups",
                "waves_per_simd",
                "llvm_fn_attrs",
            )
        )
    ):
        return False
    scale = kernel_options.get("SM_SCALE")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale):
        return False
    sizevars = V.graph.sizevars
    if (fp16_pilot is True or fp32_balanced64_pilot is True) and not any(
        all(
            tensor is not None
            and len(tensor.get_size()) == 4
            and all(
                sizevars.statically_known_equals(actual, expected)
                for actual, expected in zip(tensor.get_size(), shape)
            )
            for tensor, shape in (
                (query, (batch, 64, length, 128)),
                (key, (batch, 8, length, 128)),
                (value, (batch, 8, length, 128)),
                (grad_out, (batch, 64, length, 128)),
            )
        )
        for batch, length in (
            ((4, 2048), (4, 4096), (2, 4096), (1, 4096), (2, 2048))
            if fp16_pilot is True
            else ((4, 2048), (4, 4096), (2, 4096), (1, 4096), (1, 2048), (2, 2048))
        )
    ):
        return False
    if not all(
        sizevars.statically_known_equals(kernel_options.get(name), expected)
        for name, expected in (
            ("ROCM_BWD_PHASE", 0),
            ("ROCM_GQA_SPLITS", 1),
            ("GQA_SHARED_HEADS", 8),
            ("QK_HEAD_DIM", 128),
            ("V_HEAD_DIM", 128),
            ("QK_HEAD_DIM_ROUNDED", 128),
            ("V_HEAD_DIM_ROUNDED", 128),
            ("BLOCK_M1", 16 if fp32_balanced64_pilot is True else 32),
            ("BLOCK_N1", 64 if fp32_balanced64_pilot is True else 128),
            ("BLOCK_M2", 64 if fp32_balanced64_pilot is True else 128),
            ("BLOCK_N2", 16 if fp32_balanced64_pilot is True else 32),
            ("num_warps", 4),
            ("num_stages", 1 if fp32_balanced64_pilot is True else 2),
            ("matrix_instr_nonkdim", 0),
            ("waves_per_eu", 0),
            ("SPARSE_Q_BLOCK_SIZE", 128),
            ("SPARSE_KV_BLOCK_SIZE", 128),
        )
    ) or not any(
        sizevars.statically_known_equals(kernel_options.get("kpack"), expected)
        for expected in (1, 2)
    ):
        return False
    shapes = next(
        (
            (
                (batch, q_heads, q_len, dim),
                (batch, kv_heads, kv_len, dim),
                (batch, kv_heads, kv_len, dim),
                (batch, q_heads, q_len, dim),
            )
            for batch, q_heads, kv_heads, q_len, kv_len, dim in (
                (4, 64, 8, 4096, 4096, 128),
                (4, 64, 8, 2048, 2048, 128),
                (4, 64, 8, 1024, 1024, 128),
                (2, 64, 8, 4096, 4096, 128),
                *(
                    ((1, 64, 8, 4096, 4096, 128), (2, 64, 8, 2048, 2048, 128))
                    if fp16_pilot is True
                    else ()
                ),
                *(
                    (
                        (1, 64, 8, 4096, 4096, 128),
                        (1, 64, 8, 2048, 2048, 128),
                        (2, 64, 8, 2048, 2048, 128),
                    )
                    if fp32_balanced64_pilot is True
                    else ()
                ),
            )
            if all(
                tensor is not None
                and len(tensor.get_size()) == 4
                and all(
                    sizevars.statically_known_equals(actual, expected)
                    for actual, expected in zip(tensor.get_size(), shape)
                )
                for tensor, shape in (
                    (query, (batch, q_heads, q_len, dim)),
                    (key, (batch, kv_heads, kv_len, dim)),
                    (value, (batch, kv_heads, kv_len, dim)),
                    (grad_out, (batch, q_heads, q_len, dim)),
                )
            )
        ),
        None,
    )
    if shapes is None:
        return False
    for tensor, shape in zip((query, key, value, grad_out), shapes):
        if (
            tensor is None
            or tensor.get_dtype()
            != (
                torch.float32
                if fp32_balanced64_pilot is True
                else torch.float16
                if fp16_pilot is True
                else torch.bfloat16
            )
            or len(tensor.get_size()) != 4
            or not all(
                sizevars.statically_known_equals(actual, expected)
                for actual, expected in zip(tensor.get_size(), shape)
            )
        ):
            return False
        stride = 1
        for size, actual in zip(
            reversed(tensor.get_size()), reversed(tensor.get_stride())
        ):
            if not sizevars.statically_known_equals(actual, stride):
                return False
            stride *= size
    return True


def _use_rocm_d128_dv_view_carry_split64(
    use_separate_dq,
    is_gfx950,
    query,
    key,
    value,
    grad_out,
    kernel_options,
    score_is_trivial,
    has_captures,
    *,
    requested=False,
    reverse_requested=False,
    mask_is_causal=False,
    grad_logsumexp_is_none=False,
    other_requests=False,
    reverse_other_requests=False,
    fp16_pilot=False,
    fp32_balanced64_pilot=False,
):
    if requested is not True or other_requests:
        return False
    return _use_rocm_d128_reverse_causal_dq_rows(
        use_separate_dq,
        is_gfx950,
        query,
        key,
        value,
        grad_out,
        kernel_options,
        score_is_trivial,
        has_captures,
        requested=reverse_requested,
        mask_is_causal=mask_is_causal,
        grad_logsumexp_is_none=grad_logsumexp_is_none,
        other_requests=reverse_other_requests,
        fp16_pilot=fp16_pilot,
        fp32_balanced64_pilot=fp32_balanced64_pilot,
    )


def _use_rocm_separate_dq(
    is_gfx950,
    query,
    key,
    value,
    kernel_options,
    score_is_trivial,
    has_captures,
    *,
    mask_is_scalar_true=False,
    block_mask_is_default=False,
    fp16_pilot=False,
    fp32_balanced64_pilot=False,
):
    # Measured D128 gains also use early dP and XCD remapping.
    # The D128 phase split alone runs slower on measured short and causal cases.
    if fp32_balanced64_pilot is not False and (
        fp32_balanced64_pilot is not True
        or fp16_pilot is not False
        or any(
            type(kernel_options.get(option)) in (bool, float, str)
            for option in (
                "BLOCK_M1",
                "BLOCK_N1",
                "BLOCK_M2",
                "BLOCK_N2",
                "num_warps",
                "num_stages",
            )
        )
    ):
        return False
    if (
        not kernel_options.get("ROCM_SEPARATE_DQ", False)
        or not is_gfx950
        or not score_is_trivial
        or has_captures
        or kernel_options["USE_TMA"]
        or kernel_options["PRESCALE_QK"]
        or kernel_options.get("num_consumer_groups", 0)
        or (
            query.get_dtype() != torch.bfloat16
            and fp16_pilot is not True
            and fp32_balanced64_pilot is not True
        )
    ):
        return False
    sizevars = V.graph.sizevars
    if (fp16_pilot is True or fp32_balanced64_pilot is True) and not any(
        all(
            tensor is not None
            and len(tensor.get_size()) == 4
            and all(
                sizevars.statically_known_equals(actual, expected)
                for actual, expected in zip(tensor.get_size(), shape)
            )
            for tensor, shape in (
                (query, (batch, 64, length, 128)),
                (key, (batch, 8, length, 128)),
                (value, (batch, 8, length, 128)),
            )
        )
        for batch, length in (
            ((4, 2048), (4, 4096), (2, 4096), (1, 4096), (2, 2048))
            if fp16_pilot is True
            else ((4, 2048), (4, 4096), (2, 4096), (1, 4096), (1, 2048), (2, 2048))
        )
    ):
        return False
    if sizevars.statically_known_equals(query.get_size()[-1], 64):
        if (
            fp16_pilot is not False
            or fp32_balanced64_pilot is not False
            or not mask_is_scalar_true
            or not block_mask_is_default
            or kernel_options.get("ROCM_EARLY_DP") is not True
            or kernel_options.get("XCD_REMAP") is not False
            or kernel_options.get("HAS_FULL_BLOCKS") is not False
            or kernel_options.get("SAFE_HEAD_DIM") is not True
            or kernel_options.get("IS_DIVISIBLE") is not True
            or kernel_options.get("SM_SCALE") != 0.125
            or type(kernel_options.get("SM_SCALE")) not in (int, float)
        ):
            return False
        for tensor, shape in (
            (query, (4, 64, 4096, 64)),
            (key, (4, 8, 4096, 64)),
            (value, (4, 8, 4096, 64)),
        ):
            if (
                tensor.get_dtype() != torch.bfloat16
                or len(tensor.get_size()) != 4
                or len(tensor.get_stride()) != 4
                or not all(
                    type(actual) not in (bool, float, str)
                    and sizevars.statically_known_equals(actual, expected)
                    for actual, expected in zip(tensor.get_size(), shape)
                )
            ):
                return False
            stride = 1
            for size, actual in zip(reversed(shape), reversed(tensor.get_stride())):
                if not sizevars.statically_known_equals(actual, stride):
                    return False
                stride *= size
        return all(
            kernel_options.get(name) is not None
            and type(kernel_options.get(name)) not in (bool, float, str)
            and sizevars.statically_known_equals(kernel_options.get(name), expected)
            for name, expected in (
                ("BLOCK_M1", 32),
                ("BLOCK_N1", 128),
                ("BLOCK_M2", 128),
                ("BLOCK_N2", 32),
                ("num_warps", 4),
                ("num_stages", 2),
                ("GQA_SHARED_HEADS", 8),
                ("matrix_instr_nonkdim", 0),
                ("waves_per_eu", 0),
                ("kpack", 1),
                ("QK_HEAD_DIM", 64),
                ("V_HEAD_DIM", 64),
                ("QK_HEAD_DIM_ROUNDED", 64),
                ("V_HEAD_DIM_ROUNDED", 64),
                ("SPARSE_Q_BLOCK_SIZE", 1 << 30),
                ("SPARSE_KV_BLOCK_SIZE", 1 << 30),
            )
        )
    if sizevars.statically_known_equals(query.get_size()[-1], 256):
        if (
            not mask_is_scalar_true
            or not block_mask_is_default
            or not kernel_options.get("ROCM_D256_PANEL_DOTS", False)
        ):
            return False
        if not all(
            tensor.get_dtype() == torch.bfloat16
            and sizevars.statically_known_equals(tensor.get_size()[-1], 256)
            and sizevars.statically_known_equals(
                tensor.get_size()[0], query.get_size()[0]
            )
            and all(
                sizevars.statically_known_gt(size, 0) for size in tensor.get_size()[:3]
            )
            for tensor in (query, key, value)
        ):
            return False
        gqa = kernel_options["GQA_SHARED_HEADS"]
        if (
            not any(sizevars.statically_known_equals(gqa, ratio) for ratio in (1, 4))
            or not sizevars.statically_known_equals(
                query.get_size()[1], key.get_size()[1] * gqa
            )
            or not all(
                sizevars.statically_known_equals(k_size, v_size)
                for k_size, v_size in zip(key.get_size(), value.get_size())
            )
        ):
            return False
        for tensor in (query, key, value):
            stride = 1
            for size, actual in zip(
                reversed(tensor.get_size()), reversed(tensor.get_stride())
            ):
                if not sizevars.statically_known_equals(actual, stride):
                    return False
                stride *= size
        selected = tuple(
            kernel_options[option]
            for option in (
                "BLOCK_M1",
                "BLOCK_N1",
                "BLOCK_M2",
                "BLOCK_N2",
                "num_warps",
                "num_stages",
            )
        )
        return selected in ((16, 64, 64, 16, 4, 2), (16, 64, 128, 32, 4, 2))
    if not all(
        tensor.get_dtype()
        == (
            torch.float32
            if fp32_balanced64_pilot is True
            else torch.float16
            if fp16_pilot is True
            else torch.bfloat16
        )
        and sizevars.statically_known_equals(tensor.get_size()[-1], 128)
        and sizevars.statically_known_equals(tensor.get_size()[0], query.get_size()[0])
        and all(sizevars.statically_known_gt(size, 0) for size in tensor.get_size()[:3])
        for tensor in (query, key, value)
    ):
        return False
    if not sizevars.statically_known_equals(
        query.get_size()[1], key.get_size()[1] * 8
    ) or not all(
        sizevars.statically_known_equals(k_size, v_size)
        for k_size, v_size in zip(key.get_size(), value.get_size())
    ):
        return False
    for tensor in (query, key, value):
        stride = 1
        for size, actual in zip(
            reversed(tensor.get_size()), reversed(tensor.get_stride())
        ):
            if not sizevars.statically_known_equals(actual, stride):
                return False
            stride *= size
    options = (
        "BLOCK_M1",
        "BLOCK_N1",
        "BLOCK_M2",
        "BLOCK_N2",
        "num_warps",
        "num_stages",
        "GQA_SHARED_HEADS",
    )
    if fp32_balanced64_pilot is True:
        return tuple(kernel_options[option] for option in options) == (
            16,
            64,
            64,
            16,
            4,
            1,
            8,
        )
    return tuple(kernel_options[option] for option in options) == (
        32,
        128,
        128,
        32,
        4,
        2,
        8,
    )


def _use_rocm_early_probability_compact(
    is_gfx950,
    query,
    key,
    value,
    kernel_options,
    score_is_trivial,
    has_captures,
    *,
    requested=False,
    mask_is_scalar_true=False,
    block_mask_is_default=False,
):
    """Admit backward-scoped compaction for the qualified dense phase1 body."""
    if (
        requested is not True
        or kernel_options.get("ROCM_EARLY_DP", False) is not True
        or not V.graph.sizevars.statically_known_equals(query.get_size()[-1], 256)
        or tuple(
            kernel_options.get(option)
            for option in (
                "BLOCK_M1",
                "BLOCK_N1",
                "BLOCK_M2",
                "BLOCK_N2",
                "num_warps",
                "num_stages",
            )
        )
        != (16, 64, 128, 32, 4, 2)
    ):
        return False
    return _use_rocm_separate_dq(
        is_gfx950,
        query,
        key,
        value,
        kernel_options,
        score_is_trivial,
        has_captures,
        mask_is_scalar_true=mask_is_scalar_true,
        block_mask_is_default=block_mask_is_default,
    )


def _is_rocm_default_block_mask(block_mask, query, key):
    # This checks the default mask structure, not the runtime block-list values.
    sizevars = V.graph.sizevars
    if not all(
        sizevars.statically_known_equals(length, 1) for length in block_mask[:2]
    ):
        return False
    if not all(
        sizevars.statically_known_equals(size, _LARGE_SPARSE_BLOCK_SIZE)
        for size in block_mask[14:16]
    ):
        return False
    if not all(block_mask[index] is None for index in (4, 5, 8, 9, 10, 11, 12, 13)):
        return False
    for index, rank in ((2, 3), (3, 4), (6, 3), (7, 4)):
        node = block_mask[index]
        if node is None or len(node.get_size()) != rank:
            return False
        if not all(
            sizevars.statically_known_equals(size, 1) for size in node.get_size()
        ):
            return False
    return sizevars.statically_known_geq(
        _LARGE_SPARSE_BLOCK_SIZE, query.get_size()[2]
    ) and sizevars.statically_known_geq(_LARGE_SPARSE_BLOCK_SIZE, key.get_size()[2])


def _is_rocm_scalar_true_mask_graph(graph_module):
    nodes = list(graph_module.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    outputs = [node for node in nodes if node.op == "output"]
    if len(placeholders) != 4 or len(outputs) != 1 or len(outputs[0].args) != 1:
        return False
    output = outputs[0].args[0]
    if (
        not isinstance(output, torch.fx.Node)
        or output.op != "call_function"
        or output.target is not torch.ops.aten.full.default
        or len(output.args) != 2
    ):
        return False
    shape, fill = output.args
    return (
        isinstance(shape, (tuple, list))
        and len(shape) == 0
        and fill is True
        and output.kwargs.get("dtype") is torch.bool
    )


def _is_rocm_simple_causal_mask_graph(graph_module):
    nodes = list(graph_module.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    outputs = [node for node in nodes if node.op == "output"]
    if len(nodes) != 6 or len(placeholders) != 4 or len(outputs) != 1:
        return False
    if len(outputs[0].args) != 1:
        return False
    output = outputs[0].args[0]
    if (
        not isinstance(output, torch.fx.Node)
        or output.op != "call_function"
        or len(output.args) != 2
        or output.kwargs
    ):
        return False
    left, right = output.args
    return (
        output.target is torch.ops.aten.ge.Tensor
        and left is placeholders[2]
        and right is placeholders[3]
    ) or (
        output.target is torch.ops.aten.le.Tensor
        and left is placeholders[3]
        and right is placeholders[2]
    )


def _use_rocm_xcd_snake(
    is_gfx950,
    query,
    key,
    value,
    kernel_options,
    score_is_trivial,
    mask_is_simple_causal,
    has_captures,
):
    sizevars = V.graph.sizevars
    if (
        kernel_options.get("ROCM_XCD_SNAKE", False) is not True
        or not is_gfx950
        or not score_is_trivial
        or not mask_is_simple_causal
        or has_captures
        or not sizevars.statically_known_equals(
            kernel_options.get("num_consumer_groups", 0), 0
        )
        or any(
            kernel_options.get(option, False) is not False
            for option in (
                "USE_TMA",
                "PRESCALE_QK",
                "ROCM_D256_PANEL_DOTS",
                "ROCM_SEPARATE_DQ",
                "ROCM_GQA_SPLIT_DKDV",
                "ROCM_EARLY_DP",
                "XCD_REMAP",
            )
        )
    ):
        return False
    if not all(
        tensor.get_dtype() == torch.bfloat16
        and sizevars.statically_known_equals(tensor.get_size()[-1], 256)
        and sizevars.statically_known_equals(tensor.get_size()[0], query.get_size()[0])
        and all(sizevars.statically_known_gt(size, 0) for size in tensor.get_size()[:3])
        for tensor in (query, key, value)
    ):
        return False
    gqa = kernel_options["GQA_SHARED_HEADS"]
    if (
        not any(sizevars.statically_known_equals(gqa, ratio) for ratio in (1, 4))
        or not sizevars.statically_known_equals(
            query.get_size()[1], key.get_size()[1] * gqa
        )
        or not all(
            sizevars.statically_known_equals(k_size, v_size)
            for k_size, v_size in zip(key.get_size(), value.get_size())
        )
    ):
        return False
    for tensor in (query, key, value):
        stride = 1
        for size, actual in zip(
            reversed(tensor.get_size()), reversed(tensor.get_stride())
        ):
            if not sizevars.statically_known_equals(actual, stride):
                return False
            stride *= size
    if not all(
        sizevars.statically_known_geq(2147483647, tensor.get_size()[2] + 63)
        for tensor in (query, key)
    ):
        return False
    options = (
        "BLOCK_M1",
        "BLOCK_N1",
        "BLOCK_M2",
        "BLOCK_N2",
        "num_warps",
        "num_stages",
        "ROCM_BWD_PHASE",
        "ROCM_GQA_SPLITS",
    )
    return all(
        sizevars.statically_known_equals(kernel_options[option], expected)
        for option, expected in zip(options, (32, 64, 64, 64, 4, 1, 0, 1))
    )


def _use_rocm_d128_gqa_query_tile_outer(
    input_nodes, kernel_options, *, single_config=False
):
    if type(kernel_options.get("BLOCKS_ARE_CONTIGUOUS")) is not bool:
        return False
    query = input_nodes[0] if len(input_nodes) == 16 else None
    dtype = query.get_dtype() if query is not None else None
    query_shape = query.get_size() if query is not None else ()
    fp16 = dtype == torch.float16
    if dtype == torch.float32 or fp16:
        if (
            single_config is not True
            or input_nodes[6] is not input_nodes[0]
            or any(
                kernel_options.get(name) is not True
                for name in (
                    "ROCM_D128_DV_VIEW_CARRY_SPLIT64",
                    "ROCM_SEPARATE_DQ",
                    "ROCM_EARLY_DP",
                    "XCD_REMAP",
                    "HAS_FULL_BLOCKS",
                    "IS_DIVISIBLE",
                    "SAFE_HEAD_DIM",
                )
            )
            or any(
                kernel_options.get(name) is not False
                for name in ("USE_TMA", "PRESCALE_QK")
            )
            or type(kernel_options.get("FLOAT32_PRECISION")) is not str
            or kernel_options["FLOAT32_PRECISION"] != "'ieee'"
        ):
            return False
        scale = kernel_options.get("SM_SCALE")
        if type(scale) not in (int, float) or scale not in (
            0.08838834764831843,
            0.0,
            -0.125,
        ):
            return False
        sizevars = V.graph.sizevars
        if not all(
            kernel_options.get(name) is not None
            and type(kernel_options[name]) not in (bool, float, str)
            and sizevars.statically_known_equals(kernel_options[name], expected)
            for name, expected in (
                ("ROCM_BWD_PHASE", 1),
                ("ROCM_GQA_SPLITS", 1),
                ("GQA_SHARED_HEADS", 8),
                ("QK_HEAD_DIM", 128),
                ("V_HEAD_DIM", 128),
                ("QK_HEAD_DIM_ROUNDED", 128),
                ("V_HEAD_DIM_ROUNDED", 128),
                ("BLOCK_M1", 32 if fp16 else 16),
                ("BLOCK_N1", 128 if fp16 else 64),
                ("BLOCK_M2", 128 if fp16 else 64),
                ("BLOCK_N2", 32 if fp16 else 16),
                ("num_warps", 4),
                ("num_stages", 2 if fp16 else 1),
                ("matrix_instr_nonkdim", 0),
                ("waves_per_eu", 0),
                ("kpack", 1),
                ("SPARSE_Q_BLOCK_SIZE", 128),
                ("SPARSE_KV_BLOCK_SIZE", 128),
            )
        ):
            return False
        if len(query_shape) != 4 or any(
            value is None or type(value) in (bool, float, str)
            for value in (query_shape[0], query_shape[2])
        ):
            return False
        shape_pairs = (
            ((4, 2048), (4, 4096), (2, 2048), (2, 4096), (1, 4096))
            if fp16
            else ((4, 4096), (2, 4096), (1, 4096), (4, 2048))
        )
        shape_pair = next(
            (
                (batch, length)
                for batch, length in shape_pairs
                if sizevars.statically_known_equals(query_shape[0], batch)
                and sizevars.statically_known_equals(query_shape[2], length)
            ),
            None,
        )
        if shape_pair is None:
            return False
        batch, length = shape_pair
        sparse_blocks = length // 128
        device = input_nodes[0].get_device()
        if getattr(device, "type", None) != "cuda":
            return False
        for index, shape in enumerate(
            (
                (batch, 64, length, 128),
                (batch, 8, length, 128),
                (batch, 8, length, 128),
                (batch, 64, length),
                (batch, 64, length),
                (batch, 64, length, 128),
                (batch, 64, length, 128),
                (batch, 8, length, 128),
                *(
                    (1, 1, sparse_blocks)
                    if index % 2 == 0
                    else (1, 1, sparse_blocks, sparse_blocks)
                    for index in range(8, 16)
                ),
            )
        ):
            tensor = input_nodes[index]
            expected_dtype = (
                torch.float32
                if index in (3, 4)
                else dtype
                if index < 8
                else torch.int32
            )
            if (
                tensor is None
                or tensor.get_dtype() != expected_dtype
                or tensor.get_device() != device
                or len(tensor.get_size()) != len(shape)
                or len(tensor.get_stride()) != len(shape)
                or not all(
                    actual is not None
                    and type(actual) not in (bool, float, str)
                    and sizevars.statically_known_equals(actual, expected)
                    for actual, expected in zip(tensor.get_size(), shape)
                )
            ):
                return False
            stride = 1
            for size, actual in zip(reversed(shape), reversed(tensor.get_stride())):
                if (
                    actual is None
                    or type(actual) in (bool, float, str)
                    or not sizevars.statically_known_equals(actual, stride)
                ):
                    return False
                stride *= size
        return True
    return False


def _use_rocm_gqa_split_dkdv(
    is_gfx950,
    query,
    key,
    value,
    kernel_options,
    score_is_trivial,
    mask_is_trivial,
    block_mask_is_default,
    has_captures,
):
    if (
        not kernel_options.get("ROCM_GQA_SPLIT_DKDV", False)
        or not is_gfx950
        or not score_is_trivial
        or not mask_is_trivial
        or not block_mask_is_default
        or has_captures
        or kernel_options["USE_TMA"]
        or kernel_options["PRESCALE_QK"]
        or kernel_options.get("num_consumer_groups", 0)
    ):
        return False
    sizevars = V.graph.sizevars
    if not all(
        tensor.get_dtype() == torch.float16
        and sizevars.statically_known_equals(tensor.get_size()[-1], 64)
        and sizevars.statically_known_equals(tensor.get_size()[0], query.get_size()[0])
        and all(sizevars.statically_known_gt(size, 0) for size in tensor.get_size()[:3])
        for tensor in (query, key, value)
    ):
        return False
    if not sizevars.statically_known_equals(
        query.get_size()[1], key.get_size()[1] * 4
    ) or not all(
        sizevars.statically_known_equals(k_size, v_size)
        for k_size, v_size in zip(key.get_size(), value.get_size())
    ):
        return False
    for tensor in (query, key, value):
        stride = 1
        for size, actual in zip(
            reversed(tensor.get_size()), reversed(tensor.get_stride())
        ):
            if not sizevars.statically_known_equals(actual, stride):
                return False
            stride *= size
    options = (
        "BLOCK_M1",
        "BLOCK_N1",
        "BLOCK_M2",
        "BLOCK_N2",
        "num_warps",
        "num_stages",
        "GQA_SHARED_HEADS",
    )
    return tuple(kernel_options[option] for option in options) == (
        32,
        128,
        128,
        32,
        4,
        2,
        4,
    )


def _sanitize_kernel_options_for_triton(
    kernel_options: dict[str, Any],
) -> tuple[dict[str, Any], _Backend]:
    """We always strip quotes around str values, we only need this in lowering, so we pop it here
    to avoid passing to triton constexpr dict
    """
    sanitized = dict(kernel_options)
    backend = cast(_Backend, sanitized.pop("BACKEND", "AUTO"))
    return sanitized, backend


def raise_flex_kernel_options_error(
    kernel_name: str,
    kernel_options: dict[str, Any],
    option_names: Sequence[str],
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> None:
    option_values = ", ".join(f"{name}={kernel_options[name]}" for name in option_names)
    raise ValueError(
        f"Invalid FlexAttention {kernel_name} kernel options: Q and KV block sizes "
        f"must be divisible by the selected tile sizes. Got "
        f"SPARSE_Q_BLOCK_SIZE={sparse_q_block_size}, "
        f"SPARSE_KV_BLOCK_SIZE={sparse_kv_block_size}, and {option_values}. "
        f"Pass compatible values with kernel_options. Available {kernel_name} "
        f"tuning options are {_flex_kernel_tuning_options(kernel_name)}. For example: "
        f"{_flex_kernel_options_example(kernel_name)}. If you did not pin "
        f"these options, and the default choice errors, compiling with "
        f"mode='max-autotune-no-cudagraphs' can also fix this by trying more "
        f"FlexAttention configs."
    )


def _check_flash_supported_scalar_captures(
    score_mod_other_buffers,
    mask_mod_other_buffers,
    *,
    backward: bool = False,
) -> None:
    if has_unsupported_cpu_scalar_tensor_captures(
        score_mod_other_buffers, mask_mod_other_buffers
    ):
        direction = " backward" if backward else ""
        raise RuntimeError(
            f"BACKEND='FLASH' but flash attention{direction} cannot be used: "
            "NYI: score_mod or mask_mod captures a 0-dim CPU tensor scalar. "
            "Workarounds: use BACKEND='TRITON' or pass the value as a tensor "
            "on device instead of capturing a CPU scalar tensor."
        )


@SymbolicGridFn
def flex_attention_grid(batch_size, q_heads, num_queries, d_model, meta, *, cdiv):
    """How is this kernel parallelized?
    We create a grid of (ceil_div(n_queries, query_block_size), batch_size, num_heads)
    Each block is responsible for iterating over blocks of keys and values calculating
    the final attention output.
    """
    return (cdiv(num_queries, meta["BLOCK_M"]), batch_size, q_heads)


def set_float32_precision(kernel_options: dict[str, Any], dtype: torch.dtype) -> None:
    precision = torch.backends.cuda.matmul.fp32_precision
    if precision == "none":
        # Unset at every level of the per-backend hierarchy; the legacy
        # default is "highest". Do not fall back to the legacy getter,
        # which throws under mixed legacy/new API state.
        precision = "ieee"
    if dtype == torch.float32 and precision == "bfx9":
        # See Note [BF16x9 precision] in torch/_inductor/utils.py.
        warning_once(
            log,
            "FP32 FlexAttention does not support bfx9 precision; using IEEE precision instead.",
        )
        kernel_options["FLOAT32_PRECISION"] = "'ieee'"
        return
    precision = (
        "ieee"
        if precision == "ieee" or torch.version.hip or torch.mtia.is_available()
        else "tf32"
    )
    kernel_options.setdefault("FLOAT32_PRECISION", repr(precision))


flex_attention_template = TritonTemplate(
    name="flex_attention",
    grid=flex_attention_grid,
    source=load_flex_template("flex_attention")
    + load_flex_template("utilities")
    + load_flex_template("common"),
    always_freeze_layout=True,
)


@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
def flex_attention(
    query,
    key,
    value,
    subgraph,
    block_mask,
    scale,
    kernel_options: dict[str, Any],
    score_mod_other_buffers,
    mask_mod_other_buffers,
):
    """The main lowering for the flex_attention hop
    This can currently lower to one of 4 templates:
    1. Base Triton Template
    2. Flex Decode Triton Template
    3. Cpu specific CPP template
    4. MPS specific Metal template
    """
    if query.get_device().type == "cpu":
        return lower_cpu(
            query,
            key,
            value,
            subgraph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
    if query.get_device().type == "mps":
        from .flex_mps import lower_mps

        return lower_mps(
            query,
            key,
            value,
            subgraph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
    # below is cuda path if device is not cpu or mps
    # tl.dot does not support embedding size less than 16
    small_dqk = V.graph.sizevars.evaluate_expr(sympy.Lt(query.get_size()[-1], 16))
    small_dv = V.graph.sizevars.evaluate_expr(sympy.Lt(value.get_size()[-1], 16))
    if small_dqk or small_dv:
        raise NotImplementedError(
            f"NYI: embedding dimension of the query, key, and value must be "
            f"at least 16 but got E={query.get_size()[-1]} and Ev={value.get_size()[-1]}"
        )

    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        _,  # dq_write_order (backward-only)
        _,  # dq_write_order_full (backward-only)
        _,  # dq_kv_order (backward-only)
        _,  # dq_kv_order_spt (backward-only)
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mask_graph,
    ) = block_mask

    kernel_options, backend = _sanitize_kernel_options_for_triton(kernel_options)
    _strip_rocm_query_tile_outer_options(kernel_options)
    for fp16_alias in (
        "ROCM_D128_FP16_COVERAGE",
        "fwd_ROCM_D128_FP16_COVERAGE",
        "bwd_ROCM_D128_FP16_COVERAGE",
    ):
        kernel_options.pop(fp16_alias, None)
    for fp32_balanced64_alias in (
        "ROCM_D128_FP32_BALANCED64_COVERAGE",
        "fwd_ROCM_D128_FP32_BALANCED64_COVERAGE",
        "bwd_ROCM_D128_FP32_BALANCED64_COVERAGE",
    ):
        kernel_options.pop(fp32_balanced64_alias, None)

    # Early check for FLASH backend: reject scalar captures that cannot be
    # represented by flash-attn aux_scalars before building subgraph buffers.
    if backend == "FLASH":
        _check_flash_supported_scalar_captures(
            score_mod_other_buffers, mask_mod_other_buffers
        )

    if backend == "FLASH":
        score_mod_other_buffers = realize_captures_for_cutedsl(score_mod_other_buffers)
        mask_mod_other_buffers = realize_captures_for_cutedsl(mask_mod_other_buffers)

    placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("score", query.get_dtype()),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    subgraph_buffer = build_subgraph_buffer(
        placeholder_inps + list(score_mod_other_buffers), subgraph
    )
    freeze_irnodes(subgraph_buffer)

    mask_graph_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    mask_graph_buffer = build_subgraph_buffer(
        mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
    )
    freeze_irnodes(mask_graph_buffer)
    # Mark symbols in custom kernel options as static shapes and add guards.
    kernel_options = {
        k: V.graph.sizevars.guard_int(v) if isinstance(v, sympy.Symbol) else v
        for k, v in kernel_options.items()
    }
    set_float32_precision(kernel_options, query.get_dtype())
    enable_gqa = V.graph.sizevars.evaluate_expr(
        sympy.Ne(query.get_size()[1], key.get_size()[1]),
    )

    can_use_decode = _use_flex_decoding(
        query, kv_indices, value, kernel_options, enable_gqa
    )
    use_decode = (backend == "TRITON_DECODE") or (backend == "AUTO" and can_use_decode)

    if backend == "TRITON_DECODE" and not can_use_decode:
        raise RuntimeError(
            "BACKEND='TRITON_DECODE' was specified but flex_decoding cannot be used for this input. "
            "flex_decoding is only available for short sequence lengths with specific configurations."
        )

    if use_decode:
        return create_flex_decoding_kernel(
            query,
            key,
            value,
            block_mask,
            scale,
            kernel_options,
            subgraph_buffer,
            mask_graph_buffer,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )

    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ]
    )

    if _use_flex_flash_attention(
        subgraph,
        mask_graph,
        kernel_options,
        num_score_mod_placeholders=len(placeholder_inps),
        backend=backend,
    ):
        return create_flex_flash_attention_kernel(
            query,
            key,
            value,
            block_mask,
            scale,
            kernel_options,
            subgraph_buffer,
            mask_graph_buffer,
            score_mod_other_buffers,
            mask_mod_other_buffers,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
            mask_graph=mask_graph,
            subgraph=subgraph,
        )

    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    freeze_irnodes(score_mod_other_buffers)
    freeze_irnodes(mask_mod_other_buffers)

    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()
    if not V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)):
        raise AssertionError(
            f"Bq and Bkv must be broadcastable. Got Bq={Bq} and Bkv={Bkv}"
        )
    if not V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_q, 0)):
        raise AssertionError("Query length must be greater than 0")
    if not V.graph.sizevars.evaluate_expr(sympy.Gt(seq_len_kv, 0)):
        raise AssertionError("Key length must be greater than 0")

    B = Bq

    seq_q_divisible = can_skip_boundary_checks(seq_len_q, SPARSE_Q_BLOCK_SIZE)
    seq_kv_divisible = can_skip_boundary_checks(seq_len_kv, SPARSE_KV_BLOCK_SIZE)
    if seq_q_divisible and seq_kv_divisible:
        kernel_options.setdefault("IS_DIVISIBLE", True)
    else:
        kernel_options.setdefault("IS_DIVISIBLE", False)

    # NB it is okay that the v_head_dim is different
    # We are using these to match fill order of the output.
    q_strides = query.get_stride()
    # Construct output layout with strides matching the query.
    out_size = [B, Hq, seq_len_q, v_head_dim]
    out_strides = infer_dense_strides(out_size, q_strides)

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        [B, Hq, seq_len_q, v_head_dim],
        stride=[sympy.sympify(s) for s in out_strides],
    )
    # see NOTE:[TritonTemplates with multiple outputs]
    logsumexp_shape = [B, Hq, seq_len_q]
    logsumexp = empty_strided(
        logsumexp_shape,
        None,
        dtype=torch.float32,  # The logsumexp is always stored in fp32 regardless of the input dtype
        device=query.get_device(),
    )
    max_scores = empty_strided(
        logsumexp_shape,  # Same shape as logsumexp
        None,
        dtype=torch.float32,  # The max scores are always stored in fp32 regardless of the input dtype
        device=query.get_device(),
    )
    kernel_options.setdefault("SM_SCALE", scale)

    # Determine GQA broadcast factor.
    gqa_shared_heads = FloorDiv(Hq, Hkv)
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

    # Inside of Triton kernel, only apply partial masking if partial blocks are computed.
    # full_kv_num_blocks is None if partial blocks are not computed
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        full_kv_num_blocks, full_kv_indices = (
            empty(0, device=query.get_device()) for _ in range(2)
        )

    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

    choices: list[Any] = []

    dtype = query.get_dtype()
    head_dim = V.graph.sizevars.guard_int(query.get_size()[-1])
    configs: list[FlexConfig] = V.choices.get_flex_attention_fwd_configs(
        head_dim, seq_len_q, dtype, query.get_device().type
    )

    # Mark SPARSE_KV_BLOCK_SIZE & SPARSE_Q_BLOCK_SIZE as static shapes and add guards.
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)
    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)

    original_kernel_options = kernel_options.copy()
    # Default config for warp specialization
    num_consumer_groups, num_buffers_warp_spec = 0, 0
    invalid_block_options: dict[str, Any] | None = None

    for conf in configs:
        cur_kernel_options = original_kernel_options.copy()
        # Performance tuning
        # Triton parameters
        # Remove prefix for forward kernels options and delete backward kernel options.
        for k in list(cur_kernel_options.keys()):
            if k.startswith("fwd_"):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith("bwd_"):
                cur_kernel_options.pop(k)
        cur_kernel_options["ROCM_D128_DV_VIEW_CARRY_SPLIT64"] = False
        cur_kernel_options.setdefault("num_stages", conf.num_stages)
        cur_kernel_options.setdefault("num_warps", conf.num_warps)
        if cur_kernel_options.get("num_consumer_groups", False):
            cur_kernel_options.setdefault("num_consumer_groups", num_consumer_groups)
            cur_kernel_options.setdefault(
                "num_buffers_warp_spec", num_buffers_warp_spec
            )

        # Shrink default tiles to fit smaller pow2 sparse block sizes;
        # user-pinned tiles and non-pow2 sparse sizes still error out below.
        block_m, block_n = conf.block_m, conf.block_n
        if len(configs) == 1 and all(
            is_power_of_2(s) and s >= 16
            for s in (SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE)
        ):
            block_m = min(block_m, SPARSE_Q_BLOCK_SIZE)
            block_n = min(block_n, SPARSE_KV_BLOCK_SIZE)
        cur_kernel_options.setdefault("BLOCK_M", block_m)
        cur_kernel_options.setdefault("BLOCK_N", block_n)

        # Blocksparse options
        cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)

        if (
            cur_kernel_options["SPARSE_KV_BLOCK_SIZE"] % cur_kernel_options["BLOCK_N"]
            != 0
            or cur_kernel_options["SPARSE_Q_BLOCK_SIZE"] % cur_kernel_options["BLOCK_M"]
            != 0
        ):
            invalid_block_options = cur_kernel_options
            if len(configs) == 1:
                raise_flex_kernel_options_error(
                    "forward",
                    cur_kernel_options,
                    ("BLOCK_M", "BLOCK_N"),
                    SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE,
                )
            continue

        # Descriptor selection runs after the tile rejection above so a config
        # that is about to be discarded cannot install descriptor range bounds.
        #
        # A default of True means "absent": omission keeps automatic selection,
        # while any explicit falsy value (False, 0, None) forces pointer loads.
        tdm_requested = bool(cur_kernel_options.get("USE_TMA", True))

        # ROCm reports device type "cuda", so route it exclusively. The generic
        # probe excludes HIP only in its CUDA arm, so it can read true on a ROCm
        # host and would then enable descriptors under NVIDIA's rules, skipping
        # the ROCm floor, the gfx1250 probe and the operand policy.
        if torch.version.hip is not None and query.get_device().type == "cuda":
            cur_kernel_options["USE_TMA"] = tdm_requested and use_flex_tdm_descriptor(
                query,
                key,
                value,
                block_shapes=[
                    (
                        cur_kernel_options["BLOCK_M"],
                        cur_kernel_options["QK_HEAD_DIM_ROUNDED"],
                    ),
                    (
                        cur_kernel_options["BLOCK_N"],
                        cur_kernel_options["QK_HEAD_DIM_ROUNDED"],
                    ),
                    (
                        cur_kernel_options["BLOCK_N"],
                        cur_kernel_options["V_HEAD_DIM_ROUNDED"],
                    ),
                ],
            )
        else:
            # Intel GPU enables TMA by default
            cur_kernel_options.setdefault("USE_TMA", bool(torch.xpu.is_available()))

            if cur_kernel_options["USE_TMA"] and not can_use_tma(query, key, value):
                cur_kernel_options["USE_TMA"] = False

        # ROCm specific kernargs
        for attrib in ["kpack", "matrix_instr_nonkdim", "waves_per_eu"]:
            if hasattr(conf, attrib):
                cur_kernel_options[attrib] = getattr(conf, attrib)

        error = flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                logsumexp,
                max_scores,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
            ],
            layout=layout,
            subgraphs=[
                subgraph_buffer,
                mask_graph_buffer,
            ],
            mutated_inputs=[
                logsumexp,
                max_scores,
            ],
            call_sizes=query.get_size(),
            **cur_kernel_options,
        )
        if error is not None and len(configs) == 1:
            raise error

    # Let the active choices handler append any backend-specific flex-attention
    # template choices (e.g. TLX on Blackwell in fbcode). No-op by default.
    choices = V.choices.append_flex_attention_choices(
        choices,
        configs,
        [
            query,
            key,
            value,
            logsumexp,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ],
        [subgraph_buffer, mask_graph_buffer],
        layout,
        original_kernel_options,
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
    )

    if not choices and invalid_block_options is not None:
        raise_flex_kernel_options_error(
            "forward",
            invalid_block_options,
            ("BLOCK_M", "BLOCK_N"),
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
        )

    input_gen_fns = {
        5: create_num_blocks_fake_generator(kv_indices),
        6: create_indices_fake,
        7: create_num_blocks_fake_generator(full_kv_indices),
        8: create_indices_fake,
    }

    out, _ = autotune_select_algorithm(
        "flex_attention",
        choices,
        # Use generated inputs because codegen can inline capture producers.
        list(choices[0].input_nodes) if choices else [],
        layout,
        input_gen_fns=input_gen_fns,
    )

    # need subgraph inputs and outputs to analyze all symints used in flex attention
    out.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    out.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        subgraph_buffer, mask_graph_buffer
    )

    return (out, logsumexp, max_scores)


# ---------------------------- Backward HOP Implementation ----------------------------


@SymbolicGridFn
def flex_attention_backward_grid(
    batch_size, q_heads, num_queries, d_model, kv_heads, num_key_value, meta, *, cdiv
):
    """How is this kernel parallelized?
    We create a grid of (ceil_div(n_queries, query_block_size) * heads_ratio + ceil_div(n_kv, kv_block_size), batch_size, kv_heads)
    Currently this is only parallelizing over batch* kv_heads, but we can, and want to
    parallelize over ceil_div(q_heads//kv_heads * num_key_value, key_value_block_size).
    To do this will either require atomic updates to some grad values or to have a two pass kernel design.
    """
    query_blocks = cdiv(num_queries, meta["BLOCK_M2"]) * (q_heads // kv_heads)
    kv_blocks = cdiv(num_key_value, meta["BLOCK_N1"])
    phase = meta.get("ROCM_BWD_PHASE", 0)
    if phase == 1:
        return kv_blocks, batch_size, kv_heads * meta.get("ROCM_GQA_SPLITS", 1)
    if phase == 2:
        return query_blocks, batch_size, kv_heads
    blocks = query_blocks + kv_blocks
    return blocks, batch_size, kv_heads


flex_attention_backward_template = TritonTemplate(
    name="flex_attention_backward",
    grid=flex_attention_backward_grid,
    source=load_flex_template("flex_backwards") + load_flex_template("utilities"),
    always_freeze_layout=True,
)


def _fuse_nested_index_backward(joint_graph: torch.fx.GraphModule) -> None:
    r"""Fuse the backward of table[i][j] into one scatter for table[i, j]."""
    from torch._inductor.pattern_matcher import (
        CallFunction,
        KeywordArg,
        Match,
        PatternMatcherPass,
        register_graph_pattern,
    )

    scatter = torch.ops.flex_lib.zeros_and_scatter.default
    patterns = PatternMatcherPass(pass_name="flex_attention_index_backward")

    def same_shape(match: Match) -> bool:
        args = match.kwargs
        shapes = (args["shape"][len(args["indices"]) :], args["inner_shape"])
        remaining, inner = torch.fx.map_arg(shapes, lambda n: n.meta["val"])
        return statically_known_true(sym_eq(remaining, inner))

    @register_graph_pattern(
        CallFunction(
            scatter,
            KeywordArg("shape"),
            KeywordArg("indices"),
            CallFunction(
                scatter,
                KeywordArg("inner_shape"),
                KeywordArg("inner_indices"),
                KeywordArg("value"),
            ),
        ),
        extra_check=same_shape,
        # pyrefly: ignore [bad-argument-type]
        pass_dict=patterns,
    )
    def fuse(match: Match, shape, indices, inner_shape, inner_indices, value):
        match.output_node().args = (shape, [*indices, *inner_indices], value)
        match.erase_nodes()

    # Matching runs backward through the graph; repeat to collapse longer chains.
    changed = False
    while patterns.apply(joint_graph):
        changed = True
    if changed:
        joint_graph.graph.lint()
        joint_graph.recompile()


def validate_joint_graph(joint_graph: torch.fx.Graph):
    """We do some pre lowering graph checks in order to raise nicer error messages"""
    for node in joint_graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.flex_lib.zeros_and_scatter.default
        ):
            for user in node.users:
                if user.op != "output":
                    raise NotImplementedError(
                        "Using multiple indexing operations on the same tensor that requires gradients "
                        "in a score_mod function is not currently supported. "
                        "This typically happens when indexing the same tensor multiple times, like:\n\n"
                        "    def score_mod(score, b, h, q_idx, kv_idx):\n"
                        "        return score + bias[q_idx] + bias[kv_idx]  # bias used twice!\n\n"
                        "A valid workaround is to clone() the tensors that will be indexed multiple times. For example:\n\n"
                        "    bias1 = bias.clone()\n"
                        "    def score_mod(score, b, h, q_idx, kv_idx):\n"
                        "        return score + bias[q_idx] + bias1[kv_idx]\n\n"
                        "Note that this solution will use additional memory.\n\n"
                        "For chained indexing with operations in between, such as "
                        "(table[idx] * 2)[0], apply all indices in one indexing "
                        "expression instead: table[idx, 0] * 2."
                    )
    return


@dataclass(frozen=True)
class JointOutputResult:
    """Results from processing joint outputs."""

    grad_input: ComputedBuffer
    captured_grads_compute: list[ComputedBuffer]
    captured_grads: list[TensorBox | None]
    mutated_grads: list[TensorBox]


def process_joint_outputs(
    all_joint_outputs: SubgraphResults, num_placeholders: int
) -> JointOutputResult:
    """Process joint outputs and extract various buffers needed for lowering

    Args:
        all_joint_outputs: List of all the outputs from build_subgraphs
        num_placeholders: The number of placeholder inputs, used to skip over unused backward compute buffers

    Returns:
        JointOutputResult containing processed buffers and gradients
    """
    if not isinstance(all_joint_outputs, list):
        raise AssertionError(f"Expected list, got {type(all_joint_outputs)}")
    if all_joint_outputs[0] is None:
        raise AssertionError("joint_subgraph_buffer is None - this is a bug!")

    joint_buffer = all_joint_outputs[0]
    other_grads = all_joint_outputs[num_placeholders - 1 :]

    # outer_grads has the structure: Len(other_buffer_grads) if buffer doesn't require grad than it will be None
    # We only grab the buffers that require grad for inlining into kernel
    grads_compute = [buf for buf in other_grads if buf is not None]

    def get_out(buf):
        if buf is None:
            return None
        if not isinstance(buf, ComputedBuffer):
            raise AssertionError(f"Expected ComputedBuffer, got {type(buf)}")
        if buf.name is None:
            raise AssertionError("ComputedBuffer name must not be None")
        return TensorBox.create(V.graph.get_buffer(buf.name))

    grads_out = [get_out(x) for x in other_grads]
    mutated_grads = [buf for buf in grads_out if buf is not None]

    return JointOutputResult(
        grad_input=joint_buffer,
        captured_grads_compute=grads_compute,
        captured_grads=grads_out,
        mutated_grads=mutated_grads,
    )


# TODO: We probably also need a layout constraint?
@register_lowering(
    torch.ops.higher_order.flex_attention_backward, type_promotion_kind=None
)
def flex_attention_backward(*args, **kwargs):
    """Lowering for the flex_attention_backward op in triton"""
    (
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        joint_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    ) = args
    if query.get_device().type == "mps":
        raise NotImplementedError(
            "flex_attention backward is not yet supported on MPS. "
            "Use torch.no_grad() for inference."
        )
    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        dq_write_order,
        dq_write_order_full,
        dq_kv_order,
        dq_kv_order_spt,
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mask_graph,
    ) = block_mask

    (
        query,
        key,
        value,
        logsumexp,
        grad_out,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
        dq_write_order,
        dq_write_order_full,
        dq_kv_order,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            logsumexp,
            grad_out,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
            dq_write_order,
            dq_write_order_full,
            dq_kv_order,
        ]
    )

    device = query.get_device()
    dtype = query.get_dtype()
    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()

    if not V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)):
        raise AssertionError(
            f"Bq and Bkv must be broadcastable. Got Bq={Bq} and Bkv={Bkv}"
        )

    kernel_options, backend = _sanitize_kernel_options_for_triton(kernel_options)
    user_kernel_options = kernel_options.copy()
    rocm_requests = _normalize_rocm_backward_options(kernel_options)
    if backend == "FLASH":
        _check_flash_supported_scalar_captures(
            score_mod_other_buffers, mask_mod_other_buffers, backward=True
        )
        score_mod_other_buffers = realize_captures_for_cutedsl(score_mod_other_buffers)
        mask_mod_other_buffers = realize_captures_for_cutedsl(mask_mod_other_buffers)

    # Add check for mixed dtypes
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Backward pass with mixed query, key, and value dtype is not supported, "
            f"got query.dtype={query.dtype}, key.dtype={key.dtype}, "
            f"and value.dtype={value.dtype}"
        )
    # Mark symbols in custom kernel options as static shapes and add guards.
    kernel_options = {
        k: V.graph.sizevars.guard_int(v) if isinstance(v, sympy.Symbol) else v
        for k, v in kernel_options.items()
    }
    set_float32_precision(kernel_options, query.get_dtype())
    kernel_options.setdefault("PRESCALE_QK", False)
    kernel_options.setdefault("ROWS_GUARANTEED_SAFE", False)
    kernel_options.setdefault("BLOCKS_ARE_CONTIGUOUS", False)
    kernel_options.setdefault("WRITE_DQ", True)
    seq_q_divisible = can_skip_boundary_checks(seq_len_q, SPARSE_Q_BLOCK_SIZE)
    seq_kv_divisible = can_skip_boundary_checks(seq_len_kv, SPARSE_KV_BLOCK_SIZE)
    if seq_q_divisible and seq_kv_divisible:
        kernel_options.setdefault("IS_DIVISIBLE", True)
    else:
        kernel_options.setdefault("IS_DIVISIBLE", False)

    fwd_placeholder_inps = [
        create_placeholder(name, dtype, device)
        for name, dtype in [
            ("score", dtype),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    fw_subgraph_buffer = build_subgraph_buffer(
        fwd_placeholder_inps + list(score_mod_other_buffers), fw_graph
    )
    freeze_irnodes(fw_subgraph_buffer)

    joint_placeholder_inps = fwd_placeholder_inps + [
        create_placeholder("grad_score_mod", dtype, device)
    ]
    # Sometimes we have weird unused nodes here
    joint_graph.graph_module.graph.eliminate_dead_code()
    _fuse_nested_index_backward(joint_graph.graph_module)

    # It is hard to raise nice errors for some joint graphs during subgraph lowering
    # This lets us do some checks before attempting to lower
    validate_joint_graph(joint_graph.graph_module.graph)

    all_joint_outputs = build_subgraph_buffer(
        joint_placeholder_inps + list(score_mod_other_buffers),
        joint_graph,
    )

    freeze_irnodes(all_joint_outputs)

    joint_outputs = process_joint_outputs(
        all_joint_outputs, len(joint_placeholder_inps)
    )

    mask_graph_placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    mask_graph_buffer = build_subgraph_buffer(
        mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
    )
    freeze_irnodes(mask_graph_buffer)

    if _use_flex_flash_attention_backward(
        fw_graph,
        mask_graph,
        backend=backend,
        joint_outputs=joint_outputs,
        score_mod_other_buffers=score_mod_other_buffers,
    ):
        needs_block_mask = not is_trivial_mask_graph(mask_graph.graph_module)
        if grad_logsumexp is not None:
            (grad_logsumexp,) = maybe_realize([grad_logsumexp])

        score_is_trivial = is_trivial_score_graph(fw_graph.graph_module)
        return create_flex_flash_attention_backward_kernel(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            scale,
            kernel_options,
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
            fw_subgraph_buffer=None if score_is_trivial else fw_subgraph_buffer,
            joint_subgraph_buffer=None
            if score_is_trivial
            else joint_outputs.grad_input,
            score_mod_other_buffers=list(score_mod_other_buffers),
            mask_graph_buffer=mask_graph_buffer if needs_block_mask else None,
            mask_mod_other_buffers=list(mask_mod_other_buffers),
            q_num_blocks=q_num_blocks if needs_block_mask else None,
            q_indices=q_indices if needs_block_mask else None,
            full_q_num_blocks=full_q_num_blocks if needs_block_mask else None,
            full_q_indices=full_q_indices if needs_block_mask else None,
            dq_write_order=dq_write_order if needs_block_mask else None,
            dq_write_order_full=dq_write_order_full if needs_block_mask else None,
            dq_kv_order=dq_kv_order if needs_block_mask else None,
            dq_kv_order_spt=dq_kv_order_spt,
        )

    # Construct layout with stride order matching K
    key_size = [Bq, Hkv, seq_len_kv, qk_head_dim]
    key_strides = infer_dense_strides(key_size, key.get_stride())

    layout_broadcasted_k = FixedLayout(
        key.get_device(),
        key.get_dtype(),
        key_size,
        stride=[sympy.sympify(s) for s in key_strides],
    )

    # Create delta which will is needed for the bwd's kernel
    mul_delta = lowerings[aten.mul](out, grad_out)
    delta = lowerings[aten.sum](mul_delta, axis=-1)
    delta = lowerings[prims.convert_element_type](delta, torch.float32)
    if grad_logsumexp is not None:
        grad_lse_exp2 = lowerings[aten.mul](grad_logsumexp, 1 / math.log(2))
        grad_lse_exp2 = ExternKernel.require_contiguous(grad_lse_exp2)
        delta = lowerings[aten.sub](delta, grad_lse_exp2)
        delta = ExternKernel.require_contiguous(delta)
        delta, grad_lse_exp2 = maybe_realize([delta, grad_lse_exp2])
    else:
        delta = ExternKernel.require_contiguous(delta)
        (delta,) = maybe_realize([delta])

    # # see NOTE:[TritonTemplates with multiple outputs]
    query_size = [Bq, Hq, seq_len_q, qk_head_dim]
    grad_query_strides = infer_dense_strides(query_size, query.get_stride())
    grad_query = empty_strided(
        query_size,
        stride=[sympy.sympify(s) for s in grad_query_strides],
        dtype=query.get_dtype(),
        device=query.get_device(),
    )

    # Construct output layout with stride order matching value
    value_size = [Bq, Hkv, seq_len_kv, v_head_dim]
    value_strides = infer_dense_strides(value_size, value.get_stride())

    broadcasted_grad_value = empty_strided(
        value_size,
        stride=[sympy.sympify(s) for s in value_strides],
        dtype=value.get_dtype(),
        device=value.get_device(),
    )

    kernel_options.setdefault("SM_SCALE", scale)

    # Determine GQA factor
    gqa_shared_heads = FloorDiv(Hq, Hkv)
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)

    # Inside of Triton kernel, only apply partial masking if partial blocks are computed.
    # full_kv_num_blocks is torch.zeros([1, 1, 1]) if partial blocks are not computed.
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    if not has_full_blocks:
        full_kv_num_blocks, full_kv_indices, full_q_num_blocks, full_q_indices = (
            empty(0, device=query.get_device()) for _ in range(4)
        )

    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)

    choices: list[Any] = []

    dkdv_choices: list[Any] = []
    dq_choices: list[Any] = []
    dkdv_input_nodes = None
    dq_input_nodes = None
    partial_grad_value = None
    dkdv_options = None
    split_dkdv = False
    dkdv_layout = layout_broadcasted_k
    layout_grad_query = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        query_size,
        stride=[sympy.sympify(s) for s in grad_query_strides],
    )

    dtype = query.get_dtype()
    head_dim = V.graph.sizevars.guard_int(query.get_size()[-1])
    configs: list[FlexBwDConfig] = V.choices.get_flex_attention_bwd_configs(
        head_dim, dtype, query.get_device().type
    )

    # Default config for warp specialization
    num_consumer_groups, num_buffers_warp_spec = 0, 0
    invalid_block_options: dict[str, Any] | None = None

    original_kernel_options = kernel_options.copy()
    bwd_input_nodes = [
        query,
        key,
        value,
        logsumexp,
        delta,
        grad_out,
        grad_query,
        broadcasted_grad_value,
        kv_num_blocks,
        kv_indices,
        q_num_blocks,
        q_indices,
        full_kv_num_blocks,
        full_kv_indices,
        full_q_num_blocks,
        full_q_indices,
    ]
    bwd_subgraphs = [
        fw_subgraph_buffer,
        joint_outputs.grad_input,
        mask_graph_buffer,
        joint_outputs.captured_grads_compute,
    ]
    bwd_mutated_inputs = [
        grad_query,
        broadcasted_grad_value,
        *joint_outputs.mutated_grads,
    ]

    is_gfx950 = (
        bool(torch.version.hip)
        and query.get_device().type == "cuda"
        and torch.cuda.get_device_properties(query.get_device()).gcnArchName.split(
            ":", 1
        )[0]
        == "gfx950"
    )
    for conf in configs:
        # Performance tuning
        # Triton heuristics
        cur_kernel_options = _get_rocm_backward_config_options(
            original_kernel_options, is_gfx950=is_gfx950
        )
        cur_kernel_options.setdefault("num_warps", conf.num_warps)
        cur_kernel_options.setdefault("num_stages", conf.num_stages)

        if cur_kernel_options.get("num_consumer_groups", False):
            cur_kernel_options.setdefault("num_consumer_groups", num_consumer_groups)
            cur_kernel_options.setdefault(
                "num_buffers_warp_spec", num_buffers_warp_spec
            )

        # Intel GPU enables TMA by default
        cur_kernel_options.setdefault("USE_TMA", bool(torch.xpu.is_available()))

        if cur_kernel_options["USE_TMA"] and not can_use_tma(query, key, value):
            cur_kernel_options["USE_TMA"] = False

        cur_kernel_options["ROCM_D256_PANEL_DOTS"] = _use_rocm_d256_panel_dots(
            is_gfx950,
            dtype,
            head_dim,
            V.graph.sizevars.guard_int(v_head_dim),
            cur_kernel_options,
        )

        # Shrink default tiles to fit smaller pow2 sparse block sizes;
        # user-pinned tiles and non-pow2 sparse sizes still error out below.
        block_m1, block_n1 = conf.block_m1, conf.block_n1
        block_m2, block_n2 = conf.block_m2, conf.block_n2
        if len(configs) == 1 and all(
            is_power_of_2(s) and s >= 16
            for s in (SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE)
        ):
            block_m1 = min(block_m1, SPARSE_Q_BLOCK_SIZE)
            block_n1 = min(block_n1, SPARSE_KV_BLOCK_SIZE)
            block_m2 = min(block_m2, SPARSE_Q_BLOCK_SIZE)
            block_n2 = min(block_n2, SPARSE_KV_BLOCK_SIZE)
            # Kernel static asserts: BLOCK_N1 % BLOCK_M1 == BLOCK_M2 % BLOCK_N2 == 0
            block_m1 = min(block_m1, block_n1)
            block_n2 = min(block_n2, block_m2)
        cur_kernel_options.setdefault("BLOCK_M1", block_m1)
        cur_kernel_options.setdefault("BLOCK_N1", block_n1)
        cur_kernel_options.setdefault("BLOCK_M2", block_m2)
        cur_kernel_options.setdefault("BLOCK_N2", block_n2)

        # Blocksparse options
        cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)

        if (
            cur_kernel_options["SPARSE_KV_BLOCK_SIZE"] % cur_kernel_options["BLOCK_N1"]
            != 0
            or cur_kernel_options["SPARSE_Q_BLOCK_SIZE"]
            % cur_kernel_options["BLOCK_M1"]
            != 0
            or cur_kernel_options["SPARSE_KV_BLOCK_SIZE"]
            % cur_kernel_options["BLOCK_N2"]
            != 0
            or cur_kernel_options["SPARSE_Q_BLOCK_SIZE"]
            % cur_kernel_options["BLOCK_M2"]
            != 0
            # Kernel static asserts; validate here for a friendly error.
            or cur_kernel_options["BLOCK_N1"] % cur_kernel_options["BLOCK_M1"] != 0
            or cur_kernel_options["BLOCK_M2"] % cur_kernel_options["BLOCK_N2"] != 0
        ):
            invalid_block_options = cur_kernel_options
            if len(configs) == 1:
                raise_flex_kernel_options_error(
                    "backward",
                    cur_kernel_options,
                    ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"),
                    SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE,
                )
            continue

        # ROCm specific kernargs
        for attrib in ["kpack", "matrix_instr_nonkdim", "waves_per_eu"]:
            if hasattr(conf, attrib):
                if (
                    (
                        rocm_requests.d192_panel_phase_requested is True
                        and head_dim == 192
                        or rocm_requests.d96_panel_phase_requested is True
                        and head_dim == 96
                    )
                    and is_gfx950
                    and dtype == torch.bfloat16
                ):
                    cur_kernel_options.setdefault(attrib, getattr(conf, attrib))
                else:
                    cur_kernel_options[attrib] = getattr(conf, attrib)

        cur_kernel_options["ROCM_XCD_SNAKE"] = len(
            configs
        ) == 1 and _use_rocm_xcd_snake(
            is_gfx950,
            query,
            key,
            value,
            cur_kernel_options,
            is_trivial_score_graph(fw_graph.graph_module),
            _is_rocm_simple_causal_mask_graph(mask_graph.graph_module),
            bool(score_mod_other_buffers or mask_mod_other_buffers),
        )

        flex_attention_backward_template.maybe_append_choice(
            choices=choices,
            input_nodes=bwd_input_nodes,
            layout=layout_broadcasted_k,  # We use store_output only for grad_key
            subgraphs=bwd_subgraphs,
            mutated_inputs=bwd_mutated_inputs,
            call_sizes=query.get_size() + key.get_size()[1:3],
            **cur_kernel_options,
        )

        auto_options = _get_rocm_backward_auto_profile(
            user_kernel_options,
            cur_kernel_options,
            bwd_input_nodes,
            out,
            backend=backend,
            is_gfx950=is_gfx950,
            single_config=len(configs) == 1,
            score_is_trivial=is_trivial_score_graph(fw_graph.graph_module),
            has_captures=bool(score_mod_other_buffers or mask_mod_other_buffers),
            mask_is_causal=_is_rocm_simple_causal_mask_graph(mask_graph.graph_module),
            mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(
                mask_graph.graph_module
            ),
            block_mask_is_default=_is_rocm_default_block_mask(block_mask, query, key),
            grad_logsumexp_is_none=grad_logsumexp is None,
        )
        active_rocm_requests = rocm_requests
        if auto_options is not None:
            active_rocm_requests = _normalize_rocm_backward_options(auto_options)
            cur_kernel_options = _get_rocm_backward_config_options(
                auto_options, is_gfx950=is_gfx950
            )

        fp32_balanced64_pilot = len(
            configs
        ) == 1 and _use_rocm_d128_fp32_balanced64_coverage(
            is_gfx950,
            query,
            key,
            value,
            grad_out,
            out,
            logsumexp,
            cur_kernel_options,
            is_trivial_score_graph(fw_graph.graph_module),
            bool(score_mod_other_buffers or mask_mod_other_buffers),
            requested=active_rocm_requests.fp32_balanced64_coverage_requested,
            other_requests=active_rocm_requests.fp32_balanced64_coverage_other_requests,
            mask_is_causal=_is_rocm_simple_causal_mask_graph(mask_graph.graph_module),
            grad_logsumexp_is_none=grad_logsumexp is None,
        )
        fp16_pilot = len(configs) == 1 and _use_rocm_d128_fp16_coverage(
            is_gfx950,
            query,
            key,
            value,
            grad_out,
            out,
            logsumexp,
            cur_kernel_options,
            is_trivial_score_graph(fw_graph.graph_module),
            bool(score_mod_other_buffers or mask_mod_other_buffers),
            requested=active_rocm_requests.fp16_coverage_requested,
            other_requests=active_rocm_requests.fp16_coverage_other_requests,
            mask_is_causal=_is_rocm_simple_causal_mask_graph(mask_graph.graph_module),
            grad_logsumexp_is_none=grad_logsumexp is None,
        )
        use_separate_dq = len(configs) == 1 and _use_rocm_separate_dq(
            is_gfx950,
            query,
            key,
            value,
            cur_kernel_options,
            is_trivial_score_graph(fw_graph.graph_module),
            bool(score_mod_other_buffers or mask_mod_other_buffers),
            mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(
                mask_graph.graph_module
            ),
            block_mask_is_default=_is_rocm_default_block_mask(block_mask, query, key),
            fp16_pilot=fp16_pilot,
            fp32_balanced64_pilot=fp32_balanced64_pilot,
        )
        use_d192_panel_phase = len(configs) == 1 and _use_rocm_d192_panel_phase(
            is_gfx950,
            query,
            key,
            value,
            cur_kernel_options,
            is_trivial_score_graph(fw_graph.graph_module),
            bool(score_mod_other_buffers or mask_mod_other_buffers),
            requested=active_rocm_requests.d192_panel_phase_requested,
            mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(
                mask_graph.graph_module
            ),
            block_mask_is_default=_is_rocm_default_block_mask(block_mask, query, key),
            grad_logsumexp_is_none=grad_logsumexp is None,
            grad_out=grad_out,
        )
        use_separate_dq = use_separate_dq or use_d192_panel_phase
        use_d96_panel_phase = len(configs) == 1 and _use_rocm_d96_panel_phase(
            is_gfx950,
            query,
            key,
            value,
            cur_kernel_options,
            is_trivial_score_graph(fw_graph.graph_module),
            bool(score_mod_other_buffers or mask_mod_other_buffers),
            requested=active_rocm_requests.d96_panel_phase_requested
            if active_rocm_requests.d192_panel_phase_requested is not True
            else False,
            mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(
                mask_graph.graph_module
            ),
            block_mask_is_default=_is_rocm_default_block_mask(block_mask, query, key),
            grad_logsumexp_is_none=grad_logsumexp is None,
            grad_out=grad_out,
        )
        use_separate_dq = use_separate_dq or use_d96_panel_phase
        use_gqa_split = len(configs) == 1 and _use_rocm_gqa_split_dkdv(
            is_gfx950,
            query,
            key,
            value,
            cur_kernel_options,
            is_trivial_score_graph(fw_graph.graph_module),
            is_trivial_mask_graph(mask_graph.graph_module),
            _is_rocm_default_block_mask(block_mask, query, key),
            bool(score_mod_other_buffers or mask_mod_other_buffers),
        )
        if use_separate_dq or use_gqa_split:
            dkdv_input_nodes = list(bwd_input_nodes)
            # The unused gradient inputs provide strides without extra allocations.
            dkdv_input_nodes[6] = query
            split_dkdv = use_gqa_split
            dkdv_options, dq_options = _get_rocm_backward_phase_options(
                cur_kernel_options,
                dkdv_input_nodes,
                requests=active_rocm_requests,
                single_config=len(configs) == 1,
                is_gfx950=is_gfx950,
                query=query,
                key=key,
                value=value,
                grad_out=grad_out,
                use_separate_dq=use_separate_dq,
                use_d192_panel_phase=use_d192_panel_phase,
                use_d96_panel_phase=use_d96_panel_phase,
                fp16_pilot=fp16_pilot,
                fp32_balanced64_pilot=fp32_balanced64_pilot,
                score_is_trivial=is_trivial_score_graph(fw_graph.graph_module),
                has_captures=bool(score_mod_other_buffers or mask_mod_other_buffers),
                mask_is_causal=_is_rocm_simple_causal_mask_graph(
                    mask_graph.graph_module
                ),
                mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(
                    mask_graph.graph_module
                ),
                block_mask_is_default=_is_rocm_default_block_mask(
                    block_mask, query, key
                ),
                grad_logsumexp_is_none=grad_logsumexp is None,
                is_automatic=auto_options is not None,
            )
            if split_dkdv:
                gqa_splits = (
                    4
                    if auto_options is not None
                    and dtype == torch.float16
                    and head_dim == 64
                    else 2
                )
                partial_size = [Bq, Hkv * gqa_splits, seq_len_kv, qk_head_dim]
                partial_strides = infer_dense_strides(partial_size, key.get_stride())
                dkdv_layout = FixedLayout(
                    key.get_device(),
                    torch.float32,
                    partial_size,
                    stride=partial_strides,
                )
                partial_grad_value = empty_strided(
                    partial_size,
                    stride=partial_strides,
                    dtype=torch.float32,
                    device=value.get_device(),
                )
                dkdv_input_nodes[7] = partial_grad_value
                dkdv_options["ROCM_GQA_SPLITS"] = gqa_splits
            flex_attention_backward_template.maybe_append_choice(
                choices=dkdv_choices,
                input_nodes=dkdv_input_nodes,
                layout=dkdv_layout,
                subgraphs=bwd_subgraphs,
                mutated_inputs=[dkdv_input_nodes[7]],
                call_sizes=query.get_size() + key.get_size()[1:3],
                **dkdv_options,
            )
            dq_input_nodes = list(dkdv_input_nodes)
            dq_input_nodes[7] = value
            flex_attention_backward_template.maybe_append_choice(
                choices=dq_choices,
                input_nodes=dq_input_nodes,
                layout=layout_grad_query,
                subgraphs=bwd_subgraphs,
                mutated_inputs=[],
                call_sizes=query.get_size() + key.get_size()[1:3],
                **dq_options,
            )

    choices, separate_dq = _append_flex_attention_backward_choices(
        choices,
        configs,
        bwd_input_nodes,
        bwd_subgraphs,
        layout_broadcasted_k,
        original_kernel_options,
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        mutated_inputs=bwd_mutated_inputs,
        split_available=bool(dkdv_choices and dq_choices),
    )

    if not choices and invalid_block_options is not None:
        raise_flex_kernel_options_error(
            "backward",
            invalid_block_options,
            ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"),
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
        )

    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    inputs_for_autotuning = (
        bwd_input_nodes
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
        + joint_outputs.mutated_grads
    )
    input_gen_fns = {
        8: create_num_blocks_fake_generator(kv_indices),  # kv_num_blocks
        9: create_indices_fake,
        10: create_num_blocks_fake_generator(q_indices),  # q_num_blocks
        11: create_indices_fake,
        12: create_num_blocks_fake_generator(full_kv_indices),  # full_kv_num_blocks
        13: create_indices_fake,
        14: create_num_blocks_fake_generator(full_q_indices),  # full_q_num_blocks
        15: create_indices_fake,
    }

    if separate_dq:
        if dkdv_input_nodes is None:
            raise AssertionError("Expected dK/dV inputs for separate backward phases")
        choices = dkdv_choices
        inputs_for_autotuning = dkdv_input_nodes

    broadcasted_grad_key, _ = autotune_select_algorithm(
        "flex_attention_backward",
        choices,
        # Autotuning materializes benchmark tensors. Scalar shape captures stay
        # in subgraph_inps below for dependency tracking and codegen.
        [x for x in inputs_for_autotuning if is_tensor_ir_node(x)],
        dkdv_layout if separate_dq else layout_broadcasted_k,
        input_gen_fns=input_gen_fns,
    )  # [Bq, Hkv, seq_len_kv, k_head_dim]

    # need subgraph inputs and outputs to analyze all symints used in flex attention
    broadcasted_grad_key.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    broadcasted_grad_key.data.data.subgraph_outs = get_bwd_subgraph_outputs(
        fw_subgraph_buffer, mask_graph_buffer, joint_outputs
    )

    if separate_dq:
        if dq_input_nodes is None:
            raise AssertionError("Expected dQ inputs for separate backward phases")
        grad_query, _ = autotune_select_algorithm(
            "flex_attention_backward_dq",
            dq_choices,
            dq_input_nodes,
            layout_grad_query,
            input_gen_fns=input_gen_fns,
        )
        grad_query.data.data.subgraph_inps = list(mask_mod_other_buffers)
        grad_query.data.data.subgraph_outs = get_bwd_subgraph_outputs(
            fw_subgraph_buffer, mask_graph_buffer, joint_outputs
        )
        if split_dkdv:
            if partial_grad_value is None or dkdv_options is None:
                raise AssertionError("Expected split dK/dV buffer and options")
            owner_size = [
                Bq,
                Hkv,
                dkdv_options["ROCM_GQA_SPLITS"],
                seq_len_kv,
                qk_head_dim,
            ]
            grad_key_owners = lowerings[aten.reshape](broadcasted_grad_key, owner_size)
            grad_value_owners = lowerings[aten.reshape](partial_grad_value, owner_size)
            broadcasted_grad_key = lowerings[aten.sum](grad_key_owners, axis=2)
            broadcasted_grad_key = lowerings[aten.mul](
                broadcasted_grad_key, dkdv_options["SM_SCALE"]
            )
            broadcasted_grad_key = to_dtype(broadcasted_grad_key, key.get_dtype())
            broadcasted_grad_value = to_dtype(
                lowerings[aten.sum](grad_value_owners, axis=2), value.get_dtype()
            )

    if V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv)):
        grad_key = broadcasted_grad_key
        grad_value = broadcasted_grad_value
    else:
        if not V.graph.sizevars.evaluate_expr(sympy.Gt(Bq, 1) & sympy.Eq(Bkv, 1)):
            raise AssertionError(
                f"Bq and Bkv must broadcastable. "
                f"Got Bq={V.graph.sizevars.evaluate_expr(Bq)} "
                f"and Bkv={V.graph.sizevars.evaluate_expr(Bkv)}"
            )
        grad_key = lowerings[aten.sum](broadcasted_grad_key, axis=0, keepdims=True)
        grad_value = lowerings[aten.sum](broadcasted_grad_value, axis=0, keepdims=True)

    # Cast captured grads to match original buffer dtypes. Gradients are accumulated
    # in fp32 for precision, then cast to the original dtype (e.g., bf16) here.
    captured_grads = tuple(
        to_dtype(g, orig.get_dtype())
        if g is not None and g.get_dtype() != orig.get_dtype()
        else g
        for g, orig in zip(joint_outputs.captured_grads, score_mod_other_buffers)
    )

    return (grad_query, grad_key, grad_value, captured_grads)


def get_bwd_subgraph_outputs(
    subgraph_buffer: SubgraphResults,
    mask_graph_buffer: SubgraphResults,
    joint_outputs: JointOutputResult,
) -> list[ComputedBuffer | TensorBox | None]:
    subgraph_buffer = (
        subgraph_buffer if isinstance(subgraph_buffer, Sequence) else [subgraph_buffer]
    )
    mask_graph_buffer = (
        mask_graph_buffer
        if isinstance(mask_graph_buffer, Sequence)
        else [mask_graph_buffer]
    )
    joint_output_buffers = [
        joint_outputs.grad_input,
        *joint_outputs.captured_grads_compute,
        *joint_outputs.captured_grads,
        *joint_outputs.mutated_grads,
    ]

    return [*subgraph_buffer, *mask_graph_buffer, *joint_output_buffers]
