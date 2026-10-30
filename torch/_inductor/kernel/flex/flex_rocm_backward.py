# mypy: allow-untyped-defs
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch._inductor.virtualized import V
from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...ir import FixedLayout
from ...lowering import empty_strided
from ...select_algorithm import SymbolicGridFn, TritonTemplate
from .common import infer_dense_strides, load_flex_template
from .flex_flash_attention import is_trivial_mask_graph, is_trivial_score_graph


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


def _use_rocm_backward_template(is_gfx950, kernel_options):
    return is_gfx950 and any(
        kernel_options.get(option, False)
        for option in (
            "ROCM_EARLY_DP",
            "XCD_REMAP",
            "ROCM_D256_PANEL_DOTS",
            "ROCM_XCD_SNAKE",
        )
    )


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


def _get_rocm_backward_autotune_options(
    user_kernel_options,
    kernel_options,
    input_nodes,
    out,
    *,
    backend,
    is_gfx950,
    score_is_trivial,
    has_captures,
    mask_is_causal,
    mask_is_scalar_true,
    block_mask_is_default,
    grad_logsumexp_is_none,
):
    """Return legal split candidates independently of the measured default profiles."""
    if (
        is_gfx950 is not True
        or backend not in ("AUTO", "TRITON")
        or score_is_trivial is not True
        or has_captures is not False
        or grad_logsumexp_is_none is not True
        or len(input_nodes) != 16
        or kernel_options.get("USE_TMA", False) is not False
        or kernel_options.get("PRESCALE_QK") is not False
        or kernel_options.get("num_consumer_groups", 0)
    ):
        return []
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
            return []
    query, key, value = input_nodes[:3]
    if any(
        tensor is None or len(tensor.get_size()) != 4 for tensor in (query, key, value)
    ):
        return []
    sizevars = V.graph.sizevars
    dimension = next(
        (
            dim
            for dim in (128, 256)
            if sizevars.statically_known_equals(query.get_size()[3], dim)
        ),
        None,
    )
    if dimension is None:
        return []
    batch, heads, query_length, _ = query.get_size()
    kv_heads, kv_length = key.get_size()[1:3]
    if (
        not all(
            sizevars.statically_known_gt(size, 0)
            for size in (batch, heads, query_length, kv_heads, kv_length)
        )
        or not all(
            sizevars.statically_known_multiple_of(length, 128)
            for length in (query_length, kv_length)
        )
        or kernel_options.get("IS_DIVISIBLE") is not True
        or kernel_options.get("SAFE_HEAD_DIM") is not True
        or type(kernel_options.get("SM_SCALE")) not in (int, float)
        or kernel_options.get("SM_SCALE") != 1 / math.sqrt(dimension)
        or not all(
            kernel_options.get(name) is not None
            and sizevars.statically_known_equals(kernel_options[name], dimension)
            for name in (
                "QK_HEAD_DIM",
                "V_HEAD_DIM",
                "QK_HEAD_DIM_ROUNDED",
                "V_HEAD_DIM_ROUNDED",
            )
        )
    ):
        return []
    gqa = 8 if dimension == 128 else kernel_options.get("GQA_SHARED_HEADS")
    if gqa is None or not any(
        sizevars.statically_known_equals(gqa, ratio)
        for ratio in ((8,) if dimension == 128 else (1, 4))
    ):
        return []
    if not sizevars.statically_known_equals(heads, kv_heads * gqa):
        return []
    causal = dimension == 128 and mask_is_causal is True
    if causal:
        if (
            kernel_options.get("HAS_FULL_BLOCKS") is not True
            or not sizevars.statically_known_equals(query_length, kv_length)
            or not all(
                kernel_options.get(name) is not None
                and sizevars.statically_known_equals(kernel_options[name], 128)
                for name in ("SPARSE_Q_BLOCK_SIZE", "SPARSE_KV_BLOCK_SIZE")
            )
        ):
            return []
    elif (
        mask_is_scalar_true is not True
        or block_mask_is_default is not True
        or kernel_options.get("HAS_FULL_BLOCKS") is not False
        or not all(
            kernel_options.get(name) is not None
            and sizevars.statically_known_equals(
                kernel_options[name], _LARGE_SPARSE_BLOCK_SIZE
            )
            for name in ("SPARSE_Q_BLOCK_SIZE", "SPARSE_KV_BLOCK_SIZE")
        )
    ):
        return []
    device = query.get_device()
    if getattr(device, "type", None) != "cuda":
        return []
    query_shape = (batch, heads, query_length, dimension)
    kv_shape = (batch, kv_heads, kv_length, dimension)
    stats_shape = (batch, heads, query_length)
    operands: list[tuple[Any, tuple[Any, ...], torch.dtype]] = [
        (query, query_shape, torch.bfloat16),
        (key, kv_shape, torch.bfloat16),
        (value, kv_shape, torch.bfloat16),
        (input_nodes[3], stats_shape, torch.float32),
        (input_nodes[4], stats_shape, torch.float32),
        (input_nodes[5], query_shape, torch.bfloat16),
        (out, query_shape, torch.bfloat16),
    ]
    if causal:
        blocks = query_length // 128
        operands.extend(
            (
                tensor,
                (1, 1, blocks) if index % 2 == 0 else (1, 1, blocks, blocks),
                torch.int32,
            )
            for index, tensor in enumerate(input_nodes[8:])
        )
    else:
        operands.extend(
            (tensor, (1, 1, 1) if index % 2 == 0 else (1, 1, 1, 1), torch.int32)
            for index, tensor in enumerate(input_nodes[8:12])
        )
        operands.extend((tensor, (0,), torch.float32) for tensor in input_nodes[12:])
    for tensor, shape, dtype in operands:
        if (
            tensor is None
            or tensor.get_dtype() != dtype
            or tensor.get_device() != device
            or len(tensor.get_size()) != len(shape)
            or len(tensor.get_stride()) != len(shape)
            or not all(
                sizevars.statically_known_equals(actual, expected)
                for actual, expected in zip(tensor.get_size(), shape)
            )
        ):
            return []
        stride = 1
        for size, actual in zip(reversed(shape), reversed(tensor.get_stride())):
            if not sizevars.statically_known_equals(actual, stride):
                return []
            stride *= size
    options = _get_rocm_backward_config_options(kernel_options, is_gfx950=True)
    options.update(
        USE_TMA=False,
        ROCM_SEPARATE_DQ=True,
        ROCM_EARLY_DP=True,
        ROCM_D256_PANEL_DOTS=dimension == 256,
        ROCM_XCD_SNAKE=False,
        ROCM_D128_GQA_QUERY_TILE_OUTER=False,
        ROCM_GQA_SPLIT_DKDV=False,
        num_warps=4,
        num_stages=2,
        matrix_instr_nonkdim=0,
        waves_per_eu=0,
        kpack=1,
    )
    tiles = (32, 128, 128, 32) if dimension == 128 else (16, 64, 128, 32)
    options.update(zip(("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"), tiles))
    if not _use_rocm_separate_dq(
        True,
        query,
        key,
        value,
        options,
        score_is_trivial,
        has_captures,
        mask_is_scalar_true=mask_is_scalar_true,
        block_mask_is_default=block_mask_is_default,
    ):
        return []
    return [dict(options, XCD_REMAP=remap) for remap in (False, True)]


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
    name="flex_attention_backward_rocm",
    grid=flex_attention_backward_grid,
    source=load_flex_template("flex_backwards_rocm")
    + load_flex_template("flex_backwards")
    + load_flex_template("utilities"),
    always_freeze_layout=True,
)


@dataclass
class RocmBackwardPlan:
    dkdv_choices: list[Any]
    dq_choices: list[Any]
    dkdv_input_nodes: list[Any] | None
    dq_input_nodes: list[Any] | None
    partial_grad_value: Any
    dkdv_options: dict[str, Any] | None
    split_dkdv: bool
    dkdv_layout: Any


def create_backward_plan(
    bwd_input_nodes,
    *,
    cur_kernel_options,
    user_kernel_options,
    rocm_requests,
    out,
    backend,
    single_config,
    fw_graph,
    mask_graph,
    block_mask,
    grad_logsumexp,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    bwd_subgraphs,
    dkdv_layout,
    layout_grad_query,
    candidate_options=None,
    profile_candidate=False,
):
    r"""Create gfx950 backward phase choices after device dispatch."""
    query, key, value, logsumexp, _, grad_out = bwd_input_nodes[:6]
    Bq, _, _, qk_head_dim = query.get_size()
    _, Hkv, seq_len_kv, _ = key.get_size()
    dtype = query.get_dtype()
    head_dim = V.graph.sizevars.guard_int(qk_head_dim)
    is_gfx950 = True
    dkdv_choices = []
    dq_choices = []
    dkdv_input_nodes = None
    dq_input_nodes = None
    partial_grad_value = None
    dkdv_options = None
    split_dkdv = False
    auto_options = None
    if candidate_options is None and not config.max_autotune:
        auto_options = _get_rocm_backward_auto_profile(
            user_kernel_options,
            cur_kernel_options,
            bwd_input_nodes,
            out,
            backend=backend,
            is_gfx950=is_gfx950,
            single_config=single_config,
            score_is_trivial=is_trivial_score_graph(fw_graph),
            has_captures=bool(score_mod_other_buffers or mask_mod_other_buffers),
            mask_is_causal=_is_rocm_simple_causal_mask_graph(mask_graph),
            mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(mask_graph),
            block_mask_is_default=_is_rocm_default_block_mask(block_mask, query, key),
            grad_logsumexp_is_none=grad_logsumexp is None,
        )
    active_rocm_requests = rocm_requests
    selected_options = (
        candidate_options if candidate_options is not None else auto_options
    )
    if selected_options is not None:
        selected_options = selected_options.copy()
        active_rocm_requests = _normalize_rocm_backward_options(selected_options)
        cur_kernel_options = _get_rocm_backward_config_options(
            selected_options, is_gfx950=is_gfx950
        )
        single_config = True

    fp32_balanced64_pilot = single_config and _use_rocm_d128_fp32_balanced64_coverage(
        is_gfx950,
        query,
        key,
        value,
        grad_out,
        out,
        logsumexp,
        cur_kernel_options,
        is_trivial_score_graph(fw_graph),
        bool(score_mod_other_buffers or mask_mod_other_buffers),
        requested=active_rocm_requests.fp32_balanced64_coverage_requested,
        other_requests=active_rocm_requests.fp32_balanced64_coverage_other_requests,
        mask_is_causal=_is_rocm_simple_causal_mask_graph(mask_graph),
        grad_logsumexp_is_none=grad_logsumexp is None,
    )
    fp16_pilot = single_config and _use_rocm_d128_fp16_coverage(
        is_gfx950,
        query,
        key,
        value,
        grad_out,
        out,
        logsumexp,
        cur_kernel_options,
        is_trivial_score_graph(fw_graph),
        bool(score_mod_other_buffers or mask_mod_other_buffers),
        requested=active_rocm_requests.fp16_coverage_requested,
        other_requests=active_rocm_requests.fp16_coverage_other_requests,
        mask_is_causal=_is_rocm_simple_causal_mask_graph(mask_graph),
        grad_logsumexp_is_none=grad_logsumexp is None,
    )
    use_separate_dq = single_config and _use_rocm_separate_dq(
        is_gfx950,
        query,
        key,
        value,
        cur_kernel_options,
        is_trivial_score_graph(fw_graph),
        bool(score_mod_other_buffers or mask_mod_other_buffers),
        mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(mask_graph),
        block_mask_is_default=_is_rocm_default_block_mask(block_mask, query, key),
        fp16_pilot=fp16_pilot,
        fp32_balanced64_pilot=fp32_balanced64_pilot,
    )
    use_d192_panel_phase = single_config and _use_rocm_d192_panel_phase(
        is_gfx950,
        query,
        key,
        value,
        cur_kernel_options,
        is_trivial_score_graph(fw_graph),
        bool(score_mod_other_buffers or mask_mod_other_buffers),
        requested=active_rocm_requests.d192_panel_phase_requested,
        mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(mask_graph),
        block_mask_is_default=_is_rocm_default_block_mask(block_mask, query, key),
        grad_logsumexp_is_none=grad_logsumexp is None,
        grad_out=grad_out,
    )
    use_separate_dq = use_separate_dq or use_d192_panel_phase
    use_d96_panel_phase = single_config and _use_rocm_d96_panel_phase(
        is_gfx950,
        query,
        key,
        value,
        cur_kernel_options,
        is_trivial_score_graph(fw_graph),
        bool(score_mod_other_buffers or mask_mod_other_buffers),
        requested=active_rocm_requests.d96_panel_phase_requested
        if active_rocm_requests.d192_panel_phase_requested is not True
        else False,
        mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(mask_graph),
        block_mask_is_default=_is_rocm_default_block_mask(block_mask, query, key),
        grad_logsumexp_is_none=grad_logsumexp is None,
        grad_out=grad_out,
    )
    use_separate_dq = use_separate_dq or use_d96_panel_phase
    use_gqa_split = single_config and _use_rocm_gqa_split_dkdv(
        is_gfx950,
        query,
        key,
        value,
        cur_kernel_options,
        is_trivial_score_graph(fw_graph),
        is_trivial_mask_graph(mask_graph),
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
            single_config=single_config,
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
            score_is_trivial=is_trivial_score_graph(fw_graph),
            has_captures=bool(score_mod_other_buffers or mask_mod_other_buffers),
            mask_is_causal=_is_rocm_simple_causal_mask_graph(mask_graph),
            mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(mask_graph),
            block_mask_is_default=_is_rocm_default_block_mask(block_mask, query, key),
            grad_logsumexp_is_none=grad_logsumexp is None,
            is_automatic=auto_options is not None or profile_candidate is True,
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

    return RocmBackwardPlan(
        dkdv_choices=dkdv_choices,
        dq_choices=dq_choices,
        dkdv_input_nodes=dkdv_input_nodes,
        dq_input_nodes=dq_input_nodes,
        partial_grad_value=partial_grad_value,
        dkdv_options=dkdv_options,
        split_dkdv=split_dkdv,
        dkdv_layout=dkdv_layout,
    )


def append_backward_autotune_choices(
    choices,
    bwd_input_nodes,
    *,
    cur_kernel_options,
    user_kernel_options,
    rocm_requests,
    out,
    backend,
    fw_graph,
    mask_graph,
    block_mask,
    grad_logsumexp,
    score_mod_other_buffers,
    mask_mod_other_buffers,
    bwd_subgraphs,
    dkdv_layout,
    layout_grad_query,
):
    """Append complete split candidates while retaining the fused candidates."""
    from .flex_rocm_backward_autotune import RocmBackwardChoiceCaller

    query, key = bwd_input_nodes[:2]
    kernel_options = _get_rocm_backward_config_options(
        cur_kernel_options, is_gfx950=True
    )
    kernel_options.setdefault("USE_TMA", False)
    kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", block_mask[14])
    kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", block_mask[15])
    kernel_options.setdefault("ROCM_D256_PANEL_DOTS", False)
    kernel_options.setdefault("ROCM_XCD_SNAKE", False)
    kernel_options.setdefault("ROCM_D128_GQA_QUERY_TILE_OUTER", False)
    semantics = dict(
        backend=backend,
        is_gfx950=True,
        score_is_trivial=is_trivial_score_graph(fw_graph),
        has_captures=bool(score_mod_other_buffers or mask_mod_other_buffers),
        mask_is_causal=_is_rocm_simple_causal_mask_graph(mask_graph),
        mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(mask_graph),
        block_mask_is_default=_is_rocm_default_block_mask(block_mask, query, key),
        grad_logsumexp_is_none=grad_logsumexp is None,
    )
    candidates = [
        (options, False)
        for options in _get_rocm_backward_autotune_options(
            user_kernel_options,
            kernel_options,
            bwd_input_nodes,
            out,
            **semantics,
        )
    ]
    profile = _get_rocm_backward_auto_profile(
        user_kernel_options,
        kernel_options,
        bwd_input_nodes,
        out,
        single_config=True,
        **semantics,
    )
    if profile is not None and not profile.get("ROCM_GQA_SPLIT_DKDV", False):
        candidates.append((profile, True))
    seen = OrderedSet([choice.hash_key() for choice in choices])
    for options, profile_candidate in candidates:
        plan = create_backward_plan(
            bwd_input_nodes,
            cur_kernel_options=kernel_options,
            user_kernel_options=user_kernel_options,
            rocm_requests=rocm_requests,
            out=out,
            backend=backend,
            single_config=True,
            fw_graph=fw_graph,
            mask_graph=mask_graph,
            block_mask=block_mask,
            grad_logsumexp=grad_logsumexp,
            score_mod_other_buffers=score_mod_other_buffers,
            mask_mod_other_buffers=mask_mod_other_buffers,
            bwd_subgraphs=bwd_subgraphs,
            dkdv_layout=dkdv_layout,
            layout_grad_query=layout_grad_query,
            candidate_options=options,
            profile_candidate=profile_candidate,
        )
        if plan.split_dkdv or plan.partial_grad_value is not None:
            continue
        for dkdv in plan.dkdv_choices:
            for dq in plan.dq_choices:
                choice = RocmBackwardChoiceCaller(bwd_input_nodes, dkdv, dq)
                choice_key = choice.hash_key()
                if choice_key not in seen:
                    choices.append(choice)
                    seen.add(choice_key)
    return choices
