# mypy: allow-untyped-defs
"""PyTorch Inductor pattern matcher pass for MoE (Mixture-of-Experts) expert computation.

Matches the batched expert computation used in Llama4TextExperts and similar models:

    gate_up = bmm(hidden_3d, gate_up_proj)      # (E, T, 2*D)
    gate, up = gate_up.chunk(2, dim=-1)
    out = bmm(up * silu(gate), down_proj)        # (E, T, H)

and replaces it with torch.nn.functional.grouped_mm which uses specialized
grouped-GEMM kernels (currently supports BF16 on CUDA SM>=80 and Intel XPU).

Two variants are registered:
  • SiLU / SwiGLU  (silu_gate / (1 + exp(-silu_gate)))
  • GELU           (gate * 0.5 * (1 + erf(gate * sqrt(0.5))))

Uses register_graph_pattern so no serialized pattern files are needed.
"""

import functools
import operator

import torch

from ..._dynamo.utils import counters
from ..pattern_matcher import (
    Ignored,
    KeywordArg,
    Match,
    register_graph_pattern,
    CallFunction,
)

aten = torch.ops.aten
prims = torch.ops.prims


# ---------------------------------------------------------------------------
# Replacement helper
# ---------------------------------------------------------------------------


def _replace_with_grouped_mm(match: Match, activation: str) -> None:
    """Replace bmm(up * act(gate), down_proj) with grouped_mm."""
    hidden_3d = match.kwargs["hidden_3d"]
    gate_up_proj = match.kwargs["gate_up_proj"]
    down_proj = match.kwargs["down_proj"]

    # Check BF16 (grouped_mm requires bfloat16)
    val = hidden_3d.meta.get("val")
    if val is None or val.dtype != torch.bfloat16:
        return

    graph = match.graph
    output_node = match.output_nodes()[0]

    with graph.inserting_before(output_node):
        # gate_up = grouped_mm(hidden_3d, gate_up_proj)
        gate_up = graph.call_function(
            aten._grouped_mm.default, (hidden_3d, gate_up_proj), {"offs": None}
        )
        gate_up.meta["val"] = torch.ops.aten.bmm.default(
            hidden_3d.meta["val"], gate_up_proj.meta["val"]
        )

        # half = gate_up.shape[-1] // 2
        half_val = gate_up.meta["val"].shape[-1] // 2

        # gate, up = gate_up.split(half, dim=-1)
        split = graph.call_function(aten.split.Tensor, (gate_up, half_val, -1))
        split.meta["val"] = [
            gate_up.meta["val"][..., :half_val],
            gate_up.meta["val"][..., half_val:],
        ]

        gate_item = graph.call_function(operator.getitem, (split, 0))
        gate_item.meta["val"] = split.meta["val"][0]

        up_item = graph.call_function(operator.getitem, (split, 1))
        up_item.meta["val"] = split.meta["val"][1]

        # activation(gate) — use decomposed forms since silu/gelu are not
        # registered inductor lowerings (they're in decompositions).
        if activation == "silu":
            # silu(x) = x * sigmoid(x)
            sig = graph.call_function(aten.sigmoid.default, (gate_item,))
            sig.meta["val"] = gate_item.meta["val"]
            act = graph.call_function(aten.mul.Tensor, (gate_item, sig))
        else:  # gelu: x * 0.5 * (1 + erf(x * sqrt(0.5)))
            import math
            mul_half = graph.call_function(aten.mul.Tensor, (gate_item, 0.5))
            mul_half.meta["val"] = gate_item.meta["val"]
            mul_sqrt = graph.call_function(aten.mul.Tensor, (gate_item, math.sqrt(0.5)))
            mul_sqrt.meta["val"] = gate_item.meta["val"]
            erf_val = graph.call_function(aten.erf.default, (mul_sqrt,))
            erf_val.meta["val"] = gate_item.meta["val"]
            add_one = graph.call_function(aten.add.Tensor, (erf_val, 1.0))
            add_one.meta["val"] = gate_item.meta["val"]
            act = graph.call_function(aten.mul.Tensor, (mul_half, add_one))
        act.meta["val"] = gate_item.meta["val"]

        # up * act(gate)
        mul = graph.call_function(aten.mul.Tensor, (up_item, act))
        mul.meta["val"] = up_item.meta["val"]

        # grouped_mm(up * act(gate), down_proj)
        result = graph.call_function(
            aten._grouped_mm.default, (mul, down_proj), {"offs": None}
        )
        result.meta["val"] = output_node.meta.get("val")

    output_node.replace_all_uses_with(result)
    match.erase_nodes()
    counters["inductor"]["fuse_moe"] += 1


# ---------------------------------------------------------------------------
# Pattern definitions (BF16 — inductor decomposes silu/chunk to prims)
# ---------------------------------------------------------------------------
#
# For BF16 inputs, inductor decomposes:
#   chunk(2, dim=-1)   →  split.Tensor
#   silu(bf16_tensor)  →  convert_type(x, f32) → neg → exp → add → div
#                         → convert_type(result, bf16)
#   gelu(bf16_tensor)  →  convert_type(x, f32) → mul(0.5) → erf → ... → bf16
#
# For float32 inputs, the decomposition is similar but without convert_element_type.
#
# We register both so that _moe_dtype_check (inside the handler) gates on BF16.

def _build_silu_bf16_inference_pattern():
    """BF16 silu pattern for inference (fwd_only decomposition)."""
    bmm_default = CallFunction(aten.bmm.default, KeywordArg("hidden_3d"), KeywordArg("gate_up_proj"))
    split_Tensor = CallFunction(aten.split.Tensor, bmm_default, Ignored(), Ignored(), _users=2)
    getitem_up = CallFunction(operator.getitem, split_Tensor, 1)
    getitem_gate = CallFunction(operator.getitem, split_Tensor, 0)
    convert_f32 = CallFunction(prims.convert_element_type.default, getitem_gate, Ignored(), _users=2)
    neg = CallFunction(aten.neg.default, convert_f32)
    exp = CallFunction(aten.exp.default, neg)
    add = CallFunction(aten.add.Tensor, exp, Ignored())
    div = CallFunction(aten.div.Tensor, convert_f32, add)
    convert_bf16 = CallFunction(prims.convert_element_type.default, div, Ignored())
    mul = CallFunction(aten.mul.Tensor, getitem_up, convert_bf16)
    return CallFunction(aten.bmm.default, mul, KeywordArg("down_proj"), _users=0)


def _build_silu_f32_inference_pattern():
    """Float32 silu pattern for inference."""
    bmm_default = CallFunction(aten.bmm.default, KeywordArg("hidden_3d"), KeywordArg("gate_up_proj"))
    split_Tensor = CallFunction(aten.split.Tensor, bmm_default, Ignored(), Ignored(), _users=2)
    getitem_up = CallFunction(operator.getitem, split_Tensor, 1)
    getitem_gate = CallFunction(operator.getitem, split_Tensor, 0, _users=2)
    neg = CallFunction(aten.neg.default, getitem_gate)
    exp = CallFunction(aten.exp.default, neg)
    add = CallFunction(aten.add.Tensor, exp, Ignored())
    div = CallFunction(aten.div.Tensor, getitem_gate, add)
    mul = CallFunction(aten.mul.Tensor, getitem_up, div)
    return CallFunction(aten.bmm.default, mul, KeywordArg("down_proj"), _users=0)


def _build_gelu_bf16_inference_pattern():
    """BF16 gelu pattern for inference."""
    bmm_default = CallFunction(aten.bmm.default, KeywordArg("hidden_3d"), KeywordArg("gate_up_proj"))
    split_Tensor = CallFunction(aten.split.Tensor, bmm_default, Ignored(), Ignored(), _users=2)
    getitem_up = CallFunction(operator.getitem, split_Tensor, 1)
    getitem_gate = CallFunction(operator.getitem, split_Tensor, 0)
    convert_f32 = CallFunction(prims.convert_element_type.default, getitem_gate, Ignored(), _users=2)
    mul_half = CallFunction(aten.mul.Tensor, convert_f32, Ignored())
    mul_invsqrt = CallFunction(aten.mul.Tensor, convert_f32, Ignored())
    erf = CallFunction(aten.erf.default, mul_invsqrt)
    add_one = CallFunction(aten.add.Tensor, erf, Ignored())
    mul_gate = CallFunction(aten.mul.Tensor, mul_half, add_one)
    convert_bf16 = CallFunction(prims.convert_element_type.default, mul_gate, Ignored())
    mul = CallFunction(aten.mul.Tensor, getitem_up, convert_bf16)
    return CallFunction(aten.bmm.default, mul, KeywordArg("down_proj"), _users=0)


def _build_gelu_f32_inference_pattern():
    """Float32 gelu pattern for inference."""
    bmm_default = CallFunction(aten.bmm.default, KeywordArg("hidden_3d"), KeywordArg("gate_up_proj"))
    split_Tensor = CallFunction(aten.split.Tensor, bmm_default, Ignored(), Ignored(), _users=2)
    getitem_up = CallFunction(operator.getitem, split_Tensor, 1)
    getitem_gate = CallFunction(operator.getitem, split_Tensor, 0, _users=2)
    mul_half = CallFunction(aten.mul.Tensor, getitem_gate, Ignored())
    mul_invsqrt = CallFunction(aten.mul.Tensor, getitem_gate, Ignored())
    erf = CallFunction(aten.erf.default, mul_invsqrt)
    add_one = CallFunction(aten.add.Tensor, erf, Ignored())
    mul_gate = CallFunction(aten.mul.Tensor, mul_half, add_one)
    mul = CallFunction(aten.mul.Tensor, getitem_up, mul_gate)
    return CallFunction(aten.bmm.default, mul, KeywordArg("down_proj"), _users=0)


# ---------------------------------------------------------------------------
# Pattern registration
# ---------------------------------------------------------------------------


@functools.cache
def _moe_init(input_device=None):
    from .joint_graph import patterns

    register_graph_pattern(
        _build_silu_bf16_inference_pattern(),
        pass_dict=patterns,
    )(lambda match, *a, **kw: _replace_with_grouped_mm(match, "silu"))

    register_graph_pattern(
        _build_silu_f32_inference_pattern(),
        pass_dict=patterns,
    )(lambda match, *a, **kw: _replace_with_grouped_mm(match, "silu"))

    register_graph_pattern(
        _build_gelu_bf16_inference_pattern(),
        pass_dict=patterns,
    )(lambda match, *a, **kw: _replace_with_grouped_mm(match, "gelu"))

    register_graph_pattern(
        _build_gelu_f32_inference_pattern(),
        pass_dict=patterns,
    )(lambda match, *a, **kw: _replace_with_grouped_mm(match, "gelu"))
