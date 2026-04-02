# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import operator

import torch
import torch._inductor

aten = torch.ops.aten
prims = torch.ops.prims

from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    CallMethod,
    CallMethodVarArgs,
    CallModule,
    CallModuleVarArgs,
    ExclusiveKeywordArg,
    Ignored,
    KeywordArg,
    ListOf,
    MultiOutputPattern,
    PatternExpr,
    RepeatedExpr,
    _TargetArgsExpr,
    _TargetExpr,
    _TargetExprVarArgs,
)

# ---------------------------------------------------------------------------
# _moe_pattern_1: bmm + chunk(silu * up) + bmm  (SiLU / SwiGLU, Llama4 style)
#
#   hidden_3d:    (E, T, H)
#   gate_up_proj: (E, H, 2*D)
#   down_proj:    (E, D, H)
#
#   gate_up            = bmm(hidden_3d, gate_up_proj)   # (E, T, 2*D)
#   gate, up           = chunk(gate_up, 2, dim=-1)      # each (E, T, D)
#   activated          = up * silu(gate)
#   out                = bmm(activated, down_proj)      # (E, T, H)
# ---------------------------------------------------------------------------

# Inference (forward-only)
bmm_default = CallFunction(aten.bmm.default, KeywordArg("hidden_3d"), KeywordArg("gate_up_proj"))
chunk_default = CallFunction(aten.chunk.default, bmm_default, Ignored(), Ignored(), _users=2)
getitem = CallFunction(operator.getitem, chunk_default, 0)
getitem_1 = CallFunction(operator.getitem, chunk_default, 1)
silu_default = CallFunction(aten.silu.default, getitem)
mul_tensor = CallFunction(aten.mul.Tensor, getitem_1, silu_default)
_moe_pattern_1_inference = CallFunction(aten.bmm.default, mul_tensor, KeywordArg("down_proj"), _users=0)


# Training (joint fwd+bwd): silu and mul outputs are consumed by the backward pass
# as well as the forward pass, so they appear with _users=2.
bmm_default = CallFunction(aten.bmm.default, KeywordArg("hidden_3d"), KeywordArg("gate_up_proj"))
chunk_default = CallFunction(aten.chunk.default, bmm_default, Ignored(), Ignored(), _users=2)
getitem = CallFunction(operator.getitem, chunk_default, 0)
getitem_1 = CallFunction(operator.getitem, chunk_default, 1)
silu_default = CallFunction(aten.silu.default, getitem, _users=2)
mul_tensor = CallFunction(aten.mul.Tensor, getitem_1, silu_default, _users=2)
_moe_pattern_1_training = MultiOutputPattern([
    CallFunction(aten.bmm.default, mul_tensor, KeywordArg("down_proj")),
    silu_default,
    mul_tensor,
    None,
])


# ---------------------------------------------------------------------------
# _moe_gelu_pattern_1: same topology with GELU activation
# ---------------------------------------------------------------------------

# Inference
bmm_default = CallFunction(aten.bmm.default, KeywordArg("hidden_3d"), KeywordArg("gate_up_proj"))
chunk_default = CallFunction(aten.chunk.default, bmm_default, Ignored(), Ignored(), _users=2)
getitem = CallFunction(operator.getitem, chunk_default, 0)
getitem_1 = CallFunction(operator.getitem, chunk_default, 1)
gelu_default = CallFunction(aten.gelu.default, getitem)
mul_tensor = CallFunction(aten.mul.Tensor, getitem_1, gelu_default)
_moe_gelu_pattern_1_inference = CallFunction(aten.bmm.default, mul_tensor, KeywordArg("down_proj"), _users=0)


# Training
bmm_default = CallFunction(aten.bmm.default, KeywordArg("hidden_3d"), KeywordArg("gate_up_proj"))
chunk_default = CallFunction(aten.chunk.default, bmm_default, Ignored(), Ignored(), _users=2)
getitem = CallFunction(operator.getitem, chunk_default, 0)
getitem_1 = CallFunction(operator.getitem, chunk_default, 1)
gelu_default = CallFunction(aten.gelu.default, getitem, _users=2)
mul_tensor = CallFunction(aten.mul.Tensor, getitem_1, gelu_default, _users=2)
_moe_gelu_pattern_1_training = MultiOutputPattern([
    CallFunction(aten.bmm.default, mul_tensor, KeywordArg("down_proj")),
    gelu_default,
    mul_tensor,
    None,
])
