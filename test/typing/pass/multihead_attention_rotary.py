from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _rotary(x: Tensor) -> Tensor:
    return x


EMBED_DIM = 8
HEADS = 2
_query = torch.randn(3, 1, EMBED_DIM)
_key = torch.randn(3, 1, EMBED_DIM)
_value = torch.randn(3, 1, EMBED_DIM)
_proj = torch.randn(3 * EMBED_DIM, EMBED_DIM)
_bias = torch.randn(3 * EMBED_DIM)
_out_proj_w = torch.randn(EMBED_DIM, EMBED_DIM)
_out_proj_b = torch.randn(EMBED_DIM)

F.multi_head_attention_forward(
    query=_query,
    key=_key,
    value=_value,
    embed_dim_to_check=EMBED_DIM,
    num_heads=HEADS,
    in_proj_weight=_proj,
    in_proj_bias=_bias,
    bias_k=None,
    bias_v=None,
    add_zero_attn=False,
    dropout_p=0.0,
    out_proj_weight=_out_proj_w,
    out_proj_bias=_out_proj_b,
    rotary_pos_emb=_rotary,
)
