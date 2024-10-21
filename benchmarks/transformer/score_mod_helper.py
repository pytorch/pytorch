import math
from functools import partial
from typing import Tuple

import torch
from torch.nn.attention.flex_attention import BlockMask, create_mask


def scaled_dot_product_attention_debug(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    B, H, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(B, H, L, S, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    print("scores", attn_weight[0, 0])
    print("attn_bias", attn_bias[0, 0])
    attn_weight += attn_bias
    print("post weight attn", attn_weight[0, 0])
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def generate_OG_pytorch(
    attn_type: str, shape: Tuple[int], dtype: torch.dtype, block_mask: BlockMask
):
    B, Hq, M, Hkv, N, D = shape
    ## From T5x

    ## From transfusion-pytorch
    def transfusion_attn(
        query,
        key,
        value,
        attn_mask,
        softcap_value,
        causal=True,
        dropout_p=0.0,
        scale=D**-0.5,
    ):
        import einx

        from einops import einsum

        def softclamp(t, value=50.0):
            return (t / value).tanh() * value

        q = query * scale
        k = key
        v = value

        sim = einsum(q, k, "b h i d, b h j d -> b h i j")

        sim = softclamp(sim, softcap_value)

        mask_value = -torch.finfo(sim.dtype).max
        mask_value = float("-inf")

        sim = einx.where("b i j, b h i j, -> b h i j", attn_mask, sim, mask_value)

        attn = sim.softmax(dim=-1)

        dropout = torch.nn.Dropout(dropout_p)
        attn = dropout(attn)

        out = einsum(attn, v, "b h i j, b h j d -> b h i d")

        return out

    if attn_type == "transfusion":
        attn_mask = create_mask(block_mask.mask_mod, B, 1, M, N, device="cuda").squeeze(
            1
        )

    from score_mod import dropout_p, softcap_value

    function_dict = {
        "noop": None,
        "causal": None,
        "offset": None,
        "rel": None,
        "head_bias": None,
        "alibi": None,
        "sliding_window": None,
        "document_mask": None,
        "prefix_ml": None,
        "transfusion": partial(
            transfusion_attn,
            softcap_value=softcap_value,
            dropout_p=dropout_p,
            attn_mask=attn_mask,
        )
        if Hq == Hkv
        else None,
    }

    return function_dict[attn_type]
