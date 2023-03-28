import functools
import logging

import torch
from ..pattern_matcher import inference_graph, register_replacement, training_graph

log = logging.getLogger(__name__)
aten = torch.ops.aten


def _sfdp_pattern_1(query, key, value, inv_scale):
    return (
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale)
        .softmax(dim=-1)
        .matmul(value)
    )


def _sfdp_replacement_1(query, key, value, inv_scale):
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_2(query, key, value, scale_factor):
    return (
        torch.matmul(query, key.transpose(-2, -1))
        .mul(scale_factor)
        .softmax(dim=-1)
        .matmul(value)
    )


def _sfdp_replacement_2(query, key, value, scale_factor):
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=scale_factor,
    )


def _sfdp_pattern_3(query, key, value, inv_scale_factor, dropout_p):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale_factor)
        .softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


def _sfdp_replacement_3(query, key, value, inv_scale_factor, dropout_p):
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale_factor,
    )


def _sfdp_pattern_4(query, key, value, scale_factor, dropout_p):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


def _sfdp_replacement_4(query, key, value, scale_factor, dropout_p):
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale_factor,
    )


def _sfdp_pattern_5(query, key, value, inv_scale_factor, attn_mask):
    return (
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale_factor)
        .masked_fill(attn_mask, float("-inf"))
        .softmax(dim=-1)
        .matmul(value)
    )


def _sfdp_replacement_5(query, key, value, inv_scale_factor, attn_mask):
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / inv_scale_factor,
    )


def _sfdp_pattern_6(query, key, value, inv_scale_factor, attn_mask, dropout_p):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale_factor)
        .masked_fill(attn_mask, float("-inf"))
        .softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


def _sfdp_replacement_6(query, key, value, inv_scale_factor, attn_mask, dropout_p):
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale_factor,
    )


@functools.lru_cache(None)
def _sfdp_init():
    from .joint_graph import patterns

    # sizes/values don't actually matter for initial trace
    g = functools.partial(torch.empty, (2, 4, 8, 16), requires_grad=True)
    b = functools.partial(torch.empty, (1, 1, 8, 8), dtype=torch.bool)
    c = torch.tensor(0.5)

    for pattern, replacement, args in [
        (_sfdp_pattern_1, _sfdp_replacement_1, [g(), g(), g(), c]),
        (_sfdp_pattern_2, _sfdp_replacement_2, [g(), g(), g(), c]),
        # need to fix dropout for these:
        # (_sfdp_pattern_3, _sfdp_replacement_3, [g(), g(), g(), c, c]),
        # (_sfdp_pattern_4, _sfdp_replacement_4, [g(), g(), g(), c, c]),
        # (_sfdp_pattern_6, _sfdp_replacement_6, ...),
        # This replacement just gets expanded again in the dispatcher
        # (_sfdp_pattern_5, _sfdp_replacement_5, [g(), g(), g(), c, b()]),
    ]:
        register_replacement(pattern, replacement, args, inference_graph, patterns)
        register_replacement(pattern, replacement, args, training_graph, patterns)
