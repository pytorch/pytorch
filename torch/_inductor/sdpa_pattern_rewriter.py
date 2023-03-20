import logging
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn
from torch.fx import GraphModule
from torch.fx.subgraph_rewriter import replace_multiple_patterns_with_filters

log = logging.getLogger(__name__)

# expected function signature: (query, key, value, attn_mask, dropout_p, is_causal, scale_factor) -> Tensor
_scaled_dot_product_attention_impl = torch.nn.functional.scaled_dot_product_attention


def set_scaled_dot_product_attention_impl(
    scaled_dot_product_attention_impl: Callable,
) -> None:
    """
    Sets the scaled_dot_product_attention function to be used to replace inefficient patterns.

    Call this first, before calling fuse_scaled_dot_product_attention(...)

    The passed function has to match the function signature of torch.nn.functional.scaled_dot_product with an additional
    scale factor parameter which is commonly set to sqrt(EMBED_DIM) where EMBED_DIM equals query.size(-1)

    IMPORTANT: Tested only for native functions (i.e. similar to torch.nn.functional.scaled_dot_product_attention, not custom Python functions which might be broken down further by torch dynamo )
    This function is called once per graph module be

    Arguments:
        scaled_dot_product_attention_impl : (see above) sdpa implementation callable with signature (query, key, value, attn_mask, dropout_p, is_causal, scale_factor) -> Tensor
    """
    global _scaled_dot_product_attention_impl
    _scaled_dot_product_attention_impl = scaled_dot_product_attention_impl


def fuse_scaled_dot_product_attention(gm: torch.fx.GraphModule):
    """
    Fuses Convolution/BN layers for inference purposes. Modifies passed graph module in-place and returns it.

    Arguments:
        gm : torch.fx.GraphModule to apply sdpa graph optimizations to ( in-place)
    Returns:
        Same graph module after (inplace) optimizations have been applied.
    """
    global _scale_factor_dot_product_attention_replacement_graph_modules, _scaled_dot_product_attention_impl
    if _scaled_dot_product_attention_impl is None:
        log.warning(
            "fuse_scaled_dot_product_attention(gm) called, without calling set_scaled_dot_product_attention_impl() first. Graph replacements will be dysfunctional."
        )
    _ensure_scale_factor_dot_product_attention_replacement_graph_modules()  # Compiles patterns and replacements on first call
    # for (
    #    pattern,
    #    replacement,
    # ) in _scale_factor_dot_product_attention_replacement_graph_modules:
    #    replace_multiple_patterns_with_filters(gm, { pattern : replacement}, filters=None)
    pmap = {
        pattern: replacement
        for pattern, replacement in _scale_factor_dot_product_attention_replacement_graph_modules
    }
    replace_multiple_patterns_with_filters(gm, pmap, match_filters=None)
    # for pattern, replacement in _scale_factor_dot_product_attention_replacement_graph_modules:
    #    replace_multiple_patterns_with_filters(gm, { pattern : replacement }, match_filters=None)
    # replace_pattern(gm, pattern, replacement)

    gm.graph.lint()
    gm.recompile()
    return gm


@torch.fx.wrap
def _scale_factor_dot_product_attention(
    query, key, value, attn_mask, dropout_p, is_causal, scale_factor
):
    """
    Wrapper for the configured scaled_dot_product_attention_impl, which ensures
    the function is not broken down further by torch fx when compiling the replacement Graph
    """
    # If we do not torch.fx.wrap the call, dynamo compilation step could fail
    # with TypeError: scaled_dot_product_attention(): argument 'scale_factor' (position 7) must be Number, not Proxy
    # error thrown in python_arg_parser.cpp
    return _scaled_dot_product_attention_impl(
        query, key, value, attn_mask, dropout_p, is_causal, scale=scale_factor
    )


def _sfdp_pattern_1(query, key, value, inv_scale_factor):
    return (
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale_factor)
        .softmax(dim=-1)
        .matmul(value)
    )


def _sfdp_replacement_1(query, key, value, inv_scale_factor):
    return _scale_factor_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale_factor=1.0 / inv_scale_factor,
    )


def _sfdp_pattern_2(query, key, value, scale_factor):
    return (
        torch.matmul(query, key.transpose(-2, -1))
        .mul(scale_factor)
        .softmax(dim=-1)
        .matmul(value)
    )


def _sfdp_replacement_2(query, key, value, scale_factor):
    return _scale_factor_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale_factor=scale_factor,
    )


def _sfdp_pattern_3(query, key, value, inv_scale_factor, dropout_p):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale_factor)
        .softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


def _sfdp_replacement_3(query, key, value, inv_scale_factor, dropout_p):
    return _scale_factor_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale_factor=1.0 / inv_scale_factor,
    )


def _sfdp_pattern_4(query, key, value, scale_factor, dropout_p):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


def _sfdp_replacement_4(query, key, value, scale_factor, dropout_p):
    return _scale_factor_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale_factor=scale_factor,
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
    return _scale_factor_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale_factor=1.0 / inv_scale_factor,
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
    return _scale_factor_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale_factor=1.0 / inv_scale_factor,
    )


# TODO: Add more patterns
_scale_factor_dot_product_attention_replacements: List[Tuple[Callable, Callable]] = [
    (_sfdp_pattern_1, _sfdp_replacement_1),
    (_sfdp_pattern_2, _sfdp_replacement_2),
    (_sfdp_pattern_3, _sfdp_replacement_3),
    (_sfdp_pattern_4, _sfdp_replacement_4),
    (_sfdp_pattern_5, _sfdp_replacement_5),
    (_sfdp_pattern_6, _sfdp_replacement_6),
]

_scale_factor_dot_product_attention_replacement_graph_modules: Optional[
    List[Tuple[GraphModule, GraphModule]]
] = None


def _ensure_scale_factor_dot_product_attention_replacement_graph_modules():
    """
    Builds the list of (scaled_dot_product_attention, replacement) tuples, if they haven't been built yet.
    this operation is a bit expensive and should be done only once, since it requires torch.fx.symbolic_trace
    for every pattern and replacement.
    """
    global _scale_factor_dot_product_attention_replacement_graph_modules
    if _scale_factor_dot_product_attention_replacement_graph_modules is not None:
        return
    _scale_factor_dot_product_attention_replacement_graph_modules = [
        (torch.fx.symbolic_trace(pattern), torch.fx.symbolic_trace(replacement))
        for pattern, replacement in _scale_factor_dot_product_attention_replacements
    ]
