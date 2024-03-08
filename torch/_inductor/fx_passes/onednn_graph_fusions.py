import functools
import logging

import torch
from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import fwd_only, joint_fwd_bwd, register_replacement
from ..utils import is_avx512_bf16_supported, is_avx512_vnni_supported
from .fuse_attention import (
    _sfdp_params_check,
    _sfdp_pattern_18,
    _sfdp_pattern_19,
    partialize_and_update_signature,
)

log = logging.getLogger(__name__)


def _sfdp_replacement_18(query, key, value, inv_scale, causal_mask_value, causal_mask):
    counters["inductor"]["fuse_attention"] += 1
    query_permuted = query.permute([0, 2, 1, 3])
    key_permuted = key.permute([0, 2, 1, 3])
    value_permuted = value.permute([0, 2, 1, 3])
    attn_bias = torch.zeros(1, 1, query.size(1), query.size(1))
    return (
        torch.ops.mkldnn._graph_sdpa_fusion(
            query_permuted,
            key_permuted.transpose(-1, -2),
            value_permuted,
            None,
            inv_scale,
            attn_bias,  # oneDNN v3.5 will not require this workaround
            causal_mask,
            causal_mask_value,
            transpose_query=False,
            transpose_key_twice=False,
            transpose_key_once=False,
            transpose_value=False,
            apply_mask_before_scale=True,
            choose_causal_mask_over_attn_score=False,
            output_requires_transpose_and_reorder=True,
        ),
        key_permuted,
        value_permuted,
    )


def _sfdp_replacement_19(query, key, value, inv_scale, causal_mask_value, causal_mask):
    counters["inductor"]["fuse_attention"] += 1
    attn_bias = torch.zeros(1, 1, query.size(2), query.size(2))
    return torch.ops.mkldnn._graph_sdpa_fusion(
        query,
        key.transpose(-1, -2),
        value,
        None,
        inv_scale,
        attn_bias,  # oneDNN v3.5 will not require this workaround
        causal_mask,
        causal_mask_value,
        transpose_query=False,
        transpose_key_twice=False,
        transpose_key_once=False,
        transpose_value=False,
        apply_mask_before_scale=True,
        choose_causal_mask_over_attn_score=False,
        output_requires_transpose_and_reorder=True,
    )


def _onednn_graph_extra_check(match):
    if not (
        config.onednn_graph
        and torch._C._has_onednn_graph
        and is_avx512_vnni_supported()
    ):
        return False
    query = match.kwargs["query"].meta["val"]
    if query.dtype not in [torch.float32, torch.bfloat16]:
        return False
    if str(query.device) != "cpu":
        return False
    if (query.dtype == torch.bfloat16) and not is_avx512_bf16_supported():
        return False
    return _sfdp_params_check(match)


def _get_onednn_graph_sfdp_patterns():
    from .joint_graph import patterns

    device = "cpu"

    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    g_inp = functools.partial(
        torch.empty, (2, 4, 8, 16), device=device, requires_grad=True
    )
    # inv_scale
    c_inp = functools.partial(torch.tensor, 2.0, device=device)

    # causal_mask
    cmask_inp = functools.partial(torch.empty, (1, 1, 4, 4), device=device)
    cmask_q_post_permute_inp = functools.partial(
        torch.empty, (1, 1, 8, 8), device=device
    )

    # softmax will generate a dtype conversion on inputs if they are in half,
    # but will not in float, so we generate a pattern for both
    for dtype in [torch.float, torch.half]:
        g = functools.partial(g_inp, dtype=dtype)
        c = functools.partial(c_inp, dtype=dtype)
        cmask = functools.partial(cmask_inp, dtype=torch.bool)
        cmask_q_post_permute = functools.partial(
            cmask_q_post_permute_inp, dtype=torch.bool
        )

        candidates = [
            (
                _sfdp_pattern_18,
                _sfdp_replacement_18,
                [g(), g(), g(), c(), c(), cmask()],
                {},
                _onednn_graph_extra_check,
                False,
            ),
            (
                _sfdp_pattern_19,
                _sfdp_replacement_19,
                [g(), g(), g(), c(), c(), cmask_q_post_permute()],
                {},
                _onednn_graph_extra_check,
                False,
            ),
        ]

        for (
            pattern,
            replacement,
            args,
            workaround,
            extra_check,
            register_training_pattern,
        ) in candidates:
            # XXX: when adding a new pattern, re-run `gen_attention_patterns` so the pattern
            # gets serialized to a python file and does not require tracing at runtime.
            assert isinstance(workaround, dict)
            name = pattern.__name__

            if dtype != torch.float:
                name += "_half"

            if register_training_pattern:
                training_name = name + "_training"
                yield training_name, {
                    "search_fn": pattern,
                    "replace_fn": replacement,
                    "example_inputs": args,
                    "trace_fn": joint_fwd_bwd,
                    "pass_dicts": patterns[0],
                    "extra_check": extra_check,
                    "scalar_workaround": workaround,
                }

                if workaround:
                    assert len(workaround) == 1 and "dropout_p" in workaround
                    # functools.partial insufficient because we look at signature downstream
                    pattern = partialize_and_update_signature(pattern, dropout_p=0.0)
                    replacement = partialize_and_update_signature(
                        replacement, dropout_p=0.0
                    )
                    workaround = {}

            inference_name = name + "_inference"
            yield inference_name, {
                "search_fn": pattern,
                "replace_fn": replacement,
                "example_inputs": args,
                "trace_fn": fwd_only,
                "pass_dicts": patterns[0],
                "extra_check": extra_check,
                "scalar_workaround": workaround,
            }


@functools.lru_cache(None)
def _onednn_graph_sfdp_init():
    from .serialized_patterns.central_index import get_serialized_pattern

    for key, register_replacement_kwargs in _get_onednn_graph_sfdp_patterns():
        search_fn_pattern = get_serialized_pattern(key)
        register_replacement(
            **register_replacement_kwargs, search_fn_pattern=search_fn_pattern
        )
