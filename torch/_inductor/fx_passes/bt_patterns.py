import functools

from bangtransformer.torch import bt_ops

import torch
from torch._inductor.pattern_matcher import fwd_only, register_replacement


# 该函数用来进行参数检查，当前跳过
def _bt_extra_check(*args, **kwargs):
    def fn(match):
        return True

    return fn


# search pattern，从demo中匹配到的pattern，对应FusedFeedForward中的forward()函数
# 对于torch.nn.module这类函数，需要将其转换为torch.nn.functional函数
def _bt_pattern_1(
    x,
    up_linear_weight,
    up_linear_bias,
    down_linear_weight,
    down_linear_bias,
    gated_linear_weight,
    gated_linear_bias,
    layernorm_weight,
    layernorm_bias,
    layernorm_eps,
    alpha,
    beta,
):
    normalized_shape = tuple(layernorm_weight.size())
    norm_x = torch.nn.functional.layer_norm(
        x, normalized_shape, layernorm_weight, layernorm_bias, layernorm_eps
    )
    return torch.add(
        torch.nn.functional.linear(
            torch.mul(
                torch.nn.functional.relu(
                    torch.nn.functional.linear(
                        norm_x, up_linear_weight, up_linear_bias
                    ).float()
                ).to(norm_x.dtype),
                torch.nn.functional.linear(
                    norm_x,
                    gated_linear_weight,
                    gated_linear_bias,
                ),
            ),
            down_linear_weight,
            down_linear_bias,
        ).mul(alpha),
        x.mul(beta),
    )


# replacement pattern，替换的pattern，对应bt_ops.fused_norm_residual_ffn
def _bt_replacement_1(
    x,
    up_linear_weight,
    up_linear_bias,
    down_linear_weight,
    down_linear_bias,
    gated_linear_weight,
    gated_linear_bias,
    layernorm_weight,
    layernorm_bias,
    layernorm_eps,
    alpha,
    beta,
):
    return bt_ops.fused_norm_residual_ffn(
        x,
        up_linear_weight,
        up_linear_bias,
        down_linear_weight,
        down_linear_bias,
        gated_linear_weight,
        gated_linear_bias,
        layernorm_weight,
        layernorm_bias,
        layernorm_eps,
        "relu",
        "input",
        alpha,
        beta,
    )


def _get_bt_patterns():
    from torch._inductor.fx_passes.joint_graph import patterns

    if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
        device = "cuda"
    else:
        device = "cpu"

    batch, seq_len, hidden_size, inner_size = 4, 16, 512, 512
    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    g_inp = functools.partial(torch.empty, (batch, seq_len, hidden_size), device=device)
    layer_norm_weight_inp = functools.partial(
        torch.empty,
        (hidden_size),
        device=device,
    )
    layer_norm_bias_inp = functools.partial(
        torch.empty,
        (hidden_size),
        device=device,
    )
    linear_weight_inp = functools.partial(
        torch.empty,
        (hidden_size, inner_size),
        device=device,
    )
    linear_bias_inp = functools.partial(
        torch.empty,
        (hidden_size),
        device=device,
    )
    # workaround https://github.com/pytorch/pytorch/issues/97894
    # 0.5/0.3/1e-5 is a "magic" value that lets us recover the lost input arg relationship
    d = {"alpha": 0.5, "beta": 0.3, "layernorm_eps": 1e-5}

    for dtype in [torch.float, torch.half]:
        g = functools.partial(g_inp, dtype=dtype)
        lnw = functools.partial(layer_norm_weight_inp, dtype=dtype)
        lnb = functools.partial(layer_norm_bias_inp, dtype=dtype)
        lw = functools.partial(linear_weight_inp, dtype=dtype)
        lb = functools.partial(linear_bias_inp, dtype=dtype)

        candidates = [
            (
                _bt_pattern_1,
                _bt_replacement_1,
                [g(), lw(), lb(), lw(), lb(), lw(), lb(), lnw(), lnb()],
                d,
                _bt_extra_check(None),
            ),
        ]

        for pattern, replacement, args, workaround, extra_check in candidates:
            # XXX: when adding a new pattern, re-run `gen_attention_patterns` so the pattern
            # gets serialized to a python file and does not require tracing at runtime.
            assert isinstance(workaround, dict)
            name = pattern.__name__

            if dtype != torch.float:
                name += "_half"

            inference_name = name + "_inference"
            yield inference_name, {
                "search_fn": pattern,
                "replace_fn": replacement,
                "example_inputs": args,
                "trace_fn": fwd_only,
                "pass_dicts": patterns,
                "extra_check": extra_check,
                "scalar_workaround": workaround,
            }


@functools.lru_cache(None)
def _bt_init():
    from torch._inductor.fx_passes.serialized_patterns.central_index import (
        get_serialized_pattern,
    )

    for key, register_replacement_kwargs in _get_bt_patterns():
        search_fn_pattern = get_serialized_pattern(key)
        register_replacement(
            **register_replacement_kwargs, search_fn_pattern=search_fn_pattern
        )
