import functools
import logging

import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import fwd_only, joint_fwd_bwd, register_replacement

log = logging.getLogger(__name__)
aten = torch.ops.aten


def _woq_mm_pattern_1(x, weight, scales):
    return torch.nn.functional.linear(x, weight.to(dtype=x.dtype)) * scales


def _woq_mm_replacement_1(x, weight, scales):
    counters["inductor"]["woq_mm"] += 1
    out_features = weight.size(0)
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int8pack_mm(x, weight, scales)
    new_shape = origin_x_size[:-1] + (out_features,)
    return c.reshape(new_shape)


def _woq_mm_params_check(match):
    assert all(k in match.kwargs for k in ("x", "weight"))
    x = match.kwargs["x"].meta["val"]
    weight = match.kwargs["weight"].meta["val"]
    # For now, we only support woq mm kernels
    # with x.type=bfloat16 and w.type=int8
    return (
        x.dtype == torch.bfloat16
        and weight.dtype == torch.int8
        and x.device == weight.device
    )


def _get_woq_mm_patterns():
    from .joint_graph import patterns

    if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
        device = "cuda"
    else:
        device = "cpu"

    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    x_inp = functools.partial(
        torch.empty, (1, 1, 256), device=device, requires_grad=True
    )
    w_inp = functools.partial(torch.empty, (12, 256), device=device)
    s_inp = functools.partial(torch.empty, (12), device=device)

    x = functools.partial(x_inp, dtype=torch.bfloat16)
    w = functools.partial(w_inp, dtype=torch.int8)
    s = functools.partial(s_inp, dtype=torch.bfloat16)

    workaround: dict[str, float]

    for pattern, replacement, args, workaround, extra_check in [
        (
            _woq_mm_pattern_1,
            _woq_mm_replacement_1,
            [x(), w(), s()],
            {},
            _woq_mm_params_check,
        ),
    ]:
        assert isinstance(workaround, dict)  # mypy is unable to infer the type properly
        name = pattern.__name__

        training_name = f"{name}_training"
        yield training_name, {
            "search_fn": pattern,
            "replace_fn": replacement,
            "example_inputs": args,
            "trace_fn": joint_fwd_bwd,
            "pass_dicts": patterns,
            "extra_check": extra_check,
            "scalar_workaround": workaround,
        }

        inference_name = f"{name}_inference"
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
def _woq_mm_init():
    for key, register_replacement_kwargs in _get_woq_mm_patterns():
        register_replacement(**register_replacement_kwargs)
