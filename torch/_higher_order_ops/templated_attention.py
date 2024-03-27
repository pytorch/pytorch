from typing import Callable

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_mutation,
    autograd_not_implemented,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

sdpa = HigherOrderOperator("templated_attention")


def math_attention_vmap(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
):
    scores = query @ key.transpose(-2, -1)

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    score_mod_vmap = score_mod
    score_mod_vmap = torch.vmap(score_mod_vmap, in_dims=(0, None, None, None, 0))
    score_mod_vmap = torch.vmap(score_mod_vmap, in_dims=(0, None, None, 0, None))
    score_mod_vmap = torch.vmap(score_mod_vmap, in_dims=(0, None, 0, None, None))
    score_mod_vmap = torch.vmap(score_mod_vmap, in_dims=(0, 0, None, None, None))

    scores = score_mod_vmap(scores, b, h, m, n, *other_buffers)

    scores = scores.softmax(dim=-1)
    return scores @ value


def math_attention_2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
):
    scores = query @ key.transpose(-2, -1)

    b = torch.arange(0, scores.size(0), device=scores.device).view(-1, 1, 1, 1)
    h = torch.arange(0, scores.size(1), device=scores.device).view(1, -1, 1, 1)
    m = torch.arange(0, scores.size(2), device=scores.device).view(1, 1, -1, 1)
    n = torch.arange(0, scores.size(3), device=scores.device).view(1, 1, 1, -1)

    scores = score_mod(scores, b, h, m, n, *other_buffers)

    scores = scores.softmax(dim=-1)
    return scores @ value


def math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
):
    scores = query @ key.transpose(-2, -1)

    from functorch.dim import dims

    b, h, m, n = dims()

    scores = scores[b, h, m, n]
    scores = score_mod(scores, b, h, m, n, *other_buffers)
    scores = scores.order(b, h, m, n)
    scores = scores.softmax(dim=-1)
    return scores @ value


@sdpa.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
):
    # TODO using the vmap version currently throws error in validate_and_convert_non_fake_tensors
    return math_attention_vmap(
        query, key, value, score_mod, *other_buffers
    ).contiguous()
    # return math_attention_2(query, key, value, score_mod, *other_buffers).contiguous()


# TODO For now lets disable but we need to figure this out
# TODO DOUBLE TODO I should be able to call autograd_not_implemented with sdpa, not sure why this fails functional tensor mode
sdpa.py_impl(DispatchKey.Autograd)(autograd_not_implemented(sdpa, deferred_error=False))


def trace_sdpa(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
):
    if score_mod is None:
        with proxy_mode:
            return F.scaled_dot_product_attention(query, key, value)

    with disable_proxy_modes_tracing():
        example_out = F.scaled_dot_product_attention(query, key, value)
    example_vals = [torch.zeros((), dtype=query.dtype)] + [
        torch.zeros((), dtype=torch.int) for _ in range(4)
    ]
    score_graph = make_fx(score_mod)(*example_vals, *other_buffers)
    proxy_mode.tracer.root.register_module("sdpa_score", score_graph)
    node_args = (query, key, value, score_graph, *other_buffers)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", sdpa, proxy_args, {}, name="templated_attention"
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@sdpa.py_impl(ProxyTorchDispatchMode)
def sdpa_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
):
    assert mode is not None, "Mode should always be enabled for python fallback key"
    if mode.enable_tracing:
        return trace_sdpa(mode, query, key, value, score_mod, *other_buffers)
    else:
        return sdpa(query, key, value, score_mod, *other_buffers)


@sdpa.py_functionalize_impl
def sdpa_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
):
    """Defines the functionalization rules for the sdpa operator.

    Write now we are unwrapping each tensor and then redispatching to the next, however we want to
    guard against any mutations or aliases in the score_mod function. To the other_buffers since those
    are free variables.
    """
    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    other_buffers_unwrapped = ctx.unwrap_tensors(other_buffers)
    example_vals = [torch.zeros((), dtype=query.dtype)] + [
        torch.zeros((), dtype=torch.int) for _ in range(4)
    ]
    with ctx.redispatch_to_next() as m:
        functional_score_mod = ctx.functionalize(score_mod)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        mutates = _has_potential_branch_input_mutation(
            functional_score_mod, example_vals, pre_dispatch
        )
        # The only care about mutations of existing buffers since we can't replay these.
        # However, we can just error if anything is detected
        if mutates:
            raise UnsupportedAliasMutationException("Mutations detected in score_mod")

        out = sdpa(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            functional_score_mod,
            *other_buffers_unwrapped,
        )
    return ctx.wrap_tensors(out)


@sdpa.py_impl(FakeTensorMode)
def sdpa_fake_tensor_mode(
    mode: FakeTensorMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
):
    with mode:
        return torch.empty_like(query, memory_format=torch.contiguous_format)
    # return sdpa_dense(query, key, value, score_mod, *other_buffers)
