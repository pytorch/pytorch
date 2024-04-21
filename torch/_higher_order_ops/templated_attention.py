from typing import Callable, Tuple

import torch
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
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class TemplatedAttentionHOP(HigherOrderOperator):
    def __init__(self):
        super().__init__("templated_attention")

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: Callable,
        *other_buffers: torch.Tensor,
    ):
        if not all(isinstance(buf, torch.Tensor) for buf in other_buffers):
            raise RuntimeError("Other buffers must be tensors.")
        return super().__call__(query, key, value, score_mod, *other_buffers)


templated_attention = TemplatedAttentionHOP()
templated_attention.__module__ = "torch.ops.higher_order"


def math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager implementation

    This implementation uses vmap to vectorize the score_mod function over the batch, head, m, and n dimensions.
    We then apply the vectorized score_mod function to the scores matrix. Each wrap of vmap applies one of the
    batch, head, m, or n dimensions. We need to apply vmap 4 times to vectorized over all 4 dimensions.

    Args:
        query: The query tensor
        key: The key tensor
        value: The value tensor
        score_mod: The score_mod function
        other_buffers: Other buffers that are passed to the score_mod function
    """
    assert len(other_buffers) == 0, "Other buffers are not yet supported."

    scores = (query @ key.transpose(-2, -1)).to(dtype=torch.float32)

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    in_dim_buffers = (None,) * len(other_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None) + in_dim_buffers)

    scores = score_mod(scores, b, h, m, n, *other_buffers).to(torch.float32)

    # TODO Unconditionally return logsumexp for backwards
    # if any(t.requires_grad for t in (query, key, value)):
    logsumexp = scores.logsumexp(dim=-1)

    scores = scores.softmax(dim=-1)

    return scores.to(query.dtype) @ value, logsumexp


@templated_attention.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = math_attention(query, key, value, score_mod, *other_buffers)
    out = out.contiguous()
    return out, lse


# TODO We need to implement an autograd function for this, there is some complexity to do this generically
templated_attention.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(templated_attention, deferred_error=True)
)


def trace_templated_attention(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Traces the templated_attention operator with the given score_mod function and other_buffers.

    Trace SDPA will call make_fx with "fake" example vals and then trace the score_mod function
    This will produce a GraphModule that will be stored on the root tracer as "sdpa_score". We
    access this graph module in inductor to inline the score_mod function to the triton template.
    """
    example_out = templated_attention(query, key, value, score_mod, *other_buffers)

    example_vals = [
        torch.zeros((), dtype=query.dtype, requires_grad=query.requires_grad)
    ] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
    score_graph = make_fx(score_mod)(*example_vals, *other_buffers)
    proxy_mode.tracer.root.register_module("sdpa_score", score_graph)
    node_args = (query, key, value, score_graph, *other_buffers)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", templated_attention, proxy_args, {}, name="templated_attention"
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@templated_attention.py_impl(ProxyTorchDispatchMode)
def templated_attention_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert mode is not None, "Mode should always be enabled for python fallback key"
    if mode.enable_tracing:
        return trace_templated_attention(
            mode, query, key, value, score_mod, *other_buffers
        )
    else:
        return templated_attention(query, key, value, score_mod, *other_buffers)


@templated_attention.py_functionalize_impl
def templated_attention_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Defines the functionalization rules for the templated_attention operator.

    Write now we are unwrapping each tensor and then redispatching to the next, however we want to
    guard against any mutations in the score_mod function, to the other_buffers since those
    are free variables.
    """
    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    other_buffers_unwrapped = ctx.unwrap_tensors(other_buffers)

    # Appease the mypy overlords
    assert isinstance(query_unwrapped, torch.Tensor)
    assert isinstance(key_unwrapped, torch.Tensor)
    assert isinstance(value_unwrapped, torch.Tensor)
    assert isinstance(other_buffers_unwrapped, tuple)
    assert all(isinstance(item, torch.Tensor) for item in other_buffers_unwrapped)

    example_vals = (
        [torch.zeros((), dtype=query.dtype)]
        + [torch.zeros((), dtype=torch.int) for _ in range(4)]
        + list(other_buffers_unwrapped)
    )
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

        out = templated_attention(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            functional_score_mod,
            *other_buffers_unwrapped,
        )
    return ctx.wrap_tensors(out)  # type: ignore[return-value]


@templated_attention.py_impl(FakeTensorMode)
def templated_attention_fake_tensor_mode(
    mode: FakeTensorMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    with mode:
        batch_size, num_heads, seq_len_q, _ = query.shape
        logsumexp = query.new_empty(
            batch_size, num_heads, seq_len_q, dtype=torch.float32
        )
        return torch.empty_like(query, memory_format=torch.contiguous_format), logsumexp
