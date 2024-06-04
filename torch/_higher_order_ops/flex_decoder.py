from typing import Any, Callable, Tuple, Union

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
from torch.fx.graph_module import GraphModule

from torch.overrides import TorchFunctionMode


"""Import flex attention common utils"""
from torch._higher_order_ops.flex_attention import (
    transform_getitem_args,
    TransformGetItemToIndex
)

class FlexDecoderHOP(HigherOrderOperator):
    def __init__(self):
        super().__init__("flex_decoder")

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: Callable,
        *other_buffers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not all(isinstance(buf, torch.Tensor) for buf in other_buffers):
            raise RuntimeError("Other buffers must be tensors.")
        return super().__call__(query, key, value, score_mod, *other_buffers)


flex_decoder = FlexDecoderHOP()
flex_decoder.__module__ = "torch.ops.higher_order"


def math_decoder(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
    Bc: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager implementation

    Flex decoder is a special case of flex attention where the key and value tensors are tiled along the N dimension.
    This implementation uses matmul to broadcast the query over key and values tiles along the N dimension.

    This implementation uses vmap to vectorize the score_mod function over the batch, head, tile, m, and Bc dimensions.
    We then apply the vectorized score_mod function to the scores matrix. Each wrap of vmap applies one of the
    batch, head, tile, query, or Bc dimensions. We need to apply vmap 5 times to vectorized over all 5 dimensions.

    This implementation uses partial rowsum, where each tile uses its local rowmax for softmax stability.

    Args:
        query: The query tensor
        key: The key tensor
        value: The value tensor
        score_mod: The score_mod function
        other_buffers: Other buffers that are passed to the score_mod function
        Rc: key/value tile size along N dimension
    """
    working_precision = torch.float64 if query.dtype == torch.float64 else torch.float32

    # Break key and value tensors into tiles along m dimension.
    query = query.view(query.shape[0], query.shape[1], 1, query.shape[2], -1)
    key = key.view(key.shape[0], key.shape[1], key.shape[2]//Bc, Bc, -1)
    value = value.view(value.shape[0], value.shape[1], value.shape[2]//Bc, Bc, -1)

    scores = (query @ key.transpose(-2, -1)).to(dtype=working_precision)

    B = torch.arange(0, scores.size(0), device=scores.device) # B
    H = torch.arange(0, scores.size(1), device=scores.device) # H
    Q = torch.arange(0, scores.size(-2), device=scores.device) # m
    KV = torch.arange(0, scores.size(2)*scores.size(-1), device=scores.device).view(scores.size(2), scores.size(-1)) # [N/Bc, Bc]

    in_dim_buffers = (None,) * len(other_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0) + in_dim_buffers)    # Bc. (Bc, [], [], [], Bc)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None) + in_dim_buffers)    # Q ([Q, Bc], [], [], Q, Bc)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0))                     # N/Bc ([N/Bc, Q, Bc], [], [], Q, [N/Bc, Bc])
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None) + in_dim_buffers)    # H ([H, N/Bc, Q, Bc], [], H, Q, [N/Bc, Bc])
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None) + in_dim_buffers)    # B ([B, H, N/Bc, Q, Bc], B, H, Q, [N/Bc, Bc])

    # todo: We wouldn't need these overrides in this file if Dynamo always did the
    # rewriting.

    # print(scores[0, 5, 0], "scores from decoder")
    with TransformGetItemToIndex():
        scores = score_mod(scores, B, H, Q, KV, *other_buffers).to(working_precision)

    # Calculate local output & sumexp based on local rowmax
    local_rowmax = torch.max(scores, dim=-1, keepdim=True).values
    score_exp = torch.exp(scores - local_rowmax).nan_to_num(nan=0) # when local_rowmax = -inf, set score_exp to 0
    local_sumexp = score_exp.sum(dim=-1)
    local_output = (score_exp.to(query.dtype) @ value)

    # print(scores[0, 5, 0], "scores after score_mod from decoder")
    # print(score_exp[0, 5, 0], "local score exp from decoder")
    # print(local_output[0, 5, 0], "local_output from decoder")
    # print(local_rowmax[0, 5, 0], "local_rowmax from decoder")


    ## Reduction: find global rowmax
    rowmax = torch.max(local_rowmax, dim=-3, keepdim=True).values

    # print(rowmax[0, 5], "global_rowmax from decoder")

    # Rebase local output and sumexp from local rowmax to global rowmax.
    rowmax_delta = (local_rowmax - rowmax)
    local_sumexp = local_sumexp*torch.exp(rowmax_delta.squeeze(-1))
    local_output = local_output*torch.exp(rowmax_delta)

    # print(local_output[0, 5, 1], "rebased local_output from decoder")


    ## Reduction: Aggregate local sumexp and output to calculate global logsumexp and output.
    sumexp = local_sumexp.sum(dim=-2)
    logsumexp = torch.log(sumexp) + rowmax.squeeze(2,4)
    output = local_output.sum(dim=-3).div(sumexp.unsqueeze(-1))

    # print(output[0, 5, 1], "Output from decoder")
    # print(logsumexp[0, 5], "Logsumexp from decoder")

    # Unconditionally return logsumexp
    return output, logsumexp


@flex_decoder.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = math_decoder(query, key, value, score_mod, *other_buffers)
    out = out.contiguous()
    return out, lse


def trace_flex_decoder(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Traces the flex_decoder operator with the given score_mod function and other_buffers.

    Trace SDPA will call make_fx with "fake" example vals and then trace the score_mod function
    This will produce a GraphModule that will be stored on the root tracer as "sdpa_score". We
    access this graph module in inductor to inline the score_mod function to the triton template.
    """
    example_out = flex_decoder(query, key, value, score_mod, *other_buffers)
    example_vals = [
        torch.zeros((), dtype=query.dtype, requires_grad=query.requires_grad)
    ] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
    with TransformGetItemToIndex():
        score_graph = make_fx(score_mod)(*example_vals, *other_buffers)
    qualname = proxy_mode.tracer.get_fresh_qualname("sdpa_score")
    proxy_mode.tracer.root.register_module(qualname, score_graph)
    node_args = (query, key, value, score_graph, *other_buffers)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", flex_decoder, proxy_args, {}
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@flex_decoder.py_impl(ProxyTorchDispatchMode)
def flex_decoder_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert mode is not None, "Mode should always be enabled for python fallback key"
    if mode.enable_tracing:
        return trace_flex_decoder(mode, query, key, value, score_mod, *other_buffers)
    else:
        return flex_decoder(query, key, value, score_mod, *other_buffers)


@flex_decoder.py_functionalize_impl
def flex_decoder_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Defines the functionalization rules for the flex_decoder operator.

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
        with TransformGetItemToIndex():
            mutates = _has_potential_branch_input_mutation(
                functional_score_mod, example_vals, pre_dispatch
            )
        # The only care about mutations of existing buffers since we can't replay these.
        # However, we can just error if anything is detected
        if mutates:
            raise UnsupportedAliasMutationException("Mutations detected in score_mod")

        out = flex_decoder(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            functional_score_mod,
            *other_buffers_unwrapped,
        )
    return ctx.wrap_tensors(out)  # type: ignore[return-value, arg-type]


@flex_decoder.py_impl(FakeTensorMode)
def flex_decoder_fake_tensor_mode(
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


# ---------------------------- Autograd Implementation ----------------------------
from torch._higher_order_ops.flex_attention import (
    create_fw_bw_graph
)

class FlexDecoderAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query, key, value, fw_graph, joint_graph, *other_buffers
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        any_buffer_requires_grad = any(buffer.requires_grad for buffer in other_buffers)
        assert (
            not any_buffer_requires_grad
        ), "Captured buffers that require grad are not yet supported."
        ctx._fw_graph = fw_graph
        ctx._joint_graph = joint_graph
        with torch._C._AutoDispatchBelowAutograd():
            out, logsumexp = flex_decoder(query, key, value, fw_graph, *other_buffers)

        ctx.save_for_backward(query, key, value, out, logsumexp, *other_buffers)
        return out, logsumexp

    # No backward path
    @staticmethod
    def backward(ctx, grad_out, logsumexp_grad):
        raise AssertionError("flex_decoder has no backward path.")


@flex_decoder.py_impl(DispatchKey.Autograd)
def flex_decoder_autograd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_requires_grad = any(t.requires_grad for t in (query, key, value))
    if torch.is_grad_enabled() and input_requires_grad:
        example_vals = [
            torch.zeros((), dtype=query.dtype, requires_grad=input_requires_grad)
        ] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
        fw_graph, bw_graph = create_fw_bw_graph(score_mod, example_vals, other_buffers)
    else:
        fw_graph, bw_graph = score_mod, None
    out, logsumexp = FlexDecoderAutogradOp.apply(
        query, key, value, fw_graph, bw_graph, *other_buffers
    )
    return out, logsumexp
