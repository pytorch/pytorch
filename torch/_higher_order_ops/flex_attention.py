# mypy: allow-untyped-defs
from typing import Any, Callable, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_mutation,
    autograd_not_implemented,
    reenter_make_fx,
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


def transform_getitem_args(x: torch.Tensor, index_args) -> Tuple[Any, ...]:
    if isinstance(index_args, tuple):
        return (x, list(index_args))
    elif not isinstance(index_args, (list, tuple)):
        return (x, [index_args])
    return (x, index_args)


class TransformGetItemToIndex(TorchFunctionMode):
    # This is needed since we want to support calling
    # A[q_idx], where q_idx is a scalar tensor in score_mod.
    # Today, when q_idx is a scalar tensor, we implicitly convert it to a python
    # scalar and create a view. We do not want that behavior in this case, so we
    # use this torchfunctionmode to override that behavior for score_mod
    # wherever we're running it.
    def __torch_function__(self, func, types, args, kwargs=None):
        if func == torch.Tensor.__getitem__:
            return torch.ops.aten.index(*transform_getitem_args(*args))
        return func(*args, **(kwargs or {}))


class FlexAttentionHOP(HigherOrderOperator):
    def __init__(self):
        super().__init__("flex_attention")

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: Callable,
        sparse_kv_num_blocks: torch.Tensor,
        sparse_kv_indices: torch.Tensor,
        sparse_q_num_blocks: torch.Tensor,
        sparse_q_indices: torch.Tensor,
        SPARSE_KV_BLOCK_SIZE: int,
        SPARSE_Q_BLOCK_SIZE: int,
        *other_buffers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not all(isinstance(buf, torch.Tensor) for buf in other_buffers):
            raise RuntimeError("Other buffers must be tensors.")
        return super().__call__(
            query,
            key,
            value,
            score_mod,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers,
        )


flex_attention = FlexAttentionHOP()
flex_attention.__module__ = "torch.ops.higher_order"


class FlexAttentionBackwardHOP(HigherOrderOperator):
    def __init__(self):
        super().__init__("flex_attention_backward")

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        logsumexp: torch.Tensor,
        grad_out: torch.Tensor,
        fw_graph: Union[Callable, GraphModule],
        joint_graph: GraphModule,
        sparse_kv_num_blocks: torch.Tensor,
        sparse_kv_indices: torch.Tensor,
        sparse_q_num_blocks: torch.Tensor,
        sparse_q_indices: torch.Tensor,
        SPARSE_KV_BLOCK_SIZE: int,
        SPARSE_Q_BLOCK_SIZE: int,
        *other_buffers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not all(isinstance(buf, torch.Tensor) for buf in other_buffers):
            raise RuntimeError("Other buffers must be tensors.")
        return super().__call__(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            fw_graph,
            joint_graph,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers,
        )


flex_attention_backward = FlexAttentionBackwardHOP()
flex_attention_backward.__module__ = "torch.ops.higher_order"


def math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
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
    working_precision = torch.float64 if query.dtype == torch.float64 else torch.float32

    scores = (query @ key.transpose(-2, -1)).to(dtype=working_precision)

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    in_dim_buffers = (None,) * len(other_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None) + in_dim_buffers)

    # todo: We wouldn't need these overrides in this file if Dynamo always did the
    # rewriting.
    with TransformGetItemToIndex():
        scores = score_mod(scores, b, h, m, n, *other_buffers).to(working_precision)

    # TODO Unconditionally return logsumexp for backwards
    # if any(t.requires_grad for t in (query, key, value)):
    logsumexp = scores.logsumexp(dim=-1)

    scores = scores.softmax(dim=-1)

    return scores.to(query.dtype) @ value, logsumexp


@flex_attention.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = math_attention(
        query,
        key,
        value,
        score_mod,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
        SPARSE_KV_BLOCK_SIZE,
        SPARSE_Q_BLOCK_SIZE,
        *other_buffers,
    )
    out = out.contiguous()
    return out, lse


def trace_flex_attention(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Traces the flex_attention operator with the given score_mod function and other_buffers.

    Trace SDPA will call make_fx with "fake" example vals and then trace the score_mod function
    This will produce a GraphModule that will be stored on the root tracer as "sdpa_score". We
    access this graph module in inductor to inline the score_mod function to the triton template.
    """
    example_out = flex_attention(
        query,
        key,
        value,
        score_mod,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
        SPARSE_KV_BLOCK_SIZE,
        SPARSE_Q_BLOCK_SIZE,
        *other_buffers,
    )
    example_vals = [
        torch.zeros((), dtype=query.dtype, requires_grad=query.requires_grad)
    ] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
    with TransformGetItemToIndex():
        score_graph = reenter_make_fx(score_mod)(*example_vals, *other_buffers)
    qualname = proxy_mode.tracer.get_fresh_qualname("sdpa_score")
    proxy_mode.tracer.root.register_module(qualname, score_graph)
    node_args = (
        query,
        key,
        value,
        score_graph,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
        SPARSE_KV_BLOCK_SIZE,
        SPARSE_Q_BLOCK_SIZE,
        *other_buffers,
    )
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", flex_attention, proxy_args, {}
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@flex_attention.py_impl(ProxyTorchDispatchMode)
def flex_attention_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert mode is not None, "Mode should always be enabled for python fallback key"
    if mode.enable_tracing:
        return trace_flex_attention(
            mode,
            query,
            key,
            value,
            score_mod,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers,
        )
    else:
        return flex_attention(
            query,
            key,
            value,
            score_mod,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers,
        )


@flex_attention.py_functionalize_impl
def flex_attention_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Defines the functionalization rules for the flex_attention operator.

    Write now we are unwrapping each tensor and then redispatching to the next, however we want to
    guard against any mutations in the score_mod function, to the other_buffers since those
    are free variables.
    """
    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    sparse_kv_num_blocks_unwrapped = ctx.unwrap_tensors(sparse_kv_num_blocks)
    sparse_kv_indices_unwrapped = ctx.unwrap_tensors(sparse_kv_indices)
    sparse_q_num_blocks_unwrapped = ctx.unwrap_tensors(sparse_q_num_blocks)
    sparse_q_indices_unwrapped = ctx.unwrap_tensors(sparse_q_indices)
    other_buffers_unwrapped = ctx.unwrap_tensors(other_buffers)

    # Appease the mypy overlords
    assert isinstance(query_unwrapped, torch.Tensor)
    assert isinstance(key_unwrapped, torch.Tensor)
    assert isinstance(value_unwrapped, torch.Tensor)
    assert isinstance(sparse_kv_num_blocks_unwrapped, torch.Tensor)
    assert isinstance(sparse_kv_indices_unwrapped, torch.Tensor)
    assert isinstance(sparse_q_num_blocks_unwrapped, torch.Tensor)
    assert isinstance(sparse_q_indices_unwrapped, torch.Tensor)
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

        out = flex_attention(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            functional_score_mod,
            sparse_kv_num_blocks_unwrapped,
            sparse_kv_indices_unwrapped,
            sparse_q_num_blocks_unwrapped,
            sparse_q_indices_unwrapped,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers_unwrapped,
        )
    return ctx.wrap_tensors(out)  # type: ignore[return-value, arg-type]


@flex_attention.py_impl(FakeTensorMode)
def flex_attention_fake_tensor_mode(
    mode: FakeTensorMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    with mode:
        batch_size, num_heads, seq_len_q, _ = query.shape
        logsumexp = query.new_empty(
            batch_size, num_heads, seq_len_q, dtype=torch.float32
        )
        return torch.empty_like(query), logsumexp


# ---------------------------- Autograd Implementation ----------------------------
def create_fw_bw_graph(score_mod, index_values, other_buffers):
    # See Note:[HOP create fw_bw graph]

    # All of these imports need to be here in order to avoid circular dependencies
    from torch._dispatch.python import suspend_functionalization
    from torch._functorch.aot_autograd import AOTConfig, create_joint
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():

            def _from_fun(t):
                return torch.empty_strided(
                    t.size(),
                    t.stride(),
                    device=t.device,
                    dtype=t.dtype,
                    requires_grad=t.requires_grad,
                )

            # If someone runs this hop under the default compiler backend ("eager")
            # Then this path will be run with the actual user inputs. We convert them
            # to fake tensors in order to not perform any actual compute.
            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(index_values)
            if fake_mode is None:
                fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

            with fake_mode:
                unwrapped_score_mod_indexes = pytree.tree_map(_from_fun, index_values)
                unwrapped_other_buffers = pytree.tree_map(_from_fun, other_buffers)

            assert all(isinstance(t, FakeTensor) for t in unwrapped_score_mod_indexes)
            assert all(isinstance(t, FakeTensor) for t in unwrapped_other_buffers)

            example_flat_out = pytree.tree_map(
                _from_fun,
                score_mod(*unwrapped_score_mod_indexes, *unwrapped_other_buffers),
            )
            if not isinstance(example_flat_out, torch.Tensor):
                raise RuntimeError(
                    "Expected output of score_mod to be a tensor."
                    f"Got type {type(example_flat_out)}."
                )
            example_grad = _from_fun(example_flat_out)

        def joint_f(score, b, h, m, n, example_grad, *other_buffers):
            def fw_with_masks(*args):
                fw_out = score_mod(*args)
                out_requires_grad = fw_out.requires_grad
                return ((fw_out,), (out_requires_grad,))

            joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)
            args = [score, b, h, m, n] + list(other_buffers)
            optional_grad = [example_grad] if example_grad.requires_grad else []
            _, grads = joint(args, optional_grad)

            return grads

        joint_graph = make_fx(joint_f)(
            *unwrapped_score_mod_indexes, example_grad, *unwrapped_other_buffers
        )
        return score_mod, joint_graph


class FlexAttentionAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        fw_graph,
        joint_graph,
        sparse_kv_num_blocks: torch.Tensor,
        sparse_kv_indices: torch.Tensor,
        sparse_q_num_blocks: torch.Tensor,
        sparse_q_indices: torch.Tensor,
        SPARSE_KV_BLOCK_SIZE: int,
        SPARSE_Q_BLOCK_SIZE: int,
        *other_buffers,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        any_buffer_requires_grad = any(buffer.requires_grad for buffer in other_buffers)
        assert (
            not any_buffer_requires_grad
        ), "Captured buffers that require grad are not yet supported."
        ctx._fw_graph = fw_graph
        ctx._joint_graph = joint_graph
        ctx._SPARSE_KV_BLOCK_SIZE = SPARSE_KV_BLOCK_SIZE
        ctx._SPARSE_Q_BLOCK_SIZE = SPARSE_Q_BLOCK_SIZE
        with torch._C._AutoDispatchBelowAutograd():
            out, logsumexp = flex_attention(
                query,
                key,
                value,
                fw_graph,
                sparse_kv_num_blocks,
                sparse_kv_indices,
                sparse_q_num_blocks,
                sparse_q_indices,
                SPARSE_KV_BLOCK_SIZE,
                SPARSE_Q_BLOCK_SIZE,
                *other_buffers,
            )

        ctx.save_for_backward(
            query,
            key,
            value,
            out,
            logsumexp,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            *other_buffers,
        )
        return out, logsumexp

    @staticmethod
    def backward(ctx, grad_out, logsumexp_grad):
        fw_args = ctx.saved_tensors
        (
            query,
            key,
            value,
            out,
            logsumexp,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            *other_buffers,
        ) = fw_args
        fw_graph = ctx._fw_graph
        joint_graph = ctx._joint_graph
        SPARSE_KV_BLOCK_SIZE = ctx._SPARSE_KV_BLOCK_SIZE
        SPARSE_Q_BLOCK_SIZE = ctx._SPARSE_Q_BLOCK_SIZE
        # We have asserted that other_buffers do not require grad in the forward
        none_grads = [None] * (8 + len(other_buffers))
        grad_query, grad_key, grad_value = flex_attention_backward(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            fw_graph,
            joint_graph,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers,
        )
        return grad_query, grad_key, grad_value, *none_grads


@flex_attention.py_impl(DispatchKey.Autograd)
def flex_attention_autograd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    with TransformGetItemToIndex():
        input_requires_grad = any(t.requires_grad for t in (query, key, value))
        if torch.is_grad_enabled() and input_requires_grad:
            example_vals = [
                torch.zeros((), dtype=query.dtype, requires_grad=input_requires_grad)
            ] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
            fw_graph, bw_graph = create_fw_bw_graph(
                score_mod, example_vals, other_buffers
            )
        else:
            fw_graph, bw_graph = score_mod, None
        out, logsumexp = FlexAttentionAutogradOp.apply(
            query,
            key,
            value,
            fw_graph,
            bw_graph,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers,
        )
    return out, logsumexp


# ---------------------------- Backward HOP Implementation ----------------------------


@flex_attention_backward.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    fw_graph: Callable,  # GraphModule type hint?
    joint_graph: Callable,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    working_precision = torch.float64 if query.dtype == torch.float64 else torch.float32
    scores = (query @ key.transpose(-2, -1)).to(working_precision)

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    in_dim_buffers = (None,) * len(other_buffers)
    score_mod = torch.vmap(fw_graph, in_dims=(0, None, None, None, 0) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None) + in_dim_buffers)

    with TransformGetItemToIndex():
        post_mod_scores = score_mod(scores, b, h, m, n, *other_buffers).to(
            working_precision
        )

    softmax_scores = torch.exp(post_mod_scores - logsumexp.unsqueeze(-1))

    grad_value = softmax_scores.to(query.dtype).transpose(-2, -1) @ grad_out

    grad_softmax_scores = grad_out @ value.transpose(-2, -1)

    sum_scores = torch.sum(out * grad_out, -1, keepdim=True)
    grad_score_mod = softmax_scores * (grad_softmax_scores - sum_scores)

    # Gradient of the inline score_mod function, with respect to the scores
    in_dim_buffers = (None,) * len(other_buffers)
    out_dims = [0, None, None, None, None] + [None] * len(other_buffers)
    joint_score_mod = torch.vmap(
        joint_graph,
        in_dims=(0, None, None, None, 0, 0) + in_dim_buffers,
        out_dims=out_dims,
    )
    joint_score_mod = torch.vmap(
        joint_score_mod,
        in_dims=(0, None, None, 0, None, 0) + in_dim_buffers,
        out_dims=out_dims,
    )
    joint_score_mod = torch.vmap(
        joint_score_mod,
        in_dims=(0, None, 0, None, None, 0) + in_dim_buffers,
        out_dims=out_dims,
    )
    joint_score_mod = torch.vmap(
        joint_score_mod,
        in_dims=(0, 0, None, None, None, 0) + in_dim_buffers,
        out_dims=out_dims,
    )
    with TransformGetItemToIndex():
        grad_scores, *_ = joint_score_mod(
            scores, b, h, m, n, grad_score_mod, *other_buffers
        )
    grad_scores = grad_scores.to(query.dtype)

    grad_query = grad_scores @ key
    grad_key = grad_scores.transpose(-2, -1) @ query
    return grad_query.contiguous(), grad_key.contiguous(), grad_value.contiguous()


def trace_flex_attention_backward(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """We already have the forward graph and joint graph from the forward pass, so we create a proxy attach both graphs"""
    example_out = flex_attention_backward(
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        fw_graph,
        joint_graph,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
        SPARSE_KV_BLOCK_SIZE,
        SPARSE_Q_BLOCK_SIZE,
        *other_buffers,
    )

    fw_example_vals = [
        torch.zeros((), dtype=query.dtype, requires_grad=query.requires_grad)
    ] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
    bw_example_vals = fw_example_vals + [torch.zeros((), dtype=query.dtype)]
    with TransformGetItemToIndex():
        fw_graph = reenter_make_fx(fw_graph)(*fw_example_vals, *other_buffers)
        joint_graph = reenter_make_fx(joint_graph)(*bw_example_vals, *other_buffers)
    proxy_mode.tracer.root.register_module("fw_graph", fw_graph)
    proxy_mode.tracer.root.register_module("joint_graph", joint_graph)
    node_args = (
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        fw_graph,
        joint_graph,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
        SPARSE_KV_BLOCK_SIZE,
        SPARSE_Q_BLOCK_SIZE,
        *other_buffers,
    )
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        flex_attention_backward,
        proxy_args,
        {},
        name="flex_attention_backward",
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@flex_attention_backward.py_impl(ProxyTorchDispatchMode)
def flex_attention_backward_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE,
    SPARSE_Q_BLOCK_SIZE,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mode is not None, "Mode should always be enabled for python fallback key"
    if mode.enable_tracing:
        return trace_flex_attention_backward(
            mode,
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            fw_graph,
            joint_graph,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers,
        )
    else:
        return flex_attention_backward(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            fw_graph,
            joint_graph,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers,
        )


@flex_attention_backward.py_functionalize_impl
def flex_attention_backward_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Defines the functionalization rules for the flex_attention operator.

    Write now we are unwrapping each tensor and then redispatching to the next,
    since we know that the forward score mod function is assured to be free of mutations
    to the other_buffers, we skip that mutate check and go straight to redispatching.
    """
    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    out_unwrapped = ctx.unwrap_tensors(out)
    logsumexp_unwrapped = ctx.unwrap_tensors(logsumexp)
    grad_out_unwrapped = ctx.unwrap_tensors(grad_out)
    sparse_kv_num_blocks_unwrapped = ctx.unwrap_tensors(sparse_kv_num_blocks)
    sparse_kv_indices_unwrapped = ctx.unwrap_tensors(sparse_kv_indices)
    sparse_q_num_blocks_unwrapped = ctx.unwrap_tensors(sparse_q_num_blocks)
    sparse_q_indices_unwrapped = ctx.unwrap_tensors(sparse_q_indices)
    other_buffers_unwrapped = ctx.unwrap_tensors(other_buffers)

    # Appease the mypy overlords
    assert isinstance(query_unwrapped, torch.Tensor)
    assert isinstance(key_unwrapped, torch.Tensor)
    assert isinstance(value_unwrapped, torch.Tensor)
    assert isinstance(out_unwrapped, torch.Tensor)
    assert isinstance(logsumexp_unwrapped, torch.Tensor)
    assert isinstance(grad_out_unwrapped, torch.Tensor)
    assert isinstance(sparse_kv_num_blocks_unwrapped, torch.Tensor)
    assert isinstance(sparse_kv_indices_unwrapped, torch.Tensor)
    assert isinstance(sparse_q_num_blocks_unwrapped, torch.Tensor)
    assert isinstance(sparse_q_indices_unwrapped, torch.Tensor)
    assert isinstance(other_buffers_unwrapped, tuple)
    assert all(isinstance(item, torch.Tensor) for item in other_buffers_unwrapped)

    with ctx.redispatch_to_next() as m:
        functional_fw_graph = ctx.functionalize(fw_graph)
        functional_joint_graph = ctx.functionalize(joint_graph)

        grad_query, grad_key, grad_value = flex_attention_backward(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            out_unwrapped,
            logsumexp_unwrapped,
            grad_out_unwrapped,
            functional_fw_graph,  # type: ignore[arg-type]
            functional_joint_graph,  # type: ignore[arg-type]
            sparse_kv_num_blocks_unwrapped,
            sparse_kv_indices_unwrapped,
            sparse_q_num_blocks_unwrapped,
            sparse_q_indices_unwrapped,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers_unwrapped,
        )

    return ctx.wrap_tensors((grad_query, grad_key, grad_value))  # type: ignore[return-value,arg-type]


@flex_attention_backward.py_impl(FakeTensorMode)
def flex_attention_backward_fake_tensor_mode(
    mode: FakeTensorMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with mode:
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        return grad_query, grad_key, grad_value


flex_attention_backward.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(flex_attention_backward, deferred_error=True)
)
