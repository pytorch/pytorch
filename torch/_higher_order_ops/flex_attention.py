import math
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_mutation,
    _maybe_reenter_make_fx,
    autograd_not_implemented,
    reenter_make_fx,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
    UnsupportedAliasMutationException,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.graph_module import GraphModule


# Duplicate of _inductor/kernel/flex_attention.py to avoid circular import
def _construct_strides(
    sizes: Sequence[int],
    fill_order: Sequence[int],
) -> Sequence[int]:
    """From a list of sizes and a fill order, construct the strides of the permuted tensor."""
    # Initialize strides
    assert len(sizes) == len(
        fill_order
    ), "Length of sizes must match the length of the fill order"
    strides = [0] * len(sizes)

    # Start with stride 1 for the innermost dimension
    current_stride = 1

    # Iterate through the fill order populating strides
    for dim in fill_order:
        strides[dim] = current_stride
        current_stride *= sizes[dim]

    return strides


def _permute_strides(out: torch.Tensor, query_strides: tuple[int, ...]) -> torch.Tensor:
    """
    Create a new tensor with the same data and shape as the input,
    but with strides permuted based on the input tensor's stride order.

    Args:
        out (torch.Tensor): The output tensor of attention.
        query_strides (List[int]): The stride order of the input query tensor

    Returns:
        torch.Tensor: A new tensor with same shape and data as the input,
        but with strides permuted based on the query tensor's stride order.
    """
    from torch._inductor.ir import get_fill_order

    fill_order = get_fill_order(query_strides)
    assert out.storage_offset() == 0, "Only support storage_offset == 0"
    out_strides = _construct_strides(out.shape, fill_order)
    new_out = out.new_empty(out.shape).as_strided(out.shape, out_strides)
    new_out.copy_(out)
    return new_out


class FlexAttentionHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_attention", cacheable=True)

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: Callable,
        block_mask: tuple,
        scale: float,
        kernel_options: dict[str, Any],
        score_mod_other_buffers: tuple = (),
        mask_mod_other_buffers: tuple = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validate_subgraph_args_types(score_mod_other_buffers + mask_mod_other_buffers)
        return super().__call__(
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )


flex_attention = FlexAttentionHOP()


class FlexAttentionBackwardHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_attention_backward")

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        logsumexp: torch.Tensor,
        grad_out: torch.Tensor,
        grad_logsumexp: torch.Tensor,
        fw_graph: Union[Callable, GraphModule],
        joint_graph: GraphModule,
        block_mask: tuple,
        scale: float,
        kernel_options: dict[str, Any],
        score_mod_other_buffers: tuple = (),
        mask_mod_other_buffers: tuple = (),
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, tuple[Optional[torch.Tensor], ...]
    ]:
        validate_subgraph_args_types(score_mod_other_buffers + mask_mod_other_buffers)
        return super().__call__(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            fw_graph,
            joint_graph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )


flex_attention_backward = FlexAttentionBackwardHOP()


def _math_attention_inner(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    working_precision = torch.float64 if query.dtype == torch.float64 else torch.float32

    scores = (query @ key.transpose(-2, -1)).to(dtype=working_precision)

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    captured_buffers_in_dim = (None,) * len(score_mod_other_buffers)
    from torch.nn.attention.flex_attention import _vmap_for_bhqkv

    # first input is score
    score_mod = _vmap_for_bhqkv(score_mod, prefix=(0,), suffix=captured_buffers_in_dim)

    mask_mod = block_mask[-1]
    mask_mod_in_dim_buffers = (None,) * len(mask_mod_other_buffers)
    mask_mod = _vmap_for_bhqkv(mask_mod, prefix=(), suffix=mask_mod_in_dim_buffers)

    with TransformGetItemToIndex():
        scores = (scores * scale).to(working_precision)
        post_mod_scores = torch.where(
            mask_mod(b, h, m, n, *mask_mod_other_buffers),
            score_mod(scores, b, h, m, n, *score_mod_other_buffers),
            torch.tensor(-float("inf"), dtype=working_precision, device=scores.device),
        )

    return scores, post_mod_scores


def math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor]:
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
    # broadcast query & key along head dim for GQA
    G = query.size(1) // key.size(1)
    value = torch.repeat_interleave(value, G, dim=1)
    key = torch.repeat_interleave(key, G, dim=1)

    Bq, Bkv = query.size(0), key.size(0)
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.size()[1:]))
    value = value.expand((Bq, *value.size()[1:]))

    _, post_mod_scores = _math_attention_inner(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )

    # Set fully masked rows' sumexp to 0.0
    logsumexp = post_mod_scores.logsumexp(dim=-1)
    masked_rows = torch.all(post_mod_scores == -float("inf"), dim=-1)
    logsumexp = torch.where(masked_rows, -float("inf"), logsumexp)

    post_mod_scores = torch._safe_softmax(post_mod_scores, dim=-1)

    return post_mod_scores.to(query.dtype) @ value, logsumexp / math.log(2)


@flex_attention.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = math_attention(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    out = _permute_strides(out, query.stride())
    return out, lse


def trace_flex_attention(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Traces the flex_attention operator with the given score_mod function and other_buffers.

    Trace SDPA will call make_fx with "fake" example vals and then trace the score_mod function
    This will produce a GraphModule that will be stored on the root tracer as "sdpa_score". We
    access this graph module in inductor to inline the score_mod function to the triton template.
    """
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    example_out = flex_attention(
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    example_vals = [query.new_zeros((), requires_grad=query.requires_grad)] + [
        query.new_zeros((), dtype=torch.int) for _ in range(4)
    ]
    mask_example_vals = [query.new_zeros((), dtype=torch.int) for _ in range(4)]
    mask_mod = block_mask[-1]
    with TransformGetItemToIndex():
        score_graph = reenter_make_fx(score_mod)(
            *example_vals, *score_mod_other_buffers
        )
        mask_graph = reenter_make_fx(mask_mod)(
            *mask_example_vals, *mask_mod_other_buffers
        )
    assert isinstance(proxy_mode.tracer, torch.fx.Tracer)
    block_mask = block_mask[:-1] + (mask_graph,)
    qualname = proxy_mode.tracer.get_fresh_qualname("sdpa_score")
    proxy_mode.tracer.root.register_module(qualname, score_graph)
    mask_qualname = proxy_mode.tracer.get_fresh_qualname("sdpa_mask")
    proxy_mode.tracer.root.register_module(mask_qualname, mask_graph)
    node_args = (
        query,
        key,
        value,
        score_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
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
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mode is not None, "Mode should always be enabled for python fallback key"
    return trace_flex_attention(
        mode,
        query,
        key,
        value,
        score_mod,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )


@flex_attention.py_functionalize_impl
def flex_attention_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Defines the functionalization rules for the flex_attention operator.

    Write now we are unwrapping each tensor and then redispatching to the next, however we want to
    guard against any mutations in the score_mod function, to the other_buffers since those
    are free variables.
    """
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    block_mask_unwrapped = ctx.unwrap_tensors(block_mask)
    score_mod_other_buffers_unwrapped = ctx.unwrap_tensors(score_mod_other_buffers)
    mask_mod_other_buffers_unwrapped = ctx.unwrap_tensors(mask_mod_other_buffers)

    # Appease the mypy overlords
    assert isinstance(query_unwrapped, torch.Tensor)
    assert isinstance(key_unwrapped, torch.Tensor)
    assert isinstance(value_unwrapped, torch.Tensor)
    assert isinstance(block_mask_unwrapped, tuple)
    assert isinstance(score_mod_other_buffers_unwrapped, tuple)
    assert isinstance(mask_mod_other_buffers_unwrapped, tuple)

    example_vals = (
        [query_unwrapped.new_zeros(())]
        + [query_unwrapped.new_zeros((), dtype=torch.int) for _ in range(4)]
        + list(score_mod_other_buffers_unwrapped)
    )
    with ctx.redispatch_to_next():
        functional_score_mod = ctx.functionalize(score_mod)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        with TransformGetItemToIndex():
            mutates = _has_potential_branch_input_mutation(
                score_mod, example_vals, pre_dispatch
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
            block_mask_unwrapped,
            scale,
            kernel_options,
            score_mod_other_buffers_unwrapped,
            mask_mod_other_buffers_unwrapped,
        )
    return ctx.wrap_tensors(out)  # type: ignore[return-value, arg-type]


def flex_attention_fake_impl(
    query: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Figure out a better way to handle this for NJT than using sum()
    if query.is_nested:
        out = torch.empty_like(query, memory_format=torch.contiguous_format)
        logsumexp = query.sum(dim=-1)
        return out, logsumexp

    v_head_dim = value.size(-1)
    batch_size, num_heads, seq_len_q, _q_head_dim = query.shape
    logsumexp = query.new_empty(batch_size, num_heads, seq_len_q, dtype=torch.float32)
    out_shape = (batch_size, num_heads, seq_len_q, v_head_dim)
    out = query.new_empty(out_shape)
    out = _permute_strides(out, query.stride())
    return out, logsumexp


@flex_attention.py_impl(FakeTensorMode)
def flex_attention_fake_tensor_mode(
    mode: FakeTensorMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    with mode:
        out, logsumexp = flex_attention_fake_impl(query, value)
        return out, logsumexp


# ---------------------------- Autograd Implementation ----------------------------
def create_fw_bw_graph(
    score_mod: Callable,
    index_values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    other_buffers: tuple[Tensor, ...],
) -> tuple[Callable, Callable]:
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

            def _from_fun(
                t: Union[Tensor, torch.SymInt, int]
            ) -> Union[Tensor, torch.SymInt, int]:
                if isinstance(t, torch.Tensor):
                    return torch.empty_strided(
                        t.size(),
                        t.stride(),
                        device=t.device,
                        dtype=t.dtype,
                        requires_grad=t.requires_grad,
                    )
                return t

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

            assert all(
                isinstance(t, (FakeTensor, int, torch.SymInt))
                for t in unwrapped_score_mod_indexes + unwrapped_other_buffers
            )

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

        def joint_f(
            score: Tensor,
            b: Tensor,
            h: Tensor,
            m: Tensor,
            n: Tensor,
            example_grad: Tensor,
            *other_buffers: tuple[Tensor, ...],
        ) -> tuple[Tensor, ...]:
            def fw_with_masks(
                *args: tuple[Tensor, ...]
            ) -> tuple[tuple[Tensor], tuple[bool]]:
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
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        fw_graph: Callable,
        joint_graph: Callable,
        block_mask: tuple[Any, ...],
        scale: float,
        kernel_options: dict[str, Any],
        mask_mod_other_buffers: tuple[Any, ...],
        *score_mod_other_buffers: tuple[Any, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        any_buffer_requires_grad = any(
            buffer.requires_grad
            for buffer in mask_mod_other_buffers
            if isinstance(buffer, torch.Tensor)
        )
        assert (
            not any_buffer_requires_grad
        ), "Captured buffers from mask mod that require grad are not supported."
        ctx._fw_graph = fw_graph
        ctx._joint_graph = joint_graph
        ctx._mask_graph = block_mask[-1]
        ctx.scale = scale
        ctx.kernel_options = kernel_options
        ctx._score_mod_other_buffers_len = len(score_mod_other_buffers)
        with torch._C._AutoDispatchBelowAutograd():
            out, logsumexp = flex_attention(
                query,
                key,
                value,
                fw_graph,
                block_mask,
                scale,
                kernel_options,
                score_mod_other_buffers,
                mask_mod_other_buffers,
            )

        save_tensors_and_symints_for_backward(
            ctx,
            (
                query,
                key,
                value,
                out,
                logsumexp,
                *block_mask[:-1],
                *score_mod_other_buffers,
                *mask_mod_other_buffers,
            ),
        )
        return out, logsumexp

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor, grad_logsumexp: Tensor) -> tuple[Optional[Tensor], ...]:  # type: ignore[override]
        fw_args = saved_tensors_and_symints(ctx)
        (
            query,
            key,
            value,
            out,
            logsumexp,
            query_lengths,
            kv_lengths,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
            Q_BLOCK_SIZE,
            KV_BLOCK_SIZE,
            *other_buffers,
        ) = fw_args
        fw_graph = ctx._fw_graph
        joint_graph = ctx._joint_graph
        mask_graph = ctx._mask_graph
        scale = ctx.scale
        kernel_options = ctx.kernel_options
        score_mod_other_buffers = tuple(
            other_buffers[: ctx._score_mod_other_buffers_len]
        )
        mask_mod_other_buffers = tuple(
            other_buffers[ctx._score_mod_other_buffers_len :]
        )
        # We have asserted that mask_mod_other_buffers do not require grad,
        # but score_mod_other_buffers can require grad.
        none_grads = [None] * 6
        (
            grad_query,
            grad_key,
            grad_value,
            grad_score_mod_captured,
        ) = flex_attention_backward(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            fw_graph,
            joint_graph,
            (
                query_lengths,
                kv_lengths,
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                q_num_blocks,
                q_indices,
                full_q_num_blocks,
                full_q_indices,
                Q_BLOCK_SIZE,
                KV_BLOCK_SIZE,
                mask_graph,
            ),
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
        return grad_query, grad_key, grad_value, *none_grads, *grad_score_mod_captured


# TODO: Rework DispatchKey.Autograd to py_autograd_impl
@flex_attention.py_impl(DispatchKey.Autograd)
def flex_attention_autograd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple[Tensor, ...] = (),
    mask_mod_other_buffers: tuple[Tensor, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    with TransformGetItemToIndex():
        input_requires_grad = any(
            isinstance(t, torch.Tensor) and t.requires_grad
            for t in (query, key, value, *score_mod_other_buffers)
        )
        if torch.is_grad_enabled() and input_requires_grad:
            example_vals = (
                query.new_zeros((), requires_grad=input_requires_grad),
                query.new_zeros((), dtype=torch.int),
                query.new_zeros((), dtype=torch.int),
                query.new_zeros((), dtype=torch.int),
                query.new_zeros((), dtype=torch.int),
            )
            fw_graph, bw_graph = create_fw_bw_graph(
                score_mod, example_vals, score_mod_other_buffers
            )
        else:
            fw_graph, bw_graph = score_mod, None
        out, logsumexp = FlexAttentionAutogradOp.apply(
            query,
            key,
            value,
            fw_graph,
            bw_graph,
            block_mask,
            scale,
            kernel_options,
            mask_mod_other_buffers,
            *score_mod_other_buffers,
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
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable,  # GraphModule type hint?
    joint_graph: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple,
    mask_mod_other_buffers: tuple,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, tuple[Optional[torch.Tensor], ...]
]:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    Bq, Hq, seq_len_q, qk_head_dim = query.shape
    Bkv, Hkv, seq_len_kv, v_head_dim = value.shape

    # Get outputs before calling repeat interleave and permute to input stride orders
    actual_grad_query = query.new_empty((Bq, Hq, seq_len_q, qk_head_dim))
    actual_grad_query = _permute_strides(actual_grad_query, query.stride())

    actual_grad_key = key.new_empty((Bq, Hkv, seq_len_kv, qk_head_dim))
    actual_grad_key = _permute_strides(actual_grad_key, key.stride())

    actual_grad_value = value.new_empty((Bq, Hkv, seq_len_kv, v_head_dim))
    actual_grad_value = _permute_strides(actual_grad_value, value.stride())

    def _maybe_new_buffer(
        buffer: Union[torch.Tensor, torch.SymInt, int]
    ) -> Optional[Union[torch.Tensor, torch.SymInt, int]]:
        if isinstance(buffer, torch.Tensor):
            return (
                torch.empty_like(buffer, memory_format=torch.contiguous_format)
                if buffer.requires_grad
                else None
            )
        return buffer

    actual_grad_score_mod_captured = [
        _maybe_new_buffer(buffer) for buffer in score_mod_other_buffers
    ]

    Bq, Bkv = query.size(0), key.size(0)
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.size()[1:]))
    value = value.expand((Bq, *value.size()[1:]))

    G = query.size(1) // key.size(1)
    key = torch.repeat_interleave(key, G, dim=1)
    value = torch.repeat_interleave(value, G, dim=1)

    # We're undoing the log -> log2 change of base in the forwards
    logsumexp = logsumexp * math.log(2)
    # The backwards formula for the log -> log2 change of base in the forwards
    grad_logsumexp = grad_logsumexp / math.log(2)
    scores, post_mod_scores = _math_attention_inner(
        query,
        key,
        value,
        fw_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    masked_out_rows = logsumexp == -float("inf")
    softmax_scores = torch.exp(post_mod_scores - logsumexp.unsqueeze(-1))
    softmax_scores = torch.where(masked_out_rows.unsqueeze(-1), 0, softmax_scores)

    grad_value = softmax_scores.to(query.dtype).transpose(-2, -1) @ grad_out

    grad_softmax_scores = grad_out @ value.transpose(-2, -1)

    sum_scores = torch.sum(out * grad_out, -1, keepdim=True)
    grad_score_mod = softmax_scores * (
        grad_softmax_scores - sum_scores + grad_logsumexp.unsqueeze(-1)
    )

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    mask_graph = block_mask[-1]
    # Gradient of the inline score_mod function, with respect to the scores
    captured_buffers_in_dim = (None,) * len(score_mod_other_buffers)
    out_dims = [0, None, None, None, None] + [None] * len(score_mod_other_buffers)
    from torch.nn.attention.flex_attention import _vmap_for_bhqkv

    # inputs are [score, b, h, q_idx, kv_idx, gradOut, ...]
    # score and gradOut are "fully" batched
    joint_score_mod = _vmap_for_bhqkv(
        joint_graph,
        prefix=(0,),
        suffix=(0,) + captured_buffers_in_dim,
        out_dims=out_dims,
    )
    with TransformGetItemToIndex():
        grad_scores, _, _, _, _, *grad_score_mod_captured = joint_score_mod(
            scores, b, h, m, n, grad_score_mod, *score_mod_other_buffers
        )
    grad_scores = grad_scores * scale
    grad_scores = grad_scores.to(query.dtype)

    mask_mod = _vmap_for_bhqkv(
        mask_graph, prefix=(), suffix=(None,) * len(mask_mod_other_buffers)
    )
    with TransformGetItemToIndex():
        mask_scores = mask_mod(b, h, m, n, *mask_mod_other_buffers)
        grad_scores = torch.where(
            mask_scores, grad_scores, torch.tensor(0, dtype=query.dtype)
        )

    grad_query = grad_scores @ key
    grad_key = grad_scores.transpose(-2, -1) @ query

    # Reduce DK, DV along broadcasted heads.
    grad_key = grad_key.view(
        grad_key.size(0), -1, G, grad_key.size(-2), grad_key.size(-1)
    )
    grad_value = grad_value.view(
        grad_value.size(0), -1, G, grad_value.size(-2), grad_value.size(-1)
    )

    grad_key = torch.sum(grad_key, 2, keepdim=False)
    grad_value = torch.sum(grad_value, 2, keepdim=False)

    # Fill to correctly strided outputs
    actual_grad_query.copy_(grad_query)
    actual_grad_key.copy_(grad_key)
    actual_grad_value.copy_(grad_value)

    if Bq != Bkv:
        assert (
            Bq > 1 and Bkv == 1
        ), f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}"

        actual_grad_key = torch.sum(actual_grad_key, 0, keepdim=True)
        actual_grad_value = torch.sum(actual_grad_value, 0, keepdim=True)

    score_mod_other_buffer_grads = [
        actual_grad.copy_(grad) if isinstance(actual_grad, torch.Tensor) else None
        for actual_grad, grad in zip(
            actual_grad_score_mod_captured, grad_score_mod_captured
        )
    ]

    return (
        actual_grad_query,
        actual_grad_key,
        actual_grad_value,
        tuple(score_mod_other_buffer_grads),
    )


def trace_flex_attention_backward(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, tuple[Optional[torch.Tensor], ...]
]:
    """We already have the forward graph and joint graph from the forward pass, so we create a proxy attach both graphs"""
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    example_out = flex_attention_backward(
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        joint_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )

    requires_grad = any(pytree.tree_map(lambda x: x.requires_grad, (query, key)))
    fw_example_vals = [query.new_zeros((), requires_grad=requires_grad)] + [
        query.new_zeros((), dtype=torch.int) for _ in range(4)
    ]
    bw_example_vals = fw_example_vals + [query.new_zeros(())]
    mask_example_vals = [query.new_zeros((), dtype=torch.int) for _ in range(4)]
    mask_graph = block_mask[-1]
    with TransformGetItemToIndex():
        # There's no active make_fx during the compiled autograd graph's initial capture
        fw_graph = _maybe_reenter_make_fx(fw_graph)(
            *fw_example_vals, *score_mod_other_buffers
        )
        joint_graph = _maybe_reenter_make_fx(joint_graph)(
            *bw_example_vals, *score_mod_other_buffers
        )
        mask_graph = _maybe_reenter_make_fx(mask_graph)(
            *mask_example_vals, *mask_mod_other_buffers
        )
    assert isinstance(proxy_mode.tracer, torch.fx.Tracer)
    block_mask = block_mask[:-1] + (mask_graph,)

    qualname = proxy_mode.tracer.get_fresh_qualname("fw_graph")
    proxy_mode.tracer.root.register_module(qualname, fw_graph)  # type: ignore[arg-type]
    qualname = proxy_mode.tracer.get_fresh_qualname("joint_graph")
    proxy_mode.tracer.root.register_module(qualname, joint_graph)
    qualname = proxy_mode.tracer.get_fresh_qualname("mask_graph")
    proxy_mode.tracer.root.register_module(qualname, mask_graph)

    node_args = (
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        joint_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
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
    grad_logsumexp: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, tuple[Optional[torch.Tensor], ...]
]:
    assert mode is not None, "Mode should always be enabled for python fallback key"
    return trace_flex_attention_backward(
        mode,
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        joint_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
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
    grad_logsumexp: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, tuple[Optional[torch.Tensor], ...]
]:
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
    grad_logsumexp_unwrapped = ctx.unwrap_tensors(grad_logsumexp)
    block_mask_unwrapped = ctx.unwrap_tensors(block_mask)
    score_mod_other_buffers_unwrapped = ctx.unwrap_tensors(score_mod_other_buffers)
    mask_mod_other_buffers_unwrapped = ctx.unwrap_tensors(mask_mod_other_buffers)

    # Appease the mypy overlords
    assert isinstance(query_unwrapped, torch.Tensor)
    assert isinstance(key_unwrapped, torch.Tensor)
    assert isinstance(value_unwrapped, torch.Tensor)
    assert isinstance(out_unwrapped, torch.Tensor)
    assert isinstance(logsumexp_unwrapped, torch.Tensor)
    assert isinstance(grad_out_unwrapped, torch.Tensor)
    assert isinstance(grad_logsumexp_unwrapped, torch.Tensor)
    assert isinstance(block_mask_unwrapped, tuple)
    assert isinstance(score_mod_other_buffers_unwrapped, tuple)
    assert isinstance(mask_mod_other_buffers_unwrapped, tuple)

    with ctx.redispatch_to_next():
        functional_fw_graph = ctx.functionalize(fw_graph)
        functional_joint_graph = ctx.functionalize(joint_graph)

        (
            grad_query,
            grad_key,
            grad_value,
            grad_score_mod_captured,
        ) = flex_attention_backward(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            out_unwrapped,
            logsumexp_unwrapped,
            grad_out_unwrapped,
            grad_logsumexp_unwrapped,
            functional_fw_graph,  # type: ignore[arg-type]
            functional_joint_graph,  # type: ignore[arg-type]
            block_mask_unwrapped,
            scale,
            kernel_options,
            score_mod_other_buffers_unwrapped,
            mask_mod_other_buffers_unwrapped,
        )

    return ctx.wrap_tensors((grad_query, grad_key, grad_value, grad_score_mod_captured))  # type: ignore[return-value,arg-type]


@flex_attention_backward.py_impl(FakeTensorMode)
def flex_attention_backward_fake_tensor_mode(
    mode: FakeTensorMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, tuple[Optional[torch.Tensor], ...]
]:
    with mode:
        Bq, _, _, qk_head_dim = query.shape
        Bkv, Hkv, seq_len_kv, v_head_dim = value.shape

        grad_query = torch.empty_like(query)
        # zeros_and_scatter creates a contiguous zeros tensor -> contiguous_format
        grad_score_mod_captured = tuple(
            [
                torch.empty_like(buffer, memory_format=torch.contiguous_format)
                if isinstance(buffer, torch.Tensor) and buffer.requires_grad
                else None
                for buffer in score_mod_other_buffers
            ]
        )

        broadcasted_grad_key = key.new_empty((Bq, Hkv, seq_len_kv, qk_head_dim))
        broadcasted_grad_key = _permute_strides(broadcasted_grad_key, key.stride())

        broadcasted_grad_value = value.new_empty((Bq, Hkv, seq_len_kv, v_head_dim))
        broadcasted_grad_value = _permute_strides(
            broadcasted_grad_value, value.stride()
        )

        if Bq > 1 and Bkv == 1:
            grad_key = torch.sum(broadcasted_grad_key, dim=0, keepdim=True)
            grad_value = torch.sum(broadcasted_grad_value, dim=0, keepdim=True)
        else:
            grad_key = broadcasted_grad_key
            grad_value = broadcasted_grad_value

        return grad_query, grad_key, grad_value, grad_score_mod_captured


flex_attention_backward.py_autograd_impl(
    autograd_not_implemented(flex_attention_backward, deferred_error=True)
)
