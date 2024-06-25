import contextlib
import weakref
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Protocol, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
from torch import nn
from torch.distributed._tensor import distribute_module, DTensor, Replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle


aten = torch.ops.aten


def sdpa_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    local_results = _scaled_dot_product_ring_flash_attention(
        op_info.mesh,
        *op_info.local_args,  # type: ignore[arg-type]
        **op_info.local_kwargs,  # type: ignore[arg-type]
    )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def _merge_sdpa(
    chunks: List[torch.Tensor], logsumexps: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This merges multiple scaled dot product attention chunks by using the
    provided logsumexps to rescale the chunks before summing.

    Args:
        chunks (List[torch.Tensor]): A list of scaled dot product attention chunks
        logsumexps (List[torch.Tensor]): A list of logsumexps for each chunk

    Returns:
        out (torch.Tensor): The merged scaled dot product attention
        softmax_lse (torch.Tensor): The logsumexp of the merged scaled dot product attention
    """
    assert len(chunks) == len(logsumexps)

    # LSE may be padded in the sequence dimension such as with memory efficient attention.
    seq_len = chunks[0].size(2)
    logsumexps = [lse[:, :, :seq_len] for lse in logsumexps]

    softmax_lse = torch.stack([lse.exp() for lse in logsumexps]).sum(dim=0).log_()

    out = []
    for i, (chunk, chunk_lse) in enumerate(zip(chunks, logsumexps)):
        softmax_lse_corrected = torch.exp(chunk_lse - softmax_lse)
        out_corrected = chunk * softmax_lse_corrected.unsqueeze(-1).to(chunk.dtype)
        out.append(out_corrected)
    out = torch.stack(out).sum(dim=0)

    return out, softmax_lse


def _scaled_dot_product_ring_flash_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")

    return _templated_ring_attention(
        mesh,
        torch.ops.aten._scaled_dot_product_flash_attention,
        query=query,
        key=key,
        value=value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def _scaled_dot_product_ring_efficient_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    compute_log_sumexp: bool = True,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")
    if not compute_log_sumexp:
        raise NotImplementedError("compute_log_sumexp must be set")

    return _templated_ring_attention(
        mesh,
        torch.ops.aten._scaled_dot_product_efficient_attention,
        query=query,
        key=key,
        value=value,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        compute_log_sumexp=compute_log_sumexp,
    )


def _scaled_dot_product_ring_cudnn_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = True,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    if not return_debug_mask:
        raise NotImplementedError("return_debug_mask must be set")

    return _templated_ring_attention(
        mesh,
        torch.ops.aten._scaled_dot_product_cudnn_attention,
        query=query,
        key=key,
        value=value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=return_debug_mask,
        scale=scale,
    )


def _ring_rotate(block: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    rank = dist.get_rank(pg)
    size = dist.get_world_size(pg)

    # rank 0 sends to rank 1, rank 1 sends to rank 2, ..., rank n-1 sends to rank 0
    input_split_sizes = [0] * size
    input_split_sizes[(rank + 1) % size] = len(block)
    output_split_sizes = [0] * size
    output_split_sizes[(rank - 1) % size] = len(block)

    out = ft_c.all_to_all_single_autograd(
        block, input_split_sizes, output_split_sizes, pg
    )
    return out


class AttentionOp(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args: object,
        is_causal: bool = False,
        **kwargs: object,
    ) -> Tuple[torch.Tensor, ...]:
        ...


def _templated_ring_attention(
    mesh: DeviceMesh,
    op: AttentionOp,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *args: object,
    is_causal: bool = False,
    **kwargs: object,
) -> Tuple[torch.Tensor, ...]:
    """
    This is a generalized ring attention implementation that can support multiple attention ops.

    Parameters
    ----------
    op:
        The attention op to use
    *args:
        additional args are passed to the op
    **kwargs:
        additional kwargs are passed to the op

    Returns
    -------
    out:
        The merged attention output
    softmax_lse:
        The logsumexp of the merged attention output
    """
    if is_causal and (query.size(2) != key.size(2)):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )

    if isinstance(mesh, dist.ProcessGroup):
        pg: Union[dist.ProcessGroup, List[dist.ProcessGroup]] = mesh
    else:
        pg = mesh.get_group()
    assert isinstance(pg, dist.ProcessGroup), "process group must be single dimension"
    rank = dist.get_rank(pg)
    size = dist.get_world_size(pg)

    next_kv = None

    chunks = []
    logsumexps = []
    for i in range(size):
        # overlap communication with compute
        if next_kv is not None:
            next_kv = ft_c.wait_tensor(next_kv)
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = _ring_rotate(next_kv, pg)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        if is_causal_behavior != _CausalBehavior.SKIP:
            local_results = op(
                query,
                key,
                value,
                *args,
                is_causal=is_causal_behavior.value,
                **kwargs,
            )
            chunks.append(local_results[0])
            logsumexps.append(local_results[1])

    out, softmax_lse = _merge_sdpa(chunks, logsumexps)

    local_results = (out, softmax_lse) + local_results[2:]
    return local_results


def _scaled_dot_product_chunk_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    size: int,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    This is a single node chunked implementation of
    _scaled_dot_product_ring_flash_attention used for verifying
    the correctness of the backwards pass.
    """

    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")

    if is_causal and (query.size(2) != key.size(2)):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )

    query_len = query.size(2) // size
    ctx_len = key.size(2) // size

    global_out = []
    global_softmax_lse = []

    for rank in range(size):
        chunks = []
        logsumexps = []

        chunk_query = query[:, :, rank * query_len : (rank + 1) * query_len]

        for i in range(size):
            src_rank = (rank - i) % size
            chunk_key = key[:, :, src_rank * ctx_len : (src_rank + 1) * ctx_len]
            chunk_value = value[:, :, src_rank * ctx_len : (src_rank + 1) * ctx_len]

            is_causal_behavior = _is_causal_behavior(
                rank=rank, world_size=size, i=i, is_causal=is_causal
            )

            if is_causal_behavior != _CausalBehavior.SKIP:
                local_results = torch.ops.aten._scaled_dot_product_flash_attention(
                    chunk_query,
                    chunk_key,
                    chunk_value,
                    dropout_p=dropout_p,
                    is_causal=is_causal_behavior.value,
                    scale=scale,
                )
                chunks.append(local_results[0])
                logsumexps.append(local_results[1])

        out, softmax_lse = _merge_sdpa(chunks, logsumexps)
        global_out.append(out)
        global_softmax_lse.append(softmax_lse)

    global_out = torch.concat(global_out, dim=2)
    global_softmax_lse = torch.concat(global_softmax_lse, dim=2)

    local_results = (global_out, global_softmax_lse) + local_results[2:]
    return local_results


def sdpa_backward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # Redistribute grad_output tensor to the same placement as output tensor
    args = list(args)
    assert isinstance(args[0], DTensor) and isinstance(args[4], DTensor)
    args[0] = args[0].redistribute(args[4].device_mesh, args[4].placements)
    args = tuple(args)

    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    local_results = _scaled_dot_product_ring_flash_attention_backward(
        op_info.mesh,
        *op_info.local_args,  # type: ignore[arg-type]
        **op_info.local_kwargs,  # type: ignore[arg-type]
    )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def _scaled_dot_product_ring_flash_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    pg = mesh.get_group()
    assert isinstance(pg, dist.ProcessGroup), "must be single dimension"
    rank = dist.get_rank(pg)
    size = dist.get_world_size(pg)

    # rank 0 sends to rank 1, rank 1 sends to rank 2, ..., rank n-1 sends to rank 0
    right_dsts = list(range(1, size)) + [0]

    next_kv = None

    out_grad_queries = []
    out_grad_keys = []
    out_grad_values = []

    for i in range(size):
        # overlap communication with compute
        if next_kv is not None:
            next_kv = ft_c.wait_tensor(next_kv)
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = ft_c.permute_tensor(next_kv, right_dsts, pg)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        if is_causal_behavior != _CausalBehavior.SKIP:
            # we rerun the forwards pass since we don't have a good way to save the
            # output/logsumexp
            (
                output,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                philox_seed,
                philox_offset,
                _,
            ) = torch.ops.aten._scaled_dot_product_flash_attention(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=is_causal_behavior.value,
                scale=scale,
            )

            softmax_lse_corrected = torch.exp(logsumexp - softmax_lse)

            chunk_grad = grad_out * softmax_lse_corrected.conj().unsqueeze(-1).to(
                grad_out.dtype
            )

            (
                grad_query,
                grad_key,
                grad_value,
            ) = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                grad_out=chunk_grad,
                query=query,
                key=key,
                value=value,
                out=output,
                logsumexp=logsumexp,
                cum_seq_q=cum_seq_q,
                cum_seq_k=cum_seq_k,
                max_q=max_q,
                max_k=max_k,
                dropout_p=dropout_p,
                is_causal=is_causal_behavior.value,
                philox_seed=philox_seed,
                philox_offset=philox_offset,
                scale=scale,
            )
        else:
            grad_query = torch.zeros_like(query)
            grad_key = torch.zeros_like(key)
            grad_value = torch.zeros_like(value)

        # TODO overlap grad communication
        if i == 0:
            out_grad_queries.append(grad_query)
            out_grad_keys.append(grad_key)
            out_grad_values.append(grad_value)
        elif i > 0:
            grad_dsts = [(-i) % size for i in range(size)]

            grad_kv = torch.cat([grad_key.flatten(), grad_value.flatten()])
            grad_kv = ft_c.permute_tensor(grad_kv, grad_dsts, pg)
            grad_kv = ft_c.wait_tensor(grad_kv)
            grad_key = grad_kv[: grad_key.numel()].reshape(grad_key.shape)
            grad_value = grad_kv[grad_key.numel() :].reshape(grad_value.shape)

            out_grad_queries.append(grad_query)
            out_grad_keys.append(grad_key)
            out_grad_values.append(grad_value)

    # stack and sum to avoid accumulation errors
    out_grad_query = torch.stack(out_grad_queries).sum(dim=0)
    out_grad_key = torch.stack(out_grad_keys).sum(dim=0)
    out_grad_value = torch.stack(out_grad_values).sum(dim=0)

    return out_grad_query, out_grad_key, out_grad_value


customized_ops = {
    aten._scaled_dot_product_flash_attention.default: sdpa_handler,
    aten._scaled_dot_product_flash_attention_backward.default: sdpa_backward_handler,
}


@contextlib.contextmanager
def attention_context_parallel() -> Generator[None, None, None]:
    """
    This is a context manager that force enables attention context parallel
    optimizations for all scaled_dot_product_attention ops.

    This currently only supports ring attention and the
    SDPBackend.FLASH_ATTENTION backend. See sdpa_kernel.

    Non-flash attention backends will result in incorrect results.
    """
    old_handlers = DTensor._op_dispatcher._custom_op_handlers
    DTensor._op_dispatcher._custom_op_handlers = {**old_handlers, **customized_ops}

    yield

    DTensor._op_dispatcher._custom_op_handlers = old_handlers


class AttentionContextParallel(ParallelStyle):
    """
    Applies context parallel optimizations to the attention layer.

    This will work for nn.MultiHeadedAttention and custom attention layers that
    call F.scaled_dotproduct_attention with a simliar signature.

    This expects the `forward` method consumes either:

    * a single tensor for self attention
    * one argument for each of: query, key, value

    This currently only supports ring attention and the
    SDPBackend.FLASH_ATTENTION backend. See sdpa_kernel.

    Non-flash attention backends will result in incorrect results.
    """

    # use a weakref dictionary to store context managers for each nn.Module
    _CONTEXT_MANAGERS: "weakref.WeakKeyDictionary[nn.Module, Any]" = (
        weakref.WeakKeyDictionary()
    )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(device_mesh, DeviceMesh):
            raise ValueError(
                f"{type(device_mesh)} is not supported by {type(self)} yet."
            )

        if not device_mesh.ndim == 1:
            raise ValueError

        return distribute_module(
            module,
            device_mesh,
            input_fn=self._input_fn,  # type: ignore[arg-type]
            output_fn=self._output_fn,  # type: ignore[arg-type]
        )

    @classmethod
    def _input_fn(
        cls,
        module: nn.Module,
        inputs: Tuple[Union[torch.Tensor, int, float], ...],
        device_mesh: DeviceMesh,
    ) -> Tuple[Union[torch.Tensor, int, float], ...]:
        # TODO(d4l3k); this should be Shard(2), need to fix Linear layer rules
        placement = [Replicate()]

        def backward_hook(grad: torch.Tensor) -> None:
            if module in cls._CONTEXT_MANAGERS:
                cls._CONTEXT_MANAGERS[module].__exit__(None, None, None)
                del cls._CONTEXT_MANAGERS[module]

        # convert inputs to DTensor
        inp = []
        for input in inputs:
            if isinstance(input, torch.Tensor) and not isinstance(input, DTensor):
                input = DTensor.from_local(
                    input, device_mesh, placement, run_check=False
                )

            if isinstance(input, torch.Tensor) and input.requires_grad:
                input.register_hook(backward_hook)

            inp.append(input)

        manager = attention_context_parallel()
        manager.__enter__()
        cls._CONTEXT_MANAGERS[module] = manager

        return tuple(inp)

    @classmethod
    def _output_fn(
        cls,
        module: nn.Module,
        outputs: Union[torch.Tensor, Tuple[Union[torch.Tensor, int, float], ...]],
        device_mesh: DeviceMesh,
    ) -> Union[
        Union[torch.Tensor, int, float], Tuple[Union[torch.Tensor, int, float], ...]
    ]:
        cls._CONTEXT_MANAGERS[module].__exit__(None, None, None)
        del cls._CONTEXT_MANAGERS[module]

        def backward_hook(grad: torch.Tensor) -> None:
            if module not in cls._CONTEXT_MANAGERS:
                manager = attention_context_parallel()
                manager.__enter__()
                cls._CONTEXT_MANAGERS[module] = manager

        # back to local tensor
        out = []
        for output in [outputs] if isinstance(outputs, torch.Tensor) else outputs:
            output = output.to_local() if isinstance(output, DTensor) else output

            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(backward_hook)

            out.append(output)

        if isinstance(outputs, torch.Tensor):
            return out[0]

        return tuple(out)


class _CausalBehavior(Enum):
    SKIP = None
    NOT_IS_CAUSAL = False
    IS_CAUSAL = True


def _is_causal_behavior(
    rank: int, world_size: int, i: int, is_causal: bool
) -> _CausalBehavior:
    """
    Calculate is_causal behavior for each KV block. The attention can either be
    calculated in full, not at all or with the causal mask applied.
    """
    if not is_causal:
        return _CausalBehavior.NOT_IS_CAUSAL

    if i == 0:
        return _CausalBehavior.IS_CAUSAL

    source_rank = (rank - i) % world_size
    if source_rank < rank:
        return _CausalBehavior.NOT_IS_CAUSAL
    else:
        return _CausalBehavior.SKIP
