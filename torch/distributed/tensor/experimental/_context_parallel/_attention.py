import contextlib
import itertools
import logging
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial
from typing import Any, cast, Optional, Protocol, TypeAlias

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.distributed.distributed_c10d as c10d
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.distributed.tensor.parallel import ParallelStyle
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
)
from torch.utils._pytree import tree_flatten, tree_unflatten

from ._cp_custom_ops import flex_cp_allgather
from ._load_balancer import _create_default_load_balancer, _LoadBalancer


__all__ = [
    "_CausalBehavior",
    "_context_parallel_shard",
    "_ContextParallel",
    "_cp_options",
    "_disable_context_parallel_dispatcher",
    "_enable_context_parallel_dispatcher",
    "_is_causal_behavior",
    "_RotateMethod",
    "context_parallel",
    "context_parallel_unshard",
    "set_rotate_method",
]


class _CausalBehavior(Enum):
    SKIP = None
    NOT_IS_CAUSAL = False
    IS_CAUSAL = True


class _RotateMethod(Enum):
    ALL_TO_ALL = auto()
    ALL_GATHER = auto()


aten = torch.ops.aten
logger = logging.getLogger(__name__)


class _DispatchMode(Enum):
    MONKEY_PATCH = auto()
    MODULE_WRAPPER = auto()


_dispatch_mode: _DispatchMode = _DispatchMode.MONKEY_PATCH


@dataclass
class _ContextParallelOptions:
    # Whether to upcast parameters and gradients to float32 to avoid accumulation
    # errors. It is likely this is always True, but we currently keep this variable
    # for experimental purposes.
    convert_to_f32: bool = True
    enable_load_balance: bool = True
    rotate_method: _RotateMethod = _RotateMethod.ALL_GATHER


_cp_options = _ContextParallelOptions()


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
    if source_rank < rank or _cp_options.enable_load_balance:
        return _CausalBehavior.NOT_IS_CAUSAL
    else:
        return _CausalBehavior.SKIP


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _partial_update(
    original: torch.Tensor,
    new: torch.Tensor,
    dim: int,
    n_chunks: int,
    idx: int,
    add: bool,
) -> torch.Tensor:
    """
    This API partially updates a chunk of ``original`` tensor. The ``original``
    tensor will be first chunked along ``dim`` dimension, then the ``idx`` chunk
    will be updated with ``new``. If ``add`` is True, the chunk will be added
    with ``new``, otherwise the chunk will be replaced by ``new``.

    The result is a tensor that is the same size as ``original``.
    """
    chunks = list(original.chunk(n_chunks, dim=dim))
    assert chunks[idx].shape == new.shape, (original.shape, new.shape, idx)
    if add:
        chunks[idx] += new
    else:
        chunks[idx] = new
    return torch.cat(chunks, dim=dim)


class _SDPAMerger:
    """A class to help merge the local SDPA result."""

    def __init__(self, convert_to_f32: bool, seq_dim: int):
        self._seq_dim = seq_dim
        self._out: Optional[torch.Tensor] = None
        self._lse: Optional[torch.Tensor] = None
        self._should_lse_squeeze = False
        self._convert_to_f32 = convert_to_f32
        self._out_dtype = torch.float32
        self._lse_dtype = torch.float32

    def _merge_one(
        self, block_out: torch.Tensor, block_lse: torch.Tensor, partial: bool
    ) -> None:
        # The cuDNN backend preserves the last dimension for LSE.
        # Apply unsqueeze only if the input does not already have
        # the required dimensionality.
        if len(block_lse.shape) < len(block_out.shape):
            block_lse = block_lse.unsqueeze(dim=-1)
            self._should_lse_squeeze = True
        assert len(block_lse.shape) == len(block_out.shape)

        if self._lse is None:
            self._lse = block_lse
            self._out = block_out
        else:
            ROUND_ROBIN_CYCLE = 2
            assert self._lse is not None
            assert self._out is not None
            lse = (
                self._lse.chunk(ROUND_ROBIN_CYCLE, dim=self._seq_dim)[1]
                if partial
                else self._lse
            )
            out = (
                self._out.chunk(ROUND_ROBIN_CYCLE, dim=self._seq_dim)[1]
                if partial
                else self._out
            )

            # The algorithm from
            # github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
            # gives a relatively stable result.
            out = out - F.sigmoid(block_lse - lse) * (out - block_out)
            lse = lse - F.logsigmoid(lse - block_lse)
            if partial:
                self._lse = _partial_update(
                    self._lse,
                    lse,
                    dim=self._seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=1,
                    add=False,
                )
                self._out = _partial_update(
                    self._out,
                    out,
                    dim=self._seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=1,
                    add=False,
                )
            else:
                self._lse = lse
                self._out = out

    def step(self, out: torch.Tensor, lse: torch.Tensor, partial: bool) -> None:
        self._out_dtype = out.dtype
        self._lse_dtype = lse.dtype

        if self._convert_to_f32:
            out = out.to(torch.float32)
            lse = lse.to(torch.float32)

        self._merge_one(out, lse, partial)

    def results(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._out is not None
        assert self._lse is not None
        out = self._out.to(self._out_dtype)
        if self._should_lse_squeeze:
            lse = self._lse.squeeze(-1).to(self._lse_dtype)
        else:
            lse = self._lse.to(self._lse_dtype)
        return out, lse


class _AttentionOp(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]: ...


class _RingRotater(ABC):
    @abstractmethod
    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None: ...

    @abstractmethod
    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None: ...

    @abstractmethod
    def next_buffer(self) -> torch.Tensor: ...


class _AllToAllRotater(_RingRotater):
    """Use all_to_all to send the kv to the next rank."""

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._buffer: Optional[torch.Tensor] = None

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        curr_buffer = curr_buffer.contiguous()
        size = dist.get_world_size(self._pg)
        dsts = list(range(1, size)) + [0]
        self._buffer = ft_c.permute_tensor(curr_buffer, dsts, self._pg)

    def next_buffer(self) -> torch.Tensor:
        assert self._buffer is not None
        return _maybe_wait(self._buffer)


class _AllGatherRotater(_RingRotater):
    """
    Allgather the kv and return only the required kv.
    Only one communication will be done.
    """

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._aggregated_buffer: Optional[torch.Tensor] = None
        self._idx = 0

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        # We only need to perform allgather once.
        self._idx += 1
        if self._aggregated_buffer is None:
            self._aggregated_buffer = ft_c.all_gather_tensor(
                curr_buffer.contiguous(), gather_dim=0, group=self._pg
            )

    def next_buffer(self) -> torch.Tensor:
        rank = dist.get_rank(self._pg)
        idx = rank - self._idx

        assert self._aggregated_buffer is not None
        self._aggregated_buffer = _maybe_wait(self._aggregated_buffer)
        return self._aggregated_buffer.chunk(dist.get_world_size(self._pg))[idx]


def _create_rotater(
    pg: dist.ProcessGroup, seq_dim: int, method: Optional[_RotateMethod] = None
) -> _RingRotater:
    if method is None:
        method = _cp_options.rotate_method

    if method == _RotateMethod.ALL_TO_ALL:
        return _AllToAllRotater(pg, seq_dim)
    elif method == _RotateMethod.ALL_GATHER:
        return _AllGatherRotater(pg, seq_dim)
    else:
        raise NotImplementedError(f"Unknown method {method}")


def _templated_ring_attention(
    group: dist.ProcessGroup,
    seq_dim: int,
    op: _AttentionOp,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    **kwargs: object,
) -> tuple[torch.Tensor, ...]:
    """
    A generalized ring attention implementation that can support multiple attention ops.

    Note [Context parallelism load balance algorithm for causal masking]
    =====================
    This explanation uses an example to illustrate the CP algorithm with causal
    masking.

    Consider a scenario where the sequence length of q, k, and v is 4 (e.g.,
    q = (q0, q1, q2, q3)), and there are two ranks. For simplicity, we will discuss
    only q and k, as v follows the same pattern as k.

    The diagram below represents a complete QK^T operation without parallelism.
    The `****` entries indicate that the result is not required due to causal
    masking (e.g., q0k1 is marked as `****`).

    +----+------------------------+
    |    |  k0    k1   k2     k3  |
    +----+------------------------+
    | q0 | q0k0, ****, ****, **** |
    | q1 | q1k0, q1k1, ****, **** |
    | q2 | q2k0, q2k1, q2k2, **** |
    | q3 | q3k0, q3k1, q3k2, q3k3 |
    +----+------------------------+

    ### No Load Balance:

    In this scenario, each rank owns a local chunk of q, k, and v, with each chunk
    containing two elements. Rank0 is responsible for managing (q0, q1) and (k0, k1),
    while rank1 manages (q2, q3) and (k2, k3).

    First Iteration: Both rank0 and rank1 perform SDPA with their local qkv pairs.
    Causal masking is enabled as some results are not required (e.g., q0k1).

    Second Iteration: Local queries remain the same, but local kv pairs are exchanged.
    Rank0 now has (q0, q1) and (k2, k3); rank1 has (q2, q3) and (k0, k1). Rank0 performs
    no computation, while rank1 computes locally without causal masking since all results
    (q2k0, q2k1, q3k0, q3k1) are needed.

    ### Round-robin Load Balance:

    In this setup, each rank owns two local chunks of q, k, and v, with each chunk
    containing one element. Rank0 manages (q0, q3) and (k0, k3); Rank1 manages (q1, q2)
    and (k1, k2). Although the local chunks are not consecutive, they are concatenated to
    enable SDPA to be performed in a single call for each step. Consequently, the chunk()
    function may be required to prepare the correct q, k, and v configurations.

    First Iteration: Both ranks perform SDPA with their local qkv pairs, similar to the
    no-load-balance case. This iteration corresponds to the `if` of the
    (`if, `elif`, `else`) in the implementation.

    Second Iteration: Rank0 now has (q0, q3) and (k1, k2); rank1 has (q1, q2) and
    (k0, k3). For rank0, no computation is needed for q0. However, computations for
    q3k1 and q3k2 are required, so only q3 is used for SDPA. This corresponds to the
    `else` of the (`if`, `elif`, `else`) in the implementation.
    For rank1, k3 is not needed for q1 and q2, so only k0 is used for SDPA. This
    corresponds to the `elif` of (`if`, `elif`, `else`) in the implementation.

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
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires `is_causal=True`.")

    assert isinstance(group, dist.ProcessGroup), (
        "process group must be single dimension"
    )
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)

    next_kv = None

    # Without making key and value contiguous(), the loss curve is bad.
    # TODO(fegin): figure out why this is a requirement since SDPA does not have
    # this requirement.
    key = key.contiguous()
    value = value.contiguous()

    sdpa_merger = _SDPAMerger(_cp_options.convert_to_f32, seq_dim=seq_dim)

    rest: list[Any]
    out: torch.Tensor
    logsumexp: torch.Tensor

    rotater = _create_rotater(group, 2)

    for i in range(size):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            next_kv = rotater.next_buffer()
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            # Send the k, v to the next rank
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        # For a detailed understanding of the load balancing algorithm, see
        # Note [Context parallelism load balance algorithm for causal masking]
        if is_causal_behavior == _CausalBehavior.SKIP:
            # If i > rank and load balancing is not turned on.
            continue

        if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
            # When local balance is enabled, we still need to do SDPA with
            # the both local chunks of q, k, v for the first iteration.
            q, k, v, partial = (query, key, value, False)
        elif i <= rank:
            # Round-robin load balancing case, and i <= rank.
            # We need to do SDPA with only the first local chunk of k, v.
            # Note that q, k, v each contains two local chunks.
            ROUND_ROBIN_CYCLE = 2
            q, k, v, partial = (
                query,
                key.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                value.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                False,
            )
        else:
            # Round-robin load balancing case, and i > rank.
            # We need to do SDPA with only the second half of q, and update
            # only the second part of logsumexp. So partial is True.
            # Note that q, k, v each contains two chunks.
            q, k, v, partial = query.chunk(2, dim=2)[1], key, value, True

        # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
        # for the SDPA kernel definitions.
        out, logsumexp, *rest = op(
            q,
            k,
            v,
            is_causal=is_causal_behavior.value,
            **kwargs,
        )
        sdpa_merger.step(out, logsumexp, partial)

    # pyrefly: ignore [unbound-name]
    return *sdpa_merger.results(), *rest


def _templated_ring_attention_backward(
    group: dist.ProcessGroup,
    seq_dim: int,
    op: _AttentionOp,
    grad_out: torch.Tensor,
    grad_out_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    """This API implements the backward pass of the ring attention."""
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires `is_causal=True`.")
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    next_kv = None
    next_grad_kv = None
    rest: list[Any]
    grad_query_, grad_key_, grad_value_ = None, None, None

    accum_dtype = torch.float32 if _cp_options.convert_to_f32 else query.dtype
    grad_query = torch.zeros_like(query, dtype=accum_dtype)
    grad_key = torch.zeros_like(key, dtype=accum_dtype)
    grad_value = torch.zeros_like(value, dtype=accum_dtype)

    key = key.contiguous()
    value = value.contiguous()
    kv_rotater = _create_rotater(group, 2)
    dkv_rotater = _create_rotater(group, 2, method=_RotateMethod.ALL_TO_ALL)
    for i in range(size):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            buffer = kv_rotater.next_buffer()
            pointer = 0
            key = buffer[pointer : pointer + key.numel()].reshape(key.shape)
            pointer += key.numel()
            value = buffer[pointer : pointer + value.numel()].reshape(value.shape)
            pointer += value.numel()

        if i != size - 1:
            # Send the kv to the next rank.
            next_kv = torch.cat([key.flatten(), value.flatten()])
            kv_rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        if is_causal_behavior != _CausalBehavior.SKIP:
            if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
                # We need to do SDPA with the full local q, k, v.
                q, k, v, out_, dout, lse = (query, key, value, out, grad_out, logsumexp)
            elif i <= rank:
                # Round-robin load balancing case, and i <= rank.
                # We need to do SDPA with only the first half of k, v.
                # Note that q, k, v each contains two chunks.
                q, k, v, out_, dout, lse = (
                    query,
                    key.chunk(2, dim=seq_dim)[0],
                    value.chunk(2, dim=seq_dim)[0],
                    out,
                    grad_out,
                    logsumexp,
                )
            else:
                # Round-robin load balancing case, and i > rank.
                # We need to do SDPA with only the second half of q.
                # Note that q, k, v each contains two chunks.
                q, k, v, out_, dout, lse = (
                    query.chunk(2, dim=seq_dim)[1],
                    key,
                    value,
                    out.chunk(2, dim=seq_dim)[1],
                    grad_out.chunk(2, dim=seq_dim)[1],
                    # Need to make logsumexp contiguous, otherwise there will
                    # be numerical error.
                    logsumexp.chunk(2, dim=seq_dim)[1].contiguous(),
                )

            kwargs[grad_out_name] = dout
            # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
            # for the SDPA kernel definitions.
            grad_query_, grad_key_, grad_value_, *rest = op(
                query=q,
                key=k,
                value=v,
                out=out_,
                logsumexp=lse,
                is_causal=is_causal_behavior.value,
                **kwargs,
            )
        else:
            grad_query_ = torch.zeros_like(query, dtype=accum_dtype)
            grad_key_ = torch.zeros_like(key, dtype=accum_dtype)
            grad_value_ = torch.zeros_like(value, dtype=accum_dtype)

        ROUND_ROBIN_CYCLE = 2
        if i == 0:
            grad_key += grad_key_
            grad_value += grad_value_
        else:
            pointer = 0
            # Wait for the kv gradient from (cp_rank - 1) rank.
            next_grad_kv = dkv_rotater.next_buffer()
            grad_key = next_grad_kv[pointer : pointer + grad_key.numel()].reshape(
                grad_key.shape
            )
            pointer += grad_key.numel()
            grad_value = next_grad_kv[pointer : pointer + grad_value.numel()].reshape(
                grad_value.shape
            )

            if i <= rank and _cp_options.enable_load_balance:
                grad_key = _partial_update(
                    grad_key,
                    grad_key_,
                    dim=seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=0,
                    add=True,
                )
                grad_value = _partial_update(
                    grad_value,
                    grad_value_,
                    dim=seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=0,
                    add=True,
                )
            else:
                grad_key += grad_key_
                grad_value += grad_value_

        next_grad_kv = torch.cat([grad_key.flatten(), grad_value.flatten()])
        # Send the grad key and grad value to the next rank.
        dkv_rotater.exchange_buffers(next_grad_kv)

        if i <= rank or not _cp_options.enable_load_balance:
            grad_query += grad_query_
        else:
            grad_query = _partial_update(
                grad_query,
                grad_query_,
                dim=seq_dim,
                n_chunks=ROUND_ROBIN_CYCLE,
                idx=1,
                add=True,
            )

    assert grad_key_ is not None
    assert grad_value_ is not None
    grad_query = grad_query.to(query.dtype)
    next_grad_kv = dkv_rotater.next_buffer().to(key.dtype)
    grad_key = next_grad_kv[: grad_key.numel()].reshape(grad_key.shape)
    grad_value = next_grad_kv[grad_key.numel() :].reshape(grad_value.shape)
    return (
        grad_query,
        grad_key,
        grad_value,
        # pyrefly: ignore [unbound-name]
        *rest,
    )


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
) -> tuple[torch.Tensor, ...]:
    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")

    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention(
        group,
        seq_dim,
        aten._scaled_dot_product_flash_attention,
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=scale,
    )


def _scaled_dot_product_ring_efficient_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    compute_log_sumexp: bool = True,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")

    if not compute_log_sumexp:
        # CP requires compute_log_sumexp to be True because it always merges LSE
        compute_log_sumexp = True

    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention(
        group,
        seq_dim,
        aten._scaled_dot_product_efficient_attention,
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        scale=scale,
        compute_log_sumexp=compute_log_sumexp,
    )


def _scaled_dot_product_ring_cudnn_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    compute_log_sumexp: bool = True,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")

    if not compute_log_sumexp:
        # CP requires compute_log_sumexp to be True because it always merges LSE
        compute_log_sumexp = True

    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention(
        group,
        seq_dim,
        aten._scaled_dot_product_cudnn_attention,
        query=query,
        key=key,
        value=value,
        attn_bias=attn_bias,
        compute_log_sumexp=compute_log_sumexp,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=return_debug_mask,
        scale=scale,
    )


def _scaled_dot_product_ring_flash_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
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
) -> tuple[torch.Tensor, ...]:
    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention_backward(
        group,
        seq_dim,
        aten._scaled_dot_product_flash_attention_backward.default,
        grad_out=grad_out,
        grad_out_name="grad_out",
        query=query,
        key=key,
        value=value,
        out=out,
        logsumexp=logsumexp,
        is_causal=is_causal,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        scale=scale,
    )


def _scaled_dot_product_ring_efficient_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    grad_input_mask: tuple[bool, ...],
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention_backward(
        group,
        seq_dim,
        aten._scaled_dot_product_efficient_attention_backward.default,
        grad_out=grad_out,
        grad_out_name="grad_out_",
        query=query,
        key=key,
        value=value,
        attn_bias=bias,
        out=out,
        logsumexp=logsumexp,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        dropout_p=dropout_p,
        grad_input_mask=grad_input_mask,
        is_causal=is_causal,
        scale=scale,
    )


def _scaled_dot_product_ring_cudnn_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    attn_bias: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    # TODO: remove this hardcoding
    seq_dim = 2
    group = mesh.get_group()
    return _templated_ring_attention_backward(
        group,
        seq_dim,
        aten._scaled_dot_product_cudnn_attention_backward.default,
        grad_out=grad_out,
        grad_out_name="grad_out",
        query=query,
        key=key,
        value=value,
        out=out,
        logsumexp=logsumexp,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        attn_bias=attn_bias,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def _sdpa_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    logger.debug("Dispatching op_call: %s", op_info.schema)

    # sharding propagation
    # TODO: remove the context parallel strategy from the default propagation
    # rule. Either figure out how to dynamically enable it or just don't call
    # propagate.
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    call_maps: dict[torch._ops.OpOverload, Callable] = {
        aten._scaled_dot_product_flash_attention.default: _scaled_dot_product_ring_flash_attention,
        aten._scaled_dot_product_efficient_attention.default: _scaled_dot_product_ring_efficient_attention,
        aten._scaled_dot_product_cudnn_attention.default: _scaled_dot_product_ring_cudnn_attention,
        aten._scaled_dot_product_flash_attention_backward.default: _scaled_dot_product_ring_flash_attention_backward,
        aten._scaled_dot_product_efficient_attention_backward.default: _scaled_dot_product_ring_efficient_attention_backward,
        aten._scaled_dot_product_cudnn_attention_backward.default: _scaled_dot_product_ring_cudnn_attention_backward,
    }
    if op_call in call_maps:
        local_results = call_maps[op_call](
            op_info.compute_mesh,
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    else:
        raise NotImplementedError(
            "CP only supports flash attention and memory efficient attention now."
        )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


custom_ops = {
    aten._scaled_dot_product_flash_attention.default: _sdpa_handler,
    aten._scaled_dot_product_flash_attention_backward.default: _sdpa_handler,
    aten._scaled_dot_product_efficient_attention.default: _sdpa_handler,
    aten._scaled_dot_product_efficient_attention_backward.default: _sdpa_handler,
    aten._scaled_dot_product_cudnn_attention.default: _sdpa_handler,
    aten._scaled_dot_product_cudnn_attention_backward.default: _sdpa_handler,
}
exitsing_custom_ops = DTensor._op_dispatcher._custom_op_handlers


ArgsType = tuple[Any, ...]
KwargsType = dict[str, Any]
InputFnType = Callable[[Optional[nn.Module], ArgsType, KwargsType, DeviceMesh], Any]
OutputFnType = Callable[[Optional[nn.Module], Any, Any, DeviceMesh], Any]

_replaced_functions: dict[Callable, tuple[str, Callable]] = {}


def _distribute_function(
    fn: Callable,
    fn_module: types.ModuleType,
    device_mesh: DeviceMesh,
    input_fn: InputFnType,
    output_fn: OutputFnType,
) -> None:
    """
    A helper function to replace a function with a distributed version by
    using the monkey patching approach.

    This function is for the CP internal usage only.
    """

    def wrapper(
        target_fn: Callable, input_fn: InputFnType, output_fn: OutputFnType
    ) -> Callable:
        def inner_fn(*args: ArgsType, **kwargs: KwargsType) -> Any:
            args, kwargs = input_fn(None, args, kwargs, device_mesh)
            outputs = target_fn(*args, **kwargs)
            return output_fn(None, (args, kwargs), outputs, device_mesh)

        return inner_fn

    global _replaced_functions

    if fn in _replaced_functions:
        return

    wrapper_fn = wrapper(fn, input_fn, output_fn)
    setattr(fn_module, fn.__name__, wrapper_fn)
    _replaced_functions[wrapper_fn] = (fn.__name__, fn)


def _restore_function(fn: Callable, fn_module: types.ModuleType) -> None:
    """Restore the function that is replaced by _distribute_function."""
    if fn not in _replaced_functions:
        return

    original_name, original_fn = _replaced_functions[fn]
    setattr(fn_module, original_name, original_fn)


def _enable_cp_dtensor_dispatcher() -> None:
    """Enables DTensor dispatcher to dispatch SDPA to CP."""
    DTensor._op_dispatcher._custom_op_handlers = {
        **exitsing_custom_ops,
        **custom_ops,
    }


def _disable_cp_dtensor_dispatcher() -> None:
    """Disables DTensor dispatcher to dispatch SDPA to CP."""
    DTensor._op_dispatcher._custom_op_handlers = exitsing_custom_ops


def _enable_context_parallel_dispatcher_impl(seq_dim: int, mesh: DeviceMesh) -> None:
    sdpa_cp = _ContextParallel(
        seq_dim=seq_dim,
        attention_type=_ContextParallel.AttentionType.SDPA,
    )

    if _dispatch_mode == _DispatchMode.MONKEY_PATCH:
        _distribute_function(
            F.scaled_dot_product_attention,
            F,
            mesh,
            sdpa_cp.sdpa_input_fn,
            sdpa_cp.sdpa_output_fn,
        )
        _enable_cp_dtensor_dispatcher()
    elif _dispatch_mode == _DispatchMode.MODULE_WRAPPER:
        _enable_cp_dtensor_dispatcher()
    else:
        raise ValueError(f"Unknown dispatch mode: {_dispatch_mode}")


def _disable_context_parallel_dispatcher_impl() -> None:
    if _dispatch_mode == _DispatchMode.MONKEY_PATCH:
        _restore_function(F.scaled_dot_product_attention, F)
    elif _dispatch_mode == _DispatchMode.MODULE_WRAPPER:
        pass
    else:
        raise NotImplementedError(f"Unknown dispatch mode: {_dispatch_mode}")

    _disable_cp_dtensor_dispatcher()


_compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=False, fullgraph=True
)


def _context_parallel_buffers(
    mesh: DeviceMesh,
    buffers: list[torch.Tensor | BlockMask],
    buffer_seq_dims: list[int],
    load_balancer: Optional[_LoadBalancer] = None,
) -> list[torch.Tensor | BlockMask]:
    """
    Shard the buffers along the sequence dimensions according to CP rules.
    Args:
        mesh (:class:`DeviceMesh`): the device mesh for the context parallelism.
        buffers (List[torch.Tensor]): the buffers to be sharded.
        seq_dims (List[int]): the sequence dimensions of ``buffers``. This list
            must have the same length as ``buffers``.
        load_balancer (Optional[:class:`_LoadBalancer`]): an optional `_LoadBalancer`
            object. If this argument is `None`, it means the `buffers` need no
            rearrangement before being sharded. If this argument is a `_LoadBalancer`
            object, call its `_generate_indices(restore=False)` to generate the
            rearrangement indices such that each shard of `buffer[rearrange_idx]` is
            well-balanced (i.e., having close sparsities).

    Returns:
        List[torch.Tensor]: the sharded buffers.

    Note:
        For `_context_parallel_shard` we require a non-None `load_balancer` object to be
        explicitly passed if load-balancing is needed.
    """
    # generate the index tensor for rearranging the buffer if a load-balance
    # is available
    load_balance_indices = load_balancer._generate_indices() if load_balancer else None
    assert load_balance_indices is None or load_balance_indices.ndim == 2, (
        "load balance index expects shape (1, seq_len) or (B, seq_len) "
        f"but got {load_balance_indices.shape}."
    )

    new_buffers = []
    sharded_buffer: torch.Tensor | BlockMask
    for buffer, seq_dim in zip(buffers, buffer_seq_dims):
        if isinstance(buffer, torch.Tensor):
            # TODO: the load balance doesn't perform error handling.

            # NOTE: assuming batch dim is 0

            if load_balance_indices is not None:
                # TODO: we should expclitly ask users to unsqueeze the batch dim.
                # But this is a BC breaking ask.
                # However, what we have done today is also not very safe.
                idx_batch_size = load_balance_indices.size(0)
                data_batch_size = buffer.size(0) if seq_dim > 0 else 1

                if idx_batch_size != 1 and idx_batch_size != data_batch_size:
                    raise ValueError(
                        "Cannot rearrange buffer: "
                        f"load_balance_indices has shape {load_balance_indices.shape}, "
                        f"but buffer has shape {buffer.shape}."
                    )

                if seq_dim == 0:
                    buffer = torch.index_select(
                        buffer, dim=0, index=load_balance_indices[0]
                    )
                else:
                    indices = load_balance_indices
                    if idx_batch_size == 1:
                        size = [data_batch_size] + list(indices.size())[1:]
                        indices = indices.expand(*size)

                    for i in range(data_batch_size):
                        buffer[i] = torch.index_select(
                            buffer[i], dim=seq_dim - 1, index=indices[i]
                        )

            # use DTensor to shard the buffer on sequence dimension, retain the local tensor
            sharded_buffer = distribute_tensor(
                buffer, mesh, [Shard(seq_dim)], src_data_rank=None
            ).to_local()
        elif isinstance(buffer, BlockMask):
            sharded_buffer = _create_cp_block_mask(
                mask_mod=buffer.mask_mod,
                B=buffer.kv_num_blocks.shape[0],
                H=buffer.kv_num_blocks.shape[1],
                Q_LEN=buffer.seq_lengths[0],
                KV_LEN=buffer.seq_lengths[1],
                device_mesh=mesh,
                load_balancer=load_balancer,
            )
        else:
            raise ValueError(f"Unknown buffer type: {type(buffer)}")

        new_buffers.append(sharded_buffer)

    return new_buffers


def _create_cp_block_mask(
    mask_mod: _mask_mod_signature,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device_mesh: DeviceMesh,
    load_balancer: Optional[_LoadBalancer] = None,
) -> BlockMask:
    """
    Creates a specialized BlockMask for Context Parallel FlexAttention.

    This function creates a BlockMask that enables computation of attention results
    for sharded Q attending to global KV. The mask appropriately handles the query
    index offset required when each rank operates on a shard of the query sequence
    while accessing the full key-value sequence.

    The function internally rewrites the provided mask_mod function to translate local
    query indices to global query indices, ensuring that the masking logic is applied
    correctly across the distributed computation.

    Args:
        mask_mod (Callable): Mask function that operates on global attention indices.
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Global sequence length of the query.
        KV_LEN (int): Global sequence length of the key/value.
        device_mesh (DeviceMesh): Device mesh used for context parallelism.
        load_balancer (Optional[:class:`_LoadBalancer`]): The load-balancer used to rearrange
            QKV before sharding. This will be used to modify the block_mask generated.

    Returns:
        BlockMask: A block mask configured for the local query shard that can be used
            with flex_attention() for the given cp_mesh.

    Raises:
        NotImplementedError: If Q_LEN is not divisible by (CP world size * BLOCK_SIZE).

    Warning:
        Currently requires Q_LEN to be divisible by CP mesh world size * BLOCK_SIZE
        (BLOCK_SIZE defaults to 128). This constraint exists because the BlockMask
        must handle both padding and offsets correctly. For example, if Q_LEN is 384,
        CP world size is 2, and BLOCK_SIZE is 128, the local Q_LEN would be 192. In
        such cases, both rank0 and rank1 would have paddings in their local BlockMasks.
        Support for padding in this scenario is planned for future work.

    """

    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE

    if Q_LEN % (device_mesh.size() * _DEFAULT_SPARSE_BLOCK_SIZE) != 0:
        raise NotImplementedError(
            f"Q_LEN {Q_LEN} is not divisible by CP mesh world size {device_mesh.size()} * "
            f"BLOCK_SIZE {_DEFAULT_SPARSE_BLOCK_SIZE}. This is not supported yet. "
        )

    compiled_create_block_mask = torch.compile(
        create_block_mask, dynamic=False, fullgraph=True
    )

    def _rewrite_mask_mod(
        mask_mod: _mask_mod_signature,
        rank: int,
        block_size: int,
        local_q_size: int,
        qkv_rearrange_indices: Optional[torch.Tensor] = None,
    ) -> _mask_mod_signature:
        assert qkv_rearrange_indices is None or qkv_rearrange_indices.ndim == 2, (
            "load balance index expects shape (1, seq_len) or (B, seq_len) "
            f"but got {qkv_rearrange_indices.shape}."
        )

        def qkv_idx_restore(
            b: torch.Tensor, idx_post_rearrange: torch.Tensor
        ) -> torch.Tensor:
            if qkv_rearrange_indices is not None:
                if (
                    qkv_rearrange_indices.size(0) == 1
                ):  # identical load-balance in batch
                    idx_pre_rearrange = qkv_rearrange_indices[0][idx_post_rearrange]
                else:
                    idx_pre_rearrange = qkv_rearrange_indices[b][idx_post_rearrange]
            else:
                idx_pre_rearrange = idx_post_rearrange

            return idx_pre_rearrange

        def local_q_idx_to_q_idx(local_q_idx: torch.Tensor) -> torch.Tensor:
            # calculate local block_idx and block_offset
            local_blk_idx, local_blk_offset = (
                local_q_idx // block_size,
                local_q_idx % block_size,
            )
            # NOTE: load balancing is not used
            local_num_blocks = local_q_size // block_size
            blk_idx = local_num_blocks * rank + local_blk_idx
            return blk_idx * block_size + local_blk_offset

        return lambda b, h, q_idx, kv_idx: mask_mod(
            b,
            h,
            qkv_idx_restore(b, local_q_idx_to_q_idx(q_idx)),
            qkv_idx_restore(b, kv_idx),
        )

    cp_rank = device_mesh.get_local_rank()
    cp_group_size = device_mesh.size()
    load_balancer = load_balancer or _create_default_load_balancer(
        Q_LEN, cp_group_size, device_mesh.device_type
    )
    Q_SHARD_LEN = Q_LEN // cp_group_size
    block_size = _DEFAULT_SPARSE_BLOCK_SIZE

    rearrange_indices = (
        load_balancer._generate_indices(restore=False) if load_balancer else None
    )
    block_mask = compiled_create_block_mask(
        _rewrite_mask_mod(
            mask_mod,
            cp_rank,
            block_size,
            Q_SHARD_LEN,
            qkv_rearrange_indices=rearrange_indices,
        ),
        B,
        H,
        Q_SHARD_LEN,
        KV_LEN,
        device=device_mesh.device_type,
        BLOCK_SIZE=(block_size, block_size),
    )
    return block_mask


#####################
# Experimental APIs
#####################


class _ContextParallel(ParallelStyle):
    class AttentionType(Enum):
        FLEX = "flex_attention"
        SDPA = "scaled_dot_product_attention"

    def __init__(
        self,
        seq_dim: int,
        attention_type: AttentionType,
    ) -> None:
        super().__init__()
        self.seq_dim = seq_dim
        self.attention_type = attention_type

    def _apply(self, module: nn.Module, mesh: DeviceMesh) -> nn.Module:
        if self.attention_type == self.AttentionType.FLEX:
            module.register_forward_pre_hook(
                partial(self.flex_input_fn, mesh=mesh), with_kwargs=True
            )
            return module
        elif self.attention_type == self.AttentionType.SDPA:
            module.register_forward_pre_hook(
                partial(self.sdpa_input_fn, mesh=mesh), with_kwargs=True
            )
            module.register_forward_hook(partial(self.sdpa_output_fn, mesh=mesh))
            return module
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

    def flex_input_fn(
        self, module: Optional[nn.Module], args: Any, kwargs: Any, mesh: DeviceMesh
    ) -> Any:
        args_list = list(args)
        for idx, name in enumerate(
            ("query", "key", "value", "score_mod", "block_mask")
        ):
            if idx >= len(args):
                args_list.append(kwargs.pop(name, None))

        query, key, value, score_mod, block_mask = args_list[:5]
        assert isinstance(query, torch.Tensor)
        assert isinstance(key, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        assert isinstance(block_mask, BlockMask | tuple)

        key = key.contiguous()
        value = value.contiguous()

        global_key, global_value = flex_cp_allgather(
            key, value, self.seq_dim, c10d._get_process_group_name(mesh.get_group())
        )
        args_list[1] = global_key
        args_list[2] = global_value

        return tuple(args_list), kwargs

    def sdpa_input_fn(
        self,
        module: Optional[nn.Module],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        mesh: DeviceMesh,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        placement = [Shard(self.seq_dim)]
        all_args = []

        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, torch.Tensor):
                if isinstance(arg, DTensor):
                    assert arg._spec.placements == placement
                else:
                    arg = DTensor.from_local(arg, mesh, placement, run_check=False)

            all_args.append(arg)

        new_args = tuple(all_args[0 : len(args)])
        new_kwargs = dict(zip(kwargs.keys(), all_args[len(args) :]))
        return new_args, new_kwargs

    def sdpa_output_fn(
        self, module: Optional[nn.Module], inputs: Any, outputs: Any, mesh: DeviceMesh
    ) -> Any:
        new_outputs = []
        for output in [outputs] if isinstance(outputs, torch.Tensor) else outputs:
            output = output.to_local() if isinstance(output, DTensor) else output
            new_outputs.append(output)

        if isinstance(outputs, torch.Tensor):
            return new_outputs[0]

        return tuple(new_outputs)


CPBuffer: TypeAlias = torch.Tensor | BlockMask
CPBufferContainer: TypeAlias = Sequence[CPBuffer] | Mapping[str, CPBuffer]
CPBufferSeqDims: TypeAlias = Sequence[int] | Mapping[str, int]


def _context_parallel_shard(
    mesh: DeviceMesh,
    buffers: CPBufferContainer,
    seq_dims: CPBufferSeqDims,
    load_balancer: Optional[_LoadBalancer] = None,
) -> list[torch.Tensor | BlockMask]:
    """
    Shard the buffers along the specified sequence dimensions (`seq_dims`), so that each
    rank retains only its corresponding shard according to the provided `mesh`. If a
    `load_balancer` is provided, the buffers will be rearranged by the load balancer
    before sharding to improve load balance. Buffers can be either tensors or `BlockMask`
    objects. If a buffer is a `BlockMask`, its sharding dimension is determined by the
    `BlockMask` implementation, and the corresponding `seq_dim` is ignored.

    Note:
        For `_context_parallel_shard`, a non-None `load_balancer` must be explicitly passed
        if load balancing is required.

    Args:
        mesh (DeviceMesh): The device mesh used for context parallelism.
        buffers (List[torch.Tensor | BlockMask]): Buffers whose usage depends on the sequence
            dimension. Examples include input batches, labels, and positional embedding buffers.
            These buffers must be sharded along the sequence dimension to ensure correctness.
        seq_dims (List[int]): The sequence dimensions for each buffer in `buffers`. Must have
            the same length as `buffers`.
        load_balancer (Optional[_LoadBalancer]): An optional load balancer object. If provided,
            it rearranges the buffers before sharding to achieve better load balance. If not
            provided, no rearrangement is performed.

    Returns:
        List[torch.Tensor | BlockMask]: The sharded buffers, each corresponding to the local
            shard for the current rank.
    """
    # TODO: these global variables are going to bite us someday.
    # We will have to remove them soon.
    # For the new API, we only support the module wrapper mode.
    global _dispatch_mode
    _dispatch_mode = _DispatchMode.MODULE_WRAPPER
    global _cp_options
    if load_balancer is not None:
        _cp_options.enable_load_balance = True
    else:
        _cp_options.enable_load_balance = False

    if len(buffers) != len(seq_dims):
        raise ValueError(
            "`seq_dims` must have the same number of elements as `buffers`."
        )

    flat_buffers, spec = tree_flatten(buffers)
    flat_seq_dims, _ = tree_flatten(seq_dims)
    if len(flat_buffers) != len(flat_seq_dims):
        raise ValueError("`seq_dims` must have the pytree structure as `buffers`.")

    if isinstance(flat_buffers[0], torch.Tensor):
        device = flat_buffers[0].device
    else:
        device = flat_buffers[0].kv_num_blocks.device
    for buffer in flat_buffers:
        if isinstance(buffer, torch.Tensor):
            assert device == buffer.device, "All buffers must be on the same device"
        else:
            assert device == buffer.kv_num_blocks.device, (
                "All buffers must be on the same device"
            )

    flat_sharded_buffers = _context_parallel_buffers(
        mesh, flat_buffers, flat_seq_dims, load_balancer
    )

    return tree_unflatten(flat_sharded_buffers, spec)


def _enable_context_parallel_dispatcher() -> None:
    """
    Enable the context parallel dispatcher. This API is experimental and subject to change.
    """
    _enable_cp_dtensor_dispatcher()


def _disable_context_parallel_dispatcher() -> None:
    """
    Disable the context parallel dispatcher. This API is experimental and subject to change.
    """
    _disable_cp_dtensor_dispatcher()


#####################################################
# Current public APIs, but are also subject to change
#####################################################
@contextlib.contextmanager
@torch.no_grad()
def context_parallel(
    mesh: DeviceMesh,
    *,
    buffers: Optional[list[torch.Tensor]] = None,
    buffer_seq_dims: Optional[list[int]] = None,
    no_restore_buffers: Optional[set[torch.Tensor]] = None,
) -> Generator[None, None, None]:
    """

    ``context_parallel`` is an experimental API to enable context
    parallelism (CP). This API performs two actions: 1) patch the SDPA
    (``torch.nn.functional.scaled_dot_product_attention``) with the CP-enabled
    one, 2) shard ``buffers`` along the sequence dimension and each rank will
    preserve the corresponding shard according ``mesh``.

    Args:
        mesh (:class:`DeviceMesh`): the device mesh for the context parallelism.
        buffers (Optional[List[torch.Tensor]]): buffers that the usage depend
            on the sequence dimension. Examples are input batch, labels and
            positional embedding buffers. These buffers must be sharded along
            the sequence dimension to ensure the accuracy. The sharding will
            happen in-place, the buffer's shape will change within the context.
            The buffers will be restored after the context finishes.
            ``no_restore_buffers`` can be used to specify which buffers don't
            need to be restored. Note that ``buffers`` should not contain any
            nn.Parameter.
        buffer_seq_dims (Optional[List[int]]): the sequence dimensions of ``buffers``.
        no_restore_buffers (Optional[Set[torch.Tensor]]): buffers in these set
            won't be restored after the context exits. This set must be a subset
            of ``buffers``. If the buffers won't be used after the context exits,
            these buffers can be put in this list to avoid extra restore time.

    .. warning::
        `torch.distributed.tensor.experimental.context_parallel` is a
        prototype feature in PyTorch. The API is subject to change.
    """
    # For the legacy API, we only support the monkey-patch mode.
    # We will deprecate this API once the new API is widely used.
    global _dispatch_mode
    _dispatch_mode = _DispatchMode.MONKEY_PATCH

    buffers = [] if buffers is None else buffers
    buffer_seq_dims = [] if buffer_seq_dims is None else buffer_seq_dims
    no_restore_buffers = set() if no_restore_buffers is None else no_restore_buffers

    if len(buffers) != len(buffer_seq_dims):
        raise ValueError(
            "`seq_dims` must have the same number of elements as `buffers`."
        )

    for buffer in no_restore_buffers:
        # Cannot use `if not buffer in buffers` which will incur tensor comparison.
        if not any(b is buffer for b in buffers):
            raise ValueError("`no_restore_buffers` must be a subset of `buffers`.")

    original_buffers = [None if b in no_restore_buffers else b.clone() for b in buffers]

    device = buffers[0].device
    seq_length = buffers[0].shape[buffer_seq_dims[0]]
    cp_world_size = mesh.size()

    # If `enable_load_balance` is True, the default Head-tail load balancer
    # (:class:`_HeadTailLoadBalancer`) is used to rearrange the buffers before
    # sharding. Otherwise, we don't do any load-balance rearrange by passing
    # `None` to `_context_parallel_shard()`.
    load_balancer = _create_default_load_balancer(seq_length, cp_world_size, device)
    shards = _context_parallel_buffers(
        mesh,
        cast(list[torch.Tensor | BlockMask], buffers),
        buffer_seq_dims,
        load_balancer,
    )
    for buffer, shard in zip(buffers, shards):
        assert isinstance(shard, torch.Tensor), "ContextParallel only supports Tensor"
        shard = shard.clone()
        buffer.resize_(shard.shape)
        buffer.copy_(shard)

    _enable_context_parallel_dispatcher_impl(seq_dim=2, mesh=mesh)
    yield
    _disable_context_parallel_dispatcher_impl()

    for buffer, original_buffer in zip(buffers, original_buffers):
        if original_buffer is not None:
            buffer.resize_(original_buffer.shape)
            buffer.copy_(original_buffer)


@torch.no_grad()
def context_parallel_unshard(
    mesh: DeviceMesh,
    buffers: list[torch.Tensor],
    seq_dims: list[int],
    load_balancer: Optional[_LoadBalancer] = None,
) -> list[torch.Tensor]:
    """
    Unshard the tensors (e.g., output) that are sharded due to context parallelism.

    Args:
        mesh (:class:`DeviceMesh`): the device mesh for the context parallelism.
        buffers (List[torch.Tensor]): the buffers to be unsharded.
        seq_dims (List[int]): the sequence dimensions of ``buffers``. This list
            must have the same length as ``buffers``.
        load_balancer (Optional[:class:`_Loadbalancer`]): an optional `_LoadBalancer`
            object. If this argument is `None`, it means the `buffers` were not
            rearranged when being sharded and there's no need to put it back to order
            after unsharding. If this argument is a `_LoadBalancer` object, call
            its `_generate_indices(restore=True)` to generate the restore indices such
            that `unsharded[restore_idx]` is the original buffer.

    Returns:
        List[torch.Tensor]: the unsharded buffers.

    Note:
        For `context_parallel_unshard` we require not-None `load_balancer` object be
        explicitly passed if flex_attention() is to be used and load-balancing is needed.
        This is different from the case of SDPA though we strongly suggest users follow
        the same convention.
    """
    device = buffers[0].device
    cp_world_size = mesh.size()
    seq_length = buffers[0].shape[seq_dims[0]] * cp_world_size

    # If users don't pass in a `load_balancer`:
    # - if `enable_load_balance` is True, we use the default round-robin
    #   load balancer.
    # - if `enable_load_balance` is False, we don't do any load balancing
    #   by passing in `None` as `restore_indices`.
    load_balancer = load_balancer or _create_default_load_balancer(
        seq_length, cp_world_size, device
    )
    restore_indices = (
        load_balancer._generate_indices(restore=True) if load_balancer else None
    )

    assert restore_indices is None or restore_indices.ndim == 2, (
        "load balance restore index expects shape (1, seq_len) or (B, seq_len) "
        f"but got {restore_indices.shape}."
    )
    unsharded_buffers = []
    for b, dim in zip(buffers, seq_dims):
        b = b.contiguous()
        unsharded_b = _maybe_wait(ft_c.all_gather_tensor(b, dim, mesh))

        if restore_indices is not None:
            # NOTE: assuming batch dim is 0
            idx_batch_size = restore_indices.size(0)
            data_batch_size = unsharded_b.size(0)
            if idx_batch_size != 1 and idx_batch_size != data_batch_size:
                raise ValueError(
                    "Cannot restore buffer: "
                    f"restore_indices has shape {restore_indices.shape}, "
                    f"but unsharded_b has shape {unsharded_b.shape}."
                )

            for i in range(data_batch_size):
                index = (
                    restore_indices[0]  # identical load-balance in batch
                    if idx_batch_size == 1
                    else restore_indices[i]
                )
                unsharded_b_batch_i = torch.index_select(
                    unsharded_b[i], dim=dim - 1, index=index
                )
                unsharded_b[i] = unsharded_b_batch_i

        unsharded_buffers.append(unsharded_b)

    return unsharded_buffers


def set_rotate_method(rotate_method: str) -> None:
    """
    Context Parallel SDPA requires the rotation of kv shards. Users can call this
    API to specify which rotation method to use. "alltoall" shuffles the kv shards
    using all-to-all collective. While "allgather" gathers the kv shards using
    all-gather collective after the first sub-SDPA computation. If this API has not
    been called, the default rotate method is "allgather".

    Args:
        rotate_method (str): the rotate method to use. Currently only supports
        "allgather" and "alltoall". If a different string other than these two
        is passed in, the function will raise an error.

    Returns:
        None
    """
    logger.info("Note that FlexAttention CP doesn't support alltoall yet.")
    if rotate_method == "allgather":
        _cp_options.rotate_method = _RotateMethod.ALL_GATHER
    elif rotate_method == "alltoall":
        _cp_options.rotate_method = _RotateMethod.ALL_TO_ALL
    else:
        raise NotImplementedError(
            "Context Parallel does not support "
            f"using {rotate_method} for kv shards rotation"
        )
