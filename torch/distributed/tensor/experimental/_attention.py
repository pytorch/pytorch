import contextlib
import itertools
import logging
import types
import weakref
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Callable, Optional, Protocol, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
)
from torch.overrides import TorchFunctionMode


__all__ = ["context_parallel", "set_rotate_method"]


compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=False, fullgraph=True
)


class _CausalBehavior(Enum):
    SKIP = None
    NOT_IS_CAUSAL = False
    IS_CAUSAL = True


class _RotateMethod(Enum):
    ALL_TO_ALL = auto()
    ALL_GATHER = auto()


aten = torch.ops.aten
logger = logging.getLogger(__name__)


def _need_scaling() -> bool:
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        gcn_arch_name = torch.cuda.get_device_properties("cuda").gcnArchName
        _is_ck_supported = False
        for arch in ["gfx942", "gfx950"]:
            if arch in gcn_arch_name:
                _is_ck_supported = True
        # Check the function exists
        _preferred_rocm_fa_library = torch.backends.cuda.preferred_rocm_fa_library
        _CK_BACKEND = torch.backends.cuda._ROCmFABackends["ck"]
        # Note: it is possible that CK is selected but not compiled in the binary.
        if _is_ck_supported and _preferred_rocm_fa_library() == _CK_BACKEND:
            # Unsure about CK's behavior, keep logsumexp untouched
            return False
        return True
    else:
        return False


class _DispatchMode(Enum):
    MONKEY_PATCH = auto()
    TORCH_FUNCTION = auto()
    TORCH_DISPATCH = auto()


_dispatch_mode: _DispatchMode = _DispatchMode.MONKEY_PATCH


@dataclass
class _ContextParallelOptions:
    # Whether to upcast parameters and gradients to float32 to avoid accumulation
    # errors. It is likely this is always True but we currently keep this variable
    # for the experimental purpose.
    convert_to_f32: bool = True
    enable_load_balance = True
    rotate_method: _RotateMethod = _RotateMethod.ALL_GATHER


_cp_options = _ContextParallelOptions()


@dataclass
class _ContextParallelGlobalVars:
    # The current context parallel impl requires a record of some info
    # as global vars. This dataclass stores those variables.
    # TODO: this var should be able to stored in CP context
    cp_shard_dim: int = 0
    # This variable stores the TorchFunctionMode singleton because using multiple TF
    # instances for dispatching may trigger recompilations
    torch_function_mode: Optional[TorchFunctionMode] = None


_cp_global_vars = _ContextParallelGlobalVars()


def _set_cp_global_var(name: str, value: Any) -> None:
    """Set a global variable for context parallelism."""
    setattr(_cp_global_vars, name, value)


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
    This API partially update a chunk of ``original`` tensor. The ``original``
    tensor will be first chunked along ``dim`` dimension then the ``idx`` chunk
    will be updated with ``new``. If ``add`` is True, the chunk will be added
    with ``new``, otherwise the chunk with be replaced by ``add``.

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
    """A class to help to merge the local SDPA result."""

    def __init__(self, convert_to_f32: bool, seq_dim: int):
        self._seq_dim = seq_dim
        self._out: Optional[torch.Tensor] = None
        self._lse: Optional[torch.Tensor] = None
        self._convert_to_f32 = convert_to_f32
        self._out_dtype = torch.float32
        self._lse_dtype = torch.float32

    def _merge_one(
        self, block_out: torch.Tensor, block_lse: torch.Tensor, partial: bool
    ) -> None:
        block_lse = block_lse.unsqueeze(dim=-1)
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
        out, lse = self._out, self._lse.squeeze(-1)
        return out.to(self._out_dtype), lse.to(self._lse_dtype)


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
    """Use all_to_all to send the kv to the next rank"""

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
    Allgather the kv and return the only the required kv.
    Only one communication will be done.
    """

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._aggregated_buffer: Optional[torch.Tensor] = None
        self._idx = 0

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        # We only need to perform the allgather once.
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
    This is a generalized ring attention implementation that can support multiple attention ops.

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
    For rank1, k0 is not needed for q1 and q2, so only k3 is used for SDPA. This
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

    # Without making key and value contiguous(), the lose curve is bad.
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
            # We need to do SPDA, with only the first local chunk of the k, v.
            # Note that q, k, v, each contains two local chunks.
            ROUND_ROBIN_CYCLE = 2
            q, k, v, partial = (
                query,
                key.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                value.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                False,
            )
        else:
            # Round-robin load balancing case, and i > rank.
            # We need to do SPDA with only the second half of the q, and update
            # only the the second part of  logsumexp. So partial is True.
            # Note that q, k, v, each contains two chunks.
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
        if _need_scaling():
            logsumexp *= 0.6931471805599453
        sdpa_merger.step(out, logsumexp, partial)

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
    """This API implements the backward of the ring attention."""
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
                # We need to do SPDA with only the first half of the k, v.
                # Note that q, k, v, each contains two chunks.
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
                # We need to do SPDA with only the second half of the q
                # Note that q, k, v, each contains two chunks.
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
        # Send the grad key, and grad value to the next rank.
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


customized_ops = {
    aten._scaled_dot_product_flash_attention.default: _sdpa_handler,
    aten._scaled_dot_product_flash_attention_backward.default: _sdpa_handler,
    aten._scaled_dot_product_efficient_attention.default: _sdpa_handler,
    aten._scaled_dot_product_efficient_attention_backward.default: _sdpa_handler,
    aten._scaled_dot_product_cudnn_attention.default: _sdpa_handler,
    aten._scaled_dot_product_cudnn_attention_backward.default: _sdpa_handler,
}


_replaced_functions: dict[Callable, tuple[str, Callable]] = {}


def _distribute_function(
    fn: Callable,
    fn_module: types.ModuleType,
    device_mesh: DeviceMesh,
    input_fn: Optional[Callable] = None,
    output_fn: Optional[Callable] = None,
) -> None:
    """
    ``distribute_function`` is an experimental API that allows users to "distribute"
    the inputs and outputs of a function. Similar to ``distribute_module``, this API
    installs hooks to the ``fn`` to convert the inputs and outputs. There are two
    major differences between ``distribute_function`` and ``distribute_module``.
    First, a function does not have parameters and buffers, as a result,
    ``distribute_function`` itself won't convert any parameters/buffers but simply
    install the input and output hooks.  The tensor conversion will happen in the hooks.
    Another difference is an nn.Module subclass can have several instances and each
    instance be fed into ``distribute_module`` independently with affecting other
    instance. On the other hand, function is a singleton object. So if a function
    is distributed by ``distribute_function`` all subsequent calls to the function
    will invoke the installed hooks.

    Args:
        fn (Callable): the function to be distributed.
        fn_module (types.ModuleType): the Python module that the function is declared.
            e.g., if ``fn`` is ``torch.nn.functional.scaled_dot_product_attention``,
            ``fn_module`` is ``torch.nn.functional``.
        device_mesh (:class:`DeviceMesh`): the device mesh that will be used by the
            input and output hooks to distribute the tensors.
        input_fn (Optional[Callable]): the hook to distribute or convert the input
            arguments of ``fn``.
        output_fn (Optional[Callable]): the hook to distribute or convert the output
            arguments of ``fn``.
    """

    def wrapper(
        target_fn: Callable, input_fn: Optional[Callable], output_fn: Optional[Callable]
    ) -> Callable:
        def inner_fn(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
            if input_fn is not None:
                args, kwargs = input_fn(device_mesh, *args, **kwargs)
            output = target_fn(*args, **kwargs)
            if output_fn is not None:
                output = output_fn(device_mesh, output)
            return output

        return inner_fn

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


@contextlib.contextmanager
def _enable_cp_dispatcher() -> Generator[None, None, None]:
    """Enables DTensor dispatcher to dispatch SDPA to CP."""
    old_handlers = DTensor._op_dispatcher._custom_op_handlers
    DTensor._op_dispatcher._custom_op_handlers = {**old_handlers, **customized_ops}

    yield

    DTensor._op_dispatcher._custom_op_handlers = old_handlers


class _AttentionContextParallel(ParallelStyle):
    """
    Applies context parallel optimizations to the attention layer.

    This will work for nn.MultiHeadedAttention and custom attention layers that
    call F.scaled_dotproduct_attention with a similar signature.

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
        if not device_mesh.ndim == 1:
            raise ValueError("CP only supports single dimension device mesh")

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
        inputs: tuple[Union[torch.Tensor, int, float], ...],
        device_mesh: DeviceMesh,
    ) -> tuple[Union[torch.Tensor, int, float], ...]:
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
                    input.contiguous(), device_mesh, placement, run_check=False
                )

            if isinstance(input, torch.Tensor) and input.requires_grad:
                input.register_hook(backward_hook)

            inp.append(input)

        manager = _enable_cp_dispatcher()
        manager.__enter__()
        cls._CONTEXT_MANAGERS[module] = manager

        return tuple(inp)

    @classmethod
    def _output_fn(
        cls,
        module: nn.Module,
        outputs: Union[torch.Tensor, tuple[Union[torch.Tensor, int, float], ...]],
        device_mesh: DeviceMesh,
    ) -> Union[
        Union[torch.Tensor, int, float], tuple[Union[torch.Tensor, int, float], ...]
    ]:
        cls._CONTEXT_MANAGERS[module].__exit__(None, None, None)
        del cls._CONTEXT_MANAGERS[module]

        def backward_hook(grad: torch.Tensor) -> None:
            if module not in cls._CONTEXT_MANAGERS:
                manager = _enable_cp_dispatcher()
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


def create_cp_block_mask(
    mask_mod: _mask_mod_signature,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device_mesh: DeviceMesh,
) -> BlockMask:
    """
    This API creates a special BlockMask for Context Parallel FlexAttention:
    1. This BlockMask is masking on the attention of Q shard and KV global views, by
    mapping the local q_idx to the global q_idx before sending to mask_mod.
    2. The kv_seq_length (i.e. seq_lengths[1]) of this blockMask is tailored to match
    the sequence length of KV shard instead of KV global. This is to pass the shape check
    in flex_atttention(). The correct value (i.e. the sequence length of KV global) will be
    used in flex_attention once the shape check passes.

    Args:
        mask_mod (Callable): Function to modify the mask over the global attention result.
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Sequence length of query (global view).
        KV_LEN (int): Sequence length of key/value (global view).
        device_mesh (:class:`DeviceMesh`): The device mesh for the context parallelism.

    Return:
        :class:`BlockMask`: the block_mask to be used in flex_attention() within the
        context_parallel() context.

    .. warning::
        This function cannot generate correct block_mask if the BLOCK_SIZE is not
        ``_DEFAULT_SPARSE_BLOCK_SIZE`` which usually happens when the attention
        size is smaller than 128. Please do not use context_parallel() when the
        FlexAttention size is small.
    """
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE

    def _rewrite_mask_mod(
        mask_mod: _mask_mod_signature,
        rank: int,
        world_size: int,
        block_size: int,
        local_q_size: int,
    ) -> _mask_mod_signature:
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
            local_q_idx_to_q_idx(q_idx),
            kv_idx,
        )

    cp_rank = device_mesh.get_local_rank()
    cp_group_size = device_mesh.size()
    Q_SHARD_LEN = Q_LEN // cp_group_size
    block_size = _DEFAULT_SPARSE_BLOCK_SIZE
    block_mask = compiled_create_block_mask(
        _rewrite_mask_mod(mask_mod, cp_rank, cp_group_size, block_size, Q_SHARD_LEN),
        B,
        H,
        Q_SHARD_LEN,
        KV_LEN,
        device=device_mesh.device_type,
        BLOCK_SIZE=(block_size, block_size),
    )
    # flex_attention function checks the following shape so we need to rewrite:
    # key.size(-2) == block_mask.seq_lengths[1]
    seq_lengths = block_mask.seq_lengths
    block_mask.seq_lengths = (seq_lengths[0], seq_lengths[1] // cp_group_size)
    return block_mask


@contextlib.contextmanager
def _context_parallel(seq_dim: int, mesh: DeviceMesh) -> Generator[None, None, None]:
    """Replace SDPA with the CP-wrapped version and enable DTensor CP dispatcher."""

    def attention_input_fn(
        mesh: DeviceMesh, *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        placement = [Shard(seq_dim)]
        all_args = []

        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, torch.Tensor) and not isinstance(arg, DTensor):
                arg = DTensor.from_local(arg, mesh, placement, run_check=False)

            all_args.append(arg)

        new_args = tuple(all_args[0 : len(args)])
        new_kwargs = dict(zip(kwargs.keys(), all_args[len(args) :]))
        return new_args, new_kwargs

    def attention_output_fn(mesh: DeviceMesh, outputs: Any) -> Any:
        new_outputs = []
        for output in [outputs] if isinstance(outputs, torch.Tensor) else outputs:
            output = output.to_local() if isinstance(output, DTensor) else output
            new_outputs.append(output)

        if isinstance(outputs, torch.Tensor):
            return new_outputs[0]

        return tuple(new_outputs)

    def unshard(x: torch.Tensor, mesh: DeviceMesh, shard_dim: int) -> torch.Tensor:
        x = x.contiguous()
        all_xs = [torch.empty_like(x) for _ in range(mesh.size())]
        ft_c.all_gather_inplace(all_xs, x, mesh)
        return torch.cat(all_xs, dim=shard_dim)

    class DistributeFunction(TorchFunctionMode):
        def __init__(
            self,
            fn: Callable,
            device_mesh: DeviceMesh,
            input_fn: Optional[Callable] = None,
            output_fn: Optional[Callable] = None,
        ):
            self._device_mesh = device_mesh
            self._input_fn = input_fn
            self._output_fn = output_fn
            self._fn = fn

        def __torch_function__(
            self,
            func: Callable,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: Optional[dict[str, Any]] = None,
        ) -> Any:
            kwargs = kwargs or {}

            # special handler for flex_attention
            if func == torch._higher_order_ops.flex_attention:
                query, key, value, score_mod, block_mask = args[:5]
                assert isinstance(query, torch.Tensor)
                assert isinstance(key, torch.Tensor)
                assert isinstance(value, torch.Tensor)
                assert isinstance(block_mask, tuple)

                global_key = ft_c.all_gather_tensor_autograd(
                    key, _cp_global_vars.cp_shard_dim, self._device_mesh
                )
                global_value = ft_c.all_gather_tensor_autograd(
                    value, _cp_global_vars.cp_shard_dim, self._device_mesh
                )

                # shape rewrite: because torch.nn.flex_attention() checks
                # the QKV shape against the block_mask object, we need to
                # manually rewrite the shape info in block_mask tuple to
                # make it compatible with q_shard, k_global, v_global
                if block_mask[1] != global_key.size(-2):
                    block_mask = (block_mask[0], global_key.size(-2), *block_mask[2:])

                return func(
                    query,
                    global_key,
                    global_value,
                    score_mod,
                    block_mask,
                    *args[5:],
                    **kwargs,
                )

            if func != self._fn:
                return func(*args, **kwargs)

            if self._input_fn is not None:
                args, kwargs = self._input_fn(self._device_mesh, *args, **kwargs)
            output = func(*args, **kwargs)
            if self._output_fn is not None:
                output = self._output_fn(self._device_mesh, output)
            return output

    if _dispatch_mode == _DispatchMode.MONKEY_PATCH:
        _distribute_function(
            F.scaled_dot_product_attention,
            F,
            mesh,
            attention_input_fn,
            attention_output_fn,
        )
        with _enable_cp_dispatcher():
            yield
        _restore_function(F.scaled_dot_product_attention, F)
    elif _dispatch_mode == _DispatchMode.TORCH_FUNCTION:
        tf_mode = _cp_global_vars.torch_function_mode
        if tf_mode is None:
            tf_mode = DistributeFunction(
                F.scaled_dot_product_attention,
                mesh,
                attention_input_fn,
                attention_output_fn,
            )
            _set_cp_global_var("torch_function_mode", tf_mode)

        with tf_mode:
            with _enable_cp_dispatcher():
                yield
    else:
        raise NotImplementedError("torch dispatch mode is not supported yet.")


def _generate_round_robin_indices(
    seq_length: int,
    cp_world_size: int,
    device: torch.device,
    restore: bool = False,
) -> torch.Tensor:
    """
    Generate round-robin load balancing indices or restore indices.
    Args:
        seq_length: Total sequence length
        cp_world_size: Context parallel world size
        device: Device to place the tensor on
        restore: If True, generate restore indices that map round-robin reordered
                positions back to original positions. If False, generate load
                balance indices that reorder original positions to round-robin pattern.
    Returns:
        Index tensor of shape (seq_length,) with the requested mapping.
    """
    assert seq_length % (cp_world_size * 2) == 0
    chunk_size = seq_length // (cp_world_size * 2)
    all_indices = []

    for cp_rank in range(cp_world_size):
        # Generate indices for first chunk of the cp rank
        first_chunk_start = cp_rank * chunk_size
        first_chunk_indices = list(
            range(first_chunk_start, first_chunk_start + chunk_size)
        )

        # Second chunk: positions from the complementary chunk
        second_chunk_idx = cp_world_size * 2 - cp_rank - 1
        second_chunk_start = second_chunk_idx * chunk_size
        second_chunk_indices = list(
            range(second_chunk_start, second_chunk_start + chunk_size)
        )
        # combine the indices for this rank
        all_indices.extend(first_chunk_indices + second_chunk_indices)
    all_indices_tensor = torch.tensor(all_indices, dtype=torch.int, device=device)
    if restore:
        all_indices_tensor = torch.argsort(all_indices_tensor)
    return all_indices_tensor


def _context_parallel_buffers(
    mesh: DeviceMesh,
    buffers: list[torch.Tensor],
    buffer_seq_dims: list[int],
    load_balance_indices: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    """Shard the buffers along the sequence dimensions according to CP rules."""
    new_buffers = []
    for buffer, seq_dim in zip(buffers, buffer_seq_dims):
        if load_balance_indices is not None:
            buffer = torch.index_select(buffer, dim=seq_dim, index=load_balance_indices)

        # use DTensor to shard the buffer on sequence dimension, retain the local tensor
        sharded_buffer = distribute_tensor(
            buffer, mesh, [Shard(seq_dim)], src_data_rank=None
        ).to_local()
        new_buffers.append(sharded_buffer)

    return new_buffers


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
    if _cp_options.enable_load_balance:
        load_balance_indices = _generate_round_robin_indices(
            seq_length=seq_length,
            cp_world_size=cp_world_size,
            device=device,
        )
    else:
        load_balance_indices = None
    shards = _context_parallel_buffers(
        mesh, buffers, buffer_seq_dims, load_balance_indices
    )
    for buffer, shard in zip(buffers, shards):
        shard = shard.clone()
        buffer.resize_(shard.shape)
        buffer.copy_(shard)

    with _context_parallel(seq_dim=2, mesh=mesh):
        yield

    for buffer, original_buffer in zip(buffers, original_buffers):
        if original_buffer is not None:
            buffer.resize_(original_buffer.shape)
            buffer.copy_(original_buffer)


@torch.no_grad()
def context_parallel_unshard(
    mesh: DeviceMesh,
    buffers: list[torch.Tensor],
    seq_dims: list[int],
) -> list[torch.Tensor]:
    """
    Unshard the tensors (e.g., output) that are sharded due to context parallelism.

    Args:
        mesh (:class:`DeviceMesh`): the device mesh for the context parallelism.
        buffers (List[torch.Tensor]): the buffers to be unsharded.
        seq_dims (List[int]): the sequence dimensions of ``buffers``. This list
            must have the same length as ``buffers``.

    Returns:
        List[torch.Tensor]: the unsharded buffers.
    """
    if _cp_options.enable_load_balance:
        device = buffers[0].device
        cp_world_size = mesh.size()
        seq_length = buffers[0].shape[seq_dims[0]] * cp_world_size
        restore_indices = _generate_round_robin_indices(
            seq_length=seq_length,
            cp_world_size=cp_world_size,
            device=device,
            restore=True,
        )
    else:
        restore_indices = None
    unsharded_buffers = []
    for b, dim in zip(buffers, seq_dims):
        b = b.contiguous()
        unsharded_b = _maybe_wait(ft_c.all_gather_tensor(b, dim, mesh))

        if restore_indices is not None:
            unsharded_b = torch.index_select(
                unsharded_b, dim=dim, index=restore_indices
            )

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
    if rotate_method == "allgather":
        _cp_options.rotate_method = _RotateMethod.ALL_GATHER
    elif rotate_method == "alltoall":
        _cp_options.rotate_method = _RotateMethod.ALL_TO_ALL
    else:
        raise NotImplementedError(
            "Context Parallel does not support "
            f"using {rotate_method} for kv shards rotation"
        )
