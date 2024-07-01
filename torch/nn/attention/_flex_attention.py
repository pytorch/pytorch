# mypy: allow-untyped-defs
"""This module implements the user facing API for flex_attention in PyTorch."""
import functools
from typing import Callable, Optional

import torch
from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop
from torch._higher_order_ops.utils import _set_compilation_env
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_pre_dispatch_torch_function_mode,
)
from torch.nn.attention._utils import _validate_sdpa_input


def _compose(*fs):
    """Compose a sequence of score_mod functions."""

    def compose2(f, g):
        def inner(score, b, h, m, n):
            return f(g(score, b, h, m, n), b, h, m, n)

        return inner

    return functools.reduce(compose2, fs)


_score_mod_signature = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


def _identity(
    score: torch.Tensor,
    batch: torch.Tensor,
    head: torch.Tensor,
    token_q: torch.Tensor,
    token_kv: torch.Tensor,
) -> torch.Tensor:
    return score


_DEFAULT_SPARSE_BLOCK_SIZE = 128


class _BlockSparseMask:
    kv_num_blocks: torch.Tensor
    kv_indices: torch.Tensor
    q_num_blocks: torch.Tensor
    q_indices: torch.Tensor
    KV_BLOCK_SIZE: int
    Q_BLOCK_SIZE: int

    def __init__(
        self,
        kv_num_blocks,
        kv_indices,
        q_num_blocks,
        q_indices,
        KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
        Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    ):
        self.kv_num_blocks = kv_num_blocks
        self.kv_indices = kv_indices
        self.q_num_blocks = q_num_blocks
        self.q_indices = q_indices
        self.KV_BLOCK_SIZE = KV_BLOCK_SIZE
        self.Q_BLOCK_SIZE = Q_BLOCK_SIZE


def broadcast_to_dim(x, dim):
    while x.dim() < dim:
        x = x.unsqueeze(0)
    return x


def _convert_mask_to_block_mask(
    mask,
    KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
):
    assert mask.dtype == torch.bool
    mask = broadcast_to_dim(mask, 4)
    B, H, Q, KV = mask.shape
    assert Q % Q_BLOCK_SIZE == 0
    assert KV % KV_BLOCK_SIZE == 0
    mask = mask.view(
        B, H, Q // Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV // KV_BLOCK_SIZE, KV_BLOCK_SIZE
    )  # [B, H, Q//Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask = mask.permute(
        0, 1, 2, 4, 3, 5
    )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, Q_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask = mask.sum(dim=[-2, -1]) > 0  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE]
    return mask


def _convert_block_mask_to_mask(
    block_mask,
    KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
):
    assert block_mask.dim() == 4
    B, H, Q, KV = block_mask.shape
    block_mask = block_mask.expand(Q_BLOCK_SIZE, KV_BLOCK_SIZE, *block_mask.shape)
    block_mask = block_mask.permute(2, 3, 4, 0, 5, 1).reshape(
        B, H, Q * Q_BLOCK_SIZE, KV * KV_BLOCK_SIZE
    )
    return block_mask


def _create_block_sparse_mask(
    mask: torch.Tensor,
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
):
    block_mask = _convert_mask_to_block_mask(
        mask, KV_BLOCK_SIZE=KV_BLOCK_SIZE, Q_BLOCK_SIZE=Q_BLOCK_SIZE
    )
    block_mask = block_mask.to(dtype=torch.int8)
    kv_num_blocks = block_mask.sum(dim=3)
    kv_indices = torch.argsort(block_mask, dim=3, descending=True, stable=True)
    q_num_blocks = block_mask.sum(dim=2)
    q_indices = torch.argsort(block_mask, dim=2, descending=True, stable=True).permute(
        0, 1, 3, 2
    )
    return _BlockSparseMask(
        kv_num_blocks=kv_num_blocks.to(torch.int32).to(mask.device).contiguous(),
        kv_indices=kv_indices.to(torch.int32).to(mask.device).contiguous(),
        q_num_blocks=q_num_blocks.to(torch.int32).to(mask.device).contiguous(),
        q_indices=q_indices.to(torch.int32).to(mask.device).contiguous(),
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
    )


"""
    The flex attention kernels are implemented using block sparsity,
    where only the unmasked blocks are computed to get the best perf.
    If users don't specify any block sparse mask info, we create this
    empty block sparse mask with all blocks unmasked as the default one.
"""


def _create_empty_block_sparse_mask(query, key, value):
    device = query.device
    kv_len = key.size()[-2]
    q_len = query.size()[-2]
    return _BlockSparseMask(
        kv_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32, device=device),
        kv_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device),
        q_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32, device=device),
        q_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device),
        KV_BLOCK_SIZE=kv_len,
        Q_BLOCK_SIZE=q_len,
    )


def _flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: _score_mod_signature = _identity,
    block_sparse_mask: Optional[_BlockSparseMask] = None,
) -> torch.Tensor:
    r"""This function implements scaled dot product attention with an arbitrary attention score modification function.

    This function computes the scaled dot product attention between query, key, and value tensors with a user-defined
    attention score modification function. The attention score modification function will be applied after the attention
    scores have been calculated between the query and key tensors. The attention scores are calculated as follows:

    The ``score_mod`` function should have the following signature:

    .. code-block:: python

        def score_mod(
            score: torch.Tensor,
            batch: torch.Tensor,
            head: torch.Tensor,
            token_q: torch.Tensor,
            token_kv: torch.Tensor
        ) -> torch.Tensor:

    Where:
        - ``score``: A scalar tensor representing the attention score,
          with the same data type and device as the query, key, and value tensors.
        - ``batch``, ``head``, ``token_q``, ``token_kv``: Scalar tensors indicating
          the batch index, head index, query index, and key/value index, respectively.
          These should have the ``torch.int`` data type and be located on the same device as the score tensor.

    Args:
        query (Tensor): Query tensor; shape :math:`(B, H, L, E)`.
        key (Tensor): Key tensor; shape :math:`(B, H, S, E)`.
        value (Tensor): Value tensor; shape :math:`(B, H, S, Ev)`.
        score_mod (Callable): Function to modify attention scores. By default no score_mod is applied.

    Returns:
        output (Tensor): Attention output; shape :math:`(B, H, L, Ev)`.

    Shape legend:
        - :math:`N: \text{Batch size} ... : \text{Any number of other batch dimensions (optional)}`
        - :math:`S: \text{Source sequence length}`
        - :math:`L: \text{Target sequence length}`
        - :math:`E: \text{Embedding dimension of the query and key}`
        - :math:`Ev: \text{Embedding dimension of the value}`

    .. warning::
        `torch.nn.attention.flex_attention` is a prototype feature in PyTorch. It doesn't support training currently.
        Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    """

    if block_sparse_mask is None:
        block_sparse_mask = _create_empty_block_sparse_mask(query, key, value)
    if torch.compiler.is_dynamo_compiling():
        # mark head_dim always to be static
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -1)
        out, _ = flex_attention_hop(
            query,
            key,
            value,
            score_mod,
            block_sparse_mask.kv_num_blocks,
            block_sparse_mask.kv_indices,
            block_sparse_mask.q_num_blocks,
            block_sparse_mask.q_indices,
            block_sparse_mask.KV_BLOCK_SIZE,
            block_sparse_mask.Q_BLOCK_SIZE,
        )
        return out

    # Some basic input validation
    _validate_sdpa_input(query, key, value)
    if query.size(-2) % 128 != 0:
        raise ValueError("NYI: S and L must be a multiple of 128")

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("flex_attention requires dynamo support.")

    with _set_compilation_env():
        with torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_pre_dispatch_torch_function_mode():
                out, _ = torch.compile(
                    flex_attention_hop, backend="eager", fullgraph=True
                )(
                    query,
                    key,
                    value,
                    score_mod,
                    block_sparse_mask.kv_num_blocks,
                    block_sparse_mask.kv_indices,
                    block_sparse_mask.q_num_blocks,
                    block_sparse_mask.q_indices,
                    block_sparse_mask.KV_BLOCK_SIZE,
                    block_sparse_mask.Q_BLOCK_SIZE,
                )
                return out


"""Some common used score_mod functions for flex_attention in PyTorch."""


def _causal(
    score: torch.Tensor,
    batch: torch.Tensor,
    head: torch.Tensor,
    token_q: torch.Tensor,
    token_kv: torch.Tensor,
) -> torch.Tensor:
    return torch.where(token_q >= token_kv, score, float("-inf"))


def _rel_bias(
    score: torch.Tensor,
    batch: torch.Tensor,
    head: torch.Tensor,
    token_q: torch.Tensor,
    token_kv: torch.Tensor,
) -> torch.Tensor:
    return score + (token_q - token_kv)


def _rel_causal(
    score: torch.Tensor,
    batch: torch.Tensor,
    head: torch.Tensor,
    token_q: torch.Tensor,
    token_kv: torch.Tensor,
) -> torch.Tensor:
    return torch.where(token_q >= token_kv, score + (token_q - token_kv), float("-inf"))


def _generate_alibi_bias(num_heads: int):
    def _alibi_bias(
        score: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        token_q: torch.Tensor,
        token_kv: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.exp2(-((head + 1) * 8.0 / num_heads))
        return score + (token_kv - token_q) * scale

    return _alibi_bias
