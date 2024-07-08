# mypy: allow-untyped-defs
"""This module implements the user facing API for flex_attention in PyTorch."""
import functools
import itertools
import math
import operator
from typing import Callable, Optional

import torch
from torch._higher_order_ops.flex_attention import (
    flex_attention as flex_attention_hop,
    TransformGetItemToIndex,
)
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


class BlockMask:
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
        if kv_indices.dim() < 2:
            raise RuntimeError("BlockMask kv_indices must have at least 2 dimensions")
        self.kv_num_blocks = kv_num_blocks
        self.kv_indices = kv_indices
        self.q_num_blocks = q_num_blocks
        self.q_indices = q_indices
        self.KV_BLOCK_SIZE = KV_BLOCK_SIZE
        self.Q_BLOCK_SIZE = Q_BLOCK_SIZE

    def as_tuple(self):
        return (
            self.kv_num_blocks,
            self.kv_indices,
            self.q_num_blocks,
            self.q_indices,
            self.KV_BLOCK_SIZE,
            self.Q_BLOCK_SIZE,
        )

    def __str__(self):
        s = f"BlockMask(shape={self.shape}, sparsity={self.sparsity():.2f}%, \n"
        mask_str = self.to_string().strip()
        s += mask_str
        s += "\n)"
        return s

    def __getitem__(self, index) -> "BlockMask":
        tensors = self.as_tuple()[:-2]
        tensors = [x[index] for x in tensors]
        return BlockMask(
            tensors[0],
            tensors[1],
            tensors[2],
            tensors[3],
            KV_BLOCK_SIZE=self.KV_BLOCK_SIZE,
            Q_BLOCK_SIZE=self.Q_BLOCK_SIZE,
        )

    @property
    def shape(self):
        """
        Returns the shape of the mask.
        """
        *batch_dims, q_length, _ = self.kv_indices.shape
        q_length = self.kv_num_blocks.shape[-1] * self.KV_BLOCK_SIZE
        kv_length = self.q_num_blocks.shape[-1] * self.Q_BLOCK_SIZE
        return tuple(batch_dims + [q_length, kv_length])

    def numel(self):
        """
        Returns the number of elements (not accounting for sparsity) in the mask.
        """
        shape = self.shape

        def _prod(xs):
            return functools.reduce(operator.mul, xs, 1)

        return _prod(shape)

    def sparsity(self) -> float:
        """
        Computes the percentage of blocks that are sparse (i.e. not computed)
        """
        total_size = self.numel()
        computed_size = (
            self.kv_num_blocks.sum().item() * self.KV_BLOCK_SIZE * self.Q_BLOCK_SIZE
        )
        dense_ratio = computed_size / total_size
        return 100 * (1 - dense_ratio)

    def to_dense(self) -> torch.Tensor:
        """
        Returns a dense block that is equivalent to the block mask.
        """
        num_rows = self.kv_num_blocks.shape[-1]
        num_cols = self.q_num_blocks.shape[-1]
        batch_dims = self.kv_num_blocks.shape[:-1]
        device = self.kv_num_blocks.device

        def create_dense_one(kv_num_blocks, kv_indices):
            dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)

            row_indices = torch.arange(
                num_rows, dtype=torch.int, device=device
            ).unsqueeze(-1)
            col_indices = torch.arange(num_cols, dtype=torch.int, device=device)
            index_mask = col_indices < kv_num_blocks.unsqueeze(-1)

            # We write to one spot "out of bounds"
            valid_indices = torch.where(index_mask, kv_indices, num_cols)

            # set the values in 'a' to 1 where the indices are valid
            dense_mask[row_indices, valid_indices] = torch.tensor(
                1, device=dense_mask.device, dtype=dense_mask.dtype
            )
            return dense_mask[:, :num_cols]

        create_dense_batched = create_dense_one
        for _ in range(len(batch_dims)):
            create_dense_batched = torch.vmap(create_dense_batched, in_dims=(0, 0))

        out = create_dense_batched(self.kv_num_blocks, self.kv_indices)
        return out

    def to_string(self, grid_size=(20, 20), limit=4):
        """
        Returns a string representation of the block mask. Quite nifty.

        If grid_size is None, prints out an uncompressed version. Warning, it can be quite big!
        """
        dense_mask = self.to_dense()
        *batch_dims, num_rows, num_cols = dense_mask.shape
        if isinstance(grid_size, int):
            max_rows = grid_size
            max_cols = grid_size
        elif grid_size == -1:
            max_rows = num_rows
            max_cols = num_cols
        else:
            max_rows, max_cols = grid_size

        def create_block_vis(*batch_idx):
            descriptors = []

            descriptors.append(f"{batch_idx}")

            vis = ", ".join(reversed(descriptors)) + "\n"

            def summarize_section(section):
                percentage = section.float().mean().item()
                if percentage == 1:
                    return "█"
                elif percentage == 0:
                    return " "
                else:
                    return "░"

            def cdiv(a, b):
                return (a + (b - 1)) // b

            row_step = max(1, cdiv(num_rows, max_rows))
            col_step = max(1, cdiv(num_cols, max_cols))

            for r in range(0, num_rows, row_step):
                for c in range(0, num_cols, col_step):
                    cur_mask = dense_mask
                    for idx in batch_idx:
                        cur_mask = cur_mask[idx]
                    char = summarize_section(
                        cur_mask[r : r + row_step, c : c + col_step]
                    )
                    vis += char * 2
                vis += "\n"
            return vis

        total_vis = []
        for idx, batch_idx in enumerate(
            itertools.product(*[range(i) for i in batch_dims])
        ):
            if idx == limit:
                total_vis.append("...")
                total_vis.append("To print out more, set BlockMask.to_string(limit=N)")
                total_vis.append(
                    "You can also index (BlockMask[batch, head]) to choose a specific batch or head"
                )
                break
            block_vis = create_block_vis(*batch_idx)
            total_vis.append(block_vis)

        return "\n".join(total_vis)


def _broadcast_to_dim(x, dim):
    while x.dim() < dim:
        x = x.unsqueeze(0)
    return x


def _convert_mask_to_block_mask(
    mask,
    KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
):
    assert mask.dtype == torch.bool
    mask = _broadcast_to_dim(mask, 4)
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


def _create_sparse_block_from_block_mask(
    block_mask: torch.Tensor,
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
) -> BlockMask:
    device = block_mask.device
    block_mask = block_mask.to(dtype=torch.int8)
    kv_num_blocks = block_mask.sum(dim=3)
    kv_indices = torch.argsort(block_mask, dim=3, descending=True, stable=True)
    q_num_blocks = block_mask.sum(dim=2)
    q_indices = torch.argsort(block_mask, dim=2, descending=True, stable=True).permute(
        0, 1, 3, 2
    )
    return BlockMask(
        kv_num_blocks=kv_num_blocks.to(torch.int32).to(device).contiguous(),
        kv_indices=kv_indices.to(torch.int32).to(device).contiguous(),
        q_num_blocks=q_num_blocks.to(torch.int32).to(device).contiguous(),
        q_indices=q_indices.to(torch.int32).to(device).contiguous(),
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
    )


def _create_mask(
    score_mod: _score_mod_signature,
    B: int,
    H: int,
    M: int,
    N: int,
    device: str = "cuda",
    _compiled: bool = False,
):
    r"""This function creates a mask tensor from a score_mod function.

    Args:
        score_mod (Callable): Function to modify attention scores.
        B (int): Batch size.
        H (int): Number of heads.
        M (int): Sequence length of query.
        N (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.

    Returns:
        mask (Tensor): A mask tensor with shape (B, H, M, N).
    """
    from contextlib import nullcontext

    b = torch.arange(0, B, device=device)
    h = torch.arange(0, H, device=device)
    m = torch.arange(0, M, device=device)
    n = torch.arange(0, N, device=device)
    # TODO: fix this
    # A hack required because of lack of torchfunctionmode support
    # Working around some bugs with compiling vmap
    if _compiled:
        ctx = nullcontext()
    else:
        ctx = TransformGetItemToIndex()  # type: ignore[assignment]
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0))
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None))
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None))
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None))

    with ctx:
        out = score_mod(torch.zeros(B, H, M, N, device=device), b, h, m, n)
        mask = torch.where(torch.isneginf(out), False, True)
    return mask


# Done as a workaround around torch.compile not compiling what we want in the
# presence of the torchfunctionmdoe
def _create_block_mask_inner(
    score_mod, B, H, M, N, device, KV_BLOCK_SIZE, Q_BLOCK_SIZE
):
    mask = _create_mask(score_mod, B, H, M, N, device, _compiled=True)
    block_mask = _convert_mask_to_block_mask(
        mask, KV_BLOCK_SIZE=KV_BLOCK_SIZE, Q_BLOCK_SIZE=Q_BLOCK_SIZE
    )
    return block_mask.to(dtype=torch.int8)


def create_block_mask(
    score_mod: _score_mod_signature,
    B: int,
    H: int,
    M: int,
    N: int,
    device: str = "cuda",
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    _compiled=False,
):
    r"""This function creates a block mask tuple from a score_mod function.

    Args:
        score_mod (Callable): Function to modify attention scores.
        B (int): Batch size.
        H (int): Number of heads.
        M (int): Sequence length of query.
        N (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.
        KV_BLOCK_SIZE (int): Block size of block mask for each query.
        Q_BLOCK_SIZE (int): Block size of block mask for each key/value.

    Returns:
        block_mask (tuple): A tuple of (kv_num_blocks, kv_indices, q_num_blocks, q_indices,
                            KV_BLOCK_SIZE, Q_BLOCK_SIZE) which represents the block mask.
    """
    inner_func = _create_block_mask_inner
    # This is kind of a temporary hack to workaround some issues
    if _compiled:
        inner_func = torch.compile(inner_func, fullgraph=True, dynamic=False)
    with TransformGetItemToIndex():
        block_mask = inner_func(
            score_mod, B, H, M, N, device, KV_BLOCK_SIZE, Q_BLOCK_SIZE
        )
    return _create_sparse_block_from_block_mask(block_mask)


"""
    The flex attention kernels are implemented using block sparsity,
    where only the unmasked blocks are computed to get the best perf.
    If users don't specify any block sparse mask info, we create this
    empty block sparse mask with all blocks unmasked as the default one.
"""


def _create_empty_block_mask(query, key, value) -> BlockMask:
    device = query.device
    kv_len = key.size()[-2]
    q_len = query.size()[-2]
    return BlockMask(
        kv_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32, device=device),
        kv_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device),
        q_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32, device=device),
        q_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device),
        KV_BLOCK_SIZE=kv_len,
        Q_BLOCK_SIZE=q_len,
    )


def flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: _score_mod_signature = _identity,
    block_mask: Optional[BlockMask] = None,
    scale=None,
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
        - ``b``, ``h``, ``q_idx``, ``kv_idx``: Scalar tensors indicating
          the batch index, head index, query index, and key/value index, respectively.
          These should have the ``torch.int`` data type and be located on the same device as the score tensor.

    Args:
        query (Tensor): Query tensor; shape :math:`(B, H, L, E)`.
        key (Tensor): Key tensor; shape :math:`(B, H, S, E)`.
        value (Tensor): Value tensor; shape :math:`(B, H, S, Ev)`.
        score_mod (Callable): Function to modify attention scores. By default no score_mod is applied.
        block_mask (BlockMask): BlockMask object that controls the blocksparsity pattern of the attention.
        scale (float): Scaling factor applied prior to softmax. If none, the default value is set to :math`\frac{1}{\sqrt{E}}`

    Returns:
        output (Tensor): Attention output; shape :math:`(B, H, L, Ev)`.

    Shape legend:
        - :math:`N: \text{Batch size} ... : \text{Any number of other batch dimensions (optional)}`
        - :math:`S: \text{Source sequence length}`
        - :math:`L: \text{Target sequence length}`
        - :math:`E: \text{Embedding dimension of the query and key}`
        - :math:`Ev: \text{Embedding dimension of the value}`

    .. warning::
        `torch.nn.attention.flex_attention` is a prototype feature in PyTorch.
        Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    """

    if block_mask is None:
        block_mask = _create_empty_block_mask(query, key, value)
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    if torch.compiler.is_dynamo_compiling():
        # mark head_dim always to be static
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -1)
        out, _ = flex_attention_hop(
            query, key, value, score_mod, block_mask.as_tuple(), scale=scale
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
                )(query, key, value, score_mod, block_mask.as_tuple(), scale=scale)
                return out


# Shim for some temporary BC
_flex_attention = flex_attention
_create_block_mask = create_block_mask

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
