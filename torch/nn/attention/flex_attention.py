# mypy: allow-untyped-defs
# flake8: noqa: B950
"""This module implements the user facing API for flex_attention in PyTorch."""

import functools
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Callable
from enum import Enum
from typing import Any, NamedTuple, Union

import torch
from torch import Tensor


try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop
from torch._higher_order_ops.utils import _set_compilation_env
from torch._prims_common import DeviceLikeType
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    _temp_remove_pre_dispatch_torch_function_mode,
)
from torch.nn.attention._utils import _validate_sdpa_input
from torch.utils._pytree import GetAttrKey, tree_map_only


# Private debug flag to disable internal compilation wrapping for debugging purposes.
# WARNING: This is intended ONLY for debugging score_mod and mask_mod functions.
# When enabled, this bypasses the required internal compilation that ensures correctness
# and performance. Only use this temporarily when you need to set breakpoints
# in your score_mod/mask_mod functions during development.
#
# This flag only affects the internal compilation when flex_attention is called directly.
# If you have already wrapped flex_attention in torch.compile(), this flag has no effect
# and the user's compilation will still occur.
#
# Usage:
#   import torch.nn.attention.flex_attention as fa
#   fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
#   # Now you can set breakpoints in your score_mod/mask_mod
#   output = fa.flex_attention(q, k, v, score_mod=my_score_mod)
#
_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = False

_WARNINGS_SHOWN: set[str] = set()


def _warn_once(
    warning_id: str, message: str, category: type[Warning] = UserWarning
) -> None:
    """Helper to ensure each warning is shown only once per process."""
    if warning_id not in _WARNINGS_SHOWN:
        warnings.warn(message, category, stacklevel=2)
        _WARNINGS_SHOWN.add(warning_id)


__all__ = [
    "BlockMask",
    "flex_attention",
    "AuxOutput",
    "AuxRequest",
    "FlexKernelOptions",
    "create_block_mask",
    "create_mask",
    "or_masks",
    "and_masks",
    "noop_mask",
]

_score_mod_signature = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
_mask_mod_signature = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


# pyrefly: ignore [invalid-inheritance]
class FlexKernelOptions(TypedDict, total=False):
    """Options for controlling the behavior of FlexAttention kernels.

    These options are passed to the underlying Triton kernels to control performance
    and numerical behavior. Most users will not need to specify these options as the
    default autotuning provides good performance.

    The options can be prefixed with ``fwd_`` or ``bwd_`` to apply only to forward or
    backward pass respectively. For example: ``fwd_BLOCK_M`` and ``bwd_BLOCK_M1``.

    Note:
      We currently do not provide any backward compatibility guarantees for these options.
      That being said most of these have remained pretty stable since their introduction. But
      We do not consider this part of the public API just yet. We think that some documentation
      Is better than secret hidden flags, but we may change these options in the future.

    Example Usage:
        .. code-block:: python

            # Using dictionary (backward compatible)
            kernel_opts = {"BLOCK_M": 64, "BLOCK_N": 64, "PRESCALE_QK": True}
            output = flex_attention(q, k, v, kernel_options=kernel_opts)

            # Using TypedDict (recommended for type safety)
            from torch.nn.attention.flex_attention import FlexKernelOptions

            kernel_opts: FlexKernelOptions = {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "PRESCALE_QK": True,
            }
            output = flex_attention(q, k, v, kernel_options=kernel_opts)

            # Forward/backward specific options
            kernel_opts: FlexKernelOptions = {
                "fwd_BLOCK_M": 64,
                "bwd_BLOCK_M1": 32,
                "PRESCALE_QK": False,
            }
            output = flex_attention(q, k, v, kernel_options=kernel_opts)
    """

    # Performance tuning options
    # pyrefly: ignore [invalid-annotation]
    num_warps: NotRequired[int]
    """Number of warps to use in the CUDA kernel. Higher values may improve performance
    but increase register pressure. Default is determined by autotuning."""

    # pyrefly: ignore [invalid-annotation]
    num_stages: NotRequired[int]
    """Number of pipeline stages in the CUDA kernel. Higher values may improve performance
    but increase shared memory usage. Default is determined by autotuning."""

    # pyrefly: ignore [invalid-annotation]
    BLOCK_M: NotRequired[int]
    """Thread block size for the sequence length dimension of Q in forward pass.
    Must be a power of 2. Common values: 16, 32, 64, 128. Default is determined by autotuning."""

    # pyrefly: ignore [invalid-annotation]
    BLOCK_N: NotRequired[int]
    """Thread block size for the sequence length dimension of K/V in forward pass.
    Must be a power of 2. Common values: 16, 32, 64, 128. Default is determined by autotuning."""

    # Backward-specific block sizes (when prefixed with 'bwd_')
    # pyrefly: ignore [invalid-annotation]
    BLOCK_M1: NotRequired[int]
    """Thread block size for Q dimension in backward pass. Use as 'bwd_BLOCK_M1'.
    Default is determined by autotuning."""

    # pyrefly: ignore [invalid-annotation]
    BLOCK_N1: NotRequired[int]
    """Thread block size for K/V dimension in backward pass. Use as 'bwd_BLOCK_N1'.
    Default is determined by autotuning."""

    # pyrefly: ignore [invalid-annotation]
    BLOCK_M2: NotRequired[int]
    """Thread block size for second Q dimension in backward pass. Use as 'bwd_BLOCK_M2'.
    Default is determined by autotuning."""

    # pyrefly: ignore [invalid-annotation]
    BLOCK_N2: NotRequired[int]
    """Thread block size for second K/V dimension in backward pass. Use as 'bwd_BLOCK_N2'.
    Default is determined by autotuning."""

    # pyrefly: ignore [invalid-annotation]
    PRESCALE_QK: NotRequired[bool]
    """Whether to pre-scale QK by 1/sqrt(d) and change of base. This is slightly faster but
    may have more numerical error. Default: False."""

    # pyrefly: ignore [invalid-annotation]
    ROWS_GUARANTEED_SAFE: NotRequired[bool]
    """If True, guarantees that at least one value in each row is not masked out.
    Allows skipping safety checks for better performance. Only set this if you are certain
    your mask guarantees this property. For example, causal attention is guaranteed safe
    because each query has at least 1 key-value to attend to. Default: False."""

    # pyrefly: ignore [invalid-annotation]
    BLOCKS_ARE_CONTIGUOUS: NotRequired[bool]
    """If True, guarantees that all blocks in the mask are contiguous.
    Allows optimizing block traversal. For example, causal masks would satisfy this,
    but prefix_lm + sliding window would not. Default: False."""

    # pyrefly: ignore [invalid-annotation]
    WRITE_DQ: NotRequired[bool]
    """Controls whether gradient scatters are done in the DQ iteration loop of the backward pass.
    Setting this to False will force this to happen in the DK loop which depending on your
    specific score_mod and mask_mod might be faster. Default: True."""

    # pyrefly: ignore [invalid-annotation]
    FORCE_USE_FLEX_ATTENTION: NotRequired[bool]
    """If True, forces the use of the flex attention kernel instead of potentially using
    the more optimized flex-decoding kernel for short sequences. This can be a helpful
    option for debugging. Default: False."""

    # pyrefly: ignore [invalid-annotation]
    USE_TMA: NotRequired[bool]
    """Whether to use Tensor Memory Accelerator (TMA) on supported hardware.
    This is experimental and may not work on all hardware, currently specific
    to NVIDIA GPUs Hopper+. Default: False."""

    # ROCm-specific options
    # pyrefly: ignore [invalid-annotation]
    kpack: NotRequired[int]
    """ROCm-specific kernel packing parameter."""

    # pyrefly: ignore [invalid-annotation]
    matrix_instr_nonkdim: NotRequired[int]
    """ROCm-specific matrix instruction non-K dimension."""

    # pyrefly: ignore [invalid-annotation]
    waves_per_eu: NotRequired[int]
    """ROCm-specific waves per execution unit."""

    # pyrefly: ignore [invalid-annotation]
    force_flash: NotRequired[bool]
    """ If True, forces use of the cute-dsl flash attention kernel.

    Raises an error if flash attention cannot be used instead of falling back
    to the default implementation. Useful for ensuring flash attention is used
    when expected.
    """


class AuxRequest(NamedTuple):
    """Request which auxiliary outputs to compute from flex_attention.

    Each field is a boolean indicating whether that auxiliary output should be computed.
    """

    lse: bool = False
    max_scores: bool = False


class AuxOutput(NamedTuple):
    """Auxiliary outputs from flex_attention operation.

    Fields will be None if not requested, or contain the tensor if requested.
    """

    lse: Tensor | None = None
    max_scores: Tensor | None = None


class _ModificationType(Enum):
    """Enum for the type of modification function.
    - SCORE_MOD: score_mod function which accepts a score as the first argument
    - mask_mod: mask function which does not accept a score and is only used for generating
    block mask
    """

    SCORE_MOD = 1
    MASK_MOD = 2
    UNKNOWN = 3


def _get_mod_type(fn: Callable) -> _ModificationType:
    """Get the type of modification function.
    This function inspects the number of positional arguments of the function to determine
    the type of modification function. If the function has 5 positional arguments, it is
    considered as a score_mod function. If the function has 4 positional arguments, it is
    considered as a mask function.
    """
    if hasattr(fn, "__code__"):
        code = fn.__code__
        num_positional_total = code.co_argcount
        defaults = ()
        if hasattr(fn, "__defaults__"):
            defaults = fn.__defaults__ or ()
        num_defaults = len(defaults)
        num_positional_args = num_positional_total - num_defaults
    else:
        num_positional_args = sum(
            1
            for param in inspect.signature(fn).parameters.values()
            if param.default is inspect.Parameter.empty
        )
    assert num_positional_args == 5 or num_positional_args == 4
    if num_positional_args == 5:
        return _ModificationType.SCORE_MOD
    elif num_positional_args == 4:
        return _ModificationType.MASK_MOD
    else:
        return _ModificationType.UNKNOWN


# Need to define it here so that Dynamo doesn't skip it
def _vmap_for_bhqkv(
    fn: Callable,
    prefix: tuple[int | None, ...],
    suffix: tuple[int | None, ...] = (),
    out_dims: Union[int, list[int | None]] = 0,
    group_dim: bool = False,
):
    """Used to vmap both score_mods and mask_mods over 4-dimensional/5-dimension inputs.
    Mapping over the [b, hq, q_idx, kv_idx] or [b, hkv, g, q_idx, kv_idx] dimensions.

    Args:
        fn (callable): The function to vmap.
        prefix (tuple): The prefix of the vmap. For score mod functions,
                        this should be set to (0,). For mask_mods = ()
        suffix (tuple): We need to add (0,) if gradOut is being mapped over,
                        and (None,) * len(other_buffers).
        out_dims (tuple): For forward cases, keep this as the default 0 since
                          we are only returning 1 output. For backwards, the joint
                          graph returns grads for B, H, Q_idx, KV_idx and other_buffers,
                          so we set this to (0, None, None, None, None) + (None,) * len(other_buffers).

    Returns:
        callable: The vmapped function.
    """
    # We vamp a function 4 times, broadcasting the [b, h, q_idx, kv_idx] dimensions
    dimensions: list[tuple[None | int, None | int, None | int, None | int]] = []
    dimensions = [
        (None, None, None, 0),
        (None, None, 0, None),
        (None, 0, None, None),
    ]

    if group_dim:
        dimensions += [
            (None, 0, None, None),
        ]

    dimensions += [
        (0, None, None, None),
    ]

    for dims in dimensions:
        fn = torch.vmap(fn, in_dims=prefix + dims + suffix, out_dims=out_dims)  # type: ignore[arg-type]
    return fn


def _identity(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return score


def noop_mask(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    """Returns a noop mask_mod"""
    return batch.new_ones(size=(), dtype=torch.bool, device=batch.device)


def _sliced_mask_mod_error(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    """
    Raises helpful error when using mask_mod from a sliced BlockMask.

    After slicing a BlockMask, the mask_mod is reset and cannot be used directly.
    Users must reassign mask_mod from the original (unsliced) BlockMask.
    """
    raise RuntimeError(
        "Cannot use mask_mod from a sliced BlockMask. "
        "When you slice a BlockMask using [], the mask_mod attribute is reset. "
        "You must set it from the original BlockMask's mask_mod."
        "\n\nIncorrect usage:"
        "\n  base_mask = create_block_mask(my_mask_fn, ...)"
        "\n  sliced_mask = base_mask[:, :, block_idx]"
        "\n  sliced_mask.mask_mod = apply_offset(sliced_mask.mask_mod, offset)  # WRONG!"
        "\n\nCorrect usage:"
        "\n  base_mask = create_block_mask(my_mask_fn, ...)"
        "\n  sliced_mask = base_mask[:, :, block_idx]"
        "\n  sliced_mask.mask_mod = apply_offset(base_mask.mask_mod, offset)  # Use base_mask!"
    )


_DEFAULT_SPARSE_BLOCK_SIZE = 128
_LARGE_SPARSE_BLOCK_SIZE = 1 << 30


def _ordered_to_dense(num_blocks_in_row: Tensor, col_indices: Tensor):
    num_rows = col_indices.shape[-2]
    num_cols = col_indices.shape[-1]
    batch_dims = num_blocks_in_row.shape[:-1]
    device = num_blocks_in_row.device

    def create_dense_one(kv_num_blocks, kv_indices):
        dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)

        row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
            -1
        )
        col_range = torch.arange(num_cols, dtype=torch.int, device=device)
        index_mask = col_range < kv_num_blocks.unsqueeze(-1)

        # We write to one spot "out of bounds"
        valid_indices = torch.where(index_mask, kv_indices, num_cols)

        # set the values in 'a' to 1 where the indices are valid
        dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        return dense_mask[:, :num_cols].contiguous()

    create_dense_batched = create_dense_one
    for _ in range(len(batch_dims)):
        create_dense_batched = torch.vmap(create_dense_batched, in_dims=(0, 0))

    out = create_dense_batched(num_blocks_in_row, col_indices)
    return out


def _dense_to_ordered(dense_mask) -> tuple[Tensor, Tensor]:
    dense_mask = dense_mask.to(dtype=torch.int32)
    num_blocks_in_row = dense_mask.sum(dim=-1)
    col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
    return (
        num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        col_indices.to(torch.int32, memory_format=torch.contiguous_format),
    )


def _transpose_ordered(num_blocks_in_row: Tensor, col_indices: Tensor):
    dense = _ordered_to_dense(num_blocks_in_row, col_indices)
    return _dense_to_ordered(dense.transpose(-2, -1))


def _adjust_num_blocks_and_indices(
    num_blocks: Tensor,
    indices: Tensor,
    new_num_rows: int,
    new_num_cols: int,
):
    indices = indices[:, :, :new_num_rows, :new_num_cols]
    num_blocks = num_blocks[:, :, :new_num_rows]
    num_blocks = torch.where(num_blocks < new_num_cols, num_blocks, new_num_cols)
    num_blocks = torch.sum(indices < num_blocks[:, :, :, None], dim=-1).to(torch.int32)
    return num_blocks, indices


class BlockMask:
    r"""
    BlockMask is our format for representing a block-sparse attention mask.
    It is somewhat of a cross in-between BCSR and a non-sparse format.

    **Basics**

    A block-sparse mask means that instead of representing the sparsity of
    individual elements in the mask, a KV_BLOCK_SIZE x Q_BLOCK_SIZE block is
    considered sparse only if every element within that block is sparse.
    This aligns well with hardware, which generally expects to perform
    contiguous loads and computation.

    This format is primarily optimized for 1. simplicity, and 2. kernel
    efficiency. Notably, it is *not* optimized for size, as this mask is always
    reduced by a factor of KV_BLOCK_SIZE * Q_BLOCK_SIZE. If the size is a
    concern, the tensors can be reduced in size by increasing the block size.

    The essentials of our format are:

    num_blocks_in_row: Tensor[ROWS]:
    Describes the number of blocks present in each row.

    col_indices: Tensor[ROWS, MAX_BLOCKS_IN_COL]:
    `col_indices[i]` is the sequence of block positions for row i. The values of
    this row after `col_indices[i][num_blocks_in_row[i]]` are undefined.

    For example, to reconstruct the original tensor from this format:

    .. code-block:: python

        dense_mask = torch.zeros(ROWS, COLS)
        for row in range(ROWS):
            for block_idx in range(num_blocks_in_row[row]):
                dense_mask[row, col_indices[row, block_idx]] = 1

    Notably, this format makes it easier to implement a reduction along the
    *rows* of the mask.

    **Details**

    The basics of our format require only kv_num_blocks and kv_indices. But, we
    have up to 8 tensors on this object. This represents 4 pairs:

    1. (kv_num_blocks, kv_indices): Used for the forwards pass of attention, as
    we reduce along the KV dimension.

    2. [OPTIONAL] (full_kv_num_blocks, full_kv_indices): This is optional and
    purely an optimization. As it turns out, applying masking to every block
    is quite expensive! If we specifically know which blocks are "full" and
    don't require masking at all, then we can skip applying mask_mod to these
    blocks. This requires the user to split out a separate mask_mod from the
    score_mod. For causal masks, this is about a 15% speedup.

    3. [GENERATED] (q_num_blocks, q_indices): Required for the backwards pass,
    as computing dKV requires iterating along the mask along the Q dimension. These are autogenerated from 1.

    4. [GENERATED] (full_q_num_blocks, full_q_indices): Same as above, but for
    the backwards pass. These are autogenerated from 2.
    """

    seq_lengths: tuple[int, int]
    kv_num_blocks: Tensor
    kv_indices: Tensor
    full_kv_num_blocks: Tensor | None
    full_kv_indices: Tensor | None
    q_num_blocks: Tensor | None
    q_indices: Tensor | None
    full_q_num_blocks: Tensor | None
    full_q_indices: Tensor | None
    BLOCK_SIZE: tuple[int, int]
    mask_mod: _mask_mod_signature

    # Attribute lists for pytree flatten/unflatten
    _TENSOR_ATTRS = [
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
    ]

    _CONTEXT_ATTRS = [
        "seq_lengths",
        "BLOCK_SIZE",
        "mask_mod",
    ]

    def __init__(
        self,
        seq_lengths: tuple[int, int],
        kv_num_blocks: Tensor,
        kv_indices: Tensor,
        full_kv_num_blocks: Tensor | None,
        full_kv_indices: Tensor | None,
        q_num_blocks: Tensor | None,
        q_indices: Tensor | None,
        full_q_num_blocks: Tensor | None,
        full_q_indices: Tensor | None,
        BLOCK_SIZE: tuple[int, int],
        mask_mod: _mask_mod_signature,
    ) -> None:
        if kv_indices.dim() < 2:
            raise RuntimeError("BlockMask must have at least 2 dimensions")
        assert kv_num_blocks is not None, "kv_num_blocks must be provided"
        assert kv_indices is not None, "kv_indices must be provided"
        assert (full_kv_num_blocks is None) == (full_kv_indices is None), (
            "full_kv_num_blocks and full_kv_indices must be both provided or omitted"
        )
        assert (full_q_num_blocks is None) == (full_q_indices is None), (
            "full_q_num_blocks and full_q_indices must be both provided or omitted"
        )

        self.seq_lengths = seq_lengths
        self.kv_num_blocks = kv_num_blocks
        self.kv_indices = kv_indices
        self.full_kv_num_blocks = full_kv_num_blocks
        self.full_kv_indices = full_kv_indices
        self.q_num_blocks = q_num_blocks
        self.q_indices = q_indices
        self.full_q_num_blocks = full_q_num_blocks
        self.full_q_indices = full_q_indices
        self.BLOCK_SIZE = BLOCK_SIZE
        self.mask_mod = mask_mod

    @classmethod
    def from_kv_blocks(
        cls,
        kv_num_blocks: Tensor,
        kv_indices: Tensor,
        full_kv_num_blocks: Tensor | None = None,
        full_kv_indices: Tensor | None = None,
        BLOCK_SIZE: Union[int, tuple[int, int]] = _DEFAULT_SPARSE_BLOCK_SIZE,
        mask_mod: _mask_mod_signature | None = None,
        seq_lengths: tuple[int, int] | None = None,
        compute_q_blocks: bool = True,
    ):
        """
        Creates a BlockMask instance from key-value block information.

        Args:
            kv_num_blocks (Tensor): Number of kv_blocks in each Q_BLOCK_SIZE row tile.
            kv_indices (Tensor): Indices of key-value blocks in each Q_BLOCK_SIZE row tile.
            full_kv_num_blocks (Optional[Tensor]): Number of full kv_blocks in each Q_BLOCK_SIZE row tile.
            full_kv_indices (Optional[Tensor]): Indices of full key-value blocks in each Q_BLOCK_SIZE row tile.
            BLOCK_SIZE (Union[int, tuple[int, int]]): Size of KV_BLOCK_SIZE x Q_BLOCK_SIZE tiles.
            mask_mod (Optional[Callable]): Function to modify the mask.

        Returns:
            BlockMask: Instance with full Q information generated via _transposed_ordered

        Raises:
            RuntimeError: If kv_indices has < 2 dimensions.
            AssertionError: If only one of full_kv_* args is provided.
        """
        if kv_indices.dim() < 2:
            raise RuntimeError("BlockMask must have at least 2 dimensions")

        assert (full_kv_num_blocks is None) == (full_kv_indices is None), (
            "full_kv_num_blocks and full_kv_indices must be both provided or omitted"
        )

        # Generate q_num_blocks and q_indices
        if compute_q_blocks:
            q_num_blocks, q_indices = _transpose_ordered(kv_num_blocks, kv_indices)
            if full_kv_num_blocks is not None:
                assert full_kv_indices is not None
                full_q_num_blocks, full_q_indices = _transpose_ordered(
                    full_kv_num_blocks, full_kv_indices
                )
            else:
                full_q_num_blocks, full_q_indices = None, None
        else:
            q_num_blocks, q_indices = None, None
            full_q_num_blocks, full_q_indices = None, None

        if isinstance(BLOCK_SIZE, int):
            BLOCK_SIZE = (BLOCK_SIZE, BLOCK_SIZE)

        mask_mod = mask_mod if mask_mod is not None else noop_mask
        if seq_lengths is None:
            q_length = kv_indices.shape[-2] * BLOCK_SIZE[0]
            kv_length = kv_indices.shape[-1] * BLOCK_SIZE[1]
            seq_lengths = (q_length, kv_length)

        return cls(
            seq_lengths=seq_lengths,
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            q_num_blocks=q_num_blocks,
            q_indices=q_indices,
            full_q_num_blocks=full_q_num_blocks,
            full_q_indices=full_q_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            mask_mod=mask_mod,
        )

    def as_tuple(self, flatten: bool = True):
        """
        Returns a tuple of the attributes of the BlockMask.

        Args:
            flatten (bool): If True, it will flatten the tuple of (KV_BLOCK_SIZE, Q_BLOCK_SIZE)
        """
        if flatten:
            block_size = (self.BLOCK_SIZE[0], self.BLOCK_SIZE[1])  # type: ignore[assignment]
            seq_lengths = (self.seq_lengths[0], self.seq_lengths[1])  # type: ignore[assignment]
        else:
            block_size = (self.BLOCK_SIZE,)  # type: ignore[assignment]
            seq_lengths = (self.seq_lengths,)  # type: ignore[assignment]

        # pyrefly: ignore [not-iterable]
        return (
            *seq_lengths,
            self.kv_num_blocks,
            self.kv_indices,
            self.full_kv_num_blocks,
            self.full_kv_indices,
            self.q_num_blocks,
            self.q_indices,
            self.full_q_num_blocks,
            self.full_q_indices,
            *block_size,
            self.mask_mod,
        )

    @property
    def shape(self):
        *batch_dims, _, _ = self.kv_indices.shape
        return tuple(batch_dims) + self.seq_lengths

    def __str__(self) -> str:
        s = f"BlockMask(shape={self.shape}, sparsity={self.sparsity():.2f}%, \n"
        mask_str = self.to_string().strip()
        s += mask_str
        s += "\n)"
        return s

    def __getitem__(self, index) -> "BlockMask":
        """
        Returns a new BlockMask instance by getting the mask for the given index position.

        Args:
            index: Index to apply to all attributes.

        Example Usage:
            .. code-block:: python

                def causal_mask(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx


                block_mask = create_block_mask(
                    causal_mask, 4, 2, 512, 512, device="cuda"
                )
                assert block_mask.kv_num_blocks.shape == (4, 2, 4)
                assert block_mask.kv_indices.shape == (4, 2, 4, 4)

                # Index on batch dimension
                new_block_mask = block_mask[0]
                assert new_block_mask.kv_num_blocks.shape == (2, 4)
                assert new_block_mask.kv_indices.shape == (2, 4, 4)

                # Index on batch and head dimension
                new_block_mask = block_mask[0, 1]
                assert new_block_mask.kv_num_blocks.shape == (4,)
                assert new_block_mask.kv_indices.shape == (4, 4)

                # slicing on batch and head dimension
                new_block_mask = block_mask[0:2, 1:2]
                assert new_block_mask.kv_num_blocks.shape == (2, 1, 4)
                assert new_block_mask.kv_indices.shape == (2, 1, 4, 4)

                # slicing on batch, head, and query dimension
                new_block_mask = block_mask[
                    0:2, 1:2, torch.tensor([1], dtype=torch.int32)
                ]
                assert new_block_mask.kv_num_blocks.shape == (2, 1, 1)
                assert new_block_mask.kv_indices.shape == (2, 1, 1, 4)
        """
        index = (index,) if not isinstance(index, tuple) else index
        padded = (*index, slice(None), slice(None), slice(None))[:3]
        sizes = self.kv_num_blocks.shape[:3]
        index = tuple(
            (slice(i + n, i + n + 1) if -n <= i < 0 else slice(i, i + 1))
            if isinstance(i, int)
            else i
            for i, n in zip(padded, sizes, strict=True)
        )
        new_kv_num_blocks = self.kv_num_blocks[index]
        new_kv_indices = self.kv_indices[index]
        if self.full_kv_num_blocks is not None:
            assert self.full_kv_indices is not None
            new_full_kv_num_blocks = self.full_kv_num_blocks[index]
            new_full_kv_indices = self.full_kv_indices[index]
        else:
            new_full_kv_num_blocks = None
            new_full_kv_indices = None
        return BlockMask.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            BLOCK_SIZE=self.BLOCK_SIZE,
            mask_mod=_sliced_mask_mod_error,
            seq_lengths=self.seq_lengths,
            compute_q_blocks=self.q_indices is not None,
        )

    def __repr__(self) -> str:
        def shape_or_none(x: torch.Tensor | None):
            return x.shape if x is not None else None

        return (
            f"BlockMask(\n"
            f"    kv_num_blocks={self.kv_num_blocks.shape},\n"
            f"    kv_indices={self.kv_indices.shape},\n"
            f"    full_kv_num_blocks={shape_or_none(self.full_kv_num_blocks)},\n"
            f"    full_kv_indices={shape_or_none(self.full_kv_indices)},\n"
            f"    q_num_blocks={shape_or_none(self.q_num_blocks)},\n"
            f"    q_indices={shape_or_none(self.q_indices)},\n"
            f"    full_q_num_blocks={shape_or_none(self.full_q_num_blocks)},\n"
            f"    full_q_indices={shape_or_none(self.full_q_indices)},\n"
            f"    BLOCK_SIZE={self.BLOCK_SIZE},\n"
            f"    shape={self.shape},\n"
            f"    sparsity={self.sparsity():.2f}%,\n"
            f"    mask_mod={self.mask_mod.__name__ if hasattr(self.mask_mod, '__name__') else self.mask_mod}\n"
            f")"
        )

    def _adjust(self, new_q_len: int, new_kv_len: int):
        new_num_rows = (new_q_len + self.BLOCK_SIZE[0] - 1) // self.BLOCK_SIZE[0]
        new_num_cols = (new_kv_len + self.BLOCK_SIZE[1] - 1) // self.BLOCK_SIZE[1]
        new_kv_num_blocks, new_kv_indices = _adjust_num_blocks_and_indices(
            self.kv_num_blocks, self.kv_indices, new_num_rows, new_num_cols
        )
        if self.full_kv_num_blocks is not None:
            assert self.full_kv_indices is not None
            (
                new_full_kv_num_blocks,
                new_full_kv_indices,
            ) = _adjust_num_blocks_and_indices(
                self.full_kv_num_blocks,
                self.full_kv_indices,
                new_num_rows,
                new_num_cols,
            )
        else:
            new_full_kv_num_blocks = None
            new_full_kv_indices = None
        return self.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            self.BLOCK_SIZE,
            self.mask_mod,
        )

    def numel(self):
        """Returns the number of elements (not accounting for sparsity) in the mask."""
        shape = self.shape

        def _prod(xs):
            return functools.reduce(operator.mul, xs, 1)

        return _prod(shape)

    def sparsity(self) -> float:
        """Computes the percentage of blocks that are sparse (i.e. not computed)"""
        total_size = self.numel()
        computed_blocks = self.kv_num_blocks.sum()
        if self.full_kv_num_blocks is not None:
            computed_blocks += self.full_kv_num_blocks.sum()

        computed_size = computed_blocks.item() * self.BLOCK_SIZE[0] * self.BLOCK_SIZE[1]
        dense_ratio = computed_size / total_size
        return 100 * (1 - dense_ratio)

    def to_dense(self) -> Tensor:
        """Returns a dense block that is equivalent to the block mask."""
        partial_dense = _ordered_to_dense(self.kv_num_blocks, self.kv_indices)
        if self.full_kv_num_blocks is not None:
            assert self.full_kv_indices is not None
            # pyrefly: ignore [bad-return]
            return partial_dense | _ordered_to_dense(
                self.full_kv_num_blocks, self.full_kv_indices
            )
        return partial_dense

    def to_string(self, grid_size=(20, 20), limit=4):
        """Returns a string representation of the block mask. Quite nifty.

        If grid_size is -1, prints out an uncompressed version. Warning, it can be quite big!
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

            def summarize_section(section) -> str:
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

    def to(self, device: Union[torch.device, str]) -> "BlockMask":
        """Moves the BlockMask to the specified device.

        Args:
            device (torch.device or str): The target device to move the BlockMask to.
                Can be a torch.device object or a string (e.g., 'cpu', 'cuda:0').

        Returns:
            BlockMask: A new BlockMask instance with all tensor components moved
            to the specified device.

        Note:
            This method does not modify the original BlockMask in-place.
            Instead, it returns a new BlockMask instance where individual tensor attributes
            may or may not be moved to the specified device, depending on their
            current device placement.
        """
        mapped_attributes = tree_map_only(
            torch.Tensor,
            lambda x: x.to(device),
            self.as_tuple(flatten=False),
        )
        return BlockMask(*mapped_attributes)

    def _flatten(self):
        """Flatten BlockMask into a list of tensors and context."""
        tensors = tuple(getattr(self, attr) for attr in self._TENSOR_ATTRS)
        context = tuple(getattr(self, attr) for attr in self._CONTEXT_ATTRS)
        return tensors, context

    @classmethod
    def _unflatten(cls, tensors, context):
        """Unflatten tensors and context back into a BlockMask."""
        kwargs = {
            **dict(zip(cls._CONTEXT_ATTRS, context)),
            **dict(zip(cls._TENSOR_ATTRS, tensors)),
        }
        # pyrefly: ignore [bad-argument-type]
        return cls(**kwargs)

    def _flatten_with_keys(self):
        """Flatten BlockMask with keys for better tracing."""
        tensors = tuple(
            (GetAttrKey(attr), getattr(self, attr)) for attr in self._TENSOR_ATTRS
        )
        context = tuple(
            (GetAttrKey(attr), getattr(self, attr)) for attr in self._CONTEXT_ATTRS
        )
        return tensors, context


def _broadcast_to_dim(x, dim):
    while x.dim() < dim:
        x = x.unsqueeze(0)
    return x


def _round_up_to_multiple(x, multiple):
    return (x + multiple - 1) // multiple * multiple


def _convert_mask_to_block_mask(
    mask: Tensor,
    Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    separate_full_blocks: bool = False,
) -> tuple[Tensor, Tensor | None]:
    assert mask.dtype == torch.bool
    mask = _broadcast_to_dim(mask, 4)

    def padding_needed_for_multiple(x, multiple):
        return _round_up_to_multiple(x, multiple) - x

    mask = torch.nn.functional.pad(
        mask,
        (
            0,
            padding_needed_for_multiple(mask.shape[-1], KV_BLOCK_SIZE),
            0,
            padding_needed_for_multiple(mask.shape[-2], Q_BLOCK_SIZE),
        ),
    )
    B, H, Q, KV = mask.shape
    assert Q % Q_BLOCK_SIZE == 0
    assert KV % KV_BLOCK_SIZE == 0
    mask = mask.view(
        B, H, Q // Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV // KV_BLOCK_SIZE, KV_BLOCK_SIZE
    )  # [B, H, Q//Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask = mask.permute(
        0, 1, 2, 4, 3, 5
    )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, Q_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask_block_sum = mask.sum(
        dim=[-2, -1]
    )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE]
    if separate_full_blocks:
        full_block_sum = Q_BLOCK_SIZE * KV_BLOCK_SIZE
        full_blocks = mask_block_sum == full_block_sum
        partial_blocks = (mask_block_sum > 0) & (mask_block_sum < full_block_sum)
        partial_blocks = partial_blocks.to(dtype=torch.int8)
        full_blocks = full_blocks.to(dtype=torch.int8)
        return partial_blocks, full_blocks
    else:
        partial_blocks = mask_block_sum > 0
        partial_blocks = partial_blocks.to(dtype=torch.int8)
        return partial_blocks, None


def or_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature:
    """Returns a mask_mod that's the union of provided mask_mods"""
    if not all(callable(arg) for arg in mask_mods):
        raise RuntimeError(f"All inputs should be callable mask_mods: {mask_mods}")

    def or_mask(b, h, q_idx, kv_idx):
        result = b.new_zeros((), dtype=torch.bool)
        for mask in mask_mods:
            result = result | mask(b, h, q_idx, kv_idx)
        return result

    return or_mask


def and_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature:
    """Returns a mask_mod that's the intersection of provided mask_mods"""
    if not all(callable(arg) for arg in mask_mods):
        raise RuntimeError(f"All inputs should be callable mask_mods: {mask_mods}")

    def and_mask(b, h, q_idx, kv_idx):
        result = b.new_ones((), dtype=torch.bool)
        for mask in mask_mods:
            result = result & mask(b, h, q_idx, kv_idx)
        return result

    return and_mask


def _convert_block_mask_to_mask(
    block_mask,
    KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
) -> Tensor:
    assert block_mask.dim() == 4
    B, H, Q, KV = block_mask.shape
    block_mask = block_mask.expand(Q_BLOCK_SIZE, KV_BLOCK_SIZE, *block_mask.shape)
    block_mask = block_mask.permute(2, 3, 4, 0, 5, 1).reshape(
        B, H, Q * Q_BLOCK_SIZE, KV * KV_BLOCK_SIZE
    )
    return block_mask


def _create_sparse_block_from_block_mask(
    block_mask: tuple[Tensor, Tensor | None],
    mask_mod: Callable | None,
    seq_lengths: tuple[int, int],
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
) -> BlockMask:
    partial_blocks, full_blocks = block_mask

    partial_bm = _dense_to_ordered(partial_blocks)
    if full_blocks is not None:
        full_bm: tuple[Tensor | None, Tensor | None] = _dense_to_ordered(full_blocks)
    else:
        full_bm = (None, None)

    return BlockMask.from_kv_blocks(
        partial_bm[0],
        partial_bm[1],
        full_bm[0],
        full_bm[1],
        BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
        mask_mod=mask_mod,
        seq_lengths=seq_lengths,
    )


def create_mask(
    mod_fn: Union[_score_mod_signature, _mask_mod_signature],
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: DeviceLikeType | None = None,
) -> Tensor:
    r"""This function creates a mask tensor from a mod_fn function.

    Args:
        mod_fn (Union[_score_mod_signature, _mask_mod_signature]): Function to modify attention scores.
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Sequence length of query.
        KV_LEN (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.

    Returns:
        mask (Tensor): A mask tensor with shape (B, H, M, N).
    """
    if device is None:
        device = torch.accelerator.current_accelerator() or "cpu"
    if B is None:
        B = 1
    if H is None:
        H = 1
    b = torch.arange(0, B, device=device)
    h = torch.arange(0, H, device=device)
    m = torch.arange(0, Q_LEN, device=device)
    n = torch.arange(0, KV_LEN, device=device)
    mod_type = _get_mod_type(mod_fn)

    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    with TransformGetItemToIndex():
        if mod_type == _ModificationType.SCORE_MOD:
            score_mod = mod_fn
            score_mod = _vmap_for_bhqkv(score_mod, prefix=(0,))  # first input is score
            out = score_mod(torch.zeros(B, H, Q_LEN, KV_LEN, device=device), b, h, m, n)
            mask = torch.where(torch.isneginf(out), False, True)
            return mask
        elif mod_type == _ModificationType.MASK_MOD:
            mask_mod = mod_fn
            mask_mod = _vmap_for_bhqkv(mask_mod, prefix=())
            mask = mask_mod(b, h, m, n)
            return mask
        else:
            raise AssertionError


def create_block_mask(
    mask_mod: _mask_mod_signature,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: DeviceLikeType | None = None,
    BLOCK_SIZE: Union[int, tuple[int, int]] = _DEFAULT_SPARSE_BLOCK_SIZE,
    _compile=False,
) -> BlockMask:
    r"""This function creates a block mask tuple from a mask_mod function.

    Args:
        mask_mod (Callable): mask_mod function. This is a callable that defines the
            masking pattern for the attention mechanism. It takes four arguments:
            b (batch size), h (number of heads), q_idx (query index), and kv_idx (key/value index).
            It should return a boolean tensor indicating which attention connections are allowed (True)
            or masked out (False).
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Sequence length of query.
        KV_LEN (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.
        BLOCK_SIZE (int or tuple[int, int]): Block size for the block mask. If a single int is provided it is used for both query and key/value.

    Returns:
        BlockMask:  A BlockMask object that contains the block mask information.

    Example Usage:
        .. code-block:: python

            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx


            block_mask = create_block_mask(causal_mask, 1, 1, 8192, 8192, device="cuda")
            query = torch.randn(1, 1, 8192, 64, device="cuda", dtype=torch.float16)
            key = torch.randn(1, 1, 8192, 64, device="cuda", dtype=torch.float16)
            value = torch.randn(1, 1, 8192, 64, device="cuda", dtype=torch.float16)
            output = flex_attention(query, key, value, block_mask=block_mask)
    """
    if device is None:
        device = torch.accelerator.current_accelerator() or "cpu"
    mod_type = _get_mod_type(mask_mod)
    assert mod_type == _ModificationType.MASK_MOD, (
        f"create-block_mask requires a mask_mod function! Got {mask_mod}"
    )
    if B is None:
        B = 1
    if H is None:
        H = 1
    if isinstance(BLOCK_SIZE, int):
        Q_BLOCK_SIZE = BLOCK_SIZE
        KV_BLOCK_SIZE = BLOCK_SIZE
    else:
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE

    if _compile:
        warnings.warn(
            "_compile flag on create_block_mask was originally added to work around a torch.compile limitation. That limitation has since been addressed. So, to compile create_block_mask, we suggest doing torch.compile(create_block_mask). This still works for now, but will be removed in the future.",
            DeprecationWarning,
            stacklevel=2,
        )
        return torch.compile(create_block_mask)(
            mask_mod, B, H, Q_LEN, KV_LEN, device, BLOCK_SIZE
        )

    mask_tensor = create_mask(mask_mod, B, H, Q_LEN, KV_LEN, device)
    partial_block_mask, full_block_mask = _convert_mask_to_block_mask(
        mask_tensor,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        separate_full_blocks=True,
    )
    block_mask = _create_sparse_block_from_block_mask(
        (partial_block_mask, full_block_mask),
        mask_mod,
        (Q_LEN, KV_LEN),
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    )
    return block_mask


def _create_empty_block_mask(query: Tensor, key: Tensor) -> BlockMask:
    r"""Default block mask for flex attention.
    If users don't specify any block sparse mask info, we create this
    empty block sparse mask. Which creates a BlockMask with 1 block that is the full length
    of the query and key tensors.
    """
    device = query.device
    return BlockMask.from_kv_blocks(
        kv_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32, device=device),
        kv_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device),
        BLOCK_SIZE=_LARGE_SPARSE_BLOCK_SIZE,
        seq_lengths=(1, 1),
    )


def _apply_kernel_options(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    return_lse: bool,
    kernel_options,
    return_aux: AuxRequest | None = None,
):
    kernel_options = {} if kernel_options is None else dict(kernel_options)

    kernel_options.setdefault("PRESCALE_QK", False)
    kernel_options.setdefault("ROWS_GUARANTEED_SAFE", False)
    kernel_options.setdefault("BLOCKS_ARE_CONTIGUOUS", False)
    # This forces all biases grad scatters to be done in the DQ iteration loop of the backwards
    kernel_options.setdefault("WRITE_DQ", True)

    any_inputs_on_cpu_device = (
        query.device.type == "cpu"
        or key.device.type == "cpu"
        or value.device.type == "cpu"
    )

    # Determine what auxiliary outputs are needed
    output_lse = return_lse
    output_max = False

    if return_aux is not None:
        # New API takes precedence over legacy parameters
        output_lse = return_aux.lse
        output_max = return_aux.max_scores

    # If forward kernel needs to return logsumexp is decided by this rule internally.
    assert "OUTPUT_LOGSUMEXP" not in kernel_options
    kernel_options["OUTPUT_LOGSUMEXP"] = True
    if not output_lse:
        # We used to check if q,k,v required grads but since captured buffers can require grad
        # we always write unless in no_grad
        kernel_options["OUTPUT_LOGSUMEXP"] = torch.is_grad_enabled()
        if any_inputs_on_cpu_device:
            # CPU with torch.compile now supports inference, and will not return lse
            # TODO: support CPU for training and return lse
            kernel_options["OUTPUT_LOGSUMEXP"] = False

    # If forward kernel needs to return max is decided by this rule internally.
    assert "OUTPUT_MAX" not in kernel_options
    kernel_options["OUTPUT_MAX"] = output_max
    if any_inputs_on_cpu_device and output_max:
        # CPU doesn't support returning max yet
        # TODO: support CPU for returning max
        raise NotImplementedError("Returning max scores is not supported on CPU.")
        kernel_options["OUTPUT_MAX"] = False

    return kernel_options


def _validate_embed_dim(query: Tensor, key: Tensor, value: Tensor) -> None:
    if query.size(-1) != key.size(-1):
        raise ValueError(
            f"Expect query and key/value to have the same embedding dimension "
            f"but got E={query.size(-1)} and E={key.size(-1)}."
        )


def _validate_device(query: Tensor, key: Tensor, value: Tensor) -> None:
    """TODO: Remove once non cuda/cpu devices support is added
    We only need to check query since we have already that q,k,v are on the same device
    """
    supported_devices = {"cuda", "cpu", "xpu", "hpu"}
    if query.device.type not in supported_devices:
        raise ValueError(
            "FlexAttention is only supported on CUDA, CPU or HPU devices. "
            f"Found input tensors on {query.device.type} device."
        )


def _enforce_mem_layouts(
    query: Tensor, key: Tensor, value: Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enforce memory layouts for query, key, and value tensors.

    For non-FP8 dtypes, no action is taken.

    For FP8 dtypes, we enforce the following memory layouts:
    - Query tensor must be in row-major memory layout, as it will be the left-operand in the FP8 GEMM `q @ k.T`.
    - Key tensor must be in row-major memory layout, as it will be transposed when used as the right-operand
      in the FP8 GEMM `q @ k.T`, meaning it will correctly be in column-major memory layout for the GEMM.
    - Value tensor must be in column-major memory layout, as it will be the right-operand in the FP8 GEMM `softmax_scores @ v`.

    Returns the query, key, and value tensors with the enforced memory layouts.
    """

    def is_row_major(tensor: Tensor) -> bool:
        return tensor.stride()[-1] == 1

    def is_col_major(tensor: Tensor) -> bool:
        return tensor.stride()[-2] == 1

    # These memory layout constraint are only for FP8 GEMMs on NVIDIA GPU architectures >= SM89 and < SM100.
    # This is because GPU arch < SM89 does not not support FP8 GEMMs, and
    # SM100 has support for TN, NT, TT, NN layouts for FP8 GEMMs
    # (i.e., left and right operands can be in row or column major layouts)
    # so this check is only needed for older architectures.
    # See: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md
    fp8_dtypes = (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    gemm_precision = query.dtype

    should_enforce_mem_layout = (
        gemm_precision in fp8_dtypes
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability("cuda") >= (8, 9)
        and torch.cuda.get_device_capability("cuda") < (10, 0)
    )
    if not should_enforce_mem_layout:
        return query, key, value

    # Query must be in row-major memory layout as the left-operand in the FP8 GEMM `q @ k.T`
    if not is_row_major(query):
        query = query.contiguous()

    # Key must be in row-major memory layout as it will be transposed when used as the right-operand
    # in the FP8 GEMM `q @ k.T`, meaning it will correctly be in column-major memory layout for the GEMM.
    if not is_row_major(key):
        key = key.contiguous()

    # Value must be in column-major memory layout as the right-operand in the FP8 GEMM `softmax_scores @ v`
    if not is_col_major(value):
        value = value.transpose(-2, -1).contiguous().transpose(-2, -1)
    return query, key, value


def flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: _score_mod_signature | None = None,
    block_mask: BlockMask | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: FlexKernelOptions | None = None,
    *,
    return_aux: AuxRequest | None = None,
) -> Union[Tensor, tuple[Tensor, Tensor], tuple[Tensor, AuxOutput]]:
    r"""This function implements scaled dot product attention with an arbitrary attention score modification function.

    This function computes the scaled dot product attention between query, key, and value tensors with a user-defined
    attention score modification function. The attention score modification function will be applied after the attention
    scores have been calculated between the query and key tensors. The attention scores are calculated as follows:

    The ``score_mod`` function should have the following signature:

    .. code-block:: python

        def score_mod(
            score: Tensor,
            batch: Tensor,
            head: Tensor,
            q_idx: Tensor,
            k_idx: Tensor
        ) -> Tensor:

    Where:
        - ``score``: A scalar tensor representing the attention score,
          with the same data type and device as the query, key, and value tensors.
        - ``batch``, ``head``, ``q_idx``, ``k_idx``: Scalar tensors indicating
          the batch index, query head index, query index, and key/value index, respectively.
          These should have the ``torch.int`` data type and be located on the same device as the score tensor.

    Args:
        query (Tensor): Query tensor; shape :math:`(B, Hq, L, E)`. For FP8 dtypes, should be in row-major memory layout for optimal performance.
        key (Tensor): Key tensor; shape :math:`(B, Hkv, S, E)`. For FP8 dtypes, should be in row-major memory layout for optimal performance.
        value (Tensor): Value tensor; shape :math:`(B, Hkv, S, Ev)`. For FP8 dtypes, should be in column-major memory layout for optimal performance.
        score_mod (Optional[Callable]): Function to modify attention scores. By default no score_mod is applied.
        block_mask (Optional[BlockMask]): BlockMask object that controls the blocksparsity pattern of the attention.
        scale (Optional[float]): Scaling factor applied prior to softmax. If none, the default value is set to :math:`\frac{1}{\sqrt{E}}`.
        enable_gqa (bool): If set to True, enables Grouped Query Attention (GQA) and broadcasts key/value heads to query heads.
        return_lse (bool): Whether to return the logsumexp of the attention scores. Default is False. **Deprecated**: Use ``return_aux=AuxRequest(lse=True)`` instead.
        kernel_options (Optional[FlexKernelOptions]):
            Options to control the behavior of the underlying Triton kernels.
            See :class:`FlexKernelOptions` for available options and usage examples.
        return_aux (Optional[AuxRequest]): Specifies which auxiliary outputs to compute and return.
            If None, only the attention output is returned. Use ``AuxRequest(lse=True, max_scores=True)``
            to request both auxiliary outputs.

    Returns:
        output (Tensor): Attention output; shape :math:`(B, Hq, L, Ev)`.

        When ``return_aux`` is not None:
            aux (AuxOutput): Auxiliary outputs with requested fields populated.

        When ``return_aux`` is None (deprecated paths):
            lse (Tensor): Log-sum-exp of attention scores; shape :math:`(B, Hq, L)`. Only returned if ``return_lse=True``.

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
    # Some basic input validation
    _validate_sdpa_input(query, key, value)
    _validate_embed_dim(query, key, value)
    _validate_device(query, key, value)
    query, key, value = _enforce_mem_layouts(query, key, value)
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise NotImplementedError("NYI: query, key, and value must be 4D tensors")
    if (not enable_gqa) and query.size(-3) != key.size(-3):
        raise ValueError(
            f"Expect query and key/value to have the same number of heads "
            f"but got Hq={query.size(-3)} and Hkv={key.size(-3)}. "
            f"Try setting enable_gqa=True for GQA."
        )
    if enable_gqa:
        Hq = query.size(1)
        Hkv = key.size(1)
        if Hq % Hkv != 0:
            raise ValueError(
                f"Expect number of query heads to be a multiple of kv heads for GQA "
                f"but got Hq={Hq} and Hkv={Hkv}."
            )
    if query.size(0) != key.size(0):
        if block_mask is None:
            raise ValueError(
                f"Expect query and key/value to have the same batch size, "
                f"or non-none block_mask, "
                f"but got block_mask=None, Bq={query.size(0)}, and Bkv={key.size(0)}."
            )

        if block_mask.kv_num_blocks.size(0) != query.size(0):
            raise ValueError(
                f"Expect query and key/value to have the same batch size, "
                f"or block_mask and query to have the same batch size, "
                f"but got Bq={query.size(0)}, Bkv={key.size(0)}, B_block_mask={block_mask.kv_num_blocks.size(0)}."
            )

    if score_mod is None:
        score_mod = _identity

    if block_mask is None:
        block_mask = _create_empty_block_mask(query, key)

    # If BlockMask was sliced, its mask_mod is intentionally replaced with an error-raising stub.
    # This guard ensures we surface the intended error message before any shape-based checks.
    if getattr(block_mask, "mask_mod", None) is _sliced_mask_mod_error:
        raise RuntimeError("Cannot use mask_mod from a sliced BlockMask")

    if (
        block_mask.BLOCK_SIZE[0] == _LARGE_SPARSE_BLOCK_SIZE
        and block_mask.BLOCK_SIZE[1] == _LARGE_SPARSE_BLOCK_SIZE
    ):
        # This corresponds to the case where we essentially have a "no-op" block mask.
        pass
    else:
        block_mask_q_len = block_mask.shape[-2]
        block_mask_kv_len = block_mask.shape[-1]
        if query.size(-2) > block_mask_q_len or key.size(-2) > block_mask_kv_len:
            raise ValueError(
                f"block_mask was created for block_mask.shape={block_mask.shape} but got q_len={query.size(-2)} and kv_len={key.size(-2)}. "
                "As the block mask was created for a smaller length than you're using it for, you likely need to create a new block mask."
            )
        elif (
            query.size(-2) < block_mask_q_len and key.size(-2) <= block_mask_kv_len
        ) or (query.size(-2) <= block_mask_q_len and key.size(-2) < block_mask_kv_len):
            raise ValueError(
                f"block_mask was created for block_mask.shape={block_mask.shape} but got q_len={query.size(-2)} and kv_len={key.size(-2)}. "
                "As the block mask was created for a larger length than you're using it for, you can either 1. create a new block mask with the correct length, or 2. 'adjust' the existing block mask to the correct length by calling block_mask._adjust(q_len, kv_len). This essentially 'crops' the block mask to the upper left corner, which does not work for all mask_mods!"
            )
        assert query.size(-2) == block_mask_q_len
        assert key.size(-2) == block_mask_kv_len

    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))

    if query.device != block_mask.kv_num_blocks.device:  # type: ignore[union-attr]
        raise RuntimeError(
            f"Expect q/k/v and block_mask to be on the same device "
            f"but got {query.device} and {block_mask.kv_num_blocks.device}."  # type: ignore[union-attr]
        )

    # Handle deprecation warnings for old parameters
    if return_lse and return_aux is not None:
        raise ValueError(
            "Cannot specify both return_lse and return_aux. "
            "return_lse is deprecated, please use return_aux=AuxRequest(lse=True) instead."
        )
    elif return_lse and return_aux is None:
        _warn_once(
            "deprecated_return_lse",
            "return_lse is deprecated and will be removed in v2.10. "
            "Please use return_aux=AuxRequest(lse=True) instead.",
            category=FutureWarning,
        )

    kernel_options = _apply_kernel_options(
        query,
        key,
        value,
        return_lse,
        kernel_options,
        return_aux,
    )

    def _finalize_outputs(
        out,
        lse,
        max_scores,
        *,
        return_aux: AuxRequest | None,
        return_lse: bool,
    ):
        """Normalize stats and build return value (aux-aware, legacy-compatible)."""
        ln2 = math.log(2.0)
        return_lse = return_lse or return_aux is not None and return_aux.lse
        return_max = return_aux is not None and return_aux.max_scores

        lse_scaled = lse * ln2 if (return_lse and lse.numel() > 0) else None
        max_scaled = (
            max_scores * ln2 if (return_max and max_scores.numel() > 0) else None
        )

        if return_aux is not None:
            return out, AuxOutput(
                lse=lse_scaled,
                max_scores=max_scaled,
            )

        if return_lse:
            return out, lse_scaled

        return out

    if torch.compiler.is_dynamo_compiling():
        # mark head_dim and number of heads to be static
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -3)
            torch._dynamo.mark_static(x, -1)

        out, lse, max_scores = flex_attention_hop(
            query,
            key,
            value,
            score_mod,
            block_mask.as_tuple(),
            scale,
            kernel_options,  # type: ignore[union-attr]
        )
        return _finalize_outputs(
            out, lse, max_scores, return_aux=return_aux, return_lse=return_lse
        )

    if not _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG:
        _warn_once(
            warning_id="flex_attention_performance",
            message=(
                "flex_attention called without torch.compile() - this will use an unfused implementation that materializes the full scores matrix instead of generating a fused kernel.\n\n"
                "SOLUTION: Use torch.compile(flex_attention)(...)\n\n"
                "If you want to debug your score_mod/mask_mod, you can set:\n"
                "torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True\n\n"
                "This will allow you to use print statements or breakpoints. Note: This doesn't work with the backwards pass and may produce incorrect results."
            ),
        )

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("flex_attention requires dynamo support")

    from torch._dynamo.backends.debugging import (
        make_eager_backend_with_torch_function_mode,
    )

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass hop to it. So we wrap it in a dummy function.
    def _flex_attention_hop_wrapper(*args, **kwargs):
        return flex_attention_hop(*args, **kwargs)

    with _set_compilation_env():
        with torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_pre_dispatch_torch_function_mode():
                with _temp_remove_metadata_torch_function_mode() as metadata_mode:
                    if metadata_mode:
                        backend: Union[str, Callable[..., Any]] = (
                            make_eager_backend_with_torch_function_mode(metadata_mode)
                        )
                    else:
                        backend = "eager"

                    if _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG:
                        flex_fn = _flex_attention_hop_wrapper
                    else:
                        flex_fn = torch.compile(
                            _flex_attention_hop_wrapper, backend=backend, fullgraph=True
                        )

                    out, lse, max_scores = flex_fn(
                        query,
                        key,
                        value,
                        score_mod,
                        block_mask.as_tuple(),  # type: ignore[union-attr]
                        scale,
                        kernel_options,
                    )
    return _finalize_outputs(
        out, lse, max_scores, return_aux=return_aux, return_lse=return_lse
    )
