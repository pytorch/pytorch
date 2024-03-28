"""Defines bias subclasses that work with scaled_dot_product_attention"""
from enum import auto, IntEnum
from typing import Optional
from warnings import warn

import torch
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    SDPAParams,
)
from torch.nn.attention import _raise_kernel_warnings
from torch.nn.attention._utils import (
    _input_requires_grad,
    _postprocess_flash_output,
    _preprocess_flash,
    _validate_sdpa_input,
)
from torch.nn.functional import scaled_dot_product_attention

__all__ = ["causal_upper_left", "causal_lower_right", "CausalVariant", "CausalBias"]


torch._dynamo.allow_in_graph(can_use_flash_attention)
torch._dynamo.allow_in_graph(can_use_efficient_attention)
torch._dynamo.allow_in_graph(SDPAParams)


class CausalVariant(IntEnum):
    r"""
    Enum for causal variants used in attention mechanisms.

    Defines two types of causal biases:

    `UPPER_LEFT`: Represents upper-left triangular bias for standard causal attention.
    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        torch.tril(torch.ones(size, dtype=torch.bool))

    For instance, with `shape=(3,4)`, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0]]


    `LOWER_RIGHT`: Represents lower-right triangular bias, the include values are aligned to the lower
    right corner of the matrix.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        diagonal_offset = size[1] - size[0]
        torch.tril(
            torch.ones(size, dtype=torch.bool),
            diagonal=diagonal_offset,
        )

    For instance, with `shape=(3,4)`, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Note that these variants are equivalent to each other when the sequence lengths of the query and key/value
    tensors are equal since the triangular matrix is square.

    .. warning:: This enum is a prototype and subject to change.
    """

    UPPER_LEFT = auto()
    LOWER_RIGHT = auto()


class CausalBias(torch.Tensor):
    """
    A bias representing causal attention patterns. For an overview of the bias structure, see the :class:`CausalVariant` enum.

    This class is used for defining causal (triangular) attention biases. For construing the bias, there exist
    two factory functions: :func:`causal_upper_left` and :func:`causal_lower_right`.

    Example:

    .. code-block:: python

        from torch.nn.attention.bias import causal_lower_right

        bsz, num_heads, seqlen_q, seqlen_kv, head_dim = 32, 8, 4, 12, 8

        # Create a lower-right causal bias
        attn_bias = causal_lower_right(seqlen_q, seqlen_kv)

        q = torch.randn(bsz, num_heads, seqlen_q, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(bsz, num_heads, seqlen_kv, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(bsz, num_heads, seqlen_kv, head_dim, device="cuda", dtype=torch.float16)

        out = F.scaled_dot_product_attention(q, k, v, attn_bias)

    .. warning:: This class is a prototype and subject to change.
    """

    def __init__(self, variant: CausalVariant, seq_len_q: int, seq_len_kv: int):
        """
        Initializes the CausalBias instance with a specified variant and sequence lengths.

        Args:
            variant (CausalVariant): The type of causal bias to use (either UPPER_LEFT or LOWER_RIGHT).
            seq_len_q (int): The sequence length of the query tensor.
            seq_len_kv (int): The sequence length of the key/value tensor.

        Raises a warning if the LOWER_RIGHT variant is used with seq_len_q > seq_len_kv, as it may produce NaNs.
        """
        assert isinstance(variant, CausalVariant)
        self.variant = variant
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        if seq_len_q > seq_len_kv and variant == CausalVariant.LOWER_RIGHT:
            warn(
                "Lower right causal bias will produce NaNs in the output when seq_len_q > seq_len_kv!"
            )

    def _upper_left(self, device: torch.device) -> torch.Tensor:
        """Upper left causal bias"""
        return torch.tril(
            torch.ones(self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool)
        )

    def _lower_right(self, device: torch.device) -> torch.Tensor:
        """Lower right causal bias"""
        diagonal_offset = self.seq_len_kv - self.seq_len_q
        return torch.tril(
            torch.ones(
                self.seq_len_q, self.seq_len_kv, device=device, dtype=torch.bool
            ),
            diagonal=diagonal_offset,
        )

    def _materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Materializes the causal bias into a tensor form.

        Depending on the variant, this method generates either an upper-left or lower-right
        triangular matrix to represent the causal bias.

        Args:
            device (Optional[torch.device]): The device on which to create the tensor. Defaults to CPU.

        Returns:
            torch.Tensor: The materialized bias tensor.
        """
        if device is None:
            device = torch.device("cpu")
        if self.variant == CausalVariant.UPPER_LEFT:
            return self._upper_left(device)
        elif self.variant == CausalVariant.LOWER_RIGHT:
            return self._lower_right(device)

    @staticmethod
    def _dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "CausalBias",
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        r"""
        Handles the logic for computing attention with the specified causal bias.

        Args:
            query (Tensor): Query tensor; shape :math:`(N, ..., L, E)`.
            key (Tensor): Key tensor; shape :math:`(N, ..., S, E)`.
            value (Tensor): Value tensor; shape :math:`(N, ..., S, Ev)`.
            attn_mask (CausalBias): The type of causal attention to apply.
                A boolean mask where a value of True indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
            dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
            is_causal (bool): If true, assumes upper left causal attention masking and errors if both attn_mask and is_causal
                are set.
            scale (optional float): Scaling factor applied prior to softmax. If None, the default value is set
                to :math:`\frac{1}{\sqrt{E}}`.

        Returns:
            output (Tensor): Attention output; shape :math:`(N, ..., L, Ev)`.

        Raises:
            ValueError: If the causal bias variant is not a CausalVariant type.

        """
        if is_causal:
            raise ValueError("CausalBias should not be used with causal=True")

        if (
            attn_mask.seq_len_q == attn_mask.seq_len_kv
            or attn_mask.variant == CausalVariant.UPPER_LEFT
        ):
            return scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
                scale=scale,
            )
        elif attn_mask.variant == CausalVariant.LOWER_RIGHT:
            _validate_sdpa_input(query, key, value, None, dropout_p, is_causal, scale)
            sdpa_params = SDPAParams(query, key, value, None, dropout_p, is_causal)
            if can_use_flash_attention(sdpa_params):
                query, key, value, og_head_size, og_scale = _preprocess_flash(
                    query, key, value, scale
                )
                out = torch.ops.aten._scaled_dot_product_flash_attention(
                    query,
                    key,
                    value,
                    dropout_p,
                    is_causal=True,  # TODO: Flash accepts causal = True and for this particular op it means lower right
                    return_debug_mask=False,
                    scale=og_scale,
                )[0]
                return _postprocess_flash_output(out, og_head_size)
            if can_use_efficient_attention(sdpa_params):
                compute_log_sumexp = False
                if _input_requires_grad(query, key, value):
                    compute_log_sumexp = True
                return torch.ops.aten._efficient_attention_forward(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    bias=None,
                    cu_seqlens_q=None,
                    cu_seqlens_k=None,
                    max_seqlen_q=None,
                    max_seqlen_k=None,
                    dropout_p=dropout_p,
                    custom_mask_type=int(attn_mask.variant),
                    compute_log_sumexp=compute_log_sumexp,
                    scale=scale,
                    causal_diagonal=None,
                    seqlen_k=None,
                )[0].transpose(1, 2)
            else:
                _raise_kernel_warnings(sdpa_params)
                # We cant use efficient attention the only support for lower right is via materialization
                return scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask._materialize(query.device),
                    dropout_p=dropout_p,
                    is_causal=False,
                    scale=scale,
                )
        else:
            raise ValueError(
                f"CausalBias.variant must be a CausalVariant type, but found: {attn_mask.variant}"
            )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Defines the behavior of torch.nn.functional.scaled_dot_product_attention when the attn_bias is an AttnBias"""
        if kwargs is None:
            kwargs = {}
        if func != torch.nn.functional.scaled_dot_product_attention:
            raise NotImplementedError(
                "CausalBias only supports scaled_dot_product_attention"
            )
        return cls._dispatch(*args, **kwargs)

    def __repr__(self):
        return self._materialize().__repr__()


def causal_upper_left(*size) -> CausalBias:
    """
    Creates an upper-left triangular causal bias.

    This function generates a upper-left triangular matrix to represent causal attention bias with a
    diagonal offset set so that the inclusive values are aligned to the upper left corner of the matrix.
    This equivalent to the `is_causal=True` argument in `scaled_dot_product_attention`.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        torch.tril(torch.ones(size, dtype=torch.bool))

    For instance, with `shape=(3,4)`, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0]]

    Args:
        size: The size of the bias matrix.

    Returns:
        CausalBias: The UPPER_LEFT triangular causal bias variant.
    """
    assert len(size) == 2, "causal_upper_left only supports 2D tensors"
    seq_len_q, seq_len_kv = size
    return CausalBias(CausalVariant.UPPER_LEFT, seq_len_q, seq_len_kv)


def causal_lower_right(*size) -> CausalBias:
    """
    Creates a lower-right triangular causal bias.

    This function generates a lower-right triangular matrix to represent causal attention bias with a
    diagonal offset set so that the inclusive values are aligned to the lower right corner of the matrix.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        diagonal_offset = size[1] - size[0]
        torch.tril(
            torch.ones(size, dtype=torch.bool),
            diagonal=diagonal_offset,
        )

    For instance, with `shape=(3,4)`, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Args:
        size: The size of the bias matrix.

    Returns:
        CausalBias: The LOWER_RIGHT triangular causal bias variant.
    """
    assert len(size) == 2, "causal_lower_right only supports 2D tensors"
    seq_len_q, seq_len_kv = size
    return CausalBias(CausalVariant.LOWER_RIGHT, seq_len_q, seq_len_kv)


class SlidingWindowBias(torch.Tensor):
    r"""
    Computes a sliding window attention bias that can be used with the scaled dot-product attention (SDPA) mechanism.

    The sliding window attention bias allows each query to attend to a specific window or neighborhood of keys
    around its corresponding position. The window size is defined by two parameters: `window_size_left` and
    `window_size_right`, which specify the number of positions to the left and right of the query position,
    respectively.

    For a query at position `i`, the bias allows it to attend to keys within the range:
    `[i - window_size_left, i + window_size_right]`, while masking out keys outside this range.

    The `SlidingWindowBias` class supports two variants:
     - `UPPER_LEFT`: The first row in the attention matrix corresponds to the query at position 0.
     - `LOWER_RIGHT`: The last row in the attention matrix corresponds to the query at position `seq_len_kv`.

    The `seq_len_q` and `seq_len_kv` parameters specify the sequence lengths of the query and key/value tensors,
    respectively. When `seq_len_q` is not equal to `seq_len_kv`, the behavior depends on the chosen variant.

    This class is designed to be used with the :func:`torch.nn.functional.scaled_dot_product_attention` and supports fused kernel
    execution when  the input satisfies the required constraints for Flash Attention (FAv2).

    Args:
        variant (CausalVariant): The type of causal bias to use (either `UPPER_LEFT` or `LOWER_RIGHT`).
        window_size_left (Optional[int]): The size of the left side of the window. If None, the window is unbounded.
        window_size_right (Optional[int]): The size of the right side of the window. If None, the window is unbounded.
        seq_len_q (int): The sequence length of the query tensor.
        seq_len_kv (int): The sequence length of the key/value tensor.

    Returns:
        SlidingWindowBias: A sliding window attention bias object that can be passed as the `attn_mask` parameter to
        the :func:`torch.nn.functional.scaled_dot_product_attention` function.

    Examples::

        >>> from torch.nn.attention.bias import SlidingWindowBias, CausalVariant
        >>> bias = SlidingWindowBias(CausalVariant.UPPER_LEFT, window_size_left=2, window_size_right=0, seq_len_q=5, seq_len_kv=5)
        >>> print(bias)
        tensor([[ True, False, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [False,  True,  True,  True, False],
                [False, False,  True,  True,  True]])
    """

    def __init__(
        self,
        variant: CausalVariant,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        seq_len_q: int,
        seq_len_kv: int,
    ):
        """
        Initializes the SlidingWindowBias instance with specified window size and sequence lengths.

        Args:
            variant (CausalVariant): The type of causal bias to use (either UPPER_LEFT or LOWER_RIGHT).
            window_size_left (Optional[int]): The size of the left side of the window. If none, the window is unbounded.
            window_size_right (Optional[int]): The size of the right side of the window. If none, the window is unbounded.
            seq_len_q (int): The sequence length of the query tensor.
            seq_len_kv (int): The sequence length of the key/value tensor.

        Raises a warning if the LOWER_RIGHT variant is used with seq_len_q > seq_len_kv, as it may produce NaNs.
        """
        if window_size_left is not None:
            assert (
                window_size_left >= 0
            ), "window_size_left must be non-negative or None"
        if window_size_right is not None:
            assert (
                window_size_right >= 0
            ), "window_size_right must be non-negative or None"
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.variant = variant

    def _sliding_window(self, device: torch.device) -> torch.Tensor:
        """Construct a materialized sliding window attention bias"""
        queries = torch.arange(0, self.seq_len_q, dtype=torch.long, device=device).view(
            -1, 1
        )
        keys = torch.arange(0, self.seq_len_kv, dtype=torch.long, device=device).view(
            1, -1
        )
        offset = (
            self.seq_len_kv - self.seq_len_q
            if self.variant == CausalVariant.LOWER_RIGHT
            else 0
        )
        if self.window_size_left is None:
            return (keys > (queries + offset) + self.window_size_right).logical_not()
        else:
            full_keys = torch.full_like(keys, self.seq_len_kv)
            right_half = keys > torch.minimum(
                (queries + offset) + self.window_size_right, full_keys
            )
            left_half = keys < (queries + offset) - self.window_size_left
            return torch.logical_or(left_half, right_half).logical_not_()

    def _materialize(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Materializes the sliding window attention bias into a tensor form.

        Args:
            device (Optional[torch.device]): The device on which to create the tensor. Defaults to CPU.

        Returns:
            torch.Tensor: The materialized bias tensor.
        """
        if device is None:
            device = torch.device("cpu")
        return self._sliding_window(device)

    @staticmethod
    def _dispatch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: "SlidingWindowBias",
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        r"""
        Handles the logic for computing attention with the specified sliding window bias.

        Args:
            query (Tensor): Query tensor; shape :math:`(N, ..., L, E)`.
            key (Tensor): Key tensor; shape :math:`(N, ..., S, E)`.
            value (Tensor): Value tensor; shape :math:`(N, ..., S, Ev)`.
            attn_mask (SlidingWindowBias): The type of causal attention to apply.
                A boolean mask where a value of True indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
            dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
            is_causal (bool): If true, assumes upper left causal attention masking and errors if both attn_mask and is_causal
                are set.
            scale (optional float): Scaling factor applied prior to softmax. If None, the default value is set
                to :math:`\frac{1}{\sqrt{E}}`.

        Returns:
            output (Tensor): Attention output; shape :math:`(N, ..., L, Ev)`.

        Raises:
            ValueError: If the causal bias variant is not a CausalVariant type.

        """
        if is_causal:
            raise ValueError("SlidingWindowBias should not be used with causal=True")

        if (
            attn_mask.variant == CausalVariant.LOWER_RIGHT
            or attn_mask.seq_len_q == attn_mask.seq_len_kv
        ):
            # We can try to run flash_attention with sliding window hot path
            _validate_sdpa_input(query, key, value, None, dropout_p, is_causal, scale)
            sdpa_params = SDPAParams(query, key, value, None, dropout_p, is_causal)
            if can_use_flash_attention(sdpa_params):
                query, key, value, og_head_size, og_scale = _preprocess_flash(
                    query, key, value, scale
                )
                out = torch.ops.aten._scaled_dot_product_flash_attention(
                    query,
                    key,
                    value,
                    dropout_p,
                    is_causal=is_causal,
                    return_debug_mask=False,
                    scale=og_scale,
                    window_size_left=attn_mask.window_size_left,
                    window_size_right=attn_mask.window_size_right,
                )[0]
                return _postprocess_flash_output(out, og_head_size)
            else:
                _raise_kernel_warnings(sdpa_params)
                # We cant use flash attention the only support for lower right is via materialization
                return scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask._materialize(query.device),
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
        # For anything else we materialized the mask and call regular bias
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask._materialize(query.device),
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Defines the behavior of torch.nn.functional.scaled_dot_product_attention when the attn_bias is an AttnBias"""
        if kwargs is None:
            kwargs = {}
        if func != torch.nn.functional.scaled_dot_product_attention:
            raise NotImplementedError(
                "SlidingWindowBias only supports scaled_dot_product_attention"
            )
        return cls._dispatch(*args, **kwargs)

    def __repr__(self):
        return self._materialize().__repr__()
