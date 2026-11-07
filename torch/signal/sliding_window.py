# mypy: allow-untyped-defs
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = ["sliding_window_max", "sliding_window_min"]


def sliding_window_max(
    input: Tensor,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    return_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Applies a 1D sliding window maximum over an input tensor.

    The window slides along the last dimension of :attr:`input`.

    Args:
        input (Tensor): the input tensor of shape :math:`(..., L)`.
        kernel_size (int): the size of the sliding window. Must be positive.
        stride (int, optional): the stride of the sliding window. Default: 1.
        padding (int, optional): implicit negative infinity padding to be added on both sides.
            Default: 0.
        return_indices (bool, optional): if ``True``, will return the argmax along with the max values.
            Default: ``False``.

    Returns:
        Tensor of shape :math:`(..., L_{out})` containing the maximum values, or a tuple of
        ``(values, indices)`` if :attr:`return_indices` is ``True``.

    Examples::

        >>> x = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
        >>> torch.signal.sliding_window_max(x, kernel_size=3)
        tensor([5., 8., 8., 8.])
    """
    if input.dim() < 1:
        raise ValueError(f"sliding_window_max: Expected input to have at least 1 dimension, got {input.dim()}")
    if kernel_size <= 0:
        raise ValueError(f"sliding_window_max: kernel_size must be positive, got {kernel_size}")
    if stride <= 0:
        raise ValueError(f"sliding_window_max: stride must be positive, got {stride}")
    if padding < 0:
        raise ValueError(f"sliding_window_max: padding must be non-negative, got {padding}")

    orig_shape = input.shape
    L = orig_shape[-1]
    padded_L = L + 2 * padding
    if padded_L < kernel_size:
        raise ValueError(
            f"sliding_window_max: Padded sequence length ({padded_L}) must be >= kernel_size ({kernel_size})"
        )

    # Flatten leading dimensions to (N, 1, L)
    flat_input = input.reshape(-1, 1, L)

    if padding <= kernel_size // 2:
        vals, indices = F.max_pool1d(
            flat_input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_indices=True,
        )
    else:
        padded = F.pad(flat_input, (padding, padding), mode="constant", value=float("-inf"))
        vals, indices = F.max_pool1d(
            padded,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            return_indices=True,
        )
        indices = torch.clamp(indices - padding, 0, L - 1)

    out_shape = list(orig_shape[:-1]) + [vals.shape[-1]]
    out_vals = vals.reshape(out_shape)

    if return_indices:
        out_indices = indices.reshape(out_shape)
        return out_vals, out_indices
    return out_vals


def sliding_window_min(
    input: Tensor,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    return_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Applies a 1D sliding window minimum over an input tensor.

    The window slides along the last dimension of :attr:`input`.

    Args:
        input (Tensor): the input tensor of shape :math:`(..., L)`.
        kernel_size (int): the size of the sliding window. Must be positive.
        stride (int, optional): the stride of the sliding window. Default: 1.
        padding (int, optional): implicit positive infinity padding to be added on both sides.
            Default: 0.
        return_indices (bool, optional): if ``True``, will return the argmin along with the min values.
            Default: ``False``.

    Returns:
        Tensor of shape :math:`(..., L_{out})` containing the minimum values, or a tuple of
        ``(values, indices)`` if :attr:`return_indices` is ``True``.

    Examples::

        >>> x = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
        >>> torch.signal.sliding_window_min(x, kernel_size=3)
        tensor([1., 2., 2., 3.])
    """
    if return_indices:
        max_vals, indices = sliding_window_max(
            -input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_indices=True,
        )
        return -max_vals, indices
    else:
        max_vals = sliding_window_max(
            -input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_indices=False,
        )
        return -max_vals
