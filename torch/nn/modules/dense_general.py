from __future__ import annotations

from typing import Tuple, Optional

import math
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn import init

__all__ = ["DenseGeneral"]


def _normalize_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    """Convert possibly-negative axes into a canonical 0-based form."""
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


class DenseGeneral(Module):
    r"""Applies a *generalised* dense (fully-connected) transformation.

    This layer contracts *one or more* input dimensions (``axis``) with the
    **last** dimensions of the weight tensor, producing an output whose shape
    equals the un-contracted input dims followed by ``out_features``.

    Mathematically::

        y = tensordot(x, W, dims=(axis, range(len(axis)))) + b

    Args:
        in_shapes (tuple of int):  sizes of the input dimensions *being
            contracted*.  For an input tensor of rank ``N``, these must
            match ``x.shape[axis[i]]``.
        out_features (tuple of int): shape of the non-contracted part of
            the weight tensor (and therefore of the output).
        axis (tuple of int, optional): the indices of the input dimensions
            to contract.  Defaults to ``(-1,)`` (last dim only, i.e. the
            behaviour of :class:`~torch.nn.Linear`).
        bias (bool, optional): if ``True``, adds a learnable bias.
            Default: ``True``.
        device, dtype: standard parameter-factory kwargs.

    Shape:
        * **Input:**  ``(*batch, *in_shapes, *rest)``  
          – ``axis`` refers to the positions of ``*in_shapes`` inside
          ``input.shape``.
        * **Output:** ``(*batch, *rest, *out_features)``

    Example::

        >>> x = torch.randn(2, 3, 4, 5)           # (B, H, W, C)
        >>> layer = nn.DenseGeneral(
        ...     in_shapes=(4, 5),                 # contract last two dims
        ...     out_features=(8, 16),
        ...     axis=(-2, -1)
        ... )
        >>> y = layer(x)
        >>> y.shape
        torch.Size([2, 3, 8, 16])
    """

    def __init__(
        self,
        in_shapes: Tuple[int, ...],
        out_features: Tuple[int, ...],
        *,
        axis: Tuple[int, ...] = (-1,),
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if len(in_shapes) != len(axis):
            raise ValueError(
                f"in_shapes ({len(in_shapes)}) and axis ({len(axis)}) must "
                "have the same length."
            )

        self.in_shapes = tuple(int(s) for s in in_shapes)
        self.out_features = tuple(int(s) for s in out_features)
        self.axis = tuple(int(a) for a in axis)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = Parameter(torch.empty(*self.in_shapes, *self.out_features,
                                            **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(*self.out_features,
                                              **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:  # mirrors nn.Linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # fan_in is product of in_shapes
            fan_in = int(math.prod(self.in_shapes))
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        # normalise axis (supports negative indices)
        contract_axes = _normalize_axes(self.axis, x.ndim)
        weight_axes = tuple(range(len(contract_axes)))

        out: Tensor = torch.tensordot(
            x, self.weight, dims=(contract_axes, weight_axes)
        )
        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        bias = self.bias is not None
        return (f"in_shapes={self.in_shapes}, out_features={self.out_features}, "
                f"axis={self.axis}, bias={bias}")
