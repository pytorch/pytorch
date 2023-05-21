from __future__ import annotations

import torch
from typing import Dict, List, Sequence, Tuple, Union, TYPE_CHECKING
from ._parsing import AnonymousAxis, ParsedExpression, _ellipsis
from functorch._C import dim as _C

if TYPE_CHECKING:
    from functorch.dim.dim import Dim

dims = _C.dims


def rearrange(
    tensor: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    pattern: str,
    **axes_lengths: int,
) -> torch.Tensor:
    r"""A native implementation of `einops.rearrange`, a reader-friendly smart element reordering for multidimensional
    tensors. This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations.

    See: https://einops.rocks/api/rearrange/

    Args:
        tensor (Tensor or sequence of Tensor): The tensor(s) to rearrange
        pattern (str): The rearrangement pattern
        axes_lengths (int): any additional specifications for dimensions

    Returns:
        Tensor: The rearranged tensor

    Examples:
        >>> # suppose we have a set of 32 images in "h w c" format (height-width-channel)
        >>> images = torch.randn((32, 30, 40, 3))

        >>> # stack along first (batch) axis, output is a single array
        >>> rearrange(images, 'b h w c -> b h w c').shape
        torch.Size([32, 30, 40, 3])

        >>> # concatenate images along height (vertical axis), 960 = 32 * 30
        >>> rearrange(images, 'b h w c -> (b h) w c').shape
        torch.Size([960, 40, 3])

        >>> # concatenated images along horizontal axis, 1280 = 32 * 40
        >>> rearrange(images, 'b h w c -> h (b w) c').shape
        torch.Size([30, 1280, 3])

        >>> # reordered axes to "b c h w" format for deep learning
        >>> rearrange(images, 'b h w c -> b c h w').shape
        torch.Size([32, 3, 30, 40])

        >>> # flattened each image into a vector, 3600 = 30 * 40 * 3
        >>> rearrange(images, 'b h w c -> b (c h w)').shape
        torch.Size([32, 3600])

        >>> # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
        >>> rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape
        torch.Size([128, 15, 20, 3])

        >>> # space-to-depth operation
        >>> rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape
        torch.Size([32, 15, 20, 12])
    """
    # validation taken largely from einops.einops._prepare_transformation_recipe
    # https://github.com/arogozhnikov/einops/blob/230ac1526c1f42c9e1f7373912c7f8047496df11/einops/einops.py
    try:
        left_str, right_str = pattern.split("->")
    except ValueError:
        raise ValueError("Pattern must contain a single '->' separator")

    if _ellipsis in axes_lengths:
        raise ValueError(f"'{_ellipsis}' is not an allowed axis identifier")

    left = ParsedExpression(left_str)
    right = ParsedExpression(right_str)

    if not left.has_ellipsis and right.has_ellipsis:
        raise ValueError(f'Ellipsis found in right side, but not left side of a pattern {pattern}')
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise ValueError(f'Ellipsis is parenthesis in the left side is not allowed: {pattern}')
    difference = set.symmetric_difference(left.identifiers, right.identifiers)
    if left.has_non_unitary_anonymous_axes or right.has_non_unitary_anonymous_axes:
        raise ValueError('Non-unitary anonymous axes are not supported in rearrange (exception is length 1)')
    if len(difference) > 0:
        raise ValueError(f'Identifiers only on one side of expression (should be on both): {difference}')
    unmatched_axes = axes_lengths.keys() - left.identifiers
    if len(unmatched_axes) > 0:
        raise ValueError(f'Identifiers not found in expression: {unmatched_axes}')

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.stack(tensor)

    if left.has_ellipsis:
        n_ellipsis_dims = tensor.ndim - (len(left.composition) - 1)
        n_dims = len(left.identifiers) - 1
        total_n_dims = n_dims + n_ellipsis_dims
    else:
        n_ellipsis_dims = 0
        total_n_dims = n_dims = len(left.identifiers)

    first_class_dims: Tuple[Dim, ...] = (dims(total_n_dims),) if total_n_dims == 1 else dims(total_n_dims)
    identifier_dim_map: Dict[str, Tuple[Dim, ...]] = {}

    # map the left-hand side identifiers to first class dims
    dims_i = 0
    for dimension in left.composition:
        if isinstance(dimension, list):
            for identifier in dimension:
                identifier_dim_map[identifier] = (first_class_dims[dims_i],)
                dims_i += 1
        elif dimension == _ellipsis:
            identifier = _ellipsis
            identifier_dim_map[identifier] = tuple(first_class_dims[dims_i + i] for i in range(n_ellipsis_dims))
            dims_i += n_ellipsis_dims
        else:
            raise ValueError(f'Unexpected dimension: {dimension}')

    for axis, length in axes_lengths.items():
        identifier_dim_map[axis][0].size = length

    def composition_to_dims(
        composition: Sequence[Union[List[Union[str, AnonymousAxis]], str]]
    ) -> List[Union[Dim, Tuple[Dim, ...]]]:
        """Convert a `ParsedExpression.composition` into a `Tensor.__getitem__` index of first class dims."""
        dim_composition: List[Union[Dim, Tuple[Dim, ...]]] = []
        for dimension in composition:
            if isinstance(dimension, list):
                dim_composition.append(tuple(dim for identifier in dimension for dim in identifier_dim_map[identifier]))
            elif dimension == _ellipsis:
                dim_composition.extend(identifier_dim_map[_ellipsis])
            else:
                raise ValueError(f'Unexpected dimension: {dimension}')
        return dim_composition

    left_dims = composition_to_dims(left.composition)
    right_dims = composition_to_dims(right.composition)
    # TODO: add type stubs for Tensor.order
    return tensor[left_dims].order(*right_dims)  # type: ignore[attr-defined]
