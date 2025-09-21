from __future__ import annotations

import functools
from typing import Callable, TYPE_CHECKING, Union

import torch
from functorch.dim import dims  # noqa: F401

from ._parsing import (
    _ellipsis,
    AnonymousAxis,
    comma_separate,
    parse_pattern,
    validate_rearrange_expressions,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["rearrange"]


@functools.lru_cache(256)
def _create_rearrange_callable(
    tensor_ndim: int, pattern: str, **axes_lengths: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Translate an `einops`-style pattern into a callable that performs the rearrange using first-class dimensions.

    Since the an equivalent result is computed for tensors with the same number of dimensions, with the same pattern and
    specified axes lengths, this function can be memoized.

    Args:
        tensor_ndim (int): the number of dimensions in the tensor to rearrange
        pattern (str): the `einops`-style rearrangement pattern
        axes_lengths (int): any additional length specifications for dimensions

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: a callable that performs the rearrangement
    """
    left, right = parse_pattern(pattern, axes_lengths)
    validate_rearrange_expressions(left, right, axes_lengths)

    n_anon_dims = sum(not dim for dim in left.composition)
    if left.has_ellipsis:
        n_ellipsis_dims = tensor_ndim - (len(left.composition) - 1)
        n_named_dims = len(left.identifiers) - 1

        if (pattern_ndim := n_anon_dims + n_named_dims) > tensor_ndim:
            raise ValueError(
                f"Number of dimensions in pattern ({pattern_ndim}) must be less than or equal to the number of "
                f"dimensions in the tensor ({tensor_ndim})"
            )
    else:
        n_ellipsis_dims = 0
        n_named_dims = len(left.identifiers)

        if (pattern_ndim := len(left.composition)) != tensor_ndim:
            raise ValueError(
                f"Number of dimensions in pattern ({pattern_ndim}) must be equal to the number of dimensions in "
                f"the tensor ({tensor_ndim})"
            )
    n_dims = n_named_dims + n_ellipsis_dims + n_anon_dims

    if n_dims == 0:
        # an identity rearrangement on a 0-dimension tensor
        return lambda tensor: tensor

    first_class_dims: tuple[str, ...] = tuple(f"d{i}" for i in range(n_dims))
    identifier_dim_map: dict[Union[str, AnonymousAxis], tuple[str, ...]] = {}
    anon_axes: list[AnonymousAxis] = []

    # map the left-hand side identifiers to strings representing first class dims
    dims_i = 0
    for dimension in left.composition:
        if isinstance(dimension, list):
            for identifier in dimension:
                # non-unitary anon axes are not allowed in rearrange & unitary anon axes are represented as empty lists
                assert isinstance(identifier, str)
                identifier_dim_map[identifier] = (first_class_dims[dims_i],)
                dims_i += 1
            if not dimension:
                # unitary anonymous axis
                anon_axis = AnonymousAxis("1")
                identifier_dim_map[anon_axis] = (first_class_dims[dims_i],)
                anon_axes.append(anon_axis)
                dimension.append(anon_axis)
                dims_i += 1
        elif dimension == _ellipsis:
            identifier = _ellipsis
            identifier_dim_map[identifier] = tuple(
                first_class_dims[dims_i + j] for j in range(n_ellipsis_dims)
            )
            dims_i += n_ellipsis_dims
        else:
            raise ValueError(f"Unexpected dimension: {dimension}")

    def composition_to_dims(
        composition: Sequence[Union[list[Union[str, AnonymousAxis]], str]],
    ) -> list[Union[str, tuple[str, ...]]]:
        """Convert a `ParsedExpression.composition` into a `Tensor.__getitem__` index of strings representing first
        class dims."""
        dim_composition: list[Union[str, tuple[str, ...]]] = []
        for dimension in composition:
            if isinstance(dimension, list):
                dim_composition.append(
                    tuple(
                        dim
                        for identifier in dimension
                        for dim in identifier_dim_map[identifier]
                    )
                )
            elif dimension == _ellipsis:
                dim_composition.extend(identifier_dim_map[_ellipsis])
            else:
                raise ValueError(f"Unexpected dimension: {dimension}")
        return dim_composition

    left_dims = composition_to_dims(left.composition)
    right_dims = composition_to_dims(right.composition)
    anon_dims = tuple(identifier_dim_map[axis][0] for axis in anon_axes)
    specified_lengths = tuple(
        (identifier_dim_map[axis][0], length) for axis, length in axes_lengths.items()
    )

    custom_rearrange_callable_name = "do_rearrange"
    custom_rearrange_callable_code = (
        (
            f"def {custom_rearrange_callable_name}(tensor):\n"
            f"    {comma_separate(first_class_dims)} = dims({n_dims})\n"
        )
        + (
            "".join(
                f"    {dim}.size = {length}\n" for (dim, length) in specified_lengths
            )
            if specified_lengths
            else ""
        )
        + f"    tensor = tensor[{comma_separate(left_dims)}].order({comma_separate(right_dims)})\n"
        + (
            f"    return tensor.sum({comma_separate([anon_dims])}, keepdim=False)\n"
            if anon_dims
            else "    return tensor\n"
        )
    )

    exec(custom_rearrange_callable_code)
    return locals()[custom_rearrange_callable_name]


def rearrange(
    tensor: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]],
    pattern: str,
    **axes_lengths: int,
) -> torch.Tensor:
    r"""A native implementation of `einops.rearrange`, a reader-friendly smart element reordering for multidimensional
    tensors. This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations.

    See: https://einops.rocks/api/rearrange/

    Args:
        tensor (Tensor or sequence of Tensor): the tensor(s) to rearrange
        pattern (str): the rearrangement pattern
        axes_lengths (int): any additional length specifications for dimensions

    Returns:
        Tensor: the rearranged tensor

    Examples:
        >>> # suppose we have a set of 32 images in "h w c" format (height-width-channel)
        >>> images = torch.randn((32, 30, 40, 3))

        >>> # stack along first (batch) axis, output is a single array
        >>> rearrange(images, "b h w c -> b h w c").shape
        torch.Size([32, 30, 40, 3])

        >>> # concatenate images along height (vertical axis), 960 = 32 * 30
        >>> rearrange(images, "b h w c -> (b h) w c").shape
        torch.Size([960, 40, 3])

        >>> # concatenated images along horizontal axis, 1280 = 32 * 40
        >>> rearrange(images, "b h w c -> h (b w) c").shape
        torch.Size([30, 1280, 3])

        >>> # reordered axes to "b c h w" format for deep learning
        >>> rearrange(images, "b h w c -> b c h w").shape
        torch.Size([32, 3, 30, 40])

        >>> # flattened each image into a vector, 3600 = 30 * 40 * 3
        >>> rearrange(images, "b h w c -> b (c h w)").shape
        torch.Size([32, 3600])

        >>> # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
        >>> rearrange(images, "b (h1 h) (w1 w) c -> (b h1 w1) h w c", h1=2, w1=2).shape
        torch.Size([128, 15, 20, 3])

        >>> # space-to-depth operation
        >>> rearrange(images, "b (h h1) (w w1) c -> b h w (c h1 w1)", h1=2, w1=2).shape
        torch.Size([32, 15, 20, 12])
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.stack(tensor)

    rearrange_callable = _create_rearrange_callable(
        tensor.ndim, pattern, **axes_lengths
    )

    return rearrange_callable(tensor)
