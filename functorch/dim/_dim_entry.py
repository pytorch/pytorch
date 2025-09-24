from __future__ import annotations

from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from collections.abc import Sequence

    from . import Dim

import torch  # noqa: TC002


# NB: The old code represented dimension was from as negative number, so we
# follow this convention even though it shouldn't be necessary now
class DimEntry:
    # The dimension this is from the rhs, or a FCD
    data: Union[Dim, int]

    def __init__(self, data: Union[Dim, int, None] = None) -> None:
        from . import Dim

        if type(data) is int:
            assert data < 0
        elif data is None:
            data = 0
        else:
            assert isinstance(data, Dim)
        self.data = data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DimEntry):
            return False
        # Use 'is' for Dim objects to avoid triggering __torch_function__
        # Use '==' only for positional (int) comparisons
        if self.is_positional() and other.is_positional():
            # Both are positional (ints)
            return self.data == other.data
        elif not self.is_positional() and not other.is_positional():
            # Both are Dim objects - use 'is' to avoid __eq__
            return self.data is other.data
        else:
            # One is positional, one is Dim - they can't be equal
            return False

    def is_positional(self) -> bool:
        return type(self.data) is int and self.data < 0

    def is_none(self) -> bool:
        # Use isinstance to check for Dim objects, avoid triggering __torch_function__
        from . import Dim

        if isinstance(self.data, Dim):
            # This is a Dim object, it can't be "none" (which is represented by 0)
            return False
        else:
            # This is an int or other type
            return self.data == 0

    def position(self) -> int:
        assert isinstance(self.data, int)
        return self.data

    def dim(self) -> Dim:
        assert not isinstance(self.data, int)
        return self.data

    def __repr__(self) -> str:
        return repr(self.data)


def ndim_of_levels(levels: Sequence[DimEntry]) -> int:
    r = 0
    for l in levels:
        if l.is_positional():
            r += 1
    return r


def _match_levels(
    tensor: torch.Tensor,
    from_levels: list[DimEntry],
    to_levels: list[DimEntry],
    drop_levels: bool = False,
) -> torch.Tensor:
    """
    Reshape a tensor to match target levels using as_strided.

    Args:
        tensor: Input tensor to reshape
        from_levels: Current levels of the tensor
        to_levels: Target levels to match
        drop_levels: If True, missing dimensions are assumed to have stride 0

    Returns:
        Reshaped tensor
    """
    if from_levels == to_levels:
        return tensor

    sizes = tensor.size()
    strides = tensor.stride()

    if not drop_levels:
        assert len(from_levels) <= len(to_levels), (
            "Cannot expand dimensions without drop_levels"
        )

    new_sizes = []
    new_strides = []

    for level in to_levels:
        # Find index of this level in from_levels
        try:
            idx = from_levels.index(level)
        except ValueError:
            # Level not found in from_levels
            if level.is_positional():
                new_sizes.append(1)
            else:
                new_sizes.append(level.dim().size)
            new_strides.append(0)
        else:
            new_sizes.append(sizes[idx])
            new_strides.append(strides[idx])

    return tensor.as_strided(new_sizes, new_strides, tensor.storage_offset())
