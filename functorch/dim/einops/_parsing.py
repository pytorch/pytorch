"""Adapted from https://github.com/arogozhnikov/einops/blob/36c7bb16e57d6e57f8f3050f9e07abdf3f00469f/einops/parsing.py."""
from __future__ import annotations

import functools
import keyword
import warnings
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

_ellipsis: str = "…"  # NB, this is a single unicode symbol. String is used as it is not a list, but can be iterated


class AnonymousAxis:
    """Used by `ParsedExpression` to represent an axis with a size (> 1), but no associated identifier.

    Note: Different instances of this class are not equal to each other, even if they have the same value.
    """

    def __init__(self, value: str) -> None:
        self.value = int(value)
        if self.value < 1:
            raise ValueError(f'Anonymous axis should have positive length, not {self.value}')

    def __repr__(self) -> str:
        return f"{self.value}-axis"


class ParsedExpression:
    """Structure containing information about one side of an `einops`-style pattern (e.g. 'b c (h w)')."""

    def __init__(self, expression: str, *, allow_underscore: bool = False, allow_duplicates: bool = False) -> None:
        """Parse the expression and store relevant metadata.

        Args:
            expression (str): the `einops`-pattern to parse
            allow_underscore (bool): whether to allow axis identifier names to begin with an underscore
            allow_duplicates (bool): whether to allow an identifier to appear more than once in the expression
        """
        self.has_ellipsis: bool = False
        self.has_ellipsis_parenthesized: Optional[bool] = None
        self.identifiers: Set[Union[str, AnonymousAxis]] = set()
        # that's axes like 2, 3, 4 or 5. Axes with size 1 are exceptional and replaced with empty composition
        self.has_non_unitary_anonymous_axes: bool = False
        # composition keeps structure of composite axes, see how different corner cases are handled in tests
        self.composition: List[Union[List[Union[str, AnonymousAxis]], str]] = []
        if "." in expression:
            if "..." not in expression:
                raise ValueError("Expression may contain dots only inside ellipsis (...)")
            if str.count(expression, "...") != 1 or str.count(expression, ".") != 3:
                raise ValueError(
                    "Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor ")
            expression = expression.replace("...", _ellipsis)
            self.has_ellipsis = True

        bracket_group: Optional[List[Union[str, AnonymousAxis]]] = None

        def add_axis_name(x: str) -> None:
            if x in self.identifiers:
                if not (allow_underscore and x == "_") and not allow_duplicates:
                    raise ValueError(f"Indexing expression contains duplicate dimension '{x}'")
            if x == _ellipsis:
                self.identifiers.add(_ellipsis)
                if bracket_group is None:
                    self.composition.append(_ellipsis)
                    self.has_ellipsis_parenthesized = False
                else:
                    bracket_group.append(_ellipsis)
                    self.has_ellipsis_parenthesized = True
            else:
                is_number = str.isdecimal(x)
                if is_number and int(x) == 1:
                    # handling the case of anonymous axis of length 1
                    if bracket_group is None:
                        self.composition.append([])
                    else:
                        pass  # no need to think about 1s inside parenthesis
                    return
                is_axis_name, reason = self.check_axis_name_return_reason(x, allow_underscore=allow_underscore)
                if not (is_number or is_axis_name):
                    raise ValueError(f"Invalid axis identifier: {x}\n{reason}")
                axis_name: Union[str, AnonymousAxis] = AnonymousAxis(x) if is_number else x
                self.identifiers.add(axis_name)
                if is_number:
                    self.has_non_unitary_anonymous_axes = True
                if bracket_group is None:
                    self.composition.append([axis_name])
                else:
                    bracket_group.append(axis_name)

        current_identifier = None
        for char in expression:
            if char in "() ":
                if current_identifier is not None:
                    add_axis_name(current_identifier)
                current_identifier = None
                if char == "(":
                    if bracket_group is not None:
                        raise ValueError("Axis composition is one-level (brackets inside brackets not allowed)")
                    bracket_group = []
                elif char == ")":
                    if bracket_group is None:
                        raise ValueError("Brackets are not balanced")
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ["_", _ellipsis]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise ValueError(f"Unknown character '{char}'")

        if bracket_group is not None:
            raise ValueError(f"Imbalanced parentheses in expression: '{expression}'")
        if current_identifier is not None:
            add_axis_name(current_identifier)

    @staticmethod
    def check_axis_name_return_reason(name: str, allow_underscore: bool = False) -> Tuple[bool, str]:
        """Check if the given axis name is valid, and a message explaining why if not.

        Valid axes names are python identifiers except keywords, and should not start or end with an underscore.

        Args:
            name (str): the axis name to check
            allow_underscore (bool): whether axis names are allowed to start with an underscore
        """
        if not str.isidentifier(name):
            return False, "not a valid python identifier"
        elif name[0] == "_" or name[-1] == "_":
            if name == "_" and allow_underscore:
                return True, ""
            return False, "axis name should should not start or end with underscore"
        else:
            if keyword.iskeyword(name):
                warnings.warn(f"It is discouraged to use axes names that are keywords: {name}", RuntimeWarning)
            if name in ["axis"]:
                warnings.warn(
                    "It is discouraged to use 'axis' as an axis name and will raise an error in future", FutureWarning
                )
            return True, ""

    @staticmethod
    def check_axis_name(name: str) -> bool:
        """Check if the name is a valid axis name.

        Args:
            name (str): the axis name to check
        """
        is_valid, _ = ParsedExpression.check_axis_name_return_reason(name)
        return is_valid


@functools.lru_cache(256)
def pattern_to_dim_idxs(
    tensor_ndim: int, pattern: str, **axes_lengths: int
) -> Tuple[
    int,
    List[Union[int, Tuple[int, ...]]],
    List[Union[int, Tuple[int, ...]]],
    Tuple[int, ...],
    Dict[str, int],
]:
    r"""Translate an `einops`-style pattern into lists/tuples of first class dims that can be used to perform
    `rearrange`.

    Since the an equivalent result is computed for tensors with the same number of dimensions, with the same pattern and
    specified axes lengths, this function can be memoized.

    Args:
        tensor_ndim (int): the number of dimensions in the tensor to rearrange
        pattern (str): the `einops`-style rearrangement pattern
        axes_lengths (int): any additional length specifications for dimensions

    Returns:
        A tuple of three elements:
            * a list of first class dims to use to index the tensor to rearrange
            * a list of first class dims to use to reorder the tensor
            * a tuple of first class dims representing any anonymous axes in the pattern, i.e. 1 -> 1, which should be
              summed over to produce the final tensor
    """
    # validation taken largely from einops.einops._prepare_transformation_recipe
    # https://github.com/arogozhnikov/einops/blob/230ac1526c1f42c9e1f7373912c7f8047496df11/einops/einops.py
    try:
        left_str, right_str = pattern.split("->")
    except ValueError:
        raise ValueError("Pattern must contain a single '->' separator") from None

    if _ellipsis in axes_lengths:
        raise ValueError(f"'{_ellipsis}' is not an allowed axis identifier")

    left = ParsedExpression(left_str)
    right = ParsedExpression(right_str)

    if not left.has_ellipsis and right.has_ellipsis:
        raise ValueError(f"Ellipsis found in right side, but not left side of a pattern {pattern}")
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise ValueError(f"Ellipsis is parenthesis in the left side is not allowed: {pattern}")

    # rearrange-specific validations
    difference = set.symmetric_difference(left.identifiers, right.identifiers)
    if left.has_non_unitary_anonymous_axes or right.has_non_unitary_anonymous_axes:
        raise ValueError("Non-unitary anonymous axes are not supported in rearrange (exception is length 1)")
    if len(difference) > 0:
        raise ValueError(f"Identifiers only on one side of expression (should be on both): {difference}")
    unmatched_axes = axes_lengths.keys() - left.identifiers
    if len(unmatched_axes) > 0:
        raise ValueError(f"Identifiers not found in expression: {unmatched_axes}")

    if left.has_ellipsis:
        n_ellipsis_dims = tensor_ndim - (len(left.composition) - 1)
        n_dims = len(left.identifiers) - 1
        total_n_dims = n_dims + n_ellipsis_dims
    else:
        n_ellipsis_dims = 0
        total_n_dims = n_dims = len(left.identifiers)

    n_anon_dims = sum(not dim for dim in left.composition)
    total_n_dims += n_anon_dims

    identifier_index_map: Dict[Union[str, AnonymousAxis], Tuple[int, ...]] = {}
    anon_axes: List[AnonymousAxis] = []

    # map the left-hand side identifiers to the indices of the first class dims
    dim_i = 0
    for dimension in left.composition:
        if isinstance(dimension, list):
            for identifier in dimension:
                assert isinstance(identifier, str)
                identifier_index_map[identifier] = (dim_i,)
                dim_i += 1
            if not dimension:
                anon_axis = AnonymousAxis("1")
                identifier_index_map[anon_axis] = (dim_i,)
                anon_axes.append(anon_axis)
                dimension.append(anon_axis)
                dim_i += 1
        elif dimension == _ellipsis:
            identifier = _ellipsis
            identifier_index_map[identifier] = tuple(dim_i + j for j in range(n_ellipsis_dims))
            dim_i += n_ellipsis_dims
        else:
            raise ValueError(f"Unexpected dimension: {dimension}")

    def composition_to_dims(
        composition: Sequence[Union[List[Union[str, AnonymousAxis]], str]]
    ) -> List[Union[int, Tuple[int, ...]]]:
        """Convert identifiers in a `ParsedExpression.composition` to their corresponding first class dim index."""
        index_composition: List[Union[int, Tuple[int, ...]]] = []
        for dimension in composition:
            if isinstance(dimension, list):
                index_composition.append(
                    tuple(dim for identifier in dimension for dim in identifier_index_map[identifier])
                )
            elif dimension == _ellipsis:
                index_composition.extend(identifier_index_map[_ellipsis])
            else:
                raise ValueError(f"Unexpected dimension: {dimension}")
        return index_composition

    left_idxs = composition_to_dims(left.composition)
    right_idxs = composition_to_dims(right.composition)
    anon_idxs = tuple(identifier_index_map[axis][0] for axis in anon_axes)
    axes_idx_map = {
        axis: identifier_index_map[axis][0] for axis in axes_lengths
    }

    return total_n_dims, left_idxs, right_idxs, anon_idxs, axes_idx_map
