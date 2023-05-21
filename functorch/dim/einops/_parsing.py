"""Adapted from https://github.com/arogozhnikov/einops/blob/master/einops/parsing.py."""
import keyword
import warnings
from typing import List, Optional, Set, Tuple, Union

_ellipsis: str = '…'  # NB, this is a single unicode symbol. String is used as it is not a list, but can be iterated


class AnonymousAxis:
    """Used by `ParsedExpression` to represent an axis with a size (> 1), but no associated identifier.

    Note: Different instances of this class are not equal to each other, even if they have the same value.
    """

    def __init__(self, value: str) -> None:
        self.value = int(value)
        if self.value <= 1:
            if self.value == 1:
                raise ValueError('No need to create anonymous axis of length 1. Report this as an issue')
            else:
                raise ValueError(f'Anonymous axis should have positive length, not {self.value}')

    def __repr__(self) -> str:
        return f"{self.value}-axis"


class ParsedExpression:
    """Structure containing information about one side of an `einops`-style pattern (e.g. 'b c (h w)')."""

    def __init__(self, expression: str, *, allow_underscore: bool = False, allow_duplicates: bool = False) -> None:
        self.has_ellipsis: bool = False
        self.has_ellipsis_parenthesized: Optional[bool] = None
        self.identifiers: Set[Union[str, AnonymousAxis]] = set()
        # that's axes like 2, 3, 4 or 5. Axes with size 1 are exceptional and replaced with empty composition
        self.has_non_unitary_anonymous_axes: bool = False
        # composition keeps structure of composite axes, see how different corner cases are handled in tests
        self.composition: List[Union[List[Union[str, AnonymousAxis]], str]] = []
        if '.' in expression:
            if '...' not in expression:
                raise ValueError('Expression may contain dots only inside ellipsis (...)')
            if str.count(expression, '...') != 1 or str.count(expression, '.') != 3:
                raise ValueError(
                    'Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor ')
            expression = expression.replace('...', _ellipsis)
            self.has_ellipsis = True

        bracket_group: Optional[List[Union[str, AnonymousAxis]]] = None

        def add_axis_name(x: str) -> None:
            if x in self.identifiers:
                if not (allow_underscore and x == "_") and not allow_duplicates:
                    raise ValueError(f'Indexing expression contains duplicate dimension "{x}"')
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
                    raise ValueError(f'Invalid axis identifier: {x}\n{reason}')
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
            if char in '() ':
                if current_identifier is not None:
                    add_axis_name(current_identifier)
                current_identifier = None
                if char == '(':
                    if bracket_group is not None:
                        raise ValueError("Axis composition is one-level (brackets inside brackets not allowed)")
                    bracket_group = []
                elif char == ')':
                    if bracket_group is None:
                        raise ValueError('Brackets are not balanced')
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ['_', _ellipsis]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise ValueError(f"Unknown character '{char}'")

        if bracket_group is not None:
            raise ValueError(f'Imbalanced parentheses in expression: "{expression}"')
        if current_identifier is not None:
            add_axis_name(current_identifier)

    @staticmethod
    def check_axis_name_return_reason(name: str, allow_underscore: bool = False) -> Tuple[bool, str]:
        if not str.isidentifier(name):
            return False, 'not a valid python identifier'
        elif name[0] == '_' or name[-1] == '_':
            if name == '_' and allow_underscore:
                return True, ''
            return False, 'axis name should should not start or end with underscore'
        else:
            if keyword.iskeyword(name):
                warnings.warn(f"It is discouraged to use axes names that are keywords: {name}", RuntimeWarning)
            if name in ['axis']:
                warnings.warn(
                    "It is discouraged to use 'axis' as an axis name and will raise an error in future", FutureWarning
                )
            return True, ''

    @staticmethod
    def check_axis_name(name: str) -> bool:
        """
        Valid axes names are python identifiers except keywords,
        and additionally should not start or end with underscore
        """
        is_valid, _ = ParsedExpression.check_axis_name_return_reason(name)
        return is_valid
