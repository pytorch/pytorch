import numbers
from typing import Any, Dict, List, Set, Tuple, TypeVar, Union


class Integer(numbers.Integral):
    pass


class Boolean(numbers.Integral):
    pass


# Python 'type' object is not subscriptable
# Tuple[int, List, dict] -> valid
# tuple[int, list, dict] -> invalid
# Map Python 'type' to abstract base class
TYPE2ABC = {
    bool: Boolean,
    int: Integer,
    float: numbers.Real,
    complex: numbers.Complex,
    dict: Dict,
    list: List,
    set: Set,
    tuple: Tuple,
    None: type(None),
}


def issubtype(left, right):
    r"""
    Check if the left-side type is a subtype of the right-side type.
    If any of type is a composite type like `Union` and `TypeVar` with
    bounds, it would be expanded into a list of types and check all
    of left-side types are subtypes of either one from right-side types.
    """
    left = TYPE2ABC.get(left, left)
    right = TYPE2ABC.get(right, right)

    if right is Any or left == right:
        return True

    if right == type(None):
        return False

    # Right-side type
    constraints = _decompose_type(right)

    if len(constraints) == 0 or Any in constraints:
        return True

    if left is Any:
        return False

    # Left-side type
    variants = _decompose_type(left)

    # all() will return True for empty variants
    if len(variants) == 0:
        return False

    return all(_issubtype_with_constraints(variant, constraints) for variant in variants)


def _decompose_type(t, to_list=True):
    if isinstance(t, TypeVar):  # type: ignore
        if t.__bound__ is not None:
            ts = [t.__bound__]
        else:
            # For T_co, __constraints__ is ()
            ts = t.__constraints__
    elif hasattr(t, '__origin__') and t.__origin__ == Union:
        ts = t.__args__
    else:
        if not to_list:
            return None
        ts = [t]
    ts = list(TYPE2ABC.get(_t, _t) for _t in ts)
    return ts


def _issubtype_with_constraints(variant, constraints):
    r"""
    Check if the variant is a subtype of either one from constraints.
    For composite types like `Union` and `TypeVar` with bounds, they
    would be expanded for testing.
    """
    if variant in constraints:
        return True

    # [Note: Subtype for Union and TypeVar]
    # Python typing is able to flatten Union[Union[...]] or Union[TypeVar].
    # But it couldn't flatten the following scenarios:
    #   - Union[int, TypeVar[Union[...]]]
    #   - TypeVar[TypeVar[...]]
    # So, variant and each constraint may be a TypeVar or a Union.
    # In these cases, all of inner types from the variant are required to be
    # extraced and verified as a subtype of any constraint. And, all of
    # inner types from any constraint being a TypeVar or a Union are
    # also required to be extracted and verified if the variant belongs to
    # any of them.

    # Variant
    vs = _decompose_type(variant, to_list=False)

    # Variant is TypeVar or Union
    if vs is not None:
        return all(_issubtype_with_constraints(v, constraints) for v in vs)

    # Variant is not TypeVar or Union
    if hasattr(variant, '__origin__') and variant.__origin__ is not None:
        v_origin = variant.__origin__
        v_args = variant.__args__
    else:
        v_origin = variant
        v_args = None

    # Constraints
    for constraint in constraints:
        cs = _decompose_type(constraint, to_list=False)

        # Constraint is TypeVar or Union
        if cs is not None:
            if _issubtype_with_constraints(variant, cs):
                return True
        # Constraint is not TypeVar or Union
        else:
            # __origin__ can be None for plain list, tuple, ... in Python 3.6
            if hasattr(constraint, '__origin__') and constraint.__origin__ is not None:
                c_origin = constraint.__origin__
                if v_origin == c_origin:
                    c_args = constraint.__args__
                    if c_args is None or len(c_args) == 0:
                        return True
                    if v_args is not None and len(v_args) == len(c_args) and \
                            all(issubtype(v_arg, c_arg) for v_arg, c_arg in zip(v_args, c_args)):
                        return True
            # Tuple[int] -> Tuple
            else:
                if v_origin == constraint:
                    return True

    return False
