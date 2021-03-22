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


# Check if the left-side type is a subtype for the right-side type
def issubtype(left, right):
    left = TYPE2ABC.get(left, left)
    right = TYPE2ABC.get(right, right)

    if right is Any or left == right:
        return True

    if right == type(None):
        return False

    # Right-side type
    if isinstance(right, TypeVar):  # type: ignore
        if right.__bound__ is not None:
            constraints = [right.__bound__]
        else:
            constraints = right.__constraints__
    elif hasattr(right, '__origin__') and right.__origin__ == Union:
        constraints = right.__args__
    else:
        constraints = [right]
    constraints = [TYPE2ABC.get(constraint, constraint) for constraint in constraints]

    if len(constraints) == 0 or Any in constraints:
        return True

    if left is Any:
        return False

    # Left-side type
    if isinstance(left, TypeVar):  # type: ignore
        if left.__bound__ is not None:
            variants = [left.__bound__]
        else:
            variants = left.__constraints__
    elif hasattr(left, '__origin__') and left.__origin__ == Union:
        variants = left.__args__
    else:
        variants = [left]
    variants = [TYPE2ABC.get(variant, variant) for variant in variants]

    if len(variants) == 0:
        return False

    return all(_issubtype_with_constraints(variant, constraints) for variant in variants)


# Check if the variant is a subtype for any of constraints
def _issubtype_with_constraints(variant, constraints):
    if variant in constraints:
        return True

    # [Note: Subtype for Union and TypeVar]
    # Python typing is able to flatten Union[Union[...]] to one Union[...]
    # But it couldn't flatten the following scenarios:
    #   - Union[TypeVar[Union[...]]]
    #   - TypeVar[TypeVar[...]]
    # So, variant and each constraint may be a TypeVar or a Union.
    # In these cases, all of inner types from the variant are required to be
    # extraced and verified as a subtype of any constraint. And, all of
    # inner types from any constraint being a TypeVar or a Union are
    # also required to be extracted and verified if the variant belongs to
    # any of them.

    # Variant
    vs = None
    if isinstance(variant, TypeVar):  # type: ignore
        if variant.__bound__ is not None:
            vs = [variant.__bound__]
        elif len(variant.__constraints__) > 0:
            vs = variant.__constraints__
        # Both empty like T_co
    elif hasattr(variant, '__origin__') and variant.__origin__ == Union:
        vs = variant.__args__
    # Variant is TypeVar or Union
    if vs is not None:
        vs = [TYPE2ABC.get(v, v) for v in vs]
        return all(_issubtype_with_constraints(v, constraints) for v in vs)

    # Variant is not TypeVar or Union
    if hasattr(variant, '__origin__'):
        v_origin = variant.__origin__
        v_args = variant.__args__
    else:
        v_origin = variant
        v_args = None

    for constraint in constraints:
        # Constraint
        cs = None
        if isinstance(constraint, TypeVar):  # type: ignore
            if constraint.__bound__ is not None:
                cs = [constraint.__bound__]
            elif len(constraint.__constraints__) > 0:
                cs = constraint.__constraints__
        elif hasattr(constraint, '__origin__') and constraint.__origin__ == Union:
            cs = constraint.__args__

        # Constraint is TypeVar or Union
        if cs is not None:
            cs = [TYPE2ABC.get(c, c) for c in cs]
            if _issubtype_with_constraints(variant, cs):
                return True
        # Constraint is not TypeVar or Union
        else:
            # __origin__ can be None for list, tuple, ... in Python 3.6
            if hasattr(constraint, '__origin__') and constraint.__origin__ is not None:
                c_origin = constraint.__origin__
                if v_origin == c_origin:
                    c_args = constraint.__args__
                    if v_args is None or len(c_args) == 0:
                        return True
                    if len(v_args) == len(c_args) and \
                            all(issubtype(v_arg, c_arg) for v_arg, c_arg in zip(v_args, c_args)):
                        return True
            else:
                if v_origin == constraint:
                    return v_args is None or len(v_args) == 0

    return False
