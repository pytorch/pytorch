# Taking reference from official Python typing
# https://github.com/python/cpython/blob/master/Lib/typing.py

import collections
import copy
import numbers
from typing import Any, Dict, List, Set, Tuple, TypeVar, Union, get_type_hints
from typing import _type_repr  # type: ignore


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
            if hasattr(constraint, '__origin__'):
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


def _fixed_type(param) -> bool:
    if isinstance(param, TypeVar) or param in (Any, ...):  # type: ignore
        return False
    if hasattr(param, '__args__'):
        if len(param.__args__) == 0:
            return False
        for arg in param.__args__:
            if not _fixed_type(arg):
                return False
    return True


class _DataPipeType:
    def __init__(self, param):
        self.param = param
        self.fixed = _fixed_type(param)

    def __repr__(self):
        return _type_repr(self.param)

    def __eq__(self, other):
        if isinstance(other, _DataPipeType):
            return self.param == other.param
        return NotImplementedError

    def issubtype(self, other):
        if not isinstance(other, _DataPipeType):
            raise TypeError("Expected '_DataPipeType', but found {}".format(type(other)))
        return issubtype(self.param, other.param)


# Mimic generic typing as _GenericAlias (introduced in Python3.7)
class _DataPipeAlias:
    def __init__(self, origin, param):
        self._type = _DataPipeType(param)
        self._name = 'DataPipe[' + str(self._type) + ']'
        if self._type.fixed:
            self.__origin__ = type(self._name, (origin, ),
                                   {'__init_subclass__': _DataPipeAlias.fixed_type_init,
                                    'type': self._type})
        else:
            self.__origin__ = type(self._name, (origin, ),
                                   {'__init_subclass__': _DataPipeAlias.nonfixed_type_init,
                                    'type': self._type})

    def __eq__(self, other):
        if not isinstance(other, _DataPipeAlias):
            return NotImplemented
        return (self.__origin__ == other.__origin__
                and self._type == other._type)

    def __hash__(self):
        return hash((self.__origin__, self._type))

    def __repr__(self):
        return '{}[{}]'.format(self._name, str(self._type))

    def __mro_entries__(self, bases):
        return (self.__origin__, )

    @staticmethod
    def static_check_iter(sub_cls):
        # TODO: Determine if __iter__ is strictly required for DataPipe
        if '__iter__' in sub_cls.__dict__:
            iter_fn = sub_cls.__dict__['__iter__']
            hints = get_type_hints(iter_fn)
            if 'return' not in hints:
                raise TypeError('No return annotation found for `__iter__` of {}'.format(sub_cls.__name__))
            return_hint = hints['return']
            if not hasattr(return_hint, '__origin__') or return_hint.__origin__ is not collections.abc.Iterator:
                raise TypeError('Expected Iterator as the return annotation for `__iter__` of {}'
                                ', but found {}'.format(sub_cls.__name__, hints['return']))
            data_type = return_hint.__args__[0]
            if sub_cls.type.param != data_type:
                raise TypeError('Unmatched type annotation for {} ({} vs {})'
                                .format(sub_cls.__name__, sub_cls.type, _type_repr(data_type)))

    @staticmethod
    def fixed_type_init(sub_cls, *args, **kwargs):
        _DataPipeAlias.static_check_iter(sub_cls)

    @staticmethod
    def nonfixed_type_init(sub_cls, *args, **kwargs):
        _DataPipeAlias.static_check_iter(sub_cls)
        if '__init__' in sub_cls.__dict__:
            init_fn = sub_cls.__dict__['__init__']

            def new_init(self, *args, **kwargs):
                self.type = copy.deepcopy(self.type)
                init_fn(self, *args, **kwargs)
        else:
            def new_init(self, *args, **kwargs):
                self.type = copy.deepcopy(self.type)
        sub_cls.__init__ = new_init
