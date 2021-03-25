# Taking reference from official Python typing
# https://github.com/python/cpython/blob/master/Lib/typing.py

import collections
import copy
import numbers
from typing import (Any, Dict, Iterator, List, Set, Sequence, Tuple,
                    TypeVar, Union, get_type_hints)
from typing import _tp_cache, _type_check, _type_repr  # type: ignore
try:
    from typing import GenericMeta  # Python 3.6
except ImportError:  # Python > 3.6
    class GenericMeta(type):  # type: ignore
        pass


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


# In order to keep compatibility for Python 3.6, use Meta for the typing.
# TODO: When PyTorch drops the support for Python 3.6, it can be converted
# into the Alias system and using `__class_getiterm__` for DataPipe. The
# typing system will gain benefit of performance and metaclass conflicts,
# as elaborated in https://www.python.org/dev/peps/pep-0560/


def _fixed_type(param) -> bool:
    if isinstance(param, TypeVar) or param in (Any, ...):  # type: ignore
        return False
    if hasattr(param, '__args__'):
        # For Python 3.6, `__args__` can be None
        if param.__args__ is None or len(param.__args__) == 0:
            return False
        for arg in param.__args__:
            if not _fixed_type(arg):
                return False
    return True


class _DataPipeType:
    r"""
    Save type in `param` and check if it's fixed or non-fixed type
    """

    def __init__(self, param):
        self.param = param
        self.fixed = _fixed_type(param)

    def __repr__(self):
        return _type_repr(self.param)

    def __eq__(self, other):
        if isinstance(other, _DataPipeType):
            return self.param == other.param
        return NotImplemented

    def __hash__(self):
        return hash(self.param)

    def issubtype(self, other):
        if isinstance(other, _DataPipeType):
            return issubtype(self.param, other.param)
        if isinstance(other, type):
            return issubtype(self.param, other)
        raise TypeError("Expected '_DataPipeType' or 'type', but found {}".format(type(other)))


# Default type for DataPipe without annotation
_DEFAULT_TYPE = _DataPipeType(Any)


def _mro_subclass_init(obj, fixed):
    r"""
    Run through MRO to check if any super class has already built in
    the corresponding `__init_subclass__`. If so, no need to add
    `__init_subclass__`.
    """

    mro = obj.__mro__
    for b in mro:
        if isinstance(b, _DataPipeMeta):
            if fixed:
                if b.__init_subclass__ == fixed_type_init:
                    return True
                if hasattr(b.__init_subclass__, '__func__') and \
                        b.__init_subclass__.__func__ == fixed_type_init:  # type: ignore
                    return True
            if not fixed:
                if b.__init_subclass__ == nonfixed_type_init:
                    return True
                if hasattr(b.__init_subclass__, '__func__') and \
                        b.__init_subclass__.__func__ == nonfixed_type_init:  # type: ignore
                    return True
    return False


class _DataPipeMeta(GenericMeta):
    r"""
    Metaclass for `DataPipe`. Add `type` attribute and `__init_subclass__` based
    on the type, and validate the return hint of `__iter__`.
    """
    type: _DataPipeType

    def __new__(cls, name, bases, namespace, **kargs):
        # For Python > 3.6
        cls.__origin__ = None
        # Save __init__ function
        if '__init__' in namespace:
            namespace.update({'_origin_init': namespace['__init__']})
        if 'type' in namespace:
            return super().__new__(cls, name, bases, namespace)

        # For plain derived class without annotation
        t = None
        for base in bases:
            if isinstance(base, _DataPipeMeta):
                t = base.type
                break
        if t is not None:
            namespace.update({'type': t})
        else:
            namespace.update({'type': _DEFAULT_TYPE, '__init_subclass__': nonfixed_type_init})

        return super().__new__(cls, name, bases, namespace)

    @_tp_cache
    def __getitem__(self, param):
        if param is None:
            raise TypeError('{}[t]: t can not be None'.format(self.__name__))
        if isinstance(param, Sequence):
            param = Tuple[param]
        _type_check(param, msg="{}[t]: t must be a type".format(self.__name__))
        t = _DataPipeType(param)

        if not t.issubtype(self.type):
            raise TypeError('Can not subclass a DataPipe[{}] from DataPipe[{}]'
                            .format(t, self.type))

        # Types are equal, fast path for inheritance
        if self.type.issubtype(t):
            if _mro_subclass_init(self, t.fixed):
                return self

        name = self.__name__ + '[' + str(t) + ']'
        bases = (self,) + self.__bases__

        if t.fixed:
            return self.__class__(name, bases,
                                  {'__init_subclass__': fixed_type_init,
                                   'type': t})
        else:
            return self.__class__(name, bases,
                                  {'__init_subclass__': nonfixed_type_init,
                                   'type': t})

    def __eq__(self, other):
        if not isinstance(other, _DataPipeMeta):
            return NotImplemented
        if self.__origin__ is None or other.__origin__ is None:
            return self is other
        return (self.__origin__ == other.__origin__
                and self.type == other.type)

    def __hash__(self):
        return hash((self.__name__, self.type))


def _validate_iter(sub_cls):
    # TODO:
    # - add global switch for type checking at compile-time
    # - Determine if __iter__ is strictly required for DataPipe
    if '__iter__' in sub_cls.__dict__:
        iter_fn = sub_cls.__dict__['__iter__']
        hints = get_type_hints(iter_fn)
        if sub_cls.type != _DEFAULT_TYPE and 'return' not in hints:
            raise TypeError('No return annotation found for `__iter__` of {}'.format(sub_cls.__name__))
        if 'return' in hints:
            return_hint = hints['return']
            # Plain Return Hint for Python 3.6
            if return_hint == Iterator:
                return
            if not (hasattr(return_hint, '__origin__') and
                    (return_hint.__origin__ == Iterator or
                     return_hint.__origin__ == collections.abc.Iterator)):
                raise TypeError('Expected Iterator as the return annotation for `__iter__` of {}'
                                ', but found {}'.format(sub_cls.__name__, hints['return']))
            data_type = return_hint.__args__[0]
            # Double-side subtype checking to make sure type matched
            # e.g. T_co == Any, T_co == S_co
            if not (issubtype(sub_cls.type.param, data_type) and issubtype(data_type, sub_cls.type.param)):
                raise TypeError('Unmatched type annotation for {} ({} vs {})'
                                .format(sub_cls.__name__, sub_cls.type, _type_repr(data_type)))


def fixed_type_init(sub_cls, *args, **kwargs):
    _validate_iter(sub_cls)
    if '_origin_init' in sub_cls.__dict__:
        sub_cls.__init__ = sub_cls._origin_init
    else:

        # Fake __init__ function
        def fake_init(self, *args, **kwargs):
            pass
        sub_cls.__init__ = fake_init


def nonfixed_type_init(sub_cls, *args, **kwargs):
    _validate_iter(sub_cls)
    if '_origin_init' in sub_cls.__dict__:
        init_fn = sub_cls.__dict__['_origin_init']

        def new_init(self, *args, **kwargs):
            init_fn(self, *args, **kwargs)
            self.type = copy.deepcopy(sub_cls.type)
    else:
        def new_init(self, *args, **kwargs):
            self.type = copy.deepcopy(sub_cls.type)
    sub_cls.__init__ = new_init
