# Taking reference from official Python typing
# https://github.com/python/cpython/blob/master/Lib/typing.py

import collections
import copy
import numbers
from typing import Any, Dict, Generic, List, Set, Sequence, Tuple, TypeVar, Union, get_type_hints
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


# In order to keep compatibility for Python 3.6, use Meta for the typing.
# TODO: When PyTorch drops the support for Python 3.6, it can be converted
# into the Alias system and using `__class_getiterm__` for DataPipe. The
# typing system will gain benefit of performance and metaclass conflicts,
# as elaborated in https://www.python.org/dev/peps/pep-0560/

# Took reference from
# https://github.com/python/cpython/blob/3.6/Lib/typing.py#L1180-L1201
def _generic_new(base_cls, cls, *args, **kwds):
    # Assure type is erased on instantiation,
    # but attempt to store it in __orig_class__
    if cls.__origin__ is None:
        if (base_cls.__new__ is object.__new__ and
                cls.__init__ is not object.__init__):
            return base_cls.__new__(cls)
        else:
            return base_cls.__new__(cls, *args, **kwds)
    else:
        origin = cls._gorg
        if (base_cls.__new__ is object.__new__ and
                cls.__init__ is not object.__init__):
            obj = base_cls.__new__(origin)
        else:
            obj = base_cls.__new__(origin, *args, **kwds)
        try:
            obj.__orig_class__ = cls
        except AttributeError:
            pass
        obj.__init__(*args, **kwds)
        return obj

# Took reference from
# https://github.com/python/cpython/blob/3.6/Lib/typing.py#L852-L863
def _next_in_mro(cls):
    """Helper for Generic.__new__.

    Returns the class after the last occurrence of Generic or
    Generic[...] in cls.__mro__.
    """
    next_in_mro = object
    # Look for the last occurrence of Generic or Generic[...].
    for i, c in enumerate(cls.__mro__[:-1]):
        if isinstance(c, GenericMeta) and hasattr(c, '_gorg') and c._gorg is Generic:  # type: ignore
            next_in_mro = cls.__mro__[i + 1]
    return next_in_mro

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

    def __hash__(self):
        return hash(self.param)

    def issubtype(self, other):
        if isinstance(other, _DataPipeType):
            return issubtype(self.param, other.param)
        if isinstance(other, type):
            return issubtype(self.param, other)
        raise TypeError("Expected '_DataPipeType' or 'type', but found {}".format(type(other)))


_DEFAULT_TYPE = _DataPipeType(Any)


class _DataPipeMeta(GenericMeta):
    def __new__(cls, name, bases, namespace, **kargs):
        # For Python > 3.6
        cls.__origin__ = None

        # For plain derived class without annotation
        t = None
        for base in bases:
            if hasattr(base, 'type') and isinstance(base.type, _DataPipeType):
                t = base.type
                break
        if t is not None:
            if 'type' in namespace:
                if not namespace['type'].issubtype(t):
                    raise TypeError('Can not derive a DataPipe[{}] from DataPipe[{}]'
                                    .format(namespace['type'], t))
            else:
                namespace['type'] = t
        else:
            if 'type' not in namespace:
                namespace['type'] = _DEFAULT_TYPE
                namespace['__init_subclass__'] = nonfixed_type_init
        return super().__new__(cls, name, bases, namespace)

    @_tp_cache
    def __getitem__(self, param):
        if param is None:
            raise TypeError('{}[t]: t can not be None'.format(self.__name__))
        if isinstance(param, Sequence):
            param = Tuple[param]
        _type_check(param, msg="{}[t]: t must be a type".format(self.__name__))
        t = _DataPipeType(param)
        name = 'DataPipe[' + str(t) + ']'
        bases = (self,) if self.__origin__ is None else ()
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
        return (self.__origin__ == other.__origin__
                and self.type == other.type)  # type: ignore

    def __hash__(self):
        return hash((self.__origin__, self.type))  # type: ignore


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
            if not hasattr(return_hint, '__origin__') or return_hint.__origin__ is not collections.abc.Iterator:
                raise TypeError('Expected Iterator as the return annotation for `__iter__` of {}'
                                ', but found {}'.format(sub_cls.__name__, hints['return']))
            data_type = return_hint.__args__[0]
            if not (issubtype(sub_cls.type.param, data_type) and issubtype(data_type, sub_cls.type.param)):
                raise TypeError('Unmatched type annotation for {} ({} vs {})'
                                .format(sub_cls.__name__, sub_cls.type, _type_repr(data_type)))


def fixed_type_init(sub_cls, *args, **kwargs):
    _validate_iter(sub_cls)


def nonfixed_type_init(sub_cls, *args, **kwargs):
    _validate_iter(sub_cls)
    if '__init__' in sub_cls.__dict__:
        init_fn = sub_cls.__dict__['__init__']

        def new_init(self, *args, **kwargs):
            init_fn(self, *args, **kwargs)
            self.type = copy.deepcopy(sub_cls.type)
    else:
        def new_init(self, *args, **kwargs):
            self.type = copy.deepcopy(sub_cls.type)
    sub_cls.__init__ = new_init
