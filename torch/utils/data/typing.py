# Taking reference from official Python typing
# https://github.com/python/cpython/blob/master/Lib/typing.py

import collections
import copy
from typing import get_type_hints, Any, TypeVar
from typing import _type_repr  # type: ignore


def _fixed_type(param) -> bool:
    if isinstance(param, TypeVar) or param in (Any, ...):  # type: ignore
        return False
    if hasattr(param, '__args__'):
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


# _GenericAlias from typing is introduced after Python 3.6
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
                raise TypeError('Iterator is required as the return annotation for `__iter__` of {}'
                                ', but {} is found'.format(sub_cls.__name__, hints['return']))
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
                init_fn(self, *args, **kwargs)
                self.type = copy.deepcopy(self.type)
        else:
            def new_init(self, *args, **kwargs):
                self.type = copy.deepcopy(self.type)
        sub_cls.__init__ = new_init
