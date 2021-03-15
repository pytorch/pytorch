# Taking reference from official Python typing
# https://github.com/python/cpython/blob/master/Lib/typing.py

import collections
import copy
from typing import get_type_hints, Any, TypeVar
from typing import _GenericAlias, _type_repr  # type: ignore


def _fixed_type(param):
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
        elif isinstance(other, (type, TypeVar, _GenericAlias)):  # type: ignore
            return self.param == other
        return NotImplementedError


class _DataPipeAlias(_GenericAlias, _root=True):  # type: ignore
    def __init__(self, origin, param, *, inst=True, name=None):
        super().__init__(origin, params=param, inst=inst, name=name)
        self.datapipe_type = _DataPipeType(param)
        self.datapipe_name = 'DataPipe[' + str(self.datapipe_type) + ']'
        if self.datapipe_type.fixed:
            self.__origin__ = type(self.datapipe_name, (origin, ),
                                   {'__init_subclass__': _DataPipeAlias.fixed_type_init,
                                    'type': self.datapipe_type})
        else:
            self.__origin__ = type(self.datapipe_name, (origin, ),
                                   {'__init_subclass__': _DataPipeAlias.nonfixed_type_init,
                                    'type': self.datapipe_type})

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
            if sub_cls.type != data_type:
                raise TypeError('Unmatched type annotation for {} ({} vs {})'
                                .format(sub_cls.__name__, sub_cls.type, data_type))

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
