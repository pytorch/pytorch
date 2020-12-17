def _no_mutation(self, *args, **kwargs):
    raise NotImplementedError(f"'{type(self).__name__}' object does not support mutation")

def _create_immutable_container(base, mutable_functions):
    container = type('immutable_' + base.__name__, (base,), {})
    for attr in mutable_functions:
        setattr(container, attr, _no_mutation)
    return container

immutable_list = _create_immutable_container(list,
                                             ['__delitem__', '__iadd__', '__imul__', '__setitem__', 'append',
                                              'clear', 'extend', 'insert', 'pop', 'remove'])
immutable_list.__reduce__ = lambda self: (immutable_list, (tuple(iter(self)),))

immutable_dict = _create_immutable_container(dict, ['__delitem__', '__setitem__', 'clear', 'pop', 'popitem', 'update'])
immutable_dict.__reduce__ = lambda self: (immutable_dict, (iter(self.items()),))

class FakeNamedTuple(object):
    def __init__(self, field_names, values):
        self._field_names = field_names
        self._values = values

    def __repr__(self):
        return f'torch.fx.proxy.FakeNamedTuple({self._field_names}, {self._values})'

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, i):
        return self._values[i]

    def __getattr__(self, name):
        if name in self._field_names:
            return self._values[self._field_names.index(name)]
        return super().__getattribute__(name)

    def to_named_tuple(self, named_tuple_type):
        init_dict = {k : v for k, v in zip(self._field_names, self._values)}
        return named_tuple_type(**init_dict)
