
_help_mutation = """\
If you are attempting to modify the kwargs or args of a torch.fx.Node object,
instead create a new copy of it and assign the copy to the node:
    new_args = ... # copy and mutate args
    node.args = new_args
"""

def _no_mutation(self, *args, **kwargs):
    raise NotImplementedError(f"'{type(self).__name__}' object does not support mutation. {_help_mutation}")

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
