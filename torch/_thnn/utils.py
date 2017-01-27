import os
import itertools
import importlib

THNN_H_PATH = os.path.join(os.path.dirname(__file__), '..', 'lib', 'THNN.h')
THCUNN_H_PATH = os.path.join(os.path.dirname(__file__), '..', 'lib', 'THCUNN.h')


def _unpickle_backend(backend_name):
    import torch._thnn
    return torch._thnn.type2backend[backend_name]


class THNNBackendBase(object):

    def __init__(self):
        self.methods = {}

    def __getattr__(self, name):
        method = self.methods.get(name, None)
        if method is None:
            raise NotImplementedError
        return method

    def register_method(self, name, ctypes_fn):
        self.methods[name] = ctypes_fn

    @property
    def library_state(self):
        return 0

    def __reduce__(self):
        return (_unpickle_backend, (type(self).__name__,))


class Function(object):

    def __init__(self, name):
        self.name = name
        self.arguments = []

    def add_argument(self, arg):
        assert isinstance(arg, Argument)
        self.arguments.append(arg)

    def __repr__(self):
        return self.name + '(' + ', '.join(map(lambda a: a.__repr__(), self.arguments)) + ')'


class Argument(object):

    def __init__(self, _type, name, is_optional):
        self.type = _type
        self.name = name
        self.is_optional = is_optional

    def __repr__(self):
        return self.type + ' ' + self.name


def parse_header(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')

    # Remove empty lines and preprocessor directives
    lines = filter(lambda l: l and not l.startswith('#'), lines)
    # Remove line comments
    lines = map(lambda l: l.partition('//'), lines)
    # Select line and comment part
    lines = map(lambda l: (l[0].strip(), l[2].strip()), lines)
    # Remove trailing special signs
    lines = map(lambda l: (l[0].rstrip(');').rstrip(','), l[1]), lines)
    # Split arguments
    lines = map(lambda l: (l[0].split(','), l[1]), lines)
    # Flatten lines
    new_lines = []
    for l, c in lines:
        for split in l:
            new_lines.append((split, c))
    lines = new_lines
    del new_lines
    # Remove unnecessary whitespace
    lines = map(lambda l: (l[0].strip(), l[1]), lines)
    # Remove empty lines
    lines = filter(lambda l: l[0], lines)
    generic_functions = []
    for l, c in lines:
        if l.startswith('TH_API void THNN_'):
            fn_name = l.lstrip('TH_API void THNN_')
            if fn_name[0] == '(' and fn_name[-2] == ')':
                fn_name = fn_name[1:-2]
            else:
                fn_name = fn_name[:-1]
            generic_functions.append(Function(fn_name))
        elif l:
            t, name = l.split()
            if '*' in name:
                t = t + '*'
                name = name[1:]
            generic_functions[-1].add_argument(Argument(t, name, '[OPTIONAL]' in c))
    return generic_functions


def load_backend(t, lib, generic_functions, mixins=tuple()):
    lib_handle = importlib.import_module(lib)
    backend_name = 'THNN{}Backend'.format(t)
    backend = type(backend_name, mixins + (THNNBackendBase,), {})()
    for function in generic_functions:
        full_fn_name = '{}{}'.format(t, function.name)
        fn = getattr(lib_handle, full_fn_name)
        backend.register_method(function.name, fn)
    return backend
