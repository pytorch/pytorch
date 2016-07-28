import ctypes
import itertools
from pprint import pprint


# TODO: submodule THNN
THNN_H_PATH = '/Users/apaszke/pytorch/pytorch/torch/legacy/nn/THNN.h'
THNN_LIB_PATH = '/Users/apaszke/torch/install/lib/lua/5.1/libTHNN.so'

with open(THNN_H_PATH, 'r') as f:
    lines = f.read().split('\n')

# Remove empty lines and preprocessor directives
lines = filter(lambda l: l and not l.startswith('#'), lines)
# Remove line comments
lines = map(lambda l: l.partition('//')[0], lines)
# Remove trailing special signs
lines = map(lambda l: l.rstrip(');').rstrip(','), lines)
# Split arguments
lines = map(lambda l: l.split(','), lines)
# Flatten list
lines = itertools.chain.from_iterable(lines)
# Remove unnecessary whitespace
lines = map(lambda l: l.strip(), lines)
# Remove empty lines
lines = filter(lambda l: l, lines)

class Function(object):
    def __init__(self, name):
        self.name = name
        self.arguments = []

    def add_argument(self, arg):
        self.arguments.append(arg)

    def __repr__(self):
        return self.name + '(' + ', '.join(self.arguments) + ')'

generic_functions = []
for l in lines:
    if l.startswith('TH_API void THNN_'):
        fn_name = l.lstrip('TH_API void THNN_')[1:-2]
        generic_functions.append(Function(fn_name))
    else:
        t, name = l.split(' ')
        if '*' in name:
            t = t + '*'
        generic_functions[-1].add_argument(t)

types = ['Float', 'Double']

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
        return ctypes.c_void_p()

lib_handle = ctypes.cdll.LoadLibrary(THNN_LIB_PATH)

# TODO: typechecking
class TorchArgument(object):
    @staticmethod
    def from_param(obj):
        if hasattr(obj, '_cdata'):
            return ctypes.c_void_p(obj._cdata)
        elif obj is None:
            return ctypes.c_void_p(0)
        else:
            raise ValueError('Invalid argument type: {}'.format(type(obj).__name__))

TYPE_CONVERTERS = {
    # TODO: this won't work for CUDA
    'THNNState*': ctypes.c_void_p,
    'THFloatTensor*': TorchArgument,
    'THDoubleTensor*': TorchArgument,
    'THIndexTensor*': TorchArgument,
    'THIntegerTensor*': TorchArgument,
    'THGenerator*': TorchArgument,
    'int': ctypes.c_int,
    'real': ctypes.c_double,
    'double': ctypes.c_double,
    'bool': ctypes.c_bool,
    'long': ctypes.c_long,
    'THIndex_t': ctypes.c_long,
}

class Backends(object):
    pass
_backends = Backends()

_type2tensor = {
    'Float': 'THFloatTensor*',
    'Double': 'THDoubleTensor*',
}

for t in types:
    backend_name = 'THNN{}Backend'.format(t)
    backend = THNNBackendBase()
    setattr(_backends, backend_name, backend)
    for function in generic_functions:
        full_fn_name = 'THNN_{}{}'.format(t, function.name)
        ctypes_fn = getattr(lib_handle, full_fn_name)
        ctypes_fn.restype = None  # All functions return void
        arguments = map(lambda a: a if a != 'THTensor*' else _type2tensor[t], function.arguments)
        ctypes_fn.argtypes = [TYPE_CONVERTERS[t] for t in arguments]
        backend.register_method(function.name, ctypes_fn)

type2backend = {
  'torch.DoubleTensor': _backends.THNNDoubleBackend,
  'torch.FloatTensor': _backends.THNNFloatBackend,
}
