import os
import sys
from string import Template, ascii_lowercase
from ..cwrap import cwrap
from ..cwrap.plugins import StandaloneExtension, GenericNN, NullableArguments, AutoGPU

BASE_PATH = os.path.realpath(os.path.join(__file__, '..', '..', '..'))
WRAPPER_PATH = os.path.join(BASE_PATH, 'torch', 'csrc', 'nn')
THNN_UTILS_PATH = os.path.join(BASE_PATH, 'torch', '_thnn', 'utils.py')


def import_module(name, path):
    if sys.version_info >= (3, 5):
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    elif sys.version_info >= (3, 0):
        from importlib.machinery import SourceFileLoader
        return SourceFileLoader(name, path).load_module()
    else:
        import imp
        return imp.load_source(name, path)

thnn_utils = import_module('torch._thnn.utils', THNN_UTILS_PATH)

FUNCTION_TEMPLATE = Template("""\
[[
  name: $name
  return: void
  cname: $cname
  arguments:
""")

COMMON_TRANSFORMS = {
    'THIndex_t': 'long',
    'THCIndex_t': 'long',
    'THInteger_t': 'int',
}
COMMON_CPU_TRANSFORMS = {
    'THNNState*': 'void*',
    'THIndexTensor*': 'THLongTensor*',
    'THIntegerTensor*': 'THIntTensor*',
}
COMMON_GPU_TRANSFORMS = {
    'THCState*': 'void*',
    'THCIndexTensor*': 'THCudaLongTensor*',
}

TYPE_TRANSFORMS = {
    'Float': {
        'THTensor*': 'THFloatTensor*',
        'real': 'float',
    },
    'Double': {
        'THTensor*': 'THDoubleTensor*',
        'real': 'double',
    },
    'CudaHalf': {
        'THCTensor*': 'THCudaHalfTensor*',
        'real': 'half',
    },
    'Cuda': {
        'THCTensor*': 'THCudaTensor*',
        'real': 'float',
    },
    'CudaDouble': {
        'THCTensor*': 'THCudaDoubleTensor*',
        'real': 'double',
    },
}
for t, transforms in TYPE_TRANSFORMS.items():
    transforms.update(COMMON_TRANSFORMS)

for t in ['Float', 'Double']:
    TYPE_TRANSFORMS[t].update(COMMON_CPU_TRANSFORMS)
for t in ['CudaHalf', 'Cuda', 'CudaDouble']:
    TYPE_TRANSFORMS[t].update(COMMON_GPU_TRANSFORMS)


def wrap_function(name, type, arguments):
    cname = 'THNN_' + type + name
    declaration = ''
    declaration += 'extern "C" void ' + cname + \
        '(' + ', '.join(TYPE_TRANSFORMS[type].get(arg.type, arg.type) for arg in arguments) + ');\n'
    declaration += FUNCTION_TEMPLATE.substitute(name=type + name, cname=cname)
    indent = ' ' * 4
    dict_indent = ' ' * 6
    prefix = indent + '- '
    for arg in arguments:
        if not arg.is_optional:
            declaration += prefix + TYPE_TRANSFORMS[type].get(arg.type, arg.type) + ' ' + arg.name + '\n'
        else:
            t = TYPE_TRANSFORMS[type].get(arg.type, arg.type)
            declaration += prefix + 'type: ' + t + '\n' + \
                dict_indent + 'name: ' + arg.name + '\n' + \
                dict_indent + 'nullable: True' + '\n'
    declaration += ']]\n\n\n'
    return declaration


def generate_wrappers():
    wrap_nn()
    wrap_cunn()
    wrap_generic()


def wrap_nn():
    wrapper = '#include <TH/TH.h>\n\n\n'
    nn_functions = thnn_utils.parse_header(thnn_utils.THNN_H_PATH)
    for fn in nn_functions:
        for t in ['Float', 'Double']:
            wrapper += wrap_function(fn.name, t, fn.arguments)
    with open('torch/csrc/nn/THNN.cwrap', 'w') as f:
        f.write(wrapper)
    cwrap('torch/csrc/nn/THNN.cwrap', plugins=[
        StandaloneExtension('torch._thnn._THNN'),
        NullableArguments(),
    ])


def wrap_cunn():
    wrapper = '#include <TH/TH.h>\n'
    wrapper += '#include <THC/THC.h>\n\n\n'
    cunn_functions = thnn_utils.parse_header(thnn_utils.THCUNN_H_PATH)
    for fn in cunn_functions:
        for t in ['CudaHalf', 'Cuda', 'CudaDouble']:
            wrapper += wrap_function(fn.name, t, fn.arguments)
    with open('torch/csrc/nn/THCUNN.cwrap', 'w') as f:
        f.write(wrapper)
    cwrap('torch/csrc/nn/THCUNN.cwrap', plugins=[
        StandaloneExtension('torch._thnn._THCUNN'),
        NullableArguments(),
        AutoGPU(has_self=False),
    ])

GENERIC_FUNCTION_TEMPLATE = Template("""\
[[
  name: $name
  return: void
  options:
""")


def wrap_generic_function(name, backends):
    declaration = ''
    declaration += GENERIC_FUNCTION_TEMPLATE.substitute(name=name)
    for backend in backends:
        declaration += '    - cname: ' + name + '\n'
        declaration += '      backend: ' + backend['name'] + '\n'
        declaration += '      arguments:\n'
        for arg in backend['arguments']:
            declaration += '       - arg: ' + arg.type + ' ' + arg.name + '\n'
            if arg.is_optional:
                declaration += '         optional: True\n'
    declaration += ']]\n\n\n'
    return declaration


def wrap_generic():
    from collections import OrderedDict
    defs = OrderedDict()

    def should_wrap_function(name):
        if name.startswith('LookupTable'):
            return False
        return (name.endswith('updateOutput') or
                name.endswith('updateGradInput') or
                name.endswith('accGradParameters') or
                name.endswith('backward'))

    def add_functions(name, functions):
        for fn in functions:
            if not should_wrap_function(fn.name):
                continue
            if fn.name not in defs:
                defs[fn.name] = []
            defs[fn.name] += [{
                'name': name,
                'arguments': fn.arguments[1:],
            }]

    add_functions('nn', thnn_utils.parse_header(thnn_utils.THNN_H_PATH))
    add_functions('cunn', thnn_utils.parse_header(thnn_utils.THCUNN_H_PATH))

    wrapper = ''
    for name, backends in defs.items():
        wrapper += wrap_generic_function(name, backends)
    with open('torch/csrc/nn/THNN_generic.cwrap', 'w') as f:
        f.write(wrapper)

    cwrap('torch/csrc/nn/THNN_generic.cwrap', plugins=[
        GenericNN(header=True),
    ], default_plugins=False, destination='torch/csrc/nn/THNN_generic.h')

    cwrap('torch/csrc/nn/THNN_generic.cwrap', plugins=[
        GenericNN(),
    ], default_plugins=False)
