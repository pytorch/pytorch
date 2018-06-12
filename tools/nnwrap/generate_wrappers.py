import os
import sys
from string import Template, ascii_lowercase
from ..cwrap import cwrap
from ..cwrap.plugins import NNExtension, NullableArguments, AutoGPU
from ..shared import import_module

BASE_PATH = os.path.realpath(os.path.join(__file__, '..', '..', '..'))
WRAPPER_PATH = os.path.join(BASE_PATH, 'torch', 'csrc', 'nn')
THNN_UTILS_PATH = os.path.join(BASE_PATH, 'torch', '_thnn', 'utils.py')


thnn_utils = import_module('torch._thnn.utils', THNN_UTILS_PATH)

FUNCTION_TEMPLATE = Template("""\
[[
  name: $name
  return: void
  cname: $cname
  arguments:
""")

COMMON_TRANSFORMS = {
    'THIndex_t': 'int64_t',
    'THCIndex_t': 'int64_t',
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
        'accreal': 'double',
    },
    'Double': {
        'THTensor*': 'THDoubleTensor*',
        'real': 'double',
        'accreal': 'double',
    },
    'CudaHalf': {
        'THCTensor*': 'THCudaHalfTensor*',
        'real': 'half',
        'accreal': 'float',
    },
    'Cuda': {
        'THCTensor*': 'THCudaTensor*',
        'real': 'float',
        'accreal': 'float',
    },
    'CudaDouble': {
        'THCTensor*': 'THCudaDoubleTensor*',
        'real': 'double',
        'accreal': 'double',
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
    declaration += 'TH_API void ' + cname + \
        '(' + ', '.join(TYPE_TRANSFORMS[type].get(arg.type, arg.type)
                        for arg in arguments) + ');\n'
    declaration += FUNCTION_TEMPLATE.substitute(name=type + name, cname=cname)
    indent = ' ' * 4
    dict_indent = ' ' * 6
    prefix = indent + '- '
    for arg in arguments:
        if not arg.is_optional:
            declaration += prefix + \
                TYPE_TRANSFORMS[type].get(
                    arg.type, arg.type) + ' ' + arg.name + '\n'
        else:
            t = TYPE_TRANSFORMS[type].get(arg.type, arg.type)
            declaration += prefix + 'type: ' + t + '\n' + \
                dict_indent + 'name: ' + arg.name + '\n' + \
                dict_indent + 'nullable: True' + '\n'
    declaration += ']]\n\n\n'
    return declaration


def generate_wrappers(nn_root=None):
    wrap_nn(os.path.join(nn_root, 'THNN', 'generic', 'THNN.h') if nn_root else None)
    wrap_cunn(os.path.join(nn_root, 'THCUNN', 'generic', 'THCUNN.h') if nn_root else None)


def wrap_nn(thnn_h_path):
    wrapper = '#include <TH/TH.h>\n\n\n'
    nn_functions = thnn_utils.parse_header(thnn_h_path or thnn_utils.THNN_H_PATH)
    for fn in nn_functions:
        for t in ['Float', 'Double']:
            wrapper += wrap_function(fn.name, t, fn.arguments)
    with open('torch/csrc/nn/THNN.cwrap', 'w') as f:
        f.write(wrapper)
    cwrap('torch/csrc/nn/THNN.cwrap', plugins=[
        NNExtension('torch._C._THNN'),
        NullableArguments(),
    ])


def wrap_cunn(thcunn_h_path=None):
    wrapper = '#include <TH/TH.h>\n'
    wrapper += '#include <THC/THC.h>\n\n\n'
    cunn_functions = thnn_utils.parse_header(thcunn_h_path or thnn_utils.THCUNN_H_PATH)
    for fn in cunn_functions:
        for t in ['CudaHalf', 'Cuda', 'CudaDouble']:
            wrapper += wrap_function(fn.name, t, fn.arguments)
    with open('torch/csrc/nn/THCUNN.cwrap', 'w') as f:
        f.write(wrapper)
    cwrap('torch/csrc/nn/THCUNN.cwrap', plugins=[
        NNExtension('torch._C._THCUNN'),
        NullableArguments(),
        AutoGPU(has_self=False),
    ])
