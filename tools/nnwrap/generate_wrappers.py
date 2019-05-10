import os
from string import Template
from ..cwrap import cwrap
from ..cwrap.plugins import NNExtension, NullableArguments, AutoGPU
from ..shared import import_module

from ..shared._utils_internal import get_file_path

THNN_H_PATH = get_file_path('torch', 'include', 'THNN', 'generic', 'THNN.h')
THCUNN_H_PATH = get_file_path('torch', 'include', 'THCUNN', 'generic', 'THCUNN.h')

THNN_UTILS_PATH = get_file_path('torch', '_thnn', 'utils.py')

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


def generate_wrappers(nn_root=None, install_dir=None, template_path=None):
    wrap_nn(os.path.join(nn_root, 'THNN', 'generic', 'THNN.h') if nn_root else None, install_dir, template_path)
    wrap_cunn(os.path.join(nn_root, 'THCUNN', 'generic', 'THCUNN.h') if nn_root else None, install_dir, template_path)


def wrap_nn(thnn_h_path, install_dir, template_path):
    wrapper = '#include <TH/TH.h>\n\n\n'
    nn_functions = thnn_utils.parse_header(thnn_h_path or THNN_H_PATH)
    for fn in nn_functions:
        for t in ['Float', 'Double']:
            wrapper += wrap_function(fn.name, t, fn.arguments)
    install_dir = install_dir or 'torch/csrc/nn'
    try:
        os.makedirs(install_dir)
    except OSError:
        pass
    with open(os.path.join(install_dir, 'THNN.cwrap'), 'w') as f:
        f.write(wrapper)
    cwrap(os.path.join(install_dir, 'THNN.cwrap'),
          plugins=[NNExtension('torch._C._THNN'), NullableArguments()],
          template_path=template_path)


def wrap_cunn(thcunn_h_path, install_dir, template_path):
    wrapper = '#include <TH/TH.h>\n'
    wrapper += '#include <THC/THC.h>\n\n\n'
    cunn_functions = thnn_utils.parse_header(thcunn_h_path or THCUNN_H_PATH)
    for fn in cunn_functions:
        for t in ['CudaHalf', 'Cuda', 'CudaDouble']:
            wrapper += wrap_function(fn.name, t, fn.arguments)
    install_dir = install_dir or 'torch/csrc/nn'
    with open(os.path.join(install_dir, 'THCUNN.cwrap'), 'w') as f:
        f.write(wrapper)
    cwrap(os.path.join(install_dir, 'THCUNN.cwrap'),
          plugins=[NNExtension('torch._C._THCUNN'), NullableArguments(), AutoGPU(has_self=False)],
          template_path=template_path)
