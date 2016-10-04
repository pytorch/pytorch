import os
import sys
from string import Template, ascii_lowercase
from ..cwrap import cwrap
from ..cwrap.plugins import StandaloneExtension, NullableArguments, AutoGPU

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
THIMG_H_PATH = os.path.join(thnn_utils.LIB_PATH, 'THIMG.h')

FUNCTION_TEMPLATE = Template("""\
[[
  name: $name
  return: void
  cname: $cname
  arguments:
""")

COMMON_TRANSFORMS = {
    'THIndex_t': 'long',
    'THInteger_t': 'int',
}
COMMON_CPU_TRANSFORMS = {
    'THIndexTensor*': 'THLongTensor*',
    'THIntegerTensor*': 'THIntTensor*',
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
    'Byte': {
        'THTensor*': 'THByteTensor*',
        'real': 'Byte',
    }
}
for t, transforms in TYPE_TRANSFORMS.items():
    transforms.update(COMMON_TRANSFORMS)
TYPE_TRANSFORMS['Float'].update(COMMON_CPU_TRANSFORMS)
TYPE_TRANSFORMS['Double'].update(COMMON_CPU_TRANSFORMS)


def wrap_function(name, type, arguments):
    cname = 'THIMG_' + type + name
    declaration = ''
    declaration += 'extern "C" void ' + cname + '(' + ', '.join(TYPE_TRANSFORMS[type].get(arg.type, arg.type) for arg in arguments) + ');\n'
    declaration += FUNCTION_TEMPLATE.substitute(name=type + name, cname=cname)
    indent = ' ' * 4
    dict_indent = ' ' * 6
    prefix = indent + '- '
    for arg in arguments:
        if not arg.is_optional:
            declaration += prefix + TYPE_TRANSFORMS[type].get(arg.type, arg.type) + ' ' + arg.name + '\n'
        else:
            t = TYPE_TRANSFORMS[type].get(arg.type, arg.type)
            declaration += prefix + 'type: ' + t        + '\n' + \
                      dict_indent + 'name: ' + arg.name + '\n' + \
                      dict_indent + 'nullable: True' + '\n'
    declaration += ']]\n\n\n'
    return declaration

def generate_wrappers():
    wrap_image()

def wrap_image():
    wrapper = '#include <TH/TH.h>\n\n\n'
    nn_functions = thnn_utils.parse_header(THIMG_H_PATH, 'THIMG_')
    for fn in nn_functions:
        for t in ['Float', 'Double', 'Byte']:
            wrapper += wrap_function(fn.name, t, fn.arguments)
    with open('torch/csrc/image/THIMG.cwrap', 'w') as f:
        f.write(wrapper)
    cwrap('torch/csrc/image/THIMG.cwrap', plugins=[
        StandaloneExtension('torch._image'),
        NullableArguments(),
    ])
