import os
import sys
import yaml
from optparse import OptionParser

import cwrap_parser
import nn_parse
import preprocess_declarations
import function_wrapper
import dispatch_macros
import copy_wrapper
from code_template import CodeTemplate


parser = OptionParser()
parser.add_option('-s', '--source-path', help='path to source director for tensorlib',
                  action='store', default='.')
parser.add_option('-p', '--print-dependencies',
                  help='only output a list of dependencies', action='store_true')
parser.add_option('-n', '--no-cuda', action='store_true')

options, files = parser.parse_args()

TEMPLATE_PATH = options.source_path + "/templates"
GENERATOR_DERIVED = CodeTemplate.from_file(
    TEMPLATE_PATH + "/GeneratorDerived.h")
STORAGE_DERIVED_CPP = CodeTemplate.from_file(
    TEMPLATE_PATH + "/StorageDerived.cpp")
STORAGE_DERIVED_H = CodeTemplate.from_file(TEMPLATE_PATH + "/StorageDerived.h")

TYPE_DERIVED_CPP = CodeTemplate.from_file(TEMPLATE_PATH + "/TypeDerived.cpp")
TYPE_DERIVED_H = CodeTemplate.from_file(TEMPLATE_PATH + "/TypeDerived.h")
TYPE_H = CodeTemplate.from_file(TEMPLATE_PATH + "/Type.h")
TYPE_CPP = CodeTemplate.from_file(TEMPLATE_PATH + "/Type.cpp")

TENSOR_DERIVED_CPP = CodeTemplate.from_file(
    TEMPLATE_PATH + "/TensorDerived.cpp")
TENSOR_DERIVED_H = CodeTemplate.from_file(TEMPLATE_PATH + "/TensorDerived.h")
TENSOR_H = CodeTemplate.from_file(TEMPLATE_PATH + "/Tensor.h")

FUNCTIONS_H = CodeTemplate.from_file(TEMPLATE_PATH + "/Functions.h")

generators = {
    'CPUGenerator.h': {
        'name': 'CPU',
        'th_generator': 'THGenerator * generator;',
        'header': 'TH/TH.h',
    },
    'CUDAGenerator.h': {
        'name': 'CUDA',
        'th_generator': '',
        'header': 'THC/THC.h'
    },
}

backends = ['CPU']
if not options.no_cuda:
    backends.append('CUDA')

scalar_types = [
    ('Byte', 'uint8_t', 'Long'),
    ('Char', 'int8_t', 'Long'),
    ('Double', 'double', 'Double'),
    ('Float', 'float', 'Double'),
    ('Int', 'int', 'Long'),
    ('Long', 'int64_t', 'Long'),
    ('Short', 'int16_t', 'Long'),
    ('Half', 'Half', 'Double'),
]

# shared environment for non-derived base classes Type.h Tensor.h Storage.h
top_env = {
    'type_registrations': [],
    'type_headers': [],
    'type_method_declarations': [],
    'type_method_definitions': [],
    'tensor_method_declarations': [],
    'tensor_method_definitions': [],
    'function_declarations': [],
    'function_definitions': [],
    'type_ids': [],
}


def write(filename, s):
    filename = "TensorLib/" + filename
    if options.print_dependencies:
        sys.stderr.write(filename + ";")
        return
    with open(filename, "w") as f:
        f.write(s)

def generate_storage_type_and_tensor(backend, scalar_type, declarations):
    scalar_name, c_type, accreal = scalar_type
    env = {}
    env['ScalarName'] = scalar_name
    env['ScalarType'] = c_type
    env['AccScalarName'] = accreal
    env['Storage'] = "{}{}Storage".format(backend, scalar_name)
    env['Type'] = "{}{}Type".format(backend, scalar_name)
    env['Tensor'] = "{}{}Tensor".format(backend, scalar_name)
    env['Backend'] = backend

    # used for generating switch logic for external functions
    tag = backend+scalar_name
    env['TypeID'] = 'TypeID::'+tag
    top_env['type_ids'].append(tag + ',')

    if backend == 'CUDA':
        env['th_headers'] = ['#include <THC/THC.h>', '#include <THCUNN/THCUNN.h>\n#undef THNN_']
        sname = '' if scalar_name == "Float" else scalar_name
        env['THType'] = 'Cuda{}'.format(sname)
        env['THStorage'] = 'THCuda{}Storage'.format(sname)
        env['THTensor'] = 'THCuda{}Tensor'.format(sname)
        env['THIndexTensor'] = 'THCudaLongTensor'
        env['state'] = ['context->thc_state']
        env['isCUDA'] = 'true'
        env['storage_device'] = 'return storage->device;'
    else:
        env['th_headers'] = ['#include <TH/TH.h>', '#include <THNN/THNN.h>\n#undef THNN_']
        env['THType'] = scalar_name
        env['THStorage'] = "TH{}Storage".format(scalar_name)
        env['THTensor'] = 'TH{}Tensor'.format(scalar_name)
        env['THIndexTensor'] = 'THLongTensor'
        env['state'] = []
        env['isCUDA'] = 'false'
        env['storage_device'] = 'throw std::runtime_error("CPU storage has no device");'
    env['AS_REAL'] = env['ScalarType']
    if scalar_name == "Half":
        if backend == "CUDA":
            env['to_th_half'] = 'HalfFix<__half,Half>'
            env['to_tlib_half'] = 'HalfFix<Half,__half>'
            env['AS_REAL'] = 'convert<half,double>'
        else:
            env['to_th_half'] = 'HalfFix<THHalf,Half>'
            env['to_tlib_half'] = 'HalfFix<Half,THHalf>'
    else:
        env['to_th_half'] = ''
        env['to_tlib_half'] = ''

    declarations, definitions = function_wrapper.create_derived(
        env, declarations)
    env['type_derived_method_declarations'] = declarations
    env['type_derived_method_definitions'] = definitions

    write(env['Storage'] + ".cpp", STORAGE_DERIVED_CPP.substitute(env))
    write(env['Storage'] + ".h", STORAGE_DERIVED_H.substitute(env))

    write(env['Type'] + ".cpp", TYPE_DERIVED_CPP.substitute(env))
    write(env['Type'] + ".h", TYPE_DERIVED_H.substitute(env))

    write(env['Tensor'] + ".cpp", TENSOR_DERIVED_CPP.substitute(env))
    write(env['Tensor'] + ".h", TENSOR_DERIVED_H.substitute(env))

    type_register = ('context->type_registry[static_cast<int>(Backend::{})][static_cast<int>(ScalarType::{})].reset(new {}(context));'
                     .format(backend, scalar_name, env['Type']))
    top_env['type_registrations'].append(type_register)
    top_env['type_headers'].append(
        '#include "TensorLib/{}.h"'.format(env['Type']))
    return env

cwrap_files = [f for f in files if f.endswith('.cwrap') ]
nn_files = [f for f in files if f.endswith('.h') ]

declarations = [d
                for file in cwrap_files
                for d in cwrap_parser.parse(file)]
declarations += nn_parse.run(nn_files)
declarations = preprocess_declarations.run(declarations)
# print(yaml.dump(declarations))

for fname, env in generators.items():
    write(fname, GENERATOR_DERIVED.substitute(env))


# note: this will fill in top_env['type/tensor_method_declarations/definitions']
# and modify the declarations to include any information that will all_backends
# be used by function_wrapper.create_derived
function_wrapper.create_generic(top_env, declarations)

# populated by generate_storage_type_and_tensor
all_types = []

for backend in backends:
    for scalar_type in scalar_types:
        all_types.append(generate_storage_type_and_tensor(
            backend, scalar_type, declarations))

write('Type.h', TYPE_H.substitute(top_env))
write('Type.cpp', TYPE_CPP.substitute(top_env))

write('Tensor.h', TENSOR_H.substitute(top_env))
write('Functions.h', FUNCTIONS_H.substitute(top_env))
write('Dispatch.h', dispatch_macros.create(all_types))
write('Copy.cpp', copy_wrapper.create(all_types))
