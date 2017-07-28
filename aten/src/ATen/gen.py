from optparse import OptionParser
import yaml

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
parser.add_option('-o', '--output-dependencies',
                  help='only output a list of dependencies', action='store')
parser.add_option('-n', '--no-cuda', action='store_true')

options, files = parser.parse_args()
if options.output_dependencies is not None:
    output_dependencies_file = open(options.output_dependencies, 'w')

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
TENSOR_SPARSE_CPP = CodeTemplate.from_file(
    TEMPLATE_PATH + "/TensorSparse.cpp")
TENSOR_DENSE_CPP = CodeTemplate.from_file(
    TEMPLATE_PATH + "/TensorDense.cpp")

TENSOR_DERIVED_H = CodeTemplate.from_file(TEMPLATE_PATH + "/TensorDerived.h")
TENSOR_H = CodeTemplate.from_file(TEMPLATE_PATH + "/Tensor.h")
TENSOR_METHODS_H = CodeTemplate.from_file(TEMPLATE_PATH + "/TensorMethods.h")

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

densities = ['Dense', 'Sparse']

scalar_types = [
    ('Byte', 'uint8_t', 'Long', 'unsigned char'),
    ('Char', 'int8_t', 'Long', 'char'),
    ('Double', 'double', 'Double', 'double'),
    ('Float', 'float', 'Double', 'float'),
    ('Int', 'int', 'Long', 'int'),
    ('Long', 'int64_t', 'Long', 'long'),
    ('Short', 'int16_t', 'Long', 'short'),
    ('Half', 'Half', 'Double', 'THHalf'),
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
    filename = "ATen/" + filename
    if options.output_dependencies is not None:
        output_dependencies_file.write(filename + ";")
        return
    with open(filename, "w") as f:
        f.write(s)


def format_yaml(data):
    if options.output_dependencies:
        # yaml formatting is slow so don't do it if we will ditch it.
        return ""
    noalias_dumper = yaml.dumper.SafeDumper
    noalias_dumper.ignore_aliases = lambda self, data: True
    return yaml.dump(data, default_flow_style=False, Dumper=noalias_dumper)


def generate_storage_type_and_tensor(backend, density, scalar_type, declarations):
    scalar_name, c_type, accreal, th_scalar_type = scalar_type
    env = {}
    density_tag = 'Sparse' if density == 'Sparse' else ''
    th_density_tag = 'S' if density == 'Sparse' else ''
    env['Density'] = density
    env['ScalarName'] = scalar_name
    env['ScalarType'] = c_type
    env['THScalarType'] = th_scalar_type
    env['AccScalarName'] = accreal
    env['Storage'] = "{}{}Storage".format(backend, scalar_name)
    env['Type'] = "{}{}{}Type".format(density_tag, backend, scalar_name)
    env['Tensor'] = "{}{}{}Tensor".format(density_tag, backend, scalar_name)
    env['SparseTensor'] = "Sparse{}{}Tensor".format(backend, scalar_name)
    env['Backend'] = density_tag + backend

    # used for generating switch logic for external functions
    tag = density_tag + backend + scalar_name
    env['TypeID'] = 'TypeID::' + tag
    top_env['type_ids'].append(tag + ',')

    if backend == 'CUDA':
        env['th_headers'] = ['#include <THC/THC.h>',
                             '#include <THCUNN/THCUNN.h>',
                             '#undef THNN_',
                             '#undef THCIndexTensor_']
        # if density == 'Sparse':
        env['th_headers'] += ['#include <THCS/THCS.h>',
                              '#undef THCIndexTensor_']
        sname = '' if scalar_name == "Float" else scalar_name
        env['THType'] = 'Cuda{}'.format(sname)
        env['THStorage'] = 'THCuda{}Storage'.format(sname)
        if density == 'Dense':
            env['THTensor'] = 'THCuda{}Tensor'.format(sname)
        else:
            env['THTensor'] = 'THCS{}Tensor'.format(scalar_name)
        env['THIndexTensor'] = 'THCudaLongTensor'
        env['state'] = ['context->thc_state']
        env['isCUDA'] = 'true'
        env['storage_device'] = 'return storage->device;'
        env['Generator'] = 'CUDAGenerator'
    else:
        env['th_headers'] = ['#include <TH/TH.h>',
                             '#include <THNN/THNN.h>',
                             '#undef THNN_']
        # if density == 'Sparse':
        env['th_headers'].append('#include <THS/THS.h>')

        env['THType'] = scalar_name
        env['THStorage'] = "TH{}Storage".format(scalar_name)
        env['THTensor'] = 'TH{}{}Tensor'.format(th_density_tag, scalar_name)
        env['THIndexTensor'] = 'THLongTensor'
        env['state'] = []
        env['isCUDA'] = 'false'
        env['storage_device'] = 'throw std::runtime_error("CPU storage has no device");'
        env['Generator'] = 'CPUGenerator'
    env['AS_REAL'] = env['ScalarType']
    if scalar_name == "Half":
        env['SparseTensor'] = 'Tensor'
        if backend == "CUDA":
            env['to_th_type'] = 'HalfFix<__half,Half>'
            env['to_at_type'] = 'HalfFix<Half,__half>'
            env['AS_REAL'] = 'convert<half,double>'
            env['THScalarType'] = 'half'
        else:
            env['to_th_type'] = 'HalfFix<THHalf,Half>'
            env['to_at_type'] = 'HalfFix<Half,THHalf>'
    elif scalar_name == 'Long':
        env['to_th_type'] = 'long'
        env['to_at_type'] = 'int64_t'
    else:
        env['to_th_type'] = ''
        env['to_at_type'] = ''

    declarations, definitions = function_wrapper.create_derived(
        env, declarations)
    env['type_derived_method_declarations'] = declarations
    env['type_derived_method_definitions'] = definitions

    if density != 'Sparse':
        # there are no special storage types for Sparse, they are composed
        # of Dense tensors
        write(env['Storage'] + ".cpp", STORAGE_DERIVED_CPP.substitute(env))
        write(env['Storage'] + ".h", STORAGE_DERIVED_H.substitute(env))
        env['TensorDenseOrSparse'] = TENSOR_DENSE_CPP.substitute(env)
    else:
        env['TensorDenseOrSparse'] = TENSOR_SPARSE_CPP.substitute(env)

    write(env['Type'] + ".cpp", TYPE_DERIVED_CPP.substitute(env))
    write(env['Type'] + ".h", TYPE_DERIVED_H.substitute(env))

    write(env['Tensor'] + ".cpp", TENSOR_DERIVED_CPP.substitute(env))
    write(env['Tensor'] + ".h", TENSOR_DERIVED_H.substitute(env))

    type_register = (('context->type_registry[static_cast<int>(Backend::{})]' +
                      '[static_cast<int>(ScalarType::{})].reset(new {}(context));')
                     .format(env['Backend'], scalar_name, env['Type']))
    top_env['type_registrations'].append(type_register)
    top_env['type_headers'].append(
        '#include "ATen/{}.h"'.format(env['Type']))

    return env


cwrap_files = [f for f in files if f.endswith('.cwrap')]
nn_files = [f for f in files if f.endswith('.h')]

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
output_declarations = function_wrapper.create_generic(top_env, declarations)
write("Declarations.yaml", format_yaml(output_declarations))

# populated by generate_storage_type_and_tensor
all_types = []

for backend in backends:
    for density in densities:
        for scalar_type in scalar_types:
            if density == 'Sparse' and scalar_type[0] == 'Half':
                # THS does not do half type yet.
                continue
            all_types.append(generate_storage_type_and_tensor(
                backend, density, scalar_type, declarations))

write('Type.h', TYPE_H.substitute(top_env))
write('Type.cpp', TYPE_CPP.substitute(top_env))

write('Tensor.h', TENSOR_H.substitute(top_env))
write('TensorMethods.h', TENSOR_METHODS_H.substitute(top_env))
write('Functions.h', FUNCTIONS_H.substitute(top_env))
write('Dispatch.h', dispatch_macros.create(all_types))
write('Copy.cpp', copy_wrapper.create(all_types))

if options.output_dependencies is not None:
    output_dependencies_file.close()
