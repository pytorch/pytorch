import os
from optparse import OptionParser
from CodeTemplate import CodeTemplate
import sys
parser = OptionParser()
parser.add_option('-s', '--source-path', help='path to source director for tensorlib',
    action='store', default='.')
parser.add_option('-p', '--print-dependencies',
    help='only output a list of dependencies', action='store_true')
options,args = parser.parse_args()

TEMPLATE_PATH =  options.source_path+"/templates"
GENERATOR_DERIVED = CodeTemplate.from_file(TEMPLATE_PATH+"/GeneratorDerived.h")
STORAGE_DERIVED_CPP = CodeTemplate.from_file(TEMPLATE_PATH+"/StorageDerived.cpp")
STORAGE_DERIVED_H = CodeTemplate.from_file(TEMPLATE_PATH+"/StorageDerived.h")

TYPE_DERIVED_CPP = CodeTemplate.from_file(TEMPLATE_PATH+"/TypeDerived.cpp")
TYPE_DERIVED_H = CodeTemplate.from_file(TEMPLATE_PATH+"/TypeDerived.h")
TYPE_H = CodeTemplate.from_file(TEMPLATE_PATH+"/Type.h")
TYPE_CPP = CodeTemplate.from_file(TEMPLATE_PATH+"/Type.cpp")

TENSOR_DERIVED_CPP = CodeTemplate.from_file(TEMPLATE_PATH+"/TensorDerived.cpp")
TENSOR_DERIVED_H = CodeTemplate.from_file(TEMPLATE_PATH+"/TensorDerived.h")
TENSOR_H = CodeTemplate.from_file(TEMPLATE_PATH+"/Tensor.h")
TENSOR_CPP = CodeTemplate.from_file(TEMPLATE_PATH+"/Tensor.cpp")

generators = {
    'CPUGenerator.h' : {
        'name' : 'CPU',
        'th_generator':'THGenerator * generator;',
        'header': 'TH/TH.h',
    },
    'CUDAGenerator.h' : {
        'name' : 'CUDA',
        'th_generator': '',
        'header': 'THC/THC.h'
    },
}

processors = [ 'CPU', 'CUDA']
scalar_types = [
    ('Byte','uint8_t'),
    ('Char','int8_t'),
    ('Double','double'),
    ('Float','float'),
    ('Int','int'),
    ('Long','int64_t'),
    ('Short','int16_t'),
    ('Half','Half'),
]

# shared environment for non-derived base classes Type.h Tensor.h Storage.h
top_env = {
    'type_registrations' : [],
    'type_headers' : []
}

def write(filename,s):
    filename = "TensorLib/"+filename
    if options.print_dependencies:
        sys.stdout.write(filename+";")
        return
    with open(filename,"w") as f:
        f.write(s)

def generate_storage_type_and_tensor(processor, scalar_type):
    scalar_name, c_type = scalar_type
    env = {}
    env['ScalarName'] = scalar_name
    env['ScalarType'] = c_type
    env['Storage'] = "{}{}Storage".format(processor,scalar_name)
    env['Type'] = "{}{}Type".format(processor,scalar_name)
    env['Tensor'] = "{}{}Tensor".format(processor,scalar_name)
    env['Processor'] = processor

    if processor == 'CUDA':
        env['th_header'] = "THC/THC.h"
        sname = '' if scalar_name == "Float" else scalar_name
        env['THStorage'] = 'THCuda{}Storage'.format(sname)
        env['THTensor'] = 'THCuda{}Tensor'.format(sname)
        env['state'] = ['context->thc_state']
        env['isCUDA'] = 'true'
        env['storage_device'] = 'return storage->device;'
    else:
        env['th_header'] = "TH/TH.h"
        env['THStorage'] = "TH{}Storage".format(scalar_name)
        env['THTensor'] = 'TH{}Tensor'.format(scalar_name)
        env['state'] = []
        env['isCUDA'] = 'false'
        env['storage_device'] = 'throw std::runtime_error("CPU storage has no device");'

    if scalar_name == "Half":
        if processor == "CUDA":
            env['to_th_half'] = 'HalfFix<__half,Half>'
            env['to_tlib_half'] = 'HalfFix<Half,__half>'
        else:
            env['to_th_half'] = 'HalfFix<THHalf,Half>'
            env['to_tlib_half'] = 'HalfFix<Half,THHalf>'
    else:
        env['to_th_half'] = ''
        env['to_tlib_half'] = ''


    write(env['Storage']+".cpp",STORAGE_DERIVED_CPP.substitute(env))
    write(env['Storage']+".h",STORAGE_DERIVED_H.substitute(env))

    write(env['Type']+".cpp",TYPE_DERIVED_CPP.substitute(env))
    write(env['Type']+".h",TYPE_DERIVED_H.substitute(env))

    write(env['Tensor']+".cpp",TENSOR_DERIVED_CPP.substitute(env))
    write(env['Tensor']+".h",TENSOR_DERIVED_H.substitute(env))


    type_register = ('context->type_registry[static_cast<int>(Processor::{})][static_cast<int>(ScalarType::{})].reset(new {}(context));'
        .format(processor,scalar_name,env['Type']))
    top_env['type_registrations'].append(type_register)
    top_env['type_headers'].append('#include "TensorLib/{}.h"'.format(env['Type']))

for fname,env in generators.items():
    write(fname,GENERATOR_DERIVED.substitute(env))

for processor in processors:
    for scalar_type in scalar_types:
        generate_storage_type_and_tensor(processor,scalar_type)

write('Type.h',TYPE_H.substitute(top_env))
write('Type.cpp',TYPE_CPP.substitute(top_env))

write('Tensor.h',TENSOR_H.substitute(top_env))
write('Tensor.cpp',TENSOR_CPP.substitute(top_env))

# what has to be done to add a Operation ...
# 1. add virtual dispatch declaration to Type.h
# 2. add virtual override to TypeDerived.h
# 3. add override definition to TypeDerived.cpp
# 4. add non-virtual declaration to Type.h
# 5. add non-virtual declaration to Type.cpp
