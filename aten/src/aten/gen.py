import os
from optparse import OptionParser
from CodeTemplate import CodeTemplate
import sys
parser = OptionParser()
parser.add_option('-s', '--source-path', help='path to source director for tensorlib',
    action='store', default='.')
options,args = parser.parse_args()

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

TEMPLATE_PATH =  options.source_path+"/templates"
GENERATOR_DERIVED = CodeTemplate.from_file(TEMPLATE_PATH+"/GeneratorDerived.h")

def write(f,s):
    with open(fname,"w") as f:
        f.write(s)

for fname,env in generators.items():
    write(fname,GENERATOR_DERIVED.substitute(env))
