"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m tools.code_analyzer.classify_ops

To generate static analysis results:

ANALYZE_TORCH=1 tools/code_analyzer/build.sh -debug_path=true
"""

import re
import yaml

from collections import defaultdict
from tools.autograd.utils import YamlLoader

NATIVE_FUNCTIONS = 'aten/src/ATen/native/native_functions.yaml'
DECLARATIONS_PATH = 'torch/share/ATen/Declarations.yaml'
DERIVATIVES_FILE = 'tools/autograd/derivatives.yaml'
ANALYSIS_RESULT = 'build_code_analyzer/work/torch_result.yaml'

BACKENDS = [
    'CPU',
    'CUDA',
    'QuantizedCPU',
    'QuantizedCUDA',
    'SparseCPU',
    'SparseCUDA',
    'MkldnnCPU',
    'Vulkan',
]

# Key: opname (with overload)
# Value: schema_string
OPNAME_TO_OPSCHEMA = dict()

# Key: schema_string
OPS = defaultdict(lambda: defaultdict(lambda: ''))

FIELDS = (
    'name',
    'native_functions.yaml',
    'derivatives.yaml',
    'abstract',
    'derivative',
    'autograd',
    'manual',
    'modifies_arguments',
    'has_dispatch',
    'DispatchStub',
    'legacy_dispatch',
    'type_dispatch',
    'natives',
    *BACKENDS,
)
FMT = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t' + \
      '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'


def load_codegen_parsed_results():
    from tools.autograd.gen_autograd import load_aten_declarations
    from tools.autograd.load_derivatives import load_derivatives

    aten_decls = load_aten_declarations(DECLARATIONS_PATH)
    autograd_functions = load_derivatives(DERIVATIVES_FILE, aten_decls)

    for decl in aten_decls:
        schema_string = decl['schema_string']

        opinfo = OPS[schema_string]
        opinfo['schema_string'] = schema_string

        opname = 'aten::' + decl['operator_name'] + ('.' + decl['overload_name'] if decl['overload_name'] else '')
        opinfo['name'] = opname

        if opname in OPNAME_TO_OPSCHEMA:
            raise RuntimeError('duplicate opname {}'.format(opname))
        OPNAME_TO_OPSCHEMA[opname] = schema_string

        if decl['abstract']:
            opinfo['abstract'] = True
        if decl['derivative'] is not None:
            opinfo['derivative'] = True

        name = decl['name']
        inplace = decl['inplace']
        is_out_fn = name.endswith('_out')
        if inplace or is_out_fn:
            opinfo['modifies_arguments'] = True

    for decl in autograd_functions:
        schema_string = decl['declaration']['schema_string']
        OPS[schema_string]['autograd'] = True


def load_derivatives_yaml():
    with open(DERIVATIVES_FILE, 'r') as f:
        derivatives = yaml.load(f, Loader=YamlLoader)

    for item in derivatives:
        schema_string = 'aten::' + item['name']
        OPS[schema_string]['derivatives.yaml'] = True


def load_native_functions_yaml():
    with open(NATIVE_FUNCTIONS, 'r') as f:
        native_functions = yaml.load(f, Loader=YamlLoader)

    for item in native_functions:
        schema_string = 'aten::' + item['func']
        opinfo = OPS[schema_string]
        dispatch = item.get('dispatch', None)
        if dispatch is not None:
            opinfo['has_dispatch'] = True
            for backend in BACKENDS:
                for k, v in dispatch.items():
                    if backend in k:
                        opinfo[backend] = True
        if item.get('manual_kernel_registration', None):
            opinfo['manual'] = True
        opinfo['native_functions.yaml'] = True


def load_analysis_result():
    with open(ANALYSIS_RESULT, 'r') as f:
        analysis_result = yaml.load(f, Loader=YamlLoader)

    for item in analysis_result:
        opname = item['name']
        schema_string = OPNAME_TO_OPSCHEMA.get(opname, opname)

        opinfo = OPS[schema_string]
        assert opinfo['name'] == opname or not opinfo['name']
        opinfo['name'] = opname

        for dep in item.get('depends', []):
            depname = dep.get('name')
            if depname.startswith('at::native::DispatchStub'):
                m = re.match(r'^at::native::DispatchStub.*, (.*)>::choose_cpu_impl', depname)
                if not m:
                    raise RuntimeError('unrecognized dispatch stub {}'.format(depname))
                deps = opinfo.get('DispatchStub', set())
                deps.add(m.groups()[0])
                opinfo['DispatchStub'] = deps
            elif depname.startswith('at::native::legacy'):
                m = re.match(r'^at::native::(.*)\(.*\)$', depname)
                if not m:
                    raise RuntimeError('unrecognized legacy function {}'.format(depname))
                deps = opinfo.get('legacy_dispatch', set())
                deps.add(m.groups()[0])
                opinfo['legacy_dispatch'] = deps
            elif depname.startswith('at::native::'):
                m = re.match(r'^at::(native::.*)\(.*\)$', depname)
                if not m:
                    continue
                deps = opinfo.get('natives', set())
                deps.add(m.groups()[0])
                opinfo['natives'] = deps
            elif depname.startswith('at::') and 'Type' in depname:
                m = re.match(r'^at::([^:]*Type[^:]*::.*)\(.*\)$', depname)
                if not m:
                    continue
                deps = opinfo.get('type_dispatch', set())
                deps.add(m.groups()[0])
                opinfo['type_dispatch'] = deps


def dump_ops_table():
    print(FMT.format(*FIELDS))
    for op, opinfo in OPS.items():
        values = [str(opinfo[f]) for f in FIELDS]
        print(FMT.format(*values))


if __name__ == '__main__':
    load_codegen_parsed_results()
    load_derivatives_yaml()
    load_native_functions_yaml()
    load_analysis_result()
    dump_ops_table()
