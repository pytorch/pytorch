# gen_autograd.py generates C++ autograd functions and Python bindings.
#
# It delegates to the following scripts:
#
#  gen_autograd_functions.py: generates subclasses of torch::autograd::Functions
#  gen_variable_type.py: generates VariableType.h which contains all tensor methods
#  gen_python_functions.py: generates Python bindings to THPVariable
#

import argparse
import copy
import os
import yaml
from collections import defaultdict
from .utils import YamlLoader, split_name_params

template_path = os.path.join(os.path.dirname(__file__), 'templates')
derivatives_path = os.path.join(os.path.dirname(__file__), 'derivatives.yaml')
deprecated_path = os.path.join(os.path.dirname(__file__), 'deprecated.yaml')

VIEW_FUNCTIONS = {
    'alias', 'as_strided', 'expand', 'narrow', 'permute', 'select', 'slice',
    'squeeze', 't', 'transpose', 'unfold', 'unsqueeze', 'view',
}


def format_return_type(returns):
    if len(returns) == 0:
        return 'void'
    elif len(returns) == 1:
        return returns[0]['type']
    else:
        return_types = [r['type'] for r in returns]
        return 'std::tuple<{}>'.format(','.join(return_types))


def load_aten_declarations(path):
    with open(path, 'r') as f:
        declarations = yaml.load(f, Loader=YamlLoader)

    # enrich declarations with additional information
    for declaration in declarations:
        for arg in declaration['arguments']:
            simple_type = arg['type']
            simple_type = simple_type.replace(' &', '').replace('const ', '')
            simple_type = simple_type.replace('Generator *', 'Generator')
            arg['simple_type'] = simple_type
        declaration['formals'] = [arg['type'] + ' ' + arg['name']
                                  for arg in declaration['arguments']]
        declaration['args'] = [arg['name'] for arg in declaration['arguments']]
        declaration['api_name'] = declaration['name']
        declaration['return_type'] = format_return_type(declaration['returns'])

        declaration['base_name'] = declaration['name']
    return declarations


def load_deprecated_signatures(aten_decls):
    def group_declarations_by_signature():
        d = defaultdict(list)
        for declaration in aten_decls:
            name = declaration['name']
            base_name = name[:-1] if declaration['inplace'] else name
            simple_types = [arg['simple_type'] for arg in declaration['arguments']]
            signature = '{}({})'.format(base_name, ', '.join(simple_types))
            d[signature].append(declaration)
        return d

    with open(deprecated_path, 'r') as f:
        deprecated_defs = yaml.load(f, Loader=YamlLoader)
    declarations = []
    declarations_by_signature = group_declarations_by_signature()

    def get_signature(name, params, call_args):
        # create a mapping of parameter name to parameter type
        types = dict([param.split(' ')[::-1] for param in params])
        # if the name in the call is not in the parameter list, assume it's
        # a literal Scalar
        rearranged_types = [types.get(arg, 'Scalar') for arg in call_args]
        return '{}({})'.format(name, ', '.join(rearranged_types))

    for deprecated in deprecated_defs:
        python_signature = deprecated['name']
        call_args = split_name_params(deprecated['aten'])[1]
        name, params = split_name_params(python_signature)
        signature = get_signature(name, params, call_args)

        for declaration in declarations_by_signature[signature]:
            declaration = copy.deepcopy(declaration)
            declaration['deprecated'] = True
            declaration['call_args'] = call_args
            if declaration['inplace']:
                declaration['python_signature'] = python_signature.replace(name, name + '_')
            else:
                declaration['python_signature'] = python_signature

            args_by_name = {arg['name']: arg for arg in declaration['arguments']}
            declaration['arguments'] = []
            for arg in params:
                _, arg_name = arg.split(' ')
                declaration['arguments'].append(args_by_name[arg_name])
            declarations.append(declaration)
    return declarations


def gen_autograd(aten_path, out):
    aten_decls = load_aten_declarations(aten_path)

    # Parse and load derivatives.yaml
    from .load_derivatives import load_derivatives
    autograd_functions = load_derivatives(derivatives_path, aten_decls)

    # Generate VariableType.h/cpp
    from .gen_variable_type import gen_variable_type
    gen_variable_type(out, aten_decls)

    # Generate Functions.h/cpp
    from .gen_autograd_functions import gen_autograd_functions
    gen_autograd_functions(out, autograd_functions)

    # Load deprecated signatures
    deprecated = load_deprecated_signatures(aten_decls)

    # Genereate Python bindings
    from . import gen_python_functions
    gen_python_functions.gen_py_variable_methods(out, aten_decls + deprecated)
    gen_python_functions.gen_py_torch_functions(out, aten_decls + deprecated)
    gen_python_functions.gen_py_nn_functions(out, aten_decls)


def main():
    parser = argparse.ArgumentParser(
        description='Generate autograd C++ files script')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    args = parser.parse_args()
    gen_autograd(args.declarations, args.out)


if __name__ == '__main__':
    main()
