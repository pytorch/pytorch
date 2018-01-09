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
import re
import yaml
from collections import defaultdict
from .utils import CodeTemplate, YamlLoader, split_name_params, write


FUNCTION_PROTOTYPE = CodeTemplate("""\
${name}(${typed_args})""")

template_path = os.path.join(os.path.dirname(__file__), 'templates')

PY_VARIABLE_METHODS_CPP = CodeTemplate.from_file(template_path + '/python_variable_methods.cpp')
PY_VARIABLE_DISPATCH_H = CodeTemplate.from_file(template_path + '/python_variable_methods_dispatch.h')
PY_NN_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_nn_functions.cpp')
PY_NN_FUNCTIONS_H = CodeTemplate.from_file(template_path + '/python_nn_functions.h')
PY_NN_DISPATCH_H = CodeTemplate.from_file(template_path + '/python_nn_functions_dispatch.h')

derivatives_path = os.path.join(os.path.dirname(__file__), 'derivatives.yaml')
deprecated_path = os.path.join(os.path.dirname(__file__), 'deprecated.yaml')

VIEW_FUNCTIONS = {
    'alias', 'as_strided', 'expand', 'narrow', 'permute', 'select', 'slice',
    'squeeze', 't', 'transpose', 'unfold', 'unsqueeze', 'view',
}
# These functions require manual Python bindings or are not exposed to Python
SKIP_PYTHON_BINDINGS = [
    'alias', 'contiguous', 'clamp.*', 'is_cuda', 'is_sparse', 'size', 'stride',
    'slice_dim', '.*_backward'
]


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

        # Compute the Python function prototype for argument parsing
        typed_args = []
        positional = True
        for arg in declaration['arguments']:
            if arg.get('kwarg_only', False) and positional:
                typed_args.append('*')
                positional = False
            typename = arg['simple_type']
            if arg.get('is_nullable'):
                typename = '{}?'.format(typename)
            if arg.get('size') is not None:
                typename = '{}[{}]'.format(typename, arg['size'])
            param = typename + ' ' + arg['name']
            default = None
            if arg.get('default') is not None:
                default = arg['default']
                if default == 'nullptr' or default == '{}':
                    default = 'None'
            if arg.get('python_default_init') is not None:
                default = 'None'
            if default is not None:
                param += '=' + str(default)
            typed_args.append(param)

        # Python function prototype.
        # This is the string that we give to FunctionParameter, which is
        # then parsed into the actual structure which we do parsing
        # with.
        declaration['typed_args'] = typed_args
        declaration['prototype'] = FUNCTION_PROTOTYPE.substitute(declaration)

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
        prototype = deprecated['name']
        call_args = split_name_params(deprecated['aten'])[1]
        name, params = split_name_params(prototype)
        signature = get_signature(name, params, call_args)

        for declaration in declarations_by_signature[signature]:
            declaration = copy.deepcopy(declaration)
            declaration['deprecated'] = True
            declaration['call_args'] = call_args
            if declaration['inplace']:
                declaration['prototype'] = prototype.replace(name, name + '_')
            else:
                declaration['prototype'] = prototype

            args_by_name = {arg['name']: arg for arg in declaration['arguments']}
            declaration['arguments'] = []
            for arg in params:
                _, arg_name = arg.split(' ')
                declaration['arguments'].append(args_by_name[arg_name])
            declarations.append(declaration)
    return declarations


def gen_autograd(declarations, out):
    aten_decls = load_aten_declarations(declarations)

    from .load_derivatives import load_derivatives
    autograd_functions = load_derivatives(derivatives_path, aten_decls)

    def should_generate_python_binding(declaration):
        name = declaration['name']
        for pattern in SKIP_PYTHON_BINDINGS:
            if re.match('^' + pattern + '$', name):
                return False

        # we don't currently support functions which are only defined on Type
        # such as zeros(), randn(), etc.
        method_of = declaration['method_of']
        if 'Tensor' not in method_of and 'namespace' not in method_of:
            return False

        return True

    py_variable_methods = defaultdict(list)
    py_nn_functions = defaultdict(list)
    for declaration in aten_decls:
        name = declaration['name']
        if not should_generate_python_binding(declaration):
            continue
        if declaration['mode'] == 'NN':
            py_nn_functions[name].append(declaration)
        else:
            py_variable_methods[name].append(declaration)

    for declaration in load_deprecated_signatures(aten_decls):
        py_variable_methods[declaration['name']].append(declaration)

    env = {
        'py_methods': [],
        'py_method_defs': [],
        'py_method_dispatch': [],
        'py_function_initializers': [],
        'py_nn_functions': [],
        'py_nn_function_defs': [],
        'py_nn_function_dispatch': [],
    }

    from .gen_variable_type import gen_variable_type
    gen_variable_type(out, aten_decls)

    from .gen_autograd_functions import gen_autograd_functions
    gen_autograd_functions(out, autograd_functions)

    from .gen_python_functions import create_python_bindings
    create_python_bindings(
        py_variable_methods,
        env['py_methods'],
        env['py_method_defs'],
        env['py_method_dispatch'],
        is_class=True)

    create_python_bindings(
        py_nn_functions,
        env['py_nn_functions'],
        env['py_nn_function_defs'],
        env['py_nn_function_dispatch'],
        is_class=False)

    write(out, 'python_variable_methods.cpp', PY_VARIABLE_METHODS_CPP, env)
    write(out, 'python_variable_methods_dispatch.h', PY_VARIABLE_DISPATCH_H, env)
    write(out, 'python_nn_functions.cpp', PY_NN_FUNCTIONS_CPP, env)
    write(out, 'python_nn_functions.h', PY_NN_FUNCTIONS_H, env)
    write(out, 'python_nn_functions_dispatch.h', PY_NN_DISPATCH_H, env)


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
