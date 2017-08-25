import argparse
import os
import yaml
from tools.shared.module_loader import import_module

CodeTemplate = import_module('code_template', 'torch/lib/ATen/code_template.py').CodeTemplate

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


METHOD_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${method_prefix}${api_name}(${formals}) override;
""")

METHOD_DEFINITION = CodeTemplate("""\
${return_type} VariableType::${method_prefix}${api_name}(${formals}) {
    ${type_definition_body}
}
""")

METHOD_DEFINITION_NYI = CodeTemplate("""\
throw std::runtime_error("${api_name}: NYI");""")

METHOD_DEFINITION_FALLTHROUGH = CodeTemplate("""\
return baseType->${method_prefix}${api_name}(${unpacked_args});""")

UNWRAP_TENSOR = CodeTemplate("""\
auto& ${arg_name}_ = checked_unpack(${arg_name}, "${arg_name}", ${arg_pos});""")

GENERATED_COMMENT = CodeTemplate("""\
generated from tools/autograd/templates/${filename}""")

PY_VARIABLE_METHOD_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)THPVariable_${name}, ${flags}, NULL},""")


template_path = os.path.join(os.path.dirname(__file__), 'templates')

VARIABLE_TYPE_H = CodeTemplate.from_file(template_path + '/VariableType.h')
VARIABLE_TYPE_CPP = CodeTemplate.from_file(template_path + '/VariableType.cpp')
VARIABLE_METHODS_CPP = CodeTemplate.from_file(template_path + '/python_variable_methods.cpp')

FUNCTIONS_H = CodeTemplate.from_file(template_path + '/Functions.h')
FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/Functions.cpp')

derivatives_path = os.path.join(os.path.dirname(__file__), 'derivatives.yaml')

# Functions with these return types delegate completely to the underlying
# base at::Type
FALLTHROUGH_RETURN_TYPES = {'int64_t', 'void*', 'bool', 'IntList'}


def format_return_type(returns):
    if len(returns) == 0:
        return 'void'
    elif len(returns) == 1:
        return returns[0]['type']
    else:
        return_types = [r['type'] for r in returns]
        return 'std::tuple<{}>'.format(','.join(return_types))


def write(dirname, name, template, env):
    env['generated_comment'] = GENERATED_COMMENT.substitute(filename=name)
    path = os.path.join(dirname, name)
    with open(path, 'w') as f:
        f.write(template.substitute(env))


def load_derivatives(path):
    with open(path, 'r') as f:
        # TODO: load and parse derivatives.yaml
        d = yaml.load(f, Loader=Loader)
        return [] if d is None else d


def create_autograd_functions(top_env, declarations):
    """Functions.h and Functions.cpp body

    These contain the auto-generated subclasses of torch::autograd::Function
    for each every differentiable torch function.
    """
    # function_definitions = top_env['autograd_function_definitions']
    # function_declarations = top_env['autograd_function_declarations']

    def process_function(option):
        # TODO: generate autograd::Function classes for each function specified
        # in derivatives.yaml
        pass

    for option in declarations:
        process_function(option)


def create_variable_type(top_env, aten_declarations):
    """VariableType.h and VariableType.cpp body

    This is the at::Type subclass for differentiable tensors. The
    implementation of each function dispatches to the base tensor type to
    compute the output. The grad_fn is attached to differentiable functions.
    """

    type_declarations = top_env['type_derived_method_declarations']
    type_definitions = top_env['type_derived_method_definitions']

    def unpack_args(option):
        body = []
        unpacked_args = []
        for i, arg in enumerate(option['arguments']):
            if arg['dynamic_type'] == 'Tensor':
                env = {'arg_name': arg['name'], 'arg_pos': i}
                body.append(UNWRAP_TENSOR.substitute(env))
                unpacked_args.append(arg['name'] + '_')
            else:
                unpacked_args.append(arg['name'])
        option['unpacked_args'] = unpacked_args
        return body

    def process_function(option):
        option['formals'] = [arg['type'] + ' ' + arg['name']
                             for arg in option['arguments']]
        option['args'] = [arg['name'] for arg in option['arguments']]
        option['api_name'] = option['name']
        return_type = format_return_type(option['returns'])
        option['return_type'] = return_type
        option['type_definition_body'] = emit_body(option)

        type_declarations.append(METHOD_DECLARATION.substitute(option))
        type_definitions.append(METHOD_DEFINITION.substitute(option))

    def emit_body(option):
        body = []
        body += unpack_args(option)

        if option['return_type'] in FALLTHROUGH_RETURN_TYPES:
            body.extend(METHOD_DEFINITION_FALLTHROUGH.substitute(option).split('\n'))
            return body

        return METHOD_DEFINITION_NYI.substitute(option)

    for function in aten_declarations:
        process_function(function)


def create_python_bindings(top_env, aten_decls, derivatives):
    """python_variable_methods.cpp

    Generates Python bindings to Variable methods
    """

    # py_methods = top_env['py_methods']
    # py_method_defs = top_env['py_method_defs']

    def process_option(option):
        # TODO: generate Python bindings
        pass

    for option in aten_decls:
        process_option(option)


def gen_variable_type(declarations, out):
    with open(declarations, 'r') as f:
        aten_decls = [option for option in yaml.load(f, Loader=Loader)
                      if option['has_full_argument_list']]

    derivatives = load_derivatives(derivatives_path)

    env = {
        'autograd_function_declarations': [],
        'autograd_function_definitions': [],
        'type_derived_method_declarations': [],
        'type_derived_method_definitions': [],
        'py_methods': [],
        'py_method_defs': [],
    }

    create_autograd_functions(env, derivatives)
    create_variable_type(env, aten_decls)
    create_python_bindings(env, aten_decls, derivatives)

    write(out, 'VariableType.h', VARIABLE_TYPE_H, env)
    write(out, 'VariableType.cpp', VARIABLE_TYPE_CPP, env)
    write(out, 'python_variable_methods.cpp', VARIABLE_METHODS_CPP, env)
    write(out, 'Functions.h', FUNCTIONS_H, env)
    write(out, 'Functions.cpp', FUNCTIONS_CPP, env)


def main():
    parser = argparse.ArgumentParser(
        description='Generate autograd C++ files script')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    args = parser.parse_args()
    gen_variable_type(args.declarations, args.out)


if __name__ == '__main__':
    main()
