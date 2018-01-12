# Generates Python bindings for ATen functions
#
# The bindings are generated as methods on python_variable or functions on the
# torch._C._nn object.
#
from collections import defaultdict
import re
from .nested_dict import nested_dict
from tools.shared.module_loader import import_module
from .gen_autograd import template_path
from .utils import write

CodeTemplate = import_module('code_template', 'aten/src/ATen/code_template.py').CodeTemplate

# These functions require manual Python bindings or are not exposed to Python
SKIP_PYTHON_BINDINGS = [
    'alias', 'contiguous', 'clamp.*', 'is_cuda', 'is_sparse', 'size', 'stride',
    '.*_backward'
]

PY_VARIABLE_METHODS_CPP = CodeTemplate.from_file(template_path + '/python_variable_methods.cpp')
PY_VARIABLE_DISPATCH_H = CodeTemplate.from_file(template_path + '/python_variable_methods_dispatch.h')
PY_TORCH_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_torch_functions.cpp')
PY_TORCH_DISPATCH_H = CodeTemplate.from_file(template_path + '/python_torch_functions_dispatch.h')
PY_NN_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_nn_functions.cpp')
PY_NN_FUNCTIONS_H = CodeTemplate.from_file(template_path + '/python_nn_functions.h')
PY_NN_DISPATCH_H = CodeTemplate.from_file(template_path + '/python_nn_functions_dispatch.h')

PY_VARIABLE_METHOD_VARARGS = CodeTemplate("""\
static PyObject * ${pycname}(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    ${prototypes}
  });
  ${unpack_self}
  PyObject* parsed_args[${max_args}];
  auto r = parser.parse(args, kwargs, parsed_args);
  ${dispatch}
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
""")

PY_VARIABLE_METHOD_NOARGS = CodeTemplate("""\
static PyObject * ${pycname}(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  ${unpack_self}
  return wrap(${dispatch_name}(${actuals}));
  END_HANDLE_TH_ERRORS
}
""")

PY_VARIABLE_CASE = CodeTemplate("""\
${cond} (r.idx == ${i}) {
  ${call_dispatch}
""")

PY_VARIABLE_OUT = CodeTemplate("""\
if (r.isNone(${out_idx})) {
  ${call_dispatch}
} else {
  ${call_dispatch_out}
}
""")

PY_VARIABLE_CALL_DISPATCH = CodeTemplate("""\
return wrap(${dispatch_name}(${actuals}));""")

PY_VARIABLE_DISPATCH = CodeTemplate("""\
inline ${return_type} ${dispatch_name}(${formal_args}) {
  ${AutoNoGIL}
  ${AutoGPU}
  return ${dispatch_call}(${dispatch_args});
}
""")

PY_VARIABLE_METHOD_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)${pycname}, ${flags}, NULL},""")

UNPACK_SELF = "auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;"

FUNCTION_PROTOTYPE = CodeTemplate("""\
${name}(${typed_args})""")

# XXX: if you got here because of an assertion failure, it doesn't mean
# it's enough to just extend the list here. Before you do this, make sure
# to add an appropriate wrap() overload in torch/csrc/autograd/utils/wrap_outputs.h.
SUPPORTED_RETURN_TYPES = {
    'Tensor', 'std::tuple<Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor,Tensor>',
    'std::vector<Tensor>',
    'Scalar', 'bool', 'int64_t', 'void*'
}


def should_generate_python_binding(declaration):
    name = declaration['name']
    for pattern in SKIP_PYTHON_BINDINGS:
        if re.match('^' + pattern + '$', name):
            return False

    # TODO: fix handling of SparseTensor. We don't want to generate Python
    # bindings to SparseTensor overloads, such as add(Tensor, SparseTensor),
    # since the Tensor-based signature already dynamically dispatches correctly.
    # However, _sparse_mask only has a SparseTensor signature so we need to bind
    # that function.
    for arg in declaration['arguments']:
        if arg['type'] == 'SparseTensor' and declaration['name'] != '_sparse_mask':
            return False

    # we don't currently support functions which are only defined on Type
    # such as zeros(), randn(), etc.
    method_of = declaration['method_of']
    if 'Tensor' not in method_of and 'namespace' not in method_of:
        return False

    return True


def gen_py_variable_methods(out, declarations):
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                declaration['mode'] != 'NN' and
                'Tensor' in declaration['method_of'])

    py_variable_methods = defaultdict(list)
    for declaration in declarations:
        if should_bind(declaration):
            py_variable_methods[declaration['name']].append(declaration)

    env = create_python_bindings(py_variable_methods, True)
    write(out, 'python_variable_methods.cpp', PY_VARIABLE_METHODS_CPP, env)
    write(out, 'python_variable_methods_dispatch.h', PY_VARIABLE_DISPATCH_H, env)


def gen_py_nn_functions(out, declarations):
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                declaration['mode'] == 'NN')

    py_nn_functions = defaultdict(list)
    for declaration in declarations:
        if should_bind(declaration):
            py_nn_functions[declaration['name']].append(declaration)

    env = create_python_bindings(py_nn_functions, has_self=False, is_module=True)
    write(out, 'python_nn_functions.cpp', PY_NN_FUNCTIONS_CPP, env)
    write(out, 'python_nn_functions.h', PY_NN_FUNCTIONS_H, env)
    write(out, 'python_nn_functions_dispatch.h', PY_NN_DISPATCH_H, env)


def gen_py_torch_functions(out, declarations):
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                declaration['mode'] != 'NN' and
                'namespace' in declaration['method_of'])

    py_torch_functions = defaultdict(list)
    for declaration in declarations:
        name = declaration['name']
        if should_bind(declaration):
            if name.endswith('_out'):
                py_torch_functions[name[:-4]].append(declaration)
            else:
                py_torch_functions[name].append(declaration)

    env = create_python_bindings(py_torch_functions, has_self=False)
    write(out, 'python_torch_functions.cpp', PY_TORCH_FUNCTIONS_CPP, env)
    write(out, 'python_torch_functions_dispatch.h', PY_TORCH_DISPATCH_H, env)


def create_python_bindings(python_functions, has_self, is_module=False):
    """Generates Python bindings to ATen functions"""
    py_methods = []
    py_method_defs = []
    py_method_dispatch = []

    unpack_methods = {
        'const Tensor &': 'tensor',
        'SparseTensor': 'tensor',
        'Tensor &': 'tensor',
        'Generator *': 'generator',
        'Storage &': 'storage',
        'int64_t': 'toInt64',
        'bool': 'toBool',
        'double': 'toDouble',
    }

    unpack_with_default_methods = {
        'IntList': 'setDefaultIntlist',
        'Scalar': 'scalarWithDefault',
        'int64_t': 'toInt64WithDefault',
        'bool': 'setDefaultBool',
        'double': 'setDefaultDouble',
    }

    def first_tensor_arg(arguments):
        for arg in arguments:
            if arg['simple_type'] in {'Tensor', 'TensorList'}:
                return arg['name']
        return None

    def auto_gpu(option):
        tensor_arg = first_tensor_arg(option['arguments'])
        if tensor_arg is None:
            return ''
        return 'AutoGPU auto_gpu({});'.format(tensor_arg)

    def emit_single_dispatch(declaration, base_env):
        env = {}
        simple_return_type = declaration['return_type'].replace(' &', '')
        assert simple_return_type in SUPPORTED_RETURN_TYPES, \
            declaration['name'] + ' returns unsupported type: ' + simple_return_type

        body = []
        actuals = []
        formal_args = []
        arg_idx = 0

        def is_output(arg):
            return arg.get('output', False)

        inputs = [arg for arg in declaration['arguments'] if not is_output(arg)]
        outputs = [arg for arg in declaration['arguments'] if is_output(arg)]

        def parse_arg(arg, unpack_args=False):
            name = arg['name']
            typename = arg['type']
            if typename.startswith('IntList['):
                typename = 'IntList'
            if typename.startswith('LongTensor'):
                typename = 'Tensor'

            if arg.get('python_default_init'):
                assert typename in unpack_with_default_methods, \
                    '`{}` type is not supported in python_default_init'.format(typename)
                unpack_with_default = unpack_with_default_methods.get(typename)
                default_expr = arg.get('python_default_init')
                expr = 'r.{}({}, {})'.format(unpack_with_default, arg_idx, default_expr)
            else:
                unpack = unpack_methods.get(typename, typename.lower())
                expr = 'r.{}({})'.format(unpack, arg_idx)

            if unpack_args:
                body.append('auto {} = {};'.format(name, expr))
                expr = name

            if typename == 'Storage &':
                expr = '*' + expr
            if typename == 'SparseTensor':
                expr = 'SparseTensor({})'.format(expr)

            actuals.append(expr)
            dispatch_type = typename
            if dispatch_type == 'Tensor':
                dispatch_type = 'const Tensor &'
            elif dispatch_type == 'Tensor &':
                dispatch_type = 'Tensor'
            formal_args.append('{} {}'.format(dispatch_type, name))

        unpack = any(arg.get('python_default_init') for arg in inputs)
        for arg in inputs:
            if has_self and arg['name'] == 'self':
                formal_args.append('Tensor & self')
                actuals.append('self_')
                continue
            parse_arg(arg, unpack)
            arg_idx += 1

        if len(outputs) == 1:
            parse_arg(outputs[0])
        elif len(outputs) > 1:
            N = len(outputs)
            body.append('auto results = r.tensorlist_n<{}>({});'.format(N, arg_idx))
            for i, arg in enumerate(outputs):
                formal_args.append('Tensor & {}'.format(arg['name']))
                actuals.append('results[{}]'.format(i))

        env['unpack_args'] = []
        env['formal_args'] = formal_args
        env['actuals'] = actuals
        if 'call_args' in declaration:
            env['dispatch_args'] = declaration['call_args']
        else:
            env['dispatch_args'] = [arg['name'] for arg in declaration['arguments']]
        if 'Tensor' in declaration['method_of']:
            env['dispatch_args'] = [arg for arg in env['dispatch_args'] if arg != 'self']
            env['dispatch_call'] = 'self.{}'.format(declaration['name'])
        else:
            env['dispatch_call'] = 'at::{}'.format(declaration['name'])
        env['AutoNoGIL'] = 'AutoNoGIL no_gil;'
        env['AutoGPU'] = auto_gpu(declaration)
        env = nested_dict(env, nested_dict(base_env, declaration))
        body.append(PY_VARIABLE_CALL_DISPATCH.substitute(env))
        py_method_dispatch.append(PY_VARIABLE_DISPATCH.substitute(env))
        return body

    def emit_dispatch(i, declarations, base_env):
        if len(declarations) == 1:
            body = emit_single_dispatch(declarations[0], base_env)
        else:
            assert len(declarations) == 2
            env = {
                'call_dispatch_out': emit_single_dispatch(declarations[0], base_env),
                'call_dispatch': emit_single_dispatch(declarations[1], base_env),
            }
            out_idx = len([arg for arg in declarations[0]['arguments']
                           if not arg.get('output', False)])
            body = PY_VARIABLE_OUT.substitute(env, out_idx=out_idx).split('\n')
        cond = 'if' if i == 0 else '} else if'
        return PY_VARIABLE_CASE.substitute(i=i, cond=cond, call_dispatch=body)

    def process_function(name, declarations):
        env = {
            'name': name,
            'dispatch_name': 'dispatch_{}'.format(name),
            'pycname': 'THPVariable_{}'.format(name),
            'prototypes': [],
            'max_args': max(len(o['arguments']) for o in declarations),
            'unpack_self': [],
            'dispatch': [],
        }

        if has_self:
            env['unpack_self'] = [UNPACK_SELF]

        grouped = group_declarations(declarations)
        for prototype, decls in grouped:
            if has_self:
                prototype = prototype.replace('Tensor self, ', '')
                prototype = prototype.replace('Tensor self', '')
            if not has_self:
                # Use 'input' instead of 'self' for NN functions
                prototype = prototype.replace('Tensor self', 'Tensor input')
            prototype = prototype.replace('SparseTensor', 'Tensor')
            if all('deprecated' in o for o in decls):
                prototype += '|deprecated'
            env['prototypes'].append('"{}",'.format(prototype))

        for i, (_, decls) in enumerate(grouped):
            env['dispatch'].append(emit_dispatch(i, decls, env))
        env['dispatch'].append('}')

        if len(declarations) == 1 and len(declarations[0]['args']) == 1 and has_self:
            tmpl = PY_VARIABLE_METHOD_NOARGS
            env['actuals'] = ['self_']
            env['flags'] = 'METH_NOARGS'
        else:
            tmpl = PY_VARIABLE_METHOD_VARARGS
            env['flags'] = 'METH_VARARGS | METH_KEYWORDS'

        if not is_module and not has_self:
            env['flags'] += ' | METH_STATIC'

        py_methods.append(tmpl.substitute(env))
        py_method_defs.append(PY_VARIABLE_METHOD_DEF.substitute(env))

    for name in sorted(python_functions.keys()):
        process_function(name, python_functions[name])

    return {
        'py_methods': py_methods,
        'py_method_defs': py_method_defs,
        'py_method_dispatch': py_method_dispatch,
    }


def group_declarations(declarations):
    grouped = defaultdict(list)

    # first group by prototype ignoring out arguments
    for declaration in declarations:
        grouped[get_prototype(declaration, False)].append(declaration)

    result = []
    for prototype in sorted(grouped.keys()):
        group = grouped[prototype]
        assert len(group) <= 2
        if len(group) == 2:
            assert group[0]['name'].endswith('_out')
        result.append((get_prototype(group[0], True), group))
    return result


def get_prototype(declaration, include_out):
    # Use the saved prototype for deprecated pseudo-declarations
    if 'prototype' in declaration:
        return declaration['prototype']

    # Compute the Python function prototype for argument parsing
    typed_args = []
    output_args = []
    positional = True
    for arg in declaration['arguments']:
        if arg.get('output', False):
            output_args.append(arg)
            continue
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

    # add output arguments
    name = declaration['name']
    if name.endswith('_out'):
        name = name[:-4]

    if len(output_args) > 0 and include_out:
        assert declaration['name'].endswith('_out')
        if positional:
            typed_args.append('*')
            positional = False
        typenames = [arg['simple_type'] for arg in output_args]
        if len(typenames) > 1:
            typename = 'TensorList[{}]'.format(len(typenames))
        else:
            typename = typenames[0]
        typed_args.append(typename + ' out=None')

    # Python function prototype.
    # This is the string that we give to FunctionParameter, which is
    # then parsed into the actual structure which we do parsing
    # with.
    return FUNCTION_PROTOTYPE.substitute(name=name, typed_args=typed_args)
