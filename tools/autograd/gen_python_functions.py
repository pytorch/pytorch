# Generates Python bindings for ATen functions
#
# The bindings are generated as methods on python_variable or functions on the
# torch._C._nn object.

from .nested_dict import nested_dict
from tools.shared.module_loader import import_module

CodeTemplate = import_module('code_template', 'aten/src/ATen/code_template.py').CodeTemplate


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
  return wrap(${dispatch_name}(${actuals}));
""")

PY_VARIABLE_CASE_WITH_UNPACK = CodeTemplate("""\
${cond} (r.idx == ${i}) {
  ${unpack_args}
  return wrap(${dispatch_name}(${actuals}));
""")

PY_VARIABLE_DISPATCH = CodeTemplate("""\
inline ${return_type} ${dispatch_name}(${formal_args}) {
  ${AutoNoGIL}
  ${AutoGPU}
  return ${dispatch_call}(${dispatch_args});
}
""")

PY_VARIABLE_METHOD_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)${pycname}, ${flags}, NULL},""")

UNPACK_ARG = CodeTemplate("""\
${formal_arg} = ${actual};""")

UNPACK_SELF = "auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;"

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


def create_python_bindings(
        python_functions, py_methods, py_method_defs, py_method_dispatch,
        is_class):
    """python_variable_methods.cpp

    Generates Python bindings to Variable methods
    """

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

    def emit_dispatch(i, function):
        env = {}
        simple_return_type = function['return_type'].replace(' &', '')
        assert simple_return_type in SUPPORTED_RETURN_TYPES, \
            function['name'] + ' returns unsupported type: ' + simple_return_type

        actuals = []
        unpack_args = False
        formal_args = []
        arg_idx = 0
        for arg in function['arguments']:
            name = arg['name']
            if 'Tensor' in function['method_of'] and name == 'self':
                formal_args.append('Tensor & {}'.format(name))
                actuals.append('self_')
                continue

            typename = arg['type']
            if typename.startswith('IntList['):
                typename = 'IntList'
            if typename.startswith('LongTensor'):
                typename = 'Tensor'

            if arg.get('python_default_init'):
                unpack_args = True
                assert typename in unpack_with_default_methods, \
                    '`{}` type is not supported in python_default_init'.format(typename)
                unpack_with_default = unpack_with_default_methods.get(typename)
                default_expr = arg.get('python_default_init')
                expr = 'r.{}({}, {})'.format(unpack_with_default, arg_idx, default_expr)
            else:
                unpack = unpack_methods.get(typename, typename.lower())
                expr = 'r.{}({})'.format(unpack, arg_idx)

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
            arg_idx += 1

        env['i'] = i
        env['unpack_args'] = []
        env['formal_args'] = formal_args
        if unpack_args:
            unpack_statements_no_default = []
            unpack_statements_with_default = []
            actual_names = []
            for arg, formal_arg, actual in zip(function['arguments'], formal_args, actuals):
                name = arg['name']
                actual_names.append(name)
                unpack_expr = UNPACK_ARG.substitute(formal_arg=formal_arg, actual=actual)
                if arg.get('python_default_init'):
                    unpack_statements_with_default.append(unpack_expr)
                else:
                    unpack_statements_no_default.append(unpack_expr)
            env['unpack_args'] = unpack_statements_no_default + unpack_statements_with_default
            env['actuals'] = actual_names
            code_template = PY_VARIABLE_CASE_WITH_UNPACK
        else:
            env['actuals'] = actuals
            code_template = PY_VARIABLE_CASE
        if 'call_args' in function:
            env['dispatch_args'] = function['call_args']
        else:
            env['dispatch_args'] = [arg['name'] for arg in function['arguments']]
        if 'Tensor' in function['method_of']:
            env['dispatch_args'] = [arg for arg in env['dispatch_args'] if arg != 'self']
            env['dispatch_call'] = 'self.{}'.format(function['name'])
        else:
            env['dispatch_call'] = 'at::{}'.format(function['name'])
        env['AutoNoGIL'] = 'AutoNoGIL no_gil;'
        env['AutoGPU'] = auto_gpu(function)
        env['cond'] = 'if' if i == 0 else '} else if'
        env = nested_dict(env, function)
        py_method_dispatch.append(PY_VARIABLE_DISPATCH.substitute(env))
        return code_template.substitute(env)

    def process_function(name, functions):
        env = {
            'name': name,
            'dispatch_name': 'dispatch_{}'.format(name),
            'pycname': 'THPVariable_{}'.format(name),
            'prototypes': [],
            'max_args': max(len(o['arguments']) for o in functions),
            'unpack_self': [],
            'dispatch': [],
        }

        is_method = 'Tensor' in functions[0]['method_of']
        if is_method:
            env['unpack_self'] = [UNPACK_SELF]

        for o in functions:
            prototype = o['prototype']
            if is_method:
                prototype = prototype.replace('Tensor self, ', '')
                prototype = prototype.replace('Tensor self', '')
            if not is_class:
                # Use 'input' instead of 'self' for NN functions
                prototype = prototype.replace('Tensor self', 'Tensor input')
            prototype = prototype.replace('SparseTensor', 'Tensor')
            if 'deprecated' in o:
                prototype += '|deprecated'
            env['prototypes'].append('"{}",'.format(prototype))

        for i, option in enumerate(functions):
            env['dispatch'].append(emit_dispatch(i, nested_dict(env, option)))
        env['dispatch'].append('}')

        if len(functions) == 1 and len(functions[0]['args']) == 1 and is_method:
            tmpl = PY_VARIABLE_METHOD_NOARGS
            env['actuals'] = ['self_']
            env['flags'] = 'METH_NOARGS'
        else:
            tmpl = PY_VARIABLE_METHOD_VARARGS
            env['flags'] = 'METH_VARARGS | METH_KEYWORDS'

        if is_class and not is_method:
            env['flags'] += ' | METH_STATIC'

        py_methods.append(tmpl.substitute(env))
        py_method_defs.append(PY_VARIABLE_METHOD_DEF.substitute(env))

    for name in sorted(python_functions.keys()):
        process_function(name, python_functions[name])
