from .nested_dict import nested_dict
from tools.shared.module_loader import import_module

CodeTemplate = import_module('code_template', 'torch/lib/ATen/code_template.py').CodeTemplate


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


def create_python_bindings(
        python_functions, py_methods, py_method_defs, py_method_dispatch,
        is_static):
    """python_variable_methods.cpp

    Generates Python bindings to Variable methods
    """

    unpack_methods = {
        'int64_t': 'toInt64',
        'bool': 'toBool'
    }

    def first_tensor_arg(arguments):
        for arg in arguments:
            if arg['type'] in {'Tensor', 'TensorList'}:
                return arg['name']
        return None

    def auto_gpu(option):
        tensor_arg = first_tensor_arg(option['python_arguments'])
        if tensor_arg is None:
            return ''
        return 'AutoGPU auto_gpu({});'.format(tensor_arg)

    def emit_dispatch(i, function, has_self):
        env = {}

        actuals = []
        formal_args = []
        arg_idx = 0
        for arg in function['python_arguments']:
            if arg['name'] == 'self' and not is_static:
                formal_args.append('Tensor & {}'.format(arg['name']))
                actuals.append('self_')
                continue

            typename = arg['type']
            if typename.startswith('IntList['):
                typename = 'IntList'
            if typename.startswith('LongTensor'):
                typename = 'Tensor'

            unpack = unpack_methods.get(typename, typename.lower())
            actuals.append('r.{}({})'.format(unpack, arg_idx))
            dispatch_type = typename
            dispatch_type = 'const Tensor &' if dispatch_type == 'Tensor' else dispatch_type
            formal_args.append('{} {}'.format(dispatch_type, arg['name']))
            arg_idx += 1

        env['i'] = i
        env['actuals'] = actuals
        env['formal_args'] = formal_args
        env['dispatch_args'] = function['call_args']
        if not is_static and has_self:
            env['dispatch_args'] = [arg for arg in env['dispatch_args'] if arg != 'self']
            env['dispatch_call'] = 'self.{}'.format(function['name'])
        else:
            env['dispatch_call'] = 'at::{}'.format(function['name'])
        env['AutoNoGIL'] = 'AutoNoGIL no_gil;'
        env['AutoGPU'] = auto_gpu(function)
        env['cond'] = 'if' if i == 0 else '} else if'
        env = nested_dict(env, function)
        py_method_dispatch.append(PY_VARIABLE_DISPATCH.substitute(env))
        return PY_VARIABLE_CASE.substitute(env)

    def process_function(name, functions):
        env = {
            'name': name,
            'dispatch_name': 'dispatch_{}'.format(name),
            'pycname': 'THPVariable_{}'.format(name),
            'prototypes': [],
            'max_args': max(len(o['python_arguments']) for o in functions),
            'unpack_self': [],
            'dispatch': [],
        }

        has_self = 'self' in functions[0]['args']
        if has_self:
            env['unpack_self'] = [UNPACK_SELF]

        for o in functions:
            prototype = o['prototype']
            if o['inplace']:
                prototype = prototype.replace('(', '_(')
            if not is_static:
                prototype = prototype.replace('Tensor self, ', '')
                prototype = prototype.replace('Tensor self', '')
            if 'deprecated' in o:
                prototype += '|deprecated'
            env['prototypes'].append('"{}",'.format(prototype))

        for i, option in enumerate(functions):
            env['dispatch'].append(emit_dispatch(i, nested_dict(env, option), has_self))
        env['dispatch'].append('}')

        if len(functions) == 1 and len(functions[0]['args']) == 1 and not is_static:
            tmpl = PY_VARIABLE_METHOD_NOARGS
            env['actuals'] = ['self_']
            env['flags'] = 'METH_NOARGS'
        else:
            tmpl = PY_VARIABLE_METHOD_VARARGS
            env['flags'] = 'METH_VARARGS | METH_KEYWORDS'

        py_methods.append(tmpl.substitute(env))
        py_method_defs.append(PY_VARIABLE_METHOD_DEF.substitute(env))

    for name in sorted(python_functions.keys()):
        process_function(name, python_functions[name])
