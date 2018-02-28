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
    '.*_backward', '.*_backward_out', '.*_forward', '.*_forward_out',
    'sparse_raw_resize_', '_unsafe_view', 'tensor', 'sparse_coo_tensor',
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
    ${signatures}
  });
  ${unpack_self}
  ParsedArgs<${max_args}> parsed_args;
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

PY_VARIABLE_OUT_CHECK_DTYPE = CodeTemplate("""\
if (r.isNone(${out_idx})) {
  ${call_dispatch}
} else {
  if (!r.isNone(${dtype_idx})) {
    check_out_dtype_matches(r.tensor(${out_idx}), r.type(${dtype_idx}));
  }
  ${call_dispatch_out}
}
""")

PY_VARIABLE_CALL_DISPATCH = CodeTemplate("""\
${dispatch_name}(${actuals})""")

PY_VARIABLE_SET_REQUIRES_GRAD = CodeTemplate("""\
set_requires_grad(${call_dispatch}, ${requires_grad})""")

PY_VARIABLE_WRAP = CodeTemplate("""\
return wrap(${call_dispatch});""")

PY_VARIABLE_DISPATCH = CodeTemplate("""\
inline ${return_type} ${dispatch_name}(${formal_args}) {
  ${initialize_cuda}
  ${AutoNoGIL}
  ${AutoGPU}
  return ${dispatch_call}(${dispatch_args});
}
""")

PY_VARIABLE_METHOD_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)${pycname}, ${flags}, NULL},""")

UNPACK_SELF = "auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;"

PYTHON_FUNCTION_SIGNATURE = CodeTemplate("""\
${name}(${typed_args})""")

# XXX: if you got here because of an assertion failure, it doesn't mean
# it's enough to just extend the list here. Before you do this, make sure
# to add an appropriate wrap() overload in torch/csrc/autograd/utils/wrap_outputs.h.
SUPPORTED_RETURN_TYPES = {
    'Tensor', 'std::tuple<Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>',
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

    return True


def gen_py_variable_methods(out, declarations):
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                declaration['mode'] != 'NN' and
                'Tensor' in declaration['method_of'])

    py_variable_methods = group_declarations_by_name(declarations, should_bind)

    env = create_python_bindings(py_variable_methods, True)
    write(out, 'python_variable_methods.cpp', PY_VARIABLE_METHODS_CPP, env)
    write(out, 'python_variable_methods_dispatch.h', PY_VARIABLE_DISPATCH_H, env)


def gen_py_nn_functions(out, declarations):
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                declaration['mode'] == 'NN')

    py_nn_functions = group_declarations_by_name(declarations, should_bind)

    env = create_python_bindings(py_nn_functions, has_self=False, is_module=True)
    write(out, 'python_nn_functions.cpp', PY_NN_FUNCTIONS_CPP, env)
    write(out, 'python_nn_functions.h', PY_NN_FUNCTIONS_H, env)
    write(out, 'python_nn_functions_dispatch.h', PY_NN_DISPATCH_H, env)


def gen_py_torch_functions(out, declarations):
    def is_namespace_or_type_api_function(declaration):
        # These are functions that should end up on the torch module. This
        # includes functions on the at:: namespace and ones that are typically
        # called via the type (e.g. Type::randn()). Since every function is
        # implemented on the Type, we exclude functions that are also declared
        # as methods on Tensor, since one shouldn't generally call these from
        # the Type object.
        if 'namespace' in declaration['method_of']:
            return True
        if 'Tensor' in declaration['method_of']:
            return False
        return 'Type' in declaration['method_of']

    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                declaration['mode'] != 'NN' and
                is_namespace_or_type_api_function(declaration))

    py_torch_functions = group_declarations_by_name(declarations, should_bind)

    env = create_python_bindings(py_torch_functions, has_self=False)
    write(out, 'python_torch_functions.cpp', PY_TORCH_FUNCTIONS_CPP, env)
    write(out, 'python_torch_functions_dispatch.h', PY_TORCH_DISPATCH_H, env)


def group_declarations_by_name(declarations, should_bind_fn):
    """Group declarations by name ignoring _out suffix"""
    groups = defaultdict(list)
    for declaration in declarations:
        name = declaration['name']
        if should_bind_fn(declaration):
            if name.endswith('_out'):
                groups[name[:-4]].append(declaration)
            else:
                groups[name].append(declaration)
    return groups


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
        'const Type &': 'type',
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

    def auto_gpu(option, has_device_bind):
        tensor_arg = first_tensor_arg(option['arguments'])
        if tensor_arg is not None:
            if not has_device_bind:
                return 'AutoGPU auto_gpu({});'.format(tensor_arg)
            else:  # e.g. for ones_like, the default is the device of the tensor arg
                device_to_use = '({}.type().is_cuda() ? {}.get_device() : -1)'.format(tensor_arg, tensor_arg)
                return 'AutoGPU auto_gpu(device == -1 ? {} : device);'.format(device_to_use)
        elif has_device_bind:
            return 'AutoGPU auto_gpu(device);'
        else:
            return ''

    def emit_single_dispatch(declaration, out_idx, base_env):
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

        def parse_arg(arg, arg_index, unpack_args=False):
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
                expr = 'r.{}({}, {})'.format(unpack_with_default, arg_index, default_expr)
            else:
                unpack = unpack_methods.get(typename, typename.lower())
                expr = 'r.{}({})'.format(unpack, arg_index)

            if unpack_args:
                body.append('auto {} = {};'.format(name, expr))
                expr = name

            if typename == 'Storage &':
                expr = '*' + expr
            if typename == 'SparseTensor':
                expr = 'SparseTensor({})'.format(expr)

            dispatch_type = typename
            if dispatch_type == 'Tensor':
                dispatch_type = 'const Tensor &'
            elif dispatch_type == 'Tensor &':
                dispatch_type = 'Tensor'
            formal = '{} {}'.format(dispatch_type, name)
            return expr, formal

        def append_actuals_formals(actual, formal):
            actuals.append(actual)
            formal_args.append(formal)

        unpack = any(arg.get('python_default_init') for arg in inputs)
        for arg in inputs:
            if has_self and arg['name'] == 'self':
                formal_args.append('Tensor & self')
                actuals.append('self_')
                continue
            append_actuals_formals(*parse_arg(arg, arg_idx, unpack))
            arg_idx += 1

        if len(outputs) == 1:
            append_actuals_formals(*parse_arg(outputs[0], arg_idx))
        elif len(outputs) > 1:
            N = len(outputs)
            body.append('auto results = r.tensorlist_n<{}>({});'.format(N, arg_idx))
            for i, arg in enumerate(outputs):
                formal_args.append('Tensor & {}'.format(arg['name']))
                actuals.append('results[{}]'.format(i))

        # check python_binding_arguments
        has_dtype_bind = False
        has_device_bind = False
        requires_grad = None
        python_binding_arguments = declaration.get('python_binding_arguments', [])
        bind_arg_idx = arg_idx if out_idx is None else out_idx + 1
        if 'dtype' in (a['name'] for a in python_binding_arguments):
            dtype_idx, device_idx, requires_grad_idx = (bind_arg_idx, bind_arg_idx + 1, bind_arg_idx + 2)
        else:
            device_idx, requires_grad_idx = (bind_arg_idx, bind_arg_idx + 1)

        for arg in python_binding_arguments:
            if arg['name'] == 'dtype' and arg['simple_type'] == 'Type':
                # out(s) determines the dtype if it is present, so don't pass the dtype to the dispatch.
                if len(outputs) == 0:
                    # we have to use out_idx if there is an out variant because the base variant
                    # won't have the full arg_idx count
                    has_dtype_bind = True
                    append_actuals_formals(*parse_arg(arg, dtype_idx))
                elif len(outputs) > 1:
                    raise RuntimeError("Not supported: dtype parameter with multiple outputs")
            elif arg['name'] == 'device' and arg['simple_type'] == 'int64_t':
                if len(outputs) == 0:
                    has_device_bind = True
                    append_actuals_formals(*parse_arg(arg, device_idx))
            elif arg['name'] == 'requires_grad' and arg['simple_type'] == 'bool':
                requires_grad = parse_arg(arg, requires_grad_idx)[0]
            else:
                raise RuntimeError(("found {} in python_binding_arguments but only "
                                   " \"bool requires_grad\" and \"Type dtype\" are supported".format(arg)))

        env['unpack_args'] = []
        env['formal_args'] = formal_args
        env['actuals'] = actuals
        has_any_dtype = (has_dtype_bind or any('dtype' in a['name'] for a in inputs))
        env['initialize_cuda'] = 'maybe_initialize_cuda(dtype);' if has_any_dtype else []
        if 'call_args' in declaration:
            env['dispatch_args'] = declaration['call_args']
        else:
            env['dispatch_args'] = [arg['name'] for arg in declaration['arguments']]
        if 'Tensor' in declaration['method_of']:
            env['dispatch_args'] = [arg for arg in env['dispatch_args'] if arg != 'self']
            env['dispatch_call'] = 'self.{}'.format(declaration['name'])
        elif 'namespace' in declaration['method_of']:
            if has_dtype_bind:
                raise RuntimeError(("dtype with namespace dispatch currently not supported, "
                                   "consider writing as a native function"))
            env['dispatch_call'] = 'at::{}'.format(declaration['name'])
        elif has_dtype_bind:
            env['dispatch_call'] = 'dtype.{}'.format(declaration['name'])
        else:
            env['dispatch_call'] = 'default_type().{}'.format(declaration['name'])
        env['AutoNoGIL'] = 'AutoNoGIL no_gil;'
        env['AutoGPU'] = auto_gpu(declaration, has_device_bind)

        env = nested_dict(env, nested_dict(base_env, declaration))
        call_dispatch = PY_VARIABLE_CALL_DISPATCH.substitute(env)
        if requires_grad:
            call_dispatch = PY_VARIABLE_SET_REQUIRES_GRAD.substitute(env, call_dispatch=call_dispatch,
                                                                     requires_grad=requires_grad)
        body.append(PY_VARIABLE_WRAP.substitute(env, call_dispatch=call_dispatch))
        py_method_dispatch.append(PY_VARIABLE_DISPATCH.substitute(env))
        return body

    def emit_dispatch(i, dictionary, base_env):
        if 'out' in dictionary:
            out_idx = len([arg for arg in dictionary['out']['arguments']
                           if not arg.get('output', False)])
            env = {}
            env['call_dispatch_out'] = emit_single_dispatch(dictionary['out'], out_idx, base_env)
            env['call_dispatch'] = emit_single_dispatch(dictionary['base'], out_idx, base_env)

            has_dtype_bind = 'dtype' in [d['name'] for d in dictionary['out'].get('python_binding_arguments', [])]
            if has_dtype_bind:
                body = PY_VARIABLE_OUT_CHECK_DTYPE.substitute(env, out_idx=out_idx, dtype_idx=out_idx + 1).split('\n')
            else:
                body = PY_VARIABLE_OUT.substitute(env, out_idx=out_idx).split('\n')
        else:
            body = emit_single_dispatch(dictionary['base'], None, base_env)

        cond = 'if' if i == 0 else '} else if'
        return PY_VARIABLE_CASE.substitute(i=i, cond=cond, call_dispatch=body)

    def get_python_binding_arguments(declaration):
        python_binding_arguments = []
        has_tensor_input_arg = False
        for arg in declaration['arguments']:
            if arg.get('output', False):
                continue
            typename = arg['simple_type']
            if typename in ['Tensor', 'TensorList']:
                has_tensor_input_arg = True
            if arg['name'] == 'requires_grad':
                raise ValueError("argument named requires_grad not supported")

        has_tensor_return = False
        for ret in declaration['returns']:
            if ret['dynamic_type'] in ['Tensor', 'TensorList']:
                # this probably won't work if one of the returns is not a tensor, but it will
                # produce a compile-time error that is obvious
                has_tensor_return = True

        if has_tensor_return and not has_tensor_input_arg:
            if declaration['name'].startswith('randperm'):
                default_type = 'torch.int64'
            else:
                default_type = 'None'
            dtype_arg = {
                'default': default_type,
                'dynamic_type': 'Type',
                'kwarg_only': True,
                'name': 'dtype',
                'type': 'const Type &',
                'simple_type': 'Type',
            }
            python_binding_arguments.append(dtype_arg)
        if (not has_tensor_input_arg or name.endswith('_like')) and has_tensor_return:
            device_arg = {
                'default': -1,
                'default_init': -1,
                'dynamic_type': 'int64_t',
                'kwarg_only': True,
                'name': 'device',
                'type': 'int64_t',
                'simple_type': 'int64_t'
            }
            python_binding_arguments.append(device_arg)
            requires_grad_arg = {
                'default': False,
                'dynamic_type': 'bool',
                'kwarg_only': True,
                'name': 'requires_grad',
                'type': 'bool',
                'simple_type': 'bool',
            }
            python_binding_arguments.append(requires_grad_arg)
        return python_binding_arguments

    def process_function(name, declarations):
        for declaration in declarations:
            declaration['python_binding_arguments'] = get_python_binding_arguments(declaration)

        env = {
            'name': name,
            'dispatch_name': 'dispatch_{}'.format(name),
            'pycname': 'THPVariable_{}'.format(name),
            'signatures': [],
            'max_args': max(len(o['arguments']) + len(o['python_binding_arguments']) for o in declarations),
            'unpack_self': [],
            'dispatch': [],
        }

        if has_self:
            env['unpack_self'] = [UNPACK_SELF]

        grouped = group_declarations(declarations)
        for i, dictionary in enumerate(grouped):
            signature = dictionary['signature']
            if has_self:
                signature = signature.replace('Tensor self, ', '')
                signature = signature.replace('Tensor self', '')
            if not has_self:
                # Use 'input' instead of 'self' for NN functions
                signature = signature.replace('Tensor self', 'Tensor input')
            signature = signature.replace('SparseTensor', 'Tensor')
            if dictionary['base'].get('deprecated', False):
                signature += '|deprecated'
            env['signatures'].append('"{}",'.format(signature))
            env['dispatch'].append(emit_dispatch(i, dictionary, env))

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
    """Returns a list of dictionaries containing the optional keys:

       "base": the regular ATen declaration (e.g. conv2d)
       "out": the out variant (e.g. conv2d_out)
       "signature": the signature used for Python argument parsing
    """
    grouped = defaultdict(dict)

    # first group by signature ignoring out arguments
    for declaration in declarations:
        signature = get_python_signature(declaration, False)
        v = grouped[signature]
        if declaration['name'].endswith('_out'):
            v['out'] = declaration
            # prefer the signature with optional out=... arguments
            v['signature'] = get_python_signature(declaration, True)
        else:
            v['base'] = declaration
            if 'signature' not in v:
                v['signature'] = signature

    result = []
    for _, dictionary in sorted(grouped.items()):
        assert 'base' in dictionary
        result.append(dictionary)
    return result


def get_python_signature(declaration, include_out):
    # Compute the Python function signature for argument parsing
    typed_args = []
    output_args = []
    positional = True

    def get_typed_arg(arg):
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
        return param

    for arg in declaration['arguments']:
        if arg.get('output', False):
            output_args.append(arg)
            continue
        if arg.get('kwarg_only', False) and positional:
            typed_args.append('*')
            positional = False
        param = get_typed_arg(arg)
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

    # we could put this in the loop above but we want to ensure it is after the out argument
    if len(declaration['python_binding_arguments']) > 0:
        for arg in declaration['python_binding_arguments']:
            if arg.get('kwarg_only', False) and positional:
                typed_args.append('*')
                positional = False
            typed_args.append(get_typed_arg(arg))

    # Python function signature.
    # This is the string that we give to FunctionParameter, which is
    # then parsed into the actual structure which we do parsing
    # with.
    return PYTHON_FUNCTION_SIGNATURE.substitute(name=name, typed_args=typed_args)
