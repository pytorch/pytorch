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
from .gen_variable_type import should_trace
from .utils import write

CodeTemplate = import_module('code_template', 'aten/src/ATen/code_template.py').CodeTemplate

# These functions require manual Python bindings or are not exposed to Python
SKIP_PYTHON_BINDINGS = [
    'alias', 'contiguous', 'clamp.*', 'is_cuda', 'is_sparse', 'size', 'stride',
    '.*_backward', '.*_backward_(out|input|weight|bias)', '.*_forward',
    '.*_forward_out', 'sparse_raw_resize_', '_unsafe_view', 'tensor',
    'sparse_coo_tensor', '_arange.*', '_range.*', '_linspace.*', '_logspace.*',
    '_indexCopy_', 'max_values', 'min_values', 'argmax', 'argmin',
    '_cumsum.*', '_cumprod.*', '_sum.*', '_prod.*', '_th_sum.*', '_th_prod.*',
    'arange.*', 'range.*',
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
  }, /*traceable=*/${traceable});
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

PY_VARIABLE_OUT_CHECK_TYPE = CodeTemplate("""\
if (r.isNone(${out_idx})) {
  ${call_dispatch}
} else {
  check_out_type_matches(r.tensor(${out_idx}), r.scalartype(${type_idx}), r.isNone(${type_idx}),
                         r.layout(${layout_idx}), r.isNone(${layout_idx}),
                         r.device(${device_idx}), r.isNone(${device_idx}));
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
${name}(${py_formal_args})""")

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
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                declaration['mode'] != 'NN' and
                'namespace' in declaration['method_of'])

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


def get_type_default(declaration):
    if declaration['name'].startswith('randperm'):
        return 'torch.int64'
    else:
        return 'None'


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
        'const Type &': 'scalartype',
        'const THPLayout &': 'layout',
        'const Device &': 'device',
        'optional<ScalarType>': 'scalartypeOptional',
        'int64_t': 'toInt64',
        'bool': 'toBool',
        'double': 'toDouble',
        'std::string': 'string',
    }

    unpack_with_default_methods = {
        'IntList': 'setDefaultIntlist',
        'Scalar': 'scalarWithDefault',
        'int64_t': 'toInt64WithDefault',
        'bool': 'setDefaultBool',
        'double': 'setDefaultDouble',
        'const Type &': 'scalartypeWithDefault',
        'const THPLayout &': 'layoutWithDefault',
        'const Device &': 'deviceWithDefault',
        'ScalarType': 'scalartypeWithDefault',
    }

    def first_tensor_arg(arguments):
        for arg in arguments:
            if arg['simple_type'] in {'Tensor', 'TensorList'}:
                return arg['name']
        return None

    def auto_gpu(option, has_device_bind):
        if option['auto_gpu']:
            tensor_arg = first_tensor_arg(option['arguments'])
            if tensor_arg is not None:
                if not has_device_bind:
                    return 'AutoGPU auto_gpu({});'.format(tensor_arg)
                else:  # e.g. for ones_like, the default is the device of the tensor arg
                    device_to_use = '({}.type().is_cuda() ? {}.get_device() : -1)'.format(tensor_arg, tensor_arg)
                    return 'AutoGPU auto_gpu(device == -1 ? {} : device);'.format(device_to_use)
            elif has_device_bind:
                return 'AutoGPU auto_gpu(device);'
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

        def get_type_args(args):
            return [arg for arg in args if arg['simple_type'] == 'Type']
        type_actual_args = get_type_args(declaration['arguments'])
        type_binding_args = get_type_args(declaration['python_binding_arguments'])
        assert len(type_actual_args + type_binding_args) <= 1
        if type_binding_args and len(outputs) == 0:
            # out(s) determines the dtype if it is present, so only use this if there are no outputs.
            type_args = type_binding_args
        else:
            type_args = type_actual_args

        if type_args and len(outputs) > 1:
                raise RuntimeError("Not supported: type dispatched parameter with multiple outputs")

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
                # TODO: Type currently maps to ScalarType, figure out a cleaner solution
                if typename == 'const Type &':
                    default_expr += '.scalarType()'
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
            elif dispatch_type == 'const Device &':
                dispatch_type = 'int64_t'
            formal = '{} {}'.format(dispatch_type, name)
            return expr, formal

        def append_actuals_formals(actual, formal):
            actuals.append(actual)
            formal_args.append(formal)

        unpack = any(arg.get('python_default_init') for arg in inputs)
        for arg in inputs:
            if arg['simple_type'] == 'Type':
                continue
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

        layout = None
        # type args go after the outputs to match the signature generation.
        arg_idx = arg_idx if out_idx is None else out_idx + 1
        for arg in type_args:
            parsed_type_args = parse_arg(arg, arg_idx, unpack)
            arg_idx += 1

        # check python_binding_arguments
        has_device_bind = False
        requires_grad = None
        python_binding_arguments = declaration.get('python_binding_arguments', [])
        if 'dtype' in (a['name'] for a in python_binding_arguments):
            arg_idx += 1  # we already handled this in type_dispatched_args

        if 'layout' in (a['name'] for a in python_binding_arguments):
            layout_idx, device_idx, requires_grad_idx = (arg_idx, arg_idx + 1, arg_idx + 2)
        else:
            device_idx, requires_grad_idx = (arg_idx, arg_idx + 1)

        for arg in python_binding_arguments:
            if arg['name'] == 'dtype' and arg['simple_type'] == 'Type':
                pass  # already handled by type_dispatched_args
            elif arg['name'] == 'layout' and arg['simple_type'] == 'Layout':
                # out(s) determines the type and layout if it is present, so only use this if there are no outputs.
                if len(outputs) == 0:
                    layout = parse_arg(arg, layout_idx, arg.get('python_default_init'))[0]
            elif arg['name'] == 'device' and arg['simple_type'] == 'Device':
                if len(outputs) == 0:
                    assert parsed_type_args
                    assert layout
                    device_arg = parse_arg(arg, device_idx, True)
                    # add type, device formals and corresponding actuals.
                    # The type actual isthe ATen type mapped from (ScalarType, Layout, Device)
                    # The device actual is the corresponding AutoGPU index for the Device.
                    formal_args.append(parsed_type_args[1])
                    formal_args.append(device_arg[1])
                    actuals.append("torch::getType({}, {}, {}.type)".format(parsed_type_args[0], layout, device_arg[0]))
                    actuals.append('{}.deviceInt64()'.format(device_arg[0]))
                    has_device_bind = True
            elif arg['name'] == 'requires_grad' and arg['simple_type'] == 'bool':
                requires_grad = parse_arg(arg, requires_grad_idx)[0]
            else:
                raise RuntimeError(("found {} in python_binding_arguments but only "
                                    "\"bool requires_grad\", \"ScalarType dtype\", \"Layout layout\", "
                                    "\"Device device\" are supported".format(arg)))

        env['unpack_args'] = []
        env['formal_args'] = formal_args
        env['actuals'] = actuals
        maybe_init_cuda = type_args[0]['name'] if type_args else None
        env['initialize_cuda'] = 'maybe_initialize_cuda({});'.format(maybe_init_cuda) if maybe_init_cuda else []
        if 'call_args' in declaration:
            env['dispatch_args'] = declaration['call_args']
        else:
            env['dispatch_args'] = [arg['name'] for arg in declaration['arguments']]
        if 'Tensor' in declaration['method_of']:
            env['dispatch_args'] = [arg for arg in env['dispatch_args'] if arg != 'self']
            env['dispatch_call'] = 'self.{}'.format(declaration['name'])
        elif 'namespace' in declaration['method_of']:
            env['dispatch_call'] = 'at::{}'.format(declaration['name'])
        else:
            raise RuntimeError('could not dispatch, neither namespace function nor Tensor method')
        env['AutoNoGIL'] = 'AutoNoGIL no_gil;' if not declaration['with_gil'] else ''
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
                body = PY_VARIABLE_OUT_CHECK_TYPE.substitute(env, out_idx=out_idx, type_idx=out_idx + 1,
                                                             layout_idx=out_idx + 2, device_idx=out_idx + 3).split('\n')
            else:
                body = PY_VARIABLE_OUT.substitute(env, out_idx=out_idx).split('\n')
        else:
            body = emit_single_dispatch(dictionary['base'], None, base_env)

        cond = 'if' if i == 0 else '} else if'
        return PY_VARIABLE_CASE.substitute(i=i, cond=cond, call_dispatch=body)

    def get_python_binding_arguments(declaration):
        python_binding_arguments = []
        has_tensor_input_arg = False
        has_type_input_arg = False
        for arg in declaration['arguments']:
            if arg.get('output', False):
                continue
            typename = arg['simple_type']
            if typename in ['Tensor', 'TensorList']:
                has_tensor_input_arg = True
            if arg['simple_type'] == 'Type':
                has_type_input_arg = True
            if arg['name'] == 'requires_grad':
                raise ValueError("argument named requires_grad not supported")

        has_tensor_return = False
        for ret in declaration['returns']:
            if ret['dynamic_type'] in ['Tensor', 'TensorList']:
                # this probably won't work if one of the returns is not a tensor, but it will
                # produce a compile-time error that is obvious
                has_tensor_return = True

        is_like_function = name.endswith('_like')
        is_typed_like_function = is_like_function and has_type_input_arg
        is_factory_function = has_tensor_return and not has_tensor_input_arg
        is_factory_or_like_function = has_tensor_return and (not has_tensor_input_arg or is_like_function)

        if is_factory_function and not has_type_input_arg:
            default_type = get_type_default(declaration)
            dtype_arg = {
                'default': default_type,
                'dynamic_type': 'Type',
                'kwarg_only': True,
                'name': 'dtype',
                'type': 'const Type &',
                'simple_type': 'Type',
                'is_type_dispatched': True,
            }
            python_binding_arguments.append(dtype_arg)
        if is_factory_function or is_typed_like_function:
            py_default_layout = '*torch::getLayout(self.type().backend())' if is_typed_like_function else None
            layout_arg = {
                'default': 'torch.strided',
                'dynamic_type': 'Layout',
                'kwarg_only': True,
                'name': 'layout',
                'type': 'const THPLayout &',
                'simple_type': 'Layout',
                'python_default_init': py_default_layout,
            }
            python_binding_arguments.append(layout_arg)
            py_default_device = 'torch::utils::getDevice(self)' if is_typed_like_function else None
            device_arg = {
                'default': 'None',
                'default_init': 'None',
                'dynamic_type': 'Device',
                'kwarg_only': True,
                'name': 'device',
                'type': 'const Device &',
                'simple_type': 'Device',
                'python_default_init': py_default_device
            }
            python_binding_arguments.append(device_arg)
        if is_factory_or_like_function:
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

        env['traceable'] = 'true' if all(should_trace(d) for d in declarations) else 'false'

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
        if 'base' not in dictionary:
            raise RuntimeError('\'base\' not in dictionary', dictionary)
        result.append(dictionary)
    return sort_declarations(result)


# This function declares a partial order on declarations, and sorts them according
# to its linear extension. This is necessary, because there's some ambiguity in the
# choice of overload, and we want a different order.
def sort_declarations(grouped_decls):
    def is_coord_smaller(arg1, arg2):
        return arg1['dynamic_type'] == 'real' and arg2['dynamic_type'] == 'Tensor'

    def is_smaller(d1, d2):
        """Returns True if d1 < d2 in the partial order."""
        args1, args2 = d1['base']['arguments'], d2['base']['arguments']
        if len(args1) != len(args2):
            return False
        any_smaller = any(is_coord_smaller(arg1, arg2) for arg1, arg2 in zip(args1, args2))
        all_smaller_or_equal = all(arg1['dynamic_type'] == arg2['dynamic_type'] or is_coord_smaller(arg1, arg2)
                                   for arg1, arg2 in zip(args1, args2))
        return any_smaller and all_smaller_or_equal

    # Construct the relation graph
    larger_than = defaultdict(set)
    for i1, decl1 in enumerate(grouped_decls):
        for i2, decl2 in enumerate(grouped_decls):
            if is_smaller(decl1, decl2):
                larger_than[i1].add(i2)

    if not larger_than:
        return grouped_decls

    # Use a topological sort to sort decls according to the partial order.
    sorted_deps = [(i, decl) for i, decl in enumerate(grouped_decls)
                   if i not in larger_than]
    for i, decl in sorted_deps:
        for i2 in sorted(larger_than.keys()):
            larger = larger_than[i2]
            larger.discard(i)
            if not larger:
                del larger_than[i2]
                sorted_deps.append((i2, grouped_decls[i2]))

    return [decl for i, decl in sorted_deps]


def get_python_signature(declaration, include_out):
    # Compute the Python function signature for argument parsing
    py_formal_args = []
    output_args = []
    type_args = []
    positional = True

    def get_py_formal_arg(arg):
        typename = arg['simple_type']
        opt_match = re.match(r'optional<(.+)>', typename)
        if opt_match:
            typename = opt_match.group(1)
        typename = typename if typename != 'Type' else 'ScalarType'
        if arg.get('is_nullable') or opt_match:
            typename = '{}?'.format(typename)
        if arg.get('size') is not None:
            typename = '{}[{}]'.format(typename, arg['size'])
        param = typename + ' ' + arg['name']
        default = None
        if arg.get('default') is not None:
            default = arg['default']
            if default == 'nullptr' or default == 'nullopt' or default == '{}':
                default = 'None'
        if arg.get('python_default_init') is not None:
            default = 'None'
        if default is None and arg.get('is_type_dispatched', False):
            # this is necessary because ATen does not have default_types; in this case,
            # the type exists in the public API (at:: namespace), but not in the type interface;
            # to match the PyTorch default_type API, we set the default to None.
            default = get_type_default(declaration)
        if default is not None:
            param += '=' + str(default)
        return param

    for arg in declaration['arguments']:
        if arg.get('output', False):
            output_args.append(arg)
            continue
        if arg['simple_type'] == 'Type':
            type_args.append(arg)
            continue
        if arg.get('kwarg_only', False) and positional:
            py_formal_args.append('*')
            positional = False
        param = get_py_formal_arg(arg)
        py_formal_args.append(param)

    # add output arguments
    name = declaration['name']
    if name.endswith('_out'):
        name = name[:-4]

    if len(output_args) > 0 and include_out:
        assert declaration['name'].endswith('_out')
        if positional:
            py_formal_args.append('*')
            positional = False
        typenames = [arg['simple_type'] for arg in output_args]
        if len(typenames) > 1:
            typename = 'TensorList[{}]'.format(len(typenames))
        else:
            typename = typenames[0]
        py_formal_args.append(typename + ' out=None')

    # we could put this in the loop above but we want to ensure both type dispatched args
    # and python binding arguments are after the out argument; this matches the case
    # where there is a python binding argument dtype, which is necessary to match
    # the function signatures between the out and non-out variant.
    assert len(type_args) <= 1
    for arg in type_args:
        if positional:  # assume type_args should be kwarg_only.
            py_formal_args.append('*')
            positional = False
        py_formal_args.append(get_py_formal_arg(arg))

    if len(declaration['python_binding_arguments']) > 0:
        for arg in declaration['python_binding_arguments']:
            if arg.get('kwarg_only', False) and positional:
                py_formal_args.append('*')
                positional = False
            py_formal_args.append(get_py_formal_arg(arg))

    # Python function signature.
    # This is the string that we give to FunctionParameter, which is
    # then parsed into the actual structure which we do parsing
    # with.
    return PYTHON_FUNCTION_SIGNATURE.substitute(name=name, py_formal_args=py_formal_args)
