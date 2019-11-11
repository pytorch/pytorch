# Generates Python bindings for ATen functions
#
# The bindings are generated as methods on python_variable or functions on the
# torch._C._nn object.
#
from collections import defaultdict
import re
from .nested_dict import nested_dict
from .gen_variable_type import should_trace
from .utils import write

try:
    from src.ATen.code_template import CodeTemplate
except ImportError:
    from tools.shared.module_loader import import_module
    CodeTemplate = import_module('code_template', 'aten/src/ATen/code_template.py').CodeTemplate

# These functions require manual Python bindings or are not exposed to Python
SKIP_PYTHON_BINDINGS = [
    'alias', 'contiguous', 'is_cuda', 'is_sparse', 'size', 'stride',
    '.*_backward', '.*_backward_(out|input|weight|bias)', '.*_forward',
    '.*_forward_out', '_unsafe_view', 'tensor', '_?sparse_coo_tensor.*',
    '_arange.*', '_range.*', '_linspace.*', '_logspace.*',
    '_sparse_add_out', '_sparse_div.*', '_sparse_mul.*', '_sparse_sub.*', '_sparse_dense_add_out',
    'index', 'unique_dim_consecutive',
    '_indexCopy_', 'max_values', 'min_values',
    '_cumsum.*', '_cumprod.*', '_sum.*', '_prod.*',
    '_th_.*', '_thnn_.*',
    'arange.*', 'range.*', '_solve.*', '_inverse.*',
    '_cholesky.*', '_triangular_solve.*', '_qr.*', '_symeig.*', '_svd.*',
    'slice', 'randint(_out)?',
    'item', '_local_scalar_dense', 'to',
    'copy_sparse_to_sparse_', 'copy_',
    'numpy_T',  # this needs to be an attribute in Python, not a function
    'nonzero(_(out|numpy))?',
    'set_quantizer_',  # return types not supported yet
    'set_data',
    '.*_overrideable',  # overrideable functions for backend extension
    'data', 'is_leaf', 'output_nr', '_version', 'requires_grad_'
]

# Python binary operator dunder methods
BINARY_OP_NAMES = [
    '__lt__', '__le__',
    '__gt__', '__ge__',
    '__eq__', '__ne__',

    '__add__', '__radd__', '__iadd__',
    '__sub__', '__rsub__', '__isub__',
    '__mul__', '__rmul__', '__imul__',
    '__matmul__', '__rmatmul__', '__imatmul__',
    '__truediv__', '__rtruediv__', '__itruediv__',
    '__floordiv__', '__rfloordiv__', '__ifloordiv__',
    '__mod__', '__rmod__', '__imod__',
    '__divmod__', '__rdivmod__', '__idivmod__',
    '__pow__', '__rpow__', '__ipow__',
    '__lshift__', '__rlshift__', '__ilshift__',
    '__rshift__', '__rrshift__', '__irshift__',
    '__and__', '__rand__', '__iand__',
    '__xor__', '__rxor__', '__ixor__',
    '__or__', '__ror__', '__ior__',
]

PY_VARIABLE_METHOD_VARARGS = CodeTemplate("""\
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    ${signatures}
  }, /*traceable=*/${traceable});
  ${unpack_self}
  ParsedArgs<${max_args}> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  ${declare_namedtuple_return_types}
  ${dispatch}
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
""")

PY_VARIABLE_METHOD_NOARGS = CodeTemplate("""\
static PyObject * ${pycname}(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  ${declare_namedtuple_return_types}
  ${unpack_self}
  return wrap(${namedtuple_return_type}${dispatch_name}(${actuals}));
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
${call_dispatch}.set_requires_grad(${requires_grad})""")

PY_VARIABLE_WRAP = CodeTemplate("""\
return wrap(${namedtuple_return_type}${call_dispatch});""")

PY_VARIABLE_DISPATCH = CodeTemplate("""\
inline ${simple_return_type} ${dispatch_name}(${formal_args}) {
  ${initialize_cuda}
  ${AutoNoGIL}
  return ${dispatch_call}(${dispatch_args});
}
""")

PY_VARIABLE_METHOD_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)${pycfunc_voidcast}${pycname}, ${flags}, NULL},""")

PY_VARIABLE_METHOD_BINOP_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)${pycfunc_voidcast}TypeError_to_NotImplemented_<${pycname}>, ${flags}, NULL},""")

PY_RETURN_NAMEDTUPLE_DEF = CodeTemplate("""\
static PyStructSequence_Field fields${namedtuple_type_index}[] = {
  ${namedtuple_fields} {nullptr}
};
static PyStructSequence_Desc desc${namedtuple_type_index} = {
  "torch.return_types.${name}", nullptr,
  fields${namedtuple_type_index}, ${namedtuple_size}
};
static PyTypeObject type${namedtuple_type_index};
static bool namedtuple_type_initialized${namedtuple_type_index} = false;
if (!namedtuple_type_initialized${namedtuple_type_index}) {
  PyStructSequence_InitType(&type${namedtuple_type_index}, &desc${namedtuple_type_index});
  type${namedtuple_type_index}.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  namedtuple_type_initialized${namedtuple_type_index} = true;
}
""")

UNPACK_SELF = "auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;"

PYTHON_FUNCTION_SIGNATURE = CodeTemplate("""\
${name}(${py_formal_args})""")

# XXX: if you got here because of an assertion failure, it doesn't mean
# it's enough to just extend the list here. Before you do this, make sure
# to add an appropriate wrap() overload in torch/csrc/autograd/utils/wrap_outputs.h.
SUPPORTED_RETURN_TYPES = {
    'Tensor',
    'std::tuple<Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>',
    'std::tuple<Tensor,Tensor,Tensor,int64_t>',
    'std::tuple<Tensor,Tensor,double,int64_t>',
    'std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t>',
    'std::tuple<Tensor,Tensor,double,Tensor,int64_t>',
    'std::vector<Tensor>',
    'Scalar', 'bool', 'int64_t', 'void*', 'void',
    'QScheme', 'double',
    'IntArrayRef',
    'ScalarType'
}

TENSOR_OPTIONS = CodeTemplate("""\
const auto options = TensorOptions()
    .dtype(${dtype})
    .device(${device})
    .layout(${layout}.layout)
    .requires_grad(${requires_grad})
    .pinned_memory(${pin_memory});
""")

def should_generate_python_binding(declaration):
    name = declaration['name']
    for pattern in SKIP_PYTHON_BINDINGS:
        if re.match('^' + pattern + '$', name):
            return False

    simple_types = [arg['simple_type'] for arg in declaration['arguments']]
    signature = '{}({})'.format(name, ', '.join(simple_types))
    return True


def get_py_variable_methods(declarations):
    """
    Get declarations (grouped by name) which should be generated
    as methods on Tensor.
    """
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                declaration['mode'] != 'NN' and
                declaration.get('python_module') != 'nn' and
                'Tensor' in declaration['method_of'])

    return group_declarations_by_name(declarations, should_bind)


def gen_py_variable_methods(out, declarations, template_path):
    PY_VARIABLE_METHODS_CPP = CodeTemplate.from_file(template_path + '/python_variable_methods.cpp')
    PY_VARIABLE_DISPATCH_H = CodeTemplate.from_file(template_path + '/python_variable_methods_dispatch.h')

    py_variable_methods = get_py_variable_methods(declarations)

    env = create_python_bindings(py_variable_methods, True)
    write(out, 'python_variable_methods.cpp', PY_VARIABLE_METHODS_CPP, env)
    write(out, 'python_variable_methods_dispatch.h', PY_VARIABLE_DISPATCH_H, env)


def get_py_nn_functions(declarations):
    """
    Get declarations (grouped by name) which should be generated
    as functions in the "nn" module.
    """
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                (declaration['mode'] == 'NN' or declaration.get('python_module') == 'nn'))

    return group_declarations_by_name(declarations, should_bind)


def gen_py_nn_functions(out, declarations, template_path):
    PY_NN_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_nn_functions.cpp')
    PY_NN_FUNCTIONS_H = CodeTemplate.from_file(template_path + '/python_nn_functions.h')
    PY_NN_DISPATCH_H = CodeTemplate.from_file(template_path + '/python_nn_functions_dispatch.h')

    py_nn_functions = get_py_nn_functions(declarations)

    env = create_python_bindings(py_nn_functions, has_self=False, is_module=True)
    write(out, 'python_nn_functions.cpp', PY_NN_FUNCTIONS_CPP, env)
    write(out, 'python_nn_functions.h', PY_NN_FUNCTIONS_H, env)
    write(out, 'python_nn_functions_dispatch.h', PY_NN_DISPATCH_H, env)


def get_py_torch_functions(declarations):
    """
    Get declarations (grouped by name) which should be generated
    as functions in the "torch" module.
    """
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                declaration['mode'] != 'NN' and
                declaration.get('python_module') != 'nn' and
                'namespace' in declaration['method_of'])

    return group_declarations_by_name(declarations, should_bind)


def gen_py_torch_functions(out, declarations, template_path):
    PY_TORCH_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_torch_functions.cpp')
    PY_TORCH_DISPATCH_H = CodeTemplate.from_file(template_path + '/python_torch_functions_dispatch.h')

    py_torch_functions = get_py_torch_functions(declarations)

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
    if declaration['name'].startswith('randperm') or \
            declaration['name'] == 'tril_indices' or \
            declaration['name'] == 'triu_indices':
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
        'Tensor &': 'tensor',
        'Generator *': 'generator',
        'Storage &': 'storage',
        'const Type &': 'scalartype',
        'const THPLayout &': 'layout',
        'const Device &': 'device',
        'c10::optional<DimnameList>': 'toDimnameListOptional',
        'c10::optional<ScalarType>': 'scalartypeOptional',
        'c10::optional<MemoryFormat>': 'memoryformatOptional',
        'c10::optional<Scalar>': 'scalarOptional',
        'c10::optional<int64_t>': 'toInt64Optional',
        'c10::optional<bool>': 'toBoolOptional',
        'IntArrayRef': 'intlist',
        'int64_t': 'toInt64',
        'bool': 'toBool',
        'double': 'toDouble',
        'std::string': 'string',
    }

    unpack_with_default_methods = {
        'IntArrayRef': 'setDefaultIntlist',
        'Scalar': 'scalarWithDefault',
        'int64_t': 'toInt64WithDefault',
        'bool': 'setDefaultBool',
        'double': 'setDefaultDouble',
        'const Type &': 'scalartypeWithDefault',
        'const THPLayout &': 'layoutWithDefault',
        'const Device &': 'deviceWithDefault',
        'ScalarType': 'scalartypeWithDefault',
    }

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

        has_tensor_options = any(arg['simple_type'] == 'TensorOptions' for arg in declaration['arguments'])

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

        def unpack_variable(name, unpack_expr, typename):
            # optional<ArrayRef<T>> are special. The PythonArgParser returns an
            # optional<vector<T>>, which cannot be implictly converted to
            # optional<ArrayRef<T>>. One needs to unwrap the optional and rewrap.
            if typename == 'c10::optional<DimnameList>':
                result = """\
                    auto __{name} = {expr};
                    c10::optional<{typ}> {name} = __{name} ? c10::make_optional({typ}(__{name}.value())) : c10::nullopt;
                """.format(name=name, expr=unpack_expr, typ='DimnameList')
                return [line.strip() for line in result.split('\n')]

            return ['auto {} = {};'.format(name, unpack_expr)]

        def parse_arg(arg, arg_index, unpack_args=False):
            name = arg['name']
            typename = arg['type']
            if typename.startswith('IntArrayRef['):
                typename = 'IntArrayRef'
            if typename.startswith('LongTensor'):
                typename = 'Tensor'
            if typename == 'c10::optional<DimnameList>':
                unpack_args = True

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
                body.extend(unpack_variable(name, expr, typename))
                expr = name

            dispatch_type = typename
            if dispatch_type == 'Tensor':
                dispatch_type = 'const Tensor &'
            elif dispatch_type == 'Tensor &':
                dispatch_type = 'Tensor'
            elif dispatch_type == 'const Device &':
                dispatch_type = 'c10::optional<int32_t>'
            formal = '{} {}'.format(dispatch_type, name)
            return expr, formal

        def append_actuals_formals(actual, formal):
            actuals.append(actual)
            formal_args.append(formal)

        # We always want to unpack when we have TensorOptions.
        unpack = has_tensor_options
        for arg in inputs:
            if arg['simple_type'] in ['Type', 'TensorOptions']:
                continue
            if has_self and arg['name'] == 'self':
                formal_args.append('Tensor & self')
                actuals.append('self')
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
        parsed_type_args = None
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
            if not has_tensor_options:
                arg_idx += 1

        if 'layout' in (a['name'] for a in python_binding_arguments):
            layout_idx, device_idx, pin_memory_idx, requires_grad_idx = (arg_idx, arg_idx + 1, arg_idx + 2, arg_idx + 3)
        else:
            device_idx, pin_memory_idx, requires_grad_idx = (arg_idx, arg_idx + 1, arg_idx + 2)

        device = None
        for arg in python_binding_arguments:
            if arg['name'] == 'dtype' and arg['simple_type'] == 'Type':
                pass  # already handled by type_dispatched_args
            elif arg['name'] == 'layout' and arg['simple_type'] == 'Layout':
                # out(s) determines the type and layout if it is present, so only use this if there are no outputs.
                if len(outputs) == 0:
                    layout = parse_arg(arg, layout_idx)[0]
            elif arg['name'] == 'device' and arg['simple_type'] == 'Device':
                if len(outputs) == 0:
                    assert parsed_type_args
                    assert layout
                    device, device_type = parse_arg(arg, device_idx, True)

                    if not has_tensor_options:
                        # add type, device formals and corresponding actuals.
                        # The type actual is the ATen type mapped from (ScalarType, Layout, Device)
                        # The device actual is the corresponding AutoGPU index for the Device.
                        formal_args.append(parsed_type_args[1])
                        formal_args.append(device_type)
                        actuals.append("torch::getVariableType({}, {}, {})".format(parsed_type_args[0], layout, device))
                        actuals.append('{}.index()'.format(device))

                    has_device_bind = True
            elif arg['name'] == 'requires_grad' and arg['simple_type'] == 'bool':
                requires_grad = parse_arg(arg, requires_grad_idx)[0]
            elif arg['name'] == 'pin_memory' and arg['simple_type'] == 'bool':
                pin_memory = parse_arg(arg, pin_memory_idx)[0]
            else:
                raise RuntimeError(("found {} in python_binding_arguments but only "
                                    "\"bool pin_memory\", \"bool requires_grad\", \"ScalarType dtype\", \"Layout layout\", "
                                    "\"Device device\" are supported".format(arg)))

        dtype = parsed_type_args[0] if parsed_type_args else None
        if has_tensor_options and all([dtype, device, layout, requires_grad]):
            body.append(TENSOR_OPTIONS.substitute({
                'dtype': dtype,
                'layout': layout,
                'device': device,
                'requires_grad': requires_grad,
                'pin_memory': pin_memory,
            }))
            formal_args.append('const TensorOptions & options')
            actuals.append('options')

        env['unpack_args'] = []
        env['formal_args'] = formal_args
        env['actuals'] = actuals

        if has_tensor_options:
            env['initialize_cuda'] = 'torch::utils::maybe_initialize_cuda(options);'
        else:
            env['initialize_cuda'] = ''

        if 'call_args' in declaration:
            env['dispatch_args'] = declaration['call_args']
        else:
            env['dispatch_args'] = [arg['name'] for arg in declaration['arguments']]

        if 'Tensor' in declaration['method_of']:
            env['dispatch_args'] = [arg for arg in env['dispatch_args'] if arg != 'self']
            env['dispatch_call'] = 'self.{}'.format(declaration['name'])
        elif 'namespace' in declaration['method_of']:
            namespace = 'torch' if (has_tensor_options or declaration['name'].endswith('_like')) else 'at'
            env['dispatch_call'] = '{}::{}'.format(namespace, declaration['name'])
        else:
            raise RuntimeError('could not dispatch, neither namespace function nor Tensor method')

        env['AutoNoGIL'] = 'AutoNoGIL no_gil;' if not declaration['with_gil'] else ''

        # Use the simple_return_type (Tensor) rather than the fancy return type
        # (Tensor &).  This is important because the dispatch functions take
        # mutable arguments *by value*, not by reference.  If you then return
        # a a reference to such an argument, you will now have a pointer to a
        # dangling stack entry.  Not good.
        #
        # You want:
        #
        #   Tensor dispatch_selu_(Tensor self) { return at::selu_(self); }
        #
        # *not*
        #
        #   Tensor& dispatch_selu_(Tensor self) { return at::selu_(self); }
        #
        # (NB: We can't make dispatch_selu_ take Tensor&, because the enclosing
        # codegen looks like dispatch_selu_(wrap(tensor)), and you can't take a
        # mutable reference to temporary.  Maybe we could assign it to a
        # variable itself.)
        env['simple_return_type'] = simple_return_type

        env = nested_dict(env, nested_dict(base_env, declaration))
        call_dispatch = PY_VARIABLE_CALL_DISPATCH.substitute(env)
        if requires_grad and not has_tensor_options:
            call_dispatch = PY_VARIABLE_SET_REQUIRES_GRAD.substitute(env, call_dispatch=call_dispatch,
                                                                     requires_grad=requires_grad)
        if simple_return_type == 'void':
            body.append('{call_dispatch};'.format(call_dispatch=call_dispatch))
            body.append('Py_RETURN_NONE;')
        else:
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

            has_dtype_bind = 'dtype' in (d['name'] for d in dictionary['out'].get('python_binding_arguments', []))
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
        has_options_arg = False
        for arg in declaration['arguments']:
            if arg.get('output', False):
                continue
            typename = arg['simple_type']
            if typename in ['Tensor', 'TensorList']:
                has_tensor_input_arg = True
            if arg['simple_type'] == 'Type':
                has_type_input_arg = True
            elif arg['simple_type'] == 'TensorOptions':
                has_options_arg = True
            if arg['name'] == 'requires_grad':
                raise ValueError("argument named requires_grad not supported")

        has_tensor_return = False
        for ret in declaration['returns']:
            if ret['dynamic_type'] in ['Tensor', 'TensorList']:
                # this probably won't work if one of the returns is not a tensor, but it will
                # produce a compile-time error that is obvious
                has_tensor_return = True

        category_override = declaration['category_override']
        is_like_function = name.endswith('_like') or category_override == 'like'
        is_like_function_with_options = is_like_function and has_options_arg
        is_new_function = name.startswith('new_') or category_override == 'new'
        is_new_function_with_options = is_new_function and has_options_arg
        is_factory_function = has_tensor_return and not has_tensor_input_arg or category_override == 'factory'
        is_factory_or_like_or_new_function = has_tensor_return and (is_factory_function or is_like_function or is_new_function)
        is_like_or_new_function_with_options = is_like_function_with_options or is_new_function_with_options

        if (is_factory_function and not has_type_input_arg) or has_options_arg:
            default_type = get_type_default(declaration)
            py_default_dtype = 'self.scalar_type()' if is_like_or_new_function_with_options else None
            dtype_arg = {
                'default': default_type,
                'dynamic_type': 'Type',
                'kwarg_only': True,
                'name': 'dtype',
                'type': 'const Type &',
                'simple_type': 'Type',
                'python_default_init': py_default_dtype,
            }
            python_binding_arguments.append(dtype_arg)
        if is_factory_function or is_like_or_new_function_with_options:
            py_default_layout = '*torch::getLayout(self.type().backend())' if is_like_or_new_function_with_options else None
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
            py_default_device = 'self.device()' if is_like_or_new_function_with_options else None
            device_arg = {
                'default': 'None',
                'dynamic_type': 'Device',
                'kwarg_only': True,
                'name': 'device',
                'type': 'const Device &',
                'simple_type': 'Device',
                'python_default_init': py_default_device
            }
            python_binding_arguments.append(device_arg)
            pin_memory_arg = {
                'default': False,
                'dynamic_type': 'bool',
                'kwarg_only': True,
                'name': 'pin_memory',
                'type': 'bool',
                'simple_type': 'bool',
            }
            python_binding_arguments.append(pin_memory_arg)
        if is_factory_or_like_or_new_function:
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

    def emit_namedtuple_return_type_def(declaration, next_index):
        returns = declaration['returns']
        if len(returns) <= 1 or all(['field_name' not in x for x in returns]):
            declaration['namedtuple_return_type'] = ''
            return '', next_index
        declaration['namedtuple_type_index'] = next_index
        declaration['namedtuple_fields'] = ''
        for x in returns:
            # See Note [field_name versus name]
            if 'field_name' not in x:
                # When building on Windows, `PyStructSequence_UnnamedField` could not be
                # resolved by the linker for some reason, which cause error in building:
                #
                # python_nn_functions.cpp.obj : error LNK2001: unresolved external symbol
                # PyStructSequence_UnnamedField
                #
                # Thus, at this point in time, we do not support unnamed
                # fields in namedtuple; you must either name all fields,
                # or none of them.
                raise ValueError("Unnamed field is not supported by codegen")
            else:
                declaration['namedtuple_fields'] += '{"' + x['field_name'] + '", ""}, '
        declaration['namedtuple_size'] = len(returns)
        declaration['namedtuple_return_type'] = '&type{}, '.format(next_index)
        return PY_RETURN_NAMEDTUPLE_DEF.substitute(declaration), next_index + 1

    def process_function(name, declarations):
        for declaration in declarations:
            declaration['python_binding_arguments'] = get_python_binding_arguments(declaration)

        env = {
            'name': name,
            'dispatch_name': 'dispatch_{}'.format(name),
            'pycname': 'THPVariable_{}'.format(name),
            'pycfunc_voidcast': '',
            'signatures': [],
            'max_args': max(len(o['arguments']) + len(o['python_binding_arguments']) for o in declarations),
            'unpack_self': [],
            'dispatch': [],
            'declare_namedtuple_return_types': '',
        }

        if has_self:
            env['unpack_self'] = [UNPACK_SELF]

        # generate namedtuple type declare
        next_index = 0
        for declaration in declarations:
            typedef, next_index = emit_namedtuple_return_type_def(declaration, next_index)
            env['declare_namedtuple_return_types'] += typedef

        # emit dispatch
        grouped = group_declarations(declarations)
        for i, dictionary in enumerate(grouped):
            signature = dictionary['signature']
            if has_self:
                signature = signature.replace('Tensor self, ', '')
                signature = signature.replace('Tensor self', '')
            if not has_self:
                # Use 'input' instead of 'self' for NN functions
                signature = signature.replace('Tensor self', 'Tensor input')
            if dictionary['base'].get('deprecated', False):
                signature += '|deprecated'
            env['signatures'].append('"{}",'.format(signature))
            env['dispatch'].append(emit_dispatch(i, dictionary, env))

        env['dispatch'].append('}')

        env['traceable'] = 'true' if all(should_trace(d) for d in declarations) else 'false'

        if len(declarations) == 1 and len(declarations[0]['args']) == 1 and has_self:
            tmpl = PY_VARIABLE_METHOD_NOARGS
            env['actuals'] = ['self']
            env['flags'] = 'METH_NOARGS'
            env['namedtuple_return_type'] = declarations[0]['namedtuple_return_type']
        else:
            tmpl = PY_VARIABLE_METHOD_VARARGS
            env['flags'] = 'METH_VARARGS | METH_KEYWORDS'
            env['pycfunc_voidcast'] = '(void(*)(void))'

        if not is_module and not has_self:
            env['flags'] += ' | METH_STATIC'

        py_methods.append(tmpl.substitute(env))
        if name in BINARY_OP_NAMES:
            py_method_defs.append(PY_VARIABLE_METHOD_BINOP_DEF.substitute(env))
        else:
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
    for x, dictionary in sorted(grouped.items()):
        if 'base' not in dictionary:
            raise RuntimeError("'base' not in dictionary for " + str(x), dictionary)
        result.append(dictionary)
    return sort_declarations(result)


# This function declares a partial order on declarations, and sorts them according
# to its linear extension. This is necessary, because there's some ambiguity in the
# choice of overload, and we want a different order.
#
# See Note[Order of overloads matters]
def sort_declarations(grouped_decls):

    # TODO: This is a hack!
    #
    # For some reason, when you specify a Scalar argument in a native
    # function, you get a Declarations.yaml entry that looks like this:
    #
    #   - default: 1
    #     dynamic_type: Scalar
    #     is_nullable: false
    #     kwarg_only: true
    #     name: alpha
    #     type: Scalar
    #
    # This is contrast to when there is a 'real' argument in TH
    # Declarations.cwrap; this gets (correctly?) translated into
    # dynamic_type: real, and type: Scalar.  I would like to fix this
    # at the source but I have never understood what dynamic_type is
    # supposed to be.
    def normalized_dynamic_type(arg):
        if arg['dynamic_type'] == 'real':
            return 'Scalar'
        return arg['dynamic_type']

    def is_coord_smaller(arg1, arg2):
        return normalized_dynamic_type(arg1) == 'Scalar' and arg2['dynamic_type'] == 'Tensor'

    def is_smaller(d1, d2):
        """Returns True if d1 < d2 in the partial order."""
        args1, args2 = d1['base']['arguments'], d2['base']['arguments']
        if len(args1) != len(args2):
            return False
        any_smaller = any(is_coord_smaller(arg1, arg2) for arg1, arg2 in zip(args1, args2))
        all_smaller_or_equal = all(normalized_dynamic_type(arg1) == normalized_dynamic_type(arg2) or
                                   is_coord_smaller(arg1, arg2)
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
    # Compute the Python function signature for argument parsing,
    # as specified in torch/csrc/utils/python_arg_parser.h.  WARNING:
    # this is NOT the same type signature as specified by PEP 484
    # as understood by mypy; our format was independently developed
    # and has some quirks to make it more suitable specifically
    # for error parsing.
    #
    # For a translation to mypy-valid type signatures, see
    # tools/gen_pyi.py.  If you change any logic here, please
    # check that file too.
    py_formal_args = []
    output_args = []
    type_args = []
    positional = True

    def get_py_formal_arg(arg):
        typename = arg['simple_type']
        typename = typename if typename != 'Type' else 'ScalarType'

        # TODO: remove this and make optional types in simple_type to be consistent across
        # tensor and other types after make Tensor? be optional instead of undefined
        if arg.get('is_nullable') and '?' not in typename:
            typename = '{}?'.format(typename)

        if arg.get('size') is not None:
            typename = '{}[{}]'.format(typename, arg['size'])
        param = typename + ' ' + arg['name']
        default = None
        if arg.get('default') is not None:
            default = arg['default']
            if default == 'nullptr' or default == 'c10::nullopt' or default == '{}':
                default = 'None'
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
        # Skip `TensorOptions` in Python, as it is only used on the C++ side.
        if arg['simple_type'] == 'TensorOptions':
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
        if len(output_args) == 1:
            # The nn module bindings are often not exposed to the user directly
            # but via torch.nn modules and functionals.
            py_formal_args.append(typename + ' ' + output_args[0]['name'] + '=None')
        else:
            # NB: For more than 1 output args the type name is a TensorList
            # and as such we don't (yet) need to consider the naming.
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
