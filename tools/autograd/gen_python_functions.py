# Generates Python bindings for ATen functions
#
# The bindings are generated as methods on python_variable or functions on the
# torch._C._nn object.
#
from collections import defaultdict, namedtuple
import re
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

# These function signatures are not exposed to Python. Note that this signature
# list does not support regex.
SKIP_PYTHON_BINDINGS_SIGNATURES = [
    'add(Tensor, Scalar, Scalar)', 'add_(Tensor, Scalar, Scalar)',
    'sub(Tensor, Scalar, Scalar)', 'sub_(Tensor, Scalar, Scalar)',
    'mul(Tensor, Scalar)', 'mul_(Tensor, Scalar)',
    'div(Tensor, Scalar)', 'div_(Tensor, Scalar)',
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

# python binding for all overloads of a particular function/method
PY_VARIABLE_METHOD_VARARGS = CodeTemplate("""\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  ${namedtuple_typedef}
  ${unpack_self}
  static PythonArgParser parser({
    ${signatures}
  }, /*traceable=*/${traceable});
  ParsedArgs<${max_args}> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  switch (r.idx) {
    ${dispatch}
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

""")

# python binding for singe-overload function/method
PY_VARIABLE_METHOD_VARARGS_SINGLETON = CodeTemplate("""\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  ${namedtuple_typedef}
  ${unpack_self}
  static PythonArgParser parser({
    ${signatures}
  }, /*traceable=*/${traceable});
  ParsedArgs<${max_args}> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  ${dispatch}

  END_HANDLE_TH_ERRORS
}

""")

# python binding for a method with no args, shortcuts parsing
PY_VARIABLE_METHOD_NOARGS = CodeTemplate("""\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  ${namedtuple_typedef}
  ${unpack_self}
  ${dispatch}
  END_HANDLE_TH_ERRORS
}

""")

# handler for a single parsed signature - may be a single overload or
# a pair of overloads that whose signatures only differ in output params
PY_VARIABLE_CASE = CodeTemplate("""\
case ${i}: {
  ${call_dispatch}
}

""")

PY_VARIABLE_SINGLE_CASE = CodeTemplate("""\
${call_dispatch}
""")

# handler for output/no-output overload pair
# (plugged into PY_VARIABLE_CASE as ${call_dispatch})
PY_VARIABLE_OUT = CodeTemplate("""\
if (r.isNone(${out_idx})) {
  ${call_dispatch}
}
else {
  ${call_dispatch_out}
}

""")

# variation of output/no-output handler in which tensor options params
# (if present) are checked against properties of a tensor output param
PY_VARIABLE_OUT_CHECK_TYPE = CodeTemplate("""\
if (r.isNone(${out_idx})) {
  ${call_dispatch}
}
else {
  check_out_type_matches(r.tensor(${out_idx}), r.scalartype(${type_idx}), r.isNone(${type_idx}),
                         r.layout(${layout_idx}), r.isNone(${layout_idx}),
                         r.device(${device_idx}), r.isNone(${device_idx}));
  ${call_dispatch_out}
}

""")

PY_VARIABLE_CALL_DISPATCH = CodeTemplate("""\
${dispatch_call}(${dispatch_args})""")

# Unpack parsed args to locals, call the op, and wrap the result.
# Lambda is so GIL is back on by wrap() time (wrap can allocate)
PY_VARIABLE_WRAP = CodeTemplate("""\
// ${schema_string}
${inits}
auto dispatch = [](${lambda_params}) {
  ${auto_no_gil}
  return ${dispatch_call}(${dispatch_args});
};
return wrap(${namedtuple_typeref}dispatch(${lambda_args})${set_requires_grad});
""")

PY_VARIABLE_RETURN_VOID = CodeTemplate("""\
// ${schema_string}
${inits}
${auto_no_gil}
${dispatch_call}(${dispatch_args});
Py_RETURN_NONE;
""")

# PyMethodDef entry
PY_VARIABLE_METHOD_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)${pycfunc_voidcast}${pycname}, ${flags}, NULL},""")

# PyMethodDef entry for binary op, throws not implemented error
PY_VARIABLE_METHOD_BINOP_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)${pycfunc_voidcast}TypeError_to_NotImplemented_<${pycname}>, ${flags}, NULL},""")

# named tuple PyTypeObject
PY_NAMEDTUPLE_TYPEDEF = CodeTemplate("""\
static PyTypeObject ${typename};
static bool ${typename}_initialized = false;
if (!${typename}_initialized) {
  ${typename}_initialized = true;
  static PyStructSequence_Field fields[] = { ${fields,} {nullptr} };
  static PyStructSequence_Desc desc = { "torch.return_types.${name}", nullptr, fields, ${size} };
  PyStructSequence_InitType(&${typename}, &desc);
  ${typename}.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
}
""")

# NOTE: we type the unpacked self as Tensor not Variable to avoid return type
# discrepancies on method resolution (e.g. Variable::detach_ returns void
# rather than Tensor &)
UNPACK_SELF = "Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;"

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
    for pattern in SKIP_PYTHON_BINDINGS_SIGNATURES:
        if pattern == signature:
            return False

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

    py_variable_methods = get_py_variable_methods(declarations)

    env = create_python_bindings(py_variable_methods, True)
    write(out, 'python_variable_methods.cpp', PY_VARIABLE_METHODS_CPP, env)


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

    py_nn_functions = get_py_nn_functions(declarations)

    env = create_python_bindings(py_nn_functions, has_self=False, is_module=True)
    write(out, 'python_nn_functions.cpp', PY_NN_FUNCTIONS_CPP, env)
    write(out, 'python_nn_functions.h', PY_NN_FUNCTIONS_H, env)


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

    py_torch_functions = get_py_torch_functions(declarations)

    env = create_python_bindings(py_torch_functions, has_self=False)
    write(out, 'python_torch_functions.cpp', PY_TORCH_FUNCTIONS_CPP, env)


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
        """
        Emit dispatch code for a single declared overload.

        - declaration: the single op overload to generate code for.

        - out_idx:     some overloads come in pairs, with signatures that match
                       except for optional output params. For these pairs, the
                       generated binding uses a single signature. out_idx is the
                       index of the first output argument in the signature, whether
                       we're emitting the output or no-output sibling. For singleton
                       no-output overloads, out_idx is None. (Singleton overloads with
                       output are prohibited.)

        - base_env:    dictionary with stuff that applies to all overloads
        """

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

        def parse_arg(arg, arg_index, unpack_to_local=False):
            """
            generate code to dereference a parsed arg
            - arg: a dict with the arg's properties
            - arg_index: the arg's position in the python signature
            - unpack_to_local: if true, we generate a local initializer named after the arg
              and add it to `inits`
            returns a pair:
            - a c++ param declaration, using the original arg name
              (TODO hoist/eliminate the inside baseball type conversions here)
            - an expression producing the arg value. if unpack_to_local is true, this will be
              the local's name (i.e., the original arg name), otherwise it'll be an inline
              expression like `r.tensor(0)`
            """
            name = arg['name']

            typename = arg['type']
            if typename.startswith('IntArrayRef['):
                typename = 'IntArrayRef'
            if typename.startswith('LongTensor'):
                typename = 'Tensor'

            if typename == 'c10::optional<DimnameList>':
                unpack_to_local = True

            if arg.get('python_default_init'):
                assert typename in unpack_with_default_methods, \
                    '`{}` type is not supported in python_default_init'.format(typename)
                unpack_with_default = unpack_with_default_methods.get(typename)
                default_expr = arg.get('python_default_init')
                expr = 'r.{}({}, {})'.format(unpack_with_default, arg_index, default_expr)
            else:
                unpack = unpack_methods.get(typename, typename.lower())
                expr = 'r.{}({})'.format(unpack, arg_index)

            if unpack_to_local:
                # if asked to unpack, add a local and return a reference to that
                inits.extend(unpack_variable(name, expr, typename))
                expr = name

            dispatch_type = typename
            if dispatch_type == 'Tensor':
                dispatch_type = 'const Tensor &'
            elif dispatch_type == 'Tensor &':
                dispatch_type = 'Tensor'
            elif dispatch_type == 'const Device &':
                dispatch_type = 'c10::optional<int32_t>'
            param_decl = '{} {}'.format(dispatch_type, name)

            return param_decl, expr

        pa = declaration['partitioned_args']
        inputs = pa.input_args + pa.input_kwargs + pa.type_args
        outputs = pa.output_args
        python_binding_args = pa.python_binding_args

        inits = []
        actuals = []
        params = []
        arg_idx = 0

        for arg in inputs:
            if arg['simple_type'] in ['Type', 'TensorOptions']:
                pass
            elif has_self and arg['name'] == 'self':
                # we'll have unpacked this already
                params.append('Tensor & self')
                actuals.append('self')
            else:
                # unpack self to a local because logic elsewhere in here assumes a local by that name
                unpack_to_local = arg['name'] == 'self'
                param_decl, arg_expr = parse_arg(arg, arg_idx, unpack_to_local)
                params.append(param_decl)
                actuals.append(arg_expr)
                arg_idx += 1

        if len(outputs) == 1:
            param_decl, arg_expr = parse_arg(outputs[0], arg_idx)
            params.append(param_decl)
            actuals.append(arg_expr)
        elif len(outputs) > 1:
            N = len(outputs)
            inits.append('auto results = r.tensorlist_n<{}>({});'.format(N, arg_idx))
            for i, arg in enumerate(outputs):
                params.append('Tensor & {}'.format(arg['name']))
                actuals.append('results[{}]'.format(i))

        has_tensor_options = any(arg['simple_type'] == 'TensorOptions' for arg in inputs)

        # find at most one type arg to use, check error conditions
        type_binding_args = [arg for arg in python_binding_args if arg['simple_type'] == 'Type']
        assert len(pa.type_args + type_binding_args) <= 1
        if len(type_binding_args) > 0 and len(outputs) == 0:
            # out(s) determines the dtype if it is present, so only use this if there are no outputs.
            type_arg = type_binding_args[0]
        elif len(pa.type_args) > 0:
            type_arg = pa.type_args[0]
        else:
            type_arg = None
        if type_arg and len(outputs) > 1:
            raise RuntimeError(declaration['name'] + ": type dispatched parameter with multiple outputs not supported")

        layout = None
        parsed_type_arg = None
        # type args go after the outputs to match the signature generation.
        arg_idx = arg_idx if out_idx is None else out_idx + 1
        if type_arg is not None:
            parsed_type_arg = parse_arg(type_arg, arg_idx, has_tensor_options)
            arg_idx += 1

        # check python_binding_args
        #
        requires_grad = None

        if 'dtype' in (a['name'] for a in python_binding_args):
            if not has_tensor_options:
                arg_idx += 1

        if 'layout' in (a['name'] for a in python_binding_args):
            layout_idx, device_idx, pin_memory_idx, requires_grad_idx = (arg_idx, arg_idx + 1, arg_idx + 2, arg_idx + 3)
        else:
            device_idx, pin_memory_idx, requires_grad_idx = (arg_idx, arg_idx + 1, arg_idx + 2)

        device = None
        for arg in python_binding_args:
            if arg['name'] == 'dtype' and arg['simple_type'] == 'Type':
                pass  # already handled by type_dispatched_args
            elif arg['name'] == 'layout' and arg['simple_type'] == 'Layout':
                # out(s) determines the type and layout if it is present, so only use this if there are no outputs.
                if len(outputs) == 0:
                    _, layout = parse_arg(arg, layout_idx)
            elif arg['name'] == 'device' and arg['simple_type'] == 'Device':
                if len(outputs) == 0:
                    assert parsed_type_arg
                    assert layout
                    _, device = parse_arg(arg, device_idx, unpack_to_local=True)
            elif arg['name'] == 'requires_grad' and arg['simple_type'] == 'bool':
                _, requires_grad = parse_arg(arg, requires_grad_idx)
            elif arg['name'] == 'pin_memory' and arg['simple_type'] == 'bool':
                _, pin_memory = parse_arg(arg, pin_memory_idx)
            else:
                raise RuntimeError(("found {} in python_binding_args but only "
                                    "\"bool pin_memory\", \"bool requires_grad\", \"ScalarType dtype\", \"Layout layout\", "
                                    "\"Device device\" are supported".format(arg)))

        dtype = parsed_type_arg[1] if parsed_type_arg is not None else None
        if has_tensor_options and all([dtype, device, layout, requires_grad]):
            inits.append(TENSOR_OPTIONS.substitute({
                'dtype': dtype,
                'layout': layout,
                'device': device,
                'requires_grad': requires_grad,
                'pin_memory': pin_memory,
            }))
            params.append('const TensorOptions & options')
            actuals.append('options')

        if has_tensor_options:
            inits.append('torch::utils::maybe_initialize_cuda(options);')

        if 'call_args' in declaration:
            # NOTE: declaration['call_args'] is added to a deprecated op declaration by
            # load_deprecated_signatures() in gen_autograd.py, on its way from
            # deprecated.yaml to Declarations.yaml. Typically deprecated.yaml adds extra
            # constant args to fill out a smaller deprecated param list.
            dispatch_args = declaration['call_args']
        else:
            # note that outputs have been moved to front in 'arguments' (from Declarations.yaml)
            dispatch_args = [arg['name'] for arg in declaration['arguments']]

        if 'Tensor' in declaration['method_of']:
            dispatch_call = 'self.{}'.format(declaration['name'])
            dispatch_args = [arg for arg in dispatch_args if arg != 'self']
            dispatch_actuals = [arg for arg in actuals if arg != 'self']
        elif 'namespace' in declaration['method_of']:
            namespace = 'torch' if (has_tensor_options or declaration['name'].endswith('_like')) else 'at'
            dispatch_call = '{}::{}'.format(namespace, declaration['name'])
            dispatch_actuals = actuals
        else:
            raise RuntimeError('could not dispatch, neither namespace function nor Tensor method')

        auto_no_gil = ['AutoNoGIL no_gil;'] if not declaration['with_gil'] else []

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
        simple_return_type = declaration['return_type'].replace(' &', '')
        if simple_return_type not in SUPPORTED_RETURN_TYPES:
            raise RuntimeError(declaration['name'] + " returns unsupported type " + simple_return_type)

        if requires_grad and not has_tensor_options:
            set_requires_grad = '.set_requires_grad({})'.format(requires_grad)
        else:
            set_requires_grad = ''

        if declaration['namedtuple_typedef'] != '':
            inits = [declaration['namedtuple_typedef']] + inits

        if simple_return_type == 'void':
            return PY_VARIABLE_RETURN_VOID.substitute(
                auto_no_gil=auto_no_gil,
                inits=inits,
                dispatch_call=dispatch_call,
                dispatch_args=dispatch_actuals,
                schema_string=declaration['schema_string'],
            )
        else:
            return PY_VARIABLE_WRAP.substitute(
                auto_no_gil=auto_no_gil,
                inits=inits,
                simple_return_type=simple_return_type,
                namedtuple_typeref=declaration['namedtuple_typeref'],
                dispatch_call=dispatch_call,
                dispatch_args=dispatch_args,
                lambda_params=params,
                lambda_args=actuals,
                schema_string=declaration['schema_string'],
                set_requires_grad=set_requires_grad,
            )

    def emit_dispatch(i, dictionary, base_env):
        """
        Emit dispatch code for a single parsed signature. This may correspond to either one
        or two declared overloads: those that differ only in optional output params are paired
        up in the generated binding.

        - i:            this signature's position in generated binding's signature list
                        if number of signatures > 1, otherwise None

        - dictionary:   contains a no-output overload declaration under 'base', and optionally
                        a second overload with outputs under 'out'

        - base_env:     dictionary containing stuff that applies to all overloads
        """
        base_decl = dictionary['base']
        if 'out' in dictionary:
            out_decl = dictionary['out']
            pa = out_decl['partitioned_args']
            out_idx = len(pa.input_args + pa.input_kwargs + pa.type_args)

            env = {}
            env['call_dispatch_out'] = emit_single_dispatch(out_decl, out_idx, base_env)
            env['call_dispatch'] = emit_single_dispatch(base_decl, out_idx, base_env)

            has_dtype_bind = 'dtype' in (a['name'] for a in pa.python_binding_args)
            if has_dtype_bind:
                body = PY_VARIABLE_OUT_CHECK_TYPE.substitute(env, out_idx=out_idx, type_idx=out_idx + 1,
                                                             layout_idx=out_idx + 2, device_idx=out_idx + 3).split('\n')
            else:
                body = PY_VARIABLE_OUT.substitute(env, out_idx=out_idx).split('\n')
        else:
            body = emit_single_dispatch(base_decl, None, base_env)

        if i is None:
            return PY_VARIABLE_SINGLE_CASE.substitute(call_dispatch=body)
        else:
            return PY_VARIABLE_CASE.substitute(
                i=i,
                namedtuple_typedef=base_decl['namedtuple_typedef'],
                call_dispatch=body,
            )

    def emit_namedtuple_typedef(declaration):
        returns = declaration['returns']
        if len(returns) <= 1 or all(['field_name' not in x for x in returns]):
            return '', ''
        fields = []
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
                fields.append('{{"{}", ""}}'.format(x['field_name']))
        typename = 'NamedTuple'
        typedef = PY_NAMEDTUPLE_TYPEDEF.substitute(
            name=declaration['name'],
            typename=typename,
            size=len(returns),
            fields=fields,
        )
        typeref = '&{}, '.format(typename)
        return typeref, typedef

    def process_function(name, declarations):
        """
        Generate a python binding for all overloads of an op.
        """
        for declaration in declarations:
            declaration['partitioned_args'] = get_partitioned_args(declaration)
            declaration['partitioned_formals'] = get_partitioned_formals(declaration)

        def argc(decl):
            return sum([len(sublist) for sublist in decl['partitioned_formals']])

        env = {
            'name': name,
            'dispatch_name': 'dispatch_{}'.format(name),
            'pycname': 'THPVariable_{}'.format(name),
            'pycfunc_voidcast': '',
            'signatures': [],
            'max_args': max([argc(decl) for decl in declarations]),
            'unpack_self': [],
            'dispatch': [],
        }

        if has_self:
            env['unpack_self'] = [UNPACK_SELF]

        namedtuple_typedefs = [emit_namedtuple_typedef(decl) for decl in declarations]
        if len(set(namedtuple_typedefs)) == 1:
            typeref, typedef = namedtuple_typedefs[0]
            env['namedtuple_typedef'] = typedef
            for decl in declarations:
                decl['namedtuple_typeref'] = typeref
                decl['namedtuple_typedef'] = ''
        else:
            env['namedtuple_typedef'] = ''
            for ((typeref, typedef), decl) in zip(namedtuple_typedefs, declarations):
                decl['namedtuple_typeref'] = typeref
                decl['namedtuple_typedef'] = typedef

        # emit dispatch
        if len(declarations) == 1 and len(declarations[0]['args']) == (1 if has_self else 0):
            # special codegen for 0-argument calls, omits arg parse
            tmpl = PY_VARIABLE_METHOD_NOARGS
            env['flags'] = 'METH_NOARGS' if has_self else 'METH_VARARGS | METH_KEYWORDS'
            declaration = declarations[0].copy()
            env['dispatch'].append(emit_single_dispatch(declaration, 0, env))
        else:
            grouped = group_declarations(declarations)
            singleton = len(grouped) == 1

            if singleton:
                tmpl = PY_VARIABLE_METHOD_VARARGS_SINGLETON
            else:
                tmpl = PY_VARIABLE_METHOD_VARARGS

            env['flags'] = 'METH_VARARGS | METH_KEYWORDS'
            env['pycfunc_voidcast'] = '(void(*)(void))'

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
                overload_index = i if not singleton else None
                dispatch = emit_dispatch(overload_index, dictionary, env)
                env['dispatch'].append(dispatch)

        env['traceable'] = 'true' if all(should_trace(d) for d in declarations) else 'false'

        if not is_module and not has_self:
            env['flags'] += ' | METH_STATIC'

        py_methods.append(tmpl.substitute(env))
        if name in BINARY_OP_NAMES:
            py_method_defs.append(PY_VARIABLE_METHOD_BINOP_DEF.substitute(env))
        else:
            py_method_defs.append(PY_VARIABLE_METHOD_DEF.substitute(env))

    # process_function mainline
    #
    for name in sorted(python_functions.keys()):
        process_function(name, python_functions[name])

    return {
        'py_methods': py_methods,
        'py_method_defs': py_method_defs,
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

    pf = declaration['partitioned_formals']

    output_formals = pf.output_formals if include_out else []

    kw_formals = pf.input_kw_formals + output_formals + pf.type_formals + pf.tensor_opt_formals
    if kw_formals != []:
        formals = pf.input_formals + ['*'] + kw_formals
    else:
        formals = pf.input_formals

    name = declaration['name']
    if len(pf.output_formals) > 0:
        assert name.endswith("_out"), "output params, expecting name ending with '_out'"
        name = name[:-4]

    # Python function signature.
    # This is the string that we give to FunctionParameter, which is
    # then parsed into the actual structure which we do parsing
    # with.
    return PYTHON_FUNCTION_SIGNATURE.substitute(name=name, py_formal_args=formals)


PartitionedFormals = namedtuple('PartitionedFormals', [
    'input_formals',
    'input_kw_formals',
    'output_formals',
    'type_formals',
    'tensor_opt_formals'
])

def get_partitioned_formals(declaration):
    """
    Given a declaration, return partitioned args translated into formals as follows:
    - positional input formals
    - keyword input formals
    - at most one output formal - if the original has multiple outputs, they're collected
    - at most one type formal
    - tensor_opt_formals
    Carnal knowledge:
    - TensorOptions-typed formals are omitted, since we know they would be redundant
      with the scattered equivalents in tensor_opt_formals (actually, we assume there's
      at most one)
    - collection of multiple outputs into a single list
    - assumption that multiple outputs are all tensor typed
    - implicit optionality of output params
    - other type tweaks
    """
    def get_py_formal(arg, force_default=False):
        typename = arg['simple_type']
        typename = typename if typename != 'Type' else 'ScalarType'

        # TODO: remove this and make optional types in simple_type to be consistent across
        # tensor and other types after make Tensor? be optional instead of undefined
        if arg.get('is_nullable') and '?' not in typename:
            typename = '{}?'.format(typename)

        if arg.get('size') is not None:
            typename = '{}[{}]'.format(typename, arg['size'])

        param = typename + ' ' + arg['name']

        default = arg.get('default')
        if default == 'nullptr' or default == 'c10::nullopt' or default == '{}' or force_default:
            default = 'None'
        if default is not None:
            param += '=' + str(default)

        return param

    def is_tensor_opt(arg):
        return arg['simple_type'] == 'TensorOptions'

    pa = declaration['partitioned_args']

    input_formals = [get_py_formal(a) for a in pa.input_args if not is_tensor_opt(a)]
    input_kw_formals = [get_py_formal(a) for a in pa.input_kwargs if not is_tensor_opt(a)]

    if len(pa.output_args) <= 1:
        output_formals = [get_py_formal(a, force_default=True) for a in pa.output_args]
    else:
        # NB: For more than 1 output args the type name is a TensorList
        # and as such we don't (yet) need to consider the naming.
        assert all([a['simple_type'] == 'Tensor' for a in pa.output_args])
        typename = 'TensorList[{}]'.format(len(pa.output_args))
        output_formals = [typename + ' out=None']

    assert len(pa.type_args) <= 1
    type_formals = [get_py_formal(a) for a in pa.type_args]

    tensor_opt_formals = [get_py_formal(a) for a in pa.python_binding_args]

    return PartitionedFormals(
        input_formals=input_formals,
        input_kw_formals=input_kw_formals,
        output_formals=output_formals,
        type_formals=type_formals,
        tensor_opt_formals=tensor_opt_formals,
    )


PartitionedArgs = namedtuple('PartitionedArgs', [
    'input_args',
    'input_kwargs',
    'output_args',
    'type_args',
    'python_binding_args'
])

def get_partitioned_args(declaration):
    """
    Given a declaration, return a dictionary with args split into
    - positional input args
    - keyword input args
    - output args (implicitly kw)
    - type args (implicitly kw)
    - python_binding_args:
      if the declaration has a TensorOptions-typed argument, python_binding_args is a
      generated set of scattered equivalents. Note that the contents of python_binding_args
      is context dependent, e.g. it lacks a type argument if one is present in the original.
      There are other variations - see get_python_binding_arguments for more details.
    This groups args and puts them in signature order, but there's a lot of
    translation left to do - e.g. see get_partitioned_formals and get_python_signature.
    """
    input_args = []
    input_kwargs = []
    output_args = []
    type_args = []

    current_input_args = input_args
    for arg in declaration['arguments']:
        if arg.get('output', False):
            output_args.append(arg)
        elif arg['simple_type'] == 'Type':
            type_args.append(arg)
        else:
            if arg.get('kwarg_only', False):
                current_input_args = input_kwargs
            current_input_args.append(arg)

    python_binding_args = get_python_binding_arguments(declaration)

    return PartitionedArgs(
        input_args=input_args,
        input_kwargs=input_kwargs,
        output_args=output_args,
        type_args=type_args,
        python_binding_args=python_binding_args,
    )

def get_python_binding_arguments(declaration):
    """
    Given various properties of a declaration, in particular its pattern of args,
    return values and name, build a set of scattered tensor options args.
    """
    name = declaration['name']
    python_binding_args = []
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
        python_binding_args.append(dtype_arg)
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
        python_binding_args.append(layout_arg)
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
        python_binding_args.append(device_arg)
        pin_memory_arg = {
            'default': False,
            'dynamic_type': 'bool',
            'kwarg_only': True,
            'name': 'pin_memory',
            'type': 'bool',
            'simple_type': 'bool',
        }
        python_binding_args.append(pin_memory_arg)
    if is_factory_or_like_or_new_function:
        requires_grad_arg = {
            'default': False,
            'dynamic_type': 'bool',
            'kwarg_only': True,
            'name': 'requires_grad',
            'type': 'bool',
            'simple_type': 'bool',
        }
        python_binding_args.append(requires_grad_arg)

    return python_binding_args
