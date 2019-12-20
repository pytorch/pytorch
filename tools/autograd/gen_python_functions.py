# Generates Python bindings for ATen functions
#
# The bindings are generated as methods on python_variable or functions on the
# torch._C._nn object.
#

# TODO
# 1. take is_mathod dependency out of SigInfo, it doesn't belong there
# 2. make sig_info a dict
# 3. don't store it, just regenerate it as needed
# ...
# goal: move all the arguments-to-python mapping logic into one place
# and use it for both signature and implementation CG

from collections import defaultdict, namedtuple
import re
from .gen_variable_type import should_trace
from .utils import write

try:
    from src.ATen.code_template import CodeTemplate
except ImportError:
    from tools.shared.module_loader import import_module
    CodeTemplate = import_module('code_template', 'aten/src/ATen/code_template.py').CodeTemplate

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
  ${namedtuple_typedefs}
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
  ${namedtuple_typedefs}
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
  ${namedtuple_typedefs}
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

# Unpack parsed args to locals, call the op, and wrap the result.
# Lambda is so GIL is back on by wrap() time (wrap can allocate)
PY_VARIABLE_WRAP = CodeTemplate("""\
// ${schema_string}
${inits}
auto dispatch = ${lambda_def}
return wrap(${namedtuple_typeref}dispatch(${lambda_args})${set_requires_grad});
""")

PY_VARIABLE_RETURN_VOID = CodeTemplate("""\
// ${schema_string}
${inits}
${auto_no_gil}
${dispatch_callee}(${dispatch_args});
Py_RETURN_NONE;
""")

# PyMethodDef entry
PY_VARIABLE_METHOD_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)${pycfunc_voidcast}${pycname}, ${flags}, NULL},""")

# PyMethodDef entry for binary op, throws not implemented error
PY_VARIABLE_METHOD_BINOP_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)${pycfunc_voidcast}TypeError_to_NotImplemented_<${pycname}>, ${flags}, NULL},""")

#
PY_NAMEDTUPLE_FIELDSDEF = CodeTemplate("""\
static PyStructSequence_Field ${fieldsname}[] = { ${fields,} {nullptr} };
""")

# named tuple PyTypeObject
PY_NAMEDTUPLE_TYPEDEF = CodeTemplate("""\
static PyTypeObject ${typename};
static bool ${typename}_initialized = false;
if (!${typename}_initialized) {
  ${typename}_initialized = true;
  static PyStructSequence_Desc desc = { "torch.return_types.${name}", nullptr, ${fieldsname}, ${size} };
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


TENSOR_OPTIONS_DECL = CodeTemplate("""\
const auto options = TensorOptions()
    .dtype(${dtype})
    .device(${device})
    .layout(${layout}.layout)
    .requires_grad(${requires_grad})
    .pinned_memory(${pin_memory});
""")


#
# declaration derived props, utils, etc.
# declarations are dicts loaded from Declarations.yaml,
# passed to our codegen methods by callers in gen_autograd
#

def has_outputs(declaration):
    args = declaration['arguments']
    return any([a.get('output', False) for a in args])


def has_tensor_options_arg(declaration):
    return any([is_tensor_options(arg) for arg in declaration['arguments']])


def is_tensor_method(declaration):
    if 'Tensor' in declaration['method_of']:
        # args = declaration['arguments']
        # assert any([arg['name'] == 'self' and arg['simple_type'] == 'Tensor' for arg in args]), \
        #     "No Tensor-typed self found in {}".format(declaration['name'])
        return True
    return False


def is_torch_function(declaration):
    return 'namespace' in declaration['method_of']


def function_namespace(declaration):
    if has_tensor_options_arg(declaration) or op_name(declaration).endswith('_like'):
        return 'torch'
    else:
        return 'at'


def op_name(declaration):
    name = declaration['name']
    if has_outputs(declaration):
        assert name.endswith("_out"), "output params, expecting name ending with '_out'"
        return name[:-4]
    else:
        assert not name.endswith('_out'), "name ending with '_out', expecting output params"
        return name


def group_declarations(declarations):
    groups = defaultdict(list)
    for d in declarations:
        groups[op_name(d)].append(d)
    return groups


#
# declarations blacklist
# We skip codegen for these functions, for various reasons.
# Future PRs will categorize this list and eliminate or hoist
# them out of eager-only codegen.
# See https://github.com/pytorch/pytorch/issues/30788
#

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

def should_generate_python_binding(declaration):
    name = op_name(declaration)
    for pattern in SKIP_PYTHON_BINDINGS:
        if re.match('^' + pattern + '$', name):
            return False

    simple_types = [arg['simple_type'] for arg in declaration['arguments']]
    signature = '{}({})'.format(name, ', '.join(simple_types))
    for pattern in SKIP_PYTHON_BINDINGS_SIGNATURES:
        if pattern == signature:
            return False

    return True


#
# top-level codegen functions, called from gen_autograd
#

def gen_py_variable_methods(out, declarations, template_path):
    """
    Generate Tensor methods.
    """
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                declaration['mode'] != 'NN' and
                declaration.get('python_module') != 'nn' and
                is_tensor_method(declaration))

    py_variable_methods = group_declarations([d for d in declarations if should_bind(d)])
    env = create_python_bindings(py_variable_methods, True)

    PY_VARIABLE_METHODS_CPP = CodeTemplate.from_file(template_path + '/python_variable_methods.cpp')
    write(out, 'python_variable_methods.cpp', PY_VARIABLE_METHODS_CPP, env)


def gen_py_nn_functions(out, declarations, template_path):
    """
    Generate functions in the "nn" module.
    """
    # TODO move header out of codegen, has no nontrivial generated content
    PY_NN_FUNCTIONS_H = CodeTemplate.from_file(template_path + '/python_nn_functions.h')
    write(out, 'python_nn_functions.h', PY_NN_FUNCTIONS_H, {})

    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                (declaration['mode'] == 'NN' or declaration.get('python_module') == 'nn'))

    py_nn_functions = group_declarations([d for d in declarations if should_bind(d)])
    env = create_python_bindings(py_nn_functions, has_python_self=False, is_module=True)

    PY_NN_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_nn_functions.cpp')
    write(out, 'python_nn_functions.cpp', PY_NN_FUNCTIONS_CPP, env)


def gen_py_torch_functions(out, declarations, template_path):
    """
    Generated functions in the "torch" module.
    """
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                declaration['mode'] != 'NN' and
                declaration.get('python_module') != 'nn' and
                is_torch_function(declaration))

    py_torch_functions = group_declarations([d for d in declarations if should_bind(d)])
    env = create_python_bindings(py_torch_functions, has_python_self=False)

    PY_TORCH_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_torch_functions.cpp')
    write(out, 'python_torch_functions.cpp', PY_TORCH_FUNCTIONS_CPP, env)


#
# codegen
#

def create_python_bindings(python_functions, has_python_self, is_module=False):
    """Generates Python bindings to ATen functions"""
    py_methods = []
    py_method_defs = []

    for name in sorted(python_functions.keys()):
        overload_decls = python_functions[name]
        py_methods.append(method_impl(name, overload_decls, has_python_self))
        py_method_defs.append(method_def(name, overload_decls, has_python_self, is_module))

    return {
        'py_methods': py_methods,
        'py_method_defs': py_method_defs,
    }


#
# extracting and storing parsed args
#

UNPACK_METHODS = {
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

UNPACK_WITH_DEFAULT_METHODS = {
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

def parsed_arg_expr(arg, arg_index):
    # e.g. for arg name 'foo', arg type 'bool', arg_index = 2, returns 'r.toBool(2)'
    typename = arg['type']
    default_init = arg.get('python_default_init')
    if default_init:
        assert typename in UNPACK_WITH_DEFAULT_METHODS, \
            '`{}` type is not supported in python_default_init'.format(typename)
        unpack_with_default = UNPACK_WITH_DEFAULT_METHODS.get(typename)
        return 'r.{}({}, {})'.format(unpack_with_default, arg_index, default_init)
    else:
        unpack = UNPACK_METHODS.get(typename, typename.lower())
        return 'r.{}({})'.format(unpack, arg_index)


# TODO
def unpack_optional_dimname_list_hack(name, expr):
    # optional<ArrayRef<T>> are special. The PythonArgParser returns an
    # optional<vector<T>>, which cannot be implictly converted to
    # optional<ArrayRef<T>>. One needs to unwrap the optional and rewrap.
    result = """\
        auto __{name} = {expr};
        c10::optional<{typ}> {name} = __{name} ? c10::make_optional({typ}(__{name}.value())) : c10::nullopt;
    """.format(name=name, expr=expr, typ='DimnameList')
    return [line.strip() for line in result.split('\n')]


def parse_arg(arg, arg_index, unpack_to_local):
    # get parsed rhs
    expr = parsed_arg_expr(arg, arg_index)

    # maybe unpack to local
    name = arg['name']
    typename = arg['type']
    if typename == 'c10::optional<DimnameList>':
        dispatch_type = typename
        inits = unpack_optional_dimname_list_hack(name, expr)
        expr = name
    elif unpack_to_local:
        inits = ['auto {} = {};'.format(name, expr)]
        expr = name
    else:
        inits = []

    return expr, inits


#
# schema type to cpp type conversions
# some of these are to prevent dangling refs to temps, others are more obscure
# TODO don't know if these fold into more general conversions somehere, hope so
#

TEMP_SAFE_CPP_DECL_TYPE = {
    'Tensor': 'const Tensor &',
    'Tensor &': 'Tensor',
}

CPP_DECL_TYPE_CONVERSION_HACKS = {
    'const Device &': 'c10::optional<int32_t>'
}

def get_cpp_decl_type(typename, ensure_temp_safe=True):
    if ensure_temp_safe:
        typename = TEMP_SAFE_CPP_DECL_TYPE.get(typename, typename)
    return CPP_DECL_TYPE_CONVERSION_HACKS.get(typename, typename)


def get_cpp_formal(arg, ensure_temp_safe=True):
    decl_type = get_cpp_decl_type(arg['type'], ensure_temp_safe)
    return '{} {}'.format(decl_type, arg['name'])


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

def get_simple_return_type(declaration):
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
    return simple_return_type


#
# dispatch codegen
#

def get_dispatch_callee(declaration):
    # format the name of the receiving function or method
    if is_tensor_method(declaration):
        return 'self.{}'.format(declaration['name'])
    elif is_torch_function(declaration):
        namespace = function_namespace(declaration)
        return '{}::{}'.format(namespace, declaration['name'])
    else:
        raise RuntimeError('could not dispatch, neither namespace function nor Tensor method')


def get_op_args(declaration, argmap):
    # returns a list of argmap values in op call order, with two wrinkles:
    # 1. 'self' is eliminated for methods, it's baked into the callee expression elsewhere
    # 2. declaration['call_args'] shims legacy overrides and may contain constant values,
    #    not just names (see load_deprecated_signatures() in gen_autograd.py)
    call_args_override = declaration.get('call_args')
    if call_args_override:
        # names or constants
        keys = call_args_override
    else:
        # only names
        keys = [param['name'] for param in declaration['arguments']]

    if is_tensor_method(declaration):
        # exclude self for method calls
        keys = [k for k in keys if k != 'self']

    if call_args_override:
        # assume missing keys are constants
        return [argmap.get(k, k) for k in keys]
    else:
        # error on missing keys
        return [argmap[k] for k in keys]


# we use a lambda to nest op dispatch within a GIL drop
LAMBDA_DEF = CodeTemplate("""\
[](${lambda_formals}) {
  ${auto_no_gil}
  return ${dispatch_callee}(${dispatch_args});
};
""")


def emit_single_dispatch(declaration, out_idx, has_python_self):
    """
    Emit dispatch code for a single declared overload.
    """

    python_binding_arguments = declaration['python_binding_arguments']
    has_tensor_options = has_tensor_options_arg(declaration)

    inits = []          # initializer code for unpacking parsed args
    argmap = {}

    # inputs
    if has_python_self:
        argmap['self'] = {'value': 'self', 'formal': 'Tensor & self'}
        # self is passed directly to python binding, rather than parsed
        # redo formals filtering TODO merge
        arg_filter = lambda arg: not is_tensor_options(arg) and not is_tensor_self(arg)
    else:
        # redo formals filtering TODO merge
        arg_filter = lambda arg: not is_tensor_options(arg)

    def is_input(arg):
        return (not is_tensor_options(arg) and
                not (has_python_self and is_tensor_self(arg)) and
                not arg.get('output', False))

    inputs = [arg for arg in declaration['arguments'] if is_input(arg)]
    for i, arg in enumerate(inputs):
        # make sure we have a local 'self' if we're dotting off it for tensor options
        unpack_to_local = has_tensor_options and is_tensor_self(arg)
        arg_expr, unpack = parse_arg(arg, i, unpack_to_local=unpack_to_local)
        inits.extend(unpack)
        argmap[arg['name']] = {'value': arg_expr, 'formal': get_cpp_formal(arg)}

    # outputs
    outputs = [arg for arg in declaration['arguments'] if arg.get('output', False)]
    output_idx = len(inputs)
    if len(outputs) == 1:
        arg = outputs[0]
        arg_expr, _ = parse_arg(arg, output_idx, unpack_to_local=False)
        argmap[arg['name']] = {'value': arg_expr, 'formal': get_cpp_formal(arg)}
    elif len(outputs) > 1:
        # take gathered output param from python call and scatter it to op call args
        inits.append('auto results = r.tensorlist_n<{}>({});'.format(len(outputs), output_idx))
        for i, arg in enumerate(outputs):
            argmap[arg['name']] = {
                'value': 'results[{}]'.format(i),
                'formal': 'Tensor & {}'.format(arg['name']),
            }

    #
    # wrangle tensor attribute params. common case is that we gather python binding args
    # into a tensor options param for the called op, but there are many variations.
    # TODO finish cleaning this up once TensorOptions revamp is in (if it still exists)
    #

    # find at most one type arg to use, check error conditions
    type_args = [arg for arg in declaration['arguments'] if arg['simple_type'] == 'Type']
    type_binding_args = [arg for arg in python_binding_arguments if arg['simple_type'] == 'Type']
    if len(type_binding_args) > 0 and len(outputs) == 0:
        # out(s) determines the dtype if it is present, so only use this if there are no outputs.
        type_arg = type_binding_args[0]
    elif len(type_args) > 0:
        type_arg = type_args[0]
    else:
        type_arg = None
    if type_arg and len(outputs) > 1:
        raise RuntimeError(declaration['name'] + ": type dispatched parameter with multiple outputs not supported")

    layout = None
    parsed_type_arg = None
    # type args go after the outputs to match the signature generation.
    # TODO use out_idx instead? or kill it?
    arg_idx = len(inputs) if out_idx is None else len(inputs) + 1
    if type_arg is not None:
        parsed_type_arg, unpack_type = parse_arg(type_arg, arg_idx, has_tensor_options)
        inits.extend(unpack_type)
        arg_idx += 1

    if len(type_binding_args) > 0:
        if not has_tensor_options:
            # these are _out variants TODO clean up this logic
            arg_idx += 1

    if 'layout' in (a['name'] for a in python_binding_arguments):
        layout_idx, device_idx, pin_memory_idx, requires_grad_idx = (arg_idx, arg_idx + 1, arg_idx + 2, arg_idx + 3)
    else:
        device_idx, pin_memory_idx, requires_grad_idx = (arg_idx, arg_idx + 1, arg_idx + 2)

    requires_grad = None
    device = None
    for arg in python_binding_arguments:
        if arg['name'] == 'dtype' and arg['simple_type'] == 'Type':
            pass  # already handled by type_dispatched_args
        elif arg['name'] == 'layout' and arg['simple_type'] == 'Layout':
            # out(s) determines the type and layout if it is present, so only use this if there are no outputs.
            if len(outputs) == 0:
                layout, _ = parse_arg(arg, layout_idx, unpack_to_local=False)
        elif arg['name'] == 'device' and arg['simple_type'] == 'Device':
            if len(outputs) == 0:
                assert parsed_type_arg
                assert layout
                device, unpack_device = parse_arg(arg, device_idx, unpack_to_local=True)
                inits.extend(unpack_device)
        elif arg['name'] == 'requires_grad' and arg['simple_type'] == 'bool':
            requires_grad, _ = parse_arg(arg, requires_grad_idx, unpack_to_local=False)
        elif arg['name'] == 'pin_memory' and arg['simple_type'] == 'bool':
            pin_memory, _ = parse_arg(arg, pin_memory_idx, unpack_to_local=False)
        else:
            raise RuntimeError(("found {} in python_binding_arguments but only "
                                "\"bool pin_memory\", \"bool requires_grad\", \"ScalarType dtype\", \"Layout layout\", "
                                "\"Device device\" are supported".format(arg)))

    dtype = parsed_type_arg
    if has_tensor_options:
        assert all([dtype, device, layout, requires_grad]), "{}: incomplete tensor options".format(declaration['name'])
        inits.append(TENSOR_OPTIONS_DECL.substitute({
            'dtype': dtype,
            'layout': layout,
            'device': device,
            'requires_grad': requires_grad,
            'pin_memory': pin_memory,
        }))
        inits.append('torch::utils::maybe_initialize_cuda(options);')
        argmap['options'] = {
            'value': 'options',
            'formal': 'const TensorOptions & options',
        }

    #
    # home stretch - set up misc template inputs and generate
    #

    dispatch_callee = get_dispatch_callee(declaration)

    auto_no_gil = [] if declaration['with_gil'] else ['AutoNoGIL no_gil;']

    simple_return_type = get_simple_return_type(declaration)

    if simple_return_type == 'void':
        # simpler generated codepath - no wrap(result), no lambda around GIL drop
        # so we dispatch to the op directly
        dispatch_args = get_op_args(declaration, {name: arg['value'] for name, arg in argmap.items()})
        return PY_VARIABLE_RETURN_VOID.substitute(
            auto_no_gil=auto_no_gil,
            inits=inits,
            dispatch_callee=dispatch_callee,
            dispatch_args=dispatch_args,
            schema_string=declaration['schema_string'],
        )

    # generate standard codepath
    if requires_grad and not has_tensor_options:
        set_requires_grad = '.set_requires_grad({})'.format(requires_grad)
    else:
        set_requires_grad = ''

    # lambda takes all op args
    lambda_args = [argmap[arg['name']]['value'] for arg in declaration['arguments']]
    lambda_formals = [argmap[arg['name']]['formal'] for arg in declaration['arguments']]
    # and dispatches to the op
    dispatch_args = get_op_args(declaration, {name: name for name, _ in argmap.items()})

    lambda_def = LAMBDA_DEF.substitute(
        lambda_formals=lambda_formals,
        auto_no_gil=auto_no_gil,
        dispatch_callee=dispatch_callee,
        dispatch_args=dispatch_args,
    )

    return PY_VARIABLE_WRAP.substitute(
        inits=inits,
        simple_return_type=simple_return_type,
        namedtuple_typeref=declaration['namedtuple_typeref'],
        lambda_def=lambda_def,
        lambda_args=lambda_args,
        schema_string=declaration['schema_string'],
        set_requires_grad=set_requires_grad,
    )


def emit_dispatch(i, dictionary, has_python_self):
    """
    Emit dispatch code for a single parsed signature. This may correspond to either one
    or two declared overloads: those that differ only in optional output params are paired
    up in the generated binding.

    - i:            this signature's position in generated binding's signature list
                    if number of signatures > 1, otherwise None

    - dictionary:   contains a no-output overload declaration under 'base', and optionally
                    a second overload with outputs under 'out'
    """
    base_decl = dictionary['base']
    if 'out' in dictionary:
        out_decl = dictionary['out']
        pfs = get_python_formals(out_decl, has_python_self)
        out_idx = len(pfs['input_formals'] + pfs['input_kw_formals'] + pfs['type_formals'])

        env = {}
        env['call_dispatch_out'] = emit_single_dispatch(out_decl, out_idx, has_python_self)
        env['call_dispatch'] = emit_single_dispatch(base_decl, out_idx, has_python_self)

        has_dtype_bind = any(f.startswith('ScalarType dtype') for f in pfs['python_binding_formals'])
        if has_dtype_bind:
            body = PY_VARIABLE_OUT_CHECK_TYPE.substitute(env, out_idx=out_idx, type_idx=out_idx + 1,
                                                            layout_idx=out_idx + 2, device_idx=out_idx + 3).split('\n')
        else:
            body = PY_VARIABLE_OUT.substitute(env, out_idx=out_idx).split('\n')
    else:
        body = emit_single_dispatch(base_decl, None, has_python_self)

    if i is None:
        return PY_VARIABLE_SINGLE_CASE.substitute(call_dispatch=body)
    else:
        return PY_VARIABLE_CASE.substitute(i=i, call_dispatch=body)

#
# named tuple codegen
#

def namedtuple_fieldnames(declaration):
    returns = declaration['returns']
    if len(returns) <= 1 or all(['field_name' not in x for x in returns]):
        return []
    else:
        def get_field_name(x):
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
                return x['field_name']
        return [get_field_name(x) for x in returns]


def emit_namedtuple_typedefs(declarations):
    """
    Generate block of named tuple type def inits, and add typeref snippets
    to declarations that use them
    """
    flddefnames = {}    # map from unique field name lists to field def name
    flddefs = []        # field def declarations
    typenames = {}      # map from unique name + field name lists to typedef name
    typedefs = []       # typedef declarations and init code

    for decl in declarations:
        fieldnames = namedtuple_fieldnames(decl)
        if fieldnames == []:
            decl['namedtuple_typeref'] = ''
            continue

        fn_key = '_'.join(fieldnames)
        fieldsname = flddefnames.get(fn_key)
        if fieldsname is None:
            fieldsname = 'NamedTuple_fields{}'.format('' if flddefs == [] else len(fielddefs))
            fields=['{{"{}", ""}}'.format(fn) for fn in fieldnames]
            fieldsdef = PY_NAMEDTUPLE_FIELDSDEF.substitute(
                fieldsname=fieldsname,
                fields=fields
            )
            flddefnames[fn_key] = fieldsname
            flddefs.append(fieldsdef)

        name = decl['name']
        key = '{}_{}'.format(name, '_'.join(fieldnames))
        typename = typenames.get(key)
        if typename is None:
            typename = 'NamedTuple{}'.format('' if typedefs == [] else len(typedefs))
            typedef = PY_NAMEDTUPLE_TYPEDEF.substitute(
                name=name,
                typename=typename,
                size=len(fieldnames),
                fieldsname=fieldsname
            )
            typenames[key] = typename
            typedefs.append(typedef)

        decl['namedtuple_typeref'] = '&{}, '.format(typename)

    return flddefs + typedefs

#
# method codegen
#

def pycname(name):
    return 'THPVariable_{}'.format(name)


def skip_arg_parse(declarations, has_python_self):
    if len(declarations) > 1:
        return False
    decl = declarations[0]
    argc = len(decl['arguments'])
    return argc == 0 or (has_python_self and argc == 1)


def parsed_arg_count(decl, has_python_self):
    python_formals = get_python_formals(decl, has_python_self)
    return sum([len(sublist) for sublist in python_formals.values()])


def method_impl(name, declarations, has_python_self):
    """
    Generate a python binding for all overloads of an op.
    """
    for declaration in declarations:
        # extra arguments used in python binding signature
        declaration['python_binding_arguments'] = get_python_binding_arguments(declaration)
        # formals for python binding signature
        declaration['python_formals'] = get_python_formals(declaration, has_python_self)

    namedtuple_typedefs = emit_namedtuple_typedefs(declarations)
    unpack_self = [UNPACK_SELF] if has_python_self else []

    # emit dispatch
    if skip_arg_parse(declarations, has_python_self):
        return PY_VARIABLE_METHOD_NOARGS.substitute(
            name=name,
            pycname=pycname(name),
            namedtuple_typedefs=namedtuple_typedefs,
            unpack_self=unpack_self,
            dispatch=emit_single_dispatch(declaration, 0, has_python_self),
        )

    grouped = group_overloads(declarations)
    is_singleton = len(grouped) == 1

    signatures = []
    dispatch = []
    for i, dictionary in enumerate(grouped):
        signature = dictionary['signature']
        signatures.append('"{}",'.format(signature))
        overload_index = i if not is_singleton else None
        dispatch.append(emit_dispatch(overload_index, dictionary, has_python_self))

    if is_singleton:
        impl_template = PY_VARIABLE_METHOD_VARARGS_SINGLETON
    else:
        impl_template = PY_VARIABLE_METHOD_VARARGS

    return impl_template.substitute(
        name=name,
        pycname=pycname(name),
        namedtuple_typedefs=namedtuple_typedefs,
        unpack_self=unpack_self,
        max_args=max([parsed_arg_count(decl, has_python_self) for decl in declarations]),
        traceable='true' if all(should_trace(d) for d in declarations) else 'false',
        signatures=signatures,
        dispatch=dispatch,
    )


def method_def(name, declarations, has_python_self, is_module):
    """
    Generate method def entry.
    """
    if skip_arg_parse(declarations, has_python_self):
        pycfunc_voidcast = ''
        flags = 'METH_NOARGS' if has_python_self else 'METH_VARARGS | METH_KEYWORDS'
    else:
        pycfunc_voidcast = '(void(*)(void))'
        flags = 'METH_VARARGS | METH_KEYWORDS'

    if not is_module and not has_python_self:
        flags += ' | METH_STATIC'

    if name in BINARY_OP_NAMES:
        def_template = PY_VARIABLE_METHOD_BINOP_DEF
    else:
        def_template = PY_VARIABLE_METHOD_DEF

    return def_template.substitute(
        name=name,
        pycname=pycname(name),
        pycfunc_voidcast=pycfunc_voidcast,
        flags=flags,
    )

#
#
#

def group_overloads(declarations):
    """Returns a list of dictionaries containing the optional keys:

       "base": the regular ATen declaration (e.g. conv2d)
       "out": the out variant (e.g. conv2d_out)
       "signature": the signature used for Python argument parsing

       Note that we merge pairs of declarations with signatures that
       are equivalent mod output arguments, and use a single entry in
       the python_arg_parser sig list for both (output arguments become
       optional)
    """
    grouped = defaultdict(dict)

    # first group by signature ignoring out arguments
    for declaration in declarations:
        signature = get_python_signature(declaration, include_outputs=False)
        v = grouped[signature]
        if declaration['name'].endswith('_out'):
            v['out'] = declaration
            # prefer the signature with optional out=... arguments
            v['signature'] = get_python_signature(declaration, include_outputs=True)
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


def get_python_signature(declaration, include_outputs):
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

    pf = declaration['python_formals']
    input_formals = pf['input_formals']
    input_kw_formals = pf['input_kw_formals']
    output_formals = pf['output_formals'] if include_outputs else []
    type_formals = pf['type_formals']
    python_binding_formals = pf['python_binding_formals']

    kw_formals = input_kw_formals + output_formals + type_formals + python_binding_formals

    if kw_formals != []:
        py_formal_args = input_formals + ['*'] + kw_formals
    else:
        py_formal_args = input_formals

    name = op_name(declaration)

    # Python function signature.
    # This is the string that we give to FunctionParameter, which is
    # then parsed into the actual structure which we do parsing
    # with.
    signature = PYTHON_FUNCTION_SIGNATURE.substitute(name=name, py_formal_args=py_formal_args)
    if declaration.get('deprecated', False):
        signature += '|deprecated'

    return signature


#
# arg derived props and utils
#

def is_tensor_self(arg):
    return arg['name'] == 'self' and arg['simple_type'] == 'Tensor'


def is_tensor_options(arg):
    return arg['simple_type'] == 'TensorOptions'


#
#
#

SCHEMA_TYPE_CONVERSION_HACKS = {
    'Type': 'ScalarType',
}

def get_schema_type(arg):
    typename = arg['simple_type']
    typename = SCHEMA_TYPE_CONVERSION_HACKS.get(typename, typename)

    # TODO: remove this and make optional types in simple_type to be consistent across
    # tensor and other types after make Tensor? be optional instead of undefined
    if arg.get('is_nullable') and '?' not in typename:
        typename = '{}?'.format(typename)

    size = arg.get('size')
    if size is not None:
        typename = '{}[{}]'.format(typename, size)

    return typename


SCHEMA_DEFAULT_CONVERSION_HACKS = {
    'nullptr': 'None',
    'c10::nullopt': 'None',
    '{}': 'None',
}

def get_schema_default(arg, fallback_default=None):
    default = arg.get('default', fallback_default)
    return SCHEMA_DEFAULT_CONVERSION_HACKS.get(default, default)


def get_schema_formal(arg, fallback_default=None):
    name = arg['name']
    typename = get_schema_type(arg)
    default = get_schema_default(arg, fallback_default)
    if default is not None:
        return '{} {}={}'.format(typename, name, default)
    else:
        return '{} {}'.format(typename, name)


def get_python_formals(declaration, as_method):

    args = declaration['arguments']
    input_args = []
    input_kwargs = []
    output_args = []
    type_args = []

    current_input_args = input_args
    for arg in args:
        if arg.get('output', False):
            output_args.append(arg)
        elif arg['simple_type'] == 'Type':
            type_args.append(arg)
        else:
            if arg.get('kwarg_only', False):
                current_input_args = input_kwargs
            if is_tensor_self(arg) and not as_method:
                # if we're generating a function (not method) binding, s/self/input/
                arg = arg.copy()
                arg['name'] = 'input'
            current_input_args.append(arg)

    # we omit some op-defined args from python:
    # - self: for method bindings, this is a separate Python param, not a parsed arg
    # - options: `python_binding_formals` are mapped into a TensorOptions struct
    if as_method:
        arg_filter = lambda arg: not is_tensor_options(arg) and not is_tensor_self(arg)
    else:
        arg_filter = lambda arg: not is_tensor_options(arg)

    input_args = [a for a in input_args if arg_filter(a)]
    input_kwargs = [a for a in input_kwargs if arg_filter(a)]
    output_args = [a for a in output_args if arg_filter(a)]
    type_args = [a for a in type_args if arg_filter(a)]

    input_formals = [get_schema_formal(a) for a in input_args]
    input_kw_formals = [get_schema_formal(a) for a in input_kwargs]

    if len(output_args) <= 1:
        output_formals = [get_schema_formal(a, fallback_default='None') for a in output_args]
    else:
        # NB: For more than 1 output args the type name is a TensorList
        # and as such we don't (yet) need to consider the naming.
        assert all([a['simple_type'] == 'Tensor' for a in output_args])
        typename = 'TensorList[{}]'.format(len(output_args))
        output_formals = [typename + ' out=None']

    assert len(type_args) <= 1, '{}: multiple type args'.format(declaration['name'])
    type_formals = [get_schema_formal(a) for a in type_args]

    python_binding_arguments = declaration['python_binding_arguments']
    python_binding_formals = [get_schema_formal(a) for a in python_binding_arguments]

    return {
        'input_formals': input_formals,
        'input_kw_formals': input_kw_formals,
        'output_formals': output_formals,
        'type_formals': type_formals,
        'python_binding_formals': python_binding_formals,
    }


# @@@
# python arg has ptr to orig arg
# has parse index
# is flat list
# could return other info as sig_info props, w arg list a prop
# or could recompute, so e.g. kw_only insertion point, type arg, relationship w/TO
# steps
# 1. build args here instead of formals
# 2. return flat list + sig OR return flat list + write functions
# 3. remove last pa thing from emit_dispatch
# 4. use parse_index props instead of out_idx bullshit
# 5. use python args in emit_single_dispatch
# 6. litmus test will be that big lump in the middle
# 7. swap in p_b_a thing from stash
# @@@


def get_python_args_and_signature(declaration, as_method):

    args = declaration['arguments']

    input_args = []
    input_kwargs = []
    output_args = []
    type_args = []

    current_input_args = input_args
    for arg in args:
        if arg.get('output', False):
            output_args.append(arg)
        elif arg['simple_type'] == 'Type':
            type_args.append(arg)
        else:
            if arg.get('kwarg_only', False):
                current_input_args = input_kwargs
            if is_tensor_self(arg) and not as_method:
                # if we're generating a function (not method) binding, s/self/input/
                arg = arg.copy()
                arg['name'] = 'input'
            current_input_args.append(arg)

    # we omit some op-defined args from python:
    # - self: for method bindings, this is a separate Python param, not a parsed arg
    # - options: `python_binding_formals` are mapped into a TensorOptions struct
    if as_method:
        arg_filter = lambda arg: not is_tensor_options(arg) and not is_tensor_self(arg)
    else:
        arg_filter = lambda arg: not is_tensor_options(arg)

    input_args = [a for a in input_args if arg_filter(a)]
    input_kwargs = [a for a in input_kwargs if arg_filter(a)]
    output_args = [a for a in output_args if arg_filter(a)]
    type_args = [a for a in type_args if arg_filter(a)]

    input_formals = [get_schema_formal(a) for a in input_args]
    input_kw_formals = [get_schema_formal(a) for a in input_kwargs]

    if len(output_args) <= 1:
        output_formals = [get_schema_formal(a, fallback_default='None') for a in output_args]
    else:
        # NB: For more than 1 output args the type name is a TensorList
        # and as such we don't (yet) need to consider the naming.
        assert all([a['simple_type'] == 'Tensor' for a in output_args])
        typename = 'TensorList[{}]'.format(len(output_args))
        output_formals = [typename + ' out=None']

    assert len(type_args) <= 1, '{}: multiple type args'.format(declaration['name'])
    type_formals = [get_schema_formal(a) for a in type_args]

    python_binding_arguments = declaration['python_binding_arguments']
    python_binding_formals = [get_schema_formal(a) for a in python_binding_arguments]

    return {
        'input_formals': input_formals,
        'input_kw_formals': input_kw_formals,
        'output_formals': output_formals,
        'type_formals': type_formals,
        'python_binding_formals': python_binding_formals,
    }



# TODO blowtorch
def dtype_default_type_hack(name):
    if (name.startswith('randperm') or
        name == 'tril_indices' or
        name == 'triu_indices'):
        return 'torch.int64'
    else:
        return 'None'


def get_python_binding_arguments(declaration):
    """
    Given various properties of a declaration, build a set of scattered tensor options args.
    These are added to the python binding and gathered into a TensorOptions value for the op.
    """
    name = declaration['name']
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
        default_type = dtype_default_type_hack(name)
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
