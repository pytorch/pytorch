# Generates Python bindings for ATen functions
#
# The bindings are generated as methods on python_variable or functions on the
# torch._C._nn object.
#

# Code tries to stick to the following rules:
#
# - templates should be colocated with the functions that use them.
#   no templates are currently shared between functions, but if that
#   happens, maybe put the template with the first one
#
# - don't use environment dictionaries when calling template.substitute().
#   pass named arguments directly for everything, otherwise it's much too
#   hard to track what's actually being used and by who
#
# - colocate any new hacks/adjustments with existing ones of the same kind.
#   ideally in a data structure rather than code if possible. See e.g.
#   SCHEMA_DEFAULT_CONVERSION_HACKS, etc.
#
# - similarly, conversions from one format to another should ideally happen
#   all at once in a single place.
#
# - no nontrivial nested functions. couple-liners are ok but please no more.
#   especially avoid functions that read/write outer variables defined far away.
#
# - raise RuntimeError instead of asserting, and put as much
#   information as is available into the message. I.e. no need to
#   plumb in new params whose only purpose is to fill out an error
#   message, but use what's there
#

from collections import defaultdict
import re
from .gen_variable_type import should_trace
from .utils import write

try:
    from src.ATen.code_template import CodeTemplate
except ImportError:
    from tools.shared.module_loader import import_module
    CodeTemplate = import_module('code_template', 'aten/src/ATen/code_template.py').CodeTemplate

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
    'data', 'is_leaf', 'output_nr', '_version', 'requires_grad_', 'retain_grad'
]

# These function signatures are not exposed to Python. Note that this signature
# list does not support regex.
SKIP_PYTHON_BINDINGS_SIGNATURES = [
    'add(Tensor, Scalar, Scalar)', 'add_(Tensor, Scalar, Scalar)',
    'sub(Tensor, Scalar, Scalar)', 'sub_(Tensor, Scalar, Scalar)',
    'mul(Tensor, Scalar)', 'mul_(Tensor, Scalar)',
    'div(Tensor, Scalar)', 'div_(Tensor, Scalar)',
]

NATIVE_NAMESPACE_MAPPING = {
    "torch": "THPVariableFunctionsModule",
    "torch.nn": "THPNNVariableFunctionsModule"
}

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

#
# top-level codegen functions, called from gen_autograd
#

def get_py_variable_methods(declarations):
    """
    Get declarations (grouped by name) which should be generated
    as methods on Tensor.
    """
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                not is_nn_module_function(declaration) and
                is_tensor_method(declaration))

    return group_declarations_by_op_name([d for d in declarations if should_bind(d)])


def gen_py_variable_methods(out, declarations, template_path):
    """
    Generate Tensor methods.
    """
    PY_VARIABLE_METHODS_CPP = CodeTemplate.from_file(template_path + '/python_variable_methods.cpp')

    py_variable_methods = get_py_variable_methods(declarations)

    env = create_python_bindings(py_variable_methods, is_python_method=True, module=None)

    write(out, 'python_variable_methods.cpp', PY_VARIABLE_METHODS_CPP, env)


def get_py_nn_functions(declarations):
    """
    Get declarations (grouped by name) which should be generated
    as functions in the "nn" module.
    """
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                is_nn_module_function(declaration))

    return group_declarations_by_op_name([d for d in declarations if should_bind(d)])


def gen_py_nn_functions(out, declarations, template_path):
    """
    Generate functions in the "nn" module.
    """
    PY_NN_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_nn_functions.cpp')

    py_nn_functions = get_py_nn_functions(declarations)

    env = create_python_bindings(py_nn_functions, is_python_method=False, module="torch.nn")

    write(out, 'python_nn_functions.cpp', PY_NN_FUNCTIONS_CPP, env)


def get_py_torch_functions(declarations):
    """
    Get declarations (grouped by name) which should be generated
    as functions in the "torch" module.
    """
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                not is_nn_module_function(declaration) and
                is_torch_function(declaration))

    return group_declarations_by_op_name([d for d in declarations if should_bind(d)])


def gen_py_torch_functions(out, declarations, template_path):
    """
    Generate functions in the "torch" module.
    """
    PY_TORCH_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_torch_functions.cpp')

    py_torch_functions = get_py_torch_functions(declarations)

    env = create_python_bindings(py_torch_functions, is_python_method=False, module="torch")

    write(out, 'python_torch_functions.cpp', PY_TORCH_FUNCTIONS_CPP, env)


def group_declarations_by_op_name(declarations):
    groups = defaultdict(list)
    for d in declarations:
        groups[op_name(d)].append(d)
    return groups


def create_python_bindings(python_functions, is_python_method, module):
    """Generates Python bindings to ATen functions"""
    py_methods = []
    py_method_defs = []
    py_forwards = []

    for name in sorted(python_functions.keys()):
        overload_decls = python_functions[name]
        py_methods.append(method_impl(name, overload_decls, is_python_method, module))
        py_method_defs.append(method_def(name, overload_decls, is_python_method, module))
        py_forwards.extend(forward_decls(name, overload_decls, is_python_method, module))

    return {
        'py_forwards': py_forwards,
        'py_methods': py_methods,
        'py_method_defs': py_method_defs,
    }


#
# extracting and storing parsed args
#

UNPACK_METHODS = {
    'const Tensor &': 'tensor',
    'Tensor &': 'tensor',
    'Generator': 'generator',
    'Storage': 'storage',
    'Storage &': 'storage',
    'const ScalarType &': 'scalartype',
    'const Device &': 'device',
    'c10::optional<DimnameList>': 'toDimnameListOptional',
    'c10::optional<ScalarType>': 'scalartypeOptional',
    'c10::optional<Layout>': 'layoutOptional',
    'c10::optional<MemoryFormat>': 'memoryformatOptional',
    'c10::optional<Scalar>': 'scalarOptional',
    'c10::optional<int64_t>': 'toInt64Optional',
    'c10::optional<bool>': 'toBoolOptional',
    'c10::optional<double>': 'toDoubleOptional',
    'IntArrayRef': 'intlist',
    'Scalar': 'scalar',
    'ScalarType': 'scalartype',
    'Dimname': 'dimname',
    'DimnameList': 'dimnamelist',
    'TensorList': 'tensorlist',
    'int64_t': 'toInt64',
    'bool': 'toBool',
    'double': 'toDouble',
    'std::string': 'string',
}

UNPACK_WITH_SIZE_METHODS = {
    'TensorList': 'tensorlist_n<{}>',
    'DimnameList': 'dimnamelist',
    'IntArrayRef': 'intlist',
}

UNPACK_WITH_DEFAULT_METHODS = {
    'const ScalarType &': 'scalartypeWithDefault',
    'const Device &': 'deviceWithDefault',
    'c10::optional<Layout>': 'layoutWithDefault',
}

def parsed_arg_expr(arg, arg_index):
    # e.g. for arg name 'foo', arg type 'bool', arg_index = 2, returns '_r.toBool(2)'
    typename = arg['type']

    default_init = arg.get('python_default_init')
    if default_init is not None:
        # Note: only introduced by make_python_binding_args
        default_init = arg['python_default_init']
        if typename not in UNPACK_WITH_DEFAULT_METHODS:
            raise RuntimeError(
                'type \'{}\' is not supported in python_default_init'.
                format(typename))
        unpack_with_default = UNPACK_WITH_DEFAULT_METHODS[typename]
        return '_r.{}({}, {})'.format(unpack_with_default, arg_index, default_init)

    size = arg.get('size')
    if size is not None:
        if typename not in UNPACK_WITH_SIZE_METHODS:
            raise RuntimeError(
                'type \'{}\' with definite size ({}) is not supported'.
                format(typename, size))
        unpack_with_size = UNPACK_WITH_SIZE_METHODS[typename].format(size)
        return '_r.{}({})'.format(unpack_with_size, arg_index)

    unpack = UNPACK_METHODS.get(typename)
    if unpack is None:
        raise RuntimeError('type \'{}\' is not supported'.format(typename))

    return '_r.{}({})'.format(unpack, arg_index)


# TODO make this part of something more general, or get rid of it
def unpack_optional_dimname_list_hack(name, expr):
    # optional<ArrayRef<T>> are special. The PythonArgParser returns an
    # optional<vector<T>>, which cannot be implicitly converted to
    # optional<ArrayRef<T>>. One needs to unwrap the optional and rewrap.
    result = """\
        auto __{name} = {expr};
        c10::optional<{typ}> {name} = __{name} ? c10::make_optional({typ}(__{name}.value())) : c10::nullopt;
    """.format(name=name, expr=expr, typ='DimnameList')
    return [line.strip() for line in result.split('\n')]


def parse_arg(arg, arg_index, unpack_to_local=False):
    # get parsed rhs
    expr = parsed_arg_expr(arg, arg_index)

    # maybe unpack to local
    name = arg['name']
    typename = arg['type']
    if typename == 'c10::optional<DimnameList>':
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
    'Tensor &': 'Tensor',
}

def get_cpp_decl_type(typename, ensure_temp_safe=True):
    if ensure_temp_safe:
        typename = TEMP_SAFE_CPP_DECL_TYPE.get(typename, typename)
    return typename


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
    # (Tensor &).  This is important because the dispatch lambdas take
    # mutable arguments *by value*, not by reference.  If you then return
    # a reference to such an argument, you will now have a pointer to a
    # dangling stack entry.  Not good.
    #
    # You want:
    #
    #   auto dispatch_selu_ = [](Tensor self) -> Tensor { ...; return at::selu_(self); };
    #                                            ^^^^^^
    #
    # *not*
    #
    #   auto dispatch_selu_ = [](Tensor self) -> Tensor& { ...; return at::selu_(self); };
    #                                            ^^^^^^^
    #
    # (NB: We can't make dispatch_selu_ take Tensor&, because the enclosing
    # codegen looks like dispatch_selu_(_r.tensor(0)), and you can't take a
    # mutable reference to temporary.  Maybe we could assign it to a
    # variable itself.)
    #
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
        return [argmap[k] for k in keys]


TENSOR_OPTIONS_DECL = CodeTemplate("""\
const auto ${name} = TensorOptions()
    .dtype(${dtype})
    .device(${device})
    .layout(${layout})
    .requires_grad(${requires_grad})
    .pinned_memory(${pin_memory});
""")

# addition to output-variant handler in which tensor options params
# (if present) are checked against properties of a tensor output param
# TODO remove hardcoding, use unpack logic from emit_single_dispatch
PY_VARIABLE_CHECK_OUT_TYPE_HACK = CodeTemplate("""\
check_out_type_matches(_r.tensor(${out_idx}), _r.scalartype(${type_idx}),
                       _r.isNone(${type_idx}), _r.layoutOptional(${layout_idx}),
                       _r.device(${device_idx}), _r.isNone(${device_idx}));
""")

# Unpack parsed args to locals, call the op, and wrap the result.
# Lambda is so GIL is back on by wrap() time (wrap can allocate)
PY_VARIABLE_WRAP = CodeTemplate("""\
${inits}
auto dispatch_${name} = [](${lambda_formals}) -> ${simple_return_type} {
  ${auto_no_gil}
  return ${dispatch_callee}(${dispatch_args});
};
return wrap(${namedtuple_typeref}dispatch_${name}(${lambda_args})${set_requires_grad});
""")

# void return variant
PY_VARIABLE_RETURN_VOID = CodeTemplate("""\
${inits}
auto dispatch_${name} = [](${lambda_formals}) -> ${simple_return_type} {
  ${auto_no_gil}
  ${dispatch_callee}(${dispatch_args});
};
dispatch_${name}(${lambda_args})${set_requires_grad};
Py_RETURN_NONE;
""")


def emit_single_dispatch(declaration, is_python_method, output_gap=0):
    """
    Emit dispatch code for a single declared overload.
    """
    deprecated = '[deprecated] ' if declaration.get('deprecated', False) else ''
    schema_comment = '// ' + deprecated + declaration['schema_string']
    inits = [schema_comment]

    pa = declaration['python_arglists']
    args = pa['input_args'] + pa['input_kwargs'] + pa['output_args']
    has_options = has_tensor_options(declaration)

    argmap = {}

    if is_python_method:
        # self is passed directly to python binding, rather than parsed
        argmap['self'] = {'value': 'self', 'formal': 'Tensor & self'}

    for i, arg in enumerate(args):
        unpack = is_scatter(arg) or (has_options and is_tensor_self(arg))
        arg_expr, unpack_stmts = parse_arg(arg, i, unpack_to_local=unpack)
        inits.extend(unpack_stmts)
        if is_scatter(arg):
            for j, elem in enumerate(arg['scatter_args']):
                argmap[elem['name']] = {
                    'value': '{}[{}]'.format(arg_expr, j),
                    'formal': get_cpp_formal(elem, ensure_temp_safe=False),
                }
        else:
            argmap[arg['name']] = {'value': arg_expr, 'formal': get_cpp_formal(arg)}

    # synthetic python binding args deliver op args
    binding_argmap, binding_inits, set_requires_grad = \
        handle_python_binding_args(declaration, output_gap)
    argmap.update(binding_argmap)
    inits.extend(binding_inits)

    lambda_formals = [argmap[arg['name']]['formal'] for arg in declaration['arguments']]
    lambda_args = [argmap[arg['name']]['value'] for arg in declaration['arguments']]

    dispatch_callee = get_dispatch_callee(declaration)
    dispatch_args = get_op_args(declaration, {name: name for name, _ in argmap.items()})

    auto_no_gil = [] if declaration['with_gil'] else ['pybind11::gil_scoped_release no_gil;']

    simple_return_type = get_simple_return_type(declaration)
    if simple_return_type == 'void':
        template = PY_VARIABLE_RETURN_VOID
    else:
        template = PY_VARIABLE_WRAP

    return template.substitute(
        name=declaration['name'],
        inits=inits,
        lambda_formals=lambda_formals,
        lambda_args=lambda_args,
        dispatch_callee=dispatch_callee,
        dispatch_args=dispatch_args,
        auto_no_gil=auto_no_gil,
        set_requires_grad=set_requires_grad,
        simple_return_type=simple_return_type,
        namedtuple_typeref=declaration['namedtuple_typeref'],
    )


# arg['name'] to arg['simple_type'] for scattered tensor options fields
TENSOR_OPTIONS_FIELDS = {
    'dtype': 'ScalarType',
    'device': 'Device',
    'layout': 'Layout',
    'pin_memory': 'bool',
    'requires_grad': 'bool',
}

def handle_python_binding_args(declaration, output_gap):
    # map synthetic python binding args to op args and misc other stuff
    # note: this logic shares arcane knowledge with make_python_binding_args
    # and isn't completely airtight w.r.t. the possible contents of
    # python_binding_args. TODO

    argmap = {}
    inits = []
    set_requires_grad = ''

    pa = declaration['python_arglists']
    python_binding_args = pa['python_binding_args']

    if len(python_binding_args) == 0:
        # nothing to see here
        return argmap, inits, set_requires_grad

    args = pa['input_args'] + pa['input_kwargs'] + pa['output_args']
    binding_arg_base = len(args) + output_gap
    binding_arg_offsets = {arg['name']: i for i, arg in enumerate(python_binding_args)}

    def binding_arg_index(name):
        return binding_arg_base + binding_arg_offsets[name]

    def parse_binding_arg(name):
        binding_arg = python_binding_args[binding_arg_offsets[name]]
        expr, _ = parse_arg(binding_arg, binding_arg_index(name))
        return expr

    has_output = len(pa['output_args']) == 1
    tensor_options_arg = get_tensor_options(declaration)

    if tensor_options_arg is not None:
        # if our op has a tensor options arg, these are its scattered fields.
        # first some checks
        if has_output:
            raise RuntimeError('{}: tensor options with output arg'.format(declaration['name']))
        for arg in python_binding_args:
            typename = TENSOR_OPTIONS_FIELDS.get(arg['name'])
            if typename is None:
                raise RuntimeError(
                    '{}: unrecognized tensor options field \'{}\' in python binding arguments'.
                    format(declaration['name'], arg['name']))
            if typename != arg['simple_type']:
                raise RuntimeError(
                    '{}: unrecognized type \'{}\' for tensor options field \'{}\' in python binding arguments'.
                    format(declaration['name'], arg['type'], arg['name']))
        python_binding_argnames = [arg['name'] for arg in python_binding_args]
        if not all([key in python_binding_argnames for key in TENSOR_OPTIONS_FIELDS.keys()]):
            raise RuntimeError(
                '{}: incomplete tensor options args: {}'.
                format(declaration['name'], [arg['name'] for arg in python_binding_args]))
        # generate a gathering initialization of options struct
        argname = tensor_options_arg['name']
        inits.append(TENSOR_OPTIONS_DECL.substitute({
            'name': argname,
            'dtype': parse_binding_arg('dtype'),
            'layout': parse_binding_arg('layout'),
            'device': parse_binding_arg('device'),
            'requires_grad': parse_binding_arg('requires_grad'),
            'pin_memory': parse_binding_arg('pin_memory'),
        }))
        inits.append('torch::utils::maybe_initialize_cuda({});'.format(argname))
        # and add to op arg map
        argmap['options'] = {
            'value': argname,
            'formal': get_cpp_formal(tensor_options_arg),
        }

    else:
        # not the scattered fields of a tensor options - sort of a grab bag
        if 'dtype' in binding_arg_offsets:
            # we're an output-arg variant, check these args against output tensor
            if not has_output:
                raise RuntimeError(
                    '{}: dtype in python_binding_args without output arg'.
                    format(declaration['name']))
            if not all([name in binding_arg_offsets for name in ['layout', 'device']]):
                raise RuntimeError(
                    '{}: incomplete tensor options for output check'.
                    format(declaration['name']))
            check_type = PY_VARIABLE_CHECK_OUT_TYPE_HACK.substitute(
                out_idx=get_python_output_index(declaration),
                type_idx=binding_arg_index('dtype'),
                layout_idx=binding_arg_index('layout'),
                device_idx=binding_arg_index('device'),
            )
            inits.append(check_type)
        # we'll set requires_grad on outgoing tensor
        if 'requires_grad' not in binding_arg_offsets:
            raise RuntimeError(
                '{}: expected "requires_grad" in python_binding_args absent tensor options arg but found [{}]'.
                format(declaration['name'], [arg['name'] for arg in python_binding_args]))
        requires_grad = parse_binding_arg('requires_grad')
        set_requires_grad = '.set_requires_grad({})'.format(requires_grad)

    return argmap, inits, set_requires_grad


# handler for output/no-output overload pair
# (plugged into PY_VARIABLE_CASE as ${call_dispatch})
PY_VARIABLE_OUT = CodeTemplate("""\
if (_r.isNone(${out_idx})) {
  ${call_dispatch}
} else {
  ${call_dispatch_out}
}
""")

# handler for a single parsed signature - may be a single overload or
# a pair of overloads that whose signatures only differ in output params
PY_VARIABLE_CASE = CodeTemplate("""\
case ${i}: {
  ${body}
}
""")


def emit_dispatch_case(i, dictionary, is_python_method):
    """
    Emit dispatch code for a single parsed signature. This corresponds to either
    a single overload, or a pair that differ only in output params. In the latter
    case, a single signature is used for both and dispatching switches on the
    presence/absence of passed output args.
    - i: this signature's position in generated binding's signature list if number of
      signatures > 1, otherwise None
    - dictionary: contains a no-output overload declaration under 'base', and optionally
      a second overload with outputs under 'out'
    - true if we're generating a python method, in which case self is not parsed but
      passed directly
    """
    base_decl = dictionary['base']

    if 'out' in dictionary:
        # dispatch to output or no-output variant based on arg test
        out_decl = dictionary['out']
        out_idx = get_python_output_index(out_decl)
        output_gap = get_python_argc(out_decl) - get_python_argc(base_decl)

        call_dispatch = emit_single_dispatch(base_decl, is_python_method, output_gap)
        call_dispatch_out = emit_single_dispatch(out_decl, is_python_method)

        # dispatch output and no-output variants, branch on _r.isNone(<out_idx>)
        body = PY_VARIABLE_OUT.substitute(
            out_idx=out_idx,
            call_dispatch=call_dispatch,
            call_dispatch_out=call_dispatch_out,
        )
    else:
        # no-output version only
        body = emit_single_dispatch(base_decl, is_python_method)

    if i is not None:
        # generate case for ith overload
        return PY_VARIABLE_CASE.substitute(i=i, body=body)
    else:
        # only one overload, omit case wrapper
        return body

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

PY_NAMEDTUPLE_FIELDSDEF = CodeTemplate("""\
static PyStructSequence_Field ${fieldsname}[] = { ${fields,} {nullptr} };
""")

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
            fields = ['{{"{}", ""}}'.format(fn) for fn in fieldnames]
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
# method impl codegen
#

def get_pycname(name):
    return 'THPVariable_{}'.format(name)


def is_noarg_binding(overloads):
    return len(overloads) == 1 and get_python_argc(overloads[0]) == 0


# python binding for all overloads of a particular function/method
PY_VARIABLE_METHOD_VARARGS = CodeTemplate(r"""\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  ${method_header}
  static PythonArgParser parser({
    ${signatures}
  }, /*traceable=*/${traceable});

  ParsedArgs<${max_args}> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  ${check_has_torch_function}
  switch (_r.idx) {
    ${dispatch}
  }
  ${method_footer}
}

""")

# python binding for single-overload function/method
PY_VARIABLE_METHOD_VARARGS_SINGLETON = CodeTemplate("""\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  ${method_header}
  static PythonArgParser parser({
    ${signatures}
  }, /*traceable=*/${traceable});

  ParsedArgs<${max_args}> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);
  ${check_has_torch_function}
  ${dispatch}
  ${method_footer}
}

""")

# python binding for a method with no args, shortcuts parsing
PY_VARIABLE_METHOD_NOARGS = CodeTemplate("""\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args)
{
  ${method_header}
  ${dispatch}
  ${method_footer}
}

""")

TORCH_FUNCTION_CHECK = CodeTemplate("""\
if(_r.has_torch_function()) {
  return handle_torch_function(_r, args, kwargs, ${namespace}, ${modulename});
}
""")

# NOTE: we type the unpacked self as Tensor not Variable to avoid return type
# discrepancies on method resolution (e.g. Variable::detach_ returns void
# rather than Tensor &)
UNPACK_SELF = "Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;"


def method_impl(name, declarations, is_python_method, module):
    """
    Generate a python binding for all overloads of an op.
    """
    for declaration in declarations:
        # formals for python binding signature
        declaration['python_arglists'] = make_python_arglists(declaration, is_python_method)

    pycname = get_pycname(name)

    method_header = ['HANDLE_TH_ERRORS']
    method_header += emit_namedtuple_typedefs(declarations)
    method_header += [UNPACK_SELF] if is_python_method else []

    method_footer = ['END_HANDLE_TH_ERRORS']

    # emit dispatch
    if is_noarg_binding(declarations):
        dispatch = emit_single_dispatch(declaration, is_python_method)
        return PY_VARIABLE_METHOD_NOARGS.substitute(
            name=name,
            pycname=pycname,
            method_header=method_header,
            dispatch=dispatch,
            method_footer=method_footer,
        )

    method_footer = ['Py_RETURN_NONE;'] + method_footer

    grouped = group_overloads(declarations, is_python_method)
    is_singleton = len(grouped) == 1

    signatures = []
    dispatch = []
    for i, dictionary in enumerate(grouped):
        signature = dictionary['signature']
        signatures.append('"{}",'.format(signature))
        overload_index = i if not is_singleton else None
        dispatch.append(emit_dispatch_case(overload_index, dictionary, is_python_method))

    if is_singleton:
        template = PY_VARIABLE_METHOD_VARARGS_SINGLETON
    else:
        template = PY_VARIABLE_METHOD_VARARGS

    if module:
        check_has_torch_function = TORCH_FUNCTION_CHECK.substitute(
            namespace=NATIVE_NAMESPACE_MAPPING[module],
            modulename='"' + module + '"',
        )
    else:
        check_has_torch_function = ''

    max_args = max([get_python_argc(decl) for decl in declarations])
    traceable = 'true' if all(should_trace(d) for d in declarations) else 'false'

    return template.substitute(
        name=name,
        pycname=pycname,
        method_header=method_header,
        max_args=max_args,
        signatures=signatures,
        traceable=traceable,
        check_has_torch_function=check_has_torch_function,
        dispatch=dispatch,
        method_footer=method_footer,
    )


#
# forward declarations
#

PY_VARIABLE_FUNCTION_VARARGS_FORWARD_DECLARATION = CodeTemplate("""\
static PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs);
""")

PY_VARIABLE_FUNCTION_NOARGS_FORWARD_DECLARATION = CodeTemplate("""\
static PyObject * ${pycname}(PyObject* self_, PyObject* args);
""")


def forward_decls(name, declarations, is_python_method, module):
    if is_python_method:
        return []

    if is_noarg_binding(declarations):
        template = PY_VARIABLE_FUNCTION_NOARGS_FORWARD_DECLARATION
    else:
        template = PY_VARIABLE_FUNCTION_VARARGS_FORWARD_DECLARATION

    pycname = get_pycname(name)
    return [template.substitute(pycname=pycname)]


#
# method def (binding table entry) codegen
#

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

# PyMethodDef entry for binary op, throws not implemented error
PY_VARIABLE_METHOD_BINOP_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)${pycfunc_voidcast}TypeError_to_NotImplemented_<${pycname}>, ${flags}, NULL},""")

# PyMethodDef entry
PY_VARIABLE_METHOD_DEF = CodeTemplate("""\
{"${name}", (PyCFunction)${pycfunc_voidcast}${pycname}, ${flags}, NULL},""")


def method_def(name, declarations, is_python_method, module):
    """
    Generate method def entry.
    """
    pycname = get_pycname(name)

    if is_noarg_binding(declarations):
        pycfunc_voidcast = ''
        flags = 'METH_NOARGS' if is_python_method else 'METH_VARARGS | METH_KEYWORDS'
    else:
        pycfunc_voidcast = '(void(*)(void))'
        flags = 'METH_VARARGS | METH_KEYWORDS'

    if module == "torch":
        flags += ' | METH_STATIC'

    if name in BINARY_OP_NAMES:
        def_template = PY_VARIABLE_METHOD_BINOP_DEF
    else:
        def_template = PY_VARIABLE_METHOD_DEF

    return def_template.substitute(
        name=name,
        pycname=pycname,
        pycfunc_voidcast=pycfunc_voidcast,
        flags=flags,
    )

#
# overload sorting and grouping
#

def group_overloads(declarations, is_python_method):
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
        signature = get_python_signature(declaration, is_python_method, skip_outputs=True)
        v = grouped[signature]
        if declaration['name'].endswith('_out'):
            v['out'] = declaration
            # prefer the signature with optional out=... arguments
            v['signature'] = get_python_signature(declaration, is_python_method)
        else:
            v['base'] = declaration
            if 'signature' not in v:
                v['signature'] = signature

    result = []
    for x, dictionary in sorted(grouped.items()):
        if 'base' not in dictionary:
            raise RuntimeError(
                "'base' not in dictionary for {}. keys are {}".format(
                    x, list(dictionary.keys())))
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


#
# python signature codegen
#

SCHEMA_DEFAULT_CONVERSION_HACKS = {
    'nullptr': 'None',
    'c10::nullopt': 'None',
    '{}': 'None',
}

def get_schema_formal(arg, is_python_method):
    name = arg['name']
    typename = arg['simple_type']

    # TODO: remove this and make optional types in simple_type to be consistent across
    # tensor and other types after make Tensor? be optional instead of undefined
    if arg.get('is_nullable') and '?' not in typename:
        typename = '{}?'.format(typename)

    # s/self/input/ outside method bindings.
    # TODO remove this? doesn't rename in codegen, it's just for the parse string
    if name == 'self' and typename == 'Tensor' and not is_python_method:
        name = 'input'

    size = arg.get('size')
    if size is not None:
        typename = '{}[{}]'.format(typename, size)

    # default
    default = arg.get('default')
    if default is not None:
        default = SCHEMA_DEFAULT_CONVERSION_HACKS.get(default, default)
        return '{} {}={}'.format(typename, name, default)
    else:
        return '{} {}'.format(typename, name)


PYTHON_ARG_PARSER_SCHEMA = CodeTemplate("""\
${name}(${schema_formals})${deprecated}""")


def get_python_signature(declaration, is_python_method, skip_outputs=False):
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

    python_args = get_python_args(declaration)
    if skip_outputs:
        python_args = [arg for arg in python_args if not is_output(arg)]

    schema_formals = [get_schema_formal(arg, is_python_method) for arg in python_args]
    positional_argc = len(declaration['python_arglists']['input_args'])
    if len(python_args) > positional_argc:
        schema_formals.insert(positional_argc, '*')

    # Python function signature.
    # This is the string that we give to FunctionParameter, which is
    # then parsed into the actual structure which we do parsing with.
    name = op_name(declaration)
    deprecated = '|deprecated' if declaration.get('deprecated', False) else ''
    return PYTHON_ARG_PARSER_SCHEMA.substitute(
        name=name,
        schema_formals=schema_formals,
        deprecated=deprecated,
    )

#
# op args to python parsed args transform
#

def get_python_args(decl):
    arglists = decl['python_arglists']
    return \
        arglists['input_args'] + \
        arglists['input_kwargs'] + \
        arglists['output_args'] + \
        arglists['python_binding_args']


def get_python_argc(decl):
    return sum([len(arglist) for arglist in decl['python_arglists'].values()])


def get_python_output_index(decl):
    arglists = decl['python_arglists']
    return len(arglists['input_args'] + arglists['input_kwargs'])


def make_python_arglists(declaration, is_python_method):
    # produces python-ready args converted from declaration['args'],
    # partitioned into sublists by category. subslists are order, so
    # the final python arglist can be recovered by simple flattening
    # (see get_python_args())

    # partition args into sublists

    args = declaration['arguments']

    input_args = []
    input_kwargs = []
    output_args = []

    current_input_args = input_args
    for arg in args:
        if is_output(arg):
            output_args.append(arg)
        else:
            if arg.get('kwarg_only', False):
                current_input_args = input_kwargs
            current_input_args.append(arg)

    # adjustments

    # positional inputs:
    # - filter self when we're generating a method binding.else - there, it comes in as
    #   a separate Python param, not in args array
    def include(arg):
        return not (is_tensor_self(arg) and is_python_method)
    input_args = [arg for arg in input_args if include(arg)]

    # keyword inputs:
    # - filter options. after loading the yaml, an upstream step has gathered dtype,
    #   layout et al into a single tensor options arg. here we reintroduce the originals
    input_kwargs = [arg for arg in input_kwargs if not is_tensor_options(arg)]

    # outputs:
    # - coalesce multiple output args into a single 'out' arg w/type TensorList.
    # - force a default. This is so we can use this sig for both out and non-out variants
    num_outputs = len(output_args)
    if num_outputs > 1:
        for arg in output_args:
            if not arg['simple_type'] == 'Tensor':
                raise RuntimeError(
                    '{}: unsupported output argument type {}'.
                    format(declaration['name'], arg['type']))
        typename = 'TensorList'
        output_args = [{
            'default': 'None',
            'kwarg_only': True,
            'name': 'out',
            'output': True,
            'scatter_args': output_args,
            'simple_type': typename,
            'size': num_outputs,
            'type': typename,
        }]
    elif num_outputs == 1:
        output_arg = output_args[0].copy()
        output_arg['default'] = 'None'
        output_args = [output_arg]

    # make python binding args
    # these are the (re)scattered versions of the options arg omitted above.
    # TODO because these aren't guaranteed to be 100% faithful to the original
    # versions in the yaml, this recreation is a potential source of drift between
    # eager and JIT. Pull this logic out to a shared place.
    python_binding_args = make_python_binding_args(declaration)

    return {
        'input_args': input_args,
        'input_kwargs': input_kwargs,
        'output_args': output_args,
        'python_binding_args': python_binding_args,
    }

#
# python binding args
#

# TODO blowtorch
def dtype_default_type_hack(name):
    if name.startswith('randperm') or name == 'tril_indices' or name == 'triu_indices':
        return 'torch.int64'
    else:
        return 'None'


def make_python_binding_args(declaration):
    """
    Given various properties of a declaration, build a set of scattered python binding args.
    """
    name = declaration['name']
    python_binding_arguments = []
    has_tensor_input_arg = False
    has_options_arg = False
    for arg in declaration['arguments']:
        if is_output(arg):
            continue
        typename = arg['simple_type']
        if typename in ['Tensor', 'TensorList']:
            has_tensor_input_arg = True
        elif typename == 'TensorOptions':
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

    if is_factory_function or has_options_arg:
        default_type = dtype_default_type_hack(name)
        py_default_dtype = 'self.scalar_type()' if is_like_or_new_function_with_options else None
        dtype_arg = {
            'default': default_type,
            'dynamic_type': 'ScalarType',
            'kwarg_only': True,
            'name': 'dtype',
            'type': 'const ScalarType &',
            'simple_type': 'ScalarType',
            'python_default_init': py_default_dtype,
        }
        python_binding_arguments.append(dtype_arg)

    if is_factory_function or is_like_or_new_function_with_options:
        py_default_layout = 'layout_from_backend(self.options().backend())' if is_like_or_new_function_with_options else None
        layout_arg = {
            'default': 'torch.strided',
            'dynamic_type': 'Layout',
            'kwarg_only': True,
            'name': 'layout',
            'type': 'c10::optional<Layout>',
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

#
# declaration derived props, utils, etc.
# declarations are dicts loaded from Declarations.yaml,
# passed to our codegen methods by callers in gen_autograd
#

def is_tensor_self(arg):
    return arg['name'] == 'self' and arg['simple_type'] == 'Tensor'


def is_tensor_options(arg):
    return arg['simple_type'] == 'TensorOptions'


def is_scatter(arg):
    return arg.get('scatter_args') is not None

def is_output(arg):
    return arg.get('output', False)


def has_outputs(declaration):
    return any([is_output(arg) for arg in declaration['arguments']])


def get_tensor_options(declaration):
    args = [arg for arg in declaration['arguments'] if is_tensor_options(arg)]
    if len(args) == 0:
        return None
    if len(args) != 1:
        raise RuntimeError(
            '{}: multiple tensor options arguments'.
            format(declaration['name']))
    return args[0]


def has_tensor_options(declaration):
    return get_tensor_options(declaration) is not None


def is_tensor_method(declaration):
    return 'Tensor' in declaration['method_of']


def is_torch_function(declaration):
    return 'namespace' in declaration['method_of']


def is_nn_module_function(declaration):
    return declaration.get('python_module') == 'nn'


def function_namespace(declaration):
    # TODO look into why these can't all be 'torch' calls
    if has_tensor_options(declaration) or op_name(declaration).endswith('_like'):
        return 'torch'
    else:
        return 'at'


def op_name(declaration):
    name = declaration['name']
    if has_outputs(declaration):
        if not name.endswith("_out"):
            raise RuntimeError(
                '{} has output params, expecting name ending with \'_out\''.
                format(declaration['name']))
        return name[:-4]
    else:
        if name.endswith("_out"):
            raise RuntimeError(
                '{}: name ends with \'_out\', expecting output params'.
                format(declaration['name']))
        return name
