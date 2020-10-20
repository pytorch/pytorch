# Generates Python bindings for ATen functions
#
# The bindings are generated as methods on python_variable or functions on the
# torch._C._nn. torch._C._fft, or torch._C._linalg objects.
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
from .utils import write, is_tensor_method

from tools.codegen.code_template import CodeTemplate
from tools.codegen.api.python import *
from tools.codegen.gen import cpp_string, with_native_function
from tools.codegen.model import *

from typing import Dict, Optional, List, Any

#
# declarations blocklist
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
    '_indexCopy_', '_cumsum.*', '_cumprod.*', '_sum.*', '_prod.*',
    '_th_.*', '_thnn_.*',
    'arange.*', 'range.*', '_solve.*', '_inverse.*',
    'full(_out)?',
    '_cholesky.*', '_triangular_solve.*', '_qr.*', '_symeig.*', '_svd.*',
    'slice', 'randint(_out)?',
    'item', '_local_scalar_dense', 'to',
    'copy_sparse_to_sparse_', 'copy_',
    'numpy_T',  # this needs to be an attribute in Python, not a function
    'nonzero(_(out|numpy))?',
    'set_quantizer_',  # return types not supported yet
    'set_data',
    '.*_overrideable',  # overrideable functions for backend extension
    'data', 'is_leaf', 'output_nr', '_version', 'requires_grad_', 'retain_grad', 'set_'
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
    "torch.nn": "THPNNVariableFunctionsModule",
    "torch.fft": "THPFFTVariableFunctionsModule",
    "torch.linalg": "THPLinalgVariableFunctionsModule",
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


def get_py_fft_functions(declarations):
    """
    Get declarations (grouped by name) which should be generated
    as functions in the "fft" module.
    """
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                is_fft_module_function(declaration))

    return group_declarations_by_op_name([d for d in declarations if should_bind(d)])


def gen_py_fft_functions(out, declarations, template_path):
    """
    Generate functions in the "fft" module.
    """
    PY_FFT_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_fft_functions.cpp')

    py_fft_functions = get_py_fft_functions(declarations)

    env = create_python_bindings(py_fft_functions, is_python_method=False, module="torch.fft")

    write(out, 'python_fft_functions.cpp', PY_FFT_FUNCTIONS_CPP, env)

def get_py_linalg_functions(declarations):
    """
    Get declarations (grouped by name) which should be generated
    as functions in the "linalg" module.
    """
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                is_linalg_module_function(declaration))

    return group_declarations_by_op_name([d for d in declarations if should_bind(d)])


def gen_py_linalg_functions(out, declarations, template_path):
    """
    Generate functions in the "linalg" module.
    """
    PY_LINALG_FUNCTIONS_CPP = CodeTemplate.from_file(template_path + '/python_linalg_functions.cpp')

    py_linalg_functions = get_py_linalg_functions(declarations)

    env = create_python_bindings(py_linalg_functions, is_python_method=False, module="torch.linalg")

    write(out, 'python_linalg_functions.cpp', PY_LINALG_FUNCTIONS_CPP, env)


def get_py_torch_functions(declarations):
    """
    Get declarations (grouped by name) which should be generated
    as functions in the "torch" module.
    """
    def should_bind(declaration):
        return (should_generate_python_binding(declaration) and
                not is_nn_module_function(declaration) and
                not is_fft_module_function(declaration) and
                not is_linalg_module_function(declaration) and
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

        for declaration in overload_decls:
            # TODO: change all methods to directly process python signatures instead of decls.
            declaration['python_signature'] = decl_to_python_signature(declaration, method=is_python_method)
            declaration['native_function'] = decl_to_native_function(declaration)

        py_methods.append(method_impl(name, overload_decls, is_python_method, module))
        py_method_defs.append(method_def(name, overload_decls, is_python_method, module))
        py_forwards.extend(forward_decls(name, overload_decls, is_python_method, module))

    return {
        'py_forwards': py_forwards,
        'py_methods': py_methods,
        'py_method_defs': py_method_defs,
    }


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
    python_sig = base_decl['python_signature']

    if 'out' in dictionary:
        # dispatch to output or no-output variant based on arg test
        out_decl = dictionary['out']
        python_sig = out_decl['python_signature']  # prefer output variant

        out_idx = get_python_output_index(out_decl)

        call_dispatch = emit_single_dispatch(python_sig, base_decl, is_python_method)
        call_dispatch_out = emit_single_dispatch(python_sig, out_decl, is_python_method)

        # dispatch output and no-output variants, branch on _r.isNone(<out_idx>)
        body = PY_VARIABLE_OUT.substitute(
            out_idx=out_idx,
            call_dispatch=call_dispatch,
            call_dispatch_out=call_dispatch_out,
        )
    else:
        # no-output version only
        body = emit_single_dispatch(python_sig, base_decl, is_python_method)

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
  auto _r = parser.parse(${self_}, args, kwargs, parsed_args);
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
  auto _r = parser.parse(${self_}, args, kwargs, parsed_args);
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
  ${check_has_torch_function}
  ${dispatch}
  ${method_footer}
}

""")

TORCH_FUNCTION_CHECK = CodeTemplate("""\
if(_r.has_torch_function()) {
  return handle_torch_function(_r, ${self_}, args, kwargs, ${namespace}, ${modulename});
}
""")

TORCH_FUNCTION_CHECK_NOARGS = CodeTemplate("""\
if(check_has_torch_function(self_)) {
  return handle_torch_function(self_, ${name});
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
    pycname = get_pycname(name)

    method_header = ['HANDLE_TH_ERRORS']
    method_header += emit_namedtuple_typedefs(declarations)
    method_header += [UNPACK_SELF] if is_python_method else []

    method_footer = ['END_HANDLE_TH_ERRORS']

    check_has_torch_function = TORCH_FUNCTION_CHECK_NOARGS.substitute(
        name='"' + name + '"',
    ) if is_python_method else ''

    # emit dispatch
    if is_noarg_binding(declarations):
        python_sig = declarations[0]['python_signature']
        dispatch = emit_single_dispatch(python_sig, declarations[0], is_python_method)
        return PY_VARIABLE_METHOD_NOARGS.substitute(
            name=name,
            pycname=pycname,
            method_header=method_header,
            dispatch=dispatch,
            method_footer=method_footer,
            check_has_torch_function=check_has_torch_function,
        )

    method_footer = ['Py_RETURN_NONE;'] + method_footer

    grouped = group_overloads(declarations, is_python_method)
    is_singleton = len(grouped) == 1

    signatures = []
    dispatch = []
    for i, dictionary in enumerate(grouped):
        signature = dictionary['signature']
        signatures.append(f'{cpp_string(str(signature))},')
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
            self_="self_" if is_python_method else "nullptr",
        )
    else:
        check_has_torch_function = TORCH_FUNCTION_CHECK.substitute(
            namespace="THPVariableClass",
            modulename='"torch.Tensor"',
            self_="self_" if is_python_method else "nullptr",
        )

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
        self_="self_" if is_python_method else "nullptr",
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

    def dynamic_type(arg):
        return arg['dynamic_type']

    def is_coord_smaller(arg1, arg2):
        return dynamic_type(arg1) == 'Scalar' and arg2['dynamic_type'] == 'Tensor'

    def is_smaller(d1, d2):
        """Returns True if d1 < d2 in the partial order."""
        args1, args2 = d1['base']['arguments'], d2['base']['arguments']
        if len(args1) != len(args2):
            return False
        any_smaller = any(is_coord_smaller(arg1, arg2) for arg1, arg2 in zip(args1, args2))
        all_smaller_or_equal = all(dynamic_type(arg1) == dynamic_type(arg2) or
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

def get_python_signature(declaration, is_python_method, skip_outputs=False):
    return declaration['python_signature'].signature_str(skip_outputs=skip_outputs)


#
# op args to python parsed args transform
#

def get_python_argc(decl):
    return len(decl['python_signature'].arguments())


def get_python_output_index(decl):
    ps: PythonSignature = decl['python_signature']
    return len(ps.input_args) + len(ps.input_kwargs)


#
# declaration derived props, utils, etc.
# declarations are dicts loaded from Declarations.yaml,
# passed to our codegen methods by callers in gen_autograd
#

def is_output(arg):
    return arg.get('output', False)


def has_outputs(declaration):
    return any([is_output(arg) for arg in declaration['arguments']])


def is_torch_function(declaration):
    return 'namespace' in declaration['method_of']


def is_nn_module_function(declaration):
    return declaration.get('python_module') == 'nn'


def is_fft_module_function(declaration):
    return declaration.get('python_module') == 'fft'


def is_linalg_module_function(declaration):
    return declaration.get('python_module') == 'linalg'


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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                       Codegen API Integration
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# These helper functions allow us to call the new codegen API from the
# old codegen script (which operates on Declarations.yaml).

# TODO: remove all these HACKs after migration is completed!

# function schema str -> NativeFunction
NF_TABLE: Optional[Dict[str, NativeFunction]] = None

def init(native_yaml_path: str) -> None:
    from tools.codegen.gen import parse_native_yaml
    global NF_TABLE
    NF_TABLE = {str(f.func): f for f in parse_native_yaml(native_yaml_path)}

# Multiple decl entries can map to the same native function (because of deprecated decl).
def decl_to_native_function(decl: Dict[str, Any]) -> NativeFunction:
    assert NF_TABLE is not None, 'need to initialize codegen.api.python with init()'
    function_schema_str = decl['schema_string']
    assert function_schema_str.startswith('aten::'), f'unknown namespace: {function_schema_str}'
    function_schema_str = function_schema_str[len('aten::'):]
    assert function_schema_str in NF_TABLE, f'cannot find func: {function_schema_str}'
    return NF_TABLE[function_schema_str]

# Each decl entry has unique python signature.
def decl_to_python_signature(decl: Dict[str, Any], *, method: bool) -> PythonSignature:
    f = decl_to_native_function(decl)

    @with_native_function
    def go(f: NativeFunction) -> PythonSignature:
        return signature(f, method=method)

    python_sig = go(f)

    if decl.get('deprecated', False):
        # TODO: directly load 'deprecated.yaml'.
        # deprecated.yaml doesn't have complete type information, we need
        # leverage the source signature (to which it delegates the call).
        # Deprecated signature might reorder input_args and input_kwargs,
        # but never changes output_args nor python_binding_args (if any?),
        # so here we only look into these two types of args.
        src_args: Dict[str, PythonArgument] = {a.name: PythonArgument(
            name=a.name,
            type=a.type,
            cpp_type_str=a.cpp_type_str,
            default=None,
            default_init=None,
        ) for a in itertools.chain(python_sig.input_args, python_sig.input_kwargs)}
        args: List[Dict[str, Any]] = decl['arguments']
        input_arg_names: List[str] = \
            list(str(a['name']) for a in args if not a['kwarg_only'] and not a['output'])
        input_kwarg_names: List[str] = \
            list(str(a['name']) for a in args if a['kwarg_only'] and not a['output'])
        python_sig = PythonSignatureDeprecated(
            name=python_sig.name,
            input_args=tuple(src_args[n] for n in input_arg_names if not method or n != 'self'),
            input_kwargs=tuple(src_args[n] for n in input_kwarg_names),
            output_args=python_sig.output_args,
            tensor_options_args=python_sig.tensor_options_args,
            method=python_sig.method,
            deprecated_args_names=tuple(str(a['name']) for a in args),
            deprecated_args_exprs=tuple(decl.get('call_args')),
        )
    return python_sig


def emit_single_dispatch(ps: PythonSignature, decl: Dict[str, Any], method: bool) -> str:
    """
    Emit dispatch code for a single declared overload.
    """
    f = decl['native_function']

    @with_native_function
    def go(f: NativeFunction) -> str:
        # header comments
        deprecated = '[deprecated] ' if ps.deprecated else ''
        schema_comment = f'// {deprecated}aten::{f.func}'

        # dispatch lambda signature
        name = decl['name']
        lambda_formals = ', '.join(map(lambda a: f"{a.type_str} {a.name}",
                                       dispatch_lambda_args(ps, f, method=method)))
        lambda_return = dispatch_lambda_return_str(f)

        # dispatch lambda body
        dispatch_callee = cpp_dispatch_target(f)
        dispatch_args = ', '.join(cpp_dispatch_exprs(f, method, python_signature=ps))

        # from arg parser outputs to dispatch lambda arguments
        parser_outputs = arg_parser_output_exprs(ps, f, method=method)
        lambda_arg_exprs = dispatch_lambda_exprs(ps, f, method=method)
        inits = '\n'.join(lambda_arg_exprs.inits)
        lambda_args = ', '.join(lambda_arg_exprs.exprs)

        # scatter fields
        set_requires_grad = f'.set_requires_grad({parser_outputs["requires_grad"].expr})' \
            if ps.tensor_options_args and not has_tensor_options(f) else ''

        auto_no_gil = '' if decl['with_gil'] else 'pybind11::gil_scoped_release no_gil;'

        namedtuple_typeref = decl['namedtuple_typeref']

        if lambda_return == 'void':
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  {auto_no_gil}
  {dispatch_callee}({dispatch_args});
}};
dispatch_{name}({lambda_args}){set_requires_grad};
Py_RETURN_NONE;
"""
        else:
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  {auto_no_gil}
  return {dispatch_callee}({dispatch_args});
}};
return wrap({namedtuple_typeref}dispatch_{name}({lambda_args}){set_requires_grad});
"""

    return go(f)
