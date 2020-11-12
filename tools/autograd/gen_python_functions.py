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
import itertools
import re
from .gen_trace_type import should_trace
from .utils import write, is_tensor_method

from tools.codegen.code_template import CodeTemplate
from tools.codegen.api.python import *
from tools.codegen.gen import cpp_string, with_native_function
from tools.codegen.model import *

from typing import Dict, Optional, List, Any, Tuple, Set, Sequence

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
    py_methods: List[str] = []
    py_method_defs: List[str] = []
    py_forwards: List[str] = []

    for name in sorted(python_functions.keys()):
        overloads = list(decl_to_signature_function_pair(decl, method=is_python_method)
                         for decl in python_functions[name])
        py_methods.append(method_impl(name, module, overloads, method=is_python_method))
        py_method_defs.append(method_def(name, module, overloads, method=is_python_method))
        py_forwards.extend(forward_decls(name, overloads, method=is_python_method))

    return {
        'py_forwards': py_forwards,
        'py_methods': py_methods,
        'py_method_defs': py_method_defs,
    }

#
# declaration derived props, utils, etc.
# declarations are dicts loaded from Declarations.yaml,
# passed to our codegen methods by callers in gen_autograd
#

def get_pycname(name: str) -> str:
    return f'THPVariable_{name}'


def is_noarg(overloads: Sequence[PythonSignatureNativeFunctionPair]) -> bool:
    return len(overloads) == 1 and overloads[0].signature.arguments_count() == 0


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
#                         Named Tuple Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# TODO: remove the copy of this method in 'tools/pyi/gen_pyi.py'.
@with_native_function
def namedtuple_fieldnames(f: NativeFunction) -> List[str]:
    returns = f.func.returns
    if len(returns) <= 1 or all(map(lambda r: r.name is None, returns)):
        return []
    else:
        if any(map(lambda r: r.name is None, returns)):
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

        return list(map(lambda r: str(r.name), returns))

@with_native_function
def gen_namedtuple_typename_key(f: NativeFunction) -> str:
    name = cpp.name(f.func)
    fieldnames = namedtuple_fieldnames(f)
    return '_'.join([name] + fieldnames)

def emit_namedtuple_typedefs(
    overloads: Sequence[PythonSignatureNativeFunctionPair]
) -> Tuple[List[str], Dict[str, str]]:
    """
    Generate block of named tuple type def inits, and add typeref snippets
    to declarations that use them
    """
    flddefnames: Dict[str, str] = {}  # map from unique field name lists to field def name
    flddefs: List[str] = []           # field def declarations
    typenames: Dict[str, str] = {}    # map from unique name + field name lists to typedef name
    typedefs: List[str] = []          # typedef declarations and init code

    for overload in overloads:
        fieldnames = namedtuple_fieldnames(overload.function)
        if not fieldnames:
            continue

        fn_key = '_'.join(fieldnames)
        fieldsname = flddefnames.get(fn_key)
        if fieldsname is None:
            fieldsname = f'NamedTuple_fields{"" if not flddefs else len(flddefs)}'
            flddefnames[fn_key] = fieldsname
            fields = ', '.join(f'{{"{fn}", ""}}' for fn in fieldnames)
            flddefs.append(f"""\
static PyStructSequence_Field {fieldsname}[] = {{ {fields},  {{nullptr}} }};
""")

        name = cpp.name(overload.function.func)  # use @with_native_function?
        tn_key = gen_namedtuple_typename_key(overload.function)
        typename = typenames.get(tn_key)
        if typename is None:
            typename = f'NamedTuple{"" if not typedefs else len(typedefs)}'
            typenames[tn_key] = typename
            typedefs.append(f"""\
static PyTypeObject {typename};
static bool {typename}_initialized = false;
if (!{typename}_initialized) {{
  {typename}_initialized = true;
  static PyStructSequence_Desc desc = {{ "torch.return_types.{name}", nullptr, {fieldsname}, {len(fieldnames)} }};
  PyStructSequence_InitType(&{typename}, &desc);
  {typename}.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
}}
""")

    return flddefs + typedefs, typenames

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         Method Impl Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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

# handler for a single parsed signature - may be a single overload or
# a pair of overloads that whose signatures only differ in output params
# (plugged into PY_VARIABLE_METHOD_VARARGS as an item in ${dispatch})
PY_VARIABLE_CASE = CodeTemplate("""\
case ${overload_index}: {
  ${body}
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

def method_impl(
    name: str,
    module: str,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool
) -> str:
    """
    Generate a python binding for all overloads of an op.
    """
    pycname = get_pycname(name)
    noarg = is_noarg(overloads)
    namedtuple_inits, namedtuple_typenames = emit_namedtuple_typedefs(overloads)

    method_header = ['HANDLE_TH_ERRORS']
    method_header += namedtuple_inits
    method_header += [
        "Tensor& self = reinterpret_cast<THPVariable*>(self_)->cdata;"
    ] if method else []

    method_footer = ([] if noarg else ['Py_RETURN_NONE;']) + ['END_HANDLE_TH_ERRORS']

    traceable = 'true' if all(should_trace(o.function) for o in overloads) else 'false'

    grouped_overloads: Sequence[PythonSignatureGroup] = group_overloads(overloads)
    is_singleton = len(grouped_overloads) == 1
    signatures: List[str] = []
    dispatch: List[str] = []
    for overload_index, overload in enumerate(grouped_overloads):
        signature = overload.signature.signature_str()
        signatures.append(f'{cpp_string(str(signature))},')
        dispatch_body = emit_dispatch_case(overload, namedtuple_typenames)
        dispatch.append(
            PY_VARIABLE_CASE.substitute(overload_index=overload_index, body=dispatch_body)
            if not is_singleton else dispatch_body)

    if noarg:
        template = PY_VARIABLE_METHOD_NOARGS
    elif is_singleton:
        template = PY_VARIABLE_METHOD_VARARGS_SINGLETON
    else:
        template = PY_VARIABLE_METHOD_VARARGS

    return template.substitute(
        name=name,
        pycname=pycname,
        method_header=method_header,
        max_args=max(map(lambda o: o.signature.arguments_count(), overloads)),
        signatures=signatures,
        traceable=traceable,
        check_has_torch_function=gen_has_torch_function_check(
            name=name,
            module=module,
            noarg=noarg,
            method=method,
        ),
        dispatch=dispatch,
        method_footer=method_footer,
        self_="self_" if method else "nullptr",
    )

def gen_has_torch_function_check(name: str, module: str, *, noarg: bool, method: bool) -> str:
    if noarg:
        if method:
            return f"""\
if(check_has_torch_function(self_)) {{
  return handle_torch_function(self_, "{name}");
}}
"""
        else:
            return ''

    self_ = "self_" if method else "nullptr"
    namespace = {
        "torch": "THPVariableFunctionsModule",
        "torch.nn": "THPNNVariableFunctionsModule",
        "torch.fft": "THPFFTVariableFunctionsModule",
        "torch.linalg": "THPLinalgVariableFunctionsModule",
    }[module] if module else "THPVariableClass"

    return f"""\
if(_r.has_torch_function()) {{
  return handle_torch_function(_r, {self_}, args, kwargs, {namespace}, "{module or "torch.Tensor"}");
}}
"""

# handler for output/no-output overload pair
PY_VARIABLE_OUT = CodeTemplate("""\
if (_r.isNone(${out_idx})) {
  ${call_dispatch}
} else {
  ${call_dispatch_out}
}
""")

def emit_dispatch_case(
    overload: PythonSignatureGroup,
    namedtuple_typenames: Dict[str, str],
) -> str:
    """
    Emit dispatch code for a single parsed signature. This corresponds to either
    a single native function, or a pair that differ only in output params. In the
    latter case, a single python signature is used for both and dispatching
    switches on the presence/absence of passed output args.
    """
    if overload.outplace is not None:
        # dispatch output and no-output variants, branch on _r.isNone(<out_idx>)
        return PY_VARIABLE_OUT.substitute(
            out_idx=overload.signature.output_idx(),
            call_dispatch=emit_single_dispatch(
                overload.signature, overload.base, namedtuple_typenames),
            call_dispatch_out=emit_single_dispatch(
                overload.signature, overload.outplace, namedtuple_typenames),
        )
    else:
        # no-output version only
        return emit_single_dispatch(
            overload.signature, overload.base, namedtuple_typenames)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                    Forward Declarations Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def forward_decls(
    name: str,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool
) -> Tuple[str, ...]:
    if method:
        return ()

    pycname = get_pycname(name)
    if is_noarg(overloads):
        return (f"""\
static PyObject * {pycname}(PyObject* self_, PyObject* args);
""",)
    else:
        return (f"""\
static PyObject * {pycname}(PyObject* self_, PyObject* args, PyObject* kwargs);
""",)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#              Method Def (Binding Table Entry) Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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

def method_def(
    name: str,
    module: str,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool
) -> str:
    """
    Generate method def entry.
    """
    pycname = get_pycname(name)

    if is_noarg(overloads):
        pyfunc_cast = ''
        flags = 'METH_NOARGS' if method else 'METH_VARARGS | METH_KEYWORDS'
    else:
        pyfunc_cast = 'castPyCFunctionWithKeywords'
        flags = 'METH_VARARGS | METH_KEYWORDS'

    if module == "torch":
        flags += ' | METH_STATIC'

    if name in BINARY_OP_NAMES:
        # PyMethodDef entry for binary op, throws not implemented error
        return f"""\
{{"{name}", {pyfunc_cast}(TypeError_to_NotImplemented_<{pycname}>), {flags}, NULL}},"""
    else:
        # PyMethodDef entry
        return f"""\
{{"{name}", {pyfunc_cast}({pycname}), {flags}, NULL}},"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                   Overload Sorting and Grouping
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def group_overloads(
    overloads: Sequence[PythonSignatureNativeFunctionPair]
) -> Sequence[PythonSignatureGroup]:
    bases: Dict[str, PythonSignatureNativeFunctionPair] = {}
    outplaces: Dict[str, PythonSignatureNativeFunctionPair] = {}

    # first group by signature ignoring out arguments
    for overload in overloads:
        sig = overload.signature.signature_str(skip_outputs=True)
        if overload.function.func.is_out_fn():
            if sig in outplaces:
                raise RuntimeError(
                    f'Found duplicated function definition:\n- {overload.function.func}.\n'
                    f'Existing definition:\n- {outplaces[sig].function.func}.'
                )
            outplaces[sig] = overload
        else:
            if sig in bases:
                raise RuntimeError(
                    f'Found duplicated function definition:\n- {overload.function.func}.\n'
                    f'Existing definition:\n- {bases[sig].function.func}.'
                )
            bases[sig] = overload

    for sig, out in outplaces.items():
        if sig not in bases:
            candidates: List[str] = []
            for overload in overloads:
                if str(overload.function.func.name.name) == str(out.function.func.name.name) \
                        and not overload.function.func.is_out_fn() \
                        and not overload.signature.deprecated:
                    candidates.append(overload.signature.signature_str(skip_outputs=True))
            out_sig = out.signature.signature_str()
            raise RuntimeError(
                f'While identifying overloads, we found an out schema {out_sig} without a corresponding non-out variant. '
                f'We expected the non-out variant to have schema: \n- {sig}\nPlease check that you spelled the schema '
                'correctly in native_functions.yaml. We discovered the following candidate(s): \n'
                + '\n'.join(f'- {candidate}' for candidate in candidates))

    grouped: List[PythonSignatureGroup] = []
    for sig, base in bases.items():
        outplace = outplaces.get(sig)
        grouped.append(PythonSignatureGroup(
            # prefer the signature with optional out=... arguments because it's the
            # superset that can be used to parse input for both base and outplace.
            signature=outplace.signature if outplace is not None else base.signature,
            base=base.function,
            outplace=outplace.function if outplace is not None else None,
        ))

    return sort_overloads(grouped)

# This function declares a partial order on declarations, and sorts them according
# to its linear extension. This is necessary, because there's some ambiguity in the
# choice of overload, and we want a different order.
#
# See Note[Order of overloads matters]
#
# A few examples of ambiguous python signature pairs.
#
#   All parameters have the same type, except one taking Tensor the other taking
#   Scalar. A numeric PyObject can be casted into Tensor, and a zero-dim Tensor
#   object can be accepted as Scalar type parameter (see python_arg_parser.cpp).
#   Therefore, same input arguments might be accepted by either python signature.
#   We want to always parse the one taking Tensor first.
#
#     bitwise_and(Tensor input, Tensor other, *, Tensor out=None)
#     bitwise_and(Tensor input, Scalar other, *, Tensor out=None)
#
#   If they have different number of parameters then they are not ambiguous - but
#   the difference on output param can be ignored as it's optional.
#
#     multiply(Tensor input, Tensor other, *, Tensor out=None)
#     multiply(Tensor input, Scalar other)
#
#   Both positional args and keyword-only args are considered together.
#
#     subtract(Tensor other, *, Scalar alpha=1)
#     subtract(Scalar other, Scalar alpha=1)
#
# A few ambiguous cases which it does NOT handle yet.
#
#   If there is any difference in other parameters besides the Tensor/Scalar
#   difference, then they are not considered ambiguous by this method anymore.
#   However, the difference could be too trivial to disambiguate.
#
#     foo(Tensor input, Scalar other, Scalar bar)
#     foo(Tensor input, Tensor other, double bar)
#
#   If they are taking different number of parameters then they are not considered
#   ambiguous anymore, even if the difference is only on optional kwargs.
#
#     foo(Scalar other, Scalar alpha=1)
#     foo(Tensor other, *, Scalar alpha=1, Scalar beta=1)
#

def sort_overloads(
    grouped_overloads: Sequence[PythonSignatureGroup]
) -> Sequence[PythonSignatureGroup]:

    def is_arg_smaller(t1: Type, t2: Type) -> bool:
        return str(t1) == 'Scalar' and str(t2) == 'Tensor'

    def is_smaller(s1: PythonSignature, s2: PythonSignature) -> bool:
        """Returns True if s1 < s2 in the partial order."""
        args1, args2 = s1.arguments(skip_outputs=True), s2.arguments(skip_outputs=True)
        if len(args1) != len(args2):
            return False
        # TODO: should use some canonical form instead of 'str(arg.type)' - see comments
        # above. The old codegen used the deprecated 'dynamic_type(arg.type)', which
        # ignores the optional annotation, i.e. 'Scalar' and 'Scalar?'.
        equal = all(arg1.type == arg2.type for arg1, arg2 in zip(args1, args2))
        smaller_or_equal = all(str(arg1.type) == str(arg2.type)
                               or is_arg_smaller(arg1.type, arg2.type)
                               for arg1, arg2 in zip(args1, args2))
        return smaller_or_equal and not equal

    # First sort by signature
    grouped_overloads = sorted(grouped_overloads, key=lambda x: x.signature.signature_str())

    # Construct the relation graph
    larger_than: Dict[int, Set[int]] = defaultdict(set)
    for i1, overload1 in enumerate(grouped_overloads):
        for i2, overload2 in enumerate(grouped_overloads):
            if is_smaller(overload1.signature, overload2.signature):
                larger_than[i1].add(i2)

    if not larger_than:
        return list(grouped_overloads)

    # Use a topological sort to sort overloads according to the partial order.
    N = len(grouped_overloads)
    sorted_ids: List[int] = list(filter(lambda x: x not in larger_than, range(N)))

    for idx in range(N):
        # The size of sorted_ids will grow to N eventually.
        i = sorted_ids[idx]
        for j in sorted(larger_than.keys()):
            larger = larger_than[j]
            larger.discard(i)
            if not larger:
                del larger_than[j]
                sorted_ids.append(j)

    return list(map(lambda x: grouped_overloads[x], sorted_ids))

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

# Each decl entry has unique (python signature, native function) pair.
def decl_to_signature_function_pair(
    decl: Dict[str, Any], *, method: bool
) -> PythonSignatureNativeFunctionPair:
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
    return PythonSignatureNativeFunctionPair(
        signature=python_sig,
        function=f,
    )

def emit_single_dispatch(
    ps: PythonSignature, f: NativeFunction, namedtuple_typenames: Dict[str, str]
) -> str:
    """
    Emit dispatch code for a single native function.
    """
    @with_native_function
    def go(f: NativeFunction) -> str:
        # header comments
        deprecated = '[deprecated] ' if ps.deprecated else ''
        schema_comment = f'// {deprecated}aten::{f.func}'

        # dispatch lambda signature
        name = cpp.name(f.func)
        lambda_formals = ', '.join(map(lambda a: f"{a.type_str} {a.name}",
                                       dispatch_lambda_args(ps, f)))
        lambda_return = dispatch_lambda_return_str(f)

        # dispatch lambda body
        dispatch_callee = cpp_dispatch_target(f)
        dispatch_args = ', '.join(cpp_dispatch_exprs(f, python_signature=ps))

        # from arg parser outputs to dispatch lambda arguments
        parser_outputs = arg_parser_output_exprs(ps, f)
        lambda_arg_exprs = dispatch_lambda_exprs(ps, f)
        inits = '\n'.join(lambda_arg_exprs.inits)
        lambda_args = ', '.join(lambda_arg_exprs.exprs)

        # scatter fields
        # TODO: Checking `ps.method and ('requires_grad' in parser_outputs)` is a hacky
        #       solution for enabling the 'requires_grad' argument for tensor methods
        #       new_full, new_empty, and new_zeros. A much better but more difficult to
        #       implement solution involves refactoring according to Ed's description here:
        #       https://github.com/pytorch/pytorch/issues/36455#issuecomment-614767589
        need_set_requires_grad = ps.tensor_options_args and (not has_tensor_options(f) or (
            ps.method and ('requires_grad' in parser_outputs)))
        set_requires_grad = f'.set_requires_grad({parser_outputs["requires_grad"].expr})' \
            if need_set_requires_grad else ''

        if lambda_return == 'void':
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  pybind11::gil_scoped_release no_gil;
  {dispatch_callee}({dispatch_args});
}};
dispatch_{name}({lambda_args}){set_requires_grad};
Py_RETURN_NONE;
"""
        else:
            typename = namedtuple_typenames.get(gen_namedtuple_typename_key(f))
            namedtuple_typeref = f'&{typename}, ' if typename is not None else ''
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  pybind11::gil_scoped_release no_gil;
  return {dispatch_callee}({dispatch_args});
}};
return wrap({namedtuple_typeref}dispatch_{name}({lambda_args}){set_requires_grad});
"""

    return go(f)
