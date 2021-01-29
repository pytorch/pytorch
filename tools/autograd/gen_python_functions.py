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
import yaml

from .gen_trace_type import should_trace

from tools.codegen.code_template import CodeTemplate
from tools.codegen.api.types import *
from tools.codegen.api.python import *
from tools.codegen.gen import cpp_string, parse_native_yaml, with_native_function, FileManager
from tools.codegen.model import *
from tools.codegen.utils import *

from typing import Dict, Optional, List, Tuple, Set, Sequence, Callable

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore

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
    'set_data',
    '.*_overrideable',  # overrideable functions for backend extension
    'data', 'is_leaf', 'output_nr', '_version', 'requires_grad_', 'retain_grad', 'set_',
    '_fw_primal'
]

# These function signatures are not exposed to Python. Note that this signature
# list does not support regex.
SKIP_PYTHON_BINDINGS_SIGNATURES = [
    'add(Tensor, Scalar, Scalar)', 'add_(Tensor, Scalar, Scalar)',
    'sub(Tensor, Scalar, Scalar)', 'sub_(Tensor, Scalar, Scalar)',
    'mul(Tensor, Scalar)', 'mul_(Tensor, Scalar)',
    'div(Tensor, Scalar)', 'div_(Tensor, Scalar)',
]

@with_native_function
def should_generate_py_binding(f: NativeFunction) -> bool:
    name = cpp.name(f.func)
    for pattern in SKIP_PYTHON_BINDINGS:
        if re.match('^' + pattern + '$', name):
            return False

    args = ', '.join(argument_type_str(arg.type)
                     for arg in signature(f).arguments())
    sig = f'{name}({args})'
    for pattern in SKIP_PYTHON_BINDINGS_SIGNATURES:
        if pattern == sig:
            return False

    return True

def get_pycname(name: BaseOperatorName) -> str:
    return f'THPVariable_{name}'

def is_noarg(overloads: Sequence[PythonSignatureNativeFunctionPair]) -> bool:
    return len(overloads) == 1 and overloads[0].signature.arguments_count() == 0

def is_py_variable_method(f: NativeFunction) -> bool:
    return f.python_module is None and Variant.method in f.variants

def is_py_torch_function(f: NativeFunction) -> bool:
    return f.python_module is None and Variant.function in f.variants

def is_py_nn_function(f: NativeFunction) -> bool:
    return f.python_module == 'nn'

def is_py_fft_function(f: NativeFunction) -> bool:
    return f.python_module == 'fft'

def is_py_linalg_function(f: NativeFunction) -> bool:
    return f.python_module == 'linalg'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                            Main Function
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def gen(out: str, native_yaml_path: str, deprecated_yaml_path: str, template_path: str) -> None:
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)

    methods = load_signatures(native_yaml_path, deprecated_yaml_path, method=True)
    create_python_bindings(
        fm, methods, is_py_variable_method, None, 'python_variable_methods.cpp', method=True)

    functions = load_signatures(native_yaml_path, deprecated_yaml_path, method=False)
    create_python_bindings(
        fm, functions, is_py_torch_function, 'torch', 'python_torch_functions.cpp', method=False)

    create_python_bindings(
        fm, functions, is_py_nn_function, 'torch.nn', 'python_nn_functions.cpp', method=False)

    create_python_bindings(
        fm, functions, is_py_fft_function, 'torch.fft', 'python_fft_functions.cpp', method=False)

    create_python_bindings(
        fm, functions, is_py_linalg_function, 'torch.linalg', 'python_linalg_functions.cpp', method=False)

def create_python_bindings(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    module: Optional[str],
    filename: str,
    *,
    method: bool,
) -> None:
    """Generates Python bindings to ATen functions"""
    py_methods: List[str] = []
    py_method_defs: List[str] = []
    py_forwards: List[str] = []

    grouped: Dict[BaseOperatorName, List[PythonSignatureNativeFunctionPair]] = defaultdict(list)
    for pair in pairs:
        if pred(pair.function):
            grouped[pair.function.func.name.name].append(pair)

    for name in sorted(grouped.keys(), key=lambda x: str(x)):
        overloads = grouped[name]
        py_methods.append(method_impl(name, module, overloads, method=method))
        py_method_defs.append(method_def(name, module, overloads, method=method))
        py_forwards.extend(forward_decls(name, overloads, method=method))

    fm.write_with_template(filename, filename, lambda: {
        'generated_comment': '@' + f'generated from {fm.template_dir}/{filename}',
        'py_forwards': py_forwards,
        'py_methods': py_methods,
        'py_method_defs': py_method_defs,
    })

def load_signatures(
    native_yaml_path: str,
    deprecated_yaml_path: str,
    *,
    method: bool,
    skip_deprecated: bool = False,
    pyi: bool = False,
) -> Sequence[PythonSignatureNativeFunctionPair]:
    native_functions = list(filter(should_generate_py_binding, parse_native_yaml(native_yaml_path)))

    @with_native_function
    def gen_signature_pairs(f: NativeFunction) -> PythonSignatureNativeFunctionPair:
        return PythonSignatureNativeFunctionPair(
            signature=signature(f, method=method, pyi=pyi),
            function=f,
        )

    pairs = list(map(gen_signature_pairs, native_functions))
    deprecated = load_deprecated_signatures(pairs, deprecated_yaml_path, method=method, pyi=pyi)
    return pairs if skip_deprecated else pairs + deprecated

def load_deprecated_signatures(
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    deprecated_yaml_path: str,
    *,
    method: bool,
    pyi: bool,
) -> List[PythonSignatureNativeFunctionPair]:
    # The deprecated.yaml doesn't have complete type information, we need
    # find and leverage the original ATen signature (to which it delegates
    # the call) to generate the full python signature.
    # We join the deprecated and the original signatures using type-only form.

    # native function -> type-only signature
    @with_native_function
    def signature_original(f: NativeFunction) -> str:
        # remove inplace suffix but keep outplace suffix
        opname = str(f.func.name.name.base)
        if f.func.is_out_fn():
            opname += '_out'
        if f.func.name.name.inplace and pyi:
            opname += '_'
        args = CppSignatureGroup.from_native_function(f, method=False).signature.arguments()
        # Simply ignore TensorOptionsArguments as it does not exist in deprecated.yaml.
        types = ', '.join(argument_type_str(a.argument.type)
                          for a in args if isinstance(a.argument, Argument))
        return f'{opname}({types})'

    # deprecated -> type-only native signature (according to the call order)
    def signature_deprecated(opname: str, params: List[str], call_args: List[str]) -> str:
        # create a mapping of parameter name to parameter type
        types: Dict[str, str] = {}
        for param in params:
            if param == '*':
                continue
            type, name = param.split(' ')
            types[name] = type
        # if the name in the call is not in the parameter list, assume it's
        # a literal Scalar
        rearranged_types = ', '.join(types.get(arg, 'Scalar') for arg in call_args)
        return f'{opname}({rearranged_types})'

    # group the original ATen signatures by type-only signature
    grouped: Dict[str, List[PythonSignatureNativeFunctionPair]] = defaultdict(list)
    for pair in pairs:
        grouped[signature_original(pair.function)].append(pair)

    # find matching original signatures for each deprecated signature
    results: List[PythonSignatureNativeFunctionPair] = []

    with open(deprecated_yaml_path, 'r') as f:
        deprecated_defs = yaml.load(f, Loader=Loader)

    for deprecated in deprecated_defs:
        _, params = split_name_params(deprecated['name'])
        aten_name, call_args = split_name_params(deprecated['aten'])

        for pair in grouped[signature_deprecated(aten_name, params, call_args)]:
            # It uses the types from the original ATen declaration, but the
            # ordering and parameter names from the deprecated overload. Any
            # default parameter values from the original ATen declaration are
            # ignored.
            # Deprecated signature might reorder input_args and input_kwargs,
            # but never changes output_args nor TensorOptions (if any?),
            # so here we only look into these two types of args.
            python_sig = pair.signature
            src_args: Dict[str, PythonArgument] = {a.name: PythonArgument(
                name=a.name,
                type=a.type,
                default=None,
                default_init=None,
            ) for a in itertools.chain(python_sig.input_args, python_sig.input_kwargs)}

            args: List[str] = []
            input_args: List[PythonArgument] = []
            input_kwargs: List[PythonArgument] = []

            kwarg_only = False
            for param in params:
                if param == '*':
                    kwarg_only = True
                    continue
                _, param_name = param.split(' ')
                args.append(param_name)

                if param_name not in src_args:
                    # output argument
                    continue

                if not kwarg_only:
                    if not method or param_name != 'self':
                        input_args.append(src_args[param_name])
                else:
                    input_kwargs.append(src_args[param_name])

            results.append(PythonSignatureNativeFunctionPair(
                signature=PythonSignatureDeprecated(
                    name=python_sig.name,
                    input_args=tuple(input_args),
                    input_kwargs=tuple(input_kwargs),
                    output_args=python_sig.output_args,
                    tensor_options_args=python_sig.tensor_options_args,
                    method=python_sig.method,
                    deprecated_args_names=tuple(args),
                    deprecated_args_exprs=tuple(call_args),
                    returns=python_sig.returns,
                ),
                function=pair.function,
            ))

    return results

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         Named Tuple Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@with_native_function
def gen_namedtuple_typename_key(f: NativeFunction) -> str:
    name = cpp.name(f.func)
    fieldnames = namedtuple_fieldnames(f.func.returns)
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
        fieldnames = namedtuple_fieldnames(overload.function.func.returns)
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
    name: BaseOperatorName,
    module: Optional[str],
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

def gen_has_torch_function_check(
    name: BaseOperatorName, module: Optional[str], *, noarg: bool, method: bool
) -> str:
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
    name: BaseOperatorName,
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

def method_def(
    name: BaseOperatorName,
    module: Optional[str],
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

    if name.dunder_method:
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
    overloads: Sequence[PythonSignatureNativeFunctionPair],
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
        return (str(t1) == 'Scalar' and str(t2) == 'Tensor' or
                'Dimname' in str(t1) and 'Dimname' not in str(t2))

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
