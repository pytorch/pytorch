# Generates Python bindings for ATen functions
#
# The bindings are generated as methods on python_variable or functions on the
# torch._C._nn. torch._C._fft, torch._C._linalg, torch._C._nested, torch._C._sparse
# or torch._C._special objects.
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

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from typing import Callable, Iterable, Sequence

import yaml

from torchgen.api import cpp
from torchgen.api.python import (
    arg_parser_output_exprs,
    cpp_dispatch_exprs,
    cpp_dispatch_target,
    dispatch_lambda_args,
    dispatch_lambda_exprs,
    dispatch_lambda_return_str,
    has_tensor_options,
    PythonSignature,
    PythonSignatureDeprecated,
    PythonSignatureGroup,
    PythonSignatureNativeFunctionPair,
    signature,
    signature_from_schema,
    structseq_fieldnames,
)
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.gen import cpp_string, parse_native_yaml, parse_tags_yaml
from torchgen.model import (
    Argument,
    BaseOperatorName,
    FunctionSchema,
    NativeFunction,
    SchemaKind,
    Type,
    Variant,
)
from torchgen.utils import FileManager, split_name_params
from torchgen.yaml_utils import YamlLoader

from .gen_inplace_or_view_type import is_tensor_list_type
from .gen_trace_type import should_trace


#
# declarations blocklist
# We skip codegen for these functions, for various reasons.
# Future PRs will categorize this list and eliminate or hoist
# them out of eager-only codegen.
# See https://github.com/pytorch/pytorch/issues/30788
#

# These functions require manual Python bindings or are not exposed to Python
_SKIP_PYTHON_BINDINGS = [
    "alias",
    "contiguous",
    "is_cuda",
    "is_sparse",
    "is_sparse_csr",
    "size",
    "stride",
    "sym_size",
    "sym_stride",
    "sym_storage_offset",
    "sym_numel",
    ".*_backward",
    ".*_backward_(out|input|weight|bias)",
    ".*_forward",
    ".*_forward_out",
    ".*_jvp",
    "_unsafe_view",
    "tensor",
    "_?sparse_(coo|compressed|csr|csc|bsr|bsc)_tensor.*",
    "_range.*",
    "_sparse_add_out",
    "_sparse_div.*",
    "_sparse_mul.*",
    "_sparse_sub.*",
    "_sparse_dense_add_out",
    "index",
    "index_out",
    "unique_dim_consecutive",
    "_cumsum.*",
    "_cumprod.*",
    "_sum.*",
    "_prod.*",
    "_th_.*",
    "_thnn_.*",
    "range.*",
    "_solve.*",
    "_inverse.*",
    "_cholesky.*",
    "_triangular_solve.*",
    "_qr.*",
    "_svd.*",
    "slice",
    "item",
    "_local_scalar_dense",
    "to",
    "_to_copy",
    "_to_copy_out",
    "_reshape_copy",
    "_reshape_copy_out",
    "copy_sparse_to_sparse_",
    "copy_",
    "_foreach_copy",
    "numpy_T",
    "matrix_H",
    "mT",
    "mH",  # these need to be an attributes in Python, not functions
    "nonzero(_(out|numpy))?",
    "set_data",
    ".*_overrideable",  # overrideable functions for backend extension
    "data",
    "is_leaf",
    "output_nr",
    "_version",
    "requires_grad_",
    "retains_grad",
    "set_",
    "_fw_primal",
    "fake_quantize_per_tensor_affine_cachemask",
    "fake_quantize_per_channel_affine_cachemask",
    "_new_zeros_with_same_feature_meta",
    "_has_same_storage_numel",  # used for forward AD internals
    "_reshape_alias",
    "replace_",  # only used by the functionalization pass, doesn't need to be exposed to python
    "copy",  # only used by the functionalization pass
    "fill.Tensor",  # only used by the functionalization pass
    "fill.Scalar",  # only used by the functionalization pass
    "lift.*",
    "normal_functional",  # only used by the functionalization pass
    "nbytes",
    "itemsize",
    "_batch_norm_with_update",
    "_batch_norm_with_update_out",
    "_batch_norm_no_update",
]

SKIP_PYTHON_BINDINGS = [
    re.compile(rf"^{pattern}$") for pattern in _SKIP_PYTHON_BINDINGS
]

# These function signatures are not exposed to Python. Note that this signature
# list does not support regex.
SKIP_PYTHON_BINDINGS_SIGNATURES = [
    "add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
    "sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
    "mul.Scalar(Tensor self, Scalar other) -> Tensor",
    "mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
    "div.Scalar(Tensor self, Scalar other) -> Tensor",
    "div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
]


@with_native_function
def should_generate_py_binding(f: NativeFunction) -> bool:
    # NativeFunctions that are entirely code-generated should not get python bindings
    # because these codegen implementations are often inefficient. A handful of
    # view_copy style ops were exposed accidentally when they were handwritten and now
    # that we are moving them to codegen for bc reasons we need to keep them exposed in
    # python.
    if "generated" in f.tags and "view_copy" not in f.tags:
        return False

    name = cpp.name(f.func)
    for skip_regex in SKIP_PYTHON_BINDINGS:
        if skip_regex.match(name):
            return False

    signature = str(f.func)
    for pattern in SKIP_PYTHON_BINDINGS_SIGNATURES:
        if pattern == signature:
            return False
    return True


def get_pycname(name: BaseOperatorName) -> str:
    return f"THPVariable_{name}"


def is_noarg(overloads: Sequence[PythonSignatureNativeFunctionPair]) -> bool:
    return len(overloads) == 1 and overloads[0].signature.arguments_count() == 0


def is_py_variable_method(f: NativeFunction) -> bool:
    return f.python_module is None and Variant.method in f.variants


def is_py_torch_function(f: NativeFunction) -> bool:
    return f.python_module is None and Variant.function in f.variants


def is_py_nn_function(f: NativeFunction) -> bool:
    return f.python_module == "nn"


def is_py_fft_function(f: NativeFunction) -> bool:
    return f.python_module == "fft"


def is_py_linalg_function(f: NativeFunction) -> bool:
    return f.python_module == "linalg"


def is_py_nested_function(f: NativeFunction) -> bool:
    return f.python_module == "nested"


def is_py_sparse_function(f: NativeFunction) -> bool:
    return f.python_module == "sparse"


def is_py_special_function(f: NativeFunction) -> bool:
    return f.python_module == "special"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                            Main Function
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def gen(
    out: str,
    native_yaml_path: str,
    tags_yaml_path: str,
    deprecated_yaml_path: str,
    template_path: str,
    *,
    symint: bool = True,
) -> None:
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    native_functions = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).native_functions
    native_functions = list(filter(should_generate_py_binding, native_functions))

    methods = load_signatures(native_functions, deprecated_yaml_path, method=True)
    create_python_bindings(
        fm,
        methods,
        is_py_variable_method,
        None,
        "python_variable_methods.cpp",
        method=True,
        symint=symint,
    )

    # NOTE: num_shards here must be synced with gatherTorchFunctions in
    #       torch/csrc/autograd/python_torch_functions_manual.cpp
    functions = load_signatures(native_functions, deprecated_yaml_path, method=False)
    create_python_bindings_sharded(
        fm,
        functions,
        is_py_torch_function,
        "torch",
        "python_torch_functions.cpp",
        method=False,
        num_shards=3,
        symint=symint,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_nn_function,
        "torch.nn",
        "python_nn_functions.cpp",
        method=False,
        symint=symint,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_fft_function,
        "torch.fft",
        "python_fft_functions.cpp",
        method=False,
        symint=symint,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_linalg_function,
        "torch.linalg",
        "python_linalg_functions.cpp",
        method=False,
        symint=symint,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_nested_function,
        "torch.nested",
        "python_nested_functions.cpp",
        method=False,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_sparse_function,
        "torch.sparse",
        "python_sparse_functions.cpp",
        method=False,
        symint=symint,
    )

    create_python_bindings(
        fm,
        functions,
        is_py_special_function,
        "torch.special",
        "python_special_functions.cpp",
        method=False,
        symint=symint,
    )

    # Currently, we only use `functions` to generate `return_types` bindings.
    # All methods which return structseq have function variant at this point.
    # If any method only operator with structseq is added in the future,
    # we will have to address that.
    create_python_return_type_bindings(
        fm, functions, lambda fn: True, "python_return_types.cpp"
    )
    create_python_return_type_bindings_header(
        fm, functions, lambda fn: True, "python_return_types.h"
    )

    valid_tags = parse_tags_yaml(tags_yaml_path)

    def gen_tags_enum() -> dict[str, str]:
        return {
            "enum_of_valid_tags": (
                "".join(
                    [f'\n.value("{tag}", at::Tag::{tag})' for tag in sorted(valid_tags)]
                )
            )
        }

    fm.write("python_enum_tag.cpp", gen_tags_enum)


def group_filter_overloads(
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
) -> dict[BaseOperatorName, list[PythonSignatureNativeFunctionPair]]:
    grouped: dict[BaseOperatorName, list[PythonSignatureNativeFunctionPair]] = (
        defaultdict(list)
    )
    for pair in pairs:
        if pred(pair.function):
            grouped[pair.function.func.name.name].append(pair)
    return grouped


def create_python_bindings(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    module: str | None,
    filename: str,
    *,
    method: bool,
    symint: bool = True,
) -> None:
    """Generates Python bindings to ATen functions"""
    py_methods: list[str] = []
    ops_headers: list[str] = []
    py_method_defs: list[str] = []
    py_forwards: list[str] = []

    grouped = group_filter_overloads(pairs, pred)

    for name in sorted(grouped.keys(), key=str):
        overloads = grouped[name]
        py_methods.append(
            method_impl(name, module, overloads, method=method, symint=symint)
        )
        py_method_defs.append(method_def(name, module, overloads, method=method))
        py_forwards.extend(forward_decls(name, overloads, method=method))
        ops_headers.append(f"#include <ATen/ops/{name.base}.h>")

    fm.write_with_template(
        filename,
        filename,
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
            "ops_headers": ops_headers,
            "py_forwards": py_forwards,
            "py_methods": py_methods,
            "py_method_defs": py_method_defs,
        },
    )


def create_python_return_type_bindings(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    filename: str,
) -> None:
    """
    Generate function to initialize and return named tuple for native functions
    which returns named tuple and registration invocations in `python_return_types.cpp`.
    """
    py_return_types_definition: list[str] = []
    py_return_types_registrations: list[str] = []

    grouped = group_filter_overloads(pairs, pred)

    for name in sorted(grouped.keys(), key=str):
        overloads = grouped[name]
        definitions, registrations = generate_return_type_definition_and_registrations(
            overloads
        )
        py_return_types_definition.append(
            "" if not definitions else "\n".join(definitions)
        )
        py_return_types_registrations.append(
            "" if not registrations else "\n".join(registrations)
        )

    fm.write_with_template(
        filename,
        filename,
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
            "py_return_types": py_return_types_definition,
            "py_return_types_registrations": py_return_types_registrations,
        },
    )


def create_python_return_type_bindings_header(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    filename: str,
) -> None:
    """
    Generate function to initialize and return named tuple for native functions
    which returns named tuple and relevant entry for the map in `python_return_types.cpp`.
    """
    py_return_types_declarations: list[str] = []

    grouped = group_filter_overloads(pairs, pred)

    for name in sorted(grouped.keys(), key=str):
        overloads = grouped[name]
        declarations = generate_return_type_declarations(overloads)
        py_return_types_declarations.append(
            "" if not declarations else "\n".join(declarations)
        )

    fm.write_with_template(
        filename,
        filename,
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
            "py_return_types_declarations": py_return_types_declarations,
        },
    )


def create_python_bindings_sharded(
    fm: FileManager,
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    pred: Callable[[NativeFunction], bool],
    module: str | None,
    filename: str,
    *,
    method: bool,
    num_shards: int,
    symint: bool = True,
) -> None:
    """Generates Python bindings to ATen functions"""
    grouped = group_filter_overloads(pairs, pred)

    def key_func(
        kv: tuple[BaseOperatorName, list[PythonSignatureNativeFunctionPair]],
    ) -> str:
        return kv[0].base

    def env_func(
        kv: tuple[BaseOperatorName, list[PythonSignatureNativeFunctionPair]],
    ) -> dict[str, list[str]]:
        name, fn_pairs = kv
        return {
            "ops_headers": [f"#include <ATen/ops/{name.base}.h>"],
            "py_forwards": list(forward_decls(name, fn_pairs, method=method)),
            "py_methods": [
                method_impl(name, module, fn_pairs, method=method, symint=symint)
            ],
            "py_method_defs": [method_def(name, module, fn_pairs, method=method)],
        }

    fm.write_sharded(
        filename,
        grouped.items(),
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/{filename}",
        },
        key_fn=key_func,
        env_callable=env_func,
        num_shards=num_shards,
        sharded_keys={"ops_headers", "py_forwards", "py_methods", "py_method_defs"},
    )


def load_signatures(
    native_functions: list[NativeFunction],
    deprecated_yaml_path: str,
    *,
    method: bool,
    skip_deprecated: bool = False,
    pyi: bool = False,
) -> Sequence[PythonSignatureNativeFunctionPair]:
    @with_native_function
    def gen_signature_pairs(f: NativeFunction) -> PythonSignatureNativeFunctionPair:
        return PythonSignatureNativeFunctionPair(
            signature=signature(f, method=method, pyi=pyi),
            function=f,
        )

    pairs = list(map(gen_signature_pairs, native_functions))
    deprecated = load_deprecated_signatures(
        pairs, deprecated_yaml_path, method=method, pyi=pyi
    )
    return pairs if skip_deprecated else pairs + deprecated


def load_deprecated_signatures(
    pairs: Sequence[PythonSignatureNativeFunctionPair],
    deprecated_yaml_path: str,
    *,
    method: bool,
    pyi: bool,
) -> list[PythonSignatureNativeFunctionPair]:
    # The deprecated.yaml doesn't have complete type information, we need
    # find and leverage the original ATen signature (to which it delegates
    # the call) to generate the full python signature.
    # We join the deprecated and the original signatures using type-only form.

    # group the original ATen signatures by name
    grouped: dict[str, list[PythonSignatureNativeFunctionPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.signature.name].append(pair)

    # find matching original signatures for each deprecated signature
    results: list[PythonSignatureNativeFunctionPair] = []

    with open(deprecated_yaml_path) as f:
        deprecated_defs = yaml.load(f, Loader=YamlLoader)

    for deprecated in deprecated_defs:
        schema = FunctionSchema.parse(deprecated["name"])
        aten_name, call_args = split_name_params(deprecated["aten"])
        is_out = aten_name.endswith("_out")
        if is_out:
            aten_name = aten_name.replace("_out", "")

        # HACK: these are fixed constants used to pass the aten function.
        # The type must be known ahead of time
        known_constants = {
            "1": Type.parse("Scalar"),
        }
        schema_args_by_name = {a.name: a for a in schema.arguments.flat_all}
        for name in call_args:
            assert (
                name in schema_args_by_name or name in known_constants
            ), f"deprecation definiton: Unrecognized value {name}"

        # Map deprecated signature arguments to their aten signature and test
        # if the types and alias annotation match.
        def is_schema_compatible(
            aten_schema: FunctionSchema,
        ) -> bool:
            arguments: Iterable[Argument]
            if is_out:
                arguments = itertools.chain(
                    aten_schema.arguments.out, aten_schema.arguments.flat_non_out
                )
            else:
                arguments = aten_schema.arguments.flat_all

            for i, arg in enumerate(arguments):
                if i < len(call_args):
                    arg_name = call_args[i]
                    if arg_name in known_constants:
                        schema_type = known_constants[arg_name]
                        schema_annotation = None
                    else:
                        schema_arg = schema_args_by_name[arg_name]
                        schema_type = schema_arg.type
                        schema_annotation = schema_arg.annotation

                    if schema_type != arg.type or schema_annotation != arg.annotation:
                        return False
                else:
                    if arg.default is None:
                        return False

            return len(schema.returns) == len(aten_schema.returns) and all(
                a == b for a, b in zip(schema.returns, aten_schema.returns)
            )

        any_schema_found = False
        for pair in grouped[aten_name]:
            if not is_schema_compatible(pair.function.func):
                continue
            any_schema_found = True

            python_sig = signature_from_schema(
                schema,
                category_override=pair.function.category_override,
                method=method,
                pyi=pyi,
            )

            results.append(
                PythonSignatureNativeFunctionPair(
                    signature=PythonSignatureDeprecated(
                        name=python_sig.name,
                        input_args=python_sig.input_args,
                        input_kwargs=python_sig.input_kwargs,
                        output_args=python_sig.output_args,
                        tensor_options_args=python_sig.tensor_options_args,
                        method=python_sig.method,
                        deprecated_schema=schema,
                        deprecated_args_exprs=tuple(call_args),
                        returns=python_sig.returns,
                    ),
                    function=pair.function,
                )
            )
        assert any_schema_found, f"No native function with name {aten_name} matched signature:\n  {str(schema)}"

    return results


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         Named Tuple Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@with_native_function
def gen_structseq_typename_key(f: NativeFunction) -> str:
    name = cpp.name(f.func)
    fieldnames = structseq_fieldnames(f.func.returns)
    return "_".join([name] + fieldnames)


def emit_structseq_call(
    overloads: Sequence[PythonSignatureNativeFunctionPair],
) -> tuple[list[str], dict[str, str]]:
    """
    Generate block of named tuple type def inits, and add typeref snippets
    to declarations that use them
    """
    typenames: dict[
        str, str
    ] = {}  # map from unique name + field name lists to typedef name
    typedefs: list[str] = []  # typedef declarations and init code

    for overload in overloads:
        fieldnames = structseq_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue

        name = cpp.name(overload.function.func)  # use @with_native_function?
        tn_key = gen_structseq_typename_key(overload.function)
        typename = typenames.get(tn_key)
        if typename is None:
            typename = f'NamedTuple{"" if not typedefs else len(typedefs)}'
            typenames[tn_key] = typename
            typedefs.append(
                f"""\
static PyTypeObject* {typename} = generated::get_{name}_structseq();"""
            )

    return typedefs, typenames


def generate_return_type_definition_and_registrations(
    overloads: Sequence[PythonSignatureNativeFunctionPair],
) -> tuple[list[str], list[str]]:
    """
    Generate block of function in `python_return_types.cpp` to initialize
    and return named tuple for a native function which returns named tuple
    and registration invocations in same file.
    """
    typenames: dict[
        str, str
    ] = {}  # map from unique name + field name lists to typedef name
    definitions: list[str] = []  # function definition to register the typedef
    registrations: list[str] = []  # register call for the typedef

    for overload in overloads:
        fieldnames = structseq_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue

        fields = ", ".join(f'{{"{fn}", ""}}' for fn in fieldnames)

        name = cpp.name(overload.function.func)  # use @with_native_function?
        tn_key = gen_structseq_typename_key(overload.function)
        typename = typenames.get(tn_key)

        if typename is None:
            typename = f'{name}NamedTuple{"" if not definitions else len(definitions)}'
            typenames[tn_key] = typename
            definitions.append(
                f"""\
PyTypeObject* get_{name}_structseq() {{
    static PyStructSequence_Field NamedTuple_fields[] = {{ {fields},  {{nullptr}} }};
    static PyTypeObject {typename};
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = {{ "torch.return_types.{name}", nullptr, NamedTuple_fields, {len(fieldnames)} }};
    if (!is_initialized) {{
        PyStructSequence_InitType(&{typename}, &desc);
        {typename}.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }}
    return &{typename};
}}
"""
            )
            registrations.append(
                f'addReturnType(return_types_module, "{name}", generated::get_{name}_structseq());'
            )

    return definitions, registrations


def generate_return_type_declarations(
    overloads: Sequence[PythonSignatureNativeFunctionPair],
) -> list[str]:
    """
    Generate block of function declarations in `python_return_types.h` to initialize
    and return named tuple for a native function.
    """
    typenames: dict[
        str, str
    ] = {}  # map from unique name + field name lists to typedef name
    declarations: list[str] = []  # function declaration to register the typedef

    for overload in overloads:
        fieldnames = structseq_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue

        name = cpp.name(overload.function.func)  # use @with_native_function?
        tn_key = gen_structseq_typename_key(overload.function)
        typename = typenames.get(tn_key)

        if typename is None:
            typename = (
                f'{name}NamedTuple{"" if not declarations else len(declarations)}'
            )
            typenames[tn_key] = typename
            declarations.append(f"PyTypeObject* get_{name}_structseq();")

    return declarations


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         Method Impl Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# python binding for all overloads of a particular function/method
PY_VARIABLE_METHOD_VARARGS = CodeTemplate(
    r"""\
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

"""
)

# handler for a single parsed signature - may be a single overload or
# a pair of overloads that whose signatures only differ in output params
# (plugged into PY_VARIABLE_METHOD_VARARGS as an item in ${dispatch})
PY_VARIABLE_CASE = CodeTemplate(
    """\
case ${overload_index}: {
  ${body}
}
"""
)

# python binding for single-overload function/method
PY_VARIABLE_METHOD_VARARGS_SINGLETON = CodeTemplate(
    """\
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

"""
)

# python binding for a method with no args, shortcuts parsing
PY_VARIABLE_METHOD_NOARGS = CodeTemplate(
    """\
// ${name}
static PyObject * ${pycname}(PyObject* self_, PyObject* args)
{
  ${method_header}
  ${check_has_torch_function}
  ${dispatch}
  ${method_footer}
}

"""
)


def method_impl(
    name: BaseOperatorName,
    module: str | None,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool,
    symint: bool = True,
) -> str:
    """
    Generate a python binding for all overloads of an op.
    """
    pycname = get_pycname(name)
    noarg = is_noarg(overloads)
    structseq_inits, structseq_typenames = emit_structseq_call(overloads)

    method_header = ["HANDLE_TH_ERRORS"]
    method_header += structseq_inits
    method_header += (
        ["const Tensor& self = THPVariable_Unpack(self_);"] if method else []
    )

    method_footer = ([] if noarg else ["Py_RETURN_NONE;"]) + ["END_HANDLE_TH_ERRORS"]

    traceable = "true" if all(should_trace(o.function) for o in overloads) else "false"

    grouped_overloads: Sequence[PythonSignatureGroup] = group_overloads(
        overloads, symint=symint
    )
    is_singleton = len(grouped_overloads) == 1
    signatures: list[str] = []
    dispatch: list[str] = []
    for overload_index, overload in enumerate(grouped_overloads):
        signature = overload.signature.signature_str(symint=symint)
        signatures.append(f"{cpp_string(str(signature))},")
        dispatch_body = emit_dispatch_case(overload, structseq_typenames, symint=symint)
        dispatch.append(
            PY_VARIABLE_CASE.substitute(
                overload_index=overload_index, body=dispatch_body
            )
            if not is_singleton
            else dispatch_body
        )

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
        max_args=max(o.signature.arguments_count() for o in overloads),
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
    name: BaseOperatorName, module: str | None, *, noarg: bool, method: bool
) -> str:
    if noarg:
        if method:
            return f"""\
if(check_has_torch_function(self_)) {{
  return handle_torch_function(self_, "{name}");
}}
"""
        else:
            return ""

    self_ = "self_" if method else "nullptr"
    namespace = (
        {
            "torch": "THPVariableFunctionsModule",
            "torch.nn": "THPNNVariableFunctionsModule",
            "torch.fft": "THPFFTVariableFunctionsModule",
            "torch.linalg": "THPLinalgVariableFunctionsModule",
            "torch.nested": "THPNestedVariableFunctionsModule",
            "torch.sparse": "THPSparseVariableFunctionsModule",
            "torch.special": "THPSpecialVariableFunctionsModule",
        }[module]
        if module
        else "THPVariableClass"
    )

    return f"""\
if(_r.has_torch_function()) {{
  return handle_torch_function(_r, {self_}, args, kwargs, {namespace}, "{module or "torch.Tensor"}");
}}
"""


# handler for output/no-output overload pair
PY_VARIABLE_OUT = CodeTemplate(
    """\
if (_r.isNone(${out_idx})) {
  ${call_dispatch}
} else {
  ${call_dispatch_out}
}
"""
)


def emit_dispatch_case(
    overload: PythonSignatureGroup,
    structseq_typenames: dict[str, str],
    *,
    symint: bool = True,
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
                overload.signature, overload.base, structseq_typenames, symint=symint
            ),
            call_dispatch_out=emit_single_dispatch(
                overload.signature,
                overload.outplace,
                structseq_typenames,
                symint=symint,
            ),
        )
    else:
        # no-output version only
        return emit_single_dispatch(
            overload.signature, overload.base, structseq_typenames, symint=symint
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                    Forward Declarations Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def forward_decls(
    name: BaseOperatorName,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool,
) -> tuple[str, ...]:
    if method:
        return ()

    pycname = get_pycname(name)
    if is_noarg(overloads):
        return (
            f"""\
static PyObject * {pycname}(PyObject* self_, PyObject* args);
""",
        )
    else:
        return (
            f"""\
static PyObject * {pycname}(PyObject* self_, PyObject* args, PyObject* kwargs);
""",
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#              Method Def (Binding Table Entry) Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def method_def(
    name: BaseOperatorName,
    module: str | None,
    overloads: Sequence[PythonSignatureNativeFunctionPair],
    *,
    method: bool,
) -> str:
    """
    Generate method def entry.
    """
    pycname = get_pycname(name)

    if name.dunder_method:
        # PyMethodDef entry for binary op, throws not implemented error
        pycname = f"TypeError_to_NotImplemented_<{pycname}>"

    if is_noarg(overloads):
        flags = "METH_NOARGS" if method else "METH_VARARGS | METH_KEYWORDS"
    else:
        pycname = f"castPyCFunctionWithKeywords({pycname})"
        flags = "METH_VARARGS | METH_KEYWORDS"

    if module == "torch":
        flags += " | METH_STATIC"

    return f'{{"{name}", {pycname}, {flags}, nullptr}},'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                   Overload Sorting and Grouping
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def group_overloads(
    overloads: Sequence[PythonSignatureNativeFunctionPair], *, symint: bool = True
) -> Sequence[PythonSignatureGroup]:
    bases: dict[str, PythonSignatureNativeFunctionPair] = {}
    outplaces: dict[str, PythonSignatureNativeFunctionPair] = {}

    # first group by signature ignoring out arguments
    for overload in overloads:
        sig = overload.signature.signature_str(skip_outputs=True, symint=symint)
        if overload.function.func.is_out_fn():
            if sig in outplaces:
                raise RuntimeError(
                    f"Found duplicated function definition:\n- {overload.function.func}.\n"
                    f"Existing definition:\n- {outplaces[sig].function.func}."
                )
            outplaces[sig] = overload
        else:
            if sig in bases:
                raise RuntimeError(
                    f"Found duplicated function definition:\n- {overload.function.func}.\n"
                    f"Existing definition:\n- {bases[sig].function.func}."
                )
            bases[sig] = overload

    for sig, out in outplaces.items():
        if sig not in bases:
            candidates: list[str] = []
            for overload in overloads:
                if (
                    str(overload.function.func.name.name)
                    == str(out.function.func.name.name)
                    and not overload.function.func.is_out_fn()
                    and not overload.signature.deprecated
                ):
                    candidates.append(
                        overload.signature.signature_str(
                            skip_outputs=True, symint=symint
                        )
                    )
            out_sig = out.signature.signature_str(symint=symint)
            raise RuntimeError(
                f"While identifying overloads, we found an out schema {out_sig} without a corresponding non-out variant. "
                f"We expected the non-out variant to have schema: \n- {sig}\nPlease check that you spelled the schema "
                "correctly in native_functions.yaml. We discovered the following candidate(s): \n"
                + "\n".join(f"- {candidate}" for candidate in candidates)
            )

    grouped = [
        PythonSignatureGroup.from_pairs(
            functional=base,
            out=outplaces.get(sig),
        )
        for sig, base in bases.items()
    ]
    return sort_overloads(grouped, symint=symint)


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
    grouped_overloads: Sequence[PythonSignatureGroup], *, symint: bool = True
) -> Sequence[PythonSignatureGroup]:
    # NB: Smaller here means lower priority

    def is_arg_smaller(t1: Type, t2: Type) -> bool:
        return (
            str(t1) == "Scalar"
            and str(t2) == "Tensor"
            or str(t1) == "Scalar?"
            and str(t2) == "Tensor?"
            or "Dimname" in str(t1)
            and "Dimname" not in str(t2)
            or
            # In the discussion https://github.com/pytorch/pytorch/issues/54555 it has been
            # discussed why it is important to prioritize int/int? over int[]
            str(t1) == "int[]"
            and (str(t2) == "int" or str(t2) == "int?")
            or
            # TensorList currently throws an error during argument parsing, that's why it needs to be
            # last in signature ordering. See discussion: https://github.com/pytorch/pytorch/issues/58087
            str(t1) == "Tensor[]"
            and str(t2).find("[]") != -1
            or
            # Prioritize IntArrayRef overload over SymIntArrayRef
            str(t1) == "SymInt[]"
            and str(t2) == "int[]"
            or
            # Make sure both in, SymInt are sorted consistently w.r.t. Tensor since Tensor can be implicitly
            # converted to either int or SymInt.  Prioritize the Tensor overload since it otherwise gets shadowed.
            (str(t1) == "SymInt" or str(t1) == "int")
            and str(t2) == "Tensor"
        )

    def is_smaller(s1: PythonSignature, s2: PythonSignature) -> bool:
        """Returns True if s1 < s2 in the partial order."""
        args1, args2 = s1.arguments(skip_outputs=True), s2.arguments(skip_outputs=True)
        if len(args1) != len(args2):
            return False
        # TODO: should use some canonical form instead of 'str(arg.type)' - see comments
        # above. The old codegen used the deprecated 'dynamic_type(arg.type)', which
        # ignores the optional annotation, i.e. 'Scalar' and 'Scalar?'.
        equal = all(arg1.type == arg2.type for arg1, arg2 in zip(args1, args2))
        smaller_or_equal = all(
            str(arg1.type) == str(arg2.type) or is_arg_smaller(arg1.type, arg2.type)
            for arg1, arg2 in zip(args1, args2)
        )
        return smaller_or_equal and not equal

    # First sort by signature
    grouped_overloads = sorted(
        grouped_overloads, key=lambda x: x.signature.signature_str(symint=symint)
    )

    # Construct the relation graph
    larger_than: dict[int, set[int]] = defaultdict(set)
    for i1, overload1 in enumerate(grouped_overloads):
        for i2, overload2 in enumerate(grouped_overloads):
            if is_smaller(overload1.signature, overload2.signature):
                larger_than[i1].add(i2)

    if not larger_than:
        return list(grouped_overloads)

    # Use a topological sort to sort overloads according to the partial order.
    N = len(grouped_overloads)
    sorted_ids: list[int] = list(filter(lambda x: x not in larger_than, range(N)))

    for idx in range(N):
        # The size of sorted_ids will grow to N eventually.
        i = sorted_ids[idx]
        for j in sorted(larger_than.keys()):
            larger = larger_than[j]
            larger.discard(i)
            if not larger:
                del larger_than[j]
                sorted_ids.append(j)

    return [grouped_overloads[x] for x in sorted_ids]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                       Codegen API Integration
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def emit_single_dispatch(
    ps: PythonSignature,
    f: NativeFunction,
    structseq_typenames: dict[str, str],
    *,
    symint: bool = True,
) -> str:
    """
    Emit dispatch code for a single native function.
    """

    @with_native_function
    def go(f: NativeFunction) -> str:
        # header comments
        if isinstance(ps, PythonSignatureDeprecated):
            schema_comment = f"// [deprecated] aten::{ps.deprecated_schema}"
        else:
            schema_comment = f"// aten::{f.func}"

        # dispatch lambda signature
        name = cpp.name(f.func)
        lambda_formals = ", ".join(
            f"{a.type_str} {a.name}" for a in dispatch_lambda_args(ps, f, symint=symint)
        )
        lambda_return = dispatch_lambda_return_str(f)

        # dispatch lambda body
        dispatch_callee = cpp_dispatch_target(f)
        dispatch_args = ", ".join(cpp_dispatch_exprs(f, python_signature=ps))

        # from arg parser outputs to dispatch lambda arguments
        parser_outputs = arg_parser_output_exprs(ps, f, symint=symint)
        lambda_arg_exprs = dispatch_lambda_exprs(ps, f, symint=symint)
        inits = "\n".join(lambda_arg_exprs.inits)
        lambda_args = ", ".join(lambda_arg_exprs.exprs)

        # scatter fields
        # TODO: Checking `ps.method and ('requires_grad' in parser_outputs)` is a hacky
        #       solution for enabling the 'requires_grad' argument for tensor methods
        #       new_full, new_empty, and new_zeros. A much better but more difficult to
        #       implement solution involves refactoring according to Ed's description here:
        #       https://github.com/pytorch/pytorch/issues/36455#issuecomment-614767589
        need_set_requires_grad = ps.tensor_options_args and (
            not has_tensor_options(f)
            or (ps.method and ("requires_grad" in parser_outputs))
        )
        set_requires_grad = (
            f'.set_requires_grad({parser_outputs["requires_grad"].expr})'
            if need_set_requires_grad
            else ""
        )

        if lambda_return == "void":
            # Make in-place foreach return `self` at python-binding level.
            # ref: https://github.com/pytorch/pytorch/pull/118622#pullrequestreview-1904804954
            self_arg = f.func.arguments.self_arg
            return_stmt: str
            if (
                str(f.func.name).startswith("_foreach_")
                and f.func.kind() == SchemaKind.inplace
            ):
                # note(crcrpar): `_foreach_pow.ScalarAndTensor` does NOT have its in-place
                # variant and it unlikely to have it in the future. Thus it's safe to have the following assert.
                assert self_arg is not None and is_tensor_list_type(
                    self_arg.argument.type
                )
                return_stmt = """PyObject* self_tensorlist = _r.args[0];
Py_INCREF(self_tensorlist);
return self_tensorlist;
"""
            else:
                return_stmt = "Py_RETURN_NONE;"
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  pybind11::gil_scoped_release no_gil;
  {dispatch_callee}({dispatch_args});
}};
dispatch_{name}({lambda_args}){set_requires_grad};
{return_stmt}
"""
        else:
            typename = structseq_typenames.get(gen_structseq_typename_key(f))
            structseq_typeref = f"{typename}, " if typename is not None else ""
            return f"""\
{schema_comment}
{inits}
auto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{
  pybind11::gil_scoped_release no_gil;
  return {dispatch_callee}({dispatch_args});
}};
return wrap({structseq_typeref}dispatch_{name}({lambda_args}){set_requires_grad});
"""

    return go(f)
