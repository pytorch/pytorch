import argparse
import functools
import json
import os
import pathlib
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import yaml

import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest

from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
    Binding,
    CppSignature,
    CppSignatureGroup,
    DispatcherSignature,
    NamedCType,
    NativeSignature,
    SpecialArgName,
)
from torchgen.context import (
    method_with_native_function,
    native_function_manager,
    with_native_function,
    with_native_function_and_indices,
)
from torchgen.gen_functionalization_type import (
    gen_functionalization_definition,
    gen_functionalization_registration,
    gen_functionalization_view_inverse_declaration,
    GenCompositeViewCopyKernel,
)
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing

from torchgen.model import (
    Argument,
    BackendIndex,
    BackendMetadata,
    BaseOperatorName,
    DEFAULT_KERNEL_NAMESPACE,
    DispatchKey,
    FRAGMENT_NAMESPACES,
    FunctionSchema,
    is_cuda_dispatch_key,
    is_generic_dispatch_key,
    is_ufunc_dispatch_key,
    Location,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    OperatorName,
    OptionalType,
    SchemaKind,
    SelfArgument,
    STRUCTURED_DISPATCH_KEYS,
    TensorOptionsArguments,
    Type,
    Variant,
    ViewSchemaKind,
)
from torchgen.native_function_generation import (
    add_generated_native_functions,
    gen_composite_functional_kernel,
    gen_composite_out_kernel,
    pre_group_native_functions,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
    assert_never,
    concatMap,
    context,
    FileManager,
    make_file_manager,
    mapMaybe,
    NamespaceHelper,
    Target,
)
from torchgen.yaml_utils import YamlDumper, YamlLoader

T = TypeVar("T")

# Welcome to the ATen code generator v2!  The ATen code generator is
# responsible for parsing native_functions.yaml and then generating
# various generated files (e.g., TypeDefault.cpp) based on the operators
# defined in this file.  This means that the code generator knows how to
# parse function schema, and then translate this into various C++ types
# and boilerplate code.
#
# Some things to know about this file when you modify it:
#
# - This file has STRICT mypy typechecking.  Typecheck it with
#   `mypy --config mypy-strict.ini` in the root source directory
#
# - Most of the heavy lifting lives in external modules:
#   - 'model' has the data model for native_functions.yaml.  The classes
#     in those file represent what you see when you look at
#     a native_functions.yaml
#   - 'api' has conversions for how to translate JIT schema into
#     the various C++ APIs that the codegen interacts with.  There
#     are in fact THREE different C++ APIs: the public C++ API,
#     the dispatcher API, and the legacy dispatcher API.  See each
#     of these respective files for more information

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         HELPER FUNCTIONS
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# A custom loader for YAML to let us also keep track of line numbers
# of each entry in the YAML file
class LineLoader(YamlLoader):
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        # Add 1 so line numbering starts at 1
        mapping["__line__"] = node.start_mark.line + 1
        return mapping


_GLOBAL_PARSE_NATIVE_YAML_CACHE = {}
_GLOBAL_PARSE_TAGS_YAML_CACHE = {}

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple("ParsedYaml", ["native_functions", "backend_indices"])


def parse_native_yaml_struct(
    es: object,
    valid_tags: Set[str],
    ignore_keys: Optional[Set[DispatchKey]] = None,
    path: str = "<stdin>",
    skip_native_fns_gen: bool = False,
) -> ParsedYaml:
    assert isinstance(es, list)
    rs: List[NativeFunction] = []
    bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
    for e in es:
        assert isinstance(e.get("__line__"), int), e
        loc = Location(path, e["__line__"])
        funcs = e.get("func")
        with context(lambda: f"in {loc}:\n  {funcs}"):
            func, m = NativeFunction.from_yaml(e, loc, valid_tags, ignore_keys)
            rs.append(func)
            BackendIndex.grow_index(bs, m)
    error_check_native_functions(rs)
    # Default dict is to prevent the codegen from barfing when we have a dispatch key that has no kernels yet.
    indices: Dict[DispatchKey, BackendIndex] = defaultdict(
        lambda: BackendIndex(
            dispatch_key=DispatchKey.Undefined,
            use_out_as_primary=True,
            external=False,
            device_guard=False,
            # I'm actually not sure about this; undefined could be hit on
            # empty TensorList, hypothetically that could have sizes in it
            index={},
        )
    )
    if not skip_native_fns_gen:
        add_generated_native_functions(rs, bs)
    for k, v in bs.items():
        # All structured in-tree operators are implemented in terms of their out operator.
        indices[k] = BackendIndex(
            dispatch_key=k,
            use_out_as_primary=True,
            external=False,
            # Only cuda-like devices in tree require device guards
            device_guard=is_cuda_dispatch_key(k),
            index=v,
        )
    return ParsedYaml(rs, indices)


def parse_tags_yaml_struct(es: object, path: str = "<stdin>") -> Set[str]:
    assert isinstance(es, list)
    rs: Set[str] = set()
    for e in es:
        assert isinstance(e.get("__line__"), int), e
        loc = Location(path, e["__line__"])
        tags = e.get("tag")
        with context(lambda: f"in {loc}:\n  {tags}"):
            e_i = e.copy()
            name = e_i.pop("tag")
            desc = e_i.pop("desc", "")
            # ensure that each tag has a non-empty description
            assert desc != ""
            rs.add(name)
    return rs


@functools.lru_cache(maxsize=None)
def parse_tags_yaml(path: str) -> Set[str]:
    global _GLOBAL_PARSE_TAGS_YAML_CACHE
    if path not in _GLOBAL_PARSE_TAGS_YAML_CACHE:
        with open(path, "r") as f:
            es = yaml.load(f, Loader=LineLoader)
            _GLOBAL_PARSE_TAGS_YAML_CACHE[path] = parse_tags_yaml_struct(es, path=path)

    return _GLOBAL_PARSE_TAGS_YAML_CACHE[path]


def parse_native_yaml(
    path: str,
    tags_yaml_path: str,
    ignore_keys: Optional[Set[DispatchKey]] = None,
    *,
    skip_native_fns_gen: bool = False,
) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        valid_tags = parse_tags_yaml(tags_yaml_path)
        with open(path, "r") as f:
            es = yaml.load(f, Loader=LineLoader)
        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = parse_native_yaml_struct(
            es,
            valid_tags,
            ignore_keys,
            path=path,
            skip_native_fns_gen=skip_native_fns_gen,
        )

    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]


# Some assertions are already performed during parsing, but those are only within a single NativeFunction.
# Assertions here are meant to be performed across NativeFunctions.
def error_check_native_functions(funcs: Sequence[NativeFunction]) -> None:
    func_map: Dict[OperatorName, NativeFunction] = {}
    base_func_map: Dict[BaseOperatorName, List[NativeFunction]] = defaultdict(list)
    for f in funcs:
        func_map[f.func.name] = f
        base_func_map[f.func.name.name].append(f)
    for f in funcs:
        if f.structured_delegate is not None:
            delegate_func = func_map[f.structured_delegate]
            assert delegate_func.structured, (
                f"{f.func.name} is marked as a structured_delegate pointing to "
                f"{f.structured_delegate}, but {f.structured_delegate} is not marked as structured. "
                f"Consider adding 'structured=True' to the delegated operator"
            )
        # See Note [resize_ in Functionalization]
        # resize_() is technically an inplace view op (and therefore needs the tag),
        # but it would be overkill to add a true "view" variant of resize.
        # Instead, resize_() gets special treatment in functionalization,
        # and we have a resize() op that is non-aliasing + functional.
        if (
            "inplace_view" in f.tags
            and str(f.func.name) != "resize_"
            and str(f.func.name) != "resize_as_"
        ):
            base_name = f.func.name.name
            overload_name = f.func.name.overload_name
            assert base_name.inplace, (
                f"{f.func.name} is marked with tag: inplace_view, but it doesn't follow the naming "
                "convention for inplace ops - the codegen expects the base name to have a trailing underscore. "
            )
            out_of_place_base_name = BaseOperatorName(
                base_name.base, False, base_name.dunder_method
            )
            assert len(base_func_map[out_of_place_base_name]) > 0, (
                f"{f.func.name} is marked with tag: inplace_view. The codegen expects there to be a corresponding "
                f"out-of-place view op with the name '{base_name}' and matching schema, but it didn't find one. "
            )


def cpp_string(s: str) -> str:
    """Convert a python string into a c++ string literal"""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\a", "\\a")
    s = s.replace("\b", "\\b")
    s = s.replace("\f", "\\f")
    s = s.replace("\n", "\\n")
    s = s.replace("\v", "\\v")
    s = s.replace("\t", "\\t")
    return f'"{s}"'


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                        C++ CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Most functions in this section are curried: they consist of a function
# that takes some parameters (e.g., what is to be generated) which itself
# returns a function that actually maps NativeFunction to the code
# to be generated.  This pattern makes it convenient to use map, concatMap
# and similar functional combinators.


def static_dispatch_keys(backends: List[BackendIndex]) -> List[DispatchKey]:
    if len(backends) == 0:
        return []
    else:
        return [backend.dispatch_key for backend in backends] + [
            DispatchKey.CompositeImplicitAutograd,
            DispatchKey.CompositeImplicitAutogradNestedTensor,
            DispatchKey.CompositeExplicitAutograd,
            DispatchKey.CompositeExplicitAutogradNonFunctional,
        ]


def get_static_dispatch_backend(
    f: NativeFunction, backend_index: BackendIndex
) -> Optional[DispatchKey]:
    if f.structured_delegate is not None or backend_index.has_kernel(f):
        # TODO: for ops with structured_delegate it should check the dispatch table of
        # the out variant instead. For now, these structured ops all have CPU/CUDA kernels
        # so we always dispatch to the `backend`, but this could be wrong when we
        # migrate math/default_backend ops to use structured delegate.
        return backend_index.dispatch_key
    elif f.has_composite_explicit_autograd_kernel:
        return DispatchKey.CompositeExplicitAutograd
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return DispatchKey.CompositeExplicitAutogradNonFunctional
    elif f.has_composite_implicit_autograd_kernel:
        return DispatchKey.CompositeImplicitAutograd
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return DispatchKey.CompositeImplicitAutogradNestedTensor
    return None


def static_dispatch_ops_header(
    f: NativeFunction, backend_index: List[BackendIndex]
) -> Optional[str]:
    if backend_index is None or f.manual_kernel_registration:
        return None

    output = []
    for index in backend_index:
        dispatch_key = get_static_dispatch_backend(f, index)
        if dispatch_key is not None:
            output.append(
                f"#include <ATen/ops/{f.root_name}_{dispatch_key.lower()}_dispatch.h>"
            )
    return "\n".join(output)


def static_dispatch_extra_headers(backends: List[BackendIndex]) -> List[str]:
    return [
        f"#include <ATen/{dispatch_key}Functions.h>"
        for dispatch_key in static_dispatch_keys(backends)
    ]


# Translates arguments of `sig` to CppSignature bindings.
# Note that we have a special case for `memory_format` argument and this case is not covered by
# tools.codegen.api.translate() yet as its application is limited to static dispatch.
def translate_args(
    sig: Union[CppSignature, DispatcherSignature],
    cpp_sig: CppSignature,
) -> str:
    # Adds SpecialArgName.possibly_redundant_memory_format NamedCType for memory_format bindings
    def add_spl_memory_format_binding(input_bindings: List[Binding]) -> List[Binding]:
        output_bindings: List[Binding] = []
        for binding in input_bindings:
            if binding.name == "memory_format":
                spl_mem_format_binding = Binding(
                    nctype=NamedCType(
                        SpecialArgName.possibly_redundant_memory_format,
                        binding.nctype.type,
                    ),
                    name=binding.name,
                    default=binding.default,
                    argument=binding.argument,
                )
                output_bindings.append(spl_mem_format_binding)
            else:
                output_bindings.append(binding)
        return output_bindings

    src_bindings = list(sig.arguments())
    goal_bindings = list(cpp_sig.arguments())
    # When last argument of CPP signature has SpecialArgName.possibly_redundant_memory_format NCType,
    # get memory_format bindings of dispatcher signature to have the same NCType as well
    for arg in goal_bindings:
        if arg.nctype.name == SpecialArgName.possibly_redundant_memory_format:
            src_bindings = add_spl_memory_format_binding(src_bindings)
            break
    exprs = translate(src_bindings, goal_bindings)
    return ", ".join(a.expr for a in exprs)


def generate_static_dispatch_backend_call(
    sig: Union[CppSignature, DispatcherSignature],
    f: NativeFunction,
    backend_index: BackendIndex,
) -> str:
    cpp_sigs = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    assert cpp_sig is not None
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    backend_metadata = backend_index.get_kernel(f)
    kernel_ns = (
        backend_metadata.cpp_namespace
        if backend_metadata and backend_metadata.cpp_namespace
        else DEFAULT_KERNEL_NAMESPACE
    )
    ns = kernel_ns.replace("::native", "")
    return f"return {ns}::{backend_index.dispatch_key.lower()}::{name}({exprs});"


def generate_static_dispatch_fallback_call(
    sig: Union[CppSignature, DispatcherSignature],
    f: NativeFunction,
    backend_indices: List[BackendIndex],
) -> str:
    cpp_sigs = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    assert cpp_sig is not None
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    ns = DEFAULT_KERNEL_NAMESPACE.replace("::native", "")
    if f.has_composite_explicit_autograd_kernel:
        return f"return {ns}::{DispatchKey.CompositeExplicitAutograd.lower()}::{name}({exprs});"
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return f"return {ns}::{DispatchKey.CompositeExplicitAutogradNonFunctional.lower()}::{name}({exprs});"
    elif f.has_composite_implicit_autograd_kernel:
        return f"return {ns}::{DispatchKey.CompositeImplicitAutograd.lower()}::{name}({exprs});"
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return f"return {ns}::{DispatchKey.CompositeImplicitAutogradNestedTensor.lower()}::{name}({exprs});"
    else:
        return f"""TORCH_CHECK(false, "Static dispatch does not support {name} for\
{', '.join([str(index.dispatch_key)for index in backend_indices])} ");"""


def static_dispatch(
    sig: Union[CppSignature, DispatcherSignature],
    f: NativeFunction,
    backend_indices: List[BackendIndex],
) -> str:
    """
    For a given `NativeFunction`, find out the corresponding backend and dispatch to it. If more than one
    backends exsit, fallback to static dispatch by determining dispatch key from inputs.
    Arguments:
        sig: A CppSignature or DispatcherSignature for this native function we want to use.
        f: NativeFunction to generate static dispatch.
        backend_indices: All available backends.
    Return:
        C++ code to call backend-specific functions, e.g., "return at::cpu::add(self, other, scale);"
    """
    if len(backend_indices) == 0 or f.manual_kernel_registration:
        return ""

    keys = [
        b
        for b in backend_indices
        if b.has_kernel(f)
        or (
            f.structured_delegate is not None
            and b.dispatch_key in STRUCTURED_DISPATCH_KEYS
        )
    ]
    if len(keys) == 1:
        return generate_static_dispatch_backend_call(sig, f, keys[0])
    elif len(keys) == 0:
        return generate_static_dispatch_fallback_call(sig, f, backend_indices)

    native_tensor_args = [
        a.name
        for a in sig.arguments()
        if isinstance(a.argument, SelfArgument)
        or isinstance(a.argument, Argument)
        and a.argument.type.is_tensor_like()
    ]
    tensor_args = ", ".join(native_tensor_args)
    tensor_opts = f.func.arguments.tensor_options

    stmts = []
    subexprs: List[str] = []
    if tensor_opts is not None:
        subexprs.append(
            "DispatchKeySet(c10::computeDispatchKey(dtype, layout, device))"
        )
    if tensor_args != "":
        subexprs.append(f"c10::detail::multi_dispatch_key_set({tensor_args})")
    stmts.append(f"""DispatchKeySet _dk_set = {' | '.join(subexprs)};""")
    stmts.append("DispatchKey _dk = c10::highestPriorityBackendTypeId(_dk_set);")

    dispatch_code = []
    for index in keys:
        dispatch_code.append(f"""case DispatchKey::{index.dispatch_key}:""")
        dispatch_code.append(
            f"""\t{generate_static_dispatch_backend_call(sig, f, index)};"""
        )

    fallback = generate_static_dispatch_fallback_call(sig, f, backend_indices)
    connector = "\n\t\t"

    return f"""
    {connector.join(stmts)}
    switch (_dk) {{
        {connector.join(dispatch_code)}
        default:
            {fallback}
    }}
    """


# Generates RegisterSchema.cpp.  Depending on the selector, either
# all schemas are registered, or only some are (in the case of
# selective build)
@dataclass(frozen=True)
class RegisterSchema:
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if not self.selector.is_native_function_selected(f):
            return None
        tags = "{" + ", ".join(f"at::Tag::{tag}" for tag in sorted(f.tags)) + "}"
        return f"m.def({cpp_string(str(f.func))}, {tags});\n"


# Generates Operators.h and Operators.cpp.
# These provide macros that, given an operator and overload name, allow users
# to access an "un-overloaded" function version of the operator. This
# is useful for extension writers who want to (1) want to decltype the operator
# and (2) don't want to worry about method-only operators.
@dataclass(frozen=True)
class ComputeOperators:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: List[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        sig = DispatcherSignature.from_schema(f.func)
        name = f.func.name.unambiguous_name()

        if self.target is Target.DECLARATION:
            # Note [The ATen Operators API]
            # The ATen Operators API lives in the at::_ops namespace, and contains compile-time
            # metadata about each operator + entry points into the Dispatcher.
            # The C++ function, method, and redispatch API's are all implemented as wrappers
            # into various bits of the structs defined here.
            #
            # Important characteristics about the Operators API:
            # (1) It follows the Dispatcher API.
            #     This is kind of necessary to avoid overhead.
            #     For example: if it followed the C++ API, then all of the faithful C++ factory functions
            #     would need to wrap their arguments into TensorOptions only to unwrap them again.
            # (2) Overload names are disambiguated.
            #     This is helpful for pytorch extenders who would like to decltype() an aten operator,
            #     that has overloads, e.g. decltype(at::_ops::mul_Tensor::call)
            # (3) No argument defaulting is allowed.
            #     This is more of an implementation detail to avoid #include cycles,
            #     since TensorBody.h (which defines the Tensor class) needs to include this file.
            # (4) manual_cpp_bindings and faithful names are not included in the API.
            #     This applies to stuff like __dispatch__is_complex(), and add_outf().
            #     These aren't "real aten ops", they're just additional functions provided by the C++ API.
            #     They're implemented as wrappers in Functions.h that call into the actual operators
            #     defined here, i.e. at::_ops::is_complex::call() and at::_ops::add_out::call().
            #     This means that ATEN_OP(is_complex) will not fastpath, and will go through the dispatcher.
            return f"""
struct TORCH_API {name} {{
  using schema = {sig.type()};
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::{f.func.name.name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "{f.func.name.overload_name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, {cpp_string(str(f.func))})
  static {sig.defn(name="call", is_redispatching_fn=False)};
  static {sig.defn(name="redispatch", is_redispatching_fn=True)};
}};"""

        elif self.target is Target.DEFINITION:
            defns = f"""
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, name, "aten::{f.func.name.name}")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, overload_name, "{f.func.name.overload_name}")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, schema_str, {cpp_string(str(f.func))})

// aten::{f.func}
static C10_NOINLINE c10::TypedOperatorHandle<{name}::schema> create_{name}_typed_handle() {{
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow({name}::name, {name}::overload_name)
      .typed<{name}::schema>();
}}
"""
            for is_redispatching_fn in [False, True]:
                if is_redispatching_fn:
                    dispatcher_exprs_str = ", ".join(
                        ["dispatchKeySet"] + [a.name for a in sig.arguments()]
                    )
                    method_base = "redispatch"
                else:
                    dispatcher_exprs_str = ", ".join([a.name for a in sig.arguments()])
                    method_base = "call"

                dispatcher_call = method_base
                method_name = f"{name}::{method_base}"

                fn_body = f"""
    static auto op = create_{name}_typed_handle();
    return op.{dispatcher_call}({dispatcher_exprs_str});"""

                if (
                    not is_redispatching_fn
                    and len(self.static_dispatch_backend_indices) > 0
                ):
                    # call() should go through static dispatch
                    fn_body = static_dispatch(
                        sig, f, backend_indices=self.static_dispatch_backend_indices
                    )
                defns += f"""
// aten::{f.func}
{sig.defn(name=method_name, is_redispatching_fn=is_redispatching_fn)} {{
    {fn_body}
}}
"""
            return defns
        else:
            assert_never(self.target)


# Generates Functions.h, which provides the functional public C++ API,
# and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeFunction:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding
        )
        has_symint = f.func.has_symint()

        result = ""
        for sig in sig_group.signatures():
            # See Note [The ATen Operators API]
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ", ".join([e.expr for e in exprs])

            if sig.symint:
                intlike_t = "c10::SymInt"
            else:
                intlike_t = "int64_t"

            if Variant.function in f.variants:
                result += f"""
// aten::{f.func}
inline {sig.decl()} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}"""

            # The template function can be used from template situations
            # where you want to switch between the symint or not version
            # depending on a template argument
            #
            # NB: we ALWAYS generate this even for methods.  But we put it in
            # this header so it can take advantage of per-op headers
            if has_symint:
                result += f"""
namespace symint {{
  template <typename T, typename = std::enable_if_t<std::is_same<T, {intlike_t}>::value>>
  {sig.decl(suppress_symint_suffix=True)} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
  }}
}}
"""
        return result


# Generates TensorBody.h. This file provides the object-oriented (method-based)
# public C++ API, and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeTensorMethod:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: List[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if Variant.method not in f.variants:
            return None

        assert not f.func.is_out_fn()
        assert f.func.arguments.self_arg is not None

        sig_group = CppSignatureGroup.from_native_function(
            f, method=True, fallback_binding=f.manual_cpp_binding
        )

        if self.target is Target.DECLARATION:
            result = ""
            for sig in sig_group.signatures():
                result += f"{sig.decl()} const;\n"
            return result

        if self.target is not Target.DEFINITION:
            assert_never(self.target)

        result = ""

        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments(), method=True)
            exprs_str = ", ".join([e.expr for e in exprs])

            result += f"""
// aten::{f.func}
inline {sig.defn(prefix="Tensor::")} const {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}
"""

        return result


# Generates RedispatchFunctions.h.
# This is similar to the C++ API defined in Functions.h, but provides access
# to the dispatcher's redispatch API.
@dataclass(frozen=True)
class ComputeRedispatchFunction:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        # We unconditionally generate function variants of the redispatch API.
        # This is mainly because we can namespace functions separately, but not methods,
        sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding
        )

        result = ""
        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ", ".join(["dispatchKeySet"] + [a.expr for a in exprs])

            result += f"""
// aten::{f.func}
inline {sig.decl(is_redispatching_fn=True)} {{
    return at::_ops::{f.func.name.unambiguous_name()}::redispatch({exprs_str});
}}
"""

        return result


# Generates ATenOpList.cpp, a runtime accessible list of all aten
# operators.
# TODO: This was historically used to help some JIT interop code
# figure out whether or not to treat aten namespace'd operators
# one way or another, we should reevaluate if this is actually needed.
@with_native_function
def compute_aten_op(f: NativeFunction) -> str:
    return f'{{"aten::{f.func.name.name}", "{f.func.name.overload_name}"}},'


# Generates MetaFunctions.h
def compute_meta_function_declaration(g: NativeFunctionsGroup) -> Optional[str]:
    if not g.structured:
        return None
    with native_function_manager(g.out):
        name = meta.name(g)
        args = structured.meta_arguments(g)
        args_str = ", ".join(a.decl() for a in args)
        parent_class = g.out.structured_inherits
        if parent_class is None:
            parent_class = "at::impl::MetaBase"
        meta_return = "void"
        precomputed = g.out.precomputed if g.structured else None

        if precomputed:
            # Generate the template declaration with one bool parameter for each
            # precomputed element. Each parameter is true if the corresponding (in
            # terms of position) precomputed element has been set.
            precomputed_values = [*precomputed.replace.values(), precomputed.add]
            precomputed_elements = [
                elem for replace_list in precomputed_values for elem in replace_list
            ]
            precomputed_template_parameters = [
                elem.name.upper() for elem in precomputed_elements
            ]
            precomputed_template_params_str = ", ".join(
                f"bool {param} = false" for param in precomputed_template_parameters
            )
            precompute_template_decl = f"template <{precomputed_template_params_str}>"

            # Generate a string containing declarations of all precomputed elements.
            precomputed_elements_with_cpp_types = [
                structured.argument_type(elem, binds=elem.name)
                for elem in precomputed_elements
            ]

            precomputed_elements_decl = ";\n".join(
                f"{elem.cpp_type(strip_ref=True)} {elem.name}"
                for elem in precomputed_elements_with_cpp_types
            )

            # Generate "setter" methods for each precomputed element. Each method will return
            # a new instance of precompute_out with the template parameter that corresponds to
            # the member set by the method to true (to indicate that it has been set).
            setter_methods = []
            for i, elem in enumerate(precomputed_elements):
                # Generate the signature. The return type will be the same
                # as the type of `this` but with the template parameter
                # corresponding to the element set by this method set to true.
                # The assert generated below will ensure that this template
                # parameter is false on the type of `this`.
                return_ty_templates = ", ".join(
                    precomputed_template_parameters[:i]
                    + ["true"]
                    + precomputed_template_parameters[i + 1 :]
                )
                return_ty = f"precompute_out<{return_ty_templates}>"
                elem_cpp_ty = precomputed_elements_with_cpp_types[i].cpp_type(
                    strip_ref=True
                )
                signature = f"{return_ty} set_{elem.name}({elem_cpp_ty} value)"

                # Generate an assert which checks that the
                # template parameter corresponding to the precomputed
                # element that is set by this method is false on the
                # class corresponding to the object that `this` points to.
                # This ensures that each element can be set only once.
                assert_msg = f'"{precomputed_elements[i].name} already set"'
                assert_stmt = f"static_assert({precomputed_template_parameters[i]} == false, {assert_msg});"

                # Generate the new object construction block. All state
                # except the element that this method sets is copied from the
                # object that `this` points to. The value for the element that
                # the method sets is taken from a method parameter.
                construction_stmts = []
                construction_stmts.append(f"{return_ty} ret;")

                for j, elem in enumerate(precomputed_elements):
                    if i == j:
                        construction_stmts.append(f"ret.{elem.name} = value;")
                    else:
                        construction_stmts.append(
                            f"ret.{elem.name} = this->{elem.name};"
                        )

                construction_stmts.append("return ret;")
                construction_block = "\n".join(construction_stmts)

                setter_methods.append(
                    f"""
                    {signature} {{
                        {assert_stmt}
                        {construction_block}
                    }}
                """
                )
            setter_methods_decl = "\n".join(setter_methods)

            # Meta should return an instance of the struct containing the precomputed elements.
            meta_return_template_params = ", ".join(
                ["true"] * len(precomputed_template_parameters)
            )
            # This typedef (actually a using statement) is needed so that TORCH_META_FUNC can reuse the return
            # type (which has a variable number of template parameters).
            meta_return_typedef = f"using meta_return_ty = precompute_out <{meta_return_template_params}>;"
            meta_return = "meta_return_ty"
            precomputed_decl = f"""
                {precompute_template_decl}
                struct TORCH_API precompute_out {{
                    {setter_methods_decl}
                    {precomputed_elements_decl};
            }};"""
        else:
            meta_return_typedef = ""
            precomputed_decl = ""

        return f"""\
struct TORCH_API structured_{name} : public {parent_class} {{
    {precomputed_decl}
    {meta_return_typedef}
    {meta_return} meta({args_str});
}};
"""


def needs_backend_select(f: NativeFunction, selector: SelectiveBuilder) -> bool:
    name = str(f.func.name.name)
    if name.endswith("_like") or name.startswith("new_"):
        return False
    if f.func.arguments.tensor_options is None:
        return False
    return selector.is_native_function_selected(f)


# Generates RegisterBackendSelect.cpp, a series of kernels which provide
# specialized computation of dispatch key for operator signatures which cannot
# be easily done automatically using templating.
@dataclass(frozen=True)
class ComputeBackendSelect:
    target: Literal[Target.DEFINITION, Target.REGISTRATION]

    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if not needs_backend_select(f, self.selector):
            return None

        name = native.name(f.func)
        # BackendSelect can go to Meta, so it must preserve symints
        native_sig = NativeSignature(f.func, symint=True)

        native_tensor_args = [
            a
            for a in native_sig.arguments()
            if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()
        ]

        dispatcher_sig = DispatcherSignature.from_schema(f.func)

        sig: Union[NativeSignature, DispatcherSignature]
        sig = dispatcher_sig
        dispatcher_exprs = dispatcher_sig.exprs()
        dispatch_key = "c10::computeDispatchKey(dtype, layout, device)"

        if self.target is Target.DEFINITION:
            # I don't think there's actually a good reason to generate
            # these two cases differently
            # The first case could probably be improved though- it calls computeDispatchKeySet(),
            # which looks at TLS dispatch keys- there should not be any by the time we reach backend select.
            if native_tensor_args:
                assert f.func.arguments.has_tensor_arg()
                tensor_args = ", ".join(a.name for a in native_tensor_args)
                compute_dk = f"""\
DispatchKeySet _dk_set = c10::DispatchKeySet({dispatch_key}) | c10::detail::multi_dispatch_key_set({tensor_args});
DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
DispatchKeySet _dk = c10::impl::computeDispatchKeySet(_dk_set, _dk_mask);"""
            else:
                assert not f.func.arguments.has_tensor_arg()
                compute_dk = (
                    f"DispatchKeySet _dk = c10::DispatchKeySet({dispatch_key});"
                )
            return f"""\
// aten::{f.func}
C10_ALWAYS_INLINE
{sig.defn(name)} {{
  {compute_dk}
  return at::_ops::{f.func.name.unambiguous_name()}::redispatch(
      _dk, {', '.join(a.expr for a in dispatcher_exprs)});
}}
"""
        elif self.target is Target.REGISTRATION:
            return f"""m.impl("aten::{f.func.name}", TORCH_FN({name}));"""
        else:
            assert_never(self.target)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                       YAML CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def format_yaml(data: object) -> str:
    # Ignore alias in Dumper
    YamlDumper.ignore_aliases = lambda self, data: True  # type: ignore[assignment]

    # Support serializing OrderedDict
    def dict_representer(dumper: Any, data: Any) -> Any:
        return dumper.represent_dict(data.items())

    YamlDumper.add_representer(OrderedDict, dict_representer)  # type: ignore[no-untyped-call]
    # Some yaml parsers (e.g. Haskell's) don't understand line breaks.
    # width=1e9 turns off optional line breaks and improves
    # the portability of the outputted yaml.
    return yaml.dump(data, default_flow_style=False, Dumper=YamlDumper, width=1e9)  # type: ignore[no-any-return, call-overload]


# For some reason, some defaults we write to YAML are written as native
# YAML objects, rather than doing them uniformly as strings.  This
# function detects those cases and converts them into native Python
# objects.
def pythonify_default(s: str) -> object:
    if s == "true":
        return True
    elif s == "false":
        return False

    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


# What is a dynamic type?  Over time, the semantic meaning of
# dynamic type has degraded to meaninglessness (in the old days,
# it captured dtype-ness of types, but that has gone away with
# the removal of TH).  These days, it's mostly the same thing as
# the C++ API argument type, except that Tensor and Tensor?
# arguments simply present as Tensor.
#
# TODO: Get rid of dynamic_type, after getting tools/autograd
# to use the new codegen framework
def dynamic_type(t: Type) -> str:
    if isinstance(t, OptionalType):
        return dynamic_type(t.elem)
    # Note we don't use t.is_tensor_like() here because it would
    # also include Tensor[]
    if str(t) == "Tensor":
        return "at::Tensor"
    # This is a legacy concept, so never report SymInt
    return cpp.argumenttype_type(
        t, mutable=False, binds="__placeholder__", symint=False
    ).cpp_type()


def compute_method_of_yaml(variants: Set[Variant]) -> List[str]:
    # This is written out explicitly to ensure that Tensor and
    # namespace are put into the list in the right order
    method_of = ["Type"]
    if Variant.method in variants:
        method_of.append("Tensor")
    if Variant.function in variants:
        method_of.append("namespace")
    return method_of


def compute_returns_yaml(
    f: NativeFunction,
) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    # Note [name and field_name]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # To understand name_to_field_name, we must first talk about this
    # schema:
    #
    #   lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
    #
    # There is something very odd about this schema: it is an out
    # variant of the function (that is to say, it will convert into
    # at::lstsq_out() in the C++ API), but the names of the output
    # return arguments don't match the keyword argument names of
    # the inputs.  It TURNS OUT that in this situation, the historical
    # Declarations.yaml we want to output is this (abbreviated to
    # only show relevant fields):
    #
    #   arguments:
    #     ...
    #   - field_name: solution
    #     name: X
    #   - field_name: QR
    #     name: qr
    #     ...
    #
    #   returns:
    #   - field_name: solution
    #     name: X
    #   - field_name: QR
    #     name: qr
    #
    # The name of the return fields is stored in 'field_name', and the
    # name of the arguments is stored in 'name'.  So when we process
    # arguments, we need a way to get at the corresponding return.  At
    # the moment, this is most conveniently done by constructing a
    # mapping from name (the argument concept) to field_name (the
    # return concept) while processing return arguments, since we don't
    # directly maintain this correspondence in the modeling of function
    # schema itself.
    #
    # See also https://github.com/pytorch/pytorch/issues/43114
    name_to_field_name: Dict[str, str] = {}

    # Compute the returns field of the YAML entry
    names = cpp.return_names(f)
    returns = []
    for i, (r, name) in enumerate(zip(f.func.returns, names)):
        ret = {
            "dynamic_type": dynamic_type(r.type),
            "name": name,
            # legacy, report ints
            "type": cpp.return_type(r, symint=False).cpp_type(),
        }

        if r.name:
            # See Note [name and field_name]
            ret["field_name"] = r.name
            if f.func.is_out_fn():
                name_to_field_name[f.func.arguments.out[i].name] = r.name

        returns.append(ret)

    return returns, name_to_field_name


# arguments in yaml roughly corresponds to the public C++ API
def compute_cpp_argument_yaml(
    cpp_a: Binding,
    *,
    schema_order: bool,
    kwarg_only_set: Set[str],
    out_arg_set: Set[str],
    name_to_field_name: Dict[str, str],
) -> object:
    if isinstance(cpp_a.argument, TensorOptionsArguments):
        arg: Dict[str, object] = {
            "annotation": None,
            "dynamic_type": "at::TensorOptions",
            "is_nullable": False,
            "name": cpp_a.name,
            "type": cpp_a.type,
            "kwarg_only": True,
        }
        if cpp_a.default is not None:
            arg["default"] = cpp_a.default
        return arg
    elif isinstance(cpp_a.argument, SelfArgument):
        raise AssertionError()
    elif isinstance(cpp_a.argument, Argument):
        return compute_argument_yaml(
            cpp_a.argument,
            schema_order=schema_order,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )


def compute_argument_yaml(
    a: Argument,
    *,
    schema_order: bool,
    kwarg_only_set: Set[str],
    out_arg_set: Set[str],
    name_to_field_name: Dict[str, str],
) -> object:
    arg: Dict[str, object] = {
        "annotation": str(a.annotation) if a.annotation else None,
        "dynamic_type": dynamic_type(a.type),
        "is_nullable": a.type.is_nullable(),
        "name": a.name,
        # legacy, report ints
        "type": cpp.argument_type(a, binds="__placeholder__", symint=False).cpp_type(),
    }
    if a.default is not None:
        arg["default"] = pythonify_default(
            cpp.default_expr(a.default, a.type, symint=False)
        )
    if a.name in kwarg_only_set:
        arg["kwarg_only"] = True
    if a.name in out_arg_set:
        arg["output"] = True
        arg["allocate"] = True
        # See Note [name and field_name]
        if a.name in name_to_field_name:
            arg["field_name"] = name_to_field_name[a.name]
    # Historically, booleans don't get their size recorded, because it
    # is already built into the cpp type (e.g., std::array<bool, 4>)
    l = a.type.is_list_like()
    if l is not None and l.size is not None and str(l.elem) != "bool":
        arg["size"] = l.size
    return arg


@with_native_function
def compute_declaration_yaml(f: NativeFunction) -> object:
    returns, name_to_field_name = compute_returns_yaml(f)

    # These sets are used to conveniently test if an argument is a
    # kwarg-only or out argument
    kwarg_only_set = {a.name for a in f.func.arguments.flat_kwarg_only}
    out_arg_set = {a.name for a in f.func.arguments.out}

    sig_group = CppSignatureGroup.from_native_function(
        f, method=False, fallback_binding=False
    )
    cpp_args = sig_group.signature.arguments()
    arguments = [
        compute_cpp_argument_yaml(
            cpp_a,
            schema_order=False,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )
        for cpp_a in cpp_args
    ]

    schema_order_jit_arguments = list(f.func.schema_order_arguments())

    schema_order_arguments = [
        compute_argument_yaml(
            a,
            schema_order=True,
            kwarg_only_set=kwarg_only_set,
            out_arg_set=out_arg_set,
            name_to_field_name=name_to_field_name,
        )
        for a in schema_order_jit_arguments
    ]

    cpp_schema_order_types = [
        # NB: method here doesn't matter
        r.type
        for a in schema_order_jit_arguments
        for r in cpp.argument(
            a,
            method=False,
            cpp_no_default_args=set(),
            faithful=False,
            symint=False,
            has_tensor_options=False,
        )
    ]

    # legacy, report ints
    cpp_returns = cpp.returns_type(f.func.returns, symint=False).cpp_type()
    schema_order_cpp_signature = f"{cpp_returns} ({', '.join(cpp_schema_order_types)})"

    is_factory_method = (
        any(isinstance(a.argument, TensorOptionsArguments) for a in cpp_args)
        and Variant.method not in f.variants
    )

    return OrderedDict(
        [
            ("name", cpp.name(f.func)),
            ("operator_name", str(f.func.name.name)),
            ("overload_name", str(f.func.name.overload_name)),
            ("manual_kernel_registration", f.manual_kernel_registration),
            (
                "category_override",
                f.category_override if f.category_override is not None else "",
            ),
            ("schema_string", f"aten::{f.func}"),
            ("arguments", arguments),
            ("schema_order_cpp_signature", schema_order_cpp_signature),
            ("schema_order_arguments", schema_order_arguments),
            ("method_of", compute_method_of_yaml(f.variants)),
            ("mode", "native"),
            ("python_module", "" if f.python_module is None else f.python_module),
            ("returns", returns),
            ("inplace", f.func.name.name.inplace),
            ("is_factory_method", is_factory_method),
            ("abstract", f.is_abstract),
            ("device_guard", f.device_guard),
            ("with_gil", False),
            ("deprecated", False),
            ("has_math_kernel", f.has_composite_implicit_autograd_kernel),
        ]
    )


# See Note [Auto generated composite kernels]
def has_autogenerated_composite_kernel(f: NativeFunction) -> bool:
    return (f.structured or f.structured_delegate is not None) and (
        f.func.kind() == SchemaKind.functional or f.func.kind() == SchemaKind.inplace
    )


@with_native_function_and_indices
def compute_registration_declarations(
    f: NativeFunction, backend_indices: Dict[DispatchKey, BackendIndex]
) -> str:
    name = dispatcher.name(f.func)
    returns_type = dispatcher.returns_type(
        f.func.returns
    ).cpp_type_registration_declarations()
    args = dispatcher.arguments(f.func)
    args_str = ", ".join(a.no_default().decl_registration_declarations() for a in args)
    comment_data: Dict[str, str] = {
        "schema": f"aten::{f.func}",
        # TODO: What exactly is the semantics of the 'dispatch' field?
        "dispatch": str(
            {k for k, v in backend_indices.items() if v.has_kernel(f)}
            != {DispatchKey.CompositeImplicitAutograd}
            and {k for k, v in backend_indices.items() if v.has_kernel(f)}
            != {
                DispatchKey.CompositeImplicitAutograd,
                DispatchKey.CompositeImplicitAutogradNestedTensor,
            }
        ),
        "default": str(f.has_composite_kernel or has_autogenerated_composite_kernel(f)),
    }
    return f"""{returns_type} {name}({args_str}); // {json.dumps(comment_data)}
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           RUN IT ALL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def get_custom_build_selector(
    provided_op_registration_allowlist: Optional[List[str]],
    op_selection_yaml_path: Optional[str],
) -> SelectiveBuilder:
    assert not (
        provided_op_registration_allowlist is not None
        and op_selection_yaml_path is not None
    ), (
        "Both provided_op_registration_allowlist and "
        + "op_selection_yaml_path can NOT be provided at the "
        + "same time."
    )

    op_registration_allowlist: Optional[Set[str]] = None
    if provided_op_registration_allowlist is not None:
        op_registration_allowlist = set(provided_op_registration_allowlist)

    if op_registration_allowlist is not None:
        selector = SelectiveBuilder.from_legacy_op_registration_allow_list(
            op_registration_allowlist,
            True,
            False,
        )
    elif op_selection_yaml_path is not None:
        selector = SelectiveBuilder.from_yaml_path(op_selection_yaml_path)
    else:
        selector = SelectiveBuilder.get_nop_selector()

    return selector


def get_grouped_by_view_native_functions(
    native_functions: Sequence[NativeFunction],
) -> Sequence[Union[NativeFunction, NativeFunctionsViewGroup]]:
    def maybe_create_view_group(
        d: Dict[Union[ViewSchemaKind, SchemaKind], NativeFunction]
    ) -> List[Union[NativeFunction, NativeFunctionsViewGroup]]:
        funcs: List[Union[NativeFunction, NativeFunctionsViewGroup]] = []
        if ViewSchemaKind.aliasing in d:
            view = d.pop(ViewSchemaKind.aliasing)
            view_inplace = d.pop(ViewSchemaKind.aliasing_inplace, None)
            view_copy = d.pop(SchemaKind.functional, None)

            funcs.append(
                NativeFunctionsViewGroup(
                    view=view,
                    view_copy=view_copy,
                    view_inplace=view_inplace,
                )
            )
        # Take the remaining functions that weren't part of the view group
        # and emit them separately
        for func in d.values():
            funcs.append(func)
        return funcs

    grouped_by_views: Dict[
        FunctionSchema, Dict[Union[SchemaKind, ViewSchemaKind], NativeFunction]
    ] = defaultdict(dict)
    for f in native_functions:
        schema = f.func.view_signature()
        view_kind: ViewSchemaKind = f.view_schema_kind
        # We need to group up ops relevant to the same "view", consisting of:
        # view op (ViewSchemaKind.aliasing)
        # view_inplace op (ViewSchemaKind.aliasing_inplace)
        # view_copy op (SchemaKind.functional)
        if view_kind == ViewSchemaKind.non_aliasing:
            kind = f.func.kind()
            assert kind not in grouped_by_views[schema]
            grouped_by_views[schema][kind] = f
        else:
            assert view_kind not in grouped_by_views[schema]
            grouped_by_views[schema][view_kind] = f

    return list(concatMap(maybe_create_view_group, grouped_by_views.values()))


def get_grouped_native_functions(
    native_functions: Sequence[NativeFunction],
) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
    def flatten_pre_group(
        d: Dict[SchemaKind, NativeFunction]
    ) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
        r = NativeFunctionsGroup.from_dict(d)
        if r is None:
            # Invariant: any NativeFunctions that are code-generated
            # should have been grouped into NativeFunctionsGroup objects
            assert not any("generated" in f.tags for f in d.values())
            return list(d.values())
        else:
            return [r]

    # TODO: how come ValuesView isn't a Sequence lol
    pre_grouped_native_functions = pre_group_native_functions(native_functions)
    return list(
        concatMap(flatten_pre_group, list(pre_grouped_native_functions.values()))
    )


# Return native function declarations grouped by their namespaces.
def get_native_function_declarations(
    *,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    backend_indices: Dict[DispatchKey, BackendIndex],
    native_function_decl_gen: Callable[
        [Union[NativeFunctionsGroup, NativeFunction], BackendIndex], List[str]
    ] = dest.compute_native_function_declaration,
) -> List[str]:
    """
    Generate kernel declarations, in `NativeFunction(s).h`.
    :param grouped_native_functions: a sequence of `NativeFunction` or `NativeFunctionGroup`.
    :param backend_indices: kernel collections grouped by dispatch key.
    :param native_function_decl_gen: callable to generate kernel declaration for each `NativeFunction`.
    :return: a list of string, from the string with all declarations, grouped by namespaces, split by newline.
    """
    declarations: List[str] = []
    ns_grouped_kernels: Dict[str, List[str]] = defaultdict(list)
    newline = "\n"
    for f in grouped_native_functions:
        native_function_namespaces = set()
        dispatch_keys = set()
        for dispatch_key, backend_idx in backend_indices.items():
            backend_metadata = backend_idx.get_kernel(f)
            if backend_metadata:
                namespace = backend_metadata.cpp_namespace
                dispatch_keys.add(dispatch_key)
                native_function_namespaces.add(namespace)
            else:
                namespace = DEFAULT_KERNEL_NAMESPACE
            assert (
                len(native_function_namespaces) <= 1
            ), f"Codegen only supports one namespace per operator, got {native_function_namespaces} from {dispatch_keys}"
            ns_grouped_kernels[namespace].extend(
                native_function_decl_gen(f, backend_idx)
            )

    for namespace, kernels in ns_grouped_kernels.items():
        ns_helper = NamespaceHelper(
            namespace_str=namespace,
            entity_name="",
            max_level=4,
        )
        # Convert to a set first to remove duplicate kernel names. Backends are
        # allowed to repeat kernel names; only generate the declaration once!
        ordered_kernels = list(OrderedDict.fromkeys(kernels))
        declarations.extend(
            f"""
{ns_helper.prologue}
{newline.join(ordered_kernels)}
{ns_helper.epilogue}
        """.split(
                newline
            )
        )
    return declarations


def get_kernel_namespace(
    *, f: Union[NativeFunction, NativeFunctionsGroup], backend_idx: BackendIndex
) -> str:
    backend_metadata = backend_idx.get_kernel(f)
    assert not backend_metadata or "::native" in backend_metadata.cpp_namespace, (
        f"The kernel for function {f.func.name if isinstance(f, NativeFunction) else f.functional.func.name} "
        f"with dispatch key {backend_idx.dispatch_key}"
        f" has a namespace {backend_metadata.cpp_namespace} and it's not ending with '::native'."
    )
    return (
        backend_metadata.cpp_namespace if backend_metadata else DEFAULT_KERNEL_NAMESPACE
    )


# Return native function definitions grouped by dispatch key and custom namespace.
# Used in RegisterDispatchKey.cpp and etc.
def get_native_function_definitions(
    *,
    fm: FileManager,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    dispatch_key: DispatchKey,
    backend_idx: BackendIndex,
    selector: SelectiveBuilder,
    rocm: bool,
    symint: bool,
    skip_dispatcher_op_registration: bool,
    gen_dispatch_helpers: bool,
) -> List[str]:
    definitions: List[str] = []
    ns_definitions: Dict[str, List[str]] = defaultdict(list)
    anonymous_definitions: Dict[str, List[str]] = defaultdict(list)
    registrations: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    newline = "\n"
    ns_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.NAMESPACED_DEFINITION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    anonymous_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.ANONYMOUS_DEFINITION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    reg_gen = dest.RegisterDispatchKey(
        backend_idx,
        Target.REGISTRATION,
        selector,
        rocm=rocm,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=skip_dispatcher_op_registration,
    )
    for f in grouped_native_functions:
        kernel_namespace = get_kernel_namespace(f=f, backend_idx=backend_idx).replace(
            "::native", ""
        )

        ns_definitions[kernel_namespace].extend(
            ns_gen(f),
        )
        anonymous_definitions[kernel_namespace].extend(
            anonymous_gen(f),
        )
        namespace = (
            f.namespace if isinstance(f, NativeFunction) else f.functional.namespace
        )
        if namespace not in registrations[kernel_namespace]:
            registrations[kernel_namespace] = defaultdict(list)
        registrations[kernel_namespace][namespace].extend(
            reg_gen(f),
        )

    for kernel_namespace in ns_definitions:
        if len(ns_definitions[kernel_namespace]) == 0:
            continue
        ns_helper = NamespaceHelper(namespace_str=kernel_namespace)
        registration_body = ""
        for namespace in registrations[kernel_namespace]:
            if not registrations[kernel_namespace][namespace]:
                continue
            registration_body += f"""
TORCH_LIBRARY_IMPL({namespace}, {dispatch_key}, m) {{
    {newline.join(registrations[kernel_namespace][namespace])}
}};"""
        definitions.extend(
            fm.substitute_with_template(
                "RegisterDispatchDefinitions.ini",
                lambda: {
                    "ns_prologue": ns_helper.prologue,
                    "ns_epilogue": ns_helper.epilogue,
                    "dispatch_helpers": dest.gen_registration_helpers(backend_idx)
                    if gen_dispatch_helpers
                    else [],
                    "dispatch_anonymous_definitions": anonymous_definitions[
                        kernel_namespace
                    ],
                    "static_init_dispatch_registrations": ""
                    if skip_dispatcher_op_registration
                    else registration_body,
                    "deferred_dispatch_registrations": "",
                    "dispatch_namespace": dispatch_key.lower(),
                    "dispatch_namespaced_definitions": ns_definitions[kernel_namespace],
                },
            ).split(newline)
        )

    return definitions


# Return native function declarations grouped by dispatch key and custom namespace.
# Used in CPUFunctions_inl.h and etc.
def get_namespaced_declaration(
    *,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    dispatch_key: DispatchKey,
    backend_idx: BackendIndex,
    selector: SelectiveBuilder,
    rocm: bool,
    symint: bool,
) -> List[str]:
    declarations: List[str] = []
    ns_grouped_kernels: Dict[str, List[str]] = defaultdict(list)
    newline = "\n"
    func = dest.RegisterDispatchKey(
        backend_idx,
        Target.NAMESPACED_DECLARATION,
        selector,
        rocm=rocm,
        class_method_name=None,
        skip_dispatcher_op_registration=False,
        symint=symint,
    )
    for f in grouped_native_functions:
        namespace = get_kernel_namespace(f=f, backend_idx=backend_idx).replace(
            "native", dispatch_key.lower()
        )

        ns_grouped_kernels[namespace].extend(
            func(f),
        )

    for namespace, kernels in ns_grouped_kernels.items():
        if len(kernels) == 0:
            continue
        ns_helper = NamespaceHelper(
            namespace_str=namespace, entity_name="", max_level=3
        )
        ordered_kernels = list(OrderedDict.fromkeys(kernels))
        declarations.extend(
            f"""
{ns_helper.prologue}
{newline.join(ordered_kernels)}
{ns_helper.epilogue}
        """.split(
                newline
            )
        )
    return declarations


# Return native function schema registration code for aten and other namespaces.
def get_native_function_schema_registrations(
    *,
    native_functions: Sequence[NativeFunction],
    schema_selector: SelectiveBuilder,
) -> Tuple[List[str], str]:
    ns_native_functions: Dict[str, List[NativeFunction]] = defaultdict(list)
    for native_function in native_functions:
        ns_native_functions[native_function.namespace].append(native_function)
    schema_registrations = ""
    aten_schema_registrations = []
    custom_namespace = None
    for namespace, funcs in ns_native_functions.items():
        schema_registrations_body = list(
            mapMaybe(RegisterSchema(schema_selector), funcs)
        )
        # NB: we have to separate aten namespace registration from other namespaces,
        # because in the template we hardcoded an operator for ATen already.
        if namespace == "aten":
            aten_schema_registrations = schema_registrations_body
        else:
            custom_namespace = namespace
            tab = "\t"
            # if the namespace is predefined, we should use define a library fragment
            # instead of a new library
            torch_library_macro = (
                "TORCH_LIBRARY_FRAGMENT"
                if namespace in FRAGMENT_NAMESPACES
                else "TORCH_LIBRARY"
            )
            schema_registrations += f"""
{torch_library_macro}({custom_namespace}, m) {{
  {tab.join(schema_registrations_body)}
}};"""
    return (aten_schema_registrations, schema_registrations)


def gen_aggregated_headers(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    static_dispatch_idx: List[BackendIndex],
    selector: SelectiveBuilder,
    backend_indices: Dict[DispatchKey, BackendIndex],
    cpu_fm: FileManager,
    cuda_fm: FileManager,
    functions_keys: Set[DispatchKey],
    dispatch_keys: Sequence[DispatchKey],
    rocm: bool,
) -> None:
    # Buck doesn't support dynamic output files, so we aggregate all operator
    # headers into a single file
    cpu_fm.write(
        "NativeMetaFunctions.h",
        lambda: {
            "NativeMetaFunctions_includes": [],
            "NativeMetaFunctions_declarations": list(
                mapMaybe(compute_meta_function_declaration, structured_native_functions)
            ),
        },
    )
    method_native_functions = [
        fn for fn in native_functions if Variant.method in fn.variants
    ]
    non_method_native_functions = [
        fn for fn in native_functions if fn not in method_native_functions
    ]
    cpu_fm.write(
        "MethodOperators.h",
        lambda: {
            "MethodOperators_includes": [],
            "MethodOperators_declarations": list(
                mapMaybe(
                    ComputeOperators(
                        Target.DECLARATION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    method_native_functions,
                )
            ),
        },
    )
    cpu_fm.write(
        "Operators.h",
        lambda: {
            "Operators_includes": ["#include <ATen/MethodOperators.h>"],
            "Operators_declarations": list(
                mapMaybe(
                    ComputeOperators(
                        Target.DECLARATION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    non_method_native_functions,
                )
            ),
        },
    )
    cpu_fm.write(
        "Functions.h",
        lambda: {
            "static_dispatch_extra_headers": static_dispatch_extra_headers(
                static_dispatch_idx
            ),
            "Functions_includes": ["#include <ATen/Operators.h>"],
            "Functions_declarations": list(
                mapMaybe(
                    ComputeFunction(),
                    native_functions,
                )
            ),
        },
    )
    declarations = get_native_function_declarations(
        grouped_native_functions=grouped_native_functions,
        backend_indices=backend_indices,
    )
    cpu_fm.write(
        "NativeFunctions.h",
        lambda: {
            "NativeFunctions_includes": ["#include <ATen/NativeMetaFunctions.h>"],
            "NativeFunctions_declarations": declarations,
        },
    )

    for dispatch_key in dispatch_keys:
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm
        if dispatch_key in functions_keys:
            inl_headers = f"#include <ATen/{dispatch_key}Functions_inl.h>"

            fm.write_with_template(
                f"{dispatch_key}Functions.h",
                "DispatchKeyFunctions.h",
                lambda: {
                    "dispatch_key": str(dispatch_key),
                    "inline_headers": inl_headers,
                },
            )
            fm.write_with_template(
                f"{dispatch_key}Functions_inl.h",
                "DispatchKeyFunctions_inl.h",
                lambda: {
                    "DispatchKeyFunctions_inl_includes": [],
                    "dispatch_namespace": dispatch_key.lower(),
                    "dispatch_namespaced_declarations": get_namespaced_declaration(
                        grouped_native_functions=grouped_native_functions,
                        dispatch_key=dispatch_key,
                        backend_idx=backend_indices[dispatch_key],
                        selector=selector,
                        rocm=rocm,
                        symint=True,
                    ),
                },
            )

        del fm


def gen_per_operator_headers(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    static_dispatch_idx: List[BackendIndex],
    selector: SelectiveBuilder,
    backend_indices: Dict[DispatchKey, BackendIndex],
    cpu_fm: FileManager,
    cuda_fm: FileManager,
    ops_fm: FileManager,
    functions_keys: Set[DispatchKey],
    dispatch_keys: Sequence[DispatchKey],
    rocm: bool,
) -> None:
    # For CMake builds, split operator declarations into separate headers in
    # the ATen/ops folder to split up header dependencies
    functions_by_root_name: Dict[str, List[NativeFunction]] = defaultdict(lambda: [])
    for fn in native_functions:
        functions_by_root_name[fn.root_name].append(fn)

    grouped_functions_by_root_name: Dict[
        str, List[Union[NativeFunction, NativeFunctionsGroup]]
    ] = defaultdict(lambda: [])
    for group in grouped_native_functions:
        name = group.root_name
        grouped_functions_by_root_name[name].append(group)

    for name, functions in functions_by_root_name.items():
        ops_fm.write_with_template(
            f"{name}_ops.h",
            "Operator.h",
            lambda: {
                "declarations": list(
                    mapMaybe(
                        ComputeOperators(
                            Target.DECLARATION,
                            static_dispatch_backend_indices=static_dispatch_idx,
                        ),
                        functions,
                    )
                ),
            },
        )

        ops_fm.write_with_template(
            f"{name}.h",
            "Function.h",
            lambda: {
                "static_dispatch_ops_headers": list(
                    mapMaybe(
                        lambda fn: static_dispatch_ops_header(
                            fn, backend_index=static_dispatch_idx
                        ),
                        functions,
                    )
                ),
                "operator_includes": f"#include <ATen/ops/{name}_ops.h>",
                "function_definitions": list(
                    mapMaybe(
                        ComputeFunction(),
                        functions,
                    )
                ),
            },
        )

        grouped_functions = grouped_functions_by_root_name.get(name, [])
        structured_functions = [
            fn
            for fn in grouped_functions
            if isinstance(fn, NativeFunctionsGroup) and fn.structured
        ]
        is_structured = len(structured_functions) > 0

        if is_structured:
            ops_fm.write_with_template(
                f"{name}_meta.h",
                "NativeMetaFunction.h",
                lambda: {
                    "meta_function_declarations": list(
                        mapMaybe(
                            compute_meta_function_declaration, structured_functions
                        )
                    ),
                },
            )
        declarations = get_native_function_declarations(
            grouped_native_functions=grouped_functions,
            backend_indices=backend_indices,
            native_function_decl_gen=dest.compute_native_function_declaration,
        )
        ops_fm.write_with_template(
            f"{name}_native.h",
            "NativeFunction.h",
            lambda: {
                "extra_includes": (
                    f"#include <ATen/ops/{name}_meta.h>" if is_structured else []
                ),
                "native_function_declarations": declarations,
            },
        )

    for category, suffix in [
        ("Functions", ""),
        ("Operators", "_ops"),
        ("NativeMetaFunctions", "_meta"),
        ("NativeFunctions", "_native"),
    ]:
        cpu_fm.write(
            f"{category}.h",
            lambda: {
                f"{category}_includes": [
                    f"#include <ATen/ops/{name}{suffix}.h>"
                    for name in sorted(functions_by_root_name.keys())
                ],
                f"{category}_declarations": [],
            },
        )

    for dispatch_key in dispatch_keys:
        if dispatch_key not in functions_keys:
            continue

        dispatch_namespace = dispatch_key.lower()
        dispatch_names = []

        for name, functions in functions_by_root_name.items():
            grouped_functions = grouped_functions_by_root_name.get(name, [])
            declarations = list(
                concatMap(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.NAMESPACED_DECLARATION,
                        selector,
                        rocm=rocm,
                        symint=True,
                        class_method_name=None,
                        skip_dispatcher_op_registration=False,
                    ),
                    grouped_functions,
                )
            )

            if len(declarations) == 0:
                continue

            dispatch_names.append(name)
            ops_fm.write_with_template(
                f"{name}_{dispatch_namespace}_dispatch.h",
                "DispatchKeyFunction.h",
                lambda: {
                    "dispatch_namespace": dispatch_namespace,
                    "dispatch_namespaced_declarations": declarations,
                },
            )

        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm
        inl_headers = f"#include <ATen/{dispatch_key}Functions_inl.h>"

        fm.write_with_template(
            f"{dispatch_key}Functions.h",
            "DispatchKeyFunctions.h",
            lambda: {
                "dispatch_key": str(dispatch_key),
                "inline_headers": inl_headers,
            },
        )
        fm.write_with_template(
            f"{dispatch_key}Functions_inl.h",
            "DispatchKeyFunctions_inl.h",
            lambda: {
                "dispatch_namespace": dispatch_namespace,
                "DispatchKeyFunctions_inl_includes": [
                    f"#include <ATen/ops/{name}_{dispatch_namespace}_dispatch.h>"
                    for name in sorted(dispatch_names)
                ],
                "dispatch_namespaced_declarations": [],
            },
        )
        del fm

    cpu_fm.write(
        "MethodOperators.h",
        lambda: {
            "MethodOperators_includes": sorted(
                f"#include <ATen/ops/{name}_ops.h>"
                for name, functions in functions_by_root_name.items()
                if any(Variant.method in fn.variants for fn in functions)
            ),
            "MethodOperators_declarations": [],
        },
    )


def gen_headers(
    *,
    native_functions: Sequence[NativeFunction],
    valid_tags: Set[str],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    static_dispatch_idx: List[BackendIndex],
    selector: SelectiveBuilder,
    backend_indices: Dict[DispatchKey, BackendIndex],
    core_fm: FileManager,
    cpu_fm: FileManager,
    cuda_fm: FileManager,
    ops_fm: FileManager,
    dispatch_keys: Sequence[DispatchKey],
    functions_keys: Set[DispatchKey],
    rocm: bool,
    per_operator_headers: bool,
) -> None:
    if per_operator_headers:
        gen_per_operator_headers(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            cpu_fm=cpu_fm,
            cuda_fm=cuda_fm,
            ops_fm=ops_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=rocm,
        )
    else:
        gen_aggregated_headers(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            structured_native_functions=structured_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            cpu_fm=cpu_fm,
            cuda_fm=cuda_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=rocm,
        )

    core_fm.write(
        "TensorBody.h",
        lambda: {
            "tensor_method_declarations": list(
                mapMaybe(
                    ComputeTensorMethod(
                        target=Target.DECLARATION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    native_functions,
                )
            ),
            "tensor_method_definitions": list(
                mapMaybe(
                    ComputeTensorMethod(
                        target=Target.DEFINITION,
                        static_dispatch_backend_indices=static_dispatch_idx,
                    ),
                    native_functions,
                )
            ),
        },
    )

    cpu_fm.write(
        "RedispatchFunctions.h",
        lambda: {
            "function_redispatch_definitions": list(
                mapMaybe(ComputeRedispatchFunction(), native_functions)
            ),
        },
    )

    cpu_fm.write(
        "RegistrationDeclarations.h",
        lambda: {
            "registration_declarations": [
                compute_registration_declarations(f, backend_indices)
                for f in native_functions
            ],
        },
    )

    cpu_fm.write(
        "VmapGeneratedPlumbing.h", lambda: gen_all_vmap_plumbing(native_functions)
    )

    def gen_aten_interned_strings() -> Dict[str, str]:
        attrs = set()  # All function argument names
        names = set()  # All ATen function names
        for func in native_functions:
            names.add(str(func.func.name.name))
            # Some operators don't have a functional variant but we still create a
            # symbol without the underscore
            names.add(func.func.name.name.base)

            for arg in func.func.schema_order_arguments():
                attrs.add(arg.name)

        # These are keywords in C++, so aren't valid symbol names
        # https://en.cppreference.com/w/cpp/language/operator_alternative
        names -= {
            "and",
            "and_eq",
            "bitand",
            "bitor",
            "compl",
            "not",
            "not_eq",
            "or",
            "or_eq",
            "xor",
            "xor_eq",
        }

        return {
            "aten_symbols": " \\\n".join(
                [f"_(aten, {name})" for name in sorted(names)]
            ),
            "attr_symbols": " \\\n".join(
                [f"_(attr, {name})" for name in sorted(attrs)]
            ),
        }

    core_fm.write("aten_interned_strings.h", gen_aten_interned_strings)

    def gen_tags_enum() -> Dict[str, str]:
        return {"enum_of_valid_tags": (",\n".join(sorted(valid_tags)))}

    core_fm.write("enum_tag.h", gen_tags_enum)


def gen_source_files(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    view_groups: Sequence[NativeFunctionsViewGroup],
    selector: SelectiveBuilder,
    static_dispatch_idx: List[BackendIndex],
    backend_indices: Dict[DispatchKey, BackendIndex],
    core_fm: FileManager,
    cpu_fm: FileManager,
    cpu_vec_fm: FileManager,
    cuda_fm: FileManager,
    dispatch_keys: Sequence[DispatchKey],
    functions_keys: Set[DispatchKey],
    rocm: bool,
    force_schema_registration: bool,
    per_operator_headers: bool,
    skip_dispatcher_op_registration: bool,
) -> None:
    extra_cuda_headers = """\
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>"""
    if rocm:
        extra_cuda_headers = """\
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hip/HIPDevice.h>
#include <ATen/hip/HIPContext.h>"""

    for dispatch_key in dispatch_keys:
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm

        if per_operator_headers:

            def operator_headers() -> List[str]:
                headers = []
                for g in grouped_native_functions:
                    is_registered = False
                    if backend_index.has_kernel(g):
                        is_registered = True
                    # The above has_kernel test on a group will only test for
                    # the existence of out dispatch, because that's how
                    # structured kernels work. But sometimes functions can be
                    # grouped but not be structured, and then you need to check
                    # each individual piece, as they may have manual dispatch
                    # entries.
                    elif isinstance(g, NativeFunctionsGroup) and any(
                        backend_index.has_kernel(fn) for fn in g.functions()
                    ):
                        is_registered = True
                    # TODO: this condition is a bit questionable
                    # (It has to do with the fact that structured kernels get generated kernels
                    # to the Meta + CompositeExplicitAutogradNonFunctional keys).
                    elif g.structured and dispatch_key in (
                        DispatchKey.Meta,
                        DispatchKey.CompositeExplicitAutogradNonFunctional,
                    ):
                        is_registered = True
                    if not is_registered:
                        continue

                    headers.append(f"#include <ATen/ops/{g.root_name}_native.h>")
                    if (
                        dispatch_key
                        == DispatchKey.CompositeExplicitAutogradNonFunctional
                    ):
                        headers.append(f"#include <ATen/ops/{g.root_name}.h>")
                    if dispatch_key in functions_keys:
                        headers.append(
                            f"#include <ATen/ops/{g.root_name}_{dispatch_namespace}_dispatch.h>"
                        )

                return sorted(set(headers))

        else:

            def operator_headers() -> List[str]:
                headers = ["#include <ATen/NativeFunctions.h>"]
                if dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
                    headers.append("#include <ATen/Functions.h>")
                if dispatch_key in functions_keys:
                    headers.append(f"#include <ATen/{dispatch_key!s}Functions.h>")
                return headers

        backend_index = backend_indices[dispatch_key]
        ns_grouped_native_functions = defaultdict(list)
        for grouped_native_function in grouped_native_functions:
            namespace = (
                grouped_native_function.namespace
                if isinstance(grouped_native_function, NativeFunction)
                else grouped_native_function.functional.namespace
            )
            ns_grouped_native_functions[namespace].append(grouped_native_function)

        dispatch_namespace = str(dispatch_key).lower()

        # CompositeImplicitAutogradNestdTensor does not currently user the helpers generated
        # compilation will fail when `-Werror=unused-function` flag is set
        gen_dispatch_helpers: bool = (
            dispatch_key != DispatchKey.CompositeImplicitAutogradNestedTensor
        )

        dispatch_definitions = get_native_function_definitions(
            fm=fm,
            grouped_native_functions=grouped_native_functions,
            dispatch_key=dispatch_key,
            backend_idx=backend_index,
            selector=selector,
            rocm=rocm,
            symint=True,
            skip_dispatcher_op_registration=skip_dispatcher_op_registration,
            gen_dispatch_helpers=gen_dispatch_helpers,
        )
        fm.write_with_template(
            f"Register{dispatch_key}.cpp",
            "RegisterDispatchKey.cpp",
            lambda: {
                "extra_cuda_headers": extra_cuda_headers
                if is_cuda_dispatch_key(dispatch_key)
                else "",
                "external_backend_headers": "",
                "dispatch_headers": dest.gen_registration_headers(
                    backend_index, per_operator_headers, rocm
                ),
                "ops_headers": operator_headers(),
                "dispatch_helpers": "",
                "dispatch_definitions": dispatch_definitions,
            },
        )

        for g in structured_native_functions:
            if not g.out.ufunc_inner_loop or not is_ufunc_dispatch_key(dispatch_key):
                continue
            name = g.functional.func.name.name
            if dispatch_key is DispatchKey.CPU:
                assert fm is cpu_fm
                fm.write_with_template(
                    f"UfuncCPU_{name}.cpp",
                    "UfuncCPU.cpp",
                    lambda: {
                        "meta_declaration": compute_meta_function_declaration(g),
                        "native_declaration": dest.compute_native_function_declaration(
                            g, backend_indices[dispatch_key]
                        ),
                        "native_definitions": dest.compute_ufunc_cpu(g),
                    },
                )
                cpu_vec_fm.write_with_template(
                    f"UfuncCPUKernel_{name}.cpp",
                    "UfuncCPUKernel.cpp",
                    lambda: {
                        "name": name,
                        "native_definitions": dest.compute_ufunc_cpu_kernel(g),
                    },
                )
            elif dispatch_key is DispatchKey.CUDA:
                cuda_headers = "#include <ATen/native/cuda/Loops.cuh>"
                if rocm:
                    cuda_headers = "#include <ATen/native/hip/Loops.cuh>"
                fm.write_with_template(
                    f"UfuncCUDA_{name}.cu",
                    "UfuncCUDA.cu",
                    lambda: {
                        "name": name,
                        "cuda_headers": cuda_headers,
                        "meta_declaration": compute_meta_function_declaration(g),
                        "native_declaration": dest.compute_native_function_declaration(
                            g, backend_indices[dispatch_key]
                        ),
                        "native_definitions": dest.compute_ufunc_cuda(g),
                    },
                )
            else:
                raise AssertionError(f"unrecognized {dispatch_key} for ufunc")

        del fm

    # BackendSelect is generated specially
    def gen_backend_select() -> Dict[str, List[str]]:
        relevant_fns = [
            fn for fn in native_functions if needs_backend_select(fn, selector)
        ]
        return {
            "ops_headers": [
                f"#include <ATen/ops/{fn.root_name}_ops.h>" for fn in relevant_fns
            ],
            "backend_select_method_definitions": list(
                mapMaybe(
                    ComputeBackendSelect(Target.DEFINITION, selector), relevant_fns
                )
            ),
            "backend_select_function_registrations": list(
                mapMaybe(
                    ComputeBackendSelect(Target.REGISTRATION, selector), relevant_fns
                )
            ),
        }

    cpu_fm.write("RegisterBackendSelect.cpp", gen_backend_select)

    schema_selector = selector
    if force_schema_registration:
        schema_selector = SelectiveBuilder.get_nop_selector()

    (
        aten_schema_registrations,
        schema_registrations,
    ) = get_native_function_schema_registrations(
        native_functions=native_functions, schema_selector=schema_selector
    )
    cpu_fm.write(
        "RegisterSchema.cpp",
        lambda: {
            "aten_schema_registrations": []
            if skip_dispatcher_op_registration
            else aten_schema_registrations,
            "schema_registrations": []
            if skip_dispatcher_op_registration
            else schema_registrations,
        },
    )

    def key_func(
        fn: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]
    ) -> str:
        return fn.root_name

    cpu_fm.write_sharded(
        "Operators.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            "operator_headers": [f"#include <ATen/ops/{fn.root_name}.h>"],
            "definitions": [
                ComputeOperators(
                    Target.DEFINITION,
                    static_dispatch_backend_indices=static_dispatch_idx,
                )(fn)
            ],
        },
        base_env={
            "static_dispatch_extra_headers": static_dispatch_extra_headers(
                static_dispatch_idx
            ),
        },
        num_shards=5,
        sharded_keys={
            "operator_headers",
            "definitions",
            "static_dispatch_extra_headers",
        },
    )

    cpu_fm.write("Functions.cpp", lambda: {})

    core_fm.write("TensorMethods.cpp", lambda: {})

    core_fm.write(
        "ATenOpList.cpp",
        lambda: {
            "aten_ops": list(mapMaybe(compute_aten_op, native_functions)),
        },
    )

    def functionalization_env_callable(
        g: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]
    ) -> Dict[str, List[str]]:
        def gen_op_headers(
            g: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]
        ) -> List[str]:
            if isinstance(g, NativeFunctionsViewGroup):
                # view ops always get a functionalization kernel
                headers = [
                    f"#include <ATen/ops/{g.view.root_name}_native.h>",
                    f"#include <ATen/ops/{g.view.root_name}_ops.h>",
                ]
                if g.view_copy is not None:
                    headers += [
                        f"#include <ATen/ops/{g.view_copy.root_name}_native.h>",
                        f"#include <ATen/ops/{g.view_copy.root_name}_ops.h>",
                    ]
                return headers
            elif isinstance(g, NativeFunctionsGroup):
                headers = [
                    f"#include <ATen/ops/{g.functional.root_name}_native.h>",
                    f"#include <ATen/ops/{g.functional.root_name}_ops.h>",
                    f"#include <ATen/ops/{g.out.root_name}_native.h>",
                    f"#include <ATen/ops/{g.out.root_name}_ops.h>",
                ]
                if g.inplace is not None:
                    headers += [
                        f"#include <ATen/ops/{g.inplace.root_name}_native.h>",
                        f"#include <ATen/ops/{g.inplace.root_name}_ops.h>",
                    ]
                if g.mutable is not None:
                    headers += [
                        f"#include <ATen/ops/{g.mutable.root_name}_native.h>",
                        f"#include <ATen/ops/{g.mutable.root_name}_ops.h>",
                    ]
                return headers
            else:
                return [
                    f"#include <ATen/ops/{g.root_name}_native.h>",
                    f"#include <ATen/ops/{g.root_name}_ops.h>",
                ]

        return {
            "ops_headers": gen_op_headers(g),
            "func_definitions": gen_functionalization_definition(
                selector,
                g,
            ),
            "func_registrations": gen_functionalization_registration(
                selector,
                g,
                backend_indices[DispatchKey.CompositeImplicitAutograd],
            ),
        }

    all_groups: List[
        Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]
    ] = list(structured_native_functions) + list(
        view_groups  # type: ignore[assignment, arg-type, operator]
    )
    # Note: all operators that functionalization needs to handle (mutable and aliasing ops) should be grouped properly.
    # The only reason we really need to deal with direct NativeFunctions here (instead of the groups) is because:
    # (1) We can provide better error checking (error out if someone introduces a mutable op that doesn't obey the grouping logic)
    # (2) functionalization needs to manually register CompositeImplicitAutograd kernels, which might not be grouped.
    #     Although this could go away long-term if we add a dedicated dispatch key for decompositions.
    structured_map: Dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(lambda g: list(g.functions()), structured_native_functions)
    }
    view_map: Dict[OperatorName, NativeFunction] = {
        f.func.name: f for f in concatMap(lambda g: list(g.functions()), view_groups)
    }
    for f in native_functions:
        if f.func.name not in structured_map and f.func.name not in view_map:
            all_groups.append(f)

    cpu_fm.write_sharded(
        "RegisterFunctionalization.cpp",
        all_groups,
        key_fn=key_func,
        env_callable=functionalization_env_callable,
        num_shards=4,
        sharded_keys={
            "ops_headers",
            "func_definitions",
            "func_registrations",
            "func_add_back_views_definitions",
            "func_add_back_views_registrations",
        },
    )

    cpu_fm.write(
        "FunctionalInverses.h",
        lambda: {
            "view_inverse_declarations": list(
                mapMaybe(
                    lambda g: gen_functionalization_view_inverse_declaration(
                        selector, g
                    ),
                    view_groups,
                )
            )
        },
    )

    # Note [view_copy NativeFunctions]
    # Every view operator in native_functions.yaml that is not CompositeImplicitAutograd
    # needs to have a corresponding non-aliasing {view}_copy variant.
    # Backends that use functionalization and don't know how to handle aliasing ops
    # are expected to implement kernels for these {view}_copy kernels instead.
    # The code for {view}_copy operators in core is pretty boilerplate-heavy however,
    # so we codegen the following:
    # (1) A CompositeExplicitAutogradNonFunctional kernel for every {view}_copy operator.
    #     These are never explicitly invoked by the functionalization pass,
    #     but they could theoretically be called from user code (I added these kernels for completeness,
    #     since the ops are part of the public API).
    # (2) A derivative formula for every {view}_copy operator
    #     {view}_copy operators can re-use the same derivative formulas as their {view} op counterparts,
    #     so rather than stamping all of the entries out in derivatives.yaml,
    #     we codegen them in.
    #     This is similar to how autograd codegen doesn't require inplace ops to have a derivatives.yaml entry.
    cpu_fm.write(
        "CompositeViewCopyKernels.cpp",
        lambda: {
            "ops_headers": [
                "\n".join(
                    f"#include <ATen/ops/{f.root_name}_ops.h>\n"
                    # NB: this include is important as it ensures we
                    # set the visibility on generated view_copy kernels
                    # correctly
                    f"#include <ATen/ops/{f.root_name}_native.h>"
                    for f in (
                        [g.view] if g.view_copy is None else [g.view, g.view_copy]
                    )
                )
                for g in view_groups
            ]
            + [
                "\n".join(
                    f"#include <ATen/ops/{f.root_name}_ops.h>"
                    for f in [g.inplace, g.mutable, g.functional]
                    if f is not None and "generated" not in f.tags
                )
                for g in structured_native_functions
            ],
            "CompositeViewCopyKernel_Definitions": list(
                mapMaybe(
                    GenCompositeViewCopyKernel(
                        backend_indices[
                            DispatchKey.CompositeExplicitAutogradNonFunctional
                        ]
                    ),
                    view_groups,
                )
            ),
            "GeneratedCompositeFunctional_Definitions": list(
                mapMaybe(
                    gen_composite_functional_kernel,
                    structured_native_functions,
                )
            ),
            "GeneratedCompositeOut_Definitions": list(
                mapMaybe(
                    gen_composite_out_kernel,
                    structured_native_functions,
                )
            ),
        },
    )


def gen_declarations_yaml(
    cpu_fm: FileManager, native_functions: Sequence[NativeFunction]
) -> None:
    cpu_fm.write(
        "Declarations.yaml",
        lambda: format_yaml([compute_declaration_yaml(f) for f in native_functions]),
    )


def get_torchgen_root() -> pathlib.Path:
    """
    If you're depending on torchgen out-of-tree, you can use the root to figure
    out the path to native_functions.yaml
    """
    return pathlib.Path(__file__).parent.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ATen source files")
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="aten/src/ATen",
    )
    parser.add_argument(
        "-o",
        "--output-dependencies",
        help="output a list of dependencies into the given file and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
    )
    parser.add_argument(
        "--per-operator-headers",
        action="store_true",
        help="generate separate headers per operator in ATen/ops",
    )
    parser.add_argument(
        "-d",
        "--install-dir",
        "--install_dir",
        help="output directory",
        default="build/aten/src/ATen",
    )
    parser.add_argument(
        "--rocm",
        action="store_true",
        help="reinterpret CUDA as ROCm/HIP and adjust filepaths accordingly",
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Generate MPS registration code when set",
    )
    # TODO: --op-registration-whitelist will be removed when all call-sites
    # for gen.py are moved over to using the operator YAML file for mobile
    # custom build.
    parser.add_argument(
        "--op-registration-whitelist",
        "--op_registration_whitelist",
        nargs="*",
        help="filter op registrations by the whitelist (if set); "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )
    parser.add_argument(
        "--op-selection-yaml-path",
        "--op_selection_yaml_path",
        help="Provide a path to the operator selection (for custom build) YAML "
        "that contains the information about the set of selected operators "
        "and their categories (training, ...). Each operator is either a "
        "full operator name with overload or just a bare operator name. "
        "The operator names also contain the namespace prefix (e.g. aten::)",
    )
    parser.add_argument(
        "--backend-whitelist",
        "--backend_whitelist",
        nargs="*",
        help="filter dispatch backend by the whitelist (if set), "
        "e.g.: CPU CUDA QuantizedCPU ...",
    )
    parser.add_argument(
        "--static-dispatch-backend",
        "--static_dispatch_backend",
        nargs="*",
        help="generate static dispatch code for the specific backend (if set)",
    )
    parser.add_argument(
        "--skip-dispatcher-op-registration",
        "--skip_dispatcher_op_registration",
        action="store_true",
        help="Avoid registering operators into the dispatcher.",
    )
    parser.add_argument(
        "--force-schema-registration",
        "--force_schema_registration",
        action="store_true",
        help="force it to generate schema-only registrations for all ops, including"
        "those that are not listed on --op-registration-whitelist",
    )
    parser.add_argument(
        "--generate",
        type=str,
        nargs="*",
        choices=["headers", "sources", "declarations_yaml"],
        default=["headers", "sources", "declarations_yaml"],
        help="Generate only a subset of files",
    )

    options = parser.parse_args()

    selector = get_custom_build_selector(
        options.op_registration_whitelist,
        options.op_selection_yaml_path,
    )

    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(options.source_path, "native/tags.yaml")

    from torchgen.model import dispatch_keys

    # TODO: stop generating CUDA kernels for non-CUDA builds
    ignore_keys = set()
    if not options.mps:
        ignore_keys.add(DispatchKey.MPS)

        if DispatchKey.MPS in dispatch_keys:
            del dispatch_keys[dispatch_keys.index(DispatchKey.MPS)]

    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path, ignore_keys)
    valid_tags = _GLOBAL_PARSE_TAGS_YAML_CACHE[tags_yaml_path]
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )

    grouped_native_functions = get_grouped_native_functions(native_functions)

    structured_native_functions = [
        g for g in grouped_native_functions if isinstance(g, NativeFunctionsGroup)
    ]
    native_functions_with_view_groups = get_grouped_by_view_native_functions(
        native_functions
    )
    view_groups = [
        g
        for g in native_functions_with_view_groups
        if isinstance(g, NativeFunctionsViewGroup)
    ]

    # NB: It is mandatory to NOT use os.path.join here, as the install directory
    # will eventually be ingested by cmake, which does not respect Windows style
    # path slashes.  If you switch this to use os.path.join, you'll get an error
    # like:
    #
    #   Syntax error in cmake code when parsing string
    #
    #     C:/Jenkins/workspace/pytorch-builds/pytorch-win-ws2016-cuda9-cudnn7-py3-build/build/aten/src/ATen\core/TensorMethods.h
    #
    #   Invalid character escape '\c'.
    core_install_dir = f"{options.install_dir}/core"
    pathlib.Path(core_install_dir).mkdir(parents=True, exist_ok=True)
    ops_install_dir = f"{options.install_dir}/ops"
    pathlib.Path(ops_install_dir).mkdir(parents=True, exist_ok=True)

    core_fm = make_file_manager(options=options, install_dir=core_install_dir)
    cpu_fm = make_file_manager(options=options)
    cpu_vec_fm = make_file_manager(options=options)
    cuda_fm = make_file_manager(options=options)
    ops_fm = make_file_manager(options=options, install_dir=ops_install_dir)

    # Only a limited set of dispatch keys get CPUFunctions.h headers generated
    # for them; this is the set
    functions_keys = {
        DispatchKey.CPU,
        DispatchKey.CUDA,
        DispatchKey.CompositeImplicitAutograd,
        DispatchKey.CompositeImplicitAutogradNestedTensor,
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.CompositeExplicitAutogradNonFunctional,
        DispatchKey.Meta,
    }
    if options.mps:
        functions_keys.add(DispatchKey.MPS)

    if options.backend_whitelist:
        dispatch_keys = [
            k
            for k in dispatch_keys
            if is_generic_dispatch_key(k) or str(k) in options.backend_whitelist
        ]

    static_dispatch_idx: List[BackendIndex] = []
    if options.static_dispatch_backend:
        static_dispatch_idx = [
            backend_indices[DispatchKey.parse(key)]
            for key in options.static_dispatch_backend
        ]
        for key in options.static_dispatch_backend:
            dp_key = DispatchKey.parse(key)
            if dp_key not in functions_keys:
                functions_keys.add(dp_key)

    if "sources" in options.generate:
        gen_source_files(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            structured_native_functions=structured_native_functions,
            view_groups=view_groups,
            selector=selector,
            static_dispatch_idx=static_dispatch_idx,
            backend_indices=backend_indices,
            core_fm=core_fm,
            cpu_fm=cpu_fm,
            cpu_vec_fm=cpu_vec_fm,
            cuda_fm=cuda_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=options.rocm,
            force_schema_registration=options.force_schema_registration,
            per_operator_headers=options.per_operator_headers,
            skip_dispatcher_op_registration=options.skip_dispatcher_op_registration,
        )

    if "headers" in options.generate:
        gen_headers(
            native_functions=native_functions,
            valid_tags=valid_tags,
            grouped_native_functions=grouped_native_functions,
            structured_native_functions=structured_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            core_fm=core_fm,
            cpu_fm=cpu_fm,
            cuda_fm=cuda_fm,
            ops_fm=ops_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=options.rocm,
            per_operator_headers=options.per_operator_headers,
        )

    if "declarations_yaml" in options.generate:
        gen_declarations_yaml(native_functions=native_functions, cpu_fm=cpu_fm)

    if options.output_dependencies:
        depfile_path = pathlib.Path(options.output_dependencies).resolve()
        depfile_name = depfile_path.name
        depfile_stem = depfile_path.stem

        for fm, prefix in [
            (cpu_fm, ""),
            (cpu_vec_fm, "cpu_vec_"),
            (core_fm, "core_"),
            (cuda_fm, "cuda_"),
            (ops_fm, "ops_"),
        ]:
            varname = prefix + depfile_stem
            path = depfile_path.parent / (prefix + depfile_name)
            fm.write_outputs(varname, str(path))


if __name__ == "__main__":
    main()
