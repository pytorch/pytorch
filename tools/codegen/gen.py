import os
from typing import List, Dict, Optional, Tuple, Set, Any, Union, Sequence, TypeVar
from typing_extensions import Literal
import yaml
from collections import OrderedDict, defaultdict, namedtuple
import argparse
import pathlib
import json
from dataclasses import dataclass

from tools.codegen.model import (Argument, DispatchKey, FunctionSchema,
                                 Location, NativeFunction,
                                 NativeFunctionsGroup, OperatorName,
                                 BackendIndex, BackendMetadata,
                                 OptionalType, SchemaKind, SelfArgument,
                                 TensorOptionsArguments, Type, Variant,
                                 is_cuda_dispatch_key,
                                 is_generic_dispatch_key,
                                 Tag, BaseOperatorName)
from tools.codegen.api.types import (Binding, CppSignature, CppSignatureGroup,
                                     DispatcherSignature, NativeSignature)
from tools.codegen.api import cpp
import tools.codegen.api.dispatcher as dispatcher
import tools.codegen.api.native as native
import tools.codegen.api.meta as meta
import tools.codegen.api.structured as structured
from tools.codegen.api.translate import translate
from tools.codegen.selective_build.selector import SelectiveBuilder
from tools.codegen.utils import (
    Target, concatMap, context, mapMaybe, YamlDumper, YamlLoader, FileManager, assert_never
)
from tools.codegen.context import (method_with_native_function,
                                   native_function_manager,
                                   with_native_function_and_indices,
                                   with_native_function)
import tools.codegen.dest as dest
from tools.codegen.gen_functionalization_type import (
    needs_functionalization,
    gen_functionalization_definition,
    gen_functionalization_registration,
    gen_functionalization_view_inverse_declaration
)

T = TypeVar('T')

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
#     the dispatcher API, and the legacy disaptcher API.  See each
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
        mapping['__line__'] = node.start_mark.line + 1
        return mapping

_GLOBAL_PARSE_NATIVE_YAML_CACHE = {}

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
ParsedYaml = namedtuple('ParsedYaml', ['native_functions', 'backend_indices'])
def parse_native_yaml(path: str) -> ParsedYaml:
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        with open(path, 'r') as f:
            es = yaml.load(f, Loader=LineLoader)
        assert isinstance(es, list)
        rs: List[NativeFunction] = []
        bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
        for e in es:
            assert isinstance(e.get('__line__'), int), e
            loc = Location(path, e['__line__'])
            funcs = e.get('func')
            with context(lambda: f'in {loc}:\n  {funcs}'):
                func, m = NativeFunction.from_yaml(e, loc)
                rs.append(func)
                BackendIndex.grow_index(bs, m)
        error_check_native_functions(rs)
        # Default dict is to prevent the codegen from barfing when we have a dispatch key that has no kernels yet.
        indices: Dict[DispatchKey, BackendIndex] = defaultdict(lambda: BackendIndex(
            dispatch_key=DispatchKey.Undefined,
            use_out_as_primary=True,
            external=False,
            device_guard=False,
            index={}))
        for k, v in bs.items():
            # All structured in-tree operators are implemented in terms of their out operator.
            indices[k] = BackendIndex(
                dispatch_key=k,
                use_out_as_primary=True,
                external=False,
                # Only cuda-like devices in tree require device guards
                device_guard=is_cuda_dispatch_key(k),
                index=v)
        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = ParsedYaml(rs, indices)

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
            assert delegate_func.structured, \
                f"{f.func.name} is marked as a structured_delegate pointing to " \
                f"{f.structured_delegate}, but {f.structured_delegate} is not marked as structured. " \
                f"Consider adding 'structured=True' to the delegated operator"
        if f.tag is not None and f.tag is Tag.inplace_view:
            base_name = f.func.name.name
            overload_name = f.func.name.overload_name
            assert base_name.inplace, \
                f"{f.func.name} is marked with tag: inplace_view, but it doesn't follow the naming " \
                "convention for inplace ops - the codegen expects the base name to have a trailing underscore. "
            out_of_place_base_name = BaseOperatorName(base_name.base, False, base_name.dunder_method)
            assert len(base_func_map[out_of_place_base_name]) > 0, \
                f"{f.func.name} is marked with tag: inplace_view. The codegen expects there to be a corresponding " \
                f"out-of-place view op with the name '{base_name}' and matching schema, but it didn't find one. "


def cpp_string(s: str) -> str:
    """Convert a python string into a c++ string literal """
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\a', '\\a')
    s = s.replace('\b', '\\b')
    s = s.replace('\f', '\\f')
    s = s.replace('\n', '\\n')
    s = s.replace('\v', '\\v')
    s = s.replace('\t', '\\t')
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

def static_dispatch_keys(backend: Optional[BackendIndex]) -> List[DispatchKey]:
    if backend is None:
        return []
    else:
        return [
            backend.dispatch_key,
            DispatchKey.CompositeImplicitAutograd,
            DispatchKey.CompositeExplicitAutograd
        ]

def get_static_dispatch_backend(f: NativeFunction, backend_index: BackendIndex) -> Optional[DispatchKey]:
    if (f.structured_delegate is not None or backend_index.has_kernel(f)):
        # TODO: for ops with structured_delegate it should check the dispatch table of
        # the out variant instead. For now, these structured ops all have CPU/CUDA kernels
        # so we always dispatch to the `backend`, but this could be wrong when we
        # migrate math/default_backend ops to use structured delegate.
        return backend_index.dispatch_key
    elif f.has_composite_explicit_autograd_kernel:
        return DispatchKey.CompositeExplicitAutograd
    elif f.has_composite_implicit_autograd_kernel:
        return DispatchKey.CompositeImplicitAutograd
    return None


def static_dispatch_ops_header(
        f: NativeFunction,
        backend_index: Optional[BackendIndex]) -> Optional[str]:
    if backend_index is None or f.manual_kernel_registration:
        return None

    dispatch_key = get_static_dispatch_backend(f, backend_index)
    return (f'#include <ATen/ops/{f.root_name}_{dispatch_key.lower()}_dispatch.h>'
            if dispatch_key is not None else None)


def static_dispatch_extra_headers(backend: Optional[BackendIndex], skip_tensor_include: bool = False) -> List[str]:
    if skip_tensor_include:
        # See Note [Avoiding Include Cycles In Static Dispatch]
        maybe_inl = '_inl'
    else:
        maybe_inl = ''
    return [f'#include <ATen/{dispatch_key}Functions{maybe_inl}.h>'
            for dispatch_key in static_dispatch_keys(backend)]


def static_dispatch(
    f: NativeFunction, cpp_sig: CppSignature,
    *, method: bool, backend_index: Optional[BackendIndex]
) -> Optional[str]:
    if backend_index is None or f.manual_kernel_registration:
        return None
    target_sig = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False).signature
    name = target_sig.name()
    exprs = translate(cpp_sig.arguments(), target_sig.arguments(), method=method)
    exprs_str = ', '.join(a.expr for a in exprs)

    dispatch_key = get_static_dispatch_backend(f, backend_index)
    if dispatch_key is not None:
        return f'return at::{dispatch_key.lower()}::{name}({exprs_str});'

    return f'TORCH_CHECK(false, "Static dispatch does not support {name} for {backend_index.dispatch_key}.");'

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
        return f'm.def({cpp_string(str(f.func))});\n'

# Generates Operators.h and Operators.cpp.
# These provide macros that, given an operator and overload name, allow users
# to access an "un-overloaded" function version of the operator. This
# is useful for extension writers who want to (1) want to decltype the operator
# and (2) don't want to worry about method-only operators.
@dataclass(frozen=True)
class ComputeOperators:
    target: Union[
        Literal[Target.DECLARATION],
        Literal[Target.DEFINITION]
    ]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        sig = DispatcherSignature.from_schema(f.func, structured_type_override=f.part_of_structured_group)
        name = f.func.name.unambiguous_name()
        call_method_name = 'call'
        redispatch_method_name = 'redispatch'

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
  static {sig.defn(name=call_method_name, is_redispatching_fn=False)};
  static {sig.defn(name=redispatch_method_name, is_redispatching_fn=True)};
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
                    dispatcher_exprs_str = ', '.join(['dispatchKeySet'] + [a.name for a in sig.arguments()])
                    dispatcher_call = 'redispatch'
                    method_name = f'{name}::{redispatch_method_name}'
                else:
                    dispatcher_exprs_str = ', '.join([a.name for a in sig.arguments()])
                    dispatcher_call = 'call'
                    method_name = f'{name}::{call_method_name}'

                defns += f"""
// aten::{f.func}
{sig.defn(name=method_name, is_redispatching_fn=is_redispatching_fn)} {{
    static auto op = create_{name}_typed_handle();
    return op.{dispatcher_call}({dispatcher_exprs_str});
}}
"""
            return defns
        else:
            assert_never(self.target)


# Generates Function.h, which provides the functional public C++ API,
# and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeFunction:
    static_dispatch_backend_index: Optional[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if Variant.function not in f.variants:
            return None

        sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)

        def generate_defn(faithful: bool) -> str:
            if faithful:
                sig = sig_group.faithful_signature
                assert sig is not None
            else:
                sig = sig_group.signature

            # See Note [The ATen Operators API]
            target_sig = DispatcherSignature.from_schema(f.func, structured_type_override=f.part_of_structured_group)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ', '.join([e.expr for e in exprs])

            static_dispatch_block = static_dispatch(f, sig, method=False, backend_index=self.static_dispatch_backend_index)
            if static_dispatch_block is None:
                return f"""
// aten::{f.func}
TORCH_API inline {sig.decl()} {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}
"""
            else:
                return f"""
// aten::{f.func}
TORCH_API inline {sig.decl()} {{
    {static_dispatch_block}
}}
"""
        result = generate_defn(False)
        if sig_group.faithful_signature is not None:
            result += generate_defn(True)

        return result

# Generates TensorBody.h. This file provides the object-oriented (method-based)
# public C++ API, and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeTensorMethod:
    target: Union[
        Literal[Target.DECLARATION],
        Literal[Target.DEFINITION]
    ]
    static_dispatch_backend_index: Optional[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if Variant.method not in f.variants:
            return None

        assert not f.func.is_out_fn()
        assert f.func.arguments.self_arg is not None

        sig_group = CppSignatureGroup.from_native_function(f, method=True, fallback_binding=f.manual_cpp_binding)

        if self.target is Target.DECLARATION:
            result = f"{sig_group.signature.decl()} const;\n"
            if sig_group.faithful_signature is not None:
                result += f"{sig_group.faithful_signature.decl()} const;\n"
            return result

        if self.target is not Target.DEFINITION:
            assert_never(self.target)

        def generate_defn(faithful: bool) -> str:
            if faithful:
                sig = sig_group.faithful_signature
                assert sig is not None
            else:
                sig = sig_group.signature

            target_sig = DispatcherSignature.from_schema(f.func, structured_type_override=f.part_of_structured_group)
            exprs = translate(sig.arguments(), target_sig.arguments(), method=True)
            exprs_str = ', '.join([e.expr for e in exprs])

            static_dispatch_block = static_dispatch(f, sig, method=True, backend_index=self.static_dispatch_backend_index)
            if static_dispatch_block is None:
                return f"""
// aten::{f.func}
inline {sig.defn(prefix="Tensor::")} const {{
    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});
}}
"""
            else:
                return f"""
// aten::{f.func}
inline {sig.defn(prefix="Tensor::")} const {{
    {static_dispatch_block}
}}
"""

        result = generate_defn(faithful=False)
        if sig_group.faithful_signature is not None:
            result += generate_defn(faithful=True)

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
        sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)

        def generate_defn(faithful: bool) -> str:
            if faithful:
                sig = sig_group.faithful_signature
                assert sig is not None
            else:
                sig = sig_group.signature

            target_sig = DispatcherSignature.from_schema(f.func, structured_type_override=f.part_of_structured_group)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ', '.join(['dispatchKeySet'] + [a.expr for a in exprs])

            return f"""
// aten::{f.func}
TORCH_API inline {sig.decl(is_redispatching_fn=True)} {{
    return at::_ops::{f.func.name.unambiguous_name()}::redispatch({exprs_str});
}}
"""
        result = generate_defn(False)
        if sig_group.faithful_signature is not None:
            result += generate_defn(True)

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
        args_str = ', '.join(a.decl() for a in args)
        parent_class = g.out.structured_inherits
        if parent_class is None:
            parent_class = "at::impl::MetaBase"
        meta_return = "void"
        precomputed = g.out.precomputed if g.structured else None

        if precomputed:
            # Generate the template declaration with one bool parameter for each
            # precomputed element. Each parameter is true if the corresponding (in
            # terms of position) precomputed element has been set.
            precomputed_elements = [elem for replace_list in precomputed.replace.values() for elem in replace_list]
            precomputed_template_parameters = [elem.name.upper() for elem in precomputed_elements]
            precomputed_template_params_str = ", ".join(f"bool {param} = false" for param in precomputed_template_parameters)
            precompute_template_decl = f"template <{precomputed_template_params_str}>"

            # Generate a string containing declarations of all precomputed elements.
            precomputed_elements_with_cpp_types = [
                structured.argument_type(elem, binds=elem.name)
                for elem in precomputed_elements
            ]

            precomputed_elements_decl = ";\n".join(
                f"{elem.cpp_type(strip_ref=True)} {elem.name}" for elem in precomputed_elements_with_cpp_types
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
                    precomputed_template_parameters[:i] + ["true"] + precomputed_template_parameters[i + 1:]
                )
                return_ty = f"precompute_out<{return_ty_templates}>"
                elem_cpp_ty = precomputed_elements_with_cpp_types[i].cpp_type(strip_ref=True)
                signature = f"{return_ty} set_{elem.name}({elem_cpp_ty} value)"

                # Generate an assert which checks that the
                # template parameter corresponding to the precomputed
                # element that is set by this method is false on the
                # class corresponding to the object that `this` points to.
                # This ensures that each element can be set only once.
                assert_msg = f"\"{precomputed_elements[i].name} already set\""
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
                        construction_stmts.append(f"ret.{elem.name} = this->{elem.name};")

                construction_stmts.append("return ret;")
                construction_block = "\n".join(construction_stmts)

                setter_methods.append(f"""
                    {signature} {{
                        {assert_stmt}
                        {construction_block}
                    }}
                """)
            setter_methods_decl = "\n".join(setter_methods)

            # Meta should return an instance of the struct containing the precomputed elements.
            meta_return_template_params = ", ".join(["true"] * len(precomputed_template_parameters))
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
    if name.endswith('_like') or name.startswith('new_'):
        return False
    if f.func.arguments.tensor_options is None:
        return False
    return selector.is_native_function_selected(f)


# Generates RegisterBackendSelect.cpp, a series of kernels which provide
# specialized computation of dispatch key for operator signatures which cannot
# be easily done automatically using templating.
@dataclass(frozen=True)
class ComputeBackendSelect:
    target: Union[
        Literal[Target.DEFINITION],
        Literal[Target.REGISTRATION]
    ]

    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if not needs_backend_select(f, self.selector):
            return None

        name = native.name(f.func)
        native_sig = NativeSignature(f.func, structured_type_override=f.part_of_structured_group)

        native_tensor_args = [
            a for a in native_sig.arguments()
            if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()
        ]

        dispatcher_sig = DispatcherSignature.from_schema(f.func, structured_type_override=f.part_of_structured_group)

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
                tensor_args = ', '.join(a.name for a in native_tensor_args)
                compute_dk = f"""\
DispatchKeySet _dk_set = c10::DispatchKeySet({dispatch_key}) | c10::detail::multi_dispatch_key_set({tensor_args});
  DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
  DispatchKeySet _dk = c10::impl::computeDispatchKeySet(_dk_set, _dk_mask);"""
            else:
                compute_dk = f"DispatchKeySet _dk = c10::DispatchKeySet({dispatch_key});"
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
    return yaml.dump(data, default_flow_style=False, Dumper=YamlDumper, width=1e9)  # type: ignore[no-any-return]

# For some reason, some defaults we write to YAML are written as native
# YAML objects, rather than doing them uniformly as strings.  This
# function detects those cases and converts them into native Python
# objects.
def pythonify_default(s: str) -> object:
    if s == 'true':
        return True
    elif s == 'false':
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
def dynamic_type(t: Type, structured_type_override: bool) -> str:
    if isinstance(t, OptionalType):
        return dynamic_type(t.elem, structured_type_override=structured_type_override)
    # Note we don't use t.is_tensor_like() here because it would
    # also include Tensor[]
    if str(t) == 'Tensor':
        return 'at::Tensor'
    return cpp.argumenttype_type(
        t, mutable=False, binds='__placeholder__',
        structured_type_override=structured_type_override).cpp_type()

def compute_method_of_yaml(variants: Set[Variant]) -> List[str]:
    # This is written out explicitly to ensure that Tensor and
    # namespace are put into the list in the right order
    method_of = ['Type']
    if Variant.method in variants:
        method_of.append('Tensor')
    if Variant.function in variants:
        method_of.append('namespace')
    return method_of

def compute_returns_yaml(f: NativeFunction) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
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
            'dynamic_type': dynamic_type(r.type, structured_type_override=False),
            'name': name,
            'type': cpp.return_type(r).cpp_type(),
        }

        if r.name:
            # See Note [name and field_name]
            ret['field_name'] = r.name
            if f.func.is_out_fn():
                name_to_field_name[f.func.arguments.out[i].name] = r.name

        returns.append(ret)

    return returns, name_to_field_name

# arguments in yaml roughly corresponds to the public C++ API
def compute_cpp_argument_yaml(cpp_a: Binding, *, schema_order: bool, kwarg_only_set: Set[str],
                              out_arg_set: Set[str], name_to_field_name: Dict[str, str],
                              structured_type_override: bool) -> object:
    if isinstance(cpp_a.argument, TensorOptionsArguments):
        arg: Dict[str, object] = {
            'annotation': None,
            'dynamic_type': 'at::TensorOptions',
            'is_nullable': False,
            'name': cpp_a.name,
            'type': cpp_a.type,
            'kwarg_only': True,
        }
        if cpp_a.default is not None:
            arg['default'] = cpp_a.default
        return arg
    elif isinstance(cpp_a.argument, SelfArgument):
        raise AssertionError()
    elif isinstance(cpp_a.argument, Argument):
        return compute_argument_yaml(
            cpp_a.argument, schema_order=schema_order,
            kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name,
            structured_type_override=structured_type_override)

def compute_argument_yaml(a: Argument, *, schema_order: bool, kwarg_only_set: Set[str],
                          out_arg_set: Set[str], name_to_field_name: Dict[str, str],
                          structured_type_override: bool) -> object:
    arg: Dict[str, object] = {
        'annotation': str(a.annotation) if a.annotation else None,
        'dynamic_type': dynamic_type(a.type, structured_type_override=structured_type_override),
        'is_nullable': a.type.is_nullable(),
        'name': a.name,
        'type': cpp.argument_type(a, binds="__placeholder__", structured_type_override=structured_type_override).cpp_type(),
    }
    if a.default is not None:
        arg['default'] = pythonify_default(cpp.default_expr(a.default, a.type))
    if a.name in kwarg_only_set:
        arg['kwarg_only'] = True
    if a.name in out_arg_set:
        arg['output'] = True
        arg['allocate'] = True
        # See Note [name and field_name]
        if a.name in name_to_field_name:
            arg['field_name'] = name_to_field_name[a.name]
    # Historically, booleans don't get their size recorded, because it
    # is already built into the cpp type (e.g., std::array<bool, 4>)
    l = a.type.is_list_like()
    if l is not None and l.size is not None and str(l.elem) != 'bool':
        arg['size'] = l.size
    return arg

@with_native_function
def compute_declaration_yaml(f: NativeFunction) -> object:
    returns, name_to_field_name = compute_returns_yaml(f)

    # These sets are used to conveniently test if an argument is a
    # kwarg-only or out argument
    kwarg_only_set = set(a.name for a in f.func.arguments.flat_kwarg_only)
    out_arg_set = set(a.name for a in f.func.arguments.out)

    sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
    cpp_args = sig_group.signature.arguments()
    arguments = [
        compute_cpp_argument_yaml(
            cpp_a, schema_order=False,
            kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name,
            structured_type_override=f.part_of_structured_group)
        for cpp_a in cpp_args
    ]

    schema_order_jit_arguments = list(f.func.schema_order_arguments())

    schema_order_arguments = [
        compute_argument_yaml(
            a, schema_order=True,
            kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name,
            structured_type_override=f.part_of_structured_group)
        for a in schema_order_jit_arguments
    ]

    cpp_schema_order_types = [
        # NB: method here doesn't matter
        r.type for a in schema_order_jit_arguments
        for r in cpp.argument(
                a, method=False, cpp_no_default_args=set(), faithful=False,
                has_tensor_options=False, structured_type_override=f.part_of_structured_group)
    ]

    cpp_returns = cpp.returns_type(f.func.returns).cpp_type()
    schema_order_cpp_signature = f"{cpp_returns} ({', '.join(cpp_schema_order_types)})"

    is_factory_method = any(isinstance(a.argument, TensorOptionsArguments) for a in cpp_args) \
        and Variant.method not in f.variants

    return OrderedDict([
        ('name', cpp.name(f.func)),
        ('operator_name', str(f.func.name.name)),
        ('overload_name', str(f.func.name.overload_name)),
        ('manual_kernel_registration', f.manual_kernel_registration),
        ('category_override', f.category_override if f.category_override is not None else ''),
        ('schema_string', f'aten::{f.func}'),
        ('arguments', arguments),
        ('schema_order_cpp_signature', schema_order_cpp_signature),
        ('schema_order_arguments', schema_order_arguments),
        ('method_of', compute_method_of_yaml(f.variants)),
        ('mode', 'native'),
        ('python_module', '' if f.python_module is None else f.python_module),
        ('returns', returns),
        ('inplace', f.func.name.name.inplace),
        ('is_factory_method', is_factory_method),
        ('abstract', f.is_abstract),
        ('device_guard', f.device_guard),
        ('with_gil', False),
        ('deprecated', False),
        ('has_math_kernel', f.has_composite_implicit_autograd_kernel),
    ])

# See Note [Auto generated composite kernels]
def has_autogenerated_composite_kernel(f: NativeFunction) -> bool:
    return (f.structured or f.structured_delegate is not None) and \
           (f.func.kind() == SchemaKind.functional or f.func.kind() == SchemaKind.inplace)

@with_native_function_and_indices
def compute_registration_declarations(f: NativeFunction, backend_indices: Dict[DispatchKey, BackendIndex]) -> str:
    name = dispatcher.name(f.func)
    returns_type = dispatcher.returns_type(f.func.returns).cpp_type_registration_declarations()
    args = dispatcher.arguments(f.func, structured_type_override=f.part_of_structured_group)
    args_str = ', '.join(a.no_default().decl_registration_declarations() for a in args)
    comment_data : Dict[str, str] = {
        'schema': f'aten::{f.func}',
        # TODO: What exactly is the semantics of the 'dispatch' field?
        'dispatch': str({k for k, v in backend_indices.items() if v.has_kernel(f)} != {DispatchKey.CompositeImplicitAutograd}),
        'default': str(f.has_composite_kernel or has_autogenerated_composite_kernel(f))
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
        op_selection_yaml_path: Optional[str]) -> SelectiveBuilder:
    assert not (
        provided_op_registration_allowlist is not None and
        op_selection_yaml_path is not None), (
            "Both provided_op_registration_allowlist and " +
            "op_selection_yaml_path can NOT be provided at the " +
            "same time.")

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

def pre_group_native_functions(
        native_functions: Sequence[NativeFunction]) -> Dict[FunctionSchema, Dict[SchemaKind, NativeFunction]]:
    pre_grouped_native_functions: Dict[FunctionSchema, Dict[SchemaKind, NativeFunction]] = defaultdict(dict)
    for f in native_functions:
        d = pre_grouped_native_functions[f.func.signature()]
        assert f.func.kind() not in d
        d[f.func.kind()] = f
    return pre_grouped_native_functions

def get_grouped_native_functions(
        native_functions: Sequence[NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
    def flatten_pre_group(d: Dict[SchemaKind, NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
        r = NativeFunctionsGroup.from_dict(d)
        if r is None:
            return list(d.values())
        else:
            return [r]

    # TODO: how come ValuesView isn't a Sequence lol
    pre_grouped_native_functions = pre_group_native_functions(native_functions)
    return list(concatMap(flatten_pre_group, list(pre_grouped_native_functions.values())))

def gen_aggregated_headers(
        *,
        native_functions: Sequence[NativeFunction],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        static_dispatch_idx: Optional[BackendIndex],
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
    structured_native_functions = [g for g in grouped_native_functions
                                   if isinstance(g, NativeFunctionsGroup)]
    cpu_fm.write('NativeMetaFunctions.h', lambda: {
        'NativeMetaFunctions_includes': [],
        'NativeMetaFunctions_declarations': list(
            mapMaybe(compute_meta_function_declaration, structured_native_functions)),
    })
    method_native_functions = [fn for fn in native_functions
                               if Variant.method in fn.variants]
    non_method_native_functions = [fn for fn in native_functions
                                   if fn not in method_native_functions]
    cpu_fm.write('MethodOperators.h', lambda: {
        'MethodOperators_includes': [],
        'MethodOperators_declarations': list(mapMaybe(ComputeOperators(
            Target.DECLARATION), method_native_functions)),
    })
    cpu_fm.write('Operators.h', lambda: {
        'Operators_includes': ['#include <ATen/MethodOperators.h>'],
        'Operators_declarations': list(mapMaybe(ComputeOperators(
            Target.DECLARATION), non_method_native_functions)),
    })
    cpu_fm.write('Functions.h', lambda: {
        'static_dispatch_extra_headers': static_dispatch_extra_headers(static_dispatch_idx),
        'Functions_includes': ['#include <ATen/Operators.h>'],
        'Functions_declarations': list(mapMaybe(ComputeFunction(
            static_dispatch_backend_index=static_dispatch_idx), native_functions)),
    })
    cpu_fm.write('NativeFunctions.h', lambda: {
        'NativeFunctions_includes': ['#include <ATen/NativeMetaFunctions.h>'],
        'NativeFunctions_declarations': list(concatMap(
            # Convert to a set first to remove duplicate kernel names.
            # Backends are allowed to repeat kernel names; only generate the declaration once!
            lambda f: list(OrderedDict.fromkeys(concatMap(
                lambda backend_idx:
                    dest.compute_native_function_declaration(f, backend_idx),
                backend_indices.values()))),
            grouped_native_functions)),
    })

    for dispatch_key in dispatch_keys:
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm
        if dispatch_key in functions_keys:
            if dispatch_key in static_dispatch_keys(static_dispatch_idx):
                # See Note [Avoiding Include Cycles In Static Dispatch]
                inl_headers = ''
            else:
                inl_headers = f'#include <ATen/{dispatch_key}Functions_inl.h>'

            fm.write_with_template(f'{dispatch_key}Functions.h', 'DispatchKeyFunctions.h', lambda: {
                'dispatch_key': str(dispatch_key),
                'inline_headers_for_nonstatic_build': inl_headers,
            })
            fm.write_with_template(f'{dispatch_key}Functions_inl.h', 'DispatchKeyFunctions_inl.h', lambda: {
                'DispatchKeyFunctions_inl_includes': [],
                'dispatch_namespace': dispatch_key.lower(),
                'dispatch_namespaced_declarations': list(concatMap(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.NAMESPACED_DECLARATION,
                        selector,
                        rocm=rocm,
                        cpp_namespace='at::native',
                        class_method_name=None),
                    grouped_native_functions
                )),
            })

        del fm

def gen_per_operator_headers(
        *,
        native_functions: Sequence[NativeFunction],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        static_dispatch_idx: Optional[BackendIndex],
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

    grouped_functions_by_root_name: Dict[str, List[Union[NativeFunction, NativeFunctionsGroup]]] = defaultdict(lambda: [])
    for group in grouped_native_functions:
        name = group.root_name
        grouped_functions_by_root_name[name].append(group)

    for name, functions in functions_by_root_name.items():
        ops_fm.write_with_template(
            f'{name}_ops.h', 'Operator.h', lambda: {
                'declarations': list(mapMaybe(ComputeOperators(
                    Target.DECLARATION), functions)),
            })

        ops_fm.write_with_template(
            f'{name}.h', 'Function.h', lambda: {
                'static_dispatch_ops_headers': list(mapMaybe(
                    lambda fn: static_dispatch_ops_header(fn, backend_index=static_dispatch_idx),
                    functions)),
                'operator_includes': f'#include <ATen/ops/{name}_ops.h>',
                'function_definitions': list(mapMaybe(ComputeFunction(
                    static_dispatch_backend_index=static_dispatch_idx), functions)),
            })

        grouped_functions = grouped_functions_by_root_name.get(name, [])
        structured_functions = [fn for fn in grouped_functions
                                if isinstance(fn, NativeFunctionsGroup) and fn.structured]
        is_structured = len(structured_functions) > 0


        if is_structured:
            ops_fm.write_with_template(
                f'{name}_meta.h', 'NativeMetaFunction.h', lambda: {
                    'meta_function_declarations': list(mapMaybe(
                        compute_meta_function_declaration, structured_functions)),
                })


        ops_fm.write_with_template(
            f'{name}_native.h', 'NativeFunction.h', lambda: {
                'extra_includes': (f'#include <ATen/ops/{name}_meta.h>'
                                   if is_structured else []),
                'native_function_declarations': list(concatMap(
                    # Convert to a set first to remove duplicate kernel names.
                    # Backends are allowed to repeat kernel names; only generate the declaration once!
                    lambda f: list(OrderedDict.fromkeys(concatMap(
                        lambda backend_idx:
                            dest.compute_native_function_declaration(f, backend_idx),
                        backend_indices.values()))),
                    grouped_functions)),
            })

    for category, suffix in [
            ('Functions', ''),
            ('Operators', '_ops'),
            ('NativeMetaFunctions', '_meta'),
            ('NativeFunctions', '_native'),
    ]:
        cpu_fm.write(f'{category}.h', lambda: {
            'static_dispatch_extra_headers': [],
            f'{category}_includes': [
                f'#include <ATen/ops/{name}{suffix}.h>'
                for name in sorted(functions_by_root_name.keys())
            ],
            f'{category}_declarations': [],
        })

    for dispatch_key in dispatch_keys:
        if dispatch_key not in functions_keys:
            continue

        dispatch_namespace = dispatch_key.lower()
        dispatch_names = []

        for name, functions in functions_by_root_name.items():
            grouped_functions = grouped_functions_by_root_name.get(name, [])
            declarations = list(concatMap(
                dest.RegisterDispatchKey(
                    backend_indices[dispatch_key],
                    Target.NAMESPACED_DECLARATION,
                    selector,
                    rocm=rocm,
                    cpp_namespace='at::native',
                    class_method_name=None),
                grouped_functions
            ))

            if len(declarations) == 0:
                continue

            dispatch_names.append(name)
            ops_fm.write_with_template(
                f'{name}_{dispatch_namespace}_dispatch.h',
                'DispatchKeyFunction.h', lambda: {
                    'dispatch_namespace': dispatch_namespace,
                    'dispatch_namespaced_declarations': declarations,
                })

        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm
        if dispatch_key in static_dispatch_keys(static_dispatch_idx):
            # See Note [Avoiding Include Cycles In Static Dispatch]
            inl_headers = ''
        else:
            inl_headers = f'#include <ATen/{dispatch_key}Functions_inl.h>'

        fm.write_with_template(f'{dispatch_key}Functions.h', 'DispatchKeyFunctions.h', lambda: {
            'dispatch_key': str(dispatch_key),
            'inline_headers_for_nonstatic_build': inl_headers,
        })
        fm.write_with_template(f'{dispatch_key}Functions_inl.h', 'DispatchKeyFunctions_inl.h', lambda: {
            'dispatch_namespace': dispatch_namespace,
            'DispatchKeyFunctions_inl_includes': [
                f'#include <ATen/ops/{name}_{dispatch_namespace}_dispatch.h>'
                for name in sorted(dispatch_names)
            ],
            'dispatch_namespaced_declarations': [],
        })
        del fm

    cpu_fm.write('MethodOperators.h', lambda: {
        'MethodOperators_includes': sorted(
            f'#include <ATen/ops/{name}_ops.h>'
            for name, functions in functions_by_root_name.items()
            if any(Variant.method in fn.variants for fn in functions)
        ),
        'MethodOperators_declarations': [],
    })

def gen_headers(
        *,
        native_functions: Sequence[NativeFunction],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        static_dispatch_idx: Optional[BackendIndex],
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
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            cpu_fm=cpu_fm,
            cuda_fm=cuda_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=rocm,
        )

    def static_dispatch_method_headers() -> List[str]:
        return list(mapMaybe(
            lambda fn: static_dispatch_ops_header(fn, backend_index=static_dispatch_idx),
            [fn for fn in native_functions if Variant.method in fn.variants]))


    core_fm.write('TensorBody.h', lambda: {
        'static_dispatch_ops_headers': (
            static_dispatch_method_headers() if per_operator_headers
            else static_dispatch_extra_headers(static_dispatch_idx, skip_tensor_include=True)),
        'tensor_method_declarations': list(mapMaybe(ComputeTensorMethod(
            target=Target.DECLARATION, static_dispatch_backend_index=static_dispatch_idx), native_functions)),
        'tensor_method_definitions': list(mapMaybe(ComputeTensorMethod(
            target=Target.DEFINITION, static_dispatch_backend_index=static_dispatch_idx), native_functions)),
    })

    cpu_fm.write('RedispatchFunctions.h', lambda: {
        'function_redispatch_definitions': list(mapMaybe(ComputeRedispatchFunction(), native_functions)),
    })

    cpu_fm.write('RegistrationDeclarations.h', lambda: {
        'registration_declarations': [compute_registration_declarations(f, backend_indices) for f in native_functions],
    })

    cpu_fm.write('FunctionalInverses.h', lambda: {
        'view_inverse_declarations': list(mapMaybe(gen_functionalization_view_inverse_declaration, native_functions))
    })


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
        names -= set(['and', 'and_eq', 'bitand', 'bitor', 'compl', 'not',
                      'not_eq', 'or', 'or_eq', 'xor', 'xor_eq'])

        return {
            'aten_symbols': ' \\\n'.join([
                f"_(aten, {name})" for name in sorted(names)
            ]),
            'attr_symbols': ' \\\n'.join([
                f"_(attr, {name})" for name in sorted(attrs)
            ]),
        }

    core_fm.write('aten_interned_strings.h', gen_aten_interned_strings)

def gen_source_files(
        *,
        native_functions: Sequence[NativeFunction],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        static_dispatch_idx: Optional[BackendIndex],
        selector: SelectiveBuilder,
        backend_indices: Dict[DispatchKey, BackendIndex],
        core_fm: FileManager,
        cpu_fm: FileManager,
        cuda_fm: FileManager,
        dispatch_keys: Sequence[DispatchKey],
        functions_keys: Set[DispatchKey],
        rocm: bool,
        force_schema_registration: bool,
        per_operator_headers: bool,
) -> None:
    extra_cuda_headers = '''\
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>'''
    if rocm:
        extra_cuda_headers = '''\
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hip/HIPDevice.h>
#include <ATen/hip/HIPContext.h>'''

    for dispatch_key in dispatch_keys:
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm

        if per_operator_headers:
            def operator_headers() -> List[str]:
                headers = []
                for fn in native_functions:
                    is_registered = backend_index.has_kernel(fn) or (
                        fn.structured and dispatch_key in
                        (DispatchKey.Meta, DispatchKey.CompositeExplicitAutograd))
                    if not is_registered:
                        continue

                    headers.append(f"#include <ATen/ops/{fn.root_name}_native.h>")
                    if dispatch_key == DispatchKey.CompositeExplicitAutograd:
                        headers.append(f"#include <ATen/ops/{fn.root_name}.h>")
                    if dispatch_key in functions_keys:
                        headers.append(
                            f"#include <ATen/ops/{fn.root_name}_{dispatch_namespace}_dispatch.h>")

                return sorted(set(headers))
        else:
            def operator_headers() -> List[str]:
                headers = ["#include <ATen/NativeFunctions.h>"]
                if dispatch_key == DispatchKey.CompositeExplicitAutograd:
                    headers.append("#include <ATen/Functions.h>")
                if dispatch_key in functions_keys:
                    headers.append(f"#include <ATen/{dispatch_key!s}Functions.h>")
                return headers

        backend_index = backend_indices[dispatch_key]
        dispatch_namespace = str(dispatch_key).lower()
        fm.write_with_template(f'Register{dispatch_key}.cpp', 'RegisterDispatchKey.cpp', lambda: {
            'extra_cuda_headers': extra_cuda_headers if is_cuda_dispatch_key(dispatch_key) else '',
            'external_backend_headers': '',
            'dispatch_headers': dest.gen_registration_headers(backend_index, per_operator_headers),
            'ops_headers': operator_headers(),
            'DispatchKey': dispatch_key,
            'dispatch_namespace': dispatch_key.lower(),
            'dispatch_helpers': dest.gen_registration_helpers(backend_index),
            'dispatch_namespaced_definitions': list(concatMap(
                dest.RegisterDispatchKey(
                    backend_index,
                    Target.NAMESPACED_DEFINITION,
                    selector,
                    rocm=rocm,
                    cpp_namespace='at::native',
                    class_method_name=None),
                grouped_native_functions
            )),
            'dispatch_anonymous_definitions': list(concatMap(
                dest.RegisterDispatchKey(
                    backend_index,
                    Target.ANONYMOUS_DEFINITION,
                    selector,
                    rocm=rocm,
                    cpp_namespace='at::native',
                    class_method_name=None),
                grouped_native_functions
            )),
            'dispatch_registrations': list(concatMap(
                dest.RegisterDispatchKey(
                    backend_index,
                    Target.REGISTRATION,
                    selector,
                    rocm=rocm,
                    cpp_namespace='at::native',
                    class_method_name=None),
                grouped_native_functions
            )),
        })

    # BackendSelect is generated specially
    def gen_backend_select() -> Dict[str, List[str]]:
        relevant_fns = [fn for fn in native_functions if needs_backend_select(fn, selector)]
        return {
            'ops_headers': [f'#include <ATen/ops/{fn.root_name}_ops.h>' for fn in relevant_fns],
            'backend_select_method_definitions':
                list(mapMaybe(ComputeBackendSelect(Target.DEFINITION, selector), relevant_fns)),
            'backend_select_function_registrations':
                list(mapMaybe(ComputeBackendSelect(Target.REGISTRATION, selector), relevant_fns)),
        }
    cpu_fm.write('RegisterBackendSelect.cpp', gen_backend_select)

    schema_selector = selector
    if force_schema_registration:
        schema_selector = SelectiveBuilder.get_nop_selector()
    cpu_fm.write('RegisterSchema.cpp', lambda: {
        'schema_registrations': list(mapMaybe(RegisterSchema(schema_selector), native_functions)),
    })

    def key_func(fn: Union[NativeFunction, NativeFunctionsGroup]) -> str:
        return fn.root_name

    cpu_fm.write_sharded(
        'Operators.cpp',
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            'operator_headers': [f'#include <ATen/ops/{fn.root_name}.h>'],
            'definitions': [ComputeOperators(Target.DEFINITION)(fn)]},
        num_shards=5,
        sharded_keys={'operator_headers', 'definitions'}
    )

    cpu_fm.write('Functions.cpp', lambda: {})

    core_fm.write('TensorMethods.cpp', lambda: {})

    core_fm.write('ATenOpList.cpp', lambda: {
        'aten_ops': list(mapMaybe(compute_aten_op, native_functions)),
    })

    # We need to easily map from [inplace_op_name] -> [functional_op] for the functionalization pass,
    # so here I generate a mapping from every operator name to its corresponding functional NativeFunction (if it exist).
    pre_grouped_d: Dict[FunctionSchema, Dict[SchemaKind, NativeFunction]] = pre_group_native_functions(native_functions)
    to_functional_op: Dict[OperatorName, Optional[NativeFunction]] = {
        k: v for d in [
            {f.func.name: pre_grouped_d[func][SchemaKind.functional]
                if SchemaKind.functional in pre_grouped_d[func].keys() else None
                for f in pre_grouped_d[func].values()}
            for func in pre_grouped_d.keys()]
        for k, v in d.items()
    }


    def functionalization_env_callable(
            g: Union[NativeFunction, NativeFunctionsGroup]
    ) -> Dict[str, List[str]]:
        functions = [g] if isinstance(g, NativeFunction) else list(g.functions())
        functions_needing_functionalization = [
            fn for fn in functions if needs_functionalization(selector, fn)]
        return {
            'ops_headers': ([
                f"#include <ATen/ops/{functions[0].root_name}_native.h>",
                f"#include <ATen/ops/{functions[0].root_name}_ops.h>",
            ] if functions_needing_functionalization else []),
            'func_definitions': list(mapMaybe(
                lambda f: gen_functionalization_definition(selector, f, to_functional_op[f.func.name]),
                functions_needing_functionalization)),
            'func_registrations': list(mapMaybe(
                lambda f: gen_functionalization_registration(
                    selector, f, backend_indices[DispatchKey.CompositeImplicitAutograd]),
                functions_needing_functionalization)),
        }


    cpu_fm.write_sharded(
        'RegisterFunctionalization.cpp',
        grouped_native_functions,
        key_fn=key_func,
        env_callable=functionalization_env_callable,
        num_shards=4,
        sharded_keys={'ops_headers', 'func_definitions', 'func_registrations'}
    )


def gen_declarations_yaml(
        cpu_fm: FileManager,
        native_functions: Sequence[NativeFunction]) -> None:
    cpu_fm.write('Declarations.yaml', lambda:
                 format_yaml([compute_declaration_yaml(f) for f in native_functions]))

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate ATen source files')
    parser.add_argument(
        '-s',
        '--source-path',
        help='path to source directory for ATen',
        default='aten/src/ATen')
    parser.add_argument(
        '-o',
        '--output-dependencies',
        help='output a list of dependencies into the given file and exit')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='run without writing any files (still updates outputs)')
    parser.add_argument(
        '--per-operator-headers', action='store_true',
        help='generate separate headers per operator in ATen/ops')
    parser.add_argument(
        '-d', '--install_dir', help='output directory',
        default='build/aten/src/ATen')
    parser.add_argument(
        '--rocm',
        action='store_true',
        help='reinterpret CUDA as ROCm/HIP and adjust filepaths accordingly')
    # TODO: --op_registration_whitelist will be removed when all call-sites
    # for gen.py are moved over to using the operator YAML file for mobile
    # custom build.
    parser.add_argument(
        '--op_registration_whitelist',
        nargs='*',
        help='filter op registrations by the whitelist (if set); '
             'each item is `namespace`::`operator name` without overload name; '
             'e.g.: aten::empty aten::conv2d ...')
    parser.add_argument(
        '--op_selection_yaml_path',
        help='Provide a path to the operator selection (for custom build) YAML '
             'that contains the information about the set of selected operators '
             'and their categories (training, ...). Each operator is either a '
             'full operator name with overload or just a bare operator name. '
             'The operator names also contain the namespace prefix (e.g. aten::)')
    parser.add_argument(
        '--backend_whitelist',
        nargs='*',
        help='filter dispatch backend by the whitelist (if set), '
             'e.g.: CPU CUDA QuantizedCPU ...')
    parser.add_argument(
        '--static_dispatch_backend',
        help='generate static dispatch code for the specific backend (if set)')
    parser.add_argument(
        '--force_schema_registration',
        action='store_true',
        help='force it to generate schema-only registrations for all ops, including'
             'those that are not listed on --op_registration_whitelist')
    parser.add_argument(
        '--generate',
        type=str,
        nargs='*',
        choices=['headers', 'sources', 'declarations_yaml'],
        default=['headers', 'sources', 'declarations_yaml'],
        help='Generate only a subset of files')
    options = parser.parse_args()

    selector = get_custom_build_selector(
        options.op_registration_whitelist,
        options.op_selection_yaml_path,
    )

    native_yaml_path = os.path.join(options.source_path, 'native/native_functions.yaml')
    parsed_yaml = parse_native_yaml(native_yaml_path)
    native_functions, backend_indices = parsed_yaml.native_functions, parsed_yaml.backend_indices
    grouped_native_functions = get_grouped_native_functions(native_functions)

    template_dir = os.path.join(options.source_path, "templates")

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
    core_install_dir = f'{options.install_dir}/core'
    pathlib.Path(core_install_dir).mkdir(parents=True, exist_ok=True)
    ops_install_dir = f'{options.install_dir}/ops'
    pathlib.Path(ops_install_dir).mkdir(parents=True, exist_ok=True)

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=options.dry_run)

    core_fm = make_file_manager(core_install_dir)
    cpu_fm = make_file_manager(options.install_dir)
    cuda_fm = make_file_manager(options.install_dir)
    ops_fm = make_file_manager(ops_install_dir)

    extra_cuda_headers = '''\
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>'''
    if options.rocm:
        extra_cuda_headers = '''\
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hip/HIPDevice.h>
#include <ATen/hip/HIPContext.h>'''

    dispatch_keys = [
        DispatchKey.CPU,
        DispatchKey.SparseCPU,
        DispatchKey.SparseCsrCPU,
        DispatchKey.MkldnnCPU,
        DispatchKey.CUDA,
        DispatchKey.SparseCUDA,
        DispatchKey.SparseCsrCUDA,
        DispatchKey.QuantizedCPU,
        DispatchKey.QuantizedCUDA,
        DispatchKey.CompositeImplicitAutograd,
        DispatchKey.CompositeExplicitAutograd,
        # Meta is a magic key: it is automatically generated for structured
        # kernels
        DispatchKey.Meta,
        DispatchKey.ZeroTensor,
    ]
    # Only a limited set of dispatch keys get CPUFunctions.h headers generated
    # for them; this is the set
    functions_keys = {
        DispatchKey.CPU,
        DispatchKey.CUDA,
        DispatchKey.CompositeImplicitAutograd,
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.Meta,
    }
    if options.backend_whitelist:
        dispatch_keys = [k for k in dispatch_keys if is_generic_dispatch_key(k) or str(k) in options.backend_whitelist]

    static_dispatch_idx: Optional[BackendIndex] = None
    if options.static_dispatch_backend:
        static_dispatch_idx = backend_indices[DispatchKey.parse(options.static_dispatch_backend)]

    if 'sources' in options.generate:
        gen_source_files(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
            static_dispatch_idx=static_dispatch_idx,
            selector=selector,
            backend_indices=backend_indices,
            core_fm=core_fm,
            cpu_fm=cpu_fm,
            cuda_fm=cuda_fm,
            dispatch_keys=dispatch_keys,
            functions_keys=functions_keys,
            rocm=options.rocm,
            force_schema_registration=options.force_schema_registration,
            per_operator_headers=options.per_operator_headers,
        )

    if 'headers' in options.generate:
        gen_headers(
            native_functions=native_functions,
            grouped_native_functions=grouped_native_functions,
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

    if 'declarations_yaml' in options.generate:
        gen_declarations_yaml(
            native_functions=native_functions,
            cpu_fm=cpu_fm)

    if options.output_dependencies:
        depfile_path = pathlib.Path(options.output_dependencies).resolve()
        depfile_name = depfile_path.name
        depfile_stem = depfile_path.stem

        for fm, prefix in [
                (cpu_fm, ""),
                (core_fm, "core_"),
                (cuda_fm, "cuda_"),
                (ops_fm, "ops_"),
        ]:
            varname = prefix + depfile_stem
            path = depfile_path.parent / (prefix + depfile_name)
            fm.write_outputs(varname, str(path))


if __name__ == '__main__':
    main()
