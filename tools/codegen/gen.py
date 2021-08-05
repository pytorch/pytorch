import os
from typing import List, Dict, Optional, Tuple, Set, Callable, Any, Union, Sequence
from typing_extensions import Literal
import yaml
from collections import OrderedDict, defaultdict, namedtuple
import argparse
import pathlib
import functools
import json
from dataclasses import dataclass

from tools.codegen.code_template import CodeTemplate
from tools.codegen.model import (Argument, DispatchKey, FunctionSchema,
                                 Location, NativeFunction,
                                 NativeFunctionsGroup, OperatorName,
                                 BackendIndex, BackendMetadata,
                                 OptionalType, SchemaKind, SelfArgument,
                                 TensorOptionsArguments, Type, Variant,
                                 assert_never, is_cuda_dispatch_key,
                                 is_generic_dispatch_key)
from tools.codegen.api.types import (Binding, CppSignature, CppSignatureGroup,
                                     DispatcherSignature, NativeSignature)
from tools.codegen.api import cpp
import tools.codegen.api.dispatcher as dispatcher
import tools.codegen.api.native as native
import tools.codegen.api.meta as meta
import tools.codegen.api.structured as structured
from tools.codegen.api.translate import translate
from tools.codegen.selective_build.selector import SelectiveBuilder
from tools.codegen.utils import Target, concatMap, context, mapMaybe, YamlDumper, YamlLoader
from tools.codegen.context import (method_with_native_function,
                                   native_function_manager,
                                   with_native_function_and_indices,
                                   with_native_function)
import tools.codegen.dest as dest

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
            dispatch_key=DispatchKey.Undefined, use_out_as_primary=True, external=False, index={}))
        for k, v in bs.items():
            # All structured in-tree operators are implemented in terms of their out operator.
            indices[k] = BackendIndex(dispatch_key=k, use_out_as_primary=True, external=False, index=v)
        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = ParsedYaml(rs, indices)

    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]

# Some assertions are already performed during parsing, but those are only within a single NativeFunction.
# Assertions here are meant to be performed across NativeFunctions.
def error_check_native_functions(funcs: Sequence[NativeFunction]) -> None:
    func_map: Dict[OperatorName, NativeFunction] = {}
    for f in funcs:
        func_map[f.func.name] = f
    for f in funcs:
        if f.structured_delegate is not None:
            delegate_func = func_map[f.structured_delegate]
            assert delegate_func.structured, \
                f"{f.func.name} is marked as a structured_delegate pointing to " \
                f"{f.structured_delegate}, but {f.structured_delegate} is not marked as structured. " \
                f"Consider adding 'structured=True' to the delegated operator"

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

def static_dispatch_extra_headers(backend: Optional[BackendIndex], skip_tensor_include: bool = False) -> str:
    if skip_tensor_include:
        # See Note [Avoiding Include Cycles In Static Dispatch]
        maybe_inl = '_inl'
    else:
        maybe_inl = ''
    return '\n'.join([
        f'#include <ATen/{dispatch_key}Functions{maybe_inl}.h>' for dispatch_key in static_dispatch_keys(backend)])

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

    if f.structured_delegate is not None:
        # TODO: for ops with structured_delegate it should check the dispatch table of
        # the out variant instead. For now, these structured ops all have CPU/CUDA kernels
        # so we always dispatch to the `backend`, but this could be wrong when we
        # migrate math/default_backend ops to use structured delegate.
        return f'return at::{backend_index.dispatch_key.lower()}::{name}({exprs_str});'

    if backend_index.has_kernel(f):
        return f'return at::{backend_index.dispatch_key.lower()}::{name}({exprs_str});'
    elif f.has_composite_explicit_autograd_kernel:
        return f'return at::{DispatchKey.CompositeExplicitAutograd.lower()}::{name}({exprs_str});'
    elif f.has_composite_implicit_autograd_kernel:
        return f'return at::{DispatchKey.CompositeImplicitAutograd.lower()}::{name}({exprs_str});'

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
    def __call__(self, f: NativeFunction) -> Optional[str]:
        sig = DispatcherSignature.from_schema(f.func)
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
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::{str(f.func.name.name)}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "{f.func.name.overload_name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, {cpp_string(str(f.func))})
  static {sig.defn(name=call_method_name, is_redispatching_fn=False)};
  static {sig.defn(name=redispatch_method_name, is_redispatching_fn=True)};
}};"""
        elif self.target is Target.DEFINITION:
            defns = f"""
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, name, "aten::{str(f.func.name)}")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, overload_name, "{f.func.name.overload_name}")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, schema_str, {cpp_string(str(f.func))})"""

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
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{sig.type()}>();
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
            target_sig = DispatcherSignature.from_schema(f.func)
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

            target_sig = DispatcherSignature.from_schema(f.func)
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

            target_sig = DispatcherSignature.from_schema(f.func)
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
        return f"""\
struct TORCH_API structured_{name} : public {parent_class} {{
    void meta({args_str});
}};
"""

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
        if str(f.func.name.name).endswith('_like') or str(f.func.name.name).startswith('new_'):
            return None

        name = native.name(f.func)
        native_sig = NativeSignature(f.func)

        if not any(isinstance(a.argument, TensorOptionsArguments) for a in native_sig.arguments()):
            return None

        if not self.selector.is_native_function_selected(f):
            return None

        native_tensor_args = [
            a for a in native_sig.arguments()
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
  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
    .typed<{dispatcher_sig.type()}>();
  {compute_dk}
  return op.redispatch(_dk, {', '.join(a.expr for a in dispatcher_exprs)});
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
def dynamic_type(t: Type) -> str:
    if isinstance(t, OptionalType):
        return dynamic_type(t.elem)
    # Note we don't use t.is_tensor_like() here because it would
    # also include Tensor[]
    if str(t) == 'Tensor':
        return 'at::Tensor'
    return cpp.argumenttype_type(t, mutable=False, binds='__placeholder__').cpp_type()

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
            'dynamic_type': dynamic_type(r.type),
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
                              out_arg_set: Set[str], name_to_field_name: Dict[str, str]) -> object:
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
            kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name)

def compute_argument_yaml(a: Argument, *, schema_order: bool, kwarg_only_set: Set[str],
                          out_arg_set: Set[str], name_to_field_name: Dict[str, str]) -> object:
    arg: Dict[str, object] = {
        'annotation': str(a.annotation) if a.annotation else None,
        'dynamic_type': dynamic_type(a.type),
        'is_nullable': a.type.is_nullable(),
        'name': a.name,
        'type': cpp.argument_type(a, binds="__placeholder__").cpp_type(),
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
            kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name)
        for cpp_a in cpp_args
    ]

    schema_order_jit_arguments = list(f.func.schema_order_arguments())

    schema_order_arguments = [
        compute_argument_yaml(
            a, schema_order=True,
            kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name)
        for a in schema_order_jit_arguments
    ]

    cpp_schema_order_types = [
        # NB: method here doesn't matter
        r.type for a in schema_order_jit_arguments
        for r in cpp.argument(
            a, method=False, cpp_no_default_args=set(), faithful=False, has_tensor_options=False)
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
    args = dispatcher.arguments(f.func)
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

@functools.lru_cache(maxsize=None)
def _read_template(template_fn: str) -> CodeTemplate:
    return CodeTemplate.from_file(template_fn)

# A small abstraction for writing out generated files and keeping track
# of what files have been written (so you can write out a list of output
# files)
class FileManager:
    install_dir: str
    template_dir: str
    dry_run: bool
    filenames: Set[str]

    def __init__(self, install_dir: str, template_dir: str, dry_run: bool) -> None:
        self.install_dir = install_dir
        self.template_dir = template_dir
        self.filenames = set()
        self.dry_run = dry_run

    def _write_if_changed(self, filename: str, contents: str) -> None:
        old_contents: Optional[str]
        try:
            with open(filename, 'r') as f:
                old_contents = f.read()
        except IOError:
            old_contents = None
        if contents != old_contents:
            with open(filename, 'w') as f:
                f.write(contents)

    def write_with_template(self, filename: str, template_fn: str,
                            env_callable: Callable[[], Union[str, Dict[str, object]]]) -> None:
        filename = '{}/{}'.format(self.install_dir, filename)
        assert filename not in self.filenames, "duplicate file write {filename}"
        self.filenames.add(filename)
        if not self.dry_run:
            env = env_callable()
            if isinstance(env, dict):
                # TODO: Update the comment reference to the correct location
                if 'generated_comment' not in env:
                    comment = "@" + "generated by tools/codegen/gen.py"
                    comment += " from {}".format(os.path.basename(template_fn))
                    env['generated_comment'] = comment
                template = _read_template(os.path.join(self.template_dir, template_fn))
                self._write_if_changed(filename, template.substitute(env))
            elif isinstance(env, str):
                self._write_if_changed(filename, env)
            else:
                assert_never(env)


    def write(self, filename: str, env_callable: Callable[[], Union[str, Union[str, Dict[str, object]]]]) -> None:
        self.write_with_template(filename, filename, env_callable)

    def write_outputs(self, filename: str) -> None:
        """Write a file containing the list of all outputs which are
        generated by this script."""
        self._write_if_changed(
            filename,
            ''.join(name + ";" for name in sorted(self.filenames)))

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

def get_grouped_native_functions(
        native_functions: Sequence[NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
    pre_grouped_native_functions: Dict[FunctionSchema, Dict[SchemaKind, NativeFunction]] = defaultdict(dict)
    for f in native_functions:
        d = pre_grouped_native_functions[f.func.signature()]
        assert f.func.kind() not in d
        d[f.func.kind()] = f

    def flatten_pre_group(d: Dict[SchemaKind, NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
        r = NativeFunctionsGroup.from_dict(d)
        if r is None:
            return list(d.values())
        else:
            return [r]

    # TODO: how come ValuesView isn't a Sequence lol
    return list(concatMap(flatten_pre_group, list(pre_grouped_native_functions.values())))

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
    options = parser.parse_args()

    selector = get_custom_build_selector(
        options.op_registration_whitelist,
        options.op_selection_yaml_path,
    )

    native_yaml_path = os.path.join(options.source_path, 'native/native_functions.yaml')
    parsed_yaml = parse_native_yaml(native_yaml_path)
    native_functions, backend_indices = parsed_yaml.native_functions, parsed_yaml.backend_indices
    grouped_native_functions = get_grouped_native_functions(native_functions)
    structured_native_functions = [g for g in grouped_native_functions if isinstance(g, NativeFunctionsGroup)]

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

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=options.output_dependencies)

    core_fm = make_file_manager(core_install_dir)
    cpu_fm = make_file_manager(options.install_dir)
    cuda_fm = make_file_manager(options.install_dir)

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

    for dispatch_key in dispatch_keys:
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm

        fm.write_with_template(f'Register{dispatch_key}.cpp', 'RegisterDispatchKey.cpp', lambda: {
            'extra_cuda_headers': extra_cuda_headers if is_cuda_dispatch_key(dispatch_key) else '',
            'legacy_th_headers':
                '#include <ATen/LegacyTHFunctionsCUDA.h>' if dispatch_key == DispatchKey.CUDA else
                '',
            'external_backend_headers': '',
            'namespaced_headers': f'#include <ATen/{dispatch_key}Functions.h>' if dispatch_key in functions_keys else '',
            'DispatchKey': dispatch_key,
            'dispatch_namespace': dispatch_key.lower(),
            'dispatch_namespaced_definitions': list(concatMap(
                dest.RegisterDispatchKey(
                    backend_indices[dispatch_key],
                    Target.NAMESPACED_DEFINITION,
                    selector,
                    rocm=options.rocm,
                    cpp_namespace='at::native',
                    class_method_name=None),
                grouped_native_functions
            )),
            'dispatch_anonymous_definitions': list(concatMap(
                dest.RegisterDispatchKey(
                    backend_indices[dispatch_key],
                    Target.ANONYMOUS_DEFINITION,
                    selector,
                    rocm=options.rocm,
                    cpp_namespace='at::native',
                    class_method_name=None),
                grouped_native_functions
            )),
            'dispatch_registrations': list(concatMap(
                dest.RegisterDispatchKey(
                    backend_indices[dispatch_key],
                    Target.REGISTRATION,
                    selector,
                    rocm=options.rocm,
                    cpp_namespace='at::native',
                    class_method_name=None),
                grouped_native_functions
            )),
        })

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
                'dispatch_namespace': dispatch_key.lower(),
                'dispatch_namespaced_declarations': list(concatMap(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.NAMESPACED_DECLARATION,
                        selector,
                        rocm=options.rocm,
                        cpp_namespace='at::native',
                        class_method_name=None),
                    grouped_native_functions
                )),
            })

        del fm

    # BackendSelect is generated specially
    cpu_fm.write('RegisterBackendSelect.cpp', lambda: {
        'backend_select_method_definitions':
            list(mapMaybe(ComputeBackendSelect(Target.DEFINITION, selector), native_functions)),
        'backend_select_function_registrations':
            list(mapMaybe(ComputeBackendSelect(Target.REGISTRATION, selector), native_functions)),
    })

    cpu_fm.write('NativeMetaFunctions.h', lambda: {
        'declarations': list(mapMaybe(compute_meta_function_declaration, structured_native_functions)),
    })

    schema_selector = selector
    if options.force_schema_registration:
        schema_selector = SelectiveBuilder.get_nop_selector()
    cpu_fm.write('RegisterSchema.cpp', lambda: {
        'schema_registrations': list(mapMaybe(RegisterSchema(schema_selector), native_functions)),
    })

    cpu_fm.write('Operators.cpp', lambda: {
        'definitions': list(mapMaybe(ComputeOperators(
            Target.DEFINITION), native_functions)),
    })
    cpu_fm.write('Operators.h', lambda: {
        'declarations': list(mapMaybe(ComputeOperators(
            Target.DECLARATION), native_functions)),
    })

    cpu_fm.write('Functions.h', lambda: {
        'static_dispatch_extra_headers': static_dispatch_extra_headers(static_dispatch_idx),
        'function_definitions': list(mapMaybe(ComputeFunction(
            static_dispatch_backend_index=static_dispatch_idx), native_functions)),
    })

    cpu_fm.write('Functions.cpp', lambda: {})

    core_fm.write('TensorBody.h', lambda: {
        'static_dispatch_extra_headers': static_dispatch_extra_headers(static_dispatch_idx, skip_tensor_include=True),
        'tensor_method_declarations': list(mapMaybe(ComputeTensorMethod(
            target=Target.DECLARATION, static_dispatch_backend_index=static_dispatch_idx), native_functions)),
        'tensor_method_definitions': list(mapMaybe(ComputeTensorMethod(
            target=Target.DEFINITION, static_dispatch_backend_index=static_dispatch_idx), native_functions)),
    })

    core_fm.write('TensorMethods.cpp', lambda: {})

    cpu_fm.write('RedispatchFunctions.h', lambda: {
        'function_redispatch_definitions': list(mapMaybe(ComputeRedispatchFunction(), native_functions)),
    })

    core_fm.write('ATenOpList.cpp', lambda: {
        'aten_ops': list(mapMaybe(compute_aten_op, native_functions)),
    })

    cpu_fm.write('NativeFunctions.h', lambda: {
        'native_function_declarations': list(concatMap(
            # Convert to a set first to remove duplicate kernel names.
            # Backends are allowed to repeat kernel names; only generate the declaration once!
            lambda f: list(OrderedDict.fromkeys(concatMap(
                lambda backend_idx:
                    dest.compute_native_function_declaration(f, backend_idx),
                backend_indices.values()))),
            grouped_native_functions)),
    })

    cpu_fm.write('Declarations.yaml', lambda: format_yaml([compute_declaration_yaml(f) for f in native_functions]))
    cpu_fm.write('RegistrationDeclarations.h', lambda: {
        'registration_declarations': [compute_registration_declarations(f, backend_indices) for f in native_functions],
    })

    if options.output_dependencies:
        cpu_fm.write_outputs(options.output_dependencies)
        core_fm.write_outputs(f"{options.output_dependencies}-core")
        cuda_fm.write_outputs(f"{options.output_dependencies}-cuda")

if __name__ == '__main__':
    main()
