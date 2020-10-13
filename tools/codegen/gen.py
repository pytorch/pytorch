import os
import contextlib
import textwrap
import itertools
from typing import List, Dict, Optional, Iterator, Tuple, Set, Callable, Any, TypeVar, Union, Sequence
import yaml
from enum import Enum
from collections import OrderedDict
import argparse
import pathlib
import functools
import json

from tools.codegen.code_template import CodeTemplate
from tools.codegen.model import *
from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp
from tools.codegen.api.cpp import CppSignature
import tools.codegen.api.dispatcher as dispatcher
import tools.codegen.api.legacy_dispatcher as legacy_dispatcher
import tools.codegen.local as local

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore

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

# Conveniently add error context to exceptions raised.  Lets us
# easily say that an error occurred while processing a specific
# context.
@contextlib.contextmanager
def context(msg: str) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        # TODO: this does the wrong thing with KeyError
        msg = textwrap.indent(msg, '  ')
        msg = f'{e.args[0]}\n{msg}' if e.args else msg
        e.args = (msg,) + e.args[1:]
        raise

# A custom loader for YAML to let us also keep track of line numbers
# of each entry in the YAML file
class LineLoader(Loader):
    def construct_mapping(self, node, deep=False):  # type: ignore
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore
        # Add 1 so line numbering starts at 1
        mapping['__line__'] = node.start_mark.line + 1
        return mapping

# Parse native_functions.yaml into a sequence of NativeFunctions
def parse_native_yaml(path: str) -> List[NativeFunction]:
    with open(path, 'r') as f:
        es = yaml.load(f, Loader=LineLoader)
    assert isinstance(es, list)
    rs: List[NativeFunction] = []
    for e in es:
        assert isinstance(e.get('__line__'), int), e
        loc = Location(path, e['__line__'])
        funcs = e.get('func')
        with context(f'in {loc}:\n  {funcs}'):
            rs.append(NativeFunction.from_yaml(e, loc))
    return rs

T = TypeVar('T')
S = TypeVar('S')

# Given a function that operates on NativeFunction, wrap it into a new function
# that sets some appropriate context managers for that native function.
# YOU MUST WRAP FUNCTIONS IN THIS for calls to api modules to be sound
# (you will get an error if we try to access the local variables without having
# set them).
def with_native_function(func: Callable[[NativeFunction], T]) -> Callable[[NativeFunction], T]:
    @functools.wraps(func)
    def wrapper(f: NativeFunction) -> T:
        with context(f'in {f.loc}:\n  {f.func}'):
            with local.parametrize(
                use_c10_dispatcher=f.use_c10_dispatcher,
            ):
                return func(f)
    return wrapper

# These two functions purposely return generators in analogy to map()
# so that you don't mix up when you need to list() them

# Map over function that may return None; omit Nones from output sequence
def mapMaybe(func: Callable[[T], Optional[S]], xs: Sequence[T]) -> Iterator[S]:
    for x in xs:
        r = func(x)
        if r is not None:
            yield r

# Map over function that returns sequences and cat them all together
def concatMap(func: Callable[[T], Sequence[S]], xs: Sequence[T]) -> Iterator[S]:
    for x in xs:
        for r in func(x):
            yield r

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

# Many of these functions share logic for defining both the definition
# and declaration (for example, the function signature is the same), so
# we organize them into one function that takes a Target to say which
# code we want.
Target = Enum('Target', ('DEFINITION', 'DECLARATION', 'REGISTRATION'))

# Generates {dispatch}Type.cpp and {dispatch}Type.h (e.g., CPUType.cpp
# and CPUType.h).  This function is also reused to implement per-operator
# registration.  It also generates TypeDefault.cpp and TypeDefault.h when
# dispatch is None.
#
# {dispatch}Type.cpp
#   - The primary function of this file is to register all of the
#     implementations for the given dispatch key to the dispatcher,
#     so they are available for use in PyTorch.  If dispatch is
#     None, we generate schema (def) registrations and catchall
#     registrations.
#   - The secondary function of this file is to generate a wrapper
#     around functions.  In CPUType these wrappers do nothing
#     (and should be removed), but in other cases they handle
#     DeviceGuard. A small extra benefit of wrappers is they
#     are not overloaded, so they can be used in the registration
#     API without having to disambiguate which overload you want
#     (as would be the case if you directly registered native::
#     functions).
#
# {dispatch}Type.h
#   - In principle, this file shouldn't exist at all; historically,
#     it existed so that we could directly access these functions
#     outside of the registration API for the implementation of
#     static dispatch.  Should be deleted now!
#
# This function is also used for a secondary purpose: the registration
# logic is also reused to implement per-operator registration.
def compute_type_method(
    dispatch: Optional[str], *,
    target: Target,
    # Which operators to actually generate code for.  If None, generate
    # code for all operators
    op_registration_whitelist: Optional[Set[str]],
    # Only valid for generating registrations.  If True, only generate
    # def() invocations (for schema registration); do not generate
    # any impl() invocations for, e.g., catch-all kernels
    def_only: bool = False
) -> Callable[[NativeFunction], Optional[str]]:

    if def_only:
        assert target is Target.REGISTRATION and dispatch is None

    @with_native_function
    def func(f: NativeFunction) -> Optional[str]:
        if dispatch is not None:
            if f.dispatch is None or dispatch not in f.dispatch:
                return None
        else:
            if f.dispatch is not None and target is not Target.REGISTRATION:
                return None

        if op_registration_whitelist is not None and \
                f"aten::{f.func.name.name}" not in op_registration_whitelist and target is Target.REGISTRATION:
            return None

        name = legacy_dispatcher.name(f.func)
        returns_type = legacy_dispatcher.returns_type(f.func.returns)
        args = legacy_dispatcher.arguments(f.func)
        args_str = ', '.join(map(str, args))

        if target is Target.DECLARATION:
            return f"{returns_type} {name}({args_str});"
        elif target is Target.DEFINITION:
            if f.dispatch is None:
                cpp_name = cpp.name(f.func)
                impl_name = f"at::native::{cpp_name}"
            else:
                assert dispatch is not None
                impl_name = f"at::native::{f.dispatch[dispatch]}"

            args_exprs_str = ', '.join(map(lambda a: a.name, args))

            return_kw = "    return "

            cuda_guard = ""
            if dispatch is None or 'CUDA' in dispatch or 'Vulkan' == dispatch:
                self_args = (a for a in f.func.arguments if a.name == "self")

                # There is precedence for which argument we use to do
                # device guard.  This describes the precedence order.
                candidate_args = itertools.chain(self_args, f.func.out_arguments, f.func.arguments)

                # Only tensor like arguments are eligible
                device_of = next((f'{a.name}' for a in candidate_args if a.type.is_tensor_like()), None)

                has_tensor_options = any(isinstance(a.argument, TensorOptionsArguments) for a in args)

                if local.use_c10_dispatcher() == UseC10Dispatcher.full:
                    cuda_guard_from_tensor_options = """\
    const DeviceGuard device_guard(device_or_default(device));
"""
                else:
                    assert local.use_c10_dispatcher() in [UseC10Dispatcher.with_codegenerated_unboxing_wrapper,
                                                          UseC10Dispatcher.hacky_wrapper_for_legacy_signatures]
                    cuda_guard_from_tensor_options = """\
    const DeviceGuard device_guard(options.device());
"""

                # TODO: There is probably a simpler version of this that
                # works just as well.
                if f.device_guard and (dispatch is None or 'Vulkan' == dispatch) and has_tensor_options:
                    cuda_guard = cuda_guard_from_tensor_options
                elif f.device_guard and dispatch is not None and 'CUDA' in dispatch and has_tensor_options:
                    cuda_guard = f"""\
    globalContext().lazyInitCUDA();
    {cuda_guard_from_tensor_options}
"""
                elif f.device_guard and device_of is not None:
                    cuda_guard = f"""\
    const OptionalDeviceGuard device_guard(device_of({device_of}));
"""
                else:
                    cuda_guard = """\
    // DeviceGuard omitted
"""

            return f"""\
{returns_type} {name}({args_str}) {{
{cuda_guard}{return_kw}{impl_name}({args_exprs_str});
}}
"""

        elif target is Target.REGISTRATION:
            assert returns_type == dispatcher.returns_type(f.func.returns)
            dispatcher_args = dispatcher.arguments(f.func)
            dispatcher_args_types_str = ', '.join(map(lambda a: a.type, dispatcher_args))
            if dispatch is None or dispatch == 'Math' or dispatch == 'DefaultBackend':
                type_name = f'TypeDefault::{name}'
            else:
                type_name = f'{dispatch}Type::{name}'

            # def registration only happens in TypeDefault
            def_registration = ""
            if dispatch is None:
                def_registration = f'm.def({cpp_string(str(f.func))});\n'

            impl_registration = ""
            if not def_only and not f.manual_kernel_registration and (dispatch is not None or f.dispatch is None):
                # Figure out which signature the function is
                if local.use_c10_dispatcher() is UseC10Dispatcher.full:
                    payload = f"TORCH_FN({type_name})"
                elif local.use_c10_dispatcher() is UseC10Dispatcher.hacky_wrapper_for_legacy_signatures:
                    payload = "c10::impl::hacky_wrapper_for_legacy_signatures<" \
                        f"{returns_type} ({dispatcher_args_types_str})>(TORCH_FN({type_name}))"
                else:
                    assert local.use_c10_dispatcher() is UseC10Dispatcher.with_codegenerated_unboxing_wrapper
                    payload = f"torch::CppFunction::makeUnboxedOnly(&{type_name})"

                # Annotate it with dispatch information if necessary
                #
                # NB: In the ordinary, TypeDerived code generation work flow, specification
                # of the backend is handled by the enclosing block, so the torch::dispatch
                # invocation here is strictly unnecessary.  However, in the fbcode mobile
                # only workflow using per-op registration, these registrations will get dumped
                # in a TORCH_LIBRARY_FRAGMENT that does not have an ambient backend.  So
                # the torch::dispatch specification here is important!  See
                # Note [Redundancy in registration code is OK] for how we handle redundant info.
                if dispatch is not None:
                    payload = f"torch::dispatch(DispatchKey::{dispatch},\n{payload})\n"

                impl_registration = f'm.impl("{f.func.name}",\n{payload});\n'

            return f"{def_registration}{impl_registration}"
        else:
            assert_never(target)

    return func

# Return a string with a comma separated list of expressions that could be used
# to call this operator. This can be used to generate code that wraps operators
# and calls back into them. The process_tensoroptions argument determines how
# tensor options should be treated. They can be
# - PASS_THROUGH: Don't do anything, just handle them as regular arguments
# - SCATTER: Expect a `TensorOptions options` in the scope and scatter it into `options.dtype, ...`
# - GATHER: Expect `dtype, ...` in the scope and gather them into a TensorOptions for calling
def exprs_str(signature: CppSignature,
              process_tensoroptions: dispatcher.ProcessTensoroptions = dispatcher.ProcessTensoroptions.PASS_THROUGH,
              exclude_this: bool = False,
              ) -> str:
    args = signature.cpp_arguments()
    if exclude_this:
        args = [a for a in args if not isinstance(a.argument, ThisArgument)]
    exprs = dispatcher.cpparguments_exprs(args, process_tensoroptions=process_tensoroptions)
    return ', '.join(map(lambda a: a.expr, exprs))

def types_str(signature: CppSignature) -> str:
    args = signature.cpp_arguments()
    exprs = dispatcher.cpparguments_exprs(args, process_tensoroptions=dispatcher.ProcessTensoroptions.PASS_THROUGH)
    return ', '.join(map(lambda a: a.type, exprs))

# Generates Function.cpp and Function.h.  These files provide the
# functional public C++ API, and the scaffolding to call into
# the dispatcher from these functions.  See also compute_tensor_method.
def compute_function(*, target: Target) -> Callable[[NativeFunction], Optional[str]]:
    @with_native_function
    def go(f: NativeFunction) -> Optional[str]:
        if f.manual_kernel_registration:
            return None
        if Variant.function not in f.variants:
            return None

        cpp_returns_type = cpp.returns_type(f.func.returns)
        cpp_name = cpp.name(f.func)
        signature_group = cpp.signature_group(f.func, method=False)

        if target is Target.DECLARATION:
            if signature_group.gathered_signature is None:
                # There's no TensorOptions
                return f"""
CAFFE2_API {cpp_returns_type} {cpp_name}({signature_group.signature.cpp_arguments_str(with_defaults=True)});
"""
            else:
                # There's TensorOptions in the API. Create 2 APIs - one taking the TensorOptions object ("gathered_signature"),
                # and one taking a scattered signature with ScalarType, Layout, Device separately ("signature").
                # The gathered_signature already exists in several older PyTorch versions and had default arguments.
                # For backward compatibility, we left it unchanged and added the scattered API on top of it.
                # Note that the scattered API cannot have default arguments or calls will be ambigious.
                return f"""
CAFFE2_API {cpp_returns_type} {cpp_name}({signature_group.gathered_signature.cpp_arguments_str(with_defaults=True)});
CAFFE2_API {cpp_returns_type} {cpp_name}({signature_group.signature.cpp_arguments_str(with_defaults=False)});
"""

        assert target is Target.DEFINITION

        dispatcher_returns_type = dispatcher.returns_type(f.func.returns)

        if signature_group.gathered_signature is None:
            # There's no TensorOptions
            return f"""
// aten::{f.func}
{cpp_returns_type} {cpp_name}({signature_group.signature.cpp_arguments_str(with_defaults=False)}) {{
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatcher_returns_type} ({types_str(signature_group.signature)})>();
    return op.call({exprs_str(signature_group.signature)});
}}
"""
        elif local.use_c10_dispatcher().dispatcher_uses_new_style():
            # for c10-full ops, the scattered version is the real op and the gathered version is a proxy
            # calling into the scattered version
            return f"""
// aten::{f.func}
{cpp_returns_type} {cpp_name}({signature_group.signature.cpp_arguments_str(with_defaults=False)}) {{
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatcher_returns_type} ({types_str(signature_group.signature)})>();
    return op.call({exprs_str(signature_group.signature)});
}}
{cpp_returns_type} {cpp_name}({signature_group.gathered_signature.cpp_arguments_str(with_defaults=False)}) {{
    return {cpp_name}({exprs_str(signature_group.gathered_signature, dispatcher.ProcessTensoroptions.SCATTER)});
}}
"""
        else:
            # for non-c10-full ops, the gathered version is the real op and the scattered version is a proxy
            # calling into the gathered version
            return f"""
// aten::{f.func}
{cpp_returns_type} {cpp_name}({signature_group.gathered_signature.cpp_arguments_str(with_defaults=False)}) {{
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatcher_returns_type} ({types_str(signature_group.gathered_signature)})>();
    return op.call({exprs_str(signature_group.gathered_signature)});
}}
{cpp_returns_type} {cpp_name}({signature_group.signature.cpp_arguments_str(with_defaults=False)}) {{
    return {cpp_name}({exprs_str(signature_group.gathered_signature, dispatcher.ProcessTensoroptions.GATHER)});
}}
"""

    return go

# Generates TensorBody.h (sic) and TensorMethods.cpp.  These files provide the
# object-oriented (method-based) public C++ API, and the scaffolding to call into
# the dispatcher from these functions.  See also compute_function.
def compute_tensor_method(*, target: Target) -> Callable[[NativeFunction], Optional[str]]:
    @with_native_function
    def go(f: NativeFunction) -> Optional[str]:
        if Variant.method not in f.variants:
            return None

        assert not f.func.is_out_fn()
        assert len(f.func.arguments) > 0
        assert sum(a.name == 'self' for a in f.func.arguments) == 1

        cpp_name = cpp.name(f.func)
        cpp_returns_type = cpp.returns_type(f.func.returns)
        signature_group = cpp.signature_group(f.func, method=True)

        if target is Target.DECLARATION:
            if signature_group.gathered_signature is None:
                # There's no TensorOptions. Just create the API without concern for TensorOptions.
                return f"{cpp_returns_type} {cpp_name}({signature_group.signature.cpp_arguments_str(with_defaults=True)}) const;"
            else:
                # There's TensorOptions in the API. Create 2 APIs - one taking the TensorOptions object ("gathered_signature"),
                # and one taking a scattered signature with ScalarType, Layout, Device separately ("signature").
                # The gathered_signature already exists in several older PyTorch versions and had default arguments.
                # For backward compatibility, we left it unchanged and added the scattered API on top of it.
                # Note that the scattered API cannot have default arguments or calls will be ambigious.
                return f"""
{cpp_returns_type} {cpp_name}({signature_group.gathered_signature.cpp_arguments_str(with_defaults=True)}) const;
{cpp_returns_type} {cpp_name}({signature_group.signature.cpp_arguments_str(with_defaults=False)}) const;
"""

        assert target is Target.DEFINITION

        dispatcher_returns_type = dispatcher.returns_type(f.func.returns)

        result = f"""
// aten::{f.func}
{cpp_returns_type} Tensor::{cpp_name}({signature_group.signature.cpp_arguments_str(with_defaults=False)}) const {{
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatcher_returns_type} ({types_str(signature_group.signature)})>();
    return op.call({exprs_str(signature_group.signature)});
}}
"""

        if signature_group.gathered_signature is None:
            # There's no TensorOptions
            return f"""
// aten::{f.func}
{cpp_returns_type} Tensor::{cpp_name}({signature_group.signature.cpp_arguments_str(with_defaults=False)}) const {{
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatcher_returns_type} ({types_str(signature_group.signature)})>();
    return op.call({exprs_str(signature_group.signature)});
}}
"""
        elif local.use_c10_dispatcher().dispatcher_uses_new_style():
            # for c10-full ops, the scattered version is the real op and the gathered version is a proxy
            # calling into the scattered version
            return f"""
// aten::{f.func}
{cpp_returns_type} Tensor::{cpp_name}({signature_group.signature.cpp_arguments_str(with_defaults=False)}) const {{
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatcher_returns_type} ({types_str(signature_group.signature)})>();
    return op.call({exprs_str(signature_group.signature)});
}}
{cpp_returns_type} Tensor::{cpp_name}({signature_group.gathered_signature.cpp_arguments_str(with_defaults=False)}) const {{
    return {cpp_name}({exprs_str(signature_group.gathered_signature, dispatcher.ProcessTensoroptions.SCATTER, exclude_this=True)});
}}
"""
        else:
            # for non-c10-full ops, the gathered version is the real op and the scattered version is a proxy
            # calling into the gathered version
            return f"""
// aten::{f.func}
{cpp_returns_type} Tensor::{cpp_name}({signature_group.gathered_signature.cpp_arguments_str(with_defaults=False)}) const {{
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatcher_returns_type} ({types_str(signature_group.gathered_signature)})>();
    return op.call({exprs_str(signature_group.gathered_signature)});
}}
{cpp_returns_type} Tensor::{cpp_name}({signature_group.signature.cpp_arguments_str(with_defaults=False)}) const {{
    return {cpp_name}({exprs_str(signature_group.gathered_signature, dispatcher.ProcessTensoroptions.GATHER, exclude_this=True)});
}}
"""

    return go

# Generates ATenOpList.cpp, a runtime accessible list of all aten
# operators.
# TODO: This was historically used to help some JIT interop code
# figure out whether or not to treat aten namespace'd operators
# one way or another, we should reevaluate if this is actually needed.
@with_native_function
def compute_aten_op(f: NativeFunction) -> str:
    return f'{{"aten::{f.func.name.name}", "{f.func.name.overload_name}"}},'

# Generates NativeFunctions.h, a list of forward declarations of all
# actual kernel definitions we keep in aten/src/ATen/native/
@with_native_function
def compute_native_function_declaration(f: NativeFunction) -> List[str]:
    if f.dispatch is None:
        ns = [cpp.name(f.func)]
    else:
        ns = list(f.dispatch.values())

    rs = []
    # Sometimes a function name shows up multiple times; only generate
    # it once!
    seen = set()
    for n in ns:
        if n in seen:
            continue
        if "legacy::" in n:
            continue
        seen.add(n)
        returns_type = legacy_dispatcher.returns_type(f.func.returns)
        args = legacy_dispatcher.arguments(f.func)
        rs.append(f"CAFFE2_API {returns_type} {n}({', '.join(map(lambda a: a.str_with_default(), args))});")

    return rs

# Generates BackendSelectRegister.cpp, a series of kernels which provide
# specialized computation of dispatch key for operator signatures which cannot
# be easily done automatically using templating.
def compute_backend_select(*, target: Target) -> Callable[[NativeFunction], Optional[str]]:
    @with_native_function
    def go(f: NativeFunction) -> Optional[str]:
        if str(f.func.name.name).endswith('_like') or str(f.func.name.name).startswith('new_'):
            return None

        name = legacy_dispatcher.name(f.func)
        legacy_dispatcher_returns_type = legacy_dispatcher.returns_type(f.func.returns)
        legacy_dispatcher_args = legacy_dispatcher.arguments(f.func)

        if not any(isinstance(a.argument, TensorOptionsArguments) for a in legacy_dispatcher_args):
            return None

        legacy_dispatcher_tensor_args = [
            a for a in legacy_dispatcher_args
            if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()
        ]

        dispatcher_returns_type = dispatcher.returns_type(f.func.returns)
        dispatcher_args = dispatcher.arguments(f.func)

        args: Union[Sequence[DispatcherArgument], Sequence[LegacyDispatcherArgument]]
        if local.use_c10_dispatcher().dispatcher_uses_new_style():
            returns_type = dispatcher_returns_type
            args = dispatcher_args
            exprs = dispatcher.exprs(dispatcher_args)
            dispatch_key = "c10::computeDispatchKey(dtype, layout, device)"
        else:
            returns_type = legacy_dispatcher_returns_type
            args = legacy_dispatcher_args
            exprs = dispatcher.legacydispatcherarguments_exprs(legacy_dispatcher_args)
            dispatch_key = "options.computeDispatchKey()"

        if target is Target.DEFINITION:
            # I don't think there's actually a good reason to generate
            # these two cases differently
            # The first case could probably be improved though- it calls dispatchTypeId(),
            # which looks at TLS dispatch keys- there should not be any by the time we reach backend select.
            if legacy_dispatcher_tensor_args:
                tensor_args = ', '.join(a.name for a in legacy_dispatcher_tensor_args)
                compute_dk = f"""\
DispatchKeySet _dk_set = c10::DispatchKeySet({dispatch_key}) | c10::detail::multi_dispatch_key_set({tensor_args});
  DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
  DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);"""
            else:
                compute_dk = f"DispatchKey _dk = {dispatch_key};"
            return f"""\
// aten::{f.func}
{returns_type} {name}({', '.join(str(a) for a in args)}) {{
  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
    .typed<{dispatcher_returns_type} ({', '.join(a.type for a in dispatcher_args)})>();
  {compute_dk}
  DispatchKey _autograd_dk = c10::getAutogradKeyFromBackend(_dk);
  // This trick allows calling Autograd backend kernel first and then backend kernel,
  // without adding another AutogradBackendSelect dispatch key.
  DispatchKey _current_dk = at::impl::variable_excluded_from_dispatch() ? _dk : _autograd_dk;
  return op.callWithDispatchKey(_current_dk, {', '.join(a.expr for a in exprs)});
}}
"""
        elif target is Target.REGISTRATION:
            if local.use_c10_dispatcher() is UseC10Dispatcher.full:
                return f"""m.impl("aten::{f.func.name}", TORCH_FN({name}));"""
            elif local.use_c10_dispatcher() is UseC10Dispatcher.hacky_wrapper_for_legacy_signatures:
                return f"""m.impl("aten::{f.func.name}",
          c10::impl::hacky_wrapper_for_legacy_signatures<{dispatcher_returns_type} ({', '.join(a.type for a in dispatcher_args)})>(
            TORCH_FN({name})));"""
            else:
                assert local.use_c10_dispatcher() is UseC10Dispatcher.with_codegenerated_unboxing_wrapper
                return f"""m.impl_UNBOXED("aten::{f.func.name}", {name});"""
        elif target is Target.DECLARATION:
            raise AssertionError()
        else:
            assert_never(target)
    return go

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                       YAML CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def dict_representer(dumper: Any, data: Any) -> Any:
    return dumper.represent_dict(data.items())

def format_yaml(data: object) -> str:
    noalias_dumper = yaml.dumper.SafeDumper
    noalias_dumper.ignore_aliases = lambda self, data: True  # type: ignore
    # Support serializing OrderedDict
    noalias_dumper.add_representer(OrderedDict, dict_representer)  # type: ignore
    # Some yaml parsers (e.g. Haskell's) don't understand line breaks.
    # width=float('Inf') turns off optional line breaks and improves
    # the portability of the outputted yaml.
    return yaml.dump(data, default_flow_style=False, Dumper=noalias_dumper, width=float('Inf'))  # type: ignore

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
        return 'Tensor'
    return cpp.argumenttype_type(t, mutable=False)

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
    returns = []
    for i, r in enumerate(f.func.returns):
        # If we have an inplace function, the return argument is
        # implicitly named self.
        # TODO: Consider incorporating this into the data model
        if f.func.name.name.inplace:
            assert i == 0, "illegal inplace function with multiple returns"
            name = 'self'
        # If we are out function, the name is the name of the
        # corresponding output function (r.name will get recorded
        # in field_name later.)
        elif f.func.is_out_fn():
            name = f.func.out_arguments[i].name
        # If the return argument is explicitly named...
        elif r.name:
            name_conflict = any(r.name == a.name for a in f.func.schema_order_arguments())
            if name_conflict and not f.func.is_out_fn():
                name = f'{r.name}_return'
            else:
                name = r.name
        # If there is no explicit name, we just name the output result,
        # unless it's a multi-return, in which case it's result0,
        # result1, etc (zero-indexed)
        else:
            name = 'result' if len(f.func.returns) == 1 else f'result{i}'

        ret = {
            'dynamic_type': dynamic_type(r.type),
            'name': name,
            'type': cpp.return_type(r),
        }

        if r.name:
            # See Note [name and field_name]
            ret['field_name'] = r.name
            if f.func.is_out_fn():
                name_to_field_name[f.func.out_arguments[i].name] = r.name

        returns.append(ret)

    return returns, name_to_field_name

# arguments in yaml roughly corresponds to the public C++ API
def compute_cpp_argument_yaml(cpp_a: CppArgument, *, schema_order: bool, kwarg_only_set: Set[str],
                              out_arg_set: Set[str], name_to_field_name: Dict[str, str]) -> object:
    if isinstance(cpp_a.argument, TensorOptionsArguments):
        arg: Dict[str, object] = {
            'annotation': None,
            'dynamic_type': 'TensorOptions',
            'is_nullable': False,
            'name': cpp_a.name,
            'type': cpp_a.type,
            'kwarg_only': True,
        }
        if cpp_a.default is not None:
            arg['default'] = cpp_a.default
        return arg
    elif isinstance(cpp_a.argument, ThisArgument):
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
        'type': cpp.argument_type(a),
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
    kwarg_only_set = set(a.name for a in f.func.kwarg_only_arguments)
    out_arg_set = set(a.name for a in f.func.out_arguments)

    signature_group = cpp.signature_group(f.func)
    cpp_args = signature_group.signature_prefer_gathered().cpp_arguments()
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

    cpp_schema_order_types = [cpp.argument(a).type for a in schema_order_jit_arguments]
    cpp_returns = cpp.returns_type(f.func.returns)
    schema_order_cpp_signature = f"{cpp_returns} ({', '.join(cpp_schema_order_types)})"

    is_factory_method = any(isinstance(a.argument, TensorOptionsArguments) for a in cpp_args) \
        and Variant.method not in f.variants

    return OrderedDict([
        ('name', cpp.name(f.func)),
        ('operator_name', str(f.func.name.name)),
        ('overload_name', str(f.func.name.overload_name)),
        ('use_c10_dispatcher', f.use_c10_dispatcher.name),
        ('manual_kernel_registration', f.manual_kernel_registration),
        ('category_override', f.category_override if f.category_override is not None else ''),
        ('matches_jit_signature', True),
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
        # Note [Abstract ATen methods]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # An abstract ATen method is one whose dispatch differs between
        # types.  These are implemented in derived types (with a
        # standard (throwing) definition in Type).  A concrete ATen
        # method is one which has the same dispatch for all types;
        # we just implement it in the base Type.  This is exposed
        # in Declarations.yaml via a field named 'abstract'.
        #
        # Although this is what we have historically exposed, it is
        # actually not all that useful for end users, who are also interested
        # whether or not there is an explicit entry in derivatives.yaml
        # for the entry or not (as this affects whether or not the operation is
        # overrideable or not.)  Once this all gets cleaned up, this
        # property will be obsolete.
        ('abstract', f.dispatch is not None),
        ('device_guard', f.device_guard),
        ('with_gil', False),
        ('deprecated', False),
        ('has_math_kernel', f.dispatch is not None and 'Math' in f.dispatch),
    ])

@with_native_function
def compute_registration_declarations(f: NativeFunction) -> str:
    name = dispatcher.name(f.func)
    returns_type = dispatcher.returns_type(f.func.returns)
    args = dispatcher.arguments(f.func)
    args_str = ', '.join(map(str, args))
    comment_data : Dict[str, str] = {
        'schema': f'aten::{f.func}',
        'dispatch': str(f.dispatch is not None),
        'math': str(f.dispatch is not None and 'Math' in f.dispatch)
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
                comment = "@" + "generated by aten/src/ATen/gen.py"
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
    # TODO: remove this, we should just unconditionally generate Vulkan
    parser.add_argument(
        '--vulkan',
        action='store_true',
        help='Generate Vulkan backend functions')
    parser.add_argument(
        '--op_registration_whitelist',
        nargs='*',
        help='filter op registrations by the whitelist (if set); '
             'each item is `namespace`::`operator name` without overload name; '
             'e.g.: aten::empty aten::conv2d ...')
    parser.add_argument(
        '--backend_whitelist',
        nargs='*',
        help='filter dispatch backend by the whitelist (if set), '
             'e.g.: CPU CUDA QuantizedCPU ...')
    parser.add_argument(
        '--force_schema_registration',
        action='store_true',
        help='force it to generate schema-only registrations for all ops, including'
             'those that are not listed on --op_registration_whitelist')
    options = parser.parse_args()

    op_registration_whitelist: Optional[Set[str]]
    if options.op_registration_whitelist is not None:
        op_registration_whitelist = set(options.op_registration_whitelist)
    else:
        op_registration_whitelist = None

    native_functions = parse_native_yaml(os.path.join(options.source_path, 'native/native_functions.yaml'))

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
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>'''
    if options.rocm:
        extra_cuda_headers = '''\
#include <ATen/DeviceGuard.h>
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hip/HIPDevice.h>
#include <ATen/hip/HIPContext.h>'''

    backends = ["CPU", "SparseCPU", "MkldnnCPU", "CUDA", "SparseCUDA", "QuantizedCPU", "QuantizedCUDA"]
    if options.vulkan:
        backends.append("Vulkan")
    if options.backend_whitelist:
        backends = [b for b in backends if b in options.backend_whitelist]

    for dispatch in backends:
        h_template = 'TypeDerived.h'
        cpp_template = 'TypeDerived.cpp'
        # TODO: delete this special case
        if 'Sparse' in dispatch:
            cpp_template = 'SparseTypeDerived.cpp'

        fm = cuda_fm if 'CUDA' in dispatch else cpu_fm

        fm.write_with_template(f'{dispatch}Type.h', h_template, lambda: {
            'Type': f'{dispatch}Type',
            'extra_cuda_headers': extra_cuda_headers if 'CUDA' in dispatch else '',  # TODO: remove this
            'type_derived_method_declarations': list(mapMaybe(
                compute_type_method(dispatch, target=Target.DECLARATION, op_registration_whitelist=op_registration_whitelist),
                native_functions
            )),
        })
        fm.write_with_template(f'{dispatch}Type.cpp', cpp_template, lambda: {
            'Type': f'{dispatch}Type',
            # TODO: remove this
            'extra_cuda_headers': extra_cuda_headers if 'CUDA' in dispatch else '',
            # TODO: remove this
            'storage_tensor_headers': '#include <c10/core/TensorImpl.h>',
            # TODO: remove this
            'Generator': 'CUDAGeneratorImpl' if 'CUDA' in dispatch else 'CPUGeneratorImpl',
            'legacy_th_headers':
                '#include <ATen/LegacyTHFunctionsCPU.h>' if dispatch == "CPU" else
                '#include <ATen/LegacyTHFunctionsCUDA.h>' if dispatch == "CUDA" else
                '',
            'Backend': dispatch,
            'type_derived_method_definitions': list(mapMaybe(
                compute_type_method(dispatch, target=Target.DEFINITION, op_registration_whitelist=op_registration_whitelist),
                native_functions
            )),
            'function_registrations': list(mapMaybe(
                compute_type_method(
                    dispatch, target=Target.REGISTRATION, op_registration_whitelist=op_registration_whitelist),
                native_functions)),
        })
        del fm

    cpu_fm.write('TypeDefault.h', lambda: {
        'type_method_declarations':
        list(mapMaybe(
            compute_type_method(None, target=Target.DECLARATION, op_registration_whitelist=op_registration_whitelist),
            native_functions)) +
        list(mapMaybe(
            compute_type_method('Math', target=Target.DECLARATION, op_registration_whitelist=op_registration_whitelist),
            native_functions)),

    })
    cpu_fm.write('TypeDefault.cpp', lambda: {
        'type_method_definitions':
        list(mapMaybe(
            compute_type_method(None, target=Target.DEFINITION, op_registration_whitelist=op_registration_whitelist),
            native_functions)) +
        list(mapMaybe(
            compute_type_method('Math', target=Target.DEFINITION, op_registration_whitelist=op_registration_whitelist),
            native_functions)) +
        list(mapMaybe(
            compute_type_method('DefaultBackend', target=Target.DEFINITION, op_registration_whitelist=op_registration_whitelist),
            native_functions)),

        'function_registrations': list(mapMaybe(
            compute_type_method(None, target=Target.REGISTRATION, op_registration_whitelist=op_registration_whitelist),
            native_functions)),

        'default_backend_function_registrations': list(mapMaybe(
            compute_type_method('DefaultBackend', target=Target.REGISTRATION, op_registration_whitelist=op_registration_whitelist),
            native_functions)),

        'math_function_registrations': list(mapMaybe(
            compute_type_method('Math', target=Target.REGISTRATION, op_registration_whitelist=op_registration_whitelist),
            native_functions)),
    })
    cpu_fm.write('Functions.h', lambda: {
        'function_declarations': list(mapMaybe(compute_function(target=Target.DECLARATION), native_functions)),
    })
    cpu_fm.write('Functions.cpp', lambda: {
        'function_definitions': list(mapMaybe(compute_function(target=Target.DEFINITION), native_functions)),
    })
    core_fm.write('TensorBody.h', lambda: {
        'tensor_method_declarations': list(mapMaybe(compute_tensor_method(target=Target.DECLARATION), native_functions)),
    })
    core_fm.write('TensorMethods.cpp', lambda: {
        'tensor_method_definitions': list(mapMaybe(compute_tensor_method(target=Target.DEFINITION), native_functions)),
    })
    core_fm.write('ATenOpList.cpp', lambda: {
        'aten_ops': list(mapMaybe(compute_aten_op, native_functions)),
    })
    cpu_fm.write('NativeFunctions.h', lambda: {
        'native_function_declarations': list(concatMap(compute_native_function_declaration, native_functions)),
    })
    cpu_fm.write('BackendSelectRegister.cpp', lambda: {
        'backend_select_method_definitions':
            list(mapMaybe(compute_backend_select(target=Target.DEFINITION), native_functions)),
        'backend_select_function_registrations':
            list(mapMaybe(compute_backend_select(target=Target.REGISTRATION), native_functions)),
    })

    if options.force_schema_registration:
        def computeSchemaRegister() -> Dict[str, object]:
            schema_registrations = list(mapMaybe(
                compute_type_method(None, target=Target.REGISTRATION, op_registration_whitelist=None, def_only=True),
                native_functions))
            return {
                'schema_registrations': schema_registrations,
            }
        cpu_fm.write('SchemaRegister.cpp', computeSchemaRegister)

    cpu_fm.write('Declarations.yaml', lambda: format_yaml(list(map(compute_declaration_yaml, native_functions))))
    cpu_fm.write('RegistrationDeclarations.h', lambda: {
        'registration_declarations': list(map(compute_registration_declarations, native_functions)),
    })

    if options.output_dependencies:
        cpu_fm.write_outputs(options.output_dependencies)
        core_fm.write_outputs(f"{options.output_dependencies}-core")
        cuda_fm.write_outputs(f"{options.output_dependencies}-cuda")

if __name__ == '__main__':
    main()
