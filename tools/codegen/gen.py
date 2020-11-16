import os
import contextlib
import textwrap
import itertools
from typing import List, Dict, Optional, Iterator, Tuple, Set, Callable, Any, TypeVar, Union, Sequence
import yaml
from enum import Enum
from collections import OrderedDict, defaultdict
import argparse
import pathlib
import functools
import json
from dataclasses import dataclass

from tools.codegen.code_template import CodeTemplate
from tools.codegen.model import *
from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp
import tools.codegen.api.dispatcher as dispatcher
import tools.codegen.api.native as native
import tools.codegen.local as local
from tools.codegen.selective_build.selector import SelectiveBuilder

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
        with native_function_manager(f):
            return func(f)
    return wrapper

def method_with_native_function(func: Callable[[S, NativeFunction], T]) -> Callable[[S, NativeFunction], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: NativeFunction) -> T:
        with native_function_manager(f):
            return func(slf, f)
    return wrapper

@contextlib.contextmanager
def native_function_manager(f: NativeFunction) -> Iterator[None]:
    with context(f'in {f.loc}:\n  {f.func}'):
        with local.parametrize(
            use_c10_dispatcher=f.use_c10_dispatcher,
        ):
            yield

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

# Dispatch keys that "support all backends".  These codegen slightly differently
# then backend specific keys.
def is_generic_dispatch_key(dk: str) -> bool:
    return dk in {'DefaultBackend', 'Math'}

# CUDA specific dispatch keys
def is_cuda_dispatch_key(dk: str) -> bool:
    return 'CUDA' in dk

# Generates RegisterSchema.cpp.  Depending on the selector, either
# all schemas are registered, or only some are (in the case of
# selective build)
@dataclass(frozen=True)
class RegisterSchema:
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        op_name = f"aten::{f.func.name}"
        if not self.selector.is_operator_selected(op_name):
            return None
        return f'm.def({cpp_string(str(f.func))});\n'

# Generates Register{dispatch}.cpp (e.g., RegisterCPU.cpp).
#
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
@dataclass(frozen=True)
class RegisterDispatchKey:
    dispatch_key: str

    # TODO: Give more precise type Union[Literal[Target.DEFINITION,
    # Target.REGISTRATION]]; requires Literal from typing_extensions
    # which we don't have a dep for yet.
    target: Target

    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    def __post_init__(self) -> None:
        assert self.target is not Target.DECLARATION

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        # for mypy type refinement; would be fixed by TODO on target
        assert self.target is not Target.DECLARATION

        if self.dispatch_key not in f.dispatch:
            return None

        op_name = f"aten::{f.func.name}"
        if self.target is Target.REGISTRATION and not self.selector.is_operator_selected(op_name):
            return None

        name = native.name(f.func)
        returns_type = native.returns_type(f.func.returns)
        args = native.arguments(f.func)
        args_str = ', '.join(map(str, args))

        if self.target is Target.DEFINITION:
            impl_name = f"at::native::{f.dispatch[self.dispatch_key]}"

            args_exprs_str = ', '.join(a.name for a in args)

            return_kw = "    return "

            cuda_guard = ""
            if is_generic_dispatch_key(self.dispatch_key) or is_cuda_dispatch_key(self.dispatch_key):
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
                if f.device_guard and is_generic_dispatch_key(self.dispatch_key) and has_tensor_options:
                    cuda_guard = cuda_guard_from_tensor_options
                elif f.device_guard and is_cuda_dispatch_key(self.dispatch_key) and has_tensor_options:
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

        elif self.target is Target.REGISTRATION:
            if f.manual_kernel_registration:
                return None
            else:
                dispatcher_sig = DispatcherSignature.from_schema(f.func)

                # Figure out which signature the function is
                if local.use_c10_dispatcher() is UseC10Dispatcher.full:
                    payload = f"TORCH_FN({name})"
                elif local.use_c10_dispatcher() is UseC10Dispatcher.hacky_wrapper_for_legacy_signatures:
                    payload = "c10::impl::hacky_wrapper_for_legacy_signatures<" \
                        f"{dispatcher_sig.type()}>(TORCH_FN({name}))"

                else:
                    assert local.use_c10_dispatcher() is UseC10Dispatcher.with_codegenerated_unboxing_wrapper
                    payload = f"torch::CppFunction::makeUnboxedOnly(&{name})"

                return f'm.impl("{f.func.name}",\n{payload});\n'
        else:
            assert_never(self.target)

# Generates Function.cpp and Function.h.  These files provide the
# functional public C++ API, and the scaffolding to call into
# the dispatcher from these functions.  See also compute_tensor_method.
@dataclass(frozen=True)
class ComputeFunction:
    target: Target

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if f.manual_kernel_registration:
            return None
        if Variant.function not in f.variants:
            return None

        name = cpp.name(f.func)

        sig_group = CppSignatureGroup.from_schema(f.func, method=False)

        if self.target is Target.DECLARATION:
            result = f"CAFFE2_API {sig_group.signature.decl()};\n"
            if sig_group.faithful_signature is not None:
                result += f"CAFFE2_API {sig_group.faithful_signature.decl()};\n"
            return result

        assert self.target is Target.DEFINITION

        def generate_defn(sig: CppSignature) -> str:
            dispatcher_sig = DispatcherSignature.from_schema(f.func)

            dispatcher_exprs = dispatcher.cpparguments_exprs(sig.argument_packs())
            dispatcher_exprs_str = ', '.join(a.expr for a in dispatcher_exprs)

            return f"""
// aten::{f.func}
{sig.defn()} {{
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatcher_sig.type()}>();
    return op.call({dispatcher_exprs_str});
}}
"""

        result = generate_defn(sig_group.signature)
        if sig_group.faithful_signature is not None:
            if local.use_c10_dispatcher().dispatcher_uses_new_style():
                result += generate_defn(sig_group.faithful_signature)

        return result

# Generates TensorBody.h (sic) and TensorMethods.cpp.  These files provide the
# object-oriented (method-based) public C++ API, and the scaffolding to call into
# the dispatcher from these functions.  See also compute_function.
@dataclass(frozen=True)
class ComputeTensorMethod:
    target: Target

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if Variant.method not in f.variants:
            return None

        assert not f.func.is_out_fn()
        assert len(f.func.arguments) > 0
        assert sum(a.name == 'self' for a in f.func.arguments) == 1

        name = cpp.name(f.func)

        sig_group = CppSignatureGroup.from_schema(f.func, method=True)

        if self.target is Target.DECLARATION:
            result = f"{sig_group.signature.decl()} const;\n"
            if sig_group.faithful_signature is not None:
                result += f"{sig_group.faithful_signature.decl()} const;\n"
            return result

        assert self.target is Target.DEFINITION

        def generate_defn(sig: CppSignature) -> str:
            dispatcher_sig = DispatcherSignature.from_schema(f.func)

            dispatcher_exprs = dispatcher.cpparguments_exprs(sig.argument_packs())
            dispatcher_exprs_str = ', '.join(a.expr for a in dispatcher_exprs)

            return f"""
// aten::{f.func}
{sig.defn(prefix="Tensor::")} const {{
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatcher_sig.type()}>();
    return op.call({dispatcher_exprs_str});
}}
"""

        result = generate_defn(sig_group.signature)
        if sig_group.faithful_signature is not None:
            result += generate_defn(sig_group.faithful_signature)

        return result

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
        returns_type = native.returns_type(f.func.returns)
        args = native.arguments(f.func)
        rs.append(f"CAFFE2_API {returns_type} {n}({', '.join(a.str_with_default() for a in args)});")

    return rs

# Generates RegisterBackendSelect.cpp, a series of kernels which provide
# specialized computation of dispatch key for operator signatures which cannot
# be easily done automatically using templating.
@dataclass(frozen=True)
class ComputeBackendSelect:
    target: Target

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if str(f.func.name.name).endswith('_like') or str(f.func.name.name).startswith('new_'):
            return None

        name = native.name(f.func)
        native_sig = NativeSignature.from_schema(f.func)

        if not any(isinstance(a.argument, TensorOptionsArguments) for a in native_sig.arguments()):
            return None

        native_tensor_args = [
            a for a in native_sig.arguments()
            if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()
        ]

        dispatcher_sig = DispatcherSignature.from_schema(f.func)

        sig: Union[NativeSignature, DispatcherSignature]
        if local.use_c10_dispatcher().dispatcher_uses_new_style():
            sig = dispatcher_sig
            dispatcher_exprs = dispatcher_sig.exprs()
            dispatch_key = "c10::computeDispatchKey(dtype, layout, device)"
        else:
            sig = native_sig
            dispatcher_exprs = native_sig.dispatcher_exprs()
            dispatch_key = "options.computeDispatchKey()"

        if self.target is Target.DEFINITION:
            # I don't think there's actually a good reason to generate
            # these two cases differently
            # The first case could probably be improved though- it calls dispatchTypeId(),
            # which looks at TLS dispatch keys- there should not be any by the time we reach backend select.
            if native_tensor_args:
                tensor_args = ', '.join(a.name for a in native_tensor_args)
                compute_dk = f"""\
DispatchKeySet _dk_set = c10::DispatchKeySet({dispatch_key}) | c10::detail::multi_dispatch_key_set({tensor_args});
  DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
  DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);"""
            else:
                compute_dk = f"DispatchKey _dk = {dispatch_key};"
            return f"""\
// aten::{f.func}
{sig.defn(name)} {{
  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
    .typed<{dispatcher_sig.type()}>();
  {compute_dk}
  return op.callWithDispatchKey(_dk, {', '.join(a.expr for a in dispatcher_exprs)});
}}
"""
        elif self.target is Target.REGISTRATION:
            if local.use_c10_dispatcher() is UseC10Dispatcher.full:
                return f"""m.impl("aten::{f.func.name}", TORCH_FN({name}));"""
            elif local.use_c10_dispatcher() is UseC10Dispatcher.hacky_wrapper_for_legacy_signatures:
                return f"""m.impl("aten::{f.func.name}",
          c10::impl::hacky_wrapper_for_legacy_signatures<{dispatcher_sig.type()}>(
            TORCH_FN({name})));"""
            else:
                assert local.use_c10_dispatcher() is UseC10Dispatcher.with_codegenerated_unboxing_wrapper
                return f"""m.impl_UNBOXED("aten::{f.func.name}", {name});"""
        elif self.target is Target.DECLARATION:
            raise AssertionError()
        else:
            assert_never(self.target)

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
    names = cpp.return_names(f)
    returns = []
    for i, (r, name) in enumerate(zip(f.func.returns, names)):
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

    sig_group = CppSignatureGroup.from_schema(f.func, method=False)
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

    cpp_schema_order_types = [cpp.argument(a).type for a in schema_order_jit_arguments]
    cpp_returns = cpp.returns_type(f.func.returns)
    schema_order_cpp_signature = f"{cpp_returns} ({', '.join(cpp_schema_order_types)})"

    is_factory_method = any(isinstance(a.argument, TensorOptionsArguments) for a in cpp_args) \
        and Variant.method not in f.variants

    is_abstract = f.dispatch.keys() != {'Math'}

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
        ('abstract', is_abstract),
        ('device_guard', f.device_guard),
        ('with_gil', False),
        ('deprecated', False),
        ('has_math_kernel', 'Math' in f.dispatch),
    ])

@with_native_function
def compute_registration_declarations(f: NativeFunction) -> str:
    name = dispatcher.name(f.func)
    returns_type = dispatcher.returns_type(f.func.returns)
    args = dispatcher.arguments(f.func)
    args_str = ', '.join(map(str, args))
    comment_data : Dict[str, str] = {
        'schema': f'aten::{f.func}',
        # TODO: What exactly is the semantics of the 'dispatch' field?
        'dispatch': str(f.dispatch.keys() != {'Math'}),
        'default': str(any(is_generic_dispatch_key(k) for k in f.dispatch))
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
        '--force_schema_registration',
        action='store_true',
        help='force it to generate schema-only registrations for all ops, including'
             'those that are not listed on --op_registration_whitelist')
    options = parser.parse_args()

    selector = get_custom_build_selector(
        options.op_registration_whitelist,
        options.op_selection_yaml_path,
    )

    native_functions = parse_native_yaml(os.path.join(options.source_path, 'native/native_functions.yaml'))

    pre_grouped_native_functions: Dict[FunctionSchema, Dict[SchemaKind, NativeFunction]]
    pre_grouped_native_functions = defaultdict(dict)
    for f in native_functions:
        d = pre_grouped_native_functions[f.func.signature()]
        assert f.func.kind() not in d
        d[f.func.kind()] = f
    grouped_native_functions = [NativeFunctionGroup.from_dict(v) for v in pre_grouped_native_functions.values()]
    # NB: At the moment, grouped_native_functions isn't used by anything,
    # this code lives here to help potential future consumers; for a live
    # example see https://github.com/pytorch/pytorch/pull/45277

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
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>'''
    if options.rocm:
        extra_cuda_headers = '''\
#include <ATen/hip/ATenHIPGeneral.h>
#include <ATen/hip/HIPDevice.h>
#include <ATen/hip/HIPContext.h>'''

    # NB: substrings in these dispatch keys matter, we do tests to see if
    # a key contains, e.g., CUDA to classify it as a CUDA backend
    dispatch_keys = [
        "CPU",
        "SparseCPU",
        "MkldnnCPU",
        "CUDA",
        "SparseCUDA",
        "QuantizedCPU",
        "QuantizedCUDA",
        "Math",
        "DefaultBackend",
    ]
    if options.backend_whitelist:
        dispatch_keys = [k for k in dispatch_keys if is_generic_dispatch_key(k) or k in options.backend_whitelist]

    for dispatch_key in dispatch_keys:
        cpp_template = 'RegisterDispatchKey.cpp'

        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm

        fm.write_with_template(f'Register{dispatch_key}.cpp', cpp_template, lambda: {
            'extra_cuda_headers': extra_cuda_headers if is_cuda_dispatch_key(dispatch_key) else '',
            'legacy_th_headers':
                '#include <ATen/LegacyTHFunctionsCPU.h>' if dispatch_key == "CPU" else
                '#include <ATen/LegacyTHFunctionsCUDA.h>' if dispatch_key == "CUDA" else
                '',
            'DispatchKey': dispatch_key,
            'dispatch_definitions': list(mapMaybe(
                RegisterDispatchKey(dispatch_key, Target.DEFINITION, selector),
                native_functions
            )),
            'dispatch_registrations': list(mapMaybe(
                RegisterDispatchKey(dispatch_key, Target.REGISTRATION, selector),
                native_functions
            )),
        })
        del fm

    # BackendSelect is generated specially
    cpu_fm.write('RegisterBackendSelect.cpp', lambda: {
        'backend_select_method_definitions':
            list(mapMaybe(ComputeBackendSelect(Target.DEFINITION), native_functions)),
        'backend_select_function_registrations':
            list(mapMaybe(ComputeBackendSelect(Target.REGISTRATION), native_functions)),
    })

    schema_selector = selector
    if options.force_schema_registration:
        schema_selector = SelectiveBuilder.get_nop_selector()
    cpu_fm.write('RegisterSchema.cpp', lambda: {
        'schema_registrations': list(mapMaybe(RegisterSchema(schema_selector), native_functions)),
    })

    cpu_fm.write('Functions.h', lambda: {
        'function_declarations': list(mapMaybe(ComputeFunction(Target.DECLARATION), native_functions)),
    })
    cpu_fm.write('Functions.cpp', lambda: {
        'function_definitions': list(mapMaybe(ComputeFunction(Target.DEFINITION), native_functions)),
    })
    core_fm.write('TensorBody.h', lambda: {
        'tensor_method_declarations': list(mapMaybe(ComputeTensorMethod(Target.DECLARATION), native_functions)),
    })
    core_fm.write('TensorMethods.cpp', lambda: {
        'tensor_method_definitions': list(mapMaybe(ComputeTensorMethod(Target.DEFINITION), native_functions)),
    })
    core_fm.write('ATenOpList.cpp', lambda: {
        'aten_ops': list(mapMaybe(compute_aten_op, native_functions)),
    })
    cpu_fm.write('NativeFunctions.h', lambda: {
        'native_function_declarations': list(concatMap(compute_native_function_declaration, native_functions)),
    })

    cpu_fm.write('Declarations.yaml', lambda: format_yaml([compute_declaration_yaml(f) for f in native_functions]))
    cpu_fm.write('RegistrationDeclarations.h', lambda: {
        'registration_declarations': [compute_registration_declarations(f) for f in native_functions],
    })

    if options.output_dependencies:
        cpu_fm.write_outputs(options.output_dependencies)
        core_fm.write_outputs(f"{options.output_dependencies}-core")
        cuda_fm.write_outputs(f"{options.output_dependencies}-cuda")

if __name__ == '__main__':
    main()
