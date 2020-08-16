import sys
import os
import contextlib
import textwrap
import itertools
from typing import List, Sequence, Dict, Optional, Iterator, Tuple, Set, Callable, Union, Any, TypeVar, DefaultDict
import yaml
from enum import Enum
from collections import OrderedDict, defaultdict
import argparse

# Reusing CodeTemplate from existing codegen
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../aten/src/ATen"))
from code_template import CodeTemplate

from tools.codegen.model import *
from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp
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


# Note [Byte-for-byte compatibility]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Some special cases we have made in this codegen have been strictly
# to make sure that git diff -w reports no changes, but we believe
# they are not semantically meaningful.  After landing the new codegen,
# we should remove these special cases

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

# Map a function over native functions entries, concatenating
# the returned lists into a final returned list.  Unlike a generic
# concatmap, this function also sets up error reporting context and
# some dynamically scoped variables which are used within code
# generator.
#
# NB about type: accepting Sequence is modestly more flexible but then
# runs afoul of https://github.com/python/mypy/issues/5090
def concatmap_nf(func: Callable[[NativeFunction], List[T]], native_functions: Sequence[NativeFunction]) -> List[T]:
    rs: List[T] = []
    for f in native_functions:
        with context(f'in {f.loc}:\n  {f.func}'):
            with local.parametrize(
                use_c10_dispatcher_full=f.use_c10_dispatcher_full,
                # See Note [Byte-for-byte compatibility]
                hack_const_mutable_self=str(f.func.name) in ["set_data", "retain_grad"],
            ):
                rs.extend(func(f))
    return rs

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                        C++ CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

Target = Enum('Target', ('DEFINITION', 'DECLARATION', 'REGISTRATION'))

def compute_type_method(dispatch: Optional[str], *, target: Target, op_registration_whitelist: Optional[Set[str]], def_only: bool = False) -> Callable[[NativeFunction], List[str]]:
    if def_only:
        assert target is Target.REGISTRATION
    def func(f: NativeFunction) -> List[str]:
        if dispatch is not None:
            if f.dispatch is None or dispatch not in f.dispatch:
                return []
        else:
            if f.dispatch is not None and target is not Target.REGISTRATION:
                return []

        if op_registration_whitelist is not None and f"aten::{f.func.name.name}" not in op_registration_whitelist and target is Target.REGISTRATION:
            return []

        name = legacy_dispatcher.name(f.func)
        returns_type = legacy_dispatcher.returns_type(f.func.returns)
        args = legacy_dispatcher.arguments(f.func)
        args_str = ', '.join(map(str, args))

        if target is Target.DECLARATION:
            return [f"{returns_type} {name}({args_str});"]
        elif target is Target.DEFINITION:
            if f.dispatch is None:
                cpp_name = cpp.name(f.func)
                impl_name = f"at::native::{cpp_name}"
            else:
                assert dispatch is not None
                impl_name = f"at::native::{f.dispatch[dispatch]}"

            args_exprs_str = ', '.join(map(lambda a: a.name, args))

            # See Note [Byte-for-byte compatibility]
            # (return void_func() is valid C++)
            return_kw = "    return "
            if returns_type == "void":
                return_kw = " "

            cuda_guard = ""
            if dispatch is None or 'CUDA' in dispatch or 'Vulkan' == dispatch:
                self_args = (a for a in f.func.arguments if a.name == "self")

                # There is precedence for which argument we use to do
                # device guard.  This describes the precedence order.
                candidate_args = itertools.chain(self_args, f.func.out_arguments, f.func.arguments)

                # Only tensor like arguments are eligible
                device_of = next((f'{a.name}' for a in candidate_args if a.type.is_tensor_like()), None)

                # See Note [Byte-for-byte compatibility]
                # I wasn't able to figure out the internal logic for
                # these device guards
                if str(f.func.name) == "_thnn_fused_lstm_cell_backward":
                    device_of = "cx"
                elif str(f.func.name) == "_thnn_differentiable_lstm_cell_backward":
                    device_of = "input_gates"

                has_tensor_options = any(isinstance(a.argument, TensorOptionsArguments) for a in args)

                # TODO: There is probably a simpler version of this that
                # works just as well.
                if f.device_guard and (dispatch is None or 'Vulkan' == dispatch) and has_tensor_options:
                    cuda_guard = """\
    const DeviceGuard device_guard(options.device());
"""
                    # See Note [Byte-for-byte compatibility]
                    if dispatch is not None:
                        cuda_guard = f"\n{cuda_guard}"
                elif f.device_guard and dispatch is not None and 'CUDA' in dispatch and has_tensor_options:
                    cuda_guard = """\
    globalContext().lazyInitCUDA();
    const DeviceGuard device_guard(options.device());
"""
                elif f.device_guard and device_of is not None:
                    cuda_guard = f"""\
    const OptionalDeviceGuard device_guard(device_of({device_of}));
"""
                    # See Note [Byte-for-byte compatibility]
                    if dispatch is not None:
                        cuda_guard = f"\n{cuda_guard}"
                else:
                    cuda_guard = """\
    // DeviceGuard omitted
"""
                    # See Note [Byte-for-byte compatibility]
                    if dispatch is not None:
                        cuda_guard = f"\n{cuda_guard}"

            return [f"""\
{returns_type} {name}({args_str}) {{
{cuda_guard}{return_kw}{impl_name}({args_exprs_str});
}}
"""]

        elif target is Target.REGISTRATION:
            assert returns_type == dispatcher.returns_type(f.func.returns)
            dispatcher_args = dispatcher.arguments(f.func)
            dispatcher_args_types_str = ', '.join(map(lambda a: a.type, dispatcher_args))
            if dispatch is None:
                type_name = f'TypeDefault::{name}'
            else:
                type_name = f'{dispatch}Type::{name}'

            # def registration only happens in TypeDefault
            def_registration = ""
            if dispatch is None:
                def_registration = f'm.def("{f.func}");\n'

            impl_registration = ""
            if not def_only and not f.manual_kernel_registration and (dispatch is not None or f.dispatch is None):
                # Figure out which signature the function is
                if local.use_c10_dispatcher_full():
                    # See Note [Byte-for-byte compatibility]
                    if dispatch is not None:
                        nl = "\n"
                    else:
                        nl = ""

                    payload = f"c10::impl::hacky_wrapper_for_legacy_signatures<{returns_type} ({dispatcher_args_types_str})>({nl}TORCH_FN({type_name}))"

                else:
                    payload = f"torch::CppFunction::makeUnboxedOnly(&{type_name})"

                # Annotate it with dispatch information if necessary
                if dispatch is not None:
                    payload = f"torch::dispatch(DispatchKey::{dispatch},\n{payload})\n"

                impl_registration = f'm.impl("{f.func.name}",\n{payload});\n'

            return [f"{def_registration}{impl_registration}"]
        else:
            assert_never(target)

    return func

def compute_function(*, target: Target) -> Callable[[NativeFunction], List[str]]:
    def go(f: NativeFunction) -> List[str]:
        if f.manual_kernel_registration:
            return []
        if Variant.function not in f.variants:
            return []

        name = cpp.name(f.func)

        cpp_returns_type = cpp.returns_type(f.func.returns)
        cpp_args = cpp.arguments(f.func)
        cpp_args_str = ', '.join(map(str, cpp_args))

        if target is Target.DECLARATION:
            return [f"CAFFE2_API {cpp_returns_type} {name}({cpp_args_str});"]

        assert target is Target.DEFINITION

        dispatcher_exprs = dispatcher.cpparguments_exprs(cpp_args)
        cpp_args_str_no_default = ', '.join(map(lambda a: a.str_no_default(), cpp_args))
        dispatcher_returns_type = dispatcher.returns_type(f.func.returns)
        dispatcher_types_str = ', '.join(map(lambda a: a.type, dispatcher_exprs))
        dispatcher_exprs_str = ', '.join(map(lambda a: a.expr, dispatcher_exprs))

        return [f"""
// aten::{f.func}
{cpp_returns_type} {name}({cpp_args_str_no_default}) {{
#ifdef USE_STATIC_DISPATCH
#else
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatcher_returns_type} ({dispatcher_types_str})>();
    return op.call({dispatcher_exprs_str});
#endif
}}
"""]
    return go

def compute_tensor_method(*, target: Target) -> Callable[[NativeFunction], List[str]]:
    def go(f: NativeFunction) -> List[str]:
        if Variant.method not in f.variants:
            return []

        assert not f.func.is_out_fn()
        assert len(f.func.arguments) > 0
        assert sum(a.name == 'self' for a in f.func.arguments) == 1

        name = cpp.name(f.func)
        cpp_returns_type = cpp.returns_type(f.func.returns)
        cpp_args = cpp.arguments(f.func, method=True)
        cpp_args_exclude_this = [a for a in cpp_args if not isinstance(a.argument, ThisArgument)]
        cpp_args_exclude_this_str = ', '.join(str(a) for a in cpp_args_exclude_this)

        if target is Target.DECLARATION:
            return [f"{cpp_returns_type} {name}({cpp_args_exclude_this_str}) const;"]

        assert target is Target.DEFINITION

        dispatcher_exprs = dispatcher.cpparguments_exprs(cpp_args)
        cpp_args_exclude_this_str_no_default = ', '.join(a.str_no_default() for a in cpp_args_exclude_this)
        dispatcher_returns_type = dispatcher.returns_type(f.func.returns)
        dispatcher_types_str = ', '.join(map(lambda a: a.type, dispatcher_exprs))
        dispatcher_exprs_str = ', '.join(map(lambda a: a.expr, dispatcher_exprs))

        return [f"""
// aten::{f.func}
{cpp_returns_type} Tensor::{name}({cpp_args_exclude_this_str_no_default}) const {{
#ifdef USE_STATIC_DISPATCH
#else
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatcher_returns_type} ({dispatcher_types_str})>();
    return op.call({dispatcher_exprs_str});
#endif
}}
"""]

    return go

def compute_aten_op(f: NativeFunction) -> List[str]:
    return [f'{{"aten::{f.func.name.name}", "{f.func.name.overload_name}"}},']

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

def compute_backend_select(*, target: Target) -> Callable[[NativeFunction], List[str]]:
    def go(f: NativeFunction) -> List[str]:
        if str(f.func.name.name).endswith('_like') or str(f.func.name.name).startswith('new_'):
            return []

        name = legacy_dispatcher.name(f.func)
        legacy_dispatcher_returns_type = legacy_dispatcher.returns_type(f.func.returns)
        legacy_dispatcher_args = legacy_dispatcher.arguments(f.func)

        if not any(isinstance(a.argument, TensorOptionsArguments) for a in legacy_dispatcher_args):
            return []

        legacy_dispatcher_tensor_args = [
            a for a in legacy_dispatcher_args
            if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()
        ]

        dispatcher_returns_type = dispatcher.returns_type(f.func.returns)
        dispatcher_args = dispatcher.arguments(f.func)
        dispatcher_exprs = dispatcher.legacydispatcherarguments_exprs(legacy_dispatcher_args)

        if target is Target.DEFINITION:
            # See Note [Byte-for-byte compatibility]
            # I don't think there's actually a good reason to generate
            # these two cases differently
            if legacy_dispatcher_tensor_args:
                compute_dk = f"""\
DispatchKeySet _dk_set = DispatchKeySet(options.computeDispatchKey()) | c10::detail::multi_dispatch_key_set({', '.join(a.name for a in legacy_dispatcher_tensor_args)});
  DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
  DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);"""
            else:
                compute_dk = "DispatchKey _dk = options.computeDispatchKey();"
            return [f"""\
// aten::{f.func}
{legacy_dispatcher_returns_type} {name}({', '.join(a.str_with_default() for a in legacy_dispatcher_args)}) {{
  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
    .typed<{dispatcher_returns_type} ({', '.join(a.type for a in dispatcher_args)})>();
  {compute_dk}
  return op.callWithDispatchKey(_dk, {', '.join(a.expr for a in dispatcher_exprs)});
}}
"""]
        elif target is Target.REGISTRATION:
            if local.use_c10_dispatcher_full():
                return [f"""m.impl("aten::{f.func.name}",
          c10::impl::hacky_wrapper_for_legacy_signatures<{dispatcher_returns_type} ({', '.join(a.type for a in dispatcher_args)})>(
            TORCH_FN({name})));"""]
            else:
                return [f"""m.impl_UNBOXED("aten::{f.func.name}", {name});"""]
        elif target is Target.DECLARATION:
            assert False
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
def dynamic_type(t: Type) -> str:
    # Note we don't use t.is_tensor_like() here because it would
    # also include Tensor[]
    if isinstance(t, OptionalType):
        return dynamic_type(t.elem)
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
            # See Note [Byte-for-byte compatibility]
            #
            # Check if it would conflict with an existing argument.
            # Downstream codegen assumes that return names and argument
            # names don't conflict with each other, so we disambiguate
            # (by adding a trailing _return) this case.  Notice that
            # historically, the collision check was buggy: it just did a
            # straight string contains test on the entirety of the
            # inputs part of the format string, meaning that it also
            # picked up occurrences of the argument name in the NAME of
            # the function, as well as substring occurrences of the name
            # in arguments.  We have simulated the old logic here...
            buggy_name_conflict = r.name in str(f.func.name) or \
                any(r.name in a.name for a in f.func.schema_order_arguments())
            # ... but a more correct version is simply
            # name_conflict = any(r.name == a.name for a in f.func.schema_order_arguments())
            if buggy_name_conflict and not f.func.is_out_fn():
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
        assert False
    elif isinstance(cpp_a.argument, Argument):
        return compute_argument_yaml(cpp_a.argument, schema_order=schema_order, kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name)

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
    # See Note [Byte-for-byte compatibility]
    # The default value of kwarg_only is False; this case exists for
    # byte-for-byte compatibility
    elif a.name in out_arg_set:
        arg['kwarg_only'] = False
    if a.name in out_arg_set:
        arg['output'] = True
        # See Note [Byte-for-byte compatibility]
        # This is probably a bug in the original implementation, where
        # the specification of allocate was not properly propagated to
        # the schema-order arguments.  In any case, this field
        # is redundant with the output field
        if not schema_order:
            arg['allocate'] = True
        # See Note [name and field_name]
        if a.name in name_to_field_name:
            arg['field_name'] = name_to_field_name[a.name]
    # Historically, booleans don't get their size recorded, because it
    # is already built into the cpp type (e.g., std::array<bool, 4>)
    if isinstance(a.type, ListType) and a.type.size is not None and str(a.type.elem) != 'bool':
        arg['size'] = a.type.size
    return arg

def compute_declaration_yaml(f: NativeFunction) -> List[object]:
    returns, name_to_field_name = compute_returns_yaml(f)

    # These sets are used to conveniently test if an argument is a
    # kwarg-only or out argument
    kwarg_only_set = set(a.name for a in f.func.kwarg_only_arguments)
    out_arg_set = set(a.name for a in f.func.out_arguments)

    cpp_args = cpp.arguments(f.func)
    arguments = [compute_cpp_argument_yaml(cpp_a, schema_order=False, kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name) for cpp_a in cpp_args]

    # See Note [Byte-for-byte compatibility]
    # NB: NOT actually schema order.  This is almost certainly a BUG.
    schema_order_jit_arguments = list(itertools.chain(f.func.arguments, f.func.out_arguments, f.func.kwarg_only_arguments))

    schema_order_arguments = [compute_argument_yaml(a, schema_order=True, kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name) for a in schema_order_jit_arguments]

    cpp_schema_order_types = [cpp.argument(a).type for a in schema_order_jit_arguments]
    cpp_returns = cpp.returns_type(f.func.returns)
    schema_order_cpp_signature = f"{cpp_returns} ({', '.join(cpp_schema_order_types)})"

    is_factory_method = any(isinstance(a.argument, TensorOptionsArguments) for a in cpp_args) \
        and Variant.method not in f.variants

    return [OrderedDict([
        ('name', cpp.name(f.func)),
        ('operator_name', str(f.func.name.name)),
        ('overload_name', str(f.func.name.overload_name)),
        ('use_c10_dispatcher', 'full' if f.use_c10_dispatcher_full else 'with_codegenerated_unboxing_wrapper'),
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
        ('abstract', f.dispatch is not None),
        ('device_guard', f.device_guard),
        ('with_gil', False),
        ('deprecated', False),
    ])]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           RUN IT ALL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser(description='Generate ATen source files')
parser.add_argument(
    '-s',
    '--source-path',
    help='path to source directory for ATen',
    default='.')
parser.add_argument(
    '-o',
    '--output-dependencies',
    help='output a list of dependencies into the given file and exit')
parser.add_argument(
    '-d', '--install_dir', help='output directory', default='ATen')
# TODO: remove this, it does nothing
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
    '--per_op_registration',
    action='store_true',
    help='group function registrations by op name and write to separate files; '
         'must also set --op_registration_whitelist param')
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

native_functions = parse_native_yaml('aten/src/ATen/native/native_functions.yaml')

TEMPLATE_PATH = "aten/src/ATen/templates"

def generated_comment(fn: str) -> str:
    return "@" + f"generated by aten/src/ATen/gen.py from {fn}"

extra_cuda_headers = '''\
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>'''

def write_file(fn: str, env_arg: Dict[str, Union[str, Sequence[str]]]) -> None:
    env = env_arg.copy()
    derived = ['Vulkan', 'CPU', 'CUDA', 'QuantizedCPU', 'QuantizedCUDA', 'MkldnnCPU']
    sparse_derived = ['SparseCPU', 'SparseCUDA']
    if fn in [f'{x}Type.h' for x in itertools.chain(derived, sparse_derived)]:
        template_fn = 'TypeDerived.h'
    elif fn in [f'{x}Type.cpp' for x in derived]:
        template_fn = 'TypeDerived.cpp'
    elif fn in [f'{x}Type.cpp' for x in sparse_derived]:
        template_fn = 'SparseTypeDerived.cpp'
    else:
        template_fn = fn
    env['generated_comment'] = generated_comment(template_fn)
    template = CodeTemplate.from_file(os.path.join(TEMPLATE_PATH, template_fn))
    if fn in ['TensorBody.h', 'ATenOpList.cpp', 'TensorMethods.cpp']:
        out_path = f'build/aten/src/ATen_new/core/{fn}'
    else:
        out_path = f'build/aten/src/ATen_new/{fn}'
    with open(out_path, 'w') as f:
        f.write(template.substitute(env))

backends = ["CPU", "SparseCPU", "MkldnnCPU", "CUDA", "SparseCUDA", "QuantizedCPU", "QuantizedCUDA"]
if options.vulkan:
    backend.append("Vulkan")
if options.backend_whitelist:
    backends = [b for b in backends if b in options.backend_whitelist]

for dispatch in backends:
    write_file(f'{dispatch}Type.h', {
        'Type': f'{dispatch}Type',
        'extra_cuda_headers': extra_cuda_headers if 'CUDA' in dispatch else '',  # TODO: remove this
        'type_derived_method_declarations': concatmap_nf(compute_type_method(dispatch, target=Target.DECLARATION, op_registration_whitelist=op_registration_whitelist), native_functions),
    })
    write_file(f'{dispatch}Type.cpp', {
        'Type': f'{dispatch}Type',
        'extra_cuda_headers': extra_cuda_headers if 'CUDA' in dispatch else '',  # TODO: remove this
        'storage_tensor_headers': '#include <c10/core/TensorImpl.h>',  # TODO: remove this
        'Generator': f'{dispatch.replace("Quantized", "").replace("Sparse", "").replace("Mkldnn", "")}GeneratorImpl' if dispatch != 'Vulkan' else 'CPUGeneratorImpl',  # TODO: remove this
        'legacy_th_headers': f'#include <ATen/LegacyTHFunctions{dispatch}.h>' if dispatch in ["CPU", "CUDA"] else "",
        'Backend': dispatch,
        'type_derived_method_definitions': concatmap_nf(compute_type_method(dispatch, target=Target.DEFINITION, op_registration_whitelist=op_registration_whitelist), native_functions),
        'function_registrations': concatmap_nf(compute_type_method(dispatch, target=Target.REGISTRATION, op_registration_whitelist=op_registration_whitelist), native_functions) if options.per_op_registration is None else [],
    })

write_file('TypeDefault.h', {
    'type_method_declarations': concatmap_nf(compute_type_method(None, target=Target.DECLARATION, op_registration_whitelist=op_registration_whitelist), native_functions),
})
write_file('TypeDefault.cpp', {
    'type_method_definitions': concatmap_nf(compute_type_method(None, target=Target.DEFINITION, op_registration_whitelist=op_registration_whitelist), native_functions),
    'function_registrations': concatmap_nf(compute_type_method(None, target=Target.REGISTRATION, op_registration_whitelist=op_registration_whitelist), native_functions) if options.per_op_registration is None else [],
})
write_file('Functions.h', {
    'function_declarations': concatmap_nf(compute_function(target=Target.DECLARATION), native_functions),
})
write_file('Functions.cpp', {
    'function_definitions': concatmap_nf(compute_function(target=Target.DEFINITION), native_functions),
})
write_file('TensorBody.h', {
    'tensor_method_declarations': concatmap_nf(compute_tensor_method(target=Target.DECLARATION), native_functions),
})
write_file('TensorMethods.cpp', {
    'tensor_method_definitions': concatmap_nf(compute_tensor_method(target=Target.DEFINITION), native_functions),
})
write_file('ATenOpList.cpp', {
    'aten_ops': concatmap_nf(compute_aten_op, native_functions),
})
write_file('NativeFunctions.h', {
    'native_function_declarations': concatmap_nf(compute_native_function_declaration, native_functions),
})
write_file('BackendSelectRegister.cpp', {
    'backend_select_method_definitions': concatmap_nf(compute_backend_select(target=Target.DEFINITION), native_functions),
    'backend_select_function_registrations': concatmap_nf(compute_backend_select(target=Target.REGISTRATION), native_functions),
})

if options.force_schema_registration:
    schema_registrations = concatmap_nf(compute_type_method(None, target=Target.REGISTRATION, op_registration_whitelist=None, def_only=True), native_functions)
    # See Note [Byte-for-byte compatibility]
    schema_registrations.sort()
    write_file('SchemaRegister.cpp', {
        'schema_registrations': schema_registrations,
    })

def gen_per_op_registration_filename(opname: str) -> str:
    return 'pt_op_register_{}.cpp'.format(opname.replace(':', '-'))
if options.per_op_registration:
    if op_registration_whitelist is None:
        raise Exception("Must set --op_registration_whitelist for per-op registration.")
    base_fn = 'PerOpRegistration.cpp'
    template = CodeTemplate.from_file(os.path.join(TEMPLATE_PATH, base_fn))
    # First, group all native functions by unoverloaded operator name
    grouped_functions : DefaultDict[str, List[NativeFunction]] = DefaultDict(list)
    for f in native_functions:
        grouped_functions[f"aten::{f.func.name.name}"].append(f)
    extra_headers = []
    for b in backends:
        extra_headers.append(f'#include <ATen/{b}Type.h>')
    # Next, generate registration for each one
    for name in op_registration_whitelist:
        fs = grouped_functions[name]
        registrations = []
        for dispatch in itertools.chain([None], backends):
            # or you could pass in op_registration_whitelist, it doesn't
            # matter!
            # NB: Use of compute_type_method here is kind of an abuse
            registrations.extend(concatmap_nf(compute_type_method(dispatch, target=Target.REGISTRATION, op_registration_whitelist=None), fs))
        fn = gen_per_op_registration_filename(name)
        with open(os.path.join('build/aten/src/ATen_new', fn), 'w') as fil:
            fil.write(template.substitute({
                'generated_comment': generated_comment(base_fn),
                'extra_headers': extra_headers,
                'function_registrations': registrations,
            }))

with open('build/aten/src/ATen_new/Declarations.yaml', 'w') as fil:
    fil.write(format_yaml(concatmap_nf(compute_declaration_yaml, native_functions)))
