import sys
import os
import contextlib
import textwrap
import re
import pprint
import itertools
from typing import List, Sequence, Dict, Optional, Iterator, Tuple, Set, NoReturn, Callable, Union, Any
import yaml
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict

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
# I went into this rewrite with some goals:
#
# - Completely excise all legacy TH handling.  Declarations.cwrap isn't
#   a thing.  Preprocessing declarations isn't a thing.
#   native_functions.yaml is the only place where we get information
#   about operators.
#
# - Strict mypy typechecking.  You can typecheck this file using
#   `mypy -config mypy-strict.ini aten/src/ATen/gen_cpu.py` and
#   this will enforce that everything is annotated.
#
# - Better data modeling.  See tools.codegen.model for more details.
#
# Some non-goals:
#
# - Change native_functions.yaml format.  One step at a time!
#
# The general structure:
#
# - We define a lot of dataclasses to represent all of the various
#   semantic entities in native_functions.yaml (schema! types!)
#   These classes come with parsing and pretty-printing functionality.
#
# - We parse native_functions.yaml into our dataclasses
#
# - We do code generation on it (under construction!)
#


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           PROCESSING
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
        # TODO: this does the wrong thing with KeyErorr
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

native_functions = parse_native_yaml('aten/src/ATen/native/native_functions.yaml')

def map_native_functions(func: Callable[[NativeFunction], Union[Optional[str], Sequence[str]]]) -> List[str]:
    rs: List[str] = []
    for f in native_functions:
        with context(f'in {f.loc}:\n  {f.func}'):
            with local.parametrize(
                use_c10_dispatcher_full=f.use_c10_dispatcher_full,
                hack_const_mutable_self=str(f.func.name) in ["set_data", "retain_grad"],
            ):
                r = func(f)
            if r is None:
                continue
            elif isinstance(r, str):
                rs.append(r)
            else:
                rs.extend(r)
    return rs

# pprint.pprint([dataclasses.asdict(f) for f in native_functions])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

Target = Enum('Target', ('DEFINITION', 'DECLARATION', 'REGISTRATION'))

def compute_type_method(dispatch: Optional[str], *, target: Target) -> Callable[[NativeFunction], Optional[str]]:
    def func(f: NativeFunction) -> Optional[str]:
        if dispatch is not None:
            if f.dispatch is None or dispatch not in f.dispatch:
                return None
        else:
            if f.dispatch is not None and target is not Target.REGISTRATION:
                return None

        name = legacy_dispatcher.name(f.func)
        returns_type = legacy_dispatcher.returns_type(f.func.returns)
        args = legacy_dispatcher.arguments(f.func)
        args_str = ', '.join(map(str, args))

        if target is Target.DECLARATION:
            return f"{returns_type} {name}({args_str});"
        elif target is Target.DEFINITION:
            # TODO: refactor
            if f.dispatch is None:
                cpp_name = cpp.name(f.func)
                impl_name = f"at::native::{cpp_name}"
            else:
                assert dispatch is not None  # FIXME
                impl_name = f"at::native::{f.dispatch[dispatch]}"

            args_exprs_str = ', '.join(map(lambda a: a.name, args))

            # TODO: This isn't actually necessary, just for compat
            return_kw = "    return "
            if returns_type == "void":
                return_kw = " "  # bit for bit compat

            cuda_guard = ""
            if dispatch is None or 'CUDA' in dispatch:
                self_args = (a for a in f.func.arguments if a.name == "self")
                candidate_args = itertools.chain(self_args, f.func.out_arguments, f.func.arguments)
                # this is pretty funky, ngl
                device_of = next((f'{a.name}' for a in candidate_args if a.type.is_tensor_like()), None)
                # DON'T ASK ME WHY
                if str(f.func.name) == "_thnn_fused_lstm_cell_backward":
                    device_of = "cx"
                elif str(f.func.name) == "_thnn_differentiable_lstm_cell_backward":
                    device_of = "input_gates"
                has_tensor_options = any(isinstance(a.argument, TensorOptionsArguments) for a in args)
                if f.device_guard and dispatch is None and has_tensor_options:
                    cuda_guard = f"""\
    const DeviceGuard device_guard(options.device());
"""
                elif f.device_guard and dispatch is not None and 'CUDA' in dispatch and has_tensor_options:
                    cuda_guard = f"""\
    globalContext().lazyInitCUDA();
    const DeviceGuard device_guard(options.device());
"""
                elif f.device_guard and device_of is not None:
                    cuda_guard = f"""\
    const OptionalDeviceGuard device_guard(device_of({device_of}));
"""
                    if dispatch is not None:
                        cuda_guard = f"\n{cuda_guard}"
                else:
                    cuda_guard = """\
    // DeviceGuard omitted
"""
                    if dispatch is not None:
                        cuda_guard = f"\n{cuda_guard}"

            return f"""\
{returns_type} {name}({args_str}) {{
{cuda_guard}{return_kw}{impl_name}({args_exprs_str});
}}
"""

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
            if dispatch == None:
                def_registration = f'm.def("{f.func}");\n'

            impl_registration = ""
            if not f.manual_kernel_registration and (dispatch is not None or f.dispatch is None):
                # Figure out which signature the function is
                if local.use_c10_dispatcher_full():
                    # the price we pay for -w clean compat; newlines
                    # must be in right places
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

            return f"{def_registration}{impl_registration}"
        else:
            assert_never(target)

    return func

def compute_function(*, target: Target) -> Callable[[NativeFunction], Optional[str]]:
    def go(f: NativeFunction) -> Optional[str]:
        if f.manual_kernel_registration:
            return None
        if Variant.function not in f.variants:
            return None

        name = cpp.name(f.func)

        cpp_returns_type = cpp.returns_type(f.func.returns)
        cpp_args = cpp.arguments(f.func)
        cpp_args_str = ', '.join(map(str, cpp_args))

        if target is Target.DECLARATION:
            return f"CAFFE2_API {cpp_returns_type} {name}({cpp_args_str});"

        assert target is Target.DEFINITION

        dispatcher_exprs = dispatcher.cpparguments_exprs(cpp_args)
        cpp_args_str_no_default = ', '.join(map(lambda a: a.str_no_default(), cpp_args))
        dispatcher_returns_type = dispatcher.returns_type(f.func.returns)
        dispatcher_types_str = ', '.join(map(lambda a: a.type, dispatcher_exprs))
        dispatcher_exprs_str = ', '.join(map(lambda a: a.expr, dispatcher_exprs))

        return f"""
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
"""
    return go

def compute_tensor_method(*, target: Target) -> Callable[[NativeFunction], Optional[str]]:
    def go(f: NativeFunction) -> Optional[str]:
        if Variant.method not in f.variants:
            return None

        assert not f.func.is_out_fn()
        assert len(f.func.arguments) > 0
        assert sum(a.name == 'self' for a in f.func.arguments) == 1

        name = cpp.name(f.func)
        cpp_returns_type = cpp.returns_type(f.func.returns)
        cpp_args = cpp.arguments(f.func, method=True)
        cpp_args_exclude_this = [a for a in cpp_args if not isinstance(a.argument, ThisArgument)]
        cpp_args_exclude_this_str = ', '.join(str(a) for a in cpp_args_exclude_this)

        if target is Target.DECLARATION:
                return f"{cpp_returns_type} {name}({cpp_args_exclude_this_str}) const;"

        assert target is Target.DEFINITION

        dispatcher_exprs = dispatcher.cpparguments_exprs(cpp_args)
        cpp_args_exclude_this_str_no_default = ', '.join(a.str_no_default() for a in cpp_args_exclude_this)
        dispatcher_returns_type = dispatcher.returns_type(f.func.returns)
        dispatcher_types_str = ', '.join(map(lambda a: a.type, dispatcher_exprs))
        dispatcher_exprs_str = ', '.join(map(lambda a: a.expr, dispatcher_exprs))

        return f"""
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
"""

    return go

def compute_aten_op(f: NativeFunction) -> Optional[str]:
    return f'{{"aten::{f.func.name.name}", "{f.func.name.overload_name}"}},'


def compute_native_function_declaration(f: NativeFunction) -> Sequence[str]:
    # Order wobbling
    if f.dispatch is None:
        ns = [cpp.name(f.func)]
    else:
        ns = list(f.dispatch.values())

    rs = []
    seen = set()
    for n in ns:
        if n in seen:
            continue
        if "legacy::" in n:
            continue
        seen.add(n)
        returns_type = legacy_dispatcher.returns_type(f.func.returns)
        args = legacy_dispatcher.arguments(f.func)
        rs.append(f"CAFFE2_API {returns_type} {n}({', '.join(map(lambda a: a.str_with_default(), args))});");

    return rs

def compute_backend_select(*, target: Target) -> Callable[[NativeFunction], Optional[str]]:
    def go(f: NativeFunction) -> Optional[str]:
        if str(f.func.name.name).endswith('_like') or str(f.func.name.name).startswith('new_'):
            return None

        name = legacy_dispatcher.name(f.func)
        legacy_dispatcher_returns_type = legacy_dispatcher.returns_type(f.func.returns)
        legacy_dispatcher_args = legacy_dispatcher.arguments(f.func)

        if not any(isinstance(a.argument, TensorOptionsArguments) for a in legacy_dispatcher_args):
            return None

        legacy_dispatcher_tensor_args = [a for a in legacy_dispatcher_args if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()]

        dispatcher_returns_type = dispatcher.returns_type(f.func.returns)
        dispatcher_args = dispatcher.arguments(f.func)
        dispatcher_exprs = dispatcher.legacydispatcherarguments_exprs(legacy_dispatcher_args)

        if target is Target.DEFINITION:
            # TODO: Not really any good reason to special case here
            if legacy_dispatcher_tensor_args:
                compute_dk = f"""\
DispatchKeySet _dk_set = DispatchKeySet(options.computeDispatchKey()) | c10::detail::multi_dispatch_key_set({', '.join(a.name for a in legacy_dispatcher_tensor_args)});
  DispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
  DispatchKey _dk = c10::impl::dispatchTypeId(_dk_set, _dk_mask);"""
            else:
                compute_dk = "DispatchKey _dk = options.computeDispatchKey();"
            return f"""\
// aten::{f.func}
{legacy_dispatcher_returns_type} {name}({', '.join(a.str_with_default() for a in legacy_dispatcher_args)}) {{
  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
    .typed<{dispatcher_returns_type} ({', '.join(a.type for a in dispatcher_args)})>();
  {compute_dk}
  return op.callWithDispatchKey(_dk, {', '.join(a.expr for a in dispatcher_exprs)});
}}
"""
        elif target is Target.REGISTRATION:
            if local.use_c10_dispatcher_full():
                return f"""m.impl("aten::{f.func.name}",
          c10::impl::hacky_wrapper_for_legacy_signatures<{dispatcher_returns_type} ({', '.join(a.type for a in dispatcher_args)})>(
            TORCH_FN({name})));"""
            else:
                return f"""m.impl_UNBOXED("aten::{f.func.name}", {name});"""
        elif target is Target.DECLARATION:
            assert False
        else:
            assert_never(target)
    return go


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           WRITING
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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
    derived = ['CPU', 'CUDA', 'QuantizedCPU', 'QuantizedCUDA', 'MkldnnCPU']
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

for dispatch in ["CPU", "CUDA", "QuantizedCPU", "QuantizedCUDA", "MkldnnCPU", "SparseCPU", "SparseCUDA"]:
    write_file(f'{dispatch}Type.h', {
        'Type': f'{dispatch}Type',
        'extra_cuda_headers': extra_cuda_headers if 'CUDA' in dispatch else '',  # TODO: remove this
        'type_derived_method_declarations': map_native_functions(compute_type_method(dispatch, target=Target.DECLARATION)),
    })
    write_file(f'{dispatch}Type.cpp', {
        'Type': f'{dispatch}Type',
        'extra_cuda_headers': extra_cuda_headers if 'CUDA' in dispatch else '',  # TODO: remove this
        'storage_tensor_headers': '#include <c10/core/TensorImpl.h>', # TODO: remove this
        'Generator': f'{dispatch.replace("Quantized", "").replace("Sparse", "").replace("Mkldnn", "")}GeneratorImpl', # TODO: remove this
        'legacy_th_headers': f'#include <ATen/LegacyTHFunctions{dispatch}.h>' if dispatch in ["CPU", "CUDA"] else "",
        'Backend': dispatch,
        'type_derived_method_definitions': map_native_functions(compute_type_method(dispatch, target=Target.DEFINITION)),
        'function_registrations': map_native_functions(compute_type_method(dispatch, target=Target.REGISTRATION)),
    })
write_file('TypeDefault.h', {
    'type_method_declarations': map_native_functions(compute_type_method(None, target=Target.DECLARATION)),
})
write_file('TypeDefault.cpp', {
    'type_method_definitions': map_native_functions(compute_type_method(None, target=Target.DEFINITION)),
    'function_registrations': map_native_functions(compute_type_method(None, target=Target.REGISTRATION)),
})
write_file('Functions.h', {
    'function_declarations': map_native_functions(compute_function(target=Target.DECLARATION)),
})
write_file('Functions.cpp', {
    'function_definitions': map_native_functions(compute_function(target=Target.DEFINITION)),
})
write_file('TensorBody.h', {
    'tensor_method_declarations': map_native_functions(compute_tensor_method(target=Target.DECLARATION)),
})
write_file('TensorMethods.cpp', {
    'tensor_method_definitions': map_native_functions(compute_tensor_method(target=Target.DEFINITION)),
})
write_file('ATenOpList.cpp', {
    'aten_ops': map_native_functions(compute_aten_op),
})
write_file('NativeFunctions.h', {
    'native_function_declarations': map_native_functions(compute_native_function_declaration),
})
write_file('BackendSelectRegister.cpp', {
    'backend_select_method_definitions': map_native_functions(compute_backend_select(target=Target.DEFINITION)),
    'backend_select_function_registrations': map_native_functions(compute_backend_select(target=Target.REGISTRATION)),
})


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

# this makes no fucking sense
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

def dynamic_type(t: Type) -> str:
    if isinstance(t, OptionalType):
        return dynamic_type(t.elem)
    if str(t) == 'Tensor':
        return 'Tensor'
    return cpp.argumenttype_type(t, mutable=False)

def compute_declarations_yaml() -> object:
    rs: List[object] = []
    for f in native_functions:
        with context(f'in {f.loc}:\n  {f.func}'):
            with local.parametrize(
                use_c10_dispatcher_full=f.use_c10_dispatcher_full,
                hack_const_mutable_self=str(f.func.name) in ["set_data", "retain_grad"],
            ):
                dispatcher_args = dispatcher.arguments(f.func)
                dispatcher_returns = dispatcher.returns_type(f.func.returns)
                cpp_args = cpp.arguments(f.func)

                kwarg_only_set = set(a.name for a in f.func.kwarg_only_arguments)
                out_arg_set = set(a.name for a in f.func.out_arguments)
                args_set = set(a.name for a in f.func.schema_order_arguments())

                field_name_prop_set: Dict[str, str] = {}
                if f.func.name.name.inplace:
                    if len(f.func.returns) == 1:
                        returns = [
                            {
                                'dynamic_type': 'Tensor',
                                'name': 'self',
                                'type': 'Tensor &',
                            }
                        ]
                    else:
                        assert len(f.func.returns) == 0
                        returns = []
                else:
                    result = 'result'
                    if f.func.is_out_fn():
                        result = 'out'
                    returns = []
                    for i, r in enumerate(f.func.returns):
                        if f.func.is_out_fn():
                            name = f.func.out_arguments[i].name
                        elif r.name:
                            # Yes I can't make this up
                            if (r.name in args_set or r.name in str(f.func.name) or any(r.name in x for x in args_set)) and not f.func.is_out_fn():
                                name = f'{r.name}_return'
                            else:
                                name = r.name
                        else:
                            name = result if len(f.func.returns) == 1 else f'{result}{i}'
                        ret = {
                            'dynamic_type': dynamic_type(r.type),
                            'name': name,
                            'type': cpp.return_type(r),
                        }
                        if r.name:
                            ret['field_name'] = r.name
                            if r.annotation is not None:
                                field_name_prop_set[str(r.annotation)] = r.name
                        returns.append(ret)

                def make_argument(a: Argument, *, schema_order: bool) -> object:
                    arg: Dict[str, object] = {
                        'annotation': str(a.annotation) if a.annotation else None,
                        'dynamic_type': dynamic_type(a.type),
                        'is_nullable': a.type.is_nullable(),
                        'name': a.name,
                        'type': cpp.argument_type(a),
                    }
                    # TODO: type confusion
                    if a.default is not None:
                        arg['default'] = pythonify_default(cpp.default_expr(a.default, a.type))
                    if a.name in kwarg_only_set:
                        arg['kwarg_only'] = True
                    elif a.name in out_arg_set:
                        arg['kwarg_only'] = False  # LOL
                    if a.name in out_arg_set:
                        arg['output'] = True
                        if not schema_order:
                            arg['allocate'] = True
                        if a.annotation is not None and str(a.annotation) in field_name_prop_set:
                            arg['field_name'] = field_name_prop_set[str(a.annotation)]
                    if isinstance(a.type, ListType) and a.type.size is not None and str(a.type.elem) != 'bool':
                        arg['size'] = a.type.size
                    return arg

                arguments = []
                for cpp_a in cpp_args:
                    assert not isinstance(cpp_a.argument, ThisArgument)
                    if isinstance(cpp_a.argument, Argument):
                        arguments.append(make_argument(cpp_a.argument, schema_order=False))
                    else:
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
                        arguments.append(arg)

                buggy_schema_order_args = list(itertools.chain(f.func.arguments, f.func.out_arguments, f.func.kwarg_only_arguments))
                schema_order_arguments = []
                # NB: NOT actually schema order LOLOLOL
                # for a in f.func.schema_order_arguments():
                for a in buggy_schema_order_args:
                    schema_order_arguments.append(make_argument(a, schema_order=True))
                schema_order_args = [cpp.argument(a) for a in buggy_schema_order_args]
                schema_order_cpp_signature = f"{dispatcher_returns} ({', '.join(a.type for a in schema_order_args)})"

                method_of = ['Type']
                if Variant.method in f.variants:
                    method_of.append('Tensor')
                if Variant.function in f.variants:
                    method_of.append('namespace')

                rs.append(OrderedDict([
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
                    ('method_of', method_of),
                    ('mode', 'native'),
                    ('python_module', '' if f.python_module is None else f.python_module),
                    ('returns', returns),
                    ('inplace', f.func.name.name.inplace),
                    ('is_factory_method', any(isinstance(a.argument, TensorOptionsArguments) for a in cpp_args) and Variant.method not in f.variants),
                    ('abstract', f.dispatch is not None),
                    ('device_guard', f.device_guard),
                    ('with_gil', False),
                    ('deprecated', False),
                ]))
    return rs

with open('build/aten/src/ATen_new/Declarations.yaml', 'w') as f:
    f.write(format_yaml(compute_declarations_yaml()))
