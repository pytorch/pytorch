import sys
import os
import contextlib
import textwrap
import re
import pprint
import itertools
from typing import List, Sequence, Dict, Optional, Iterator, Tuple, Set, NoReturn, Callable, Union
import yaml
from dataclasses import dataclass

# Reusing CodeTemplate from existing codegen
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../aten/src/ATen"))
from code_template import CodeTemplate

from tools.codegen.model import *

try:
    # use faster C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore

# A little trick from https://github.com/python/mypy/issues/6366
# for getting mypy to do exhaustiveness checking
def _assert_never(x: NoReturn) -> NoReturn:
    assert False, "Unhandled type: {}".format(type(x).__name__)

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

def map_native_functions(func: Callable[[NativeFunction], Optional[str]]) -> List[str]:
    rs: List[str] = []
    for f in native_functions:
        with context(f'in {f.loc}:\n  {f.func}'):
            r = func(f)
            if r is None:
                continue
            rs.append(r)
    return rs

# pprint.pprint([dataclasses.asdict(f) for f in native_functions])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           CODE GENERATION
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# General concept: relate BINDERS to USES, generate TYPES with
# EXPRESSIONS

# The only conversions you need in old codegen:
#
#   C++ API to Dispatcher API (Functions.h)
#   C++ API to Legacy Dispatcher API (STATIC_DISPATCH)
#   Legacy Dispatcher API to Legacy Dispatcher API
#
#   Legacy dispatcher API is controlled by use_c10_dispatcher_full
#
# In ideal world, we would have
#
#   C++ API to Dispatcher API
#   Dispatcher API to Dispatcher API


# What users see in the public C++ API.  Multiple arguments may refer to
# a single formal
@dataclass(frozen=True)
class CppApiFormal:
    type: str
    name: str
    # Only used by the header, but we work it out in all cases anyway
    default: Optional[str]

    def str_no_default(self) -> str:
        return f"{self.type} {self.name}"

    def __str__(self) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{self.type} {self.name}{mb_default}"

# The "modern" unboxed dispatcher C++ API.  This is what you actually
# pass in as the types to the dispatcher calls itself.
@dataclass(frozen=True)
class DispatcherExpr:
    type: str
    expr: str

# Legacy dispatcher API.  Similar to the dispatcher, but things like
# optionals are handled differently.  To be deleted eventually.
# Triggered when you don't have use_c10_dispatcher_full
@dataclass(frozen=True)
class LegacyDispatcherExpr:
    type: str
    expr: str

# jit types which map to value types in c++ look the same whether or not
# their arguments or return types
def cpp_value_type(t: Type) -> Optional[str]:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            return None
        elif t.name == BaseTy.int:
            return 'int64_t'
        elif t.name == BaseTy.float:
            return 'double'
        elif t.name == BaseTy.str:
            return 'std::string'
        elif t.name in [BaseTy.bool, BaseTy.QScheme, BaseTy.Scalar,
                BaseTy.ScalarType, BaseTy.Generator, BaseTy.Storage,
                BaseTy.Layout, BaseTy.Device, BaseTy.MemoryFormat,
                BaseTy.Dimname, BaseTy.ConstQuantizerPtr]:
            # These C++ names coincidentally line up with their schema
            # names
            return t.name.name
        else:
            assert False, f"unsupported type: {t}"
    elif isinstance(t, OptionalType):
        elem = cpp_value_type(t.elem)
        if elem is None:
            return None
        return f"c10::optional<{elem}>"
    elif isinstance(t, ListType):
        if str(t.elem) == 'bool':
            assert t.size is not None
            return f"std::array<bool,{t.size}>"
        else:
            return None
    else:
        assert False, f"unrecognized type {repr(t)}"


def cpp_api_argument_type(t: Type, *, mutable: bool, use_c10_dispatcher_full: bool) -> str:
    r = cpp_value_type(t)
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                return 'Tensor &'
            else:
                return 'const Tensor &'
        else:
            assert False, f"base type should have been value type {t}"
    elif isinstance(t, OptionalType):
        if str(t.elem) == 'Tensor':
            if mutable:
                return 'Tensor &'  # TODO: fix this discrepancy
            else:
                if use_c10_dispatcher_full:
                    return 'const c10::optional<Tensor>&'
                else:
                    return 'const Tensor &'
        elem = cpp_api_argument_type(t.elem, mutable=mutable, use_c10_dispatcher_full=use_c10_dispatcher_full)
        return f"c10::optional<{elem}>"
    elif isinstance(t, ListType):
        # TODO: remove these special cases
        if str(t.elem) == 'int':
            return "IntArrayRef"
        elif str(t.elem) == 'Tensor':
            return "TensorList"
        elif not use_c10_dispatcher_full and str(t.elem) == 'Tensor?':
            return "TensorList"
        elif str(t.elem) == 'Dimname':
            return "DimnameList"
        elem = cpp_api_argument_type(t.elem, mutable=mutable, use_c10_dispatcher_full=use_c10_dispatcher_full)
        # TODO: explicitly qualify namespace here
        return f"ArrayRef<{elem}>"
    else:
        assert False, f"unrecognized type {repr(t)}"

def legacy_dispatcher_argument_type(t: Type, *, mutable: bool) -> str:
    if isinstance(t, OptionalType):
        if str(t.elem) == 'Tensor':
            if mutable:
                return 'Tensor &'
            else:
                return 'const Tensor &'
    elif isinstance(t, ListType):
        if str(t.elem) == 'Tensor?':
            return 'TensorList'
    return cpp_api_argument_type(t, mutable=mutable, use_c10_dispatcher_full=False)


def dispatcher_argument_type(t: Type, *, mutable: bool, use_c10_dispatcher_full: bool) -> str:
    # It would be OK to write special cases here
    if use_c10_dispatcher_full:
        return cpp_api_argument_type(t, mutable=mutable, use_c10_dispatcher_full=True)
    else:
        return legacy_dispatcher_argument_type(t, mutable=mutable)

def cpp_api_single_return_type(t: Type, *, mutable: bool) -> str:
    r = cpp_value_type(t)
    if r is not None:
        return r

    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                return 'Tensor &'
            else:
                return 'Tensor'
    elif isinstance(t, ListType):
        elem = cpp_api_single_return_type(t.elem, mutable=mutable)
        assert t.size is None, f"fixed size list returns not supported: {t}"
        return f"std::vector<{elem}>"

    assert False, f"unrecognized return type {t}"

def cpp_api_return_type(rs: Sequence[Return]) -> str:
    if len(rs) == 0:
        return 'void'
    elif len(rs) == 1:
        return cpp_api_single_return_type(rs[0].type, mutable=rs[0].is_write)
    else:
        args = ','.join([cpp_api_single_return_type(r.type, mutable=r.is_write) for r in rs])
        return f'std::tuple<{args}>'

def dispatcher_return_type(rs: Sequence[Return]) -> str:
    # At present, there is no difference. But there could be!
    return cpp_api_return_type(rs)

JIT_TO_CPP_DEFAULT = {
    'False': 'false',
    'True': 'true',
    'None': 'c10::nullopt',  # UGH this one is type directed
    'Mean': 'at::Reduction::Mean',
    '[]': '{}',
    '[0,1]': '{0,1}',  # TODO: stop special casing
    'contiguous_format': 'MemoryFormat::Contiguous',
}

def cpp_api_default(d: str, t: Type) -> str:
    if d == 'None' and str(t) == 'Tensor?':
        return '{}'
    return JIT_TO_CPP_DEFAULT.get(d, d)


def bind_simple_argument_cpp_api_to_dispatcher(a: Argument, *, use_c10_dispatcher_full: bool) -> Tuple[CppApiFormal, DispatcherExpr]:
    # Input from C++ API format
    cpp_api_formal = CppApiFormal(
        type=cpp_api_argument_type(a.type, mutable=a.is_write, use_c10_dispatcher_full=use_c10_dispatcher_full),
        name=a.name,
        default=cpp_api_default(a.default, a.type) if a.default is not None else None,
    )
    # Output in format appropriate for dispatcher
    dispatcher_expr = DispatcherExpr(
        type=dispatcher_argument_type(a.type, mutable=a.is_write, use_c10_dispatcher_full=use_c10_dispatcher_full),
        expr=a.name,
    )
    return (cpp_api_formal, dispatcher_expr)


@dataclass
class TensorOptionsArguments:
    dtype: Argument
    layout: Argument
    device: Argument
    pin_memory: Argument

    def all(self) -> Sequence[Argument]:
        return [self.dtype, self.layout, self.device, self.pin_memory]


# one to many binding
def bind_tensor_options_cpp_api_to_dispatcher(
    ta: TensorOptionsArguments,
    *, use_c10_dispatcher_full: bool
) -> Tuple[CppApiFormal, Sequence[DispatcherExpr]]:
    default = None
    if all(a.default == "None" for a in ta.all()):
        default = '{}'
    elif ta.dtype.default == "long":
        default = 'at::kLong'  # TODO: this is wrong

    cpp_api_formal = CppApiFormal(
        type='const TensorOptions &',
        name='options',
        default=default,
    )
    if not use_c10_dispatcher_full:
        dispatcher_exprs = [
            DispatcherExpr(type='const TensorOptions &', expr='options',)
        ]
    else:
        # TODO: Beef up defaulting to fix tril bug
        def arg_type(ty: Type) -> str:
            return dispatcher_argument_type(ty, mutable=False, use_c10_dispatcher_full=True)
        dispatcher_exprs = [
            DispatcherExpr(type=arg_type(ta.dtype.type), expr='optTypeMetaToScalarType(options.dtype_opt())'),
            DispatcherExpr(type=arg_type(ta.layout.type), expr='options.layout_opt()'),
            DispatcherExpr(type=arg_type(ta.device.type), expr='options.device_opt()'),
            DispatcherExpr(type=arg_type(ta.pin_memory.type), expr='options.pinned_memory_opt()'),  # weird discrep
        ]
    return (cpp_api_formal, dispatcher_exprs)


def bind_arguments_cpp_api_to_dispatcher(
    func: FunctionSchema, *, use_c10_dispatcher_full: bool
) -> Tuple[Sequence[CppApiFormal], Sequence[DispatcherExpr]]:

    # group up arguments for tensor options
    args: List[Union[Argument, TensorOptionsArguments]] = []
    args.extend(func.out_arguments)
    args.extend(func.arguments)
    i = 0
    def pred(name: str, ty: Type) -> Callable[[Argument], bool]:
        return lambda a: a.name == name and a.type in [ty, OptionalType(ty)]
    predicates = [  # order matters
        pred('dtype', Type.parse('ScalarType')),
        pred('layout', Type.parse('Layout')),
        pred('device', Type.parse('Device')),
        pred('pin_memory', Type.parse('bool')),
    ]
    while i < len(func.kwarg_only_arguments):
        # If there is enough space...
        if i <= len(func.kwarg_only_arguments) - len(predicates):
            # And the next len(predicates) arguments look like TensorOptions arguments
            if all(p(a) for p, a in zip(predicates, func.kwarg_only_arguments[i:i+len(predicates)])):
                # Group them together as one argument
                args.append(TensorOptionsArguments(
                    dtype=func.kwarg_only_arguments[i],
                    layout=func.kwarg_only_arguments[i+1],
                    device=func.kwarg_only_arguments[i+2],
                    pin_memory=func.kwarg_only_arguments[i+3],
                ))
                i += len(predicates)
                continue
        args.append(func.kwarg_only_arguments[i])
        i += 1

    # process each group, computing the formal that binds the argument,
    # and the expression that lets us refer to each argument in the
    # dispatcher domain
    cpp_api_formals: List[CppApiFormal] = []
    dispatcher_exprs: List[DispatcherExpr] = []
    for a in args:
        if isinstance(a, Argument):
            formal, expr = bind_simple_argument_cpp_api_to_dispatcher(a, use_c10_dispatcher_full=use_c10_dispatcher_full)
            exprs: Sequence[DispatcherExpr] = [expr]
        elif isinstance(a, TensorOptionsArguments):
            formal, exprs = bind_tensor_options_cpp_api_to_dispatcher(a, use_c10_dispatcher_full=use_c10_dispatcher_full)
        else:
            _assert_never(a)
        cpp_api_formals.append(formal)
        dispatcher_exprs.extend(exprs)

    return (cpp_api_formals, dispatcher_exprs)



# When we interpret things as C++ types, there are a bunch of
# different modalities we have to consider
#
# - Return versus argument type
# - Mutable type (inplace, out argument)
# - Public API versus internal calling convention versus legacy calling
#   convention
#
# I'm not really sure how to structure this logic yet, but here is a
# sketch.  This function is ONLY correct for CPUType.h at the moment;
# I bet I am going to need another parameter before I'm done
def cpp_type(t: Type, *, mutable: bool, argument: bool, legacy_optional: bool) -> str:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            if mutable:
                return 'Tensor &'
            else:
                if argument:
                    return 'const Tensor &'
                else:
                    return 'Tensor'
        elif t.name == BaseTy.int:
            return 'int64_t'
        elif t.name == BaseTy.float:
            return 'double'
        elif t.name == BaseTy.str:
            return 'std::string'
        elif t.name in [BaseTy.bool, BaseTy.QScheme, BaseTy.Scalar,
                BaseTy.ScalarType, BaseTy.Generator, BaseTy.Storage,
                BaseTy.Layout, BaseTy.Device, BaseTy.MemoryFormat,
                BaseTy.Dimname, BaseTy.ConstQuantizerPtr]:
            # These C++ names coincidentally line up with their schema
            # names
            return t.name.name
        else:
            assert False, f"unsupported type: {t}"
    elif isinstance(t, OptionalType):
        # TODO: these arguments are smoothed over by the hacky wrapper
        if argument and legacy_optional and str(t.elem) == 'Tensor':
            if mutable:
                return 'Tensor &'
            else:
                return 'const Tensor &'
        if argument and str(t.elem) == 'Tensor':
            if mutable:
                return 'Tensor &'  # WUUUT
            else:
                return 'const c10::optional<Tensor>&'
        elem = cpp_type(t.elem, mutable=mutable, argument=argument, legacy_optional=legacy_optional)
        return f"c10::optional<{elem}>"
    elif isinstance(t, ListType):
        if str(t.elem) == 'bool':
            assert t.size is not None
            return f"std::array<bool,{t.size}>"
        # TODO: remove this special case
        if str(t.elem) == 'int' and argument:
            return "IntArrayRef"
        elif str(t.elem) == 'Tensor' and argument:
            return "TensorList"
        elif str(t.elem) == 'Tensor?' and argument and legacy_optional:
            return "TensorList"
        elif str(t.elem) == 'Dimname' and argument:
            return "DimnameList"
        elem = cpp_type(t.elem, mutable=mutable, argument=argument, legacy_optional=legacy_optional)
        if argument:
            # TODO: explicitly qualify namespace here
            return f"ArrayRef<{elem}>"
        else:
            assert t.size is None, f"fixed size list returns not supported: {t}"
            return f"std::vector<{elem}>"
    else:
        assert False

def cpp_type_return(rs: Sequence[Return]) -> str:
    if len(rs) == 0:
        return 'void'
    elif len(rs) == 1:
        return cpp_type(rs[0].type, mutable=rs[0].is_write, argument=False, legacy_optional=False)
    else:
        args = ','.join([cpp_type(r.type, mutable=r.is_write, argument=False, legacy_optional=False) for r in rs])
        return f'std::tuple<{args}>'



# public = public c++ api
# dispatcher = dispatcher calling convention
# native = native functions convention (legacy, post wrapper)

# defaulting is handled in public, no where else

def public_function_name(func: FunctionSchema) -> str:
    name = str(func.name.name)
    if func.is_out_fn():
        name += '_out'
    return name

def type_method_name(func: FunctionSchema) -> str:
    name = str(func.name.name)
    # TODO: delete this!
    if func.is_out_fn():
        name += '_out'
    if func.name.overload_name:
        name += f'_{func.name.overload_name}'
    return name

def compute_type_method_declaration(dispatch: Optional[str]) -> Callable[[NativeFunction], Optional[str]]:
    def func(f: NativeFunction) -> Optional[str]:
        if dispatch is not None:
            if f.dispatch is None or dispatch not in f.dispatch:
                return None
        else:
            if f.dispatch is not None:
                return None

        name = type_method_name(f.func)
        cpp_return = cpp_type_return(f.func.returns)

        def format_arg(a: Argument) -> str:
            mutable = a.is_write
            # SPECIAL CASE
            # I'm pretty sure I know how the actual calculation is done:
            # if the mutable argument doesn't show up in the output, we
            # drop Tensor&.  But I don't want to code it.
            if str(f.func.name) in ["retain_grad", "set_data"]:
                mutable = False
            return f"{cpp_type(a.type, mutable=mutable, argument=True, legacy_optional=True)} {a.name}"
        cpp_args: List[str] = []
        cpp_args.extend(map(format_arg, f.func.out_arguments))
        cpp_args.extend(map(format_arg, f.func.arguments))

        cpp_kwarg_only = list(map(format_arg, f.func.kwarg_only_arguments))
        if f.func.tensor_options_info is not None:
            cpp_kwarg_only[f.func.tensor_options_info.slice()] = ['const TensorOptions & options']
        cpp_args.extend(cpp_kwarg_only)

        return f"{cpp_return} {name}({', '.join(cpp_args)});"
    return func

def public_argument(*, include_name: bool, include_default: bool, legacy_optional: bool) -> Callable[[Argument], str]:
    assert not (not include_name and include_default)
    def func(a: Argument) -> str:
        if include_default:
            default = f"={cpp_api_default(a.default, a.type)}" if a.default is not None else ""
        else:
            default = ""
        if include_name:
            name = f" {a.name}"
        else:
            name = ""
        # TODO: Always NO legacy optional
        ty = cpp_type(a.type, mutable=a.is_write, argument=True, legacy_optional=legacy_optional)
        return f"{ty}{name}{default}"
    return func


def compute_function(*, definition: bool) -> Callable[[NativeFunction], Optional[str]]:
    def go(f: NativeFunction) -> Optional[str]:
        if f.manual_kernel_registration:
            return None
        if Variant.function not in f.variants:
            return None

        name = public_function_name(f.func)

        return_type = cpp_api_return_type(f.func.returns)
        formals, exprs = bind_arguments_cpp_api_to_dispatcher(f.func, use_c10_dispatcher_full=f.use_c10_dispatcher_full)
        formals_str = ', '.join(map(str, formals))

        if not definition:
            return f"CAFFE2_API {return_type} {name}({formals_str});"

        formals_no_default_str = ', '.join(map(lambda a: a.str_no_default(), formals))
        dispatch_return_type = dispatcher_return_type(f.func.returns)
        dispatch_arg_types_str = ', '.join(map(lambda a: a.type, exprs))
        dispatch_args_str = ', '.join(map(lambda a: a.expr, exprs))

        return f"""
// aten::{f.func}
{return_type} {name}({formals_no_default_str}) {{
#ifdef USE_STATIC_DISPATCH
#else
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("aten::{f.func.name.name}", "{f.func.name.overload_name}")
        .typed<{dispatch_return_type} ({dispatch_arg_types_str})>();
    return op.call({dispatch_args_str});
#endif
}}
"""
    return go

def compute_tensor_method_declaration(f: NativeFunction) -> Optional[str]:
    if Variant.method not in f.variants:
        return None

    name = str(f.func.name.name)

    assert not f.func.is_out_fn()
    assert len(f.func.arguments) > 0
    assert sum(a.name == 'self' for a in f.func.arguments) == 1

    cpp_return = cpp_type_return(f.func.returns)

    format_arg = public_argument(legacy_optional=not f.use_c10_dispatcher_full, include_default=True, include_name=True)
    cpp_args: List[str] = []
    # self may occur in the middle of the argument list; we drop
    # it wherever it occurs
    cpp_args.extend(map(format_arg, filter(lambda a: a.name != 'self', f.func.arguments)))

    cpp_kwarg_only = list(map(format_arg, f.func.kwarg_only_arguments))
    if f.func.tensor_options_info is not None:
        cpp_kwarg_only[f.func.tensor_options_info.slice()] = ['const TensorOptions & options={}']
    cpp_args.extend(cpp_kwarg_only)

    return f"{cpp_return} {name}({', '.join(cpp_args)}) const;"



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
    if fn in ['CPUType.h', 'CUDAType.h']:
        template_fn = 'TypeDerived.h'
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

for dispatch in ["CPU", "CUDA"]:
    write_file(f'{dispatch}Type.h', {
        'Type': f'{dispatch}Type',
        'extra_cuda_headers': extra_cuda_headers if dispatch == 'CUDA' else '',  # TODO: remove this
        'type_derived_method_declarations': map_native_functions(compute_type_method_declaration(dispatch)),
    })
write_file('TypeDefault.h', {
    'type_method_declarations': map_native_functions(compute_type_method_declaration(None)),
})
write_file('Functions.h', {
    'function_declarations': map_native_functions(compute_function(definition=False)),
})
write_file('Functions.cpp', {
    'function_definitions': map_native_functions(compute_function(definition=True)),
})
write_file('TensorBody.h', {
    'tensor_method_declarations': map_native_functions(compute_tensor_method_declaration),
})
