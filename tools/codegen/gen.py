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

def map_native_functions(func: Callable[[NativeFunction], Optional[str]]) -> List[str]:
    rs: List[str] = []
    for f in native_functions:
        with context(f'in {f.loc}:\n  {f.func}'):
            with local.parametrize(use_c10_dispatcher_full=f.use_c10_dispatcher_full):
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

def compute_type_method_declaration(dispatch: Optional[str]) -> Callable[[NativeFunction], Optional[str]]:
    def func(f: NativeFunction) -> Optional[str]:
        if dispatch is not None:
            if f.dispatch is None or dispatch not in f.dispatch:
                return None
        else:
            if f.dispatch is not None:
                return None

        name = legacy_dispatcher.name(f.func)
        returns_type = legacy_dispatcher.returns_type(f.func.returns)
        args = legacy_dispatcher.arguments(f.func)
        return f"{returns_type} {name}({', '.join(map(str, args))});"
    return func

def compute_function(*, definition: bool) -> Callable[[NativeFunction], Optional[str]]:
    def go(f: NativeFunction) -> Optional[str]:
        if f.manual_kernel_registration:
            return None
        if Variant.function not in f.variants:
            return None

        name = cpp.name(f.func)

        cpp_returns_type = cpp.returns_type(f.func.returns)
        cpp_args = cpp.arguments(f.func)
        dispatcher_exprs = dispatcher.cpparguments_exprs(cpp_args)
        cpp_args_str = ', '.join(map(str, cpp_args))

        if not definition:
            return f"CAFFE2_API {cpp_returns_type} {name}({cpp_args_str});"

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

def compute_tensor_method_declaration(f: NativeFunction) -> Optional[str]:
    if Variant.method not in f.variants:
        return None

    assert not f.func.is_out_fn()
    assert len(f.func.arguments) > 0
    assert sum(a.name == 'self' for a in f.func.arguments) == 1

    name = cpp.name(f.func)
    cpp_returns_type = cpp.returns_type(f.func.returns)
    cpp_args = cpp.arguments(f.func, exclude_self=True)
    return f"{cpp_returns_type} {name}({', '.join(map(str, cpp_args))}) const;"



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
