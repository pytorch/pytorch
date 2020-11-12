# Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

import re
from typing import Optional, List

from tools.codegen.api.types import *
import tools.codegen.api.cpp as cpp
import tools.codegen.api.python as python
from tools.codegen.gen import with_native_function, parse_native_yaml, FileManager, mapMaybe
from tools.codegen.model import *

OPTIONAL_TYPE_PATTERN = re.compile(r"c10::optional<(.+)>")
TYPE_PATTERN = re.compile(r"(?:const\s+)?([A-Z]\w+)")

# Add 'at::' to types defined in ATen namespace, e.g. Tensor, TensorList, IntArrayRef and etc.
# TODO: maybe update the cpp argument API to take optional namespace argument?
def fully_qualified_type(argument_type: str) -> str:
    def maybe_optional_type(type: str, opt: bool) -> str:
        return f'c10::optional<{type}>' if opt else type

    opt_match = OPTIONAL_TYPE_PATTERN.match(argument_type)
    if opt_match:
        argument_type = argument_type[opt_match.start(1):opt_match.end(1)]
    match = TYPE_PATTERN.match(argument_type)
    if match is None:
        return maybe_optional_type(argument_type, opt_match is not None)
    index = match.start(1)
    qualified_type = f'{argument_type[:index]}at::{argument_type[index:]}'
    return maybe_optional_type(qualified_type, opt_match is not None)

def gen_variable_factories(out: str, native_yaml_path: str, template_path: str) -> None:
    native_functions = parse_native_yaml(native_yaml_path)
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template('variable_factories.h', 'variable_factories.h', lambda: {
        'generated_comment': f'@generated from {fm.template_dir}/variable_factories.h',
        'function_definitions': list(mapMaybe(process_function, native_functions)),
    })

@with_native_function
def process_function(f: NativeFunction) -> Optional[str]:
    name = cpp.name(f.func)
    has_tensor_options = python.has_tensor_options(f)

    if Variant.function not in f.variants or \
            not has_tensor_options and not name.endswith("_like"):
        return None

    sig = CppSignatureGroup.from_schema(f.func, method=False).signature
    formals: List[str] = []
    exprs: List[str] = []
    for arg in sig.arguments():
        qualified_type = fully_qualified_type(arg.type)
        if arg.default:
            formals.append(f'{qualified_type} {arg.name} = {arg.default}')
        else:
            formals.append(f'{qualified_type} {arg.name}')

        if isinstance(arg.argument, TensorOptionsArguments):
            # note: we remove the requires_grad setting from the TensorOptions because
            # it is ignored anyways (and we actually have an assertion that it isn't set
            # which would fail otherwise). We handle requires_grad explicitly here
            # instead of passing it through to the kernel.
            exprs.append(f'at::TensorOptions({arg.name}).requires_grad(c10::nullopt)')
        else:
            exprs.append(arg.name)

    requires_grad = "options.requires_grad()" if has_tensor_options else "false"

    return f"""\
inline at::Tensor {name}({', '.join(formals)}) {{
  at::Tensor tensor = ([&]() {{
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::{name}({', '.join(exprs)});
  }})();
  at::Tensor result =
    autograd::make_variable(std::move(tensor), /*requires_grad=*/{requires_grad});
  return result;
}}
"""
