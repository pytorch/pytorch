import os
import textwrap
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
from tools.codegen.utils import (
    Target, concatMap, context, mapMaybe, YamlDumper, YamlLoader, FileManager, assert_never
)
from tools.codegen.gen_vmap_plumbing import gen_all_vmap_plumbing

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
#                           RUN IT ALL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate ATen source files')
    parser.add_argument(
        '-s',
        '--source-path',
        help='path to source directory for ATen',
        default='/scratch/rzou/pt/workbench/aten/src/ATen')
    parser.add_argument(
        '-o',
        '--output-dependencies',
        help='output a list of dependencies into the given file and exit')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='run without writing any files (still updates outputs)')
    parser.add_argument(
        '-d', '--install_dir', help='output directory',
        default='functorch/csrc')
    options = parser.parse_args()

    native_yaml_path = os.path.join(options.source_path, 'native/native_functions.yaml')
    parsed_yaml = parse_native_yaml(native_yaml_path)
    native_functions, backend_indices = parsed_yaml.native_functions, parsed_yaml.backend_indices
    grouped_native_functions = get_grouped_native_functions(native_functions)
    template_dir = os.path.join(options.source_path, "templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=options.dry_run)

    cpu_fm = make_file_manager(options.install_dir)
    cpu_fm.write('OutOfPlacePlumbing.h', lambda: gen_all_vmap_plumbing(native_functions))

    if options.output_dependencies:
        depfile_path = pathlib.Path(options.output_dependencies).resolve()
        depfile_name = depfile_path.name
        depfile_stem = depfile_path.stem

        for fm, prefix in [
                (cpu_fm, ""),
        ]:
            varname = prefix + depfile_stem
            path = depfile_path.parent / (prefix + depfile_name)
            fm.write_outputs(varname, str(path))


if __name__ == '__main__':
    main()
