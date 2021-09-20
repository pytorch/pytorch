import pathlib
import argparse
import os
import yaml
from collections import namedtuple
from typing import List, Dict, Union, Sequence, Optional, Callable, Iterable, Iterator
from tools.codegen.gen import FileManager, get_grouped_native_functions, parse_native_yaml
from tools.codegen.model import (DispatchKey,
                                 NativeFunction, NativeFunctionsGroup, OperatorName)
from tools.codegen.selective_build.selector import SelectiveBuilder
from tools.codegen.utils import concatMap, YamlLoader
import tools.codegen.dest as dest
from .gen_backend_stubs import (parse_backend_yaml, error_on_missing_kernels,
                                gen_dispatchkey_nativefunc_headers,
                                gen_dispatcher_registrations)

# Parses the external backend's yaml, and adds a new BackendIndex for the backend's dispatch key.
# Returns a Tuple of (backend_key, autograd_key, cpp_namespace, updated BackendIndex mapping, full_codegen)
ParsedExternalYaml = namedtuple('ParsedExternalYaml', [
    'backend_key', 'autograd_key', 'cpp_namespace', 'backend_indices', 'full_codegen'])


def parse_full_codegen_ops(
        backend_yaml_path: str,
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
) -> List[OperatorName]:

    native_functions_map: Dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()), grouped_native_functions)
    }

    with open(backend_yaml_path, 'r') as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    assert isinstance(yaml_values, dict)

    full_codegen = yaml_values.pop('full_codegen', [])
    assert isinstance(full_codegen, list), f'expected "full_codegen" to be a list, but got: {full_codegen}'
    full_codegen = [OperatorName.parse(name) for name in full_codegen]

    return full_codegen


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Lazy Tensor backend files')
    parser.add_argument(
        '-s',
        '--source_yaml',
        help='path to source yaml file containing operator external definitions')
    parser.add_argument(
        '-o', '--output_dir', help='output directory')
    parser.add_argument(
        '--dry_run', type=bool, default=False, help='output directory')
    parser.add_argument(
        '--impl_path', type=str, default=None, help='path to the source C++ file containing kernel definitions')
    parser.add_argument(
        '--gen_ts_lowerings', action="store_true", help='Generate TorchScript lowerings in addition to Lazy IR and NativeFunctions')
    options = parser.parse_args()

    run(options.source_yaml, options.output_dir, options.dry_run, options.impl_path, options.gen_ts_lowerings)


def run(source_yaml: str, output_dir: str, dry_run: bool, impl_path: Optional[str], gen_ts_lowerings: bool) -> None:

    # Assumes that this file lives at PYTORCH_ROOT/tools/codegen/gen_backend_stubs.py
    pytorch_root = pathlib.Path(__file__).parent.parent.parent.absolute()
    template_dir = os.path.join(pytorch_root, "aten/src/ATen/templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=dry_run)

    fm = make_file_manager(output_dir)

    native_yaml_path = os.path.join(pytorch_root, 'aten/src/ATen/native/native_functions.yaml')
    parsed_yaml = parse_native_yaml(native_yaml_path)
    native_functions, backend_indices = parsed_yaml.native_functions, parsed_yaml.backend_indices
    grouped_native_functions = get_grouped_native_functions(native_functions)
    parsed_backend_yaml = parse_backend_yaml(source_yaml, grouped_native_functions, backend_indices)
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    backend_indices = parsed_backend_yaml.backend_indices
    full_codegen = parse_full_codegen_ops(source_yaml, grouped_native_functions)

    def concatMapCodegen(func: Callable[[NativeFunction], Sequence[str]], xs: Iterable[Union[NativeFunctionsGroup, NativeFunction]]) -> Iterator[str]:
        for x in xs:
            f = x.functional if isinstance(x, NativeFunctionsGroup) else x
            if f.func.name in full_codegen:
                for r in func(f):
                    yield r

    selector = SelectiveBuilder.get_nop_selector()

    # TODO: handle cases when yaml contains zero ops properly in a later PR.
    if backend_key is not None and autograd_key is not None:
        backend_dispatch_key: DispatchKey = backend_key
        autograd_dispatch_key: DispatchKey = autograd_key
        class_name = backend_indices[backend_dispatch_key].native_function_class_name()

        if impl_path is not None:
            error_on_missing_kernels(native_functions, backend_indices, backend_key,
                                     autograd_key, impl_path, full_codegen)

        assert class_name is not None

        # Generate nativefunction declarations
        gen_dispatchkey_nativefunc_headers(fm, class_name, cpp_namespace, backend_indices,
                                           grouped_native_functions, backend_dispatch_key, autograd_dispatch_key)

        # Generate Dispatcher registrations which hook up the nativefunctions
        for dispatch_key in [backend_dispatch_key, autograd_dispatch_key]:
            gen_dispatcher_registrations(fm, output_dir, cpp_namespace, backend_indices, grouped_native_functions,
                                         backend_dispatch_key, dispatch_key, selector)

        # Generate native function impls that build IR nodes
        fm.write_with_template(f'{backend_dispatch_key}NativeFunctions.cpp', 'DispatchKeyNativeFunctions.cpp', lambda: {
            'generated_comment': '',
            'includes': [f'#include "{path}"' for path in [
                "lazy_tensor_core/csrc/tensor.h",
                "lazy_tensor_core/csrc/aten_ltc_bridge.h",
                f"{output_dir}/{backend_key}NativeFunctions.h",
                f"{output_dir}/{backend_key}LazyIr.h",
                f"{output_dir}/{backend_key}ShapeDtype.h",
                "ATen/MetaFunctions.h",
            ]],
            'native_functions_include': '',
            'backend_namespace': 'torch_lazy_tensors',  # this is wrong
            'native_function_definitions':
            list(concatMapCodegen(
                lambda f: dest.gen_lazy_nativefunc_definition(
                    f,
                    backend_indices[backend_dispatch_key],
                    class_method_name=f'{backend_dispatch_key}NativeFunctions'),
                grouped_native_functions
            )),
        })

        # Generate headers for shape/dtype funcs for non-meta kernels
        fm.write_with_template(f'{backend_dispatch_key}ShapeDtype.h', 'ShapeDtype.h', lambda: {
            'lazy_ir_sysinc': [f'#include <{path}>' for path in [
                "c10/core/ScalarType.h",
                "c10/util/Optional.h",
                "vector",
            ]],
            'lazy_ir_inc': [f'#include "{path}"' for path in [
                "lazy_tensor_core/csrc/ir.h",
                "lazy_tensors/types.h",
                "lazy_tensor_core/csrc/compiler/node_lowering.h"
            ]],
            'DispatchKey': backend_dispatch_key,
            'dispatch_namespace': backend_dispatch_key.lower(),
            'func_declarations': list(concatMapCodegen(
                lambda f: dest.gen_lazy_shape_dtype_decl(f, backend_indices[backend_dispatch_key]),
                grouped_native_functions
            )),
        })

        # Generate IR node classes
        fm.write_with_template(f'{backend_dispatch_key}LazyIr.h', 'LazyIr.h', lambda: {
            'lazy_ir_sysinc': [f'#include <{path}>' for path in [
                "c10/core/ScalarType.h",
                "c10/util/Optional.h",
                "vector",
            ]],
            'lazy_ir_inc': [f'#include "{path}"' for path in [
                "lazy_tensor_core/csrc/ir.h",
                "lazy_tensors/types.h",
                "lazy_tensor_core/csrc/compiler/node_lowering.h"
            ]],
            'external_backend_headers': f'#include "{output_dir}/{backend_key}NativeFunctions.h"',
            'namespaced_headers': '',
            'DispatchKey': backend_dispatch_key,
            'dispatch_namespace': backend_dispatch_key.lower(),
            'ir_declarations': list(concatMapCodegen(
                dest.LazyIR(backend_indices[backend_dispatch_key]),
                grouped_native_functions
            )),
        })

        if gen_ts_lowerings:
            # Generate TorchScript Lowerings for the IR nodes
            fm.write_with_template('LazyTsLowering.cpp', 'LazyTsLowering.cpp', lambda: {
                'ts_lowering_sysinc': [f'#include <{path}>' for path in [
                    "vector",
                ]],
                'ts_lowering_inc': [f'#include "{path}"' for path in [
                    "lazy_tensor_core/csrc/ir.h",
                    "torch/csrc/jit/ir/named_value.h",
                    "torch/csrc/jit/frontend/sugared_value.h",
                    "lazy_tensor_core/csrc/ts_backend/ts_lowering_context.h",
                    f"{output_dir}/{backend_key}LazyIr.h",
                ]],
                'backend_namespace': backend_dispatch_key.lower(),  # TODO this is not designed yet
                'lowering_dispatches': list(concatMapCodegen(
                    dest.LazyTsLowering(
                        backend_indices[backend_dispatch_key],
                        dest.LazyTsLowering.TsLoweringTarget.DISPATCH),
                    grouped_native_functions,
                )),
                'lowering_definitions': list(concatMapCodegen(
                    dest.LazyTsLowering(
                        backend_indices[backend_dispatch_key],
                        dest.LazyTsLowering.TsLoweringTarget.LOWERING),
                    grouped_native_functions
                )),
            })


if __name__ == '__main__':
    main()
