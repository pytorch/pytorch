import pathlib
import argparse
import os
import yaml
import re
from collections import namedtuple, Counter, defaultdict
from typing import List, Dict, Union, Sequence, Optional
from tools.codegen.gen import FileManager, get_grouped_native_functions, parse_native_yaml
from tools.codegen.model import (BackendIndex, BackendMetadata, DispatchKey,
                                 NativeFunction, NativeFunctionsGroup, OperatorName)
from tools.codegen.selective_build.selector import SelectiveBuilder
from tools.codegen.utils import Target, concatMap, context, YamlLoader
from tools.codegen.context import native_function_manager
import tools.codegen.dest as dest
import tools.codegen.api.dispatcher as dispatcher
from tools.codegen.api.types import DispatcherSignature

# Parses the external backend's yaml, and adds a new BackendIndex for the backend's dispatch key.
# Returns a Tuple of (backend_key, autograd_key, cpp_namespace, updated BackendIndex mapping, codegen)
ParsedExternalYaml = namedtuple('ParsedExternalYaml', [
    'backend_key', 'autograd_key', 'cpp_namespace', 'backend_indices', 'codegen'])


def parse_backend_yaml(
        backend_yaml_path: str,
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        backend_indices: Dict[DispatchKey, BackendIndex]
) -> ParsedExternalYaml:

    native_functions_map: Dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()), grouped_native_functions)
    }

    with open(backend_yaml_path, 'r') as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    assert isinstance(yaml_values, dict)

    valid_keys = ['backend', 'cpp_namespace', 'extra_headers', 'supported', 'autograd', 'codegen']

    backend = yaml_values.pop('backend', None)
    assert backend is not None, 'You must provide a value for "backend"'

    cpp_namespace = yaml_values.pop('cpp_namespace', None)
    assert cpp_namespace is not None, 'You must provide a value for "cpp_namespace"'

    supported = yaml_values.pop('supported', [])
    if supported is None:
        supported = []  # Allow an empty list of supported ops
    assert isinstance(
        supported, list), f'expected "supported" to be a list, but got: {supported} (of type {type(supported)})'

    supported_autograd = yaml_values.pop('autograd', [])
    assert isinstance(supported, list), f'expected "autograd" to be a list, but got: {supported_autograd}'

    codegen = yaml_values.pop('codegen', [])
    assert isinstance(codegen, list), f'expected "codegen" to be a list, but got: {codegen}'
    supported.extend(codegen)
    codegen = [OperatorName.parse(name) for name in codegen]

    assert len(yaml_values.keys()) == 0, \
        f'{backend_yaml_path} contains unexpected keys: {", ".join(yaml_values.keys())}. \
Only the following keys are supported: {", ".join(valid_keys)}'

    def create_backend_index(backend_ops: List[str], dispatch_key: DispatchKey) -> BackendIndex:
        metadata: Dict[OperatorName, BackendMetadata] = {}
        for op in backend_ops:
            op_name = OperatorName.parse(op)
            assert op_name in native_functions_map, f"Found an invalid operator name: {op_name}"
            # See Note [External Backends Follow Dispatcher API]
            kernel_name = dispatcher.name(native_functions_map[op_name].func)
            # TODO: allow structured external backends later.
            m = BackendMetadata(kernel=kernel_name, structured=False)
            metadata[op_name] = m
        # TODO: currently hardcoding the fact that XLA implements out/inplace in terms of functional ops,
        # this should eventually be toggleable per-backend.
        return BackendIndex(
            dispatch_key=dispatch_key,
            use_out_as_primary=False,
            external=True,
            index=metadata)

    backend_key: Optional[DispatchKey] = None
    if len(supported) > 0:
        with context(lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'):
            backend_key = DispatchKey.parse(backend)

        backend_idx = create_backend_index(supported, backend_key)
        assert backend_key not in backend_indices
        backend_indices[backend_key] = backend_idx

    autograd_key: Optional[DispatchKey] = None
    if len(supported_autograd) > 0:
        with context(lambda: f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'):
            autograd_key = DispatchKey.parse(f'Autograd{backend}')

        autograd_idx = create_backend_index(supported_autograd, autograd_key)
        assert autograd_key not in backend_indices
        backend_indices[autograd_key] = autograd_idx

    for g in grouped_native_functions:
        if isinstance(g, NativeFunction):
            forward_kernels = [] if backend_key is None else \
                [m for m in [backend_indices[backend_key].get_kernel(g)] if m is not None]
            backward_kernels = [] if autograd_key is None else \
                [m for m in [backend_indices[autograd_key].get_kernel(g)] if m is not None]
        else:
            forward_kernels = [] if backend_key is None else [m for m in [
                backend_indices[backend_key].get_kernel(f) for f in g.functions()]
                if m is not None]
            backward_kernels = [] if autograd_key is None else [m for m in [
                backend_indices[autograd_key].get_kernel(f) for f in g.functions()]
                if m is not None]

        forward_kernels = [f for f in forward_kernels if f is not None]
        backward_kernels = [f for f in backward_kernels if f is not None]
        assert len(forward_kernels) == 0 or len(backward_kernels) == 0, \
            f'Currently, all variants of an op must either be registered to a backend key, or to a backend\'s \
autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! \
{forward_kernels[0].kernel} is listed under "supported", but {backward_kernels[0].kernel} is listed under "autograd".'

    return ParsedExternalYaml(backend_key, autograd_key, cpp_namespace, backend_indices, codegen)


def error_on_missing_kernels(
        native_functions: Sequence[NativeFunction],
        backend_indices: Dict[DispatchKey, BackendIndex],
        backend_key: DispatchKey,
        autograd_key: DispatchKey,
        codegen: List[OperatorName],
        kernel_defn_file_path: str,
) -> None:
    try:
        with open(kernel_defn_file_path, 'r') as f:
            backend_defns = f.read()
    except IOError:
        raise AssertionError(f'Unable to read from the specified impl_path file: {kernel_defn_file_path}')

    class_name: Optional[str] = backend_indices[backend_key].native_function_class_name()
    assert class_name is not None

    expected_backend_op_names: List[OperatorName] = \
        list(backend_indices[backend_key].index.keys()) + list(backend_indices[autograd_key].index.keys())
    expected_backend_native_funcs: List[NativeFunction] = [
        f for f in native_functions if f.func.name in expected_backend_op_names and f.func.name not in codegen]

    expected_backend_kernel_name_counts: Dict[str, List[NativeFunction]] = defaultdict(list)
    for native_f in expected_backend_native_funcs:
        expected_backend_kernel_name_counts[dispatcher.name(native_f.func)].append(native_f)

    kernel_defn_regex = rf'{class_name}::([\w\d]*)\([^\)]*\)\s*{{'
    actual_backend_kernel_name_counts = Counter(re.findall(kernel_defn_regex, backend_defns))

    missing_kernels_err_msg = ""
    for expected_name, funcs in expected_backend_kernel_name_counts.items():
        expected_overload_count = len(funcs)
        actual_overload_count = actual_backend_kernel_name_counts[expected_name]
        if expected_overload_count != actual_overload_count:
            def create_decl(f: NativeFunction) -> str:
                with native_function_manager(f):
                    return DispatcherSignature.from_schema(f.func).decl()
            expected_schemas_str = '\n'.join([create_decl(f) for f in funcs])
            missing_kernels_err_msg += f"""
{class_name} is missing a kernel definition for {expected_name}. We found {actual_overload_count} kernel(s) with that name,
but expected {expected_overload_count} kernel(s). The expected function schemas for the missing operator are:
{expected_schemas_str}

"""
    assert missing_kernels_err_msg == "", missing_kernels_err_msg

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
    codegen = parsed_backend_yaml.codegen

    selector = SelectiveBuilder.get_nop_selector()

    # TODO: handle cases when yaml contains zero ops properly in a later PR.
    if backend_key is not None and autograd_key is not None:
        backend_dispatch_key: DispatchKey = backend_key
        autograd_dispatch_key: DispatchKey = autograd_key
        class_name = backend_indices[backend_dispatch_key].native_function_class_name()

        if impl_path is not None:
            error_on_missing_kernels(native_functions, backend_indices, backend_key, autograd_key, codegen, impl_path)

        assert class_name is not None
        generated_comment = 'Autogenerated file by gen_backend_stubs.py. Do not edit directly!'
        fm.write_with_template(f'{backend_dispatch_key}NativeFunctions.h', 'DispatchKeyNativeFunctions.h', lambda: {
            'generated_comment': generated_comment,
            'cpp_namespace': cpp_namespace,
            'class_name': class_name,
            # Convert to a set first to remove duplicate kernel names.
            # Backends are allowed to repeat kernel names; only generate the declaration once!
            'dispatch_declarations': list(set(concatMap(
                lambda f: dest.compute_native_function_declaration(f, backend_indices[backend_dispatch_key]),
                grouped_native_functions
            ))) + list(set(concatMap(
                lambda f: dest.compute_native_function_declaration(f, backend_indices[autograd_dispatch_key]),
                grouped_native_functions
            ))),
        })

        for dispatch_key in [backend_dispatch_key, autograd_dispatch_key]:
            fm.write_with_template(f'Register{dispatch_key}.cpp', 'RegisterDispatchKey.cpp', lambda: {
                'extra_cuda_headers': '',
                'legacy_th_headers': '',
                'external_backend_headers': f'#include "{output_dir}/{backend_key}NativeFunctions.h"',
                'namespaced_headers': '',
                'DispatchKey': dispatch_key,
                'dispatch_namespace': dispatch_key.lower(),
                'dispatch_namespaced_definitions': list(concatMap(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.NAMESPACED_DEFINITION,
                        selector,
                        rocm=False,
                        cpp_namespace=cpp_namespace,
                        class_method_name=f'{backend_dispatch_key}NativeFunctions'),
                    grouped_native_functions
                )),
                'dispatch_anonymous_definitions': list(concatMap(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.ANONYMOUS_DEFINITION,
                        selector,
                        rocm=False,
                        cpp_namespace=cpp_namespace,
                        class_method_name=f'{backend_dispatch_key}NativeFunctions'),
                    grouped_native_functions
                )),
                'dispatch_registrations': list(concatMap(
                    dest.RegisterDispatchKey(
                        backend_indices[dispatch_key],
                        Target.REGISTRATION,
                        selector,
                        rocm=False,
                        cpp_namespace=cpp_namespace,
                        class_method_name=f'{backend_dispatch_key}NativeFunctions'),
                    grouped_native_functions
                )),
            })

        # generate native function impls that build IR nodes
        for dispatch_key in [backend_dispatch_key]:
            fm.write_with_template(f'{dispatch_key}NativeFunctions.cpp', 'DispatchKeyNativeFunctions.cpp', lambda: {
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
                list(concatMap(
                    lambda f: dest.compute_lazy_native_function_definition(
                        f,
                        backend_indices[dispatch_key],
                        codegen,
                        class_method_name=f'{backend_dispatch_key}NativeFunctions'),
                    grouped_native_functions
                )),
            })

        # and generate headers for shape/dtype funcs for non-meta kernels
        for dispatch_key in [backend_dispatch_key]:  # , autograd_dispatch_key
            fm.write_with_template(f'{dispatch_key}ShapeDtype.h', 'ShapeDtype.h', lambda: {
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
                'DispatchKey': dispatch_key,
                'dispatch_namespace': dispatch_key.lower(),
                'func_declarations': list(concatMap(
                    lambda f: dest.compute_lazy_shape_dtype_decl(f, backend_indices[dispatch_key], codegen),
                    grouped_native_functions
                )),
            })

        # and generate IR nodes
        for dispatch_key in [backend_dispatch_key]:  # , autograd_dispatch_key
            fm.write_with_template(f'{dispatch_key}LazyIr.h', 'LazyIr.h', lambda: {
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
                'DispatchKey': dispatch_key,
                'dispatch_namespace': dispatch_key.lower(),
                'ir_declarations': list(concatMap(
                    dest.LazyIR(backend_indices[dispatch_key], codegen),
                    grouped_native_functions
                )),
            })

        if gen_ts_lowerings:
            # and TorchScript Lowerings for the IR nodes
            for dispatch_key in [backend_dispatch_key]:  # , autograd_dispatch_key
                fm.write_with_template(f'TSLowering.cpp', 'TSLowering.cpp', lambda: {
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
                    'backend_namespace': dispatch_key.lower(),  # TODO this is not designed yet
                    'lowering_dispatches': list(concatMap(
                        dest.TsLowering(
                            backend_indices[dispatch_key],
                            codegen,
                            dest.TsLowering.TsLoweringTarget.DISPATCH),
                        grouped_native_functions
                    )),
                    'lowering_definitions': list(concatMap(
                        dest.TsLowering(
                            backend_indices[dispatch_key],
                            codegen,
                            dest.TsLowering.TsLoweringTarget.LOWERING),
                        grouped_native_functions
                    )),
                })


if __name__ == '__main__':
    main()
