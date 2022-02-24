import pathlib
import argparse
import os
import yaml
import re
from collections import namedtuple, Counter, defaultdict
from typing import List, Dict, Union, Sequence, Optional
from tools.codegen.gen import get_grouped_native_functions, parse_native_yaml
from tools.codegen.model import (BackendIndex, BackendMetadata, DispatchKey,
                                 NativeFunction, NativeFunctionsGroup, OperatorName)
from tools.codegen.selective_build.selector import SelectiveBuilder
from tools.codegen.utils import Target, concatMap, context, YamlLoader, FileManager
from tools.codegen.context import native_function_manager
import tools.codegen.dest as dest
import tools.codegen.api.dispatcher as dispatcher
from tools.codegen.api.types import DispatcherSignature


# Parses the external backend's yaml, and adds a new BackendIndex for the backend's dispatch key.
# Returns a Tuple of (backend_key, autograd_key, cpp_namespace, updated BackendIndex mapping)
ParsedExternalYaml = namedtuple('ParsedExternalYaml', [
    'backend_key', 'autograd_key', 'cpp_namespace', 'backend_indices'])
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

    valid_keys = ['backend', 'cpp_namespace', 'extra_headers', 'supported', 'autograd', 'full_codegen']

    backend = yaml_values.pop('backend', None)
    assert backend is not None, 'You must provide a value for "backend"'

    cpp_namespace = yaml_values.pop('cpp_namespace', None)
    assert cpp_namespace is not None, 'You must provide a value for "cpp_namespace"'

    # Mostly just defaulting to false to stick with LazyTensor convention.
    use_out_as_primary = yaml_values.pop('use_out_as_primary', False)
    assert isinstance(use_out_as_primary, bool), \
        f'You must provide either True or False for use_out_as_primary. Provided: {use_out_as_primary}'

    use_device_guard = yaml_values.pop('device_guard', False)
    assert isinstance(use_device_guard, bool), \
        f'You must provide either True or False for device_guard. Provided: {use_device_guard}'

    supported = yaml_values.pop('supported', [])
    if supported is None:
        supported = []  # Allow an empty list of supported ops
    assert isinstance(supported, list), f'expected "supported" to be a list, but got: {supported} (of type {type(supported)})'

    supported_autograd = yaml_values.pop('autograd', [])
    assert isinstance(supported, list), f'expected "autograd" to be a list, but got: {supported_autograd}'

    # full_codegen is ignored by parse_backend_yaml, and re-parsed in gen_lazy_tensor.py
    full_codegen = yaml_values.pop('full_codegen', [])
    supported.extend(full_codegen)

    assert len(yaml_values.keys()) == 0, \
        f'{backend_yaml_path} contains unexpected keys: {", ".join(yaml_values.keys())}. \
Only the following keys are supported: {", ".join(valid_keys)}'

    def create_backend_index(
            backend_ops: List[str],
            dispatch_key: DispatchKey,
            *,
            use_out_as_primary: bool,
            use_device_guard: bool
    ) -> BackendIndex:
        metadata: Dict[OperatorName, BackendMetadata] = {}
        for op in backend_ops:
            op_name = OperatorName.parse(op)
            assert op_name in native_functions_map, f"Found an invalid operator name: {op_name}"
            # See Note [External Backends Follow Dispatcher API]
            kernel_name = dispatcher.name(native_functions_map[op_name].func)
            # TODO: allow structured external backends later.
            m = BackendMetadata(kernel=kernel_name, structured=False)
            metadata[op_name] = m
        return BackendIndex(
            dispatch_key=dispatch_key,
            use_out_as_primary=use_out_as_primary,
            external=True,
            device_guard=use_device_guard,
            index=metadata)

    backend_key: Optional[DispatchKey] = None
    if len(supported) > 0:
        with context(lambda: f'The provided value for "backend" must be a valid DispatchKey, but got {backend}.'):
            backend_key = DispatchKey.parse(backend)

        backend_idx = create_backend_index(
            supported, backend_key, use_out_as_primary=use_out_as_primary, use_device_guard=use_device_guard)
        assert backend_key not in backend_indices
        backend_indices[backend_key] = backend_idx

    autograd_key: Optional[DispatchKey] = None
    if len(supported_autograd) > 0:
        with context(lambda: f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey.'):
            autograd_key = DispatchKey.parse(f'Autograd{backend}')

        autograd_idx = create_backend_index(
            supported_autograd, autograd_key, use_out_as_primary=use_out_as_primary, use_device_guard=use_device_guard)
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

    return ParsedExternalYaml(backend_key, autograd_key, cpp_namespace, backend_indices)

def error_on_missing_kernels(
        native_functions: Sequence[NativeFunction],
        backend_indices: Dict[DispatchKey, BackendIndex],
        backend_key: DispatchKey,
        autograd_key: Optional[DispatchKey],
        kernel_defn_file_path: str,
        full_codegen: Optional[List[OperatorName]] = None,
) -> None:
    try:
        with open(kernel_defn_file_path, 'r') as f:
            backend_defns = f.read()
    except IOError:
        raise AssertionError(f'Unable to read from the specified impl_path file: {kernel_defn_file_path}')

    if full_codegen is None:
        full_codegen = []

    class_name: Optional[str] = backend_indices[backend_key].native_function_class_name()
    assert class_name is not None

    expected_backend_op_names: List[OperatorName] = \
        list(backend_indices[backend_key].index.keys()) + \
        [] if autograd_key is None else list(backend_indices[autograd_key].index.keys())
    expected_backend_native_funcs: List[NativeFunction] = [
        f for f in native_functions if f.func.name in expected_backend_op_names and f.func.name not in full_codegen]
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
                    return DispatcherSignature.from_schema(f.func, structured_type_override=f.part_of_structured_group).decl()
            expected_schemas_str = '\n'.join([create_decl(f) for f in funcs])
            missing_kernels_err_msg += f"""
{class_name} is missing a kernel definition for {expected_name}. We found {actual_overload_count} kernel(s) with that name,
but expected {expected_overload_count} kernel(s). The expected function schemas for the missing operator are:
{expected_schemas_str}

"""
    assert missing_kernels_err_msg == "", missing_kernels_err_msg

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate backend stub files')
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
    options = parser.parse_args()

    run(options.source_yaml, options.output_dir, options.dry_run, options.impl_path)


def gen_dispatchkey_nativefunc_headers(
        fm: FileManager,
        class_name: str,
        cpp_namespace: str,
        backend_indices: Dict[DispatchKey, BackendIndex],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        backend_dispatch_key: DispatchKey,
        autograd_dispatch_key: Optional[DispatchKey]) -> None:
    assert class_name is not None
    generated_comment = 'Autogenerated file by gen_backend_stubs.py. Do not edit directly!'

    # Convert to a set first to remove duplicate kernel names.
    # Backends are allowed to repeat kernel names; only generate the declaration once!
    # Sort for deterministic output.
    backend_declarations = list(sorted(set(concatMap(
        lambda f: dest.compute_native_function_declaration(f, backend_indices[backend_dispatch_key]),
        grouped_native_functions))))
    autograd_declarations = list(sorted(set(concatMap(
        lambda f: [] if autograd_dispatch_key is None else
        dest.compute_native_function_declaration(f, backend_indices[autograd_dispatch_key]),
        grouped_native_functions))))

    fm.write_with_template(f'{backend_dispatch_key}NativeFunctions.h', 'DispatchKeyNativeFunctions.h', lambda: {
        'generated_comment': generated_comment,
        'cpp_namespace': cpp_namespace,
        'class_name': class_name,
        'dispatch_declarations': backend_declarations + autograd_declarations,
    })


def gen_dispatcher_registrations(
        fm: FileManager,
        output_dir: str,
        cpp_namespace: str,
        backend_indices: Dict[DispatchKey, BackendIndex],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
        backend_dispatch_key: DispatchKey,
        dispatch_key: DispatchKey,
        selector: 'SelectiveBuilder') -> None:
    backend_index = backend_indices[dispatch_key]
    fm.write_with_template(f'Register{dispatch_key}.cpp', 'RegisterDispatchKey.cpp', lambda: {
        'extra_cuda_headers': '',
        'external_backend_headers': f'#include "{output_dir}/{backend_dispatch_key}NativeFunctions.h"',
        'ops_headers': '#include <ATen/Functions.h>',
        'DispatchKey': dispatch_key,
        'dispatch_namespace': dispatch_key.lower(),
        'dispatch_headers': dest.gen_registration_headers(backend_index, per_operator_headers=False),
        'dispatch_helpers': dest.gen_registration_helpers(backend_index),
        'dispatch_namespaced_definitions': list(concatMap(
            dest.RegisterDispatchKey(
                backend_index,
                Target.NAMESPACED_DEFINITION,
                selector,
                rocm=False,
                cpp_namespace=cpp_namespace,
                class_method_name=f'{backend_dispatch_key}NativeFunctions'),
            grouped_native_functions
        )),
        'dispatch_anonymous_definitions': list(concatMap(
            dest.RegisterDispatchKey(
                backend_index,
                Target.ANONYMOUS_DEFINITION,
                selector,
                rocm=False,
                cpp_namespace=cpp_namespace,
                class_method_name=f'{backend_dispatch_key}NativeFunctions'),
            grouped_native_functions
        )),
        'dispatch_registrations': list(concatMap(
            dest.RegisterDispatchKey(
                backend_index,
                Target.REGISTRATION,
                selector,
                rocm=False,
                cpp_namespace=cpp_namespace,
                class_method_name=f'{dispatch_key}NativeFunctions'),
            grouped_native_functions
        )),
    })

def run(source_yaml: str, output_dir: str, dry_run: bool, impl_path: Optional[str] = None) -> None:

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

    selector = SelectiveBuilder.get_nop_selector()


    if backend_key is None:
        # This could be useful if a backend wants to quickly set up a noop yaml file but doesn't have any kernels ready yet.
        return

    class_name = backend_indices[backend_key].native_function_class_name()

    if impl_path is not None:
        error_on_missing_kernels(native_functions, backend_indices, backend_key, autograd_key, impl_path)


        gen_dispatchkey_nativefunc_headers(fm, class_name, cpp_namespace, backend_indices,
                                           grouped_native_functions, backend_key, autograd_key)

        for dispatch_key in [backend_key] if autograd_key is None else [backend_key, autograd_key]:
            gen_dispatcher_registrations(fm, output_dir, cpp_namespace, backend_indices, grouped_native_functions,
                                         backend_key, dispatch_key, selector)
if __name__ == '__main__':
    main()
