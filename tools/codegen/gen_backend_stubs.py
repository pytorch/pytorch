import pathlib
import argparse
import os
import yaml
from typing import List, Dict, Union, Tuple, Sequence
from tools.codegen.gen import FileManager, get_grouped_native_functions, parse_native_yaml
from tools.codegen.model import (ExternalBackendFunction, ExternalBackendFunctionsGroup,
                                 NativeFunction, NativeFunctionsGroup, OperatorName,
                                 ExternalBackendMetadata, assert_never, DispatchKey,
                                 FunctionSchema)
from tools.codegen.selective_build.selector import SelectiveBuilder
from tools.codegen.utils import Target, concatMap
import tools.codegen.dest as dest
import tools.codegen.api.dispatcher as dispatcher

try:
    # use faster C loader if available
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore


def parse_backend_yaml(
        backend_yaml_path: str,
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]]
) -> Tuple[str, List[Union[ExternalBackendFunction, ExternalBackendFunctionsGroup]]]:
    with open(backend_yaml_path, 'r') as f:
        yaml_values = yaml.load(f, Loader=Loader)
    assert isinstance(yaml_values, dict)

    valid_keys = ['backend', 'cpp_namespace', 'supported', 'autograd']

    backend = yaml_values.pop('backend', None)
    assert backend is not None, 'You must provide a value for "backend"'
    backend_key = DispatchKey.try_parse(backend)
    assert backend_key is not None, f'The provided value for "backend" must be a valid DispatchKey, but got {backend}. \
The set of valid dispatch keys is: {", ".join(k for k, v in DispatchKey.__members__.items())}'

    cpp_namespace = yaml_values.pop('cpp_namespace', None)
    assert cpp_namespace is not None, 'You must provide a value for "cpp_namespace"'

    supported = yaml_values.pop('supported', [])
    if supported is None:
        supported = []  # Allow an empty list of supported ops
    assert isinstance(supported, list), f'expected "supported" to be a list, but got: {supported} (of type {type(supported)})'
    supported_autograd = yaml_values.pop('autograd', [])
    assert isinstance(supported, list), f'expected "autograd" to be a list, but got: {supported_autograd}'

    if len(supported_autograd) > 0:
        autograd_key = DispatchKey.try_parse(f'Autograd{backend}')
        assert autograd_key is not None, f'The "autograd" key was specified, which indicates that you would like to override \
the behavior of autograd for some operators on your backend. However "Autograd{backend}" is not a valid DispatchKey. \
The set of valid dispatch keys is: {", ".join(k for k, v in DispatchKey.__members__.items())}'

    assert len(yaml_values.keys()) == 0, \
        f'{backend_yaml_path} contains unexpected keys: {", ".join(yaml_values.keys())}. \
Only the following keys are supported: {", ".join(valid_keys)}'

    metadata: Dict[OperatorName, ExternalBackendMetadata] = {}
    for op in supported:
        op_name = OperatorName.parse(op)
        m = ExternalBackendMetadata(op_name, is_autograd=False)
        metadata[m.operator] = m
    for op in supported_autograd:
        op_name = OperatorName.parse(op)
        m = ExternalBackendMetadata(op_name, is_autograd=True)
        metadata[m.operator] = m

    native_functions_map: Dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()), grouped_native_functions)
    }

    def kernel_name(func: FunctionSchema) -> str:
        # For external backends, we enforce that their names and signatures match the dispatcher convention
        return dispatcher.name(func)

    def native_to_external(
            g: Union[NativeFunction, NativeFunctionsGroup]
    ) -> Union[ExternalBackendFunction, ExternalBackendFunctionsGroup]:
        if isinstance(g, NativeFunction):
            f = g
            m = metadata.get(f.func.name, None)
            dispatch_key = DispatchKey.parse(f'Autograd{backend}') \
                if m is not None and m.is_autograd else DispatchKey.parse(backend)
            kernel = kernel_name(f.func)
            return ExternalBackendFunction(NativeFunction.with_dispatch_entry(f, dispatch_key, kernel), dispatch_key, m)
        elif isinstance(g, NativeFunctionsGroup):
            out_meta = metadata.get(g.out.func.name, None)
            kernel = kernel_name(g.out.func)
            dispatch_key = DispatchKey.parse(f'Autograd{backend}') \
                if out_meta is not None and out_meta.is_autograd else DispatchKey.parse(backend)
            out = ExternalBackendFunction(NativeFunction.with_dispatch_entry(g.out, dispatch_key, kernel), dispatch_key, out_meta)

            functional_meta = metadata.get(g.functional.func.name, None)
            kernel = kernel_name(g.functional.func)
            dispatch_key = DispatchKey.parse(f'Autograd{backend}') \
                if functional_meta is not None and functional_meta.is_autograd else DispatchKey.parse(backend)
            functional = ExternalBackendFunction(
                NativeFunction.with_dispatch_entry(g.functional, dispatch_key, kernel), dispatch_key, functional_meta)

            if out_meta is not None and functional_meta is not None:
                assert out_meta.is_autograd == functional_meta.is_autograd, \
                    f'Currently, all variants of an op must either be registered to a backend key, or to a backend\'s \
autograd key. They can not be mix and matched. If this is something you need, feel free to create an issue! \
{out_meta.operator} is listed under {"autograd" if out_meta.is_autograd else "supported"}, but \
{functional_meta.operator} is listed under {"autograd" if functional_meta.is_autograd else "supported"}'

            inplace = None
            if g.inplace:
                inplace_meta = metadata.get(g.inplace.func.name, None)
                kernel = kernel_name(g.inplace.func)
                dispatch_key = DispatchKey.parse(f'Autograd{backend}') \
                    if inplace_meta is not None and inplace_meta.is_autograd else DispatchKey.parse(backend)
                inplace = ExternalBackendFunction(
                    NativeFunction.with_dispatch_entry(g.inplace, dispatch_key, kernel), dispatch_key, inplace_meta)
                if inplace_meta is not None:
                    other_meta = out_meta if out_meta is not None else functional_meta
                    if other_meta is not None:
                        assert inplace_meta.is_autograd == other_meta.is_autograd, \
                            f'Currently, all variants of an op must either be registered to a backend key, or to a backend\'s \
autograd key. They can not be mix and matched. If this is something you need, feel free to create an issue! \
{inplace_meta.operator} is listed under {"autograd" if inplace_meta.is_autograd else "supported"}, but \
{other_meta.operator} is listed under {"autograd" if other_meta.is_autograd else "supported"}'

            return ExternalBackendFunctionsGroup(functional, inplace, out)
        else:
            assert_never(g)
    for op_name in metadata.keys():
        assert op_name in native_functions_map, f"Found an invalid operator name: {op_name}"
    return cpp_namespace, [native_to_external(g) for g in grouped_native_functions]

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
    options = parser.parse_args()

    run(options.source_yaml, options.output_dir, options.dry_run)

def run(source_yaml: str, output_dir: str, dry_run: bool) -> None:

    # Assumes that this file lives at PYTORCH_ROOT/tools/codegen/gen_backend_stubs.py
    pytorch_root = pathlib.Path(__file__).parent.parent.parent.absolute()
    template_dir = os.path.join(pytorch_root, "aten/src/ATen/templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=dry_run)

    fm = make_file_manager(output_dir)

    native_yaml_path = os.path.join(pytorch_root, 'aten/src/ATen/native/native_functions.yaml')
    grouped_native_functions = get_grouped_native_functions(native_yaml_path)
    cpp_namespace, external_backend_functions = parse_backend_yaml(source_yaml, grouped_native_functions)

    native_functions = parse_native_yaml(native_yaml_path)

    selector = SelectiveBuilder.get_nop_selector()


    generated_comment = 'Autogenerated file by gen_backend_stubs.py. Do not edit directly!'
    fm.write('aten_xla_type.h', lambda: {
        'generated_comment': generated_comment,
        'cpp_namespace': cpp_namespace,
        'dispatch_xla_declarations': list(concatMap(dest.compute_native_function_declaration, external_backend_functions)),
    })

    fm.write('aten_xla_type_default.h', lambda: {
        'generated_comment': generated_comment,
        'cpp_namespace': cpp_namespace,
        'dispatch_aten_fallback_declarations': list(concatMap(
            dest.GenExternalAtenFallback(Target.NAMESPACED_DECLARATION), external_backend_functions
        )),
    })

    fm.write('aten_xla_type_default.cpp', lambda: {
        'generated_comment': generated_comment,
        'cpp_namespace': cpp_namespace,
        # TODO: after cpu fallbacks are moved to a boxed kernel,
        # merge registrations / definitions into RegisterDispatchKey
        'dispatch_aten_fallback_definitions': list(concatMap(
            dest.GenExternalAtenFallback(Target.NAMESPACED_DEFINITION), external_backend_functions
        )),
        'dispatch_registrations': list(concatMap(
            dest.GenExternalAtenFallback(Target.REGISTRATION), [e for e in external_backend_functions if not e.is_autograd_kernel]
        )),
        'dispatch_autograd_registrations': list(concatMap(
            dest.GenExternalAtenFallback(Target.REGISTRATION), [e for e in external_backend_functions if e.is_autograd_kernel]
        )),
    })

if __name__ == '__main__':
    main()
