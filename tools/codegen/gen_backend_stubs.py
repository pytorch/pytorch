from tools.codegen.model import *
from tools.codegen.api.types import *
from tools.codegen.gen import *

def parse_backend_yaml(backend_yaml_path: str, grouped_native_functions: List[Union[NativeFunction, NativeFunctionsGroup]], backend: str) -> List[Union[ExternalBackendFunction, ExternalBackendFunctionsGroup]]:
    with open(backend_yaml_path, 'r') as f:
        yaml_values = yaml.load(f, Loader=LineLoader)
    assert isinstance(yaml_values, dict)

    backend = yaml_values['backend']
    supported = yaml_values['supported']
    supported_autograd = yaml_values['autograd']

    metadata: Dict[OperatorName, ExternalBackendMetadata] = {}
    for op in supported:
        op_name = OperatorName.parse(op)
        m = ExternalBackendMetadata(op_name, backend, is_autograd=False)
        metadata[m.operator] = m
    for op in supported_autograd:
        op_name = OperatorName.parse(op)
        m = ExternalBackendMetadata(op_name, backend, is_autograd=True)
        metadata[m.operator] = m

    def native_to_external(g: Union[NativeFunction, NativeFunctionsGroup]) -> Union[ExternalBackendFunction, ExternalBackendFunctionsGroup]:
        if isinstance(g, NativeFunction):
            f = g
            m = metadata.get(f.func.name, None)
            return ExternalBackendFunction(f, m)
        elif isinstance(g, NativeFunctionsGroup):
            return ExternalBackendFunctionsGroup.from_function_group(g, metadata)
        else:
            assert_never(g)
    # TODO: error check that each metadata's operator maps back to a valid NativeFunction
    # TODO: error check that xla doesn't implement any `_out` functions themselves here.
    #       We probably want to pass in some config from xla to tell us "xla implements functional and not out kernels"
    return [native_to_external(g) for g in grouped_native_functions]

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate backend stub files')
    parser.add_argument(
        '-pt',
        '--pytorch_root',
        help='path to top-level pytorch core directory')
    parser.add_argument(
        '-s',
        '--source_yaml',
        help='path to source yaml file containing operator external definitions')
    parser.add_argument(
        '-o', '--output_dir', help='output directory')
    parser.add_argument(
        '-b', '--backend', help='The name of the external backend, e.g. XLA')
    options = parser.parse_args()

    template_dir = os.path.join(options.pytorch_root, "aten/src/ATen/templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=False)

    fm = make_file_manager(options.output_dir)

    native_yaml_path = os.path.join(options.pytorch_root, 'aten/src/ATen/native/native_functions.yaml')
    grouped_native_functions = get_grouped_native_functions(native_yaml_path)
    external_backend_functions = parse_backend_yaml(options.source_yaml, grouped_native_functions, options.backend)

    native_functions = parse_native_yaml(native_yaml_path)

    selector = SelectiveBuilder.get_nop_selector()

    fm.write('aten_xla_type.h', lambda: {
        'dispatch_xla_declarations': list(concatMap(dest.compute_native_function_declaration, external_backend_functions)),
    })

    fm.write('aten_xla_type_default.h', lambda: {
        'dispatch_aten_fallback_declarations': list(concatMap(
            dest.GenExternalAtenFallback(Target.NAMESPACED_DECLARATION), external_backend_functions
        )),
    })

    fm.write('aten_xla_type_default.cpp', lambda: {
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
