import pathlib
import argparse
import os
import re
import yaml
from collections import namedtuple, Counter
from typing import List, Dict, Union, Sequence, Optional, Callable, Iterable, Iterator, Tuple, Type
from tools.codegen.dest.lazy_ir import LazyIR, TSLazyIR
from tools.codegen.gen import get_grouped_native_functions, parse_native_yaml, NamespaceHelper
from tools.codegen.model import (FunctionSchema,
                                 NativeFunction, NativeFunctionsGroup, OperatorName)
from tools.codegen.selective_build.selector import SelectiveBuilder
from tools.codegen.utils import concatMap, YamlLoader, FileManager
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
        for f in concatMap(
            lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()), grouped_native_functions
        )
    }

    with open(backend_yaml_path, 'r') as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    assert isinstance(yaml_values, dict)

    full_codegen = yaml_values.pop('full_codegen', [])
    assert isinstance(full_codegen, list), f'expected "full_codegen" to be a list, but got: {full_codegen}'
    full_codegen = [OperatorName.parse(name) for name in full_codegen]

    return full_codegen

def validate_shape_inference_header(shape_inference_hdr: str, expected_shape_infr_decls: List[str]) -> None:
    try:
        with open(shape_inference_hdr, 'r') as f:
            shape_infr_decls = f.read()
            shape_infr_decl_lines = set(shape_infr_decls.split("\n"))
    except IOError:
        raise AssertionError(f'Unable to read from the specified shape_inference_hdr file: {shape_inference_hdr}')

    shape_infr_regex = r'compute_shape_(\w+)'
    actual_shape_infr_name_counts = Counter(re.findall(shape_infr_regex, shape_infr_decls))
    # TODO(whc) add a check for shape inference functions that have meta kernels implement and should be retired.

    for decl in expected_shape_infr_decls:
        assert decl in shape_infr_decl_lines, f"""Missing shape inference function.\n
Please add declare this function in {shape_inference_hdr}:\n
and implement it in the the corresponding shape_inference.cpp file.\n
{decl}"""

class default_args:
    node_base: str = "Node"
    node_base_hdr: Optional[str] = None
    shape_inference_hdr: str = "torch/csrc/lazy/core/shape_inference.h"
    tensor_class: str = "torch::lazy::LazyTensor"
    tensor_class_hdr: str = "torch/csrc/lazy/core/tensor.h"
    lazy_ir_cls: Type[LazyIR] = TSLazyIR

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
        '--gen_ts_lowerings', action="store_true",
        help='Generate TorchScript lowerings in addition to Lazy IR and NativeFunctions')
    parser.add_argument(
        '--node_base', type=str, default=default_args.node_base,
        help='Name of backend specific custom Lazy IR Node base class')
    parser.add_argument(
        '--node_base_hdr', type=str, default=default_args.node_base_hdr,
        help='Path to header file defining custom Lazy IR Node base class')
    parser.add_argument(
        '--shape_inference_hdr', type=str, default=default_args.shape_inference_hdr,
        help='Path to header file defining custom Lazy shape inference functions')
    parser.add_argument(
        '--tensor_class', type=str, default=default_args.tensor_class,
        help='Name of backend specific custom Lazy Tensor class')
    parser.add_argument(
        '--tensor_class_hdr', type=str, default=default_args.tensor_class_hdr,
        help='Path to header file defining custom Lazy Tensor class')
    options = parser.parse_args()

    # Assumes that this file lives at PYTORCH_ROOT/tools/codegen/gen_backend_stubs.py
    torch_root = pathlib.Path(__file__).parent.parent.parent.absolute()
    aten_path = str(torch_root / "aten" / "src" / "ATen")

    run_gen_lazy_tensor(aten_path, options.source_yaml, options.output_dir, options.dry_run, options.impl_path,
                        options.gen_ts_lowerings, options.node_base, options.node_base_hdr,
                        options.tensor_class, options.tensor_class_hdr, options.shape_inference_hdr,
                        default_args.lazy_ir_cls)


def run_gen_lazy_tensor(aten_path: str, source_yaml: str, output_dir: str,
                        dry_run: bool, impl_path: Optional[str],
                        gen_ts_lowerings: bool,
                        node_base: str = default_args.node_base,
                        node_base_hdr: Optional[str] = default_args.node_base_hdr,
                        tensor_class: str = default_args.tensor_class,
                        tensor_class_hdr: str = default_args.tensor_class_hdr,
                        shape_inference_hdr: str = default_args.shape_inference_hdr,
                        lazy_ir_cls: Type[LazyIR] = default_args.lazy_ir_cls,
                        per_operator_headers: bool = False) -> None:

    template_dir = os.path.join(aten_path, "templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=dry_run)

    fm = make_file_manager(output_dir)

    native_yaml_path = os.path.join(aten_path, 'native/native_functions.yaml')
    tags_yaml_path = os.path.join(aten_path, 'native/tags.yaml')
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = parsed_yaml.native_functions, parsed_yaml.backend_indices
    grouped_native_functions = get_grouped_native_functions(native_functions)

    def sort_native_function(f: Union[NativeFunctionsGroup, NativeFunction]) -> str:
        """
        We sort the native function because of the note in concat_map_codegen.
        TODO(alanwaketan): Remove this sorting hack once all ops are grouped properly.
        """
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        return str(func.name.name)

    grouped_native_functions = sorted(grouped_native_functions, key=sort_native_function)
    parsed_backend_yaml = parse_backend_yaml(source_yaml, grouped_native_functions, backend_indices)
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    backend_indices = parsed_backend_yaml.backend_indices
    full_codegen = parse_full_codegen_ops(source_yaml, grouped_native_functions)

    def concat_map_codegen(func: Callable[[NativeFunction], Sequence[str]],
                           xs: Iterable[Union[NativeFunctionsGroup, NativeFunction]],
                           *, codegenInplaceVariant: bool = False) -> Iterator[str]:
        """
        We code-gen for the functional variant, which is all we need for IR classes/lowerings/shape inferences, but we
        only code-gen additional entries for the inplace variant for the native functions.
        Note: If xs is not sorted, there may be an edge case when generating IR classes. Considering relu and relu_, if
        we encounter relu_ before relu. we will then generate an IR class with op = at::aten::relu_ for both relu and
        relu_ which will cause problems for relu.
        TODO(alanwaketan): Once all ops are grouped properly, we should no longer need this hack.
        """
        generated = set()

        def gen_key(func: FunctionSchema) -> Tuple[str, str]:
            # we want to generate unique entries for overloads of functional variants,
            # but not for inplace variants unless explicitly told `codegenInplaceVariant`
            return (func.name.name.base, func.name.overload_name)

        for x in xs:
            f = x.functional if isinstance(x, NativeFunctionsGroup) else x
            # For the 'or'd terms:
            # 1. codegenInplaceVariant means we can generate the in-place variant corresponding items.
            # 2. not f.func.name.name.inplace means the op is not a in-place variant, so we can generate the item.
            # 3. f.func.name.name.base not in generated means even for in-place ops we still need to generate the item
            # as if they were the functional variants for one time.
            if f.func.name in full_codegen and \
               (codegenInplaceVariant or not f.func.name.name.inplace or gen_key(f.func) not in generated):
                generated.add(gen_key(f.func))
                for r in func(f):
                    yield r

    selector = SelectiveBuilder.get_nop_selector()

    assert backend_key is not None
    class_name = backend_indices[backend_key].native_function_class_name()

    if impl_path is not None:
        error_on_missing_kernels(native_functions, backend_indices, backend_key,
                                 autograd_key, impl_path, full_codegen)


    """ Validate Shape Inference Definitions

    Generated lazy native functions all perform shape inference, by first using a meta:: kernel
    if available for that op, and otherwise using a 'compute_shape_{op}' function instead.  The generator
    knows the call signature for compute_shape_{op} becuase it matches the nativefunction (and meta::) signature,
    so it just has to check whether the op is structured and generate a call for one or the other.  It's up to the dev
    to supply the missing compute_shape_{op} function, but the codegen at least warns you about this and provides
    the expected signature which can be copy-pasted into shape_inference.h.

    compute_shape_{op} functions are handwritten and should be replaced over time as ops get ported
    to structured kernels.

    See torch/csrc/lazy/core/shape_inference.cpp #READ THIS! for more information.
    """
    if shape_inference_hdr is not None:
        expected_shape_infr_decls = list(
            concat_map_codegen(
                dest.GenLazyShapeInferenceDefinition(backend_indices[backend_key], tensor_class),
                grouped_native_functions,
                codegenInplaceVariant=True
            )
        )

        validate_shape_inference_header(shape_inference_hdr, expected_shape_infr_decls)
    assert class_name is not None

    # Generate nativefunction declarations
    gen_dispatchkey_nativefunc_headers(fm, class_name, cpp_namespace, backend_indices,
                                       grouped_native_functions, backend_key, autograd_key)

    # Generate Dispatcher registrations which hook up the nativefunctions
    for dispatch_key in [backend_key] if autograd_key is None else [backend_key, autograd_key]:
        gen_dispatcher_registrations(fm, output_dir, cpp_namespace, backend_indices, grouped_native_functions,
                                     backend_key, dispatch_key, selector,
                                     per_operator_headers=per_operator_headers)

    # Generate native function impls that build IR nodes
    ns_helper = NamespaceHelper(cpp_namespace)
    fm.write_with_template(f'{backend_key}NativeFunctions.cpp', 'DispatchKeyNativeFunctions.cpp', lambda: {
        'includes': [f'#include <{path}>' for path in [
            tensor_class_hdr,
            shape_inference_hdr,
            "ATen/Functions.h",
            "ATen/MetaFunctions.h",
            "ATen/Operators.h",
            "torch/csrc/lazy/core/lazy_graph_executor.h",
            "torch/csrc/lazy/core/metrics.h",
            "torch/csrc/lazy/core/shape.h",
            "lazy_tensor_core/csrc/ts_backend/aten_eager_fallback.h",
            f"{output_dir}/{backend_key}NativeFunctions.h",
            f"{output_dir}/{backend_key}LazyIr.h",
        ]],
        'native_functions_include': '',
        'namespace_prologue': ns_helper.prologue,
        'namespace_epilogue': ns_helper.epilogue,
        'native_function_definitions':
        list(concat_map_codegen(
            dest.GenLazyNativeFuncDefinition(f'{backend_key}NativeFunctions',
                                             backend_indices[backend_key],
                                             tensor_class),
            grouped_native_functions,
            codegenInplaceVariant=True
        )),
    })
    # Generate IR node classes
    fm.write_with_template('LazyIr.h', 'LazyIr.h', lambda: {
        'lazy_ir_sysinc': [f'#include <{path}>' for path in [
            "ATen/core/Formatting.h",
            "c10/core/ScalarType.h",
            "c10/util/Optional.h",
            "torch/csrc/lazy/core/hash.h",
            "torch/csrc/lazy/core/ir.h",
            "torch/csrc/lazy/core/shape.h",
            "vector",
        ]],
        'lazy_ir_inc': [f'#include "{path}"' for path in [
            node_base_hdr if node_base_hdr is not None else None
        ] if path is not None],
        'ir_declarations': list(concat_map_codegen(
            lazy_ir_cls(backend_indices[backend_key], node_base),
            grouped_native_functions
        )),
    })


if __name__ == '__main__':
    main()
