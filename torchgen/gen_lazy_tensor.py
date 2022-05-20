import pathlib
import argparse
import os
import re
import yaml
from collections import namedtuple, Counter
from typing import (
    List,
    Dict,
    Union,
    Sequence,
    Optional,
    Callable,
    Iterable,
    Iterator,
    Type,
)
from torchgen.api.types import BaseCppType
from torchgen.dest.lazy_ir import GenLazyIR, GenTSLazyIR
from torchgen.gen import (
    get_grouped_native_functions,
    parse_native_yaml,
    NamespaceHelper,
)

from torchgen.api.lazy import setValueT

from torchgen.model import (
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, YamlLoader, FileManager
import torchgen.dest as dest
from .gen_backend_stubs import (
    parse_backend_yaml,
    error_on_missing_kernels,
    gen_dispatchkey_nativefunc_headers,
    gen_dispatcher_registrations,
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                        Lazy Tensor Codegen
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Overview
# ~~~~~~~~
#
# This codegen script builds on existing data models and helpers used
# by all ATen backends, and adds new functionality specific to lazy
# tensor backends.
#
# Inputs:
# - <backend>_native_functions.yaml: controls which operators are
#   supported by the backend.
#
# Outputs:
# (for all backends)
# <DispatchKey>Ir.h defines Lazy IR classes to be constructed during tracing
# - opt-in: also generate 'lowering' methods for the TorchScript backend only
# <DispatchKey>NativeFunctions.cpp defines implementations of native functions which perform lazy tracing
# - opt-in: 'full_codegen' section of backend yaml; 'supported' section omits these implementations
# <DispatchKey>NativeFunctions.h declares implementations of native functions for both 'supported' and 'full_codegen'
# ops
#
# Register<DispatchKey>.cpp registers all op implementations with the dispatcher
# RegisterAutograd<DispatchKey>.cpp registers all autograd implementations with the dispatcher
#
# Validation Helpers:
# - Shape Inference: errs if any ops in backend yaml require shape inference not provided by meta kernels or
#   implementations in torch/csrc/lazy/core/shape_inference.*
# - native function impls: errs if any 'supported' ops do not have an implementation defined in the backend
#   (non-codegen) implementation file
#
#
# About the Data Model
# ~~~~~~~~~~~~~~~~~~~~
#
# Modeled after ATen codegen, the first step is to parse yaml and build a data model for the operators
# we care about.  In this case, the <backend>_native_functions yaml defines a subset of the core operators
# (defined in more detail in the main native_functions.yaml), which will be supported by your backend.
# Backends can list ops in two categories:
#  - `supported` ops require hand-implementations but still get codegenned declarations and registrations
#  - `full_codegen` ops get implementations (and IR classes) generated too
#
# Each native function is modeled as an object with a schema, and each schema has objects representing their
# arguments.  Much of the codegen is manipulation of the arguments and their types.  For example, lazy tensor
# backends need to transform 'at::Tensor' arguments into 'lazy::Value' objects, as well as replacing reference
# types (stringref) with actual string objects, and this is done by manipulating the data model objects.
# - see api/lazy.py for the lazy data model
#
# Once the data model is set up, the rest of this script processes a number of templates for output CPP file
# and fills in the template values using helpers in `dest/lazy_ir.py` and `dest/lazy_ts_lowering.py`.  These
# helpers mostly iterate over functions and their arguments, outputting different c++ snippets.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# Parses the external backend's yaml, and adds a new BackendIndex for the backend's dispatch key.
# Returns a Tuple of (backend_key, autograd_key, cpp_namespace, updated BackendIndex mapping, full_codegen)
ParsedExternalYaml = namedtuple(
    "ParsedExternalYaml",
    ["backend_key", "autograd_key", "cpp_namespace", "backend_indices", "full_codegen"],
)


def parse_full_codegen_ops(
    backend_yaml_path: str,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
) -> List[OperatorName]:

    native_functions_map: Dict[OperatorName, NativeFunction] = {
        f.func.name: f
        for f in concatMap(
            lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()),
            grouped_native_functions,
        )
    }

    with open(backend_yaml_path, "r") as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    assert isinstance(yaml_values, dict)

    full_codegen = yaml_values.pop("full_codegen", [])
    assert isinstance(
        full_codegen, list
    ), f'expected "full_codegen" to be a list, but got: {full_codegen}'
    full_codegen = [OperatorName.parse(name) for name in full_codegen]

    return full_codegen


def validate_shape_inference_header(
    shape_inference_hdr: str, expected_shape_infr_decls: List[str]
) -> None:
    try:
        with open(shape_inference_hdr, "r") as f:
            shape_infr_decls = f.read()
            shape_infr_decl_lines = set(shape_infr_decls.split("\n"))
    except IOError:
        raise AssertionError(
            f"Unable to read from the specified shape_inference_hdr file: {shape_inference_hdr}"
        )

    shape_infr_regex = r"compute_shape_(\w+)"
    actual_shape_infr_name_counts = Counter(
        re.findall(shape_infr_regex, shape_infr_decls)
    )
    # TODO(whc) add a check for shape inference functions that have meta kernels implement and should be retired.

    for decl in expected_shape_infr_decls:
        assert (
            decl in shape_infr_decl_lines
        ), f"""Missing shape inference function.\n
Please add declare this function in {shape_inference_hdr}:\n
and implement it in the the corresponding shape_inference.cpp file.\n
{decl}"""


class default_args:
    node_base: str = "Node"
    node_base_hdr: Optional[str] = None
    shape_inference_hdr: str = "torch/csrc/lazy/core/shape_inference.h"
    tensor_class: str = "torch::lazy::LazyTensor"
    tensor_class_hdr: str = "torch/csrc/lazy/core/tensor.h"
    lazy_ir_generator: Type[GenLazyIR] = GenLazyIR
    backend_name: str = "TorchScript"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lazy Tensor backend files")
    parser.add_argument(
        "-s",
        "--source_yaml",
        help="path to source yaml file containing operator external definitions",
    )
    parser.add_argument("-o", "--output_dir", help="output directory")
    parser.add_argument("--dry_run", type=bool, default=False, help="output directory")
    parser.add_argument(
        "--impl_path",
        type=str,
        default=None,
        help="path to the source C++ file containing kernel definitions",
    )
    parser.add_argument(
        "--gen_ts_lowerings",
        action="store_true",
        help="Generate TorchScript lowerings in addition to Lazy IR and NativeFunctions",
    )
    parser.add_argument(
        "--node_base",
        type=str,
        default=default_args.node_base,
        help="Name of backend specific custom Lazy IR Node base class",
    )
    parser.add_argument(
        "--node_base_hdr",
        type=str,
        default=default_args.node_base_hdr,
        help="Path to header file defining custom Lazy IR Node base class",
    )
    parser.add_argument(
        "--shape_inference_hdr",
        type=str,
        default=default_args.shape_inference_hdr,
        help="Path to header file defining custom Lazy shape inference functions",
    )
    parser.add_argument(
        "--tensor_class",
        type=str,
        default=default_args.tensor_class,
        help="Name of backend specific custom Lazy Tensor class",
    )
    parser.add_argument(
        "--tensor_class_hdr",
        type=str,
        default=default_args.tensor_class_hdr,
        help="Path to header file defining custom Lazy Tensor class",
    )
    parser.add_argument(
        "--backend_name",
        type=str,
        default=default_args.backend_name,
        help="Name of the backend to generate",
    )
    options = parser.parse_args()

    # Assumes that this file lives at PYTORCH_ROOT/torchgen/gen_backend_stubs.py
    torch_root = pathlib.Path(__file__).parent.parent.parent.absolute()
    aten_path = str(torch_root / "aten" / "src" / "ATen")
    lazy_ir_generator: Type[GenLazyIR] = default_args.lazy_ir_generator
    if options.gen_ts_lowerings:
        lazy_ir_generator = GenTSLazyIR

    run_gen_lazy_tensor(
        aten_path,
        options.source_yaml,
        options.output_dir,
        options.dry_run,
        options.impl_path,
        options.node_base,
        options.node_base_hdr,
        options.tensor_class,
        options.tensor_class_hdr,
        options.shape_inference_hdr,
        lazy_ir_generator,
        options.backend_name,
    )


def run_gen_lazy_tensor(
    aten_path: str,
    source_yaml: str,
    output_dir: str,
    dry_run: bool,
    impl_path: Optional[str],
    node_base: str = default_args.node_base,
    node_base_hdr: Optional[str] = default_args.node_base_hdr,
    tensor_class: str = default_args.tensor_class,
    tensor_class_hdr: str = default_args.tensor_class_hdr,
    shape_inference_hdr: str = default_args.shape_inference_hdr,
    lazy_ir_generator: Type[GenLazyIR] = default_args.lazy_ir_generator,
    # build_in_tree is true for TS backend and affects include paths
    build_in_tree: bool = False,
    # per_operator_headers changes whether ATen/Functions.h or individual operator headers are used
    # it must match how ATen was built
    per_operator_headers: bool = False,
    backend_name: str = default_args.backend_name,
    gen_forced_fallback_code: bool = False,
    # the following arguments are temporary customization points for xla backend migration.
    # do not rely on them otherwise, they should be removed once migration is complete
    backend_namespace: str = "torch::lazy",
    get_tensorlist: str = "GetTensorList",
    get_tensor_or_wrap_number: str = "GetLtcTensorOrCreateForWrappedNumber",
    try_get_tensor: str = "TryGetLtcTensor",
    metrics_counter: str = 'TORCH_LAZY_FN_COUNTER("lazy::")',
    create_tensor: str = "LazyTensor::Create",
    create_from_first_tensor: bool = False,
    create_aten_from_ltc_tensor: str = "torch::lazy::CreateAtenFromLtcTensor",
    tuple_aten_from_ltc_tensors: str = "torch::lazy::TupleAtenFromLtcTensors",
    lazy_value_class: str = "torch::lazy::Value",
    lazy_tensor_ptr: str = "LazyTensorPtr",
    get_device_fn: str = "torch::lazy::GetBackendDevice",
) -> None:
    lv_tokens = lazy_value_class.split("::")
    lv_class = lv_tokens[-1]
    lv_ns = "::".join(lv_tokens[:-1])
    setValueT(BaseCppType(lv_ns, lv_class))
    template_dir = os.path.join(aten_path, "templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir, dry_run=dry_run
        )

    fm = make_file_manager(output_dir)

    native_yaml_path = os.path.join(aten_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(aten_path, "native/tags.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    grouped_native_functions = get_grouped_native_functions(native_functions)

    def sort_native_function(f: Union[NativeFunctionsGroup, NativeFunction]) -> str:
        """
        We sort the native function because of the note in concat_map_codegen.
        TODO(alanwaketan): Remove this sorting hack once all ops are grouped properly.
        """
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        return str(func.name.name)

    grouped_native_functions = sorted(
        grouped_native_functions, key=sort_native_function
    )
    parsed_backend_yaml = parse_backend_yaml(
        source_yaml, grouped_native_functions, backend_indices
    )
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    backend_indices = parsed_backend_yaml.backend_indices
    full_codegen = parse_full_codegen_ops(source_yaml, grouped_native_functions)

    def concat_map_codegen(
        func: Callable[[NativeFunction], Sequence[str]],
        xs: Iterable[Union[NativeFunctionsGroup, NativeFunction]],
    ) -> Iterator[str]:
        """
        We code-gen for the functional variant, which is all we need for IR classes/lowerings/shape inferences, but we
        only code-gen additional entries for the inplace variant for the native functions.
        """

        for x in xs:
            fs = list(x.functions()) if isinstance(x, NativeFunctionsGroup) else [x]
            for f in fs:
                if f.func.name in full_codegen:
                    for r in func(f):
                        yield r

    selector = SelectiveBuilder.get_nop_selector()

    assert backend_key is not None
    class_name = backend_indices[backend_key].native_function_class_name()

    if impl_path is not None:
        error_on_missing_kernels(
            native_functions,
            backend_indices,
            backend_key,
            autograd_key,
            class_name,
            impl_path,
            full_codegen,
        )

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
                dest.GenLazyShapeInferenceDefinition(
                    backend_indices[backend_key], tensor_class
                ),
                grouped_native_functions,
            )
        )

        validate_shape_inference_header(shape_inference_hdr, expected_shape_infr_decls)
    assert class_name is not None

    # Generate nativefunction declarations
    # Note, eager registrations is set to False for the lazy TS backend as another LTC backend
    # may want to register their own lazy kernels instead of registering the TS ones.
    # The registration will lazily happen when init_ts_backend is called.
    gen_dispatchkey_nativefunc_headers(
        fm,
        class_name,
        cpp_namespace,
        backend_indices,
        grouped_native_functions,
        backend_key,
        autograd_key,
        backend_name,
    )

    # Generate Dispatcher registrations which hook up the nativefunctions
    for dispatch_key in (
        [backend_key] if autograd_key is None else [backend_key, autograd_key]
    ):
        gen_dispatcher_registrations(
            fm,
            output_dir,
            class_name,
            cpp_namespace,
            backend_indices,
            grouped_native_functions,
            backend_key,
            dispatch_key,
            selector,
            build_in_tree=build_in_tree,
            per_operator_headers=per_operator_headers,
            backend_name=backend_name,
            eager_registration=False,
        )

    # Generate native function impls that build IR nodes
    ns_helper = NamespaceHelper(cpp_namespace)
    fm.write_with_template(
        f"{backend_key}NativeFunctions.cpp",
        "DispatchKeyNativeFunctions.cpp",
        lambda: {
            "includes": [
                f"#include <{path}>"
                for path in [
                    tensor_class_hdr,
                    shape_inference_hdr,
                    "ATen/Functions.h",
                    "ATen/MetaFunctions.h",
                    "ATen/Operators.h",
                    "ATen/native/CPUFallback.h",
                    "torch/csrc/lazy/core/ir_builder.h",
                    "torch/csrc/lazy/core/lazy_graph_executor.h",
                    "torch/csrc/lazy/core/metrics.h",
                    "torch/csrc/lazy/core/shape.h",
                    f"{output_dir}/{backend_key}NativeFunctions.h",
                    f"{output_dir}/LazyIr.h",
                ]
                + (
                    ["torch/csrc/lazy/ts_backend/ts_eager_fallback.h"]
                    if gen_forced_fallback_code
                    else []
                )
            ],
            "native_functions_include": "",
            "namespace_prologue": ns_helper.prologue,
            "namespace_epilogue": ns_helper.epilogue,
            "native_function_definitions": list(
                concat_map_codegen(
                    dest.GenLazyNativeFuncDefinition(
                        f"{backend_key}NativeFunctions",
                        backend_indices[backend_key],
                        tensor_class,
                        gen_forced_fallback_code,
                        backend_namespace,
                        get_tensorlist,
                        get_tensor_or_wrap_number,
                        try_get_tensor,
                        metrics_counter,
                        create_tensor,
                        create_from_first_tensor,
                        create_aten_from_ltc_tensor,
                        tuple_aten_from_ltc_tensors,
                        lazy_tensor_ptr,
                        get_device_fn,
                    ),
                    grouped_native_functions,
                )
            ),
        },
    )
    # Generate IR node classes
    fm.write_with_template(
        "LazyIr.h",
        "LazyIr.h",
        lambda: {
            "lazy_ir_sysinc": [
                f"#include <{path}>"
                for path in [
                    "ATen/core/Formatting.h",
                    "c10/core/ScalarType.h",
                    "c10/util/Optional.h",
                    "torch/csrc/lazy/core/hash.h",
                    "torch/csrc/lazy/core/ir.h",
                    "torch/csrc/lazy/core/shape.h",
                    "vector",
                ]
            ],
            "lazy_ir_inc": [
                f'#include "{path}"'
                for path in [node_base_hdr if node_base_hdr is not None else None]
                if path is not None
            ],
            "ir_declarations": list(
                concat_map_codegen(
                    lazy_ir_generator(backend_indices[backend_key], node_base),
                    grouped_native_functions,
                )
            ),
            "namespace_prologue": ns_helper.prologue,
            "namespace_epilogue": ns_helper.epilogue,
        },
    )


if __name__ == "__main__":
    main()
