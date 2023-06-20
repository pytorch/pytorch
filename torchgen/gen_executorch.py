import argparse
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple, Union

import yaml

# Parse native_functions.yaml into a sequence of NativeFunctions and Backend Indices.
from torchgen import dest
from torchgen.api import cpp as aten_cpp
from torchgen.api.types import CppSignature, CppSignatureGroup, CType, NamedCType
from torchgen.context import (
    method_with_native_function,
    method_with_nested_native_function,
    with_native_function_and_index,
)
from torchgen.executorch.api import et_cpp
from torchgen.executorch.api.custom_ops import (
    ComputeNativeFunctionStub,
    gen_custom_ops_registration,
)
from torchgen.executorch.api.types import contextArg, ExecutorchCppSignature
from torchgen.executorch.api.unboxing import Unboxing
from torchgen.executorch.model import ETKernelIndex, ETKernelKey, ETParsedYaml
from torchgen.executorch.parse import ET_FIELDS, parse_et_yaml, parse_et_yaml_struct
from torchgen.gen import (
    get_custom_build_selector,
    get_native_function_declarations,
    get_native_function_declarations_from_ns_grouped_kernels,
    get_native_function_schema_registrations,
    LineLoader,
    parse_native_yaml,
)
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DEFAULT_KERNEL_NAMESPACE,
    DispatchKey,
    FunctionSchema,
    Location,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    Variant,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
    context,
    FileManager,
    make_file_manager,
    mapMaybe,
    NamespaceHelper,
)


def _sig_decl_wrapper(sig: Union[CppSignature, ExecutorchCppSignature]) -> str:
    """
    A wrapper function to basically get `sig.decl(include_context=True)`.
    For ATen kernel, the codegen has no idea about ET contextArg, so we
    use this wrapper to add it.
    """
    if isinstance(sig, ExecutorchCppSignature):
        return sig.decl()

    returns_type = aten_cpp.returns_type(sig.func.returns).cpp_type()
    cpp_args = [a.decl() for a in sig.arguments()]
    cpp_args_str = ", ".join([contextArg.decl()] + cpp_args)
    sig_decl = f"{returns_type} {sig.name()}({cpp_args_str})"
    return sig_decl


def static_dispatch(
    sig: Union[CppSignature, ExecutorchCppSignature],
    f: NativeFunction,
    backend_indices: List[BackendIndex],
) -> str:
    """
    For a given `NativeFunction`, find out the corresponding native function and dispatch to it. If zero or more than one
    native function exists, error out. A simplified version of register_dispatch_key.py
    Arguments:
        sig: A CppSignature for this native function we want to use.
        f: NativeFunction to generate static dispatch.
        backend_indices: All available backends.
    Return:
        C++ code to call backend-specific functions, e.g., "return at::native::add(self, other, scale);"
    """
    if len(backend_indices) == 0 or f.manual_kernel_registration:
        return ""

    backends = [b for b in backend_indices if b.has_kernel(f)]
    static_block = None
    if len(backends) == 1:
        backend_metadata = backends[0].get_kernel(f)
        if backend_metadata:
            args = ", ".join(a.name for a in sig.arguments())
            # Here we are assuming there's no difference between CppSignature and NativeSignature for Executorch.
            static_block = f"return ::{backend_metadata.cpp_namespace}::{backend_metadata.kernel}({args});"
    else:
        static_block = f"""
ET_ASSERT_UNREACHABLE_MSG("The number of native function(s) binding to {f.func.name} is {len(backends)}.");
    """
    return f"""
// {f.namespace}::{f.func}
TORCH_API inline {_sig_decl_wrapper(sig)} {{
    {static_block}
}}
"""


# Generates Functions.h, which provides the functional public C++ API,
# and the scaffolding to call into the dispatcher from these functions.
@dataclass(frozen=True)
class ComputeFunction:
    static_dispatch_backend_indices: List[BackendIndex]

    selector: SelectiveBuilder

    use_aten_lib: bool

    is_custom_op: Callable[[NativeFunction], bool]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if not self.selector.is_root_operator(f"{f.namespace}::{f.func.name}"):
            return None
        if Variant.function not in f.variants:
            return None
        sig: Union[CppSignature, ExecutorchCppSignature] = (
            CppSignatureGroup.from_native_function(
                f, method=False, fallback_binding=f.manual_cpp_binding
            ).most_faithful_signature()
            if self.use_aten_lib
            else ExecutorchCppSignature.from_native_function(f)
        )
        if self.use_aten_lib and not self.is_custom_op(f):
            comma = ", "

            return f"""
// {f.namespace}::{f.func}
TORCH_API inline {_sig_decl_wrapper(sig)} {{
    return at::{sig.name()}({comma.join(e.name for e in sig.arguments())});
}}
"""

        else:
            return static_dispatch(
                sig,
                f,
                backend_indices=self.static_dispatch_backend_indices,
            )


# Generates RegisterCodegenUnboxedKernels.cpp.
@dataclass(frozen=True)
class ComputeCodegenUnboxedKernels:
    selector: SelectiveBuilder

    use_aten_lib: bool

    @method_with_nested_native_function
    def __call__(
        self,
        unbox_kernel_entry: Tuple[NativeFunction, Tuple[ETKernelKey, BackendMetadata]],
    ) -> str:
        f: NativeFunction = unbox_kernel_entry[0]
        kernel_key: Union[ETKernelKey, List[ETKernelKey]] = unbox_kernel_entry[1][0]
        kernel_meta: BackendMetadata = unbox_kernel_entry[1][1]

        op_name = f"{f.namespace}::{f.func.name}"
        if not self.selector.is_root_operator(op_name):
            return ""

        if not isinstance(kernel_key, list):
            kernel_key = [kernel_key]
        used_kernel_keys = self.selector.et_get_selected_kernels(
            op_name, [k.to_native_string() for k in kernel_key]
        )
        if not used_kernel_keys:
            return ""
        sig: Union[CppSignature, ExecutorchCppSignature]
        argument_type_gen: Callable[..., NamedCType]
        return_type_gen: Callable[..., CType]
        if self.use_aten_lib:
            sig = CppSignatureGroup.from_native_function(
                f, method=False, fallback_binding=f.manual_cpp_binding
            ).most_faithful_signature()
            argument_type_gen = aten_cpp.argumenttype_type
            return_type_gen = aten_cpp.returns_type
            arguments = sig.arguments()
            kernel_call = f"torch::executor::{f.namespace}::{sig.name()}"
        else:
            sig = ExecutorchCppSignature.from_native_function(f)
            argument_type_gen = et_cpp.argumenttype_type
            return_type_gen = et_cpp.returns_type
            arguments = sig.arguments(include_context=False)
            kernel_call = f"{kernel_meta.cpp_namespace}::{kernel_meta.kernel}"
        # parse arguments into C++ code
        binding_list, code_list = Unboxing(
            argument_type_gen=argument_type_gen
        ).convert_arguments(arguments)

        # for each C++ argument, generate the conversion code
        code_connector = "\n\t"
        arg_connector = ", "

        args_str = f"{arg_connector.join(e.name for e in binding_list)}"

        if len(f.func.returns) == 0:
            if len(f.func.arguments.out) == 0:
                raise Exception(
                    f"Can't handle native function {f.func} with no returns and no out yet."
                )
            out = f.func.arguments.out[0]
            return_assignment = f"""stack[{len(binding_list)}] = &{out.name};"""
            ret_prefix = ""
        else:
            if len(f.func.arguments.out) == 0:
                return_assignment = (
                    f"""*stack[{len(binding_list)}] = EValue(result_);"""
                )
                ret_prefix = return_type_gen(f.func.returns).cpp_type() + " result_ = "
            else:
                return_assignment = ""
                ret_prefix = ""

        newline = "\n    "
        return "\n".join(
            [
                f"""
Kernel(
    "{f.namespace}::{f.func.name}",{newline + '"' + (k + '",') if k != 'default' else ''}
    []({contextArg.defn()}, EValue** stack) {{
        {code_connector.join(code_list)}

        EXECUTORCH_SCOPE_PROF("native_call_{f.func.name}");
        {ret_prefix}{kernel_call}(context, {args_str});

        {return_assignment}
    }}
),
"""
                for k in used_kernel_keys
            ]
        )


def gen_unboxing(
    *,
    native_functions: Sequence[NativeFunction],
    cpu_fm: FileManager,
    selector: SelectiveBuilder,
    use_aten_lib: bool,
    kernel_index: ETKernelIndex,
) -> None:
    # Iterable type for write_sharded is a Tuple of (native_function, (kernel_key, metadata))
    def key_func(
        item: Tuple[NativeFunction, Tuple[ETKernelKey, BackendMetadata]]
    ) -> str:
        return item[0].root_name + ":" + item[1][0].to_native_string()

    items: List[Tuple[NativeFunction, Tuple[ETKernelKey, BackendMetadata]]] = [
        (native_function, (kernel_key, metadata))
        for native_function in native_functions
        for kernel_key, metadata in kernel_index.get_kernels(native_function).items()
    ]

    header = ["Functions.h" if use_aten_lib else "NativeFunctions.h"]

    cpu_fm.write_sharded(
        "RegisterCodegenUnboxedKernels.cpp",
        items,
        key_fn=key_func,
        env_callable=lambda unbox_kernel_entry: {
            "unboxed_kernels": [
                ComputeCodegenUnboxedKernels(selector, use_aten_lib)(unbox_kernel_entry)
            ],
            "fn_header": header
            if unbox_kernel_entry == items[0]
            else [],  # Only write header once
        },
        num_shards=1,
        sharded_keys={"unboxed_kernels", "fn_header"},
    )


@with_native_function_and_index  # type: ignore[arg-type]
def compute_native_function_declaration(
    g: Union[NativeFunctionsGroup, NativeFunction], kernel_index: ETKernelIndex
) -> List[str]:
    assert isinstance(g, NativeFunction)
    sig = ExecutorchCppSignature.from_native_function(f=g)
    metadata_list = kernel_index.get_kernels(g).values()
    if metadata_list is None:
        return []
    prefix = "TORCH_API"

    # for kernels in lean mode, we declare two versions, one with context and one without.
    # In the end we will cleanup the unused one.
    def gen_decl(metadata: BackendMetadata, include_context: bool) -> str:
        return f"{prefix} {sig.decl(name=metadata.kernel, include_context=include_context)};"

    return [
        gen_decl(metadata, include_context)
        for include_context in [False, True]
        for metadata in metadata_list
    ]


def gen_functions_declarations(
    *,
    native_functions: Sequence[NativeFunction],
    kernel_index: ETKernelIndex,
    selector: SelectiveBuilder,
    use_aten_lib: bool,
    custom_ops_native_functions: Optional[Sequence[NativeFunction]] = None,
) -> str:
    """
    Generates namespace separated C++ function API inline declaration/definitions.
    Native functions are grouped by namespaces and the generated code is wrapped inside
    namespace blocks.

    E.g., for `custom_1::foo.out` in yaml file we will generate a C++ API as a symbol
    in `torch::executor::custom_1::foo_out`. This way we avoid symbol conflict when
    the other `custom_2::foo.out` is available.
    """

    # convert kernel index to BackendIndex. This is because we can't handle ETKernelIndex yet.
    # TODO larryliu: evaluate if this code is still needed. If yes let it handle ETKernelIndex.

    dispatch_key = DispatchKey.CPU
    backend_index = kernel_index._to_backend_index()

    ns_grouped_functions = defaultdict(list)
    for native_function in native_functions:
        ns_grouped_functions[native_function.namespace].append(native_function)
    functions_declarations = ""
    newline = "\n"
    for namespace in ns_grouped_functions:
        ns_helper = NamespaceHelper(
            namespace_str=namespace,
            entity_name="",
            max_level=3,
        )
        declarations = list(
            mapMaybe(
                ComputeFunction(
                    static_dispatch_backend_indices=[backend_index],
                    selector=selector,
                    use_aten_lib=use_aten_lib,
                    is_custom_op=lambda f: custom_ops_native_functions is not None
                    and f in custom_ops_native_functions,
                ),
                ns_grouped_functions[namespace],
            )
        )
        functions_declarations += f"""
{ns_helper.prologue}
{newline.join(declarations)}
{ns_helper.epilogue}
        """
    return functions_declarations


def get_ns_grouped_kernels(
    *,
    native_functions: Sequence[NativeFunction],
    kernel_index: ETKernelIndex,
    native_function_decl_gen: Callable[
        [
            Union[NativeFunctionsGroup, NativeFunction],
            ETKernelIndex,
        ],
        List[str],
    ],
) -> Dict[str, List[str]]:
    ns_grouped_kernels: Dict[str, List[str]] = defaultdict(list)
    for f in native_functions:
        native_function_namespaces = set()
        op_kernels = kernel_index.get_kernels(f)
        for backend_metadata in op_kernels.values():
            if backend_metadata:
                namespace = backend_metadata.cpp_namespace
                native_function_namespaces.add(namespace)
            else:
                namespace = DEFAULT_KERNEL_NAMESPACE
            assert (
                len(native_function_namespaces) <= 1
            ), f"Codegen only supports one namespace per operator, got {native_function_namespaces}"
            ns_grouped_kernels[namespace].extend(
                native_function_decl_gen(f, kernel_index)
            )
    return ns_grouped_kernels


def gen_headers(
    *,
    native_functions: Sequence[NativeFunction],
    gen_custom_ops_header: bool,
    custom_ops_native_functions: Sequence[NativeFunction],
    selector: SelectiveBuilder,
    kernel_index: ETKernelIndex,
    cpu_fm: FileManager,
    use_aten_lib: bool,
) -> None:
    """Generate headers.

    Args:
        native_functions (Sequence[NativeFunction]): a collection of NativeFunction for ATen ops.
        gen_custom_ops_header (bool): whether we should generate CustomOpsNativeFunctions.h
        custom_ops_native_functions (Sequence[NativeFunction]): a collection of NativeFunction for custom ops.
        kernel_index (ETKernelIndex): kernel collection
        cpu_fm (FileManager): file manager manages output stream
        use_aten_lib (bool): whether we are generating for PyTorch types or Executorch types.
    """
    aten_headers = ["#include <ATen/Functions.h>"]
    backend_indices = {DispatchKey.CPU: kernel_index._to_backend_index()}
    if gen_custom_ops_header:
        cpu_fm.write_with_template(
            "CustomOpsNativeFunctions.h",
            "NativeFunctions.h",
            lambda: {
                "nativeFunctions_declarations": get_native_function_declarations(
                    grouped_native_functions=custom_ops_native_functions,
                    backend_indices=backend_indices,
                    native_function_decl_gen=dest.compute_native_function_declaration,
                ),
            },
        )
        aten_headers.append('#include "CustomOpsNativeFunctions.h"')
    cpu_fm.write(
        "Functions.h",
        lambda: {
            "static_dispatch_extra_headers": aten_headers
            if use_aten_lib
            else ['#include "NativeFunctions.h"'],
            "Functions_declarations": gen_functions_declarations(
                native_functions=native_functions,
                kernel_index=kernel_index,
                selector=selector,
                use_aten_lib=use_aten_lib,
                custom_ops_native_functions=custom_ops_native_functions,
            ),
        },
    )
    if use_aten_lib:
        cpu_fm.write(
            "NativeFunctions.h",
            lambda: {
                "nativeFunctions_declarations": get_native_function_declarations(
                    grouped_native_functions=native_functions,
                    backend_indices=backend_indices,
                    native_function_decl_gen=dest.compute_native_function_declaration,
                ),
            },
        )
    else:
        ns_grouped_kernels = get_ns_grouped_kernels(
            native_functions=native_functions,
            kernel_index=kernel_index,
            native_function_decl_gen=compute_native_function_declaration,  # type: ignore[arg-type]
        )
        cpu_fm.write(
            "NativeFunctions.h",
            lambda: {
                "nativeFunctions_declarations": get_native_function_declarations_from_ns_grouped_kernels(
                    ns_grouped_kernels=ns_grouped_kernels,
                ),
            },
        )


def gen_custom_ops(
    *,
    native_functions: Sequence[NativeFunction],
    selector: SelectiveBuilder,
    kernel_index: ETKernelIndex,
    cpu_fm: FileManager,
    rocm: bool,
) -> None:
    dispatch_key = DispatchKey.CPU
    (
        anonymous_definition,
        static_init_dispatch_registrations,
    ) = gen_custom_ops_registration(
        native_functions=native_functions,
        selector=selector,
        kernel_index=kernel_index,
        rocm=rocm,
    )
    cpu_fm.write_with_template(
        f"Register{dispatch_key}CustomOps.cpp",
        "RegisterDispatchKeyCustomOps.cpp",
        lambda: {
            "ops_headers": '#include "CustomOpsNativeFunctions.h"',
            "DispatchKey": dispatch_key,
            "dispatch_namespace": dispatch_key.lower(),
            "dispatch_namespaced_definitions": "",
            "dispatch_anonymous_definitions": anonymous_definition,
            "static_init_dispatch_registrations": static_init_dispatch_registrations,
        },
    )
    cpu_fm.write_with_template(
        f"Register{dispatch_key}Stub.cpp",
        "RegisterDispatchKeyCustomOps.cpp",
        lambda: {
            "ops_headers": "",
            "DispatchKey": dispatch_key,
            "dispatch_namespace": dispatch_key.lower(),
            "dispatch_namespaced_definitions": "",
            "dispatch_anonymous_definitions": list(
                mapMaybe(ComputeNativeFunctionStub(), native_functions)
            ),
            "static_init_dispatch_registrations": static_init_dispatch_registrations,
        },
    )

    (
        aten_schema_registrations,
        schema_registrations,
    ) = get_native_function_schema_registrations(
        native_functions=native_functions,
        schema_selector=selector,
    )
    cpu_fm.write(
        "RegisterSchema.cpp",
        lambda: {
            "schema_registrations": schema_registrations,
            "aten_schema_registrations": aten_schema_registrations,
        },
    )


def translate_native_yaml(
    tags_yaml_path: str,
    aten_yaml_path: str,
    native_yaml_path: Optional[str],
    use_aten_lib: bool,
    out_file: TextIO,
) -> None:
    """Translates Executorch DSL dialect to use the same syntax as
    native_functions.yaml. The major difference is that Executorch DSL dialect
    supports "op" key, where it refers to the operator name in native_functions.yaml.

    For example, a functions.yaml may have the following entry:

    - op: add.out
      ...

    It needs to be translated to the following:

    - func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
      ...

    We go in aten_yaml_path and find the operator schema for "add.out" and add it
    to the original functions.yaml. We also add required field "variants", where for
    Executorch it will always be "function".

    For ATen mode we don't have to do the translation because native_yaml_path is
    the same as native_functions.yaml.

    Args:
        tags_yaml_path: Path to a tags.yaml file to satisfy codegen parsing.
            It is not optional.
        aten_yaml_path: Path to ATen operator yaml file native_functions.yaml.
        native_yaml_path: Path to a functions.yaml file to parse.
            If the path does not exist in the filesystem, it is treated as an
            empty file. If `custom_ops_yaml_path` exists, the contents of that
            file are appended to the yaml input to be parsed.
        use_aten_lib: We use this flag to determine if we want to generate native
            functions. In ATen mode we should generate out= variants.
        out_file: The IO object that we are writing into.
    Returns:
        None
    """
    if use_aten_lib:
        with open(aten_yaml_path, "r") as aten_yaml:
            out_file.writelines(aten_yaml.readlines())
        return

    native_functions, persisted_fields = parse_et_yaml(
        aten_yaml_path,
        tags_yaml_path,
        None,
        skip_native_fns_gen=False,
    )

    func_to_scoped_name: Dict[FunctionSchema, str] = {
        f.func: f"{f.namespace}::{f.func.name}" for f in native_functions
    }
    op_to_scoped_name: Dict[OperatorName, str] = {
        func.name: name for func, name in func_to_scoped_name.items()
    }

    schema_dict = {name: str(func) for func, name in func_to_scoped_name.items()}
    kernel_persist_dict: Dict[str, Dict[str, Any]] = {
        op_to_scoped_name[op]: v for op, v in persisted_fields.items()
    }

    if (
        not native_yaml_path
        or not os.path.exists(native_yaml_path)
        or os.stat(native_yaml_path).st_size == 0
    ):
        return
    with open(native_yaml_path, "r") as native_yaml:
        native_es = yaml.load(native_yaml, Loader=LineLoader)
        if not native_es:
            return
        for e in native_es:
            assert isinstance(e.get("__line__"), int), e
            loc = Location(native_yaml_path, e.pop("__line__"))
            with context(lambda: f"in {loc}:\n  "):
                if "variants" not in e:
                    e["variants"] = "function"
                if "func" in e:
                    continue
                assert isinstance(e.get("op"), str), e
                opname = e.pop("op")
                if "::" not in opname:
                    opname = "aten::" + opname
                assert opname in schema_dict
                e["func"] = schema_dict.get(opname)

                # Write out persisted kernel information
                if opname in kernel_persist_dict:
                    for k, v in kernel_persist_dict[opname].items():
                        e[k] = v

        yaml.dump(native_es, out_file, width=1000)


def parse_yaml(
    path: Optional[str],
    tags_yaml_path: str,
    function_filter: Callable[[NativeFunction], bool],
    skip_native_fns_gen: bool = False,
) -> Tuple[
    List[NativeFunction],
    Union[Dict[DispatchKey, Dict[OperatorName, BackendMetadata]], ETKernelIndex],
]:
    if path and os.path.exists(path) and os.stat(path).st_size > 0:
        with open(path, "r") as f:
            es = yaml.load(f, Loader=LineLoader)

        # Check for kernel index structure
        kernel_index = (
            parse_et_yaml_struct(es) if any("kernels" in e for e in es) else None
        )

        # Remove ET specific fields from entries for BC compatibility
        for entry in es:
            for field in ET_FIELDS:
                entry.pop(field, None)

        parsed_yaml = parse_native_yaml(
            path,
            tags_yaml_path,
            None,
            skip_native_fns_gen=skip_native_fns_gen,
            loaded_yaml=es,
        )
        native_functions = list(filter(function_filter, parsed_yaml.native_functions))
        op_names = [f.func.name for f in native_functions]

        # (1) Return ETKernelIndex if kernel index is present
        if kernel_index is not None:
            filtered_index = {
                op_name: kernel_mapping
                for op_name, kernel_mapping in kernel_index.index.items()
                if op_name in op_names
            }
            return native_functions, ETKernelIndex(index=filtered_index)

        # (2) Return BackendIndices if kernel index is absent
        def map_index(
            m: Dict[OperatorName, BackendMetadata]
        ) -> Dict[OperatorName, BackendMetadata]:
            return {op: m[op] for op in m if op in op_names}

        backend_indices = {
            k: map_index(b.index) for (k, b) in parsed_yaml.backend_indices.items()
        }

        return native_functions, backend_indices
    else:
        return [], {}


def parse_yaml_files(
    tags_yaml_path: str,
    aten_yaml_path: str,
    native_yaml_path: Optional[str],
    custom_ops_yaml_path: Optional[str],
    selector: SelectiveBuilder,
    use_aten_lib: bool,
) -> Tuple[ETParsedYaml, Optional[ETParsedYaml]]:
    """Parses functions.yaml and custom_ops.yaml files.

    Args:
        tags_yaml_path: Path to a tags.yaml file to satisfy codegen parsing.
            It is not optional.
        aten_yaml_path: Path to ATen operator yaml file native_functions.yaml.
        native_yaml_path: Path to a functions.yaml file to parse.
            If the path does not exist in the filesystem, it is treated as an
            empty file. If `custom_ops_yaml_path` exists, the contents of that
            file are appended to the yaml input to be parsed.
        custom_ops_yaml_path: Path to a custom_ops.yaml file to parse. If
            the path does not exist in the filesystem, it is ignored.
        selector: For selective build.
        use_aten_lib: We use this flag to determine if we want to generate native
            functions. In ATen mode we should generate out= variants.
    Returns:
        A tuple with two elements:
        [0]: The parsed results of concatenating the contents of
             `native_yaml_path` and `custom_ops_yaml_path`.
        [1]: The parsed results of the contents of `custom_ops_yaml_path`, if
             present. If not present, None.
    """
    import tempfile

    # only include selected ops, this is because we want to avoid
    def function_filter(f: NativeFunction) -> bool:
        return selector.is_native_function_selected(f)

    with tempfile.TemporaryDirectory() as tmpdirname:
        translated_yaml_path = os.path.join(tmpdirname, "translated.yaml")
        with open(translated_yaml_path, "w") as translated:
            translate_native_yaml(
                tags_yaml_path,
                aten_yaml_path,
                native_yaml_path,
                use_aten_lib,
                translated,
            )

        translated_functions, translated_indices = parse_yaml(
            translated_yaml_path, tags_yaml_path, function_filter, not use_aten_lib
        )
        custom_ops_functions, custom_ops_indices = parse_yaml(
            custom_ops_yaml_path, tags_yaml_path, function_filter, True
        )

        # Convert BackendIndices to ETKernelIndex
        if not isinstance(translated_indices, ETKernelIndex):
            translated_indices = ETKernelIndex.from_backend_indices(translated_indices)
        if not isinstance(custom_ops_indices, ETKernelIndex):
            custom_ops_indices = ETKernelIndex.from_backend_indices(custom_ops_indices)

        combined_functions = translated_functions + custom_ops_functions
        combined_kernel_index = ETKernelIndex.merge_indices(
            translated_indices, custom_ops_indices
        )
        combined_yaml = ETParsedYaml(combined_functions, combined_kernel_index)
        custom_ops_parsed_yaml = ETParsedYaml(custom_ops_functions, custom_ops_indices)

    return combined_yaml, custom_ops_parsed_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate operator source files")
    # Although we don't refer to --source-path directly, make_file_manager()
    # expects it to point to a directory that contains a templates/ subdirectory
    # containing the file templates.
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for kernel templates",
    )
    parser.add_argument(
        "--functions-yaml-path",
        "--functions_yaml_path",
        help="path to the functions.yaml file to use. Optional, but at least "
        "one of --functions-yaml-path and --custom-ops-yaml-path must be "
        "specified.",
    )
    parser.add_argument(
        "--custom-ops-yaml-path",
        "--custom_ops_yaml_path",
        help="path to the custom_ops.yaml file to use. Optional, but at least "
        "one of --functions-yaml-path and --custom-ops-yaml-path must be "
        "specified.",
    )
    parser.add_argument(
        "--aten-yaml-path",
        "--aten_yaml_path",
        help="path to native_functions.yaml file.",
    )
    # Note that make_file_manager() also looks at --install-dir.
    parser.add_argument(
        "-d",
        "--install-dir",
        "--install_dir",
        help="output directory",
        default="build/generated",
    )
    parser.add_argument(
        "-o",
        "--output-dependencies",
        help="output a list of dependencies into the given file and exit",
    )
    # Although we don't refer to --dry-run directly, make_file_manager() looks
    # for it.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
    )
    parser.add_argument(
        "--static-dispatch-backend",
        "--static_dispatch_backend",
        nargs="*",
        help="generate static dispatch code for the specific backend (if set)",
    )
    parser.add_argument(
        "--op-registration-whitelist",
        "--op_registration_whitelist",
        nargs="*",
        help="filter op registrations by the whitelist (if set); "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )
    parser.add_argument(
        "--op-selection-yaml-path",
        "--op_selection_yaml_path",
        help="Provide a path to the operator selection (for custom build) YAML "
        "that contains the information about the set of selected operators "
        "and their categories (training, ...). Each operator is either a "
        "full operator name with overload or just a bare operator name. "
        "The operator names also contain the namespace prefix (e.g. aten::)",
    )
    parser.add_argument(
        "--tags-path",
        help="Path to tags.yaml. Required by yaml parsing in codegen system.",
    )
    parser.add_argument(
        "--rocm",
        action="store_true",
        help="reinterpret CUDA as ROCm/HIP and adjust filepaths accordingly",
    )
    parser.add_argument(
        "--use-aten-lib",
        "--use_aten_lib",
        action="store_true",
        help="a boolean flag to indicate whether we use ATen kernels or not, in the future this flag will be per "
        "operator",
    )
    parser.add_argument(
        "--generate",
        type=str,
        nargs="*",
        choices=["headers", "sources"],
        default=["headers", "sources"],
        help="Generate only a subset of files",
    )
    options = parser.parse_args()
    assert options.tags_path, "tags.yaml is required by codegen yaml parsing."

    selector = get_custom_build_selector(
        options.op_registration_whitelist,
        options.op_selection_yaml_path,
    )

    parsed_yaml, custom_ops_parsed_yaml = parse_yaml_files(
        aten_yaml_path=options.aten_yaml_path,
        tags_yaml_path=options.tags_path,
        native_yaml_path=options.functions_yaml_path,
        custom_ops_yaml_path=options.custom_ops_yaml_path,
        selector=selector,
        use_aten_lib=options.use_aten_lib,
    )
    native_functions, kernel_index = (
        parsed_yaml.native_functions,
        parsed_yaml.kernel_index,
    )
    custom_ops_native_functions = (
        custom_ops_parsed_yaml.native_functions if custom_ops_parsed_yaml else []
    )

    cpu_fm = make_file_manager(options=options)

    if "headers" in options.generate:
        # generate CustomOpsNativeFunctions.h when custom_ops.yaml is present, to match the build system.
        gen_headers(
            native_functions=native_functions,
            gen_custom_ops_header=options.custom_ops_yaml_path,
            custom_ops_native_functions=custom_ops_native_functions,
            selector=selector,
            kernel_index=kernel_index,
            cpu_fm=cpu_fm,
            use_aten_lib=options.use_aten_lib,
        )

    if "sources" in options.generate:
        gen_unboxing(
            native_functions=native_functions,
            cpu_fm=cpu_fm,
            selector=selector,
            use_aten_lib=options.use_aten_lib,
            kernel_index=kernel_index,
        )
        if custom_ops_native_functions:
            gen_custom_ops(
                native_functions=custom_ops_native_functions,
                selector=selector,
                kernel_index=kernel_index,
                cpu_fm=cpu_fm,
                rocm=options.rocm,
            )

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


if __name__ == "__main__":
    main()
