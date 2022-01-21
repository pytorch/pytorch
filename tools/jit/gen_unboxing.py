# Generates CodegenUnboxingWrappers.cpp.
# This generates static unboxing wrapper for ATen ops.
import argparse
import json
import os
from dataclasses import dataclass
from tools.codegen.api import unboxing
from tools.codegen.api.types import CppSignatureGroup
from tools.codegen.context import method_with_native_function
from tools.codegen.gen import parse_native_yaml
from tools.codegen.model import NativeFunction, NativeFunctionsGroup
from tools.codegen.utils import Target, FileManager, mapMaybe
from typing import Union, Sequence
from typing_extensions import Literal


# Generates CodegenFunctions.h & CodegenFunctions.cpp.
@dataclass(frozen=True)
class ComputeUnboxingFunctions:
    target: Union[Literal[Target.DECLARATION], Literal[Target.DEFINITION]]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding
        )
        sig = sig_group.most_faithful_signature()

        if self.target is Target.DECLARATION:
            # Note [The ATen Codegen Unboxing API]
            # Similar to the ATen Operators API, ATen Codegen Unboxing API lives in the at::unboxing namespace, and
            # will be used by codegen unboxing wrappers (CodegenUnboxingWrappers.cpp).
            # The Wrappers will be registered into torch::jit::OperatorRegistry using RegisterOperators API.
            #
            # Important characteristics about the Codegen Unboxing API:
            # (1) It follows the OperatorRegistry API.
            #     This is kind of necessary to avoid overhead.
            #     For example: if it followed the C++ API, then all of the faithful C++ factory functions
            #     would need to wrap their arguments into TensorOptions only to unwrap them again.
            # (2) Under the hood it calls C++ API.
            return f"""
// aten::{f.func}
TORCH_API void {f.func.name.unambiguous_name()}(Stack & stack);
"""
        else:
            # gather all the arguments from native function, including "out"
            args = list(f.func.arguments.flat_non_out) + (
                list(f.func.arguments.out) if f.func.arguments.out else []
            )

            # parse arguments into C++ code
            arguments = unboxing.convert_arguments(
                args, f.func.arguments.tensor_options
            )

            # for each C++ argument, generate the conversion code
            code_connector = "\n\t"
            code_list = []
            for arg in arguments:
                code_list.extend(arguments[arg].code)
            code = code_connector.join(code_list)

            # function call and push back to stack
            func_call_and_push = code_connector.join(
                unboxing.generate_unboxed_kernel_call(f, sig, arguments)
            )

            return f"""
// aten::{f.func}
TORCH_API void {f.func.name.unambiguous_name()}(Stack & stack) {{
    {code}
    drop(stack, {len(args)});
    {func_call_and_push}
}}
"""


@dataclass(frozen=True)
class ComputeUnboxingWrapper:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        # We unconditionally generate function wrappers,
        sig_group = CppSignatureGroup.from_native_function(
            f, method=False, fallback_binding=f.manual_cpp_binding
        )

        sig = sig_group.signature

        # escape double quote in schema, get rid of extra double quotes
        schema = json.dumps(sig.func.__str__())[1:-1]

        return f"""
OperatorGenerator(
    TORCH_SELECTIVE_SCHEMA("aten::{schema}"),
    [](Stack & stack) {{
        RECORD_FUNCTION("{sig.name()}", std::vector<c10::IValue>());
        at::unboxing::{f.func.name.unambiguous_name()}(stack);
    }},
    aliasAnalysisFromSchema()
),
"""


def gen_unboxing(
        *,
        native_functions: Sequence[NativeFunction],
        cpu_fm: FileManager,
) -> None:
    def key_func(fn: Union[NativeFunction, NativeFunctionsGroup]) -> str:
        return fn.root_name

    cpu_fm.write_sharded(
        "CodegenFunctions.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            "definitions": [ComputeUnboxingFunctions(Target.DEFINITION)(fn)]
        },
        num_shards=5,
        sharded_keys={"definitions"},
    )
    cpu_fm.write(
        "CodegenFunctions.h",
        lambda: {
            "declarations": list(
                mapMaybe(ComputeUnboxingFunctions(Target.DECLARATION), native_functions)
            ),
        },
    )
    cpu_fm.write_sharded(
        "CodegenUnboxingWrappers.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {"unboxed_ops": [ComputeUnboxingWrapper()(fn)]},
        num_shards=5,
        sharded_keys={"unboxed_ops"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unboxing source files")
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="aten/src/ATen",
    )
    parser.add_argument(
        "-d", "--install_dir", help="output directory", default="build/aten/src/ATen"
    )

    options = parser.parse_args()

    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )

    template_dir = os.path.join(options.source_path, "templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir, dry_run=False
        )

    cpu_fm = make_file_manager(options.install_dir)
    gen_unboxing(native_functions=native_functions, cpu_fm=cpu_fm)


if __name__ == "__main__":
    main()
