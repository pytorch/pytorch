# Generates CodegenUnboxingWrappers.cpp.
# This generates static unboxing wrapper for ATen ops.
import argparse
import json
import os

from dataclasses import dataclass
from typing import Union, Sequence
from typing_extensions import Literal

from tools.codegen.api import unboxing, cpp
from tools.codegen.api.translate import translate
from tools.codegen.api.types import CppSignatureGroup, CType, BaseCType, voidT
from tools.codegen.context import method_with_native_function
from tools.codegen.gen import parse_native_yaml
from tools.codegen.model import NativeFunction, NativeFunctionsGroup, Variant
from tools.codegen.utils import Target, FileManager, mapMaybe, make_file_manager


# Generates CodegenFunctions.h & CodegenFunctions.cpp.
@dataclass(frozen=True)
class ComputeUnboxingFunctions:
    target: Union[Literal[Target.DECLARATION], Literal[Target.DEFINITION]]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        sig_group = CppSignatureGroup.from_native_function(
            f, method=(Variant.method in f.variants), fallback_binding=f.manual_cpp_binding
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
            expr_list, code_list = unboxing.convert_arguments(args)

            # for each C++ argument, generate the conversion code
            code_connector = "\n\t"
            arg_connector = ",\n\t\t"
            # function call and push back to stack
            prefix = "self_base." if sig.method else "at::"
            args_str = "" if not expr_list else f"""
        {arg_connector.join(e.expr for e in translate(expr_list, sig.arguments(), method=sig.method))}
    """
            ret_type: CType = cpp.returns_type(f.func.returns)
            if isinstance(ret_type, BaseCType) and ret_type.type == voidT:
                ret_str = ""
                push_str = ""
            else:
                ret_str = "auto result_ = "
                push_str = """
    pack(stack, std::move(result_));
                """
            return f"""
// aten::{f.func}
TORCH_API void {f.func.name.unambiguous_name()}(Stack & stack) {{
    {code_connector.join(code_list)}
    
    drop(stack, {len(args)});
    
    {ret_str}{prefix}{sig.name()}({args_str});
    {push_str}
}}
"""


@dataclass(frozen=True)
class ComputeUnboxingWrapper:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        # We unconditionally generate function wrappers,
        sig_group = CppSignatureGroup.from_native_function(
            f, method=(Variant.method in f.variants), fallback_binding=f.manual_cpp_binding
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
        "UnboxingFunctions.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            "definitions": [ComputeUnboxingFunctions(Target.DEFINITION)(fn)]
        },
        num_shards=5,
        sharded_keys={"definitions"},
    )
    cpu_fm.write(
        "UnboxingFunctions.h",
        lambda: {
            "declarations": list(
                mapMaybe(ComputeUnboxingFunctions(Target.DECLARATION), native_functions)
            ),
        },
    )
    cpu_fm.write_sharded(
        "RegisterCodegenUnboxedKernels.cpp",
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
    parser.add_argument(
        '--dry-run', action='store_true',
        help='run without writing any files (still updates outputs)')

    options = parser.parse_args()

    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )

    cpu_fm = make_file_manager(options=options)
    gen_unboxing(native_functions=native_functions, cpu_fm=cpu_fm)


if __name__ == "__main__":
    main()
