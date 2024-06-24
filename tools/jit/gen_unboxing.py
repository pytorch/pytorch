# Generates RegisterCodegenUnboxedKernels.cpp, UnboxingFunctions.h and UnboxingFunctions.cpp.
import argparse
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import List, Literal, Sequence, Union

import yaml

from torchgen.api import cpp, unboxing
from torchgen.api.translate import translate
from torchgen.api.types import CppSignatureGroup
from torchgen.api.unboxing import convert_arguments
from torchgen.context import method_with_native_function
from torchgen.gen import cpp_string, get_custom_build_selector, parse_native_yaml
from torchgen.model import Argument, NativeFunction, NativeFunctionsGroup, Variant
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import FileManager, make_file_manager, mapMaybe, Target


# Generates UnboxingFunctions.h & UnboxingFunctions.cpp.
@dataclass(frozen=True)
class ComputeUnboxingFunctions:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        if not self.selector.is_root_operator(f"aten::{f.func.name}"):
            return ""

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
            sig_group = CppSignatureGroup.from_native_function(
                f, method=(Variant.method in f.variants)
            )
            sig = sig_group.most_faithful_signature()
            # parse arguments into C++ code
            binding_list, code_list = convert_arguments(f)

            # for each C++ argument, generate the conversion code
            code_connector = "\n\t"
            arg_connector = ", "
            # function call and push back to stack
            prefix = "self_base." if sig.method else "at::"
            translated_args = translate(
                binding_list, sig.arguments(), method=sig.method
            )
            args_str = f"{arg_connector.join(e.expr for e in translated_args)}"
            if len(f.func.returns) == 0:
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

    drop(stack, {len(binding_list)});

    {ret_str}{prefix}{sig.name()}({args_str});
    {push_str}
}}
"""


# Generates RegisterCodegenUnboxedKernels.cpp.
@dataclass(frozen=True)
class ComputeCodegenUnboxedKernels:
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        if not self.selector.is_root_operator(f"aten::{f.func.name}"):
            return ""
        # We unconditionally generate function wrappers,
        sig_group = CppSignatureGroup.from_native_function(f, method=False)

        sig = sig_group.most_faithful_signature()

        # escape double quote in schema, get rid of extra double quotes
        schema = cpp_string(str(sig.func))[1:-1]

        # arguments
        args = sig.arguments()
        connector = ",\n\t\t"
        args_code = []
        for arg in args:
            # Using method=False faithful C++ API, so we should not see SelfArgument/TensorOptionsArgument
            assert isinstance(arg.argument, Argument)
            if not arg.argument.default:
                arg_cpp = "c10::IValue(c10::nullopt)"
            else:
                # The unboxing code uses the faithful C++ API to avoid the overhead
                # from wrapping/unwrapping TensorOptios.
                # However, we would look to include default args for schema parsing.
                # Default args only show up in the nonfaithful C++ API,
                arg_default = cpp.default_expr(
                    arg.argument.default, arg.argument.type, symint=False
                )
                if arg_default.startswith("{"):
                    arg_cpp = f"c10::IntArrayRef({arg_default})"
                else:
                    arg_cpp = f"c10::IValue({arg_default})"
            args_code.append(
                f"""c10::Argument("{arg.name}", nullptr, c10::nullopt, {arg_cpp})"""
            )

        returns = f.func.returns
        returns_code = []
        for ret in returns:
            returns_code.append(f"""c10::Argument("{ret.name if ret.name else ""}")""")
        return f"""
// aten::{schema}
OperatorGenerator(
    "aten::{f.func.name.name}",
    "{f.func.name.overload_name}",
    {{
        {connector.join(args_code)}
    }},
    {{
        {connector.join(returns_code)}
    }},
    [](Stack & stack) {{
        RECORD_FUNCTION("{sig.name()}", std::vector<c10::IValue>());
        at::unboxing::{unboxing.name(f)}(stack);
    }},
    aliasAnalysisFromSchema()
),
"""


def gen_unboxing(
    *,
    native_functions: Sequence[NativeFunction],
    cpu_fm: FileManager,
    selector: SelectiveBuilder,
) -> None:
    def key_func(fn: Union[NativeFunction, NativeFunctionsGroup]) -> str:
        return fn.root_name

    selected_op_num: int = len(selector.operators)
    # a best practice threshold of operators to enable sharding
    sharding_threshold: int = 100
    cpu_fm.write_sharded(
        "UnboxingFunctions.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            "definitions": [ComputeUnboxingFunctions(Target.DEFINITION, selector)(fn)]
        },
        num_shards=1 if selected_op_num < sharding_threshold else 5,
        sharded_keys={"definitions"},
    )
    cpu_fm.write(
        "UnboxingFunctions.h",
        lambda: {
            "declarations": list(
                mapMaybe(
                    ComputeUnboxingFunctions(Target.DECLARATION, selector),
                    native_functions,
                )
            ),
        },
    )
    cpu_fm.write_sharded(
        "RegisterCodegenUnboxedKernels.cpp",
        native_functions,
        key_fn=key_func,
        env_callable=lambda fn: {
            "unboxed_ops": [ComputeCodegenUnboxedKernels(selector)(fn)]
        },
        num_shards=1 if selected_op_num < sharding_threshold else 10,
        sharded_keys={"unboxed_ops"},
    )


def main(args: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Generate unboxing source files")
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="aten/src/ATen",
    )
    parser.add_argument(
        "-d",
        "--install-dir",
        "--install_dir",
        help="output directory",
        default="build/aten/src/ATen",
    )
    parser.add_argument(
        "-o",
        "--output-dependencies",
        help="output a list of dependencies into the given file and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
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
        "--op-registration-allowlist",
        "--op_registration_allowlist",
        nargs="*",
        help="filter op registrations by the allowlist (if set); "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )
    parser.add_argument(
        "--TEST-ONLY-op-registration-allowlist-yaml-path",
        "--TEST_ONLY_op_registration_allowlist_yaml_path",
        help="Provide a path to the operator selection (for custom build) YAML "
        "which contains a list of operators. It is to serve testing purpose and "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )

    options = parser.parse_args(args)
    if options.op_registration_allowlist:
        op_registration_allowlist = options.op_registration_allowlist
    elif options.TEST_ONLY_op_registration_allowlist_yaml_path:
        with open(options.TEST_ONLY_op_registration_allowlist_yaml_path) as f:
            op_registration_allowlist = yaml.safe_load(f)
    else:
        op_registration_allowlist = None

    selector = get_custom_build_selector(
        op_registration_allowlist,
        options.op_selection_yaml_path,
    )

    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(options.source_path, "native/tags.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )

    cpu_fm = make_file_manager(options=options)
    gen_unboxing(native_functions=native_functions, cpu_fm=cpu_fm, selector=selector)

    if options.output_dependencies:
        depfile_path = pathlib.Path(options.output_dependencies).resolve()
        depfile_name = depfile_path.name
        depfile_stem = depfile_path.stem

        path = depfile_path.parent / depfile_name
        cpu_fm.write_outputs(depfile_stem, str(path))


if __name__ == "__main__":
    main(sys.argv[1:])
