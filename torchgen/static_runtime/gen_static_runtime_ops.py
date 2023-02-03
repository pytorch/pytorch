import argparse
import itertools
import os
from typing import Sequence, TypeVar, Union

from libfb.py.log import set_simple_logging  # type: ignore[import]

from torchgen import gen
from torchgen.context import native_function_manager
from torchgen.model import DispatchKey, NativeFunctionsGroup, NativeFunctionsViewGroup
from torchgen.static_runtime import config, generator

# Given a list of `grouped_native_functions` sorted by their op names, return a list of
# lists each of which groups ops that share the base name. For example, `mean` and
# `mean.dim` are grouped together by this function.

NativeGroupT = TypeVar(
    "NativeGroupT",
    bound=Union[NativeFunctionsGroup, NativeFunctionsViewGroup],
)


def group_functions_by_op_name(
    grouped_native_functions: Sequence[NativeGroupT],
) -> Sequence[Sequence[NativeGroupT]]:
    if not grouped_native_functions:
        return []
    groups = []

    def is_supported(g: Union[NativeFunctionsGroup, NativeFunctionsViewGroup]) -> bool:
        with native_function_manager(g):
            return generator.is_supported(g)

    eligible_ops = (g for g in grouped_native_functions if is_supported(g))
    groups = [
        list(group)
        for k, group in (
            itertools.groupby(
                eligible_ops,
                key=lambda g: config.func_name_base_str(g),
            )
        )
    ]

    return groups


def clang_format(cpp_file_path: str) -> None:
    import subprocess

    subprocess.run(["clang-format", "-i", cpp_file_path])


def write_cpp(cpp_ops: Sequence[str], file_path: str) -> None:
    code = "\n".join(cpp_ops)
    generated = f"""// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/te_wrapper.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

namespace torch {{
namespace jit {{

{code}

}} // namespace jit
}} // namespace torch
"""
    with open(file_path, "w") as f:
        f.write(generated)
    clang_format(file_path)


def write_test_cpp(cpp_ops: Sequence[str], file_path: str) -> None:
    code = "\n".join(cpp_ops)
    generated = f"""// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/torch.h>

#include "test_utils.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using namespace torch::jit::test;
using c10::IValue;

{code}

"""
    with open(file_path, "w") as f:
        f.write(generated)
    clang_format(file_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ATen source files")
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="caffe2/aten/src/ATen",
    )
    parser.add_argument(
        "-p",
        "--generated-ops-cpp-path",
        help="path to directory to generate op dispatcher .cpp file",
        default="caffe2/torch/csrc/jit/runtime/static/generated_ops.cpp",
    )
    parser.add_argument(
        "-t",
        "--generated-ops-test-cpp-path",
        help="path to directory to generate op dispatcher .cpp file",
        default="caffe2/benchmarks/static_runtime/test_generated_ops.cc",
    )
    options = parser.parse_args()
    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(options.source_path, "native/tags.yaml")
    parsed_yaml = gen.parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )

    op_generator = generator.GenOpDispatcher()
    test_case_generator = generator.GenOpTestCase()

    native_functions_groups = [
        g
        for g in gen.get_grouped_native_functions(native_functions)
        if isinstance(g, NativeFunctionsGroup)
    ]

    supported_functions_groups = group_functions_by_op_name(native_functions_groups)

    out_variant_op_result = [
        op_generator.out_variant(groups, backend_indices[DispatchKey.CPU])
        for groups in supported_functions_groups
    ]
    out_variant_test_result = [
        test_case_generator.out_variant(groups) for groups in supported_functions_groups
    ]

    native_functions_view_groups = [
        g
        for g in gen.get_grouped_by_view_native_functions(native_functions)
        if isinstance(g, NativeFunctionsViewGroup)
    ]

    supported_functions_view_groups = group_functions_by_op_name(
        native_functions_view_groups
    )

    view_op_result = [
        op_generator.view(groups, backend_indices[DispatchKey.CPU])
        for groups in supported_functions_view_groups
    ]
    view_test_result = [
        test_case_generator.view(groups) for groups in supported_functions_view_groups
    ]

    op_result = out_variant_op_result + ["\n\n"] + view_op_result
    test_result = out_variant_test_result + ["\n\n"] + view_test_result

    write_cpp(op_result, options.generated_ops_cpp_path)
    write_test_cpp(test_result, options.generated_ops_test_cpp_path)

    print(
        "\ntotal grouped native ops: %d"
        % len(gen.get_grouped_native_functions(native_functions))
    )

    print("grouped native ops with out variant: %d" % len(native_functions_groups))
    supported_functions_num = sum(
        [len(groups) for groups in supported_functions_groups]
    )
    print("generated functions groups with out variant: %d" % supported_functions_num)

    print("\nview grouped native ops: %d" % len(native_functions_view_groups))
    supported_view_functions_num = sum(
        [len(groups) for groups in supported_functions_view_groups]
    )
    print("generated functions view groups: %d" % supported_view_functions_num)

    print(
        "\noverall generated : %d"
        % (supported_functions_num + supported_view_functions_num)
    )


if __name__ == "__main__":
    set_simple_logging(escape_newlines=False)
    main()
