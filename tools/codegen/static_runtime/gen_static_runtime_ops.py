from tools.codegen import gen
from tools.codegen.context import native_function_manager
from tools.codegen.model import NativeFunctionsGroup
from tools.codegen.static_runtime import gen_structured

import argparse
import itertools
import os
from typing import Sequence

# Given a list of `grouped_native_functions` sorted by their op names, return a list of
# lists each of which groups ops that share the base name. For example, `mean` and
# `mean.dim` are grouped together by this function.
def group_functions_by_op_name(grouped_native_functions:
                               Sequence[NativeFunctionsGroup]) -> Sequence[Sequence[NativeFunctionsGroup]]:
    if not grouped_native_functions:
        return []
    groups = []
    current_op_name = None
    current_group = None

    def is_supported(g: NativeFunctionsGroup) -> bool:
        with native_function_manager(g):
            return gen_structured.is_supported(g)

    eligible_ops = (g for g in grouped_native_functions if is_supported(g))
    groups = [list(group) for k, group in (itertools.groupby(eligible_ops, key=lambda g: g.functional.func.name.name.base))]
    return groups

def clang_format(cpp_file_path: str) -> None:
    import subprocess
    subprocess.run(["clang-format", "-i", cpp_file_path])

def write_cpp(cpp_ops: Sequence[str], file_path: str) -> None:
    code = "\n".join(cpp_ops)
    generated = f"""// @lint-ignore-every CLANGTIDY HOWTOEVEN
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
    parser = argparse.ArgumentParser(description='Generate ATen source files')
    parser.add_argument(
        '-s',
        '--source-path',
        help='path to source directory for ATen',
        default='aten/src/ATen')
    parser.add_argument(
        '-p',
        '--generated-ops-cpp-path',
        help='path to directory to generate op dispatcher .cpp file',
        default='torch/csrc/jit/runtime/static/generated_ops.cpp')
    parser.add_argument(
        '-t',
        '--generated-ops-test-cpp-path',
        help='path to directory to generate op dispatcher .cpp file',
        default='benchmarks/static_runtime/test_generated_ops.cc')
    options = parser.parse_args()
    native_yaml_path = os.path.join(options.source_path, 'native/native_functions.yaml')
    tags_yaml_path = os.path.join(options.source_path, 'native/tags.yaml')
    parsed_yaml = gen.parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = parsed_yaml.native_functions, parsed_yaml.backend_indices
    grouped_native_functions = gen.get_grouped_native_functions(native_functions)
    structured_native_functions = [g for g in grouped_native_functions
                                   if isinstance(g, NativeFunctionsGroup)]
    supported_function_groups = group_functions_by_op_name(structured_native_functions)

    gen_out_variant_dispatcher = gen_structured.GenOutVariantDispatcher()
    result = [gen_out_variant_dispatcher(groups) for groups in supported_function_groups]

    gen_out_variant_dispatcher_test_case = gen_structured.GenOutVariantDispatcherTestCase()
    test_result = [gen_out_variant_dispatcher_test_case(groups) for groups in supported_function_groups]

    write_cpp(result, options.generated_ops_cpp_path)
    write_test_cpp(test_result, options.generated_ops_test_cpp_path)

    print("total grouped native ops: %d" % len(grouped_native_functions))
    print("structured grouped native ops: %d" % len(structured_native_functions))
    supported_grouped_functions = sum([len(groups) for groups in supported_function_groups])
    print("generated grouped native ops: %d" % supported_grouped_functions)

if __name__ == '__main__':
    main()
