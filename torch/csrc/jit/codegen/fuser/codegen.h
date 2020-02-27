
copy: fbcode/caffe2/torch/csrc/jit/codegen/fuser/codegen.h
copyrev: cd53c1ad7904e4d23c10a2d538de4ca27a069e44

#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/fuser/arg_spec.h>
#include <torch/csrc/jit/codegen/fuser/partition_desc.h>
#include <torch/csrc/jit/codegen/fuser/tensor_desc.h>
#include <torch/csrc/jit/ir/ir.h>

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// Creates a CPU or CUDA kernel for the given graph.
// Returns the C++ or CUDA string implementing the kernel.
TORCH_API std::string generateKernel(
    const std::string& name,
    const Graph& graph,
    const std::vector<std::pair<const Value*, const c10::optional<TensorDesc>>>& inputs,
    const std::vector<std::pair<const Value*, const TensorDesc>>& outputs,
    const bool use_cuda);

} // namespace fuser
} // namespace jit
} // namespace torch
