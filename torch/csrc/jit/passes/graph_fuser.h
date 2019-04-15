#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

// nvrtc has a limit on the number of arguments allowed in a CUDA kernel.
// The specific limit is a function of constant memory size, amount available
// to pass arguments, and some implementation dependence. Select a safe
// limit here.
//   This limit is also applied to other devices in the fuser, because we
// don't consider a kernel with such a large number of arguments would be
// profitable.
constexpr size_t fusion_kernel_args_limit = 128;

// NB: Be sure to run DCE before fusion, because dead instructions
// can prevent fusion opportunities from being exploited.
// On Windows will noop, NYI
TORCH_API void FuseGraph(std::shared_ptr<Graph>& graph);

TORCH_API bool trackSingleGradSumToSizeToOutputs(
    Value* gradSumToSizeOutput,
    std::vector<int64_t>* outputGradSumToSizes);

} // namespace jit
} // namespace torch
