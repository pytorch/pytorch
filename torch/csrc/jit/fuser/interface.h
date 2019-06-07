#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>
#include <ATen/core/stack.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace torch {
namespace jit {

constexpr int kCPUDevice = -1;

// Assigns a "key" to the given fusion_group that it can use to run its
// fusion later (via runFusion() below).
TORCH_API int64_t registerFusion(const Node* fusion_group);

// Runs the fusion corresponding to the given key on the inputs
// found on the stack. Outputs are placed on the same stack.
// In some cases a fusion cannot be run and a fallback path where
// PyTorch's interpreter runs the graph instead is attempted.
TORCH_API void runFusion(const int64_t key, Stack& stack);

// True if the respective devices can fuse, false otherwise
TORCH_API bool canFuseOnCPU();
TORCH_API bool canFuseOnGPU();

// Sets whether fusion on the CPU is allowed (disabled by default due to
// flakiness)
TORCH_API void overrideCanFuseOnCPU(bool value);

// Treats the given graph as a fusion group and launches it on the
// specified device with the given inputs.
// Returns the outputs.
TORCH_API std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs);

// Treats the given graph as a fusion group and returns the generated code.
TORCH_API std::string debugGetFusedKernelCode(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs);

TORCH_API size_t nCompiledKernels();

} // namespace jit
} // namespace torch
