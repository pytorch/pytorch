#pragma once

#include <ATen/Context.h>
#include <torch/csrc/jit/ir/ir.h>

namespace nvfuser {

TORCH_CUDA_CU_API void TypePropagate(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace nvfuser
