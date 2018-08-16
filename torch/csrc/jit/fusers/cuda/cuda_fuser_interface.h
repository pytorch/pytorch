#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fusers/fuser_interface.h"

#include "torch/csrc/WindowsTorchApiMacro.h"

#include "ATen/ATen.h"

#include <memory>

namespace torch { namespace jit { 

TORCH_API void FuseCUDAGraphInternal(std::shared_ptr<Graph>& graph);

TORCH_API std::shared_ptr<FusionFunction> getCUDAFusionFunction(Node* fusion_group);

TORCH_API void debugCUDALaunchGraph(
    Graph& graph
  , int device
  , at::ArrayRef<at::Tensor> inputs
  , at::ArrayRef<at::Tensor> outputs);

} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
