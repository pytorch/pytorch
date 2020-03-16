#pragma once

#include <aten/src/ATen/Context.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch {
namespace jit {

// NB: Be sure to run DCE before fusion, because dead instructions
// can prevent fusion opportunities from being exploited.
// On Windows will noop, NYI
TORCH_API void CudaFuseGraph(std::shared_ptr<Graph>& graph);

// Register CudaFuseGraph in custom passes
struct TORCH_API RegisterCudaFuseGraph : public PassManager{
  static void registerPass(){
    TORCH_CHECK(
        at::globalContext().hasCUDA(),
        "CudaFuseGraph requires pytorch built with CUDA");
      TORCH_CHECK(
        !at::globalContext().hasHIP(),
        "CudaFuseGraph is not supported for HIP.");
    PassManager::registerPass(CudaFuseGraph);
  }
};

} // namespace jit
} // namespace torch
