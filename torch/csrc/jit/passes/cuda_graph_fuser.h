#pragma once

#include <aten/src/ATen/Context.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch {
namespace jit {

// Register CudaFuseGraph in custom passes
struct C10_EXPORT RegisterCudaFuseGraph
    : public PassManager<RegisterCudaFuseGraph> {
  static void registerPass() {
    TORCH_CHECK(
        at::globalContext().hasCUDA() && !at::globalContext().hasHIP(),
        "Running CUDA fuser is only supported on CUDA builds.");
    PassManager::registerPass(fuser::cuda::fuseGraph);
  }
};

} // namespace jit
} // namespace torch
