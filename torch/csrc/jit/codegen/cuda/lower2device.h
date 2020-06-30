#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <ostream>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_CUDA_API GPULower {
 public:
  // Init printer on ostream
  GPULower(Fusion* _fusion) : fusion_(_fusion) {}

  // print generated code to ostream
  std::vector<Expr*> getLoweredExprs();
  std::ostream& printKernel(
      std::ostream& _os,
      const std::string& kernel_name = "CUDAGeneratedKernel");

 private:
  Fusion* const fusion_ = nullptr;
};

} // namespace fuser
} // namespace jit
} // namespace torch
