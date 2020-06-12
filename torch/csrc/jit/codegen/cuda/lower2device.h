#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <ostream>

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_CUDA_API GPULower {
 private:
  Fusion* const fusion_;

 public:
  // Init printer on ostream
  GPULower(Fusion* _fusion) : fusion_(_fusion) {}

  // print generated code to ostream
  std::vector<Expr*> getLoweredExprs();
  std::ostream& printKernel(
      std::ostream& _os,
      const std::string& kernel_name = "CUDAGeneratedKernel");
};

} // namespace fuser
} // namespace jit
} // namespace torch
