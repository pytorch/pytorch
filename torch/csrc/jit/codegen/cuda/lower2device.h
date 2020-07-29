#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <ostream>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_CUDA_API GPULower {
 public:
  explicit GPULower(Fusion* fusion) : fusion_(fusion) {
    lower();
  }

  GPULower() = default;
  GPULower(const GPULower& lower) = default;
  GPULower& operator=(const GPULower& other) = default;

  const std::vector<Expr*>& lowered_exprs() const {
    return lowered_exprs_;
  }

  // print generated code to ostream
  std::ostream& printKernel(
      std::ostream& _os,
      const std::string& kernel_name = "CUDAGeneratedKernel");

  std::string getKernel(const std::string& kernel_name = "CUDAGeneratedKernel");

  std::vector<kir::Allocate*> global_allocations() {
    return global_allocations_;
  }

  std::vector<kir::Allocate*> sync_allocations() {
    return sync_allocations_;
  }

 private:
  void lower();

 private:
  // List of global buffers (not including buffers for grid syncronization)
  std::vector<kir::Allocate*> global_allocations_;

  // List of syncronization buffers that must be initialized to 0 when running
  // the fusion
  std::vector<kir::Allocate*> sync_allocations_;

  // Lowered IR
  std::vector<Expr*> lowered_exprs_;

  Fusion* fusion_ = nullptr;
};

} // namespace fuser
} // namespace jit
} // namespace torch
