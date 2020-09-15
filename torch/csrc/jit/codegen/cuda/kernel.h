
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_CUDA_API Kernel final {
 public:
  void print() const;

 private:
  // Lowered IR
  std::unordered_set<Val*> lowered_val_set_;
  std::unordered_set<Expr*> lowered_expr_set_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
