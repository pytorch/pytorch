
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <memory>
#include <utility>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// Container for a lowered Kernel IR
//
// TODO(kir): currently, it is just pointing to nodes owned
//  by a Fusion object. The goal is to have the Kernel object
//  own the Kernel IR nodes
//
class TORCH_CUDA_API Kernel final : public NonCopyable {
 public:
  explicit Kernel(const std::vector<Expr*>& exprs);

  const auto& globalAllocations() const {
    return global_allocations_;
  }

  const auto& dynamicAllocations() const {
    return dynamic_smem_allocations_;
  }

  const auto& staticAllocations() const {
    return static_smem_allocations_;
  }

  const auto& exprs() const {
    return exprs_;
  }

 private:
  // List of global buffers
  std::vector<kir::Allocate*> global_allocations_;

  // List of dynamic shared memory buffers
  std::vector<kir::Allocate*> dynamic_smem_allocations_;

  // List of static shared memory buffers
  std::vector<kir::Allocate*> static_smem_allocations_;

  // Lowered expressions
  std::vector<Expr*> exprs_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
