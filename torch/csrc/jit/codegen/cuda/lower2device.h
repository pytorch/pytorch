#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <memory>
#include <ostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_CU_API GpuLower {
  class KernelIrMapper;

 public:
  GpuLower() = default;

  explicit GpuLower(Fusion* fusion) : fusion_(fusion) {
    lower();
  }

  kir::Kernel* kernel() const;

  //! Converts a Fusion IR value into the Kernel IR equivalent
  kir::Val* lowerValue(const Val* val);

  //! Converts a Fusion IR expression into the Kernel IR equivalent
  kir::Expr* lowerExpr(const Expr* expr);

  //! Returns the currently active lowering object
  //! (or nullptr if no lowering is in progress)
  static GpuLower* current();

  const ComputeAtMap& caLoopMap() const {
    return ca_loop_map_;
  }

  const ComputeAtMap& caIndexMap() const {
    return ca_index_map_;
  }

  const ComputeAtMap& caParallelMap() const {
    return ca_parallel_map_;
  }

  const auto& trivialReductions() const {
    return trivial_reductions_;
  }

  const auto& kirTrivialReductions() const {
    return kir_trivial_reductions_;
  }

  bool isDerivedFromTrivialReduction(IterDomain* id) const {
    return trivialReductions().find(id) != trivialReductions().end();
  }

  bool isDerivedFromTrivialReduction(kir::IterDomain* id) const {
    return kirTrivialReductions().find(id) != kirTrivialReductions().end();
  }

 private:
  void lower();

  // TensorViews are all based on symbolic sizes. When we first initialize them
  // we don't know if they're inputs or outputs which would mean that they have
  // runtime shapes. Intermediate tensors (those not going to global memory) do
  // not have this information. Since we need to have the correct information in
  // the kernel being fetched for shapes, we want to replace input and output
  // tensors to reference the runtime structure containing sizes.
  void replaceSymbolicSizes();

 private:
  // Lowered Kernel IR
  std::unique_ptr<kir::Kernel> kernel_;

  // Fusion IR node to Kernel IR node mapping
  std::unordered_map<const Val*, kir::Val*> kir_val_map_;
  std::unordered_map<const Expr*, kir::Expr*> kir_expr_map_;

  // Some stateful information during lowering
  ComputeAtMap ca_loop_map_;
  ComputeAtMap ca_index_map_;
  ComputeAtMap ca_parallel_map_;
  std::unordered_set<IterDomain*> trivial_reductions_;
  std::unordered_set<kir::IterDomain*> kir_trivial_reductions_;

  Fusion* fusion_ = nullptr;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
