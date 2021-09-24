#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_reductions.h>
#include <torch/csrc/jit/codegen/cuda/parallel_dimension_map.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <memory>
#include <ostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO: we frequently use pairwise root mapping from consumers to producers.
// This information is implicitly in the computeAtMaps, but there's no isolated
// container for this information that we can reuse. Would be nice to generate
// such a structure and propagate it through lowering.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API GpuLower {
  class KernelIrMapper;

 public:
  GpuLower() = default;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
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

  const ThreadPredicateMap& threadPredMap() const {
    return thread_pred_map_;
  }

  const ComputeAtMap& caLoopMap() const {
    return ca_loop_map_;
  }

  const ComputeAtMap& caIndexMap() const {
    return ca_index_map_;
  }

  const ComputeAtMap& caParallelMap() const {
    return ca_parallel_map_;
  }

  const auto& trivialReductionInfo() const {
    return trivial_reduction_info_;
  }

  const HaloInfo& haloInfo() const {
    return halo_info_;
  }

  HaloInfo& haloInfo() {
    return halo_info_;
  }

  const ParallelDimensionMap& parallelDimensionMap() const {
    return parallel_dimension_map_;
  }

  ParallelDimensionMap& parallelDimensionMap() {
    return parallel_dimension_map_;
  }

  PredicateElimination& predicateElimination() {
    return pred_elimination_;
  }

  const PredicateElimination& predicateElimination() const {
    return pred_elimination_;
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
  ThreadPredicateMap thread_pred_map_;
  PredicateElimination pred_elimination_;
  ComputeAtMap ca_loop_map_;
  ComputeAtMap ca_index_map_;
  ComputeAtMap ca_parallel_map_;
  TrivialReductionInfo trivial_reduction_info_;
  HaloInfo halo_info_;
  ParallelDimensionMap parallel_dimension_map_;

  Fusion* fusion_ = nullptr;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
