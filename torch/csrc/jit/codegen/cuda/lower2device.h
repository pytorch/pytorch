#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_allocation.h>
#include <torch/csrc/jit/codegen/cuda/lower_double_buffer.h>
#include <torch/csrc/jit/codegen/cuda/lower_fused_reduction.h>
#include <torch/csrc/jit/codegen/cuda/lower_index_hoist.h>
#include <torch/csrc/jit/codegen/cuda/lower_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_predicate_elimination.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_sync_information.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_broadcast.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_reductions.h>
#include <torch/csrc/jit/codegen/cuda/lower_warp_reduce.h>
#include <torch/csrc/jit/codegen/cuda/non_divisible_split.h>
#include <torch/csrc/jit/codegen/cuda/parallel_dimension_map.h>
#include <torch/csrc/jit/codegen/cuda/partial_split_map.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/vectorization_info.h>

#include <memory>
#include <ostream>
#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO: we frequently use pairwise root mapping from consumers to producers.
// This information is implicitly in the computeAtMaps, but there's no isolated
// container for this information that we can reuse. Would be nice to generate
// such a structure and propagate it through lowering.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API GpuLower : public NonCopyable {
  class KernelIrMapper;

 public:
  GpuLower() = delete;

  // GpuLower lowers the provided fusion into a kernel which can be translated
  // into cuda code. index_type allows to compile the kernel based on int32
  // indexing instead of int64 for additional performance.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit GpuLower(Fusion* fusion, DataType index_type = DataType::Int) {
    lower(fusion, index_type);
  }

  kir::Kernel* kernel() const;

  //! Returns the currently active lowering object.
  //! It's an error if no lowering is in progress.
  static GpuLower* current();

  //! Query if lowering is in progress
  static bool hasCurrent();

  std::shared_ptr<const ConcretizedBroadcastDomains>
  concretizedBroadcastDomains() {
    return concretized_broadcast_domains_;
  }

  const ThreadPredicateMap& threadPredMap() const {
    return thread_pred_map_;
  }

  // Returns non-const reference. Necessary to reset a predicate flag
  // when a broadcast expression is fused into a reduction.
  ThreadPredicateMap& threadPredMap() {
    return thread_pred_map_;
  }

  std::shared_ptr<const ComputeAtMap> caMap() const {
    return std::const_pointer_cast<const ComputeAtMap>(compute_at_map_);
  }

  const TrivialReductionInfo& trivialReductionInfo() const {
    return trivial_reduction_info_;
  }

  std::shared_ptr<const HaloInfo> haloInfo() const {
    return std::const_pointer_cast<const HaloInfo>(halo_info_);
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

  LocalAllocationInfoMap& localAllocationInfoMap() {
    return local_allocation_info_map_;
  }

  const WarpPaddedParallelInfo& getWarpPaddedParallelInfo() const {
    return warp_pad_info_;
  }

  PartialSplitMap& partialSplitMap() {
    return partial_split_map_;
  }

  const PartialSplitMap& partialSplitMap() const {
    return partial_split_map_;
  }

  auto& nonDivisibleSplitInfo() {
    return non_divisible_split_info_;
  }

  const auto& nonDivisibleSplitInfo() const {
    return non_divisible_split_info_;
  }

  const auto& divisbleSplitSet() const {
    return divisible_splits_;
  }

  DoubleBufferInfo& doubleBufferInfo() {
    return double_buffer_info_;
  }

  CommonIndexMap& commonIndexMap() {
    return common_index_map_;
  }

  const auto& vectorizedAccesses() const {
    return vectorized_accesses_;
  }

  auto& vectorizedAccesses() {
    return vectorized_accesses_;
  }

  const auto& vectorizedSetInfo() const {
    return vectorized_set_info_;
  }

  auto& vectorizedSetInfo() {
    return vectorized_set_info_;
  }

  FusedReductionInfo& fusedReductionInfo() {
    return fused_reduction_info_;
  }

  const SyncMap& syncMap() const {
    return sync_map_;
  }

  kir::KernelPerformanceProfile& profile() {
    return profile_;
  }

  // This is an interface to propagate information after expression
  //  replacement on the kernel IR. E.g.:
  //    for ...
  //       c = a + b   (expr 0)
  //  after any pass that does replacement:
  //    for ...
  //       c1 = a1 + b1 (expr1)
  //  The previous analysis that was performed on expr0 might still
  //    be valid on expr1 but that info would be lost after replacement.
  //  This function provides an interface to manually update the info
  //    in any pass that performs replacement.
  void propagateExprInfo(const Expr* old_expr, const Expr* new_expr);

 private:
  void lower(Fusion* fusion, DataType index_type);

  // Goes through the parallelized iterdomains of the used TVs and find
  //  the parallel dimensions that need to be padded to a multiples of
  //  warp size.
  void collectPaddedParallelDims();

 private:
  // Lowered Kernel IR
  std::unique_ptr<kir::Kernel> kernel_;

  // Some stateful information during lowering
  // TODO: A lot of this information uses a define class then call build. It
  // would be safer to wrap all of these in unique pointers and remove the build
  // interface and default constructor. That way they couldn't be accessed
  // without being initialized.
  std::shared_ptr<const ConcretizedBroadcastDomains>
      concretized_broadcast_domains_;
  ThreadPredicateMap thread_pred_map_;
  PredicateElimination pred_elimination_;
  std::shared_ptr<ComputeAtMap> compute_at_map_;
  TrivialReductionInfo trivial_reduction_info_;
  std::shared_ptr<HaloInfo> halo_info_;
  LocalAllocationInfoMap local_allocation_info_map_;
  WarpPaddedParallelInfo warp_pad_info_;
  ParallelDimensionMap parallel_dimension_map_;
  PartialSplitMap partial_split_map_;
  NonDivisibleSplitInfo non_divisible_split_info_;
  DoubleBufferInfo double_buffer_info_;
  CommonIndexMap common_index_map_;
  FusedReductionInfo fused_reduction_info_;
  SyncMap sync_map_;
  kir::KernelPerformanceProfile profile_;
  std::unordered_set<Split*> divisible_splits_;

  // Track which tensor views are inputs or outputs of a vectorized operation
  // and their maximum vectorized access size
  // std::unordered_map<TensorView*, VectorizationInfo> vectorized_accesses_;
  std::unordered_map<TensorView*, int> vectorized_accesses_;
  // Info on each vectorized set op
  std::vector<VectorizedSetInfo> vectorized_set_info_;

  Fusion* fusion_ = nullptr;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
