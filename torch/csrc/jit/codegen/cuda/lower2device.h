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

  //! Returns the currently active lowering object
  //! (or nullptr if no lowering is in progress)
  static GpuLower* current();

  ConcretizedBroadcastDomains& concretizedBroadcastDomains() {
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

  const ComputeAtMap& caLoopMap() const {
    return ca_loop_map_;
  }

  const ComputeAtMap& caIndexMap() const {
    return ca_index_map_;
  }

  const ComputeAtMap& caParallelMap() const {
    return ca_parallel_map_;
  }

  const TrivialReductionInfo& trivialReductionInfo() const {
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
  ConcretizedBroadcastDomains concretized_broadcast_domains_;
  ThreadPredicateMap thread_pred_map_;
  PredicateElimination pred_elimination_;
  ComputeAtMap ca_loop_map_;
  ComputeAtMap ca_index_map_;
  ComputeAtMap ca_parallel_map_;
  TrivialReductionInfo trivial_reduction_info_;
  HaloInfo halo_info_;
  LocalAllocationInfoMap local_allocation_info_map_;
  WarpPaddedParallelInfo warp_pad_info_;
  ParallelDimensionMap parallel_dimension_map_;
  PartialSplitMap partial_split_map_;
  NonDivisibleSplitInfo non_divisible_split_info_;
  DoubleBufferInfo double_buffer_info_;
  CommonIndexMap common_index_map_;
  FusedReductionInfo fused_reduction_info_;
  SyncMap sync_map_;

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
