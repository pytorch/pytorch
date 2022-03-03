#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_allocation.h>
#include <torch/csrc/jit/codegen/cuda/lower_double_buffer.h>
#include <torch/csrc/jit/codegen/cuda/lower_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_broadcast.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_reductions.h>
#include <torch/csrc/jit/codegen/cuda/lower_warp_reduce.h>
#include <torch/csrc/jit/codegen/cuda/non_divisible_split.h>
#include <torch/csrc/jit/codegen/cuda/parallel_dimension_map.h>
#include <torch/csrc/jit/codegen/cuda/partial_split_map.h>
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
class TORCH_CUDA_CU_API GpuLower : public NonCopyable {
  class KernelIrMapper;

 public:
  GpuLower() = delete;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit GpuLower(Fusion* fusion) {
    lower(fusion);
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

 private:
  void lower(Fusion* fusion);

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

  Fusion* fusion_ = nullptr;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
