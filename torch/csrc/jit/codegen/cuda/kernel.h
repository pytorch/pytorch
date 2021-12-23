#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_warp_reduce.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <memory>
#include <utility>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

//! Summary of interesting facts about the kernel
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct KernelSummary {
  //! Count of WAR (write-after-read) hazard barriers
  int war_hazard_syncs_count = 0;

  //! List of global buffers
  std::vector<const kir::Allocate*> global_allocations;

  //! List of dynamic shared memory buffers
  std::vector<const kir::Allocate*> dynamic_smem_allocations;

  //! List of static shared memory buffers
  std::vector<const kir::Allocate*> static_smem_allocations;

  //! Indicate the need to generate random numbers
  bool is_stochastic = false;

  //! Do we have any block reductions?
  bool has_block_reductions = false;

  //! Number of static grid reductions
  bool has_grid_reductions = false;

  //! Do we have any grid reduction in a loop, or grid reductions dependent on
  //! grid reductions
  bool has_cooperative_grid_reduction = false;

  //! Do we have any block broadcasts?
  bool has_block_broadcasts = false;

  //! Do we have any welford op?
  bool has_welford = false;

  //! Do we have any welford op?
  bool has_block_welford = false;

  //! Do we have any welford op?
  bool has_grid_welford = false;

  //! Largest shared memory buffer base type
  DataType largest_smem_data_type = DataType::Null;

  //! Do we have allocations of dynamic local memory?
  bool has_dynamic_local_memory_allocations = false;

  //! List of dynamic local memory buffers.
  //! Only used for debugging.
  std::vector<const kir::Allocate*> dynamic_lmem_allocations;

  //! ceilDiv extents that must be divisible
  std::vector<std::pair<const kir::Val*, const kir::Val*>> splits_to_validate;
};

//! Container for a lowered Kernel IR
//!
//! TODO(kir): currently, it is just pointing to nodes owned
//!  by a Fusion object. The goal is to have the Kernel object
//!  own the Kernel IR nodes
//!
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API Kernel final : public NonCopyable {
 public:
  Kernel() = default;

  //! Finalize a kernel definition
  //!
  //! At this point we have a complete kernel definition and we can
  //! run analysis passes to build a KernelSummary
  //!
  void finalize(std::vector<kir::Expr*> top_level_exprs);

  //! Register input as an input of the kernel
  void addInput(Val* input) {
    inputs_.push_back(input);
    input_set_.insert(input);
  }

  //! Register output as an output of the kernel
  void addOutput(Val* output) {
    outputs_.push_back(output);
    output_set_.insert(output);
  }

  const auto& inputs() const {
    return inputs_;
  }

  const auto& outputs() const {
    return outputs_;
  }

  bool isInput(Val* val) const {
    return input_set_.find(val) != input_set_.end();
  }

  bool isOutput(Val* val) const {
    return output_set_.find(val) != output_set_.end();
  }

  const auto& topLevelExprs() const {
    return top_level_exprs_;
  }

  const auto& irNodes() const {
    return ir_nodes_;
  }

  const KernelSummary& summary() const {
    return summary_;
  }

  const ThreadPredicateMap& predicateMap() const {
    return *predicate_map_;
  }

  //! Register a new Kernel IR node
  //!
  //! \note This is a specialized helper for kir::IrBuilder, not
  //!   intendted for general use
  //!
  void registerIrNode(kir::Passkey passkey, std::unique_ptr<kir::Node> node) {
    TORCH_CHECK(passkey.kernel == this);
    ir_nodes_.push_back(std::move(node));
  }

  //! Allocates a new value identifier
  kir::ValueId newValueId(kir::Passkey passkey) {
    TORCH_CHECK(passkey.kernel == this);
    return next_value_id_++;
  }

  //! Checks if parallel type is padded
  bool isParallelTypePadded(ParallelType ptype) const {
    return ptype == ParallelType::TIDx &&
        warp_padded_parallel_info_.is_tidx_padded;
  }

  const WarpPaddedParallelInfo& getWarpPaddedParallelInfo() const {
    return warp_padded_parallel_info_;
  }

  //! Debug dump of the Kernel IR
  void print() const;

 private:
  // Analyze the kernel IR and caches the summary of interesting data
  void analyze();

 private:
  // Kernel IR nodes
  std::vector<std::unique_ptr<kir::Node>> ir_nodes_;

  // Top level statements
  std::vector<kir::Expr*> top_level_exprs_;

  // Kernel inputs and outputs
  std::vector<Val*> inputs_;
  std::vector<Val*> outputs_;
  std::unordered_set<Val*> input_set_;
  std::unordered_set<Val*> output_set_;

  // Used to allocate unique value IDs
  kir::ValueId next_value_id_ = 1;

  // Summary of interesting kernel data
  KernelSummary summary_;

  // Predicate map
  // TODO(kir): consider a simpler, kernel IR based version
  std::unique_ptr<ThreadPredicateMap> predicate_map_;
  WarpPaddedParallelInfo warp_padded_parallel_info_;
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
