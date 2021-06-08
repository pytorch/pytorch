#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <memory>
#include <utility>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Summary of interesting facts about the kernel
//!
//! TODO(kir): const node ptrs
//!
struct KernelSummary {
  //! List of Write-After-Read (WAR) synchronization barriers
  std::unordered_map<size_t, kir::Sync*> war_hazard_syncs;

  //! List of global buffers
  std::vector<kir::Allocate*> global_allocations;

  //! List of dynamic shared memory buffers
  std::vector<kir::Allocate*> dynamic_smem_allocations;

  //! List of static shared memory buffers
  std::vector<kir::Allocate*> static_smem_allocations;

  //! Indicate the need to generate random numbers
  bool is_stochastic = false;

  //! Do we have any block reductions?
  bool has_block_reductions = false;

  //! Do we have any grid reductions?
  bool has_grid_reductions = false;

  //! Do we have any block broadcasts?
  bool has_block_broadcasts = false;

  //! Largest shared memory buffer base type
  DataType largest_smem_data_type = DataType::Null;
};

//! Container for a lowered Kernel IR
//!
//! TODO(kir): currently, it is just pointing to nodes owned
//!  by a Fusion object. The goal is to have the Kernel object
//!  own the Kernel IR nodes
//!
class TORCH_CUDA_CU_API Kernel final : public NonCopyable {
 public:
  Kernel() = default;

  //! Finalize a kernel definition
  //!
  //! At this point we have a complete kernel definition and we can
  //! run analysis passes to build a KernelSummary
  //!
  void finalize(
      std::vector<Expr*> top_level_exprs,
      ThreadPredicateMap predicate_map);

  //! Register input as an input of the kernel
  void addInput(Val* input) {
    inputs_.push_back(input);
  }

  //! Register output as an output of the kernel
  void addOutput(Val* output) {
    outputs_.push_back(output);
  }

  const auto& inputs() const {
    return inputs_;
  }

  const auto& outputs() const {
    return outputs_;
  }

  const auto& topLevelExprs() const {
    return top_level_exprs_;
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
  void registerIrNode(std::unique_ptr<Statement> node) {
    ir_nodes_.push_back(std::move(node));
  }

  //! Debug dump of the Kernel IR
  void print() const;

 private:
  // Analyze the kernel IR and caches the summary of interesting data
  void analyze();

 private:
  // Kernel IR nodes
  std::vector<std::unique_ptr<Statement>> ir_nodes_;

  // Map from value to its definition expression
  std::unordered_map<const Val*, Expr*> definitions_;

  // Top level expressions
  std::vector<Expr*> top_level_exprs_;

  // Kernel inputs and outputs
  std::vector<Val*> inputs_;
  std::vector<Val*> outputs_;

  // Summary of interesting kernel data
  KernelSummary summary_;

  // Predicate map
  // TODO(kir): consider a simpler, kernel IR based version
  std::unique_ptr<ThreadPredicateMap> predicate_map_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
