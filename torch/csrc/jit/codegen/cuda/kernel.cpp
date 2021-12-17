#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <iostream>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

namespace {

//! Scan all primary expressions in the Kernel IR and build
//! lists of specialized nodes and other interesting information
class KernelIrScanner : private kir::IrVisitor {
 public:
  explicit KernelIrScanner(const Kernel* kernel) {
    for (const auto& ir_node : kernel->irNodes()) {
      ir_node->accept(this);
    }
    const auto gpu_lower = GpuLower::current();
    for (auto split : gpu_lower->nonDivisibleSplitInfo().splitsToValidate()) {
      auto extent = gpu_lower->lowerValue(split->in()->extent());
      auto factor = gpu_lower->lowerValue(split->factor());
      summary_.splits_to_validate.emplace_back(extent, factor);
    }
  }

  const auto& summary() const {
    return summary_;
  }

 private:
  void visit(const kir::Sync* sync) final {
    // TODO: Move to a dedicated validation pass
    // which is not on the common execution/compilation path
    if (sync->isWarHazardSync()) {
      ++summary_.war_hazard_syncs_count;
    }
  }

  void visit(const kir::Allocate* allocate) final {
    switch (allocate->memoryType()) {
      case MemoryType::Global:
        summary_.global_allocations.push_back(allocate);
        break;
      case MemoryType::Shared:
        if (ExpressionEvaluator::isConst(allocate->size())) {
          summary_.static_smem_allocations.push_back(allocate);
        } else {
          summary_.dynamic_smem_allocations.push_back(allocate);
        }
        break;
      case MemoryType::Local:
        if (!ExpressionEvaluator::isConst(allocate->size())) {
          summary_.has_dynamic_local_memory_allocations = true;
          summary_.dynamic_lmem_allocations.emplace_back(allocate);
        }
        break;
    }
  }

  void visit(const kir::UnaryOp* unary_op) final {
    if (unary_op->operation() == UnaryOpType::RandLike) {
      // This kernel is using random numbers
      summary_.is_stochastic = true;
    }
  }

  void visit(const kir::TensorIndex* tensor_index) final {
    const auto tv = tensor_index->view();
    const auto domain = tv->domain();

    // Do we have any reductions?
    summary_.has_block_reductions =
        summary_.has_block_reductions || domain->hasBlockReduction();

    // Do we have block broadcasts?
    summary_.has_block_broadcasts =
        summary_.has_block_broadcasts || domain->hasBlockBroadcast();

    // Update the largest smem data type
    if (domain->hasBlockReduction() || domain->hasGridReduction() ||
        tv->memoryType() == MemoryType::Shared) {
      const auto data_type = tv->dtype();
      const size_t type_size = dataTypeSize(data_type);
      if (type_size > max_smem_type_size_) {
        max_smem_type_size_ = type_size;
        summary_.largest_smem_data_type = data_type;
      }
    }

    // Update Welford
    if (tensor_index->definition() != nullptr &&
        tensor_index->definition()->isA<kir::WelfordOp>()) {
      summary_.has_welford = true;
      summary_.has_block_welford =
          summary_.has_block_welford || domain->hasBlockReduction();
      summary_.has_grid_welford =
          summary_.has_grid_welford || domain->hasGridReduction();
    }
  }

  void visit(const kir::GridWelford* grid_welford) final {
    const auto dom = grid_welford->welford_op()
                         ->out()
                         ->as<kir::TensorIndex>()
                         ->view()
                         ->domain();
    updateGridReductionInLoop(dom);
  }

  void visit(const kir::GridReduction* grid_reduction) final {
    const auto dom = grid_reduction->reduction_op()
                         ->out()
                         ->as<kir::TensorIndex>()
                         ->view()
                         ->domain();
    updateGridReductionInLoop(dom);
  }

  void visit(const kir::GridBroadcast*) final {
    summary_.has_cooperative_grid_reduction = true;
  }

 private:
  size_t max_smem_type_size_ = 0;
  KernelSummary summary_;

 private:
  void updateGridReductionInLoop(TensorDomain* dom) {
    summary_.has_grid_reductions = true;

    const auto gpu_lower = GpuLower::current();
    for (const auto i : c10::irange(dom->nDims())) {
      const auto id =
          gpu_lower->caParallelMap().getConcreteMappedID(dom->domain()[i]);

      summary_.has_cooperative_grid_reduction =
          summary_.has_cooperative_grid_reduction ||
          !(id->isThread() || id->extent()->isOneInt());
    }
  }
};

//! Make sure tensors have valid allocations even when parallelized
//! loops potentially have larger iteration counts than the number of
//! threads.
//!
//! When an IterDomain of a tensor is parallelized, the IterDomain
//! may not contribute to the allocation of the tensor. For example,
//! it is assumed that an allocation of a local-memory tensor does not
//! need to be accounted for an parallelied IterDomain. This is true
//! when it is guaranteed that each thread only needs to execute the
//! loop body once. However, if not, the allocation is invalid as it
//! only has a space for one value per thread.
//!
//! ValidateAllocation checks all tensor allocations and sees if any
//! tensor may have a parallelized loop whose iteration count may
//! be larger than the number of threads. If so, an error is thrown if
//! the tensor is not allocated on thread-shared memories. Note that
//! when allocated on a shared memory (i.e., MemoryType::Shared or
//! MemoryType::Global for tensors parallelized with threadIdx, or
//! MemoryType::Global for tensors parallelized with blockIdx), it is
//! assumed that allocation is properly extended for the iteration
//! count.
class ValidateAllocation : private kir::IrVisitor {
 public:
  static void validate(const Kernel* kernel) {
    ValidateAllocation validate_allocation(kernel);
  }

 private:
  explicit ValidateAllocation(const Kernel* kernel) {
    live_allocations_.emplace_back(std::vector<const Allocate*>());
    for (const auto& ir_node : kernel->topLevelExprs()) {
      ir_node->accept(this);
    }
    live_allocations_.pop_back();
    TORCH_INTERNAL_ASSERT(live_allocations_.empty());
  }

  void visit(const kir::Allocate* allocate) final {
    TORCH_INTERNAL_ASSERT(!live_allocations_.empty());
    live_allocations_.back().push_back(allocate);
  }

  // for_loop is parallelized and its stop value is not guaranteed to
  // be <= the number of threads, which breaks an assumption made
  // during in the allocation lowering if it's thread-parallel and not
  // allocated on shared or global memories, or if it's block-parallel
  // ando not allocated on global memory.
  void validate(const kir::ForLoop* for_loop) {
    const auto loop_id = for_loop->iter_domain();
    const auto gpu_lower = GpuLower::current();
    for (const auto& allocations : live_allocations_) {
      for (const auto& allocate : allocations) {
        const auto tv = dynamic_cast<kir::TensorView*>(allocate->buffer());
        if (tv == nullptr) {
          continue;
        }
        for (const auto& axis : tv->domain()->domain()) {
          if (!gpu_lower->caParallelMap().areMapped(loop_id, axis)) {
            continue;
          }
          if (isParallelTypeThreadDim(loop_id->parallelType())) {
            TORCH_INTERNAL_ASSERT(
                tv->memoryType() == MemoryType::Shared ||
                    tv->memoryType() == MemoryType::Global,
                "Tensor t",
                tv->name(),
                " must be allocated on SMEM or GMEM.");
          } else if (isParallelTypeBlockDim(loop_id->parallelType())) {
            TORCH_INTERNAL_ASSERT(tv->memoryType() == MemoryType::Global);
          }
        }
      }
    }
  }

  void visit(const kir::ForLoop* for_loop) final {
    if (for_loop->stop() != for_loop->iter_domain()->extent() &&
        isParallelTypeThread(for_loop->iter_domain()->parallelType())) {
      validate(for_loop);
    }

    live_allocations_.emplace_back(std::vector<const Allocate*>());
    for (const auto& expr : for_loop->body().exprs()) {
      expr->accept(this);
    }
    live_allocations_.pop_back();
  }

  void visit(const kir::IfThenElse* ite) final {
    for (const auto& expr : ite->thenBody().exprs()) {
      expr->accept(this);
    }
    for (const auto& expr : ite->elseBody().exprs()) {
      expr->accept(this);
    }
  }

 private:
  std::vector<std::vector<const Allocate*>> live_allocations_;
};

} // namespace

// TODO(kir): Kernel IR validation
void Kernel::finalize(std::vector<kir::Expr*> top_level_exprs) {
  TORCH_CHECK(top_level_exprs_.empty());
  top_level_exprs_ = std::move(top_level_exprs);
  predicate_map_ = std::make_unique<ThreadPredicateMap>(
      GpuLower::current()->threadPredMap());
  warp_padded_parallel_info_ = GpuLower::current()->getWarpPaddedParallelInfo();
  ValidateAllocation::validate(this);
  analyze();
}

void Kernel::analyze() {
  FUSER_PERF_SCOPE("Kernel::analyze");

  const KernelIrScanner ir_scanner(this);
  summary_ = ir_scanner.summary();
}

void Kernel::print() const {
  kir::IrPrinter ir_printer(std::cout);
  ir_printer.printKernel(this);
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
