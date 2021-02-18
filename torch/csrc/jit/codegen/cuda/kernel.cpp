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
        summary_.has_dynamic_local_memory_allocations =
            summary_.has_dynamic_local_memory_allocations ||
            !ExpressionEvaluator::isConst(allocate->size());
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

 private:
  size_t max_smem_type_size_ = 0;
  KernelSummary summary_;

 private:
  void updateGridReductionInLoop(TensorDomain* dom) {
    ++summary_.number_of_grid_reductions;

    const auto gpu_lower = GpuLower::current();
    for (size_t i = 0; i < dom->nDims(); ++i) {
      const auto id =
          gpu_lower->caParallelMap().getConcreteMappedID(dom->domain()[i]);
      summary_.has_grid_reduction_in_loop =
          summary_.has_grid_reduction_in_loop || !id->isThread();
    }
  }
};

} // namespace

// TODO(kir): Kernel IR validation
void Kernel::finalize(
    std::vector<kir::Expr*> top_level_exprs,
    ThreadPredicateMap predicate_map) {
  TORCH_CHECK(top_level_exprs_.empty());
  TORCH_CHECK(!predicate_map_);
  top_level_exprs_ = std::move(top_level_exprs);
  predicate_map_ =
      std::make_unique<ThreadPredicateMap>(std::move(predicate_map));
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
