#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>

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
        summary_.has_dynamic_local_memory_allocations |=
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
    summary_.has_block_reductions |= domain->hasBlockReduction();
    summary_.has_grid_reductions |= domain->hasGridReduction();

    // Do we have block broadcasts?
    summary_.has_block_broadcasts |= domain->hasBlockBroadcast();

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
  }

 private:
  size_t max_smem_type_size_ = 0;
  KernelSummary summary_;
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
