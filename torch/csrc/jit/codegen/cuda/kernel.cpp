
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/dispatch.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {

class BuffersExtractor final : OptOutDispatch {
 public:
  explicit BuffersExtractor(const std::vector<Expr*>& exprs) {
    for (auto expr : exprs) {
      handle(expr);
    }
  }

  const auto& globalAllocs() const {
    return global_allocations_;
  }

  const auto& dynamicAllocs() const {
    return dynamic_allocations_;
  }

  const auto& staticAllocs() const {
    return static_allocations_;
  }

 private:
  void handle(Expr* expr) final {
    OptOutDispatch::handle(expr);
  }

  void handle(kir::ForLoop* fl) final {
    for (auto expr : fl->body().exprs()) {
      OptOutDispatch::handle(expr);
    }
  }

  void handle(kir::IfThenElse* ite) final {
    for (auto expr : ite->body().exprs()) {
      OptOutDispatch::handle(expr);
    }
    for (auto expr : ite->elseBody().exprs()) {
      OptOutDispatch::handle(expr);
    }
  }

  void handle(kir::Allocate* a) final {
    switch (a->getMemoryType()) {
      case MemoryType::Global:
        global_allocations_.push_back(a);
        break;
      case MemoryType::Shared:
        if (a->size()->isConstScalar()) {
          static_allocations_.push_back(a);
        } else {
          dynamic_allocations_.push_back(a);
        }
        break;
      case MemoryType::Local:
        break;
    }
  }

 private:
  std::vector<kir::Allocate*> global_allocations_;
  std::vector<kir::Allocate*> dynamic_allocations_;
  std::vector<kir::Allocate*> static_allocations_;
};

} // namespace

// TODO(kir): Kernel IR validation
Kernel::Kernel(
    const std::vector<Expr*>& exprs,
    const ThreadPredicateMap& predicate_map)
    : exprs_(exprs), predicate_map_(predicate_map) {
  analyze();
}

void Kernel::analyze() {
  // Cache the list of buffers used within the kernel
  BuffersExtractor buffers_extractor(exprs_);
  summary_.global_allocations = buffers_extractor.globalAllocs();
  summary_.dynamic_smem_allocations = buffers_extractor.dynamicAllocs();
  summary_.static_smem_allocations = buffers_extractor.staticAllocs();

  // Figure out if the kernel uses random numbers
  for (auto expr : exprs_) {
    if (expr->getExprType() == ExprType::UnaryOp) {
      if (expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::RandLike) {
        summary_.is_stochastic = true;
        break;
      }
    }
  }

  // Look for reductions and shared memory buffers
  size_t max_smem_type_size = 0;
  for (auto expr : exprs_) {
    for (auto out : expr->outputs()) {
      if (out->getValType() == ValType::KirTensorView) {
        const auto tv = out->as<kir::TensorView>();
        const auto domain = tv->domain();

        // Do we have any reductions?
        summary_.has_block_reductions |= domain->hasBlockReduction();
        summary_.has_grid_reductions |= domain->hasGridReduction();

        // Do we have block broadcasts?
        summary_.has_block_broadcasts |= domain->hasBlockBroadcast();

        // Update the largest smem data type
        if (domain->hasBlockReduction() || domain->hasGridReduction() ||
            tv->memoryType() == MemoryType::Shared) {
          const auto data_type = tv->getDataType().value();
          const size_t type_size = dataTypeSize(data_type);
          if (type_size > max_smem_type_size) {
            max_smem_type_size = type_size;
            summary_.largest_smem_data_type = data_type;
          }
        }
      }
    }
  }
}

} // namespace fuser
} // namespace jit
} // namespace torch
