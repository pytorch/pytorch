
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

Kernel::Kernel(const std::vector<Expr*>& exprs) : exprs_(exprs) {
  BuffersExtractor buffers_extractor(exprs);
  global_allocations_ = buffers_extractor.globalAllocs();
  dynamic_smem_allocations_ = buffers_extractor.dynamicAllocs();
  static_smem_allocations_ = buffers_extractor.staticAllocs();
}

} // namespace fuser
} // namespace jit
} // namespace torch
