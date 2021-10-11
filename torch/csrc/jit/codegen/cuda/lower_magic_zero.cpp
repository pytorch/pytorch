#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class MagicZeroInserter : public kir::MutableIrVisitor {
 public:
  static std::vector<kir::Expr*> insert(const std::vector<kir::Expr*>& exprs) {
    MagicZeroInserter inserter(exprs);
    return inserter.loop_nests_;
  }

 private:
  struct InsertionInfo {
    kir::Scope* scope = nullptr;
    kir::ForLoop* fl = nullptr;
  };

  MagicZeroInserter(const std::vector<kir::Expr*>& exprs)
      : loop_nests_(exprs), ir_builder(GpuLower::current()->kernel()) {
    loop_nests_.insert(
        loop_nests_.begin(), ir_builder.create<kir::InitMagicZero>());
    for (auto expr : exprs) {
      handle(expr);
    }
    insertAll();
  }

  void handle(kir::Expr* expr) {
    if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle(ite);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle(for_loop);
    }
  }

  void handle(kir::IfThenElse* ite) {
    scope_nest_.push_back(&ite->thenBody());
    for (auto expr : ite->thenBody().exprs()) {
      handle(expr);
    }
    scope_nest_.pop_back();
    scope_nest_.push_back(&ite->elseBody());
    for (auto expr : ite->elseBody().exprs()) {
      handle(expr);
    }
    scope_nest_.pop_back();
  }

  void handle(kir::ForLoop* fl) {
    if (fl->isUnrollable()) {
      kir::Scope* scope = nullptr;
      if (!scope_nest_.empty()) {
        scope = scope_nest_.back();
      }
      insertion_list_.push_back({scope, fl});
    } else {
      scope_nest_.push_back(&fl->body());
      for (auto expr : fl->body().exprs()) {
        handle(expr);
      }
      scope_nest_.pop_back();
    }
  }

  void insertAll() {
    for (const auto& info : insertion_list_) {
      auto fl = info.fl;
      auto scope = info.scope;
      if (scope == nullptr) {
        // place in global scope
        auto loop_it = std::find(loop_nests_.begin(), loop_nests_.end(), fl);
        TORCH_INTERNAL_ASSERT(loop_it != loop_nests_.end());
        // Place after the loop
        loop_it++;
        loop_nests_.insert(loop_it, ir_builder.create<kir::UpdateMagicZero>());
      } else {
        scope->insert_after(fl, ir_builder.create<kir::UpdateMagicZero>());
      }
    }
  }

  //! Keep track for loop structure
  std::vector<kir::Scope*> scope_nest_;

  // Keep a copy of the expressions provided
  std::vector<kir::Expr*> loop_nests_;

  kir::IrBuilder ir_builder;

  std::vector<InsertionInfo> insertion_list_;
};

} // namespace

std::vector<kir::Expr*> insertMagicZero(const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertMagicZero");
  // Check if magic zero was even used, if not we don't have to define it or
  // update it.
  bool has_magic_zero = false;
  const auto gpu_lower = GpuLower::current();
  auto kernel = gpu_lower->kernel();
  for (auto& val : kernel->irNodes()) {
    if (val->isA<kir::NamedScalar>()) {
      auto named_scalar = val->as<kir::NamedScalar>();
      if (named_scalar->dtype() == DataType::Int &&
          named_scalar->name() == "nvfuser_zero") {
        has_magic_zero = true;
        break;
      }
    }
  }

  if (!has_magic_zero) {
    return exprs;
  }

  return MagicZeroInserter::insert(exprs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
