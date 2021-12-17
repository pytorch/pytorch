#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class MagicZeroInserter : public kir::KirVisitor {
 public:
  static std::vector<kir::Expr*> insert(const std::vector<kir::Expr*>& exprs) {
    MagicZeroInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  struct InsertionInfo {
    kir::Scope* scope = nullptr;
    kir::ForLoop* fl = nullptr;
  };

  MagicZeroInserter(const std::vector<kir::Expr*>& exprs)
      : ir_builder(GpuLower::current()->kernel()) {
    kir::KirVisitor::handle(exprs);
    // exprs_ isn't copied over until kir::KirVisitor::handle is called. This
    // will be easier once we have an insertion class as we can just mark insert
    // before the first expr
    exprs_.insert(exprs_.begin(), ir_builder.create<kir::InitMagicZero>());
    insertAll();
  }

  void handle(kir::ForLoop* fl) final {
    if (fl->isUnrolled()) {
      kir::Scope* scope = nullptr;
      if (!scope_.empty()) {
        scope = scope_.back();
      }
      insertion_list_.push_back({scope, fl});
    } else {
      kir::KirVisitor::handle(fl);
    }
  }

  void insertAll() {
    for (const auto& info : insertion_list_) {
      auto fl = info.fl;
      auto scope = info.scope;
      if (scope == nullptr) {
        // place in global scope
        auto loop_it = std::find(exprs_.begin(), exprs_.end(), fl);
        TORCH_INTERNAL_ASSERT(loop_it != exprs_.end());
        // Place after the loop
        loop_it++;
        exprs_.insert(loop_it, ir_builder.create<kir::UpdateMagicZero>());
      } else {
        scope->insert_after(fl, ir_builder.create<kir::UpdateMagicZero>());
      }
    }
  }

  kir::IrBuilder ir_builder;

  std::vector<InsertionInfo> insertion_list_;
};

} // namespace

std::vector<kir::Expr*> insertMagicZero(const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertMagicZero");
  // Check if magic zero was even used, if not we don't have to define it or
  // update it.
  const auto gpu_lower = GpuLower::current();
  auto kernel = gpu_lower->kernel();
  const bool has_magic_zero = std::any_of(
      kernel->irNodes().begin(),
      kernel->irNodes().end(),
      [](const std::unique_ptr<kir::Node>& ir_node) {
        return ir_node->isA<kir::Val>() && isMagicZero(ir_node->as<kir::Val>());
      });

  if (!has_magic_zero) {
    return exprs;
  }

  return MagicZeroInserter::insert(exprs);
}

bool isMagicZero(kir::Val* val) {
  auto ns = dynamic_cast<kir::NamedScalar*>(val);
  if (ns == nullptr) {
    return false;
  }
  return ns->dtype() == DataType::Int &&
      ns->name() == std::string(kMagicZeroName);
}

bool isProtectedWithMagicZero(kir::Val* val) {
  auto def = dynamic_cast<kir::BinaryOp*>(val->definition());
  return def && def->operation() == BinaryOpType::Add &&
      isMagicZero(def->rhs());
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
