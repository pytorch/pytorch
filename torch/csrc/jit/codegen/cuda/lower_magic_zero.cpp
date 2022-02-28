#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class MagicZeroInserter : public kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    MagicZeroInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  struct InsertionInfo {
    kir::Scope* scope = nullptr;
    kir::ForLoop* fl = nullptr;
  };

  MagicZeroInserter(const std::vector<Expr*>& exprs) {
    TORCH_INTERNAL_ASSERT(exprs.size());
    kir::ExprMutator::registerInsertBefore(
        exprs.front(), IrBuilder::create<kir::InitMagicZero>(), nullptr);
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void handle(kir::ForLoop* fl) final {
    if (fl->isUnrolled()) {
      if (scope_.empty()) {
        kir::ExprMutator::registerInsertAfter(
            fl, IrBuilder::create<kir::UpdateMagicZero>());
      } else {
        TORCH_INTERNAL_ASSERT(
            scope_.back()->exprs().size(), "Not expecting an empty loop.");
        kir::ExprMutator::registerInsertAfter(
            fl, IrBuilder::create<kir::UpdateMagicZero>(), scope_.back());
      }
    } else {
      kir::ExprMutator::handle(fl);
    }
  }

  std::vector<InsertionInfo> insertion_list_;
};

} // namespace

std::vector<Expr*> insertMagicZero(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertMagicZero");
  // Check if magic zero was even used, if not we don't have to define it or
  // update it.
  const auto gpu_lower = GpuLower::current();
  auto kernel = gpu_lower->kernel();
  const bool has_magic_zero =
      std::any_of(kernel->vals().begin(), kernel->vals().end(), [](Val* val) {
        return isMagicZero(val);
      });

  if (!has_magic_zero) {
    return exprs;
  }

  return MagicZeroInserter::insert(exprs);
}

bool isMagicZero(const Val* val) {
  if (!val->isA<NamedScalar>()) {
    return false;
  }
  auto ns = val->as<NamedScalar>();
  return ns->dtype() == DataType::Int &&
      ns->name() == std::string(kMagicZeroName);
}

bool isProtectedWithMagicZero(const Val* val) {
  if (val->definition() == nullptr || !val->definition()->isA<BinaryOp>()) {
    return false;
  }
  auto bop = val->definition()->as<BinaryOp>();
  return bop->getBinaryOpType() == BinaryOpType::Add && isMagicZero(bop->rhs());
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
