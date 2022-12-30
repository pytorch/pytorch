#include <ir_builder.h>
#include <kernel_ir_dispatch.h>
#include <lower2device.h>
#include <lower_utils.h>

#include <lower_fusion_simplifier.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Replaces Transpose, Shift, Gather, and View Ops with Unary Ops.
class UnaryOpInserter : private kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    UnaryOpInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  using kir::ExprMutator::handle;

  UnaryOpInserter(const std::vector<Expr*>& exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void registerReplaceAndPropagate(Expr* old_expr, Expr* new_expr) {
    registerReplace(old_expr, new_expr);
    GpuLower::current()->propagateExprInfo(old_expr, new_expr);
  }

  void handle(TransposeOp* top) final {
    auto out = top->out();
    auto in = top->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        top, IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in));
  }

  void handle(SqueezeOp* sop) final {
    auto out = sop->out();
    auto in = sop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        sop, IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in));
  }

  void handle(ExpandOp* eop) final {
    auto out = eop->out();
    auto in = eop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        eop, IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in));
  }

  void handle(ShiftOp* sop) final {
    auto out = sop->out();
    auto in = sop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        sop, IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in));
  }

  void handle(GatherOp* gop) final {
    auto out = gop->out();
    auto in = gop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        gop, IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in));
  }

  void handle(ViewOp* vop) final {
    auto out = vop->out();
    auto in = vop->in();
    auto container = out->container();
    registerReplaceAndPropagate(
        vop, IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in));
  }
};

} // namespace

// Transpose, Shift, Gather, and View Ops with Unary Set Ops
std::vector<Expr*> unarySetOpInserter(const std::vector<Expr*>& exprs) {
  return UnaryOpInserter::insert(exprs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
