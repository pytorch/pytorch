#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>

#include <torch/csrc/jit/codegen/cuda/lower_fusion_simplifier.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Replace trivial reductions with unary ops.
class TrivialReductionReplacement : private OptOutMutator {
 public:
  TrivialReductionReplacement(
      Fusion* fusion,
      const TrivialReductionInfo& trivial_reduction_info)
      : trivial_reduction_info_(trivial_reduction_info) {
    FusionGuard fg(fusion);
    auto exprs = StmtSort::getExprs(fusion);
    for (auto expr : exprs) {
      mutate(expr);
    }
  }

 private:
  using OptOutMutator::mutate;
  void mutate(ReductionOp* rop) final {
    if (rop->out()->isA<TensorView>()) {
      auto out_tv = rop->out()->as<TensorView>();
      if (std::all_of(
              out_tv->domain()->domain().begin(),
              out_tv->domain()->domain().end(),
              [&](IterDomain* id) {
                // If id is a reduction axis, is it a trivial reduction?
                if (id->isReduction()) {
                  return trivial_reduction_info_.isDerived(id);
                } else {
                  return true;
                }
              })) {
        auto out = rop->out();
        auto in = rop->in();
        auto container = out->container();
        removeExpr(container, rop);
        IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in);
      }
    }
  }

  const TrivialReductionInfo& trivial_reduction_info_;
};

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

  void handle(TransposeOp* top) final {
    auto out = top->out();
    auto in = top->in();
    auto container = out->container();
    registerReplace(
        top, IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in));
  }

  void handle(ShiftOp* sop) final {
    auto out = sop->out();
    auto in = sop->in();
    auto container = out->container();
    registerReplace(
        sop, IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in));
  }

  void handle(GatherOp* gop) final {
    auto out = gop->out();
    auto in = gop->in();
    auto container = out->container();
    registerReplace(
        gop, IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in));
  }

  void handle(ViewDtypeOp* vop) final {
    auto out = vop->out();
    auto in = vop->in();
    auto container = out->container();
    registerReplace(
        vop,
        IrBuilder::create<UnaryOp>(container, UnaryOpType::EraseType, out, in));
  }

  void handle(ViewOp* vop) final {
    auto out = vop->out();
    auto in = vop->in();
    auto container = out->container();
    registerReplace(
        vop, IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in));
  }
};

} // namespace

void trivialReductionReplacement(
    Fusion* fusion,
    const TrivialReductionInfo& trivial_reduction_info) {
  TrivialReductionReplacement replacement(fusion, trivial_reduction_info);
}

// Transpose, Shift, Gather, and View Ops with Unary Set Ops
std::vector<Expr*> unarySetOpInserter(const std::vector<Expr*>& exprs) {
  return UnaryOpInserter::insert(exprs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
