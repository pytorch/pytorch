#include <ir_builder.h>
#include <kernel_ir_dispatch.h>
#include <lower_utils.h>

#include <lower_fusion_simplifier.h>

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
    if (ir_utils::isTvOp(rop)) {
      auto out_tv = ir_utils::getTvOutput(rop);
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

  void mutate(GroupedReductionOp* grouped_rop) final {
    if (ir_utils::isTvOp(grouped_rop)) {
      // The inputs and outputs are all uniform in grouped reductions,
      // so just checking one of the input and output pair should be
      // sufficient.
      auto out_tv = ir_utils::getTvOutput(grouped_rop);
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
        auto outputs = grouped_rop->outputs();
        auto inputs = grouped_rop->inputs();
        auto container = out_tv->container();
        removeExpr(container, grouped_rop);
        for (const auto i : c10::irange(outputs.size())) {
          IrBuilder::create<UnaryOp>(
              container, UnaryOpType::Set, outputs.at(i), inputs.at(i));
        }
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

  void handle(ExpandOp* eop) final {
    auto out = eop->out();
    auto in = eop->in();
    auto container = out->container();
    registerReplace(
        eop, IrBuilder::create<UnaryOp>(container, UnaryOpType::Set, out, in));
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
