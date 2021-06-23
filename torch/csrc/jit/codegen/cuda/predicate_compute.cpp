#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// find the first (and only) TensorView output
//
// TODO(kir): same question as ir_utils::getTvOutput():
//    why do we assume a single TV output?
//
kir::TensorView* firstTensorViewOutput(const kir::Expr* expr) {
  TORCH_INTERNAL_ASSERT(expr != nullptr);
  for (auto out : expr->outputs()) {
    if (out->isA<kir::TensorView>()) {
      return out->as<kir::TensorView>();
    } else if (out->isA<kir::TensorIndex>()) {
      return out->as<kir::TensorIndex>()->view();
    }
  }
  TORCH_INTERNAL_ASSERT(false, "Missing kir::TensorView output");
}

bool isTensorIndexOp(kir::Expr* expr) {
  const auto& outputs = expr->outputs();
  return outputs.size() >= 1 && outputs[0]->isA<kir::TensorIndex>();
}

} // namespace

namespace {

//! Analyze whether IterDomain can be statically determined to be safe
//! without bounds-checking predicates.
class IterationDomainAnalysis : private OptOutDispatch {
 public:
  //! Return true if the expression defining tv can be safely run
  //! without a predicate
  static bool canOmitPredicate(const TensorDomain* td) {
    const auto gpu_lower = GpuLower::current();
    for (size_t i = 0; i < td->nDims(); ++i) {
      IterDomain* id = gpu_lower->caLoopMap().getConcreteMappedID(td->axis(i));
      IterationDomainAnalysis id_analysis(id->fusion());
      auto extent = id->extent();
      id_analysis.handle(extent);
      if (!id_analysis.isExact(extent)) {
        return false;
      }
    }
    return true;
  }

 private:
  IterationDomainAnalysis(Fusion* fusion) : fusion_(fusion) {}

  using OptOutDispatch::handle;

  //! Check if val has nothing that prevents a loop using val as its
  //! extent to omit a bounds-checking predicate
  bool isExact(const Val* val) {
    return exact_vals_.find(val) != exact_vals_.end();
  }

  //! Record val does not need a predicate.
  void setExact(const Val* val) {
    exact_vals_.insert(val);
  }

  void handle(Val* val) override {
    if (val->definition() != nullptr) {
      handle(val->definition());
    } else {
      setExact(val);
    }
  }

  void handle(BinaryOp* bop) override {
    const auto lhs = bop->lhs();
    const auto rhs = bop->rhs();

    handle(lhs);
    handle(rhs);

    if (!(isExact(lhs) && isExact(rhs))) {
      return;
    }

    if (bop->getBinaryOpType() == BinaryOpType::CeilDiv) {
      // CeilDiv is the only expression that can make an extent val
      // larger than the actual. Need to know the exact values.
      ExpressionEvaluator ee(fusion_);
      const auto lhs_value = ee.evaluate(lhs);
      const auto rhs_value = ee.evaluate(rhs);
      if (lhs_value.has_value() && rhs_value.has_value() &&
          (lhs_value.value() % rhs_value.value()) == 0) {
        setExact(bop->out());
      }
    } else if (bop->getBinaryOpType() == BinaryOpType::Mul) {
      setExact(bop->out());
    } else {
      // Expr on extent should be either CeilDiv or Mul, which are
      // derived from split and merge, respectively.
      TORCH_INTERNAL_ASSERT("Unexpected BinaryOpType: ", bop);
    }
  }

 private:
  Fusion* fusion_ = nullptr;
  //! Vals that are known to need no predicate if used as IterDomain extent
  std::unordered_set<const Val*> exact_vals_;
};

} // namespace

kir::Bool* PredicateCompute::getInlinePredicate(
    const kir::Expr* expr,
    const std::vector<kir::ForLoop*>& loops,
    kir::Bool* thread_pred,
    PredicateType pred_type) {
  FUSER_PERF_SCOPE("getInlinePredicate");

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  if (loops.empty()) {
    TORCH_INTERNAL_ASSERT(thread_pred != nullptr);
    return thread_pred;
  }
  auto out_tv = firstTensorViewOutput(expr);
  // If local memory and initializing a reduction buffer, we don't need a
  // predicate
  if (out_tv->memoryType() == MemoryType::Local) {
    for (auto root_id : out_tv->fuserTv()->getMaybeRFactorDomain()) {
      if (!root_id->isReduction()) {
        continue;
      }
      auto kir_root_id = gpu_lower->lowerValue(root_id)->as<kir::IterDomain>();
      if (!std::any_of(loops.begin(), loops.end(), [&](kir::ForLoop* for_loop) {
            auto loop_id = for_loop->iter_domain();
            return gpu_lower->caLoopMap().areMapped(kir_root_id, loop_id);
          })) {
        return ir_builder.trueVal();
      }
    }
  }

  // Don't generate predicates unless needed. This is just for
  // potential performance benefit.
  if (IterationDomainAnalysis::canOmitPredicate(out_tv->fuserTv()->domain())) {
    TORCH_INTERNAL_ASSERT(thread_pred != nullptr);
    return thread_pred;
  }

  auto all_preds = Index::getReferenceRootPredicates(out_tv, loops).first;

  if (thread_pred != nullptr) {
    all_preds.push_back(thread_pred);
  }

  std::vector<kir::Bool*> preds;

  for (auto pred : all_preds) {
    if (!pred->isConst() || !(pred->isConst() && pred->value().value())) {
      preds.push_back(pred);
    }
  }

  const auto extent = (pred_type == PredicateType::Misaligned)
      ? preds.size() - 1
      : preds.size();
  if (preds.empty() || extent == 0) {
    return ir_builder.trueVal();
  }

  kir::Val* cond = preds[0];
  for (size_t i = 1; i < extent; i++) {
    cond = ir_builder.andExpr(cond, preds[i]);
  }

  return cond->as<kir::Bool>();
}

kir::Bool* UnswitchPredicate::get(
    const std::vector<kir::ForLoop*>& outer_loops,
    kir::ForLoop* unrolled_loop) {
  FUSER_PERF_SCOPE("UnswitchPredicate::get");

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  UnswitchPredicate up(outer_loops, unrolled_loop);

  kir::Val* unroll_pred = nullptr;
  for (auto pred : up.predicates_) {
    if (pred->isConst() && pred->value().value()) {
      continue;
    } else if (unroll_pred == nullptr) {
      unroll_pred = pred;
    } else {
      unroll_pred = ir_builder.andExpr(unroll_pred, pred);
    }
  }

  return unroll_pred == nullptr ? ir_builder.trueVal()
                                : unroll_pred->as<kir::Bool>();
}

void UnswitchPredicate::predicateOn(kir::Expr* tv_expr) {
  FUSER_PERF_SCOPE("UnswitchPredicate::predicateOn");

  if (for_loops_.empty()) {
    return;
  }

  const auto gpu_lower = GpuLower::current();

  auto out_tv = firstTensorViewOutput(tv_expr);

  auto pred_info = Index::getReferenceRootPredicates(out_tv, for_loops_, true);

  for (auto i : c10::irange(pred_info.first.size())) {
    auto pred = pred_info.first[i];
    const auto& root_ids = pred_info.second[i];

    bool add_pred = false;

    for (auto root_id : root_ids) {
      auto kir_root_id = gpu_lower->lowerValue(root_id)->as<kir::IterDomain>();

      if (std::find(
              predicated_iter_dom_.begin(),
              predicated_iter_dom_.end(),
              kir_root_id) == predicated_iter_dom_.end()) {
        add_pred = true;
        predicated_iter_dom_.push_back(kir_root_id);
      }
    }
    if (add_pred) {
      predicates_.push_back(pred);
    }
  }
}

void UnswitchPredicate::openLoop(kir::ForLoop* fl) {
  FUSER_PERF_SCOPE("UnswitchPredicate::openLoop");

  for_loops_.push_back(fl);

  for (auto expr : fl->body().exprs()) {
    if (ir_utils::isTVOp(expr) || isTensorIndexOp(expr)) {
      predicateOn(expr);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      openIte(ite);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      openLoop(for_loop);
    }
  }

  for_loops_.pop_back();
}

void UnswitchPredicate::openIte(kir::IfThenElse* ite) {
  FUSER_PERF_SCOPE("UnswitchPredicate::openIte");

  // only expand the ite thenBody
  for (auto expr : ite->thenBody().exprs()) {
    if (ir_utils::isTVOp(expr) || isTensorIndexOp(expr)) {
      predicateOn(expr);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      openIte(ite);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      openLoop(for_loop);
    }
  }
}

UnswitchPredicate::UnswitchPredicate(
    std::vector<kir::ForLoop*> outer_loops,
    kir::ForLoop* unrolled_loop)
    : for_loops_(std::move(outer_loops)) {
  openLoop(unrolled_loop);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
