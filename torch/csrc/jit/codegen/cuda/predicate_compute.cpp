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
kir::TensorView* firstTvOutput(const kir::Expr* expr) {
  TORCH_INTERNAL_ASSERT(expr != nullptr);
  for (auto out : expr->outputs()) {
    if (out->isA<kir::TensorView>()) {
      return out->as<kir::TensorView>();
    }
  }
  TORCH_INTERNAL_ASSERT(false, "Missing kir::TensorView output");
}

kir::IterDomain* getTermIterDomainInMap(
    kir::IterDomain* root_iter_domain,
    const IterDomainMap& p2c_root_map) {
  auto iter_domain = root_iter_domain;
  while (p2c_root_map.find(iter_domain) != p2c_root_map.end()) {
    iter_domain = p2c_root_map.at(iter_domain);
  }
  return iter_domain;
}

} // namespace

std::vector<kir::Bool*> PredicateCompute::computePredicates(
    const kir::TensorView* tv,
    const std::vector<kir::Val*>& indices,
    bool buffer_init) {
  FUSER_PERF_SCOPE("computePredicates");

  const auto domain = tv->domain();
  const auto& root = (buffer_init && domain->hasRFactor())
      ? domain->rfactorDomain()
      : domain->rootDomain();

  TORCH_INTERNAL_ASSERT(root.size() == indices.size());

  bool no_pred_needed = true;
  for (auto id : domain->domain()) {
    if (!id->isSimple()) {
      no_pred_needed = false;
      break;
    }
  }

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  auto true_bool = ir_builder.create<kir::Bool>(true);
  std::vector<kir::Bool*> preds(root.size(), true_bool);

  if (no_pred_needed) {
    return preds;
  }

  kir::Val* extent = nullptr;

  for (size_t i = 0; i < indices.size(); i++) {
    const bool zero_ind = indices[i]->isZeroInt();
    const bool simple_ind = indices[i]->definition() == nullptr;

    if (root[i]->isBroadcast() || (buffer_init && root[i]->isReduction()) ||
        gpu_lower->trivialReductionInfo().isDerived(root[i])) {
      continue;
    } else if (simple_ind && !zero_ind) {
      extent = nullptr;
      continue;
    } else if (zero_ind) {
      if (root[i]->extent()->isOneInt()) {
        continue;
      }
      if (extent == nullptr) {
        extent = root[i]->extent();
      } else {
        extent = ir_builder.mulExpr(extent, root[i]->extent());
      }
    } else {
      auto local_extent = root[i]->extent();
      if (extent != nullptr) {
        local_extent = ir_builder.mulExpr(extent, local_extent);
      }
      auto pred = ir_builder.ltExpr(indices[i], local_extent);
      extent = nullptr;
      preds[i] = pred->as<kir::Bool>();
    }
  }
  return preds;
}

namespace {

//! Analyze whether IterDomain can be statically determined to be safe
//! without bounds-checking predicates.
class IterationDomainAnalysis : private OptOutDispatch {
 public:
  //! Return true if the expression defining tv can be safely run
  //! without a predicate
  static bool canOmitPredicate(const kir::TensorView* tv) {
    const auto gpu_lower = GpuLower::current();
    auto fuser_tv = tv->fuserTv();
    for (size_t i = 0; i < fuser_tv->nDims(); ++i) {
      IterDomain* id =
          gpu_lower->caLoopMap().getConcreteMappedID(fuser_tv->axis(i));
      IterationDomainAnalysis id_analysis(id->fusion());
      auto extent = id->rawExtent();
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
    bool ignore_block_grid_external_ops) {
  FUSER_PERF_SCOPE("getInlinePredicate");
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  if (loops.empty()) {
    return thread_pred;
  }

  // Handle these elsewhere
  if (ignore_block_grid_external_ops) {
    if (expr->outputs().size() > 0 &&
        expr->outputs()[0]->isA<kir::TensorView>()) {
      const auto domain = expr->outputs()[0]->as<kir::TensorView>()->domain();
      if ((expr->isA<kir::ReductionOp>() &&
           (domain->hasBlockReduction() || domain->hasGridReduction())) ||
          (expr->isA<kir::BroadcastOp>() && domain->hasBlockBroadcast())) {
        return ir_builder.create<kir::Bool>(true);
      }
    }
  }

  auto out_tv = firstTvOutput(expr);

  // For the case of generating predicates, it's safe to assume all
  // axes are contiguous and saves some redundant predicates.
  auto pred_contiguity =
      std::vector<bool>(out_tv->domain()->rootDomain().size(), true);

  auto pred_inds =
      Index::getConsumerRootPredIndices(out_tv, loops, pred_contiguity);
  auto root_indices = pred_inds.first;
  const bool buffer_init = pred_inds.second;

  // If we are indexing a buffer init expr, and the buffer is local
  // memory, predicate is not needed as we allocate enough local memory.
  if (out_tv->memoryType() == MemoryType::Local && buffer_init) {
    return ir_builder.create<kir::Bool>(true);
  }

  // Don't generate predicates unless needed. This is just for
  // potential performance benefit.
  if (IterationDomainAnalysis::canOmitPredicate(out_tv)) {
    return thread_pred;
  }

  auto all_preds =
      PredicateCompute::computePredicates(out_tv, root_indices, buffer_init);
  // If we have thread predicates, add those
  if (thread_pred != nullptr) {
    all_preds.push_back(thread_pred);
  }

  std::vector<kir::Bool*> preds;

  for (auto pred : all_preds) {
    if (!pred->isConst() || !(pred->isConst() && pred->value().value())) {
      preds.push_back(pred);
    }
  }

  if (preds.empty()) {
    return ir_builder.create<kir::Bool>(true);
  }

  kir::Val* cond = preds[0];
  for (size_t i = 1; i < preds.size(); i++) {
    cond = ir_builder.andExpr(cond, preds[i]);
  }

  return cond->as<kir::Bool>();
}

kir::Bool* UnswitchPredicate::get(
    const std::vector<kir::ForLoop*>& outer_loops,
    kir::ForLoop* unrolled_loop,
    const IterDomainMap& p2c_root_map) {
  FUSER_PERF_SCOPE("UnswitchPredicate::get");

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  UnswitchPredicate up(outer_loops, unrolled_loop, p2c_root_map);

  std::unordered_set<kir::Bool*> pred_set;
  for (auto entry : up.predicates_) {
    pred_set.emplace(entry.second);
  }

  if (up.predicates_.empty()) {
    return ir_builder.create<kir::Bool>(true);
  }

  kir::Val* unroll_pred = nullptr;
  for (auto pred : pred_set) {
    if (unroll_pred == nullptr) {
      unroll_pred = pred;
    } else {
      unroll_pred = ir_builder.andExpr(unroll_pred, pred);
    }
  }

  TORCH_INTERNAL_ASSERT(unroll_pred != nullptr);

  return unroll_pred->as<kir::Bool>();
}

void UnswitchPredicate::predicateOn(kir::Expr* tv_expr) {
  FUSER_PERF_SCOPE("UnswitchPredicate::predicateOn");

  if (for_loops_.empty()) {
    return;
  }

  auto out_tv = firstTvOutput(tv_expr);

  // For the case of generating predicates, it's safe to assume all
  // axes are contiguous and saves some redundant predicates.
  auto pred_contiguity =
      std::vector<bool>(out_tv->domain()->rootDomain().size(), true);

  auto pred_inds = Index::getConsumerRootPredIndices(
      out_tv, for_loops_, pred_contiguity, true);
  auto root_indices = pred_inds.first;
  auto use_rfactor = pred_inds.second;

  auto all_preds =
      PredicateCompute::computePredicates(out_tv, root_indices, use_rfactor);

  const auto out_domain = out_tv->domain();
  const auto root_dom = (use_rfactor && out_domain->hasRFactor())
      ? out_domain->rfactorDomain()
      : out_domain->rootDomain();

  TORCH_INTERNAL_ASSERT(
      all_preds.size() == root_dom.size(),
      "Predicates should be produced for every dimension, even if it's simply set as true.");

  for (size_t i = 0; i < all_preds.size(); i++) {
    if (all_preds[i]->isConst() && all_preds[i]->value().value()) {
      continue;
    }
    const auto term_id = getTermIterDomainInMap(root_dom[i], p2c_root_map_);
    predicates_[term_id] = all_preds[i];
  }
}

void UnswitchPredicate::openLoop(kir::ForLoop* fl) {
  FUSER_PERF_SCOPE("UnswitchPredicate::openLoop");

  for_loops_.push_back(fl);

  for (auto expr : fl->body().exprs()) {
    if (ir_utils::isTVOp(expr)) {
      predicateOn(expr);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      openLoop(for_loop);
    }
  }

  for_loops_.pop_back();
}

UnswitchPredicate::UnswitchPredicate(
    std::vector<kir::ForLoop*> outer_loops,
    kir::ForLoop* unrolled_loop,
    const IterDomainMap& _p2c_root_map)
    : for_loops_(std::move(outer_loops)), p2c_root_map_(_p2c_root_map) {
  openLoop(unrolled_loop);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
