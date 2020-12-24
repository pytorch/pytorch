#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::vector<kir::Bool*> PredicateCompute::computePredicates(
    const TensorView* tv,
    const std::vector<Val*>& indices,
    bool use_rfactor) {
  FUSER_PERF_SCOPE("computePredicates");

  const std::vector<IterDomain*>& root =
      use_rfactor ? tv->getMaybeRFactorDomain() : tv->getRootDomain();

  TORCH_INTERNAL_ASSERT(root.size() == indices.size());

  bool no_pred_needed = true;
  for (auto id : tv->domain()->domain()) {
    if (id->getOrigin() != nullptr) {
      no_pred_needed = false;
    }
  }

  if (no_pred_needed) {
    return {};
  }

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  auto true_bool = ir_builder.create<kir::Bool>(true);
  std::vector<kir::Bool*> preds(root.size(), true_bool);
  Val* extent = nullptr;

  for (size_t i = 0; i < indices.size(); i++) {
    const bool zero_ind = indices[i]->isZeroInt();
    const bool simple_ind = indices[i]->getOrigin() == nullptr;

    if (root[i]->isBroadcast()) {
      continue;
    } else if (simple_ind && !zero_ind) {
      extent = nullptr;
      continue;
    } else if (zero_ind) {
      if (root[i]->extent()->isOneInt()) {
        continue;
      }
      const auto lowered_extent = GpuLower::lowerValue(root[i]->extent());
      if (extent == nullptr) {
        extent = lowered_extent;
      } else {
        extent = ir_builder.mulExpr(extent, lowered_extent);
      }
    } else {
      auto local_extent = GpuLower::lowerValue(root[i]->extent());
      if (extent != nullptr) {
        local_extent = ir_builder.mulExpr(extent, local_extent);
      }
      auto pred = ir_builder.ltExpr(indices[i], local_extent);
      extent = nullptr;
      TORCH_INTERNAL_ASSERT(
          pred->getValType().value() == ValType::KirScalar &&
          pred->getDataType().value() == DataType::Bool);
      preds[i] = pred->as<kir::Bool>();
    }
  }
  return preds;
}

kir::Bool* PredicateCompute::getInlinePredicate(
    Expr* expr,
    const std::vector<kir::ForLoop*>& loops,
    kir::Bool* thread_pred,
    bool ignore_block_grid_reductions) {
  FUSER_PERF_SCOPE("getInlinePredicate");

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  if (loops.empty()) {
    return ir_builder.create<kir::Bool>(true);
  }

  // Handle these elsewhere
  if (ignore_block_grid_reductions &&
      expr->getExprType() == ExprType::ReductionOp &&
      (expr->as<ReductionOp>()->out()->as<TensorView>()->hasBlockReduction() ||
       expr->as<ReductionOp>()->out()->as<TensorView>()->hasGridReduction())) {
    return ir_builder.create<kir::Bool>(true);
  }

  TORCH_INTERNAL_ASSERT(
      ir_utils::isTVOp(expr),
      "Cannot generate predicate based on operation without a TensorView.");

  auto out_tv = ir_utils::getTVOutput(expr);

  auto pred_contiguity = out_tv->domain()->contiguity();

  for (auto inp : expr->inputs()) {
    if (!ir_utils::isTV(inp)) {
      continue;
    }
    auto inp_tv = inp->as<TensorView>();
    if (inp_tv->domain()->hasRFactor()) {
      continue;
    } else if (
        inp_tv->getMemoryType() == MemoryType::Shared ||
        inp_tv->getMemoryType() == MemoryType::Local) {
      continue;
    } else {
      pred_contiguity = IndexCompute::contiguityAnd(
          pred_contiguity,
          IndexCompute::contiguityPasC(inp_tv->domain(), out_tv->domain()));
    }
  }

  auto pred_inds =
      Index::getConsumerRootPredIndices(out_tv, loops, pred_contiguity);
  auto root_indices = pred_inds.first;
  bool use_maybe_rfactor = pred_inds.second;

  if (out_tv->getMemoryType() == MemoryType::Local && out_tv->hasReduction() &&
      !use_maybe_rfactor) {
    auto tv_filter_inp_view =
        ir_utils::filterByType<TensorView>(expr->inputs());
    auto has_tv_inputs = tv_filter_inp_view.begin() != tv_filter_inp_view.end();
    // If predicates doesn't need maybe_rfactor, but it has reduction axes, and
    // expr has no inputs, we're pretty confident we're intializing a reduction
    // buffer. If we're initing a reduction buffer don't generate an inline
    // predicate.
    if (!has_tv_inputs) {
      return ir_builder.create<kir::Bool>(true);
    }
  }

  auto all_preds = PredicateCompute::computePredicates(
      out_tv, root_indices, use_maybe_rfactor);

  // If we have thread predicates, add those
  if (thread_pred != nullptr) {
    all_preds.push_back(thread_pred);
  }

  std::vector<kir::Bool*> preds;

  for (auto pred : all_preds)
    if (!(pred->isConst()) || !(pred->isConst() && pred->value().value()))
      preds.push_back(pred);

  if (preds.empty()) {
    return ir_builder.create<kir::Bool>(true);
  }

  Val* cond = preds[0];

  for (decltype(preds.size()) i{1}; i < preds.size(); i++) {
    cond = ir_builder.andExpr(cond, preds[i]);
  }

  TORCH_INTERNAL_ASSERT(
      cond->getValType().value() == ValType::KirScalar &&
          cond->getDataType().value() == DataType::Bool,
      "Error computing predicate, should be returning a Bool, but returning ",
      cond->getDataType().value());

  return cond->as<kir::Bool>();
}

kir::Bool* UnrollPredicate::get(
    const std::vector<kir::ForLoop*>& outer_loops,
    kir::ForLoop* unrolled_loop,
    const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map) {
  FUSER_PERF_SCOPE("UnrollPredicate::get");

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  UnrollPredicate up(outer_loops, unrolled_loop, p2c_root_map);

  std::unordered_set<kir::Bool*> pred_set;
  for (auto entry : up.predicates_) {
    pred_set.emplace(entry.second);
  }

  if (up.predicates_.empty()) {
    return ir_builder.create<kir::Bool>(true);
  }

  Val* unroll_pred = nullptr;
  for (auto pred : pred_set) {
    if (unroll_pred == nullptr) {
      unroll_pred = pred;
    } else {
      unroll_pred = ir_builder.andExpr(unroll_pred, pred);
    }
  }
  TORCH_INTERNAL_ASSERT(
      unroll_pred->getValType().value() == ValType::KirScalar &&
      unroll_pred->getDataType().value() == DataType::Bool);
  return unroll_pred->as<kir::Bool>();
}

void UnrollPredicate::predicateOn(Expr* tv_expr) {
  FUSER_PERF_SCOPE("UnrollPredicate::predicateOn");

  if (for_loops_.empty()) {
    return;
  }

  auto out_tv = ir_utils::getTVOutput(tv_expr);

  auto pred_contiguity = out_tv->domain()->contiguity();

  for (auto inp : tv_expr->inputs()) {
    if (!ir_utils::isTV(inp)) {
      continue;
    }
    auto inp_tv = inp->as<TensorView>();
    if (inp_tv->domain()->hasRFactor()) {
      continue;
    } else if (
        inp_tv->getMemoryType() == MemoryType::Shared ||
        inp_tv->getMemoryType() == MemoryType::Local) {
      continue;
    } else {
      pred_contiguity = IndexCompute::contiguityAnd(
          pred_contiguity,
          IndexCompute::contiguityPasC(inp_tv->domain(), out_tv->domain()));
    }
  }

  auto pred_inds = Index::getConsumerRootPredIndices(
      out_tv, for_loops_, pred_contiguity, true);
  auto root_indices = pred_inds.first;
  auto use_rfactor = pred_inds.second;

  auto all_preds =
      PredicateCompute::computePredicates(out_tv, root_indices, use_rfactor);

  auto root_dom =
      use_rfactor ? out_tv->getMaybeRFactorDomain() : out_tv->getRootDomain();

  TORCH_INTERNAL_ASSERT(
      all_preds.size() == root_dom.size(),
      "Predicates should be produced for every dimension, even if it's simply set as true.");

  for (size_t i = 0; i < all_preds.size(); i++) {
    if (all_preds[i]->isConst() && all_preds[i]->value().value()) {
      continue;
    }
    auto term_id = loop_utils::getTermIDInMap(root_dom[i], p2c_root_map_);
    predicates_[term_id] = all_preds[i];
  }
}

void UnrollPredicate::openLoop(kir::ForLoop* fl) {
  FUSER_PERF_SCOPE("UnrollPredicate::openLoop");

  for_loops_.push_back(fl);

  for (auto expr : fl->body().exprs()) {
    if (ir_utils::isTVOp(expr)) {
      predicateOn(expr);
    } else if (expr->getExprType().value() == ExprType::ForLoop) {
      openLoop(expr->as<kir::ForLoop>());
    }
  }

  for_loops_.pop_back();
}

UnrollPredicate::UnrollPredicate(
    std::vector<kir::ForLoop*> outer_loops,
    kir::ForLoop* unrolled_loop,
    const std::unordered_map<IterDomain*, IterDomain*>& _p2c_root_map)
    : for_loops_(std::move(outer_loops)), p2c_root_map_(_p2c_root_map) {
  openLoop(unrolled_loop);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
