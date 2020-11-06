#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
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
const kir::TensorView* firstTvOutput(const kir::Expr* expr) {
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
    bool use_rfactor) {
  FUSER_PERF_SCOPE("computePredicates");

  const auto domain = tv->domain();
  const auto& root = (use_rfactor && domain->hasRFactor())
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

  if (no_pred_needed) {
    return {};
  }

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  auto true_bool = ir_builder.create<kir::Bool>(true);
  std::vector<kir::Bool*> preds(root.size(), true_bool);
  kir::Val* extent = nullptr;

  for (size_t i = 0; i < indices.size(); i++) {
    const bool zero_ind = indices[i]->isZeroInt();
    const bool simple_ind = indices[i]->definition() == nullptr;

    if (root[i]->isBroadcast()) {
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

kir::Bool* PredicateCompute::getInlinePredicate(
    const kir::Expr* expr,
    const std::vector<kir::ForLoop*>& loops,
    kir::Bool* thread_pred,
    const ComputeAtRootDomainMap& ca_root_map,
    bool ignore_block_grid_reductions) {
  FUSER_PERF_SCOPE("getInlinePredicate");

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  if (loops.empty()) {
    return ir_builder.create<kir::Bool>(true);
  }

  // Handle these elsewhere
  if (ignore_block_grid_reductions) {
    if (auto reduction_op = dynamic_cast<const kir::ReductionOp*>(expr)) {
      const auto domain = reduction_op->out()->as<kir::TensorView>()->domain();
      if (domain->hasBlockReduction() || domain->hasGridReduction()) {
        return ir_builder.create<kir::Bool>(true);
      }
    }
  }

  const auto out_tv = firstTvOutput(expr);

  auto pred_contiguity = out_tv->domain()->contiguity();

  for (auto inp : expr->inputs()) {
    if (auto inp_tv = dynamic_cast<kir::TensorView*>(inp)) {
      if (inp_tv->domain()->hasRFactor() ||
          inp_tv->memoryType() == MemoryType::Shared ||
          inp_tv->memoryType() == MemoryType::Local) {
        continue;
      } else {
        pred_contiguity = IndexCompute::contiguityAnd(
            pred_contiguity,
            IndexCompute::contiguityPasC(inp_tv->domain(), out_tv->domain()));
      }
    }
  }

  auto pred_inds = Index::getConsumerRootPredIndices(
      out_tv, loops, pred_contiguity, ca_root_map);
  auto root_indices = pred_inds.first;
  bool use_maybe_rfactor = pred_inds.second;

  if (out_tv->memoryType() == MemoryType::Local &&
      out_tv->domain()->hasReduction() && !use_maybe_rfactor) {
    const auto tv_filter_inp_view =
        ir_utils::filterByType<kir::TensorView>(expr->inputs());
    const auto has_tv_inputs =
        tv_filter_inp_view.begin() != tv_filter_inp_view.end();
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

kir::Bool* UnrollPredicate::get(
    const std::vector<kir::ForLoop*>& outer_loops,
    kir::ForLoop* unrolled_loop,
    const IterDomainMap& p2c_root_map,
    const ComputeAtRootDomainMap& ca_root_map) {
  FUSER_PERF_SCOPE("UnrollPredicate::get");

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  UnrollPredicate up(outer_loops, unrolled_loop, p2c_root_map, ca_root_map);

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

  return unroll_pred->as<kir::Bool>();
}

void UnrollPredicate::predicateOn(kir::Expr* tv_expr) {
  FUSER_PERF_SCOPE("UnrollPredicate::predicateOn");

  if (for_loops_.empty()) {
    return;
  }

  const auto out_tv = firstTvOutput(tv_expr);

  auto pred_contiguity = out_tv->domain()->contiguity();

  for (auto inp : tv_expr->inputs()) {
    if (auto inp_tv = dynamic_cast<kir::TensorView*>(inp)) {
      if (inp_tv->domain()->hasRFactor() ||
          inp_tv->memoryType() == MemoryType::Shared ||
          inp_tv->memoryType() == MemoryType::Local) {
        continue;
      } else {
        pred_contiguity = IndexCompute::contiguityAnd(
            pred_contiguity,
            IndexCompute::contiguityPasC(inp_tv->domain(), out_tv->domain()));
      }
    }
  }

  auto pred_inds = Index::getConsumerRootPredIndices(
      out_tv, for_loops_, pred_contiguity, ca_root_map_, true);
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

void UnrollPredicate::openLoop(kir::ForLoop* fl) {
  FUSER_PERF_SCOPE("UnrollPredicate::openLoop");

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

UnrollPredicate::UnrollPredicate(
    std::vector<kir::ForLoop*> outer_loops,
    kir::ForLoop* unrolled_loop,
    const IterDomainMap& _p2c_root_map,
    const ComputeAtRootDomainMap& ca_root_map)
    : for_loops_(std::move(outer_loops)),
      p2c_root_map_(_p2c_root_map),
      ca_root_map_(ca_root_map) {
  openLoop(unrolled_loop);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
