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

bool isOutputLocal(const kir::Expr* expr) {
  return std::all_of(
      expr->outputs().begin(),
      expr->outputs().end(),
      [](const kir::Val* output) {
        return !output->isA<kir::TensorView>() ||
            output->as<kir::TensorView>()->memoryType() == MemoryType::Local;
      });
}

} // namespace

UnswitchPredicateKey::UnswitchPredicateKey()
    : predicated_concrete_id_(nullptr) {
  for (auto pt : kParallelTypeThreads) {
    parallel_concrete_ids_.insert({pt, nullptr});
  }
}

// For a predicated concrete domain, id, find which thread parallel
// types are used. For each used parallel type, find the concrete
// domain that the paralllel type is associated with. The parallelized
// concrete domains are used to uniquely collect all necessary
// unswitch predicates.
UnswitchPredicateKey::UnswitchPredicateKey(
    IterDomain* predicated_concrete_id,
    const ReferenceTensor& reference)
    : predicated_concrete_id_(predicated_concrete_id) {
  // Initialize the parallelized domain map
  for (auto pt : kParallelTypeThreads) {
    parallel_concrete_ids_.insert({pt, nullptr});
  }

  // The id parameter is a concrete domain. Needs to find the
  // corresponding reference domain to find leaf domains that are
  // parallelized.
  IterDomain* predicated_ref_id =
      reference.concrete_to_id.at(predicated_concrete_id_);
  TensorDomain* ref_td = reference.domain;

  std::vector<Val*> all_parallelized_ref_leaf_ids;
  std::copy_if(
      ref_td->domain().begin(),
      ref_td->domain().end(),
      std::back_inserter(all_parallelized_ref_leaf_ids),
      [](IterDomain* x) { return isParallelTypeThread(x->getParallelType()); });

  // If the reference is not parallelized at all, no need to
  // differentiate keys based on how the predicated id is parallelized
  if (all_parallelized_ref_leaf_ids.empty()) {
    return;
  }

  // All domains that are parallelized descendants of predicated_ref_id
  auto all_parallelized_ref_ids = DependencyCheck::getAllValsBetween(
      {predicated_ref_id}, all_parallelized_ref_leaf_ids);
  // Just pick leaf domains
  std::vector<IterDomain*> parallelized_ref_leaf_ids;
  std::copy_if(
      ref_td->domain().begin(),
      ref_td->domain().end(),
      std::back_inserter(parallelized_ref_leaf_ids),
      [&](IterDomain* x) {
        return std::find(
                   all_parallelized_ref_ids.begin(),
                   all_parallelized_ref_ids.end(),
                   x) != all_parallelized_ref_ids.end();
      });

  if (parallelized_ref_leaf_ids.empty()) {
    // None of the parallelized leaf domains are derived from predicated_ref_id
    return;
  }

  // Find the corresponding concrete id for each parallel type
  for (auto ref_leaf : parallelized_ref_leaf_ids) {
    auto pt = ref_leaf->getParallelType();
    auto it = reference.id_to_concrete.find(ref_leaf);
    TORCH_INTERNAL_ASSERT(it != reference.id_to_concrete.end());
    auto concrete_leaf = it->second;
    parallel_concrete_ids_.at(pt) = concrete_leaf;
  }
}

std::string UnswitchPredicateKey::toString() const {
  std::stringstream ss;
  ss << "Predicated domain: " << predicatedId();
  for (auto pt : kParallelTypeThreads) {
    auto pid = parallelId(pt);
    ss << ", " << pt << ": ";
    if (pid) {
      ss << pid;
    } else {
      ss << "null";
    }
  }
  return ss.str();
}

std::size_t UnswitchPredicateKeyHash::operator()(
    const UnswitchPredicateKey& key) const {
  auto h = std::hash<const IterDomain*>{}(key.predicatedId());
  for (auto pt : kParallelTypeThreads) {
    h = h ^ std::hash<const IterDomain*>{}(key.parallelId(pt));
  }
  return h;
};

kir::Bool* PredicateCompute::getInlinePredicate(
    const kir::Expr* expr,
    const std::vector<kir::ForLoop*>& loops,
    kir::Bool* thread_pred,
    PredicateType pred_type) {
  FUSER_PERF_SCOPE("GpuLower::Lower::getInlinePredicate");

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // If outputs are registers, no need to predicate for threads
  if (isOutputLocal(expr)) {
    thread_pred = ir_builder.trueVal();
  }

  if (loops.empty()) {
    TORCH_INTERNAL_ASSERT(thread_pred != nullptr);
    return thread_pred;
  }

  auto out_tv = firstTensorViewOutput(expr);

  if (gpu_lower->predicateElimination().canOmitPredicate(expr)) {
    return thread_pred;
  }

  auto pred_info_vec = Index::getReferenceRootPredicates(out_tv, loops).first;

  std::vector<kir::Bool*> preds;

  auto is_true = [](const kir::Bool* p) {
    return p->isConst() && p->value().value();
  };

  // When pred_type is ReductionWrite, filter out predicates for
  // reduction axes. For blockReduce, this is necessary when reduction
  // axes start at non-zero offsets and parallelized with TID since
  // blockReduce returns a valid output only at offset-zero
  // threads. Similarly, for gridReduce, the last block to store the
  // output may be predicated out with the read predicate, so the
  // write predicate needs to ignore the reduction axes.
  bool non_zero_start_found = false;
  for (const auto& pred_info : pred_info_vec) {
    if (pred_type == PredicateType::ReductionWrite) {
      const auto& concrete_root_ids = pred_info.root_ids;
      bool pred_for_reduction_axis = false;
      for (auto pred_root_id : concrete_root_ids) {
        auto kir_pred_root_id =
            gpu_lower->lowerValue(pred_root_id)->as<kir::IterDomain>();
        auto it = std::find_if(
            out_tv->domain()->rootDomain().begin(),
            out_tv->domain()->rootDomain().end(),
            [&](const auto& out_root_id) {
              return gpu_lower->caIndexMap().areMapped(
                  kir_pred_root_id, out_root_id);
            });
        TORCH_INTERNAL_ASSERT(
            it != out_tv->domain()->rootDomain().end(),
            "No corresponding root ID found for ",
            pred_root_id);
        auto out_root_id = *it;
        if (out_root_id->isReduction()) {
          if (!out_root_id->start()->isZeroInt()) {
            non_zero_start_found = true;
          }
          pred_for_reduction_axis = true;
          break;
        }
      }
      // Don't add the predicate if it corresponds to a reduction axis
      if (pred_for_reduction_axis) {
        continue;
      }
    }
    // start may be nullptr. stop must be non-null
    if (pred_info.start && !is_true(pred_info.start)) {
      preds.push_back(pred_info.start);
    }
    if (!is_true(pred_info.stop)) {
      preds.push_back(pred_info.stop);
    }
  }

  // When generating a predicate for blockReduce writes and not for
  // gridReduce, if all reduction axes start with zero, we can just
  // use the same predicate for reads. nullptr is returned then.
  if (pred_type == PredicateType::ReductionWrite && !non_zero_start_found &&
      !out_tv->fuserTv()->domain()->hasGridReduction()) {
    return nullptr;
  }

  if (thread_pred != nullptr && !is_true(thread_pred)) {
    preds.push_back(thread_pred);
  }

  if (preds.empty()) {
    return ir_builder.trueVal();
  }

  kir::Val* cond = preds[0];
  for (size_t i = 1; i < preds.size(); i++) {
    cond = ir_builder.andExpr(cond, preds[i]);
  }

  return cond->as<kir::Bool>();
}

kir::Bool* UnswitchPredicate::get(
    const std::vector<kir::ForLoop*>& outer_loops,
    kir::ForLoop* unrolled_loop) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::get");

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
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::predicateOn");

  if (for_loops_.empty()) {
    return;
  }

  const auto gpu_lower = GpuLower::current();

  if (gpu_lower->predicateElimination().canOmitPredicate(tv_expr)) {
    return;
  }

  auto out_tv = firstTensorViewOutput(tv_expr);

  auto ref_pred_info =
      Index::getReferenceRootPredicates(out_tv, for_loops_, true);
  ReferenceTensor& reference = ref_pred_info.second;

  for (const auto& pred_info : ref_pred_info.first) {
    auto pred = pred_info.stop;
    if (pred->isConst() && pred->value()) {
      continue;
    }

    const auto& root_ids = pred_info.root_ids;

    bool add_pred = false;

    for (auto root_id : root_ids) {
      auto kir_root_id = gpu_lower->lowerValue(root_id)->as<kir::IterDomain>();

      if (kir_root_id->isBroadcast()) {
        continue;
      }

      UnswitchPredicateKey key(root_id, reference);

      if (predicated_keys_.find(key) == predicated_keys_.end()) {
        add_pred = true;
        predicated_keys_.insert(key);
      }
    }
    if (add_pred) {
      predicates_.push_back(pred);
    }
  }
}

void UnswitchPredicate::openLoop(kir::ForLoop* fl) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::openLoop");

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
  FUSER_PERF_SCOPE("GpuLower::Lower::UnswitchPredicate::openIte");

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
