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

  auto all_preds = Index::getReferenceRootPredicates(out_tv, loops);

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
  for (size_t i = 0; i < all_preds.first.size(); ++i) {
    auto pred = all_preds.first[i];
    if (pred_type == PredicateType::ReductionWrite) {
      const auto& concrete_root_ids = all_preds.second[i];
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
    if (!is_true(pred)) {
      preds.push_back(pred);
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

  auto pred_info = Index::getReferenceRootPredicates(out_tv, for_loops_, true);

  for (auto i : c10::irange(pred_info.first.size())) {
    auto pred = pred_info.first[i];
    if (pred->isConst() && pred->value()) {
      continue;
    }

    const auto& root_ids = pred_info.second[i];

    bool add_pred = false;

    for (auto root_id : root_ids) {
      auto kir_root_id = gpu_lower->lowerValue(root_id)->as<kir::IterDomain>();

      if (kir_root_id->isBroadcast()) {
        continue;
      }

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
