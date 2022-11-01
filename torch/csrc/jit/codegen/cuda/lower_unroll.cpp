#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_misaligned_vectorization.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Provide a new for loop matching the one provided
kir::ForLoop* cloneLoopNest(const kir::ForLoop* for_loop) {
  const auto new_loop = IrBuilder::create<kir::ForLoop>(for_loop);
  for (auto expr : for_loop->body().exprs()) {
    if (auto nested_for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      expr = cloneLoopNest(nested_for_loop);
    }
    new_loop->body().push_back(expr);
  }
  return new_loop;
}

// Returns true if expr is an expression that initializes a reduction
// buffer.
bool isReductionInitExpr(const Expr* expr) {
  // False if its output isn't a TensorView
  if (!ir_utils::isTvOp(expr)) {
    return false;
  }
  // False if it doesn't have any reduction axis
  const auto out_tv = expr->outputs()[0]->as<TensorView>();
  if (!out_tv->domain()->hasReduction()) {
    return false;
  }
  // False if it has have TensorView inputs as initialization should
  // never use TensorViews
  const auto tv_filter_inp_view =
      ir_utils::filterByType<TensorView>(expr->inputs());
  if (tv_filter_inp_view.begin() != tv_filter_inp_view.end()) {
    return false;
  }
  return true;
}

} // namespace

void UnrollPass::registerReplace(
    Expr* reference,
    Expr* new_expr,
    kir::Scope* scope) {
  kir::ExprMutator::registerReplace(reference, new_expr, scope);
  GpuLower::current()->propagateExprInfo(reference, new_expr);
}

void UnrollPass::handle(Expr* expr) {
  if (ir_utils::isTvOp(expr)) {
    // If tv op, predicate it
    const auto out_tv = ir_utils::getTvOutput(expr);
    const bool should_predicate = !for_loops_.empty() ||
        out_tv->getMemoryType() == MemoryType::Global ||
        out_tv->getMemoryType() == MemoryType::Shared;
    if (!should_predicate) {
      return;
    }

    const auto thread_pred = isReductionInitExpr(expr)
        ? GpuLower::current()->kernel()->trueVal()
        : GpuLower::current()->threadPredMap().getPredicate(out_tv);

    // When this expr is in an unswitched block, only attach the
    // thread predicate to the expr as thread predicates are not
    // grouped to the unswitch predicate.
    kir::Predicate* thread_pred_expr = nullptr;
    if (unswitched_loop_) {
      thread_pred_expr = IrBuilder::create<kir::Predicate>(thread_pred);
    }

    non_trivial_pred_found_ = true;

    Expr* expr_with_predicate = expr;

    // When a predicate needs to account for ShiftOp, it is currently
    // taken care by its own function.
    if (GpuLower::current()->haloInfo()->needsShiftPredicate(expr)) {
      expr_with_predicate = ShiftPredicateInserter::insert(
          expr, for_loops_, thread_pred, unswitched_loop_);
      if (expr_with_predicate != expr) {
        registerReplace(expr, expr_with_predicate, &for_loops_.back()->body());
      }
      return;
    }

    // Reduction may need a separate predicate for writes.
    if (!isReductionInitExpr(expr) && out_tv->domain()->hasReduction()) {
      const auto write_pred = unswitched_loop_
          ? thread_pred_expr
          : IrBuilder::create<kir::Predicate>(
                PredicateType::ReductionWrite, expr, thread_pred);
      expr_with_predicate = expr_with_predicate->withWritePredicate(write_pred);
    }

    // For expr calling a device func with block sync, don't create
    // if-then-else but pass the predicate to the device func
    if (lower_utils::hasBlockSync(expr, GpuLower::current()->threadPredMap())) {
      const auto pred = unswitched_loop_
          ? thread_pred_expr
          : IrBuilder::create<kir::Predicate>(
                PredicateType::Inline, expr, thread_pred);
      expr_with_predicate = expr_with_predicate->withPredicate(pred);
      registerReplace(expr, expr_with_predicate, &for_loops_.back()->body());
      return;
    }

    // Vectorized expressions should never use inline predicates
    kir::Predicate* pred = nullptr;
    if (!unswitched_loop_ &&
        std::any_of(
            for_loops_.begin(), for_loops_.end(), [](const kir::ForLoop* fl) {
              return fl->iter_domain()->getParallelType() ==
                  ParallelType::Vectorize;
            })) {
      pred = IrBuilder::create<kir::Predicate>(PredicateType::Vectorize);
    }

    if (pred == nullptr) {
      pred = unswitched_loop_ ? thread_pred_expr
                              : IrBuilder::create<kir::Predicate>(
                                    PredicateType::Inline, expr, thread_pred);
    }

    if (lower_utils::supportInlinePredicate(expr)) {
      expr_with_predicate = expr_with_predicate->withPredicate(pred);
      registerReplace(expr, expr_with_predicate, &for_loops_.back()->body());
      return;
    }

    // If we need a predicate, put expr inside an if then else
    kir::IfThenElse* inline_ite = IrBuilder::create<kir::IfThenElse>(pred);
    if (for_loops_.empty()) {
      // Special handling for top level output expressions that still
      // need predicates. One motivating example is a reduction op that
      // reduces to a scalar (issue #491)
      kir::ExprMutator::registerReplace(expr, inline_ite, nullptr);
    } else {
      kir::ExprMutator::registerReplace(
          expr, inline_ite, &for_loops_.back()->body());
    }
    if (expr != expr_with_predicate) {
      GpuLower::current()->propagateExprInfo(expr, expr_with_predicate);
    }
    inline_ite->thenBody().push_back(expr_with_predicate);
  } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
    handle(for_loop);
  }
}

// We should factor our actual predicate generation from unrolling but insering
// IR nodes "unroll_pred" or "inline_pred", then generate those later.
void UnrollPass::handle(kir::ForLoop* fl) {
  // Setup for loop scoping
  const bool is_unroll =
      fl->iter_domain()->getParallelType() == ParallelType::Unroll ||
      fl->iter_domain()->getParallelType() == ParallelType::Unswitch;

  // If we're not looking for an unroll loop, or didn't find one, process as
  // normal.
  if (!is_unroll || !look_for_unroll_) {
    for_loops_.push_back(fl);

    // Make copy of exprs because we replace them inplace in fl
    const auto exprs_copy = fl->body().exprs();

    // Skip Misaligned Vectorization For-Loops here
    if (!containsAnyDirectChildMisalignedVectorize(fl)) {
      for (auto expr : exprs_copy) {
        handle(expr);
      }
    }

    for_loops_.pop_back();
    return;
  }

  auto unroll_pred = IrBuilder::create<kir::Predicate>(fl);

  kir::IfThenElse* unroll_ite = IrBuilder::create<kir::IfThenElse>(unroll_pred);

  // Get the loop nest for the unrolled path
  kir::ForLoop* unrolled_loop_nest = cloneLoopNest(fl);

  // Thread predicates are not removed from the expressions. Visit
  // each expression to attach kir::Predicate.
  unswitched_loop_ = true;
  look_for_unroll_ = false;
  handle(unrolled_loop_nest);
  unswitched_loop_ = false;
  look_for_unroll_ = true;

  unroll_ite->thenBody().push_back(unrolled_loop_nest);

  // Loop nest for inlined path
  kir::ForLoop* inlined_loop = cloneLoopNest(fl);

  // Add inline predicates for inlined loop nest
  look_for_unroll_ = false;
  non_trivial_pred_found_ = false;
  handle(inlined_loop);
  look_for_unroll_ = true;
  if (!non_trivial_pred_found_) {
    kir::ExprMutator::registerReplace(
        fl,
        inlined_loop,
        for_loops_.empty() ? nullptr : &for_loops_.back()->body());
  } else {
    if (!canOmitElseClause(fl)) {
      unroll_ite->elseBody().push_back(inlined_loop);
    }
    kir::ExprMutator::registerReplace(
        fl,
        unroll_ite,
        for_loops_.empty() ? nullptr : &for_loops_.back()->body());
  }
}

bool UnrollPass::canOmitElseClause(kir::ForLoop* fl) {
  std::vector<kir::ForLoop*> loops({fl});

  const auto& pred_map = GpuLower::current()->threadPredMap();

  while (loops.size() > 0) {
    auto loop = loops.back();
    loops.pop_back();

    // If there's any expression that requires barrier
    // synchronization, the else part can't be omitted
    for (auto expr : loop->body().exprs()) {
      if (lower_utils::hasBlockSync(expr, pred_map)) {
        return false;
      }
    }
    // If the number of visits of the loop body per thread is one, the
    // unswitch predicate is sufficient.
    // When the loop stop is the same as the extent of its IterDomain,
    // the per-thread visit count is guaranteed to be one at most (see
    // CudaKernelGenerator::handle(kir::ForLoop*) as well. Also, when a
    // loop is vectorized (not misaligned), the count must be one at
    // most. Even if not parallelized nor vectoirzed, it is also
    // sufficient if the loop stop is in fact one.
    bool visit_once = false;
    auto id = loop->iter_domain();
    if ((id->isThread() && (loop->stop() == id->extent())) ||
        id->getParallelType() == ParallelType::Vectorize) {
      visit_once = true;
    }
    if (!visit_once) {
      if (loop->stop()->isConstInt() && loop->stop()->evaluateInt() == 1) {
        visit_once = true;
      }
    }

    // The visit count is not guaranteed to be one, so the else part
    // must be created.
    if (!visit_once) {
      return false;
    }

    // The unswitch predicate is sufficient for this loop. Proceed to
    // nested loops.
    for (auto nested_loop :
         ir_utils::filterByType<kir::ForLoop>(loop->body().exprs())) {
      loops.push_back(nested_loop);
    }
  }

  return true;
}

UnrollPass::UnrollPass(const std::vector<Expr*>& exprs) {
  kir::ExprMutator::traverseAndInsert(exprs);
}

std::vector<Expr*> UnrollPass::runPass(
    Fusion* fusion,
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnrollPass::runPass");

  UnrollPass unroll_pass(exprs);
  return unroll_pass.exprs_;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
