#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
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
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  const auto new_loop = ir_builder.create<kir::ForLoop>(
      for_loop->iter_domain(),
      for_loop->index(),
      for_loop->start(),
      for_loop->stop(),
      for_loop->step(),
      for_loop->vectorize(),
      for_loop->vectorize_shift());
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
bool isReductionInitExpr(const kir::Expr* expr) {
  // False if its output isn't a TensorView
  if (!ir_utils::isTVOp(expr)) {
    return false;
  }
  // False if it doesn't have any reduction axis
  const auto out_tv = expr->outputs()[0]->as<kir::TensorView>();
  if (!out_tv->domain()->hasReduction()) {
    return false;
  }
  // False if it has have TensorView inputs as initialization should
  // never use TensorViews
  const auto tv_filter_inp_view =
      ir_utils::filterByType<kir::TensorView>(expr->inputs());
  if (tv_filter_inp_view.begin() != tv_filter_inp_view.end()) {
    return false;
  }
  return true;
}

} // namespace

void UnrollPass::handle(kir::Expr* expr) {
  if (ir_utils::isTVOp(expr)) {
    // If tv op, predicate it
    const auto out_tv = ir_utils::getTVOutput(expr);
    const bool should_predicate = !for_loops_.empty() ||
        out_tv->memoryType() == MemoryType::Global ||
        out_tv->memoryType() == MemoryType::Shared;
    if (!should_predicate) {
      return;
    }

    kir::IrBuilder ir_builder(GpuLower::current()->kernel());
    const auto thread_pred = isReductionInitExpr(expr)
        ? ir_builder.trueVal()
        : GpuLower::current()->threadPredMap().getPredicate(out_tv->fuserTv());

    // When a predicate needs to account for ShiftOp, it is currently
    // taken care by its own function.
    if (GpuLower::current()->haloInfo().needsShiftPredicate(expr)) {
      ShiftPredicateInserter::insert(expr, for_loops_, thread_pred);
      return;
    }

    // Reduction may need a separate predicate for writes.
    if (!isReductionInitExpr(expr) && out_tv->domain()->hasReduction()) {
      const auto write_pred = ir_builder.create<kir::Predicate>(
          PredicateType::ReductionWrite, expr, thread_pred);
      expr->setWritePredicate(write_pred);
    }

    // For expr calling a device func with block sync, don't create
    // if-then-else but pass the predicate to the device func
    if (ir_utils::hasBlockSync(expr, GpuLower::current()->threadPredMap())) {
      const auto pred = ir_builder.create<kir::Predicate>(
          PredicateType::Inline, expr, thread_pred);
      expr->setPredicate(pred);
      return;
    }

    // Vectorized expressions should never use inline predicates
    kir::Predicate* vectorized_pred = nullptr;
    if (std::any_of(
            for_loops_.begin(), for_loops_.end(), [](const kir::ForLoop* fl) {
              return fl->iter_domain()->parallelType() ==
                  ParallelType::Vectorize;
            })) {
      vectorized_pred =
          ir_builder.create<kir::Predicate>(PredicateType::Vectorize);
    }

    const auto pred = vectorized_pred == nullptr
        ? ir_builder.create<kir::Predicate>(
              PredicateType::Inline, expr, thread_pred)
        : vectorized_pred;

    TORCH_INTERNAL_ASSERT(pred != nullptr);

    // If we need a predicate, put expr inside an if then else
    non_trivial_pred_found_ = true;
    kir::IfThenElse* inline_ite = ir_builder.create<kir::IfThenElse>(pred);
    if (for_loops_.empty()) {
      // Special handling for top level output expressions that still
      // need predicates. One motivating example is a reduction op that
      // reduces to a scalar (issue #491)
      expr_replacement_map_.insert({expr, inline_ite});
    } else {
      for_loops_.back()->body().insert_before(expr, inline_ite);
      for_loops_.back()->body().erase(expr);
    }
    inline_ite->thenBody().push_back(expr);
  } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
    handle(for_loop);
  }
}

// We should factor our actual predicate generation from unrolling but insering
// IR nodes "unroll_pred" or "inline_pred", then generate those later.
void UnrollPass::handle(kir::ForLoop* fl) {
  // Setup for loop scoping
  const bool is_unroll =
      fl->iter_domain()->parallelType() == ParallelType::Unroll ||
      fl->iter_domain()->parallelType() == ParallelType::Unswitch ||
      fl->iter_domain()->parallelType() == ParallelType::Vectorize;

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

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  auto unroll_pred = ir_builder.create<kir::Predicate>(fl);

  kir::IfThenElse* unroll_ite = ir_builder.create<kir::IfThenElse>(unroll_pred);

  // Get the loop nest for the unrolled path
  kir::ForLoop* unrolled_loop_nest = cloneLoopNest(fl);

  unroll_ite->thenBody().push_back(unrolled_loop_nest);
  if (fl->iter_domain()->parallelType() == ParallelType::Vectorize) {
    expr_replacement_map_.insert({fl, unroll_ite});
    return;
  }

  // Loop nest for inlined path
  kir::ForLoop* inlined_loop = cloneLoopNest(fl);

  // Add inline predicates for inlined loop nest
  look_for_unroll_ = false;
  non_trivial_pred_found_ = false;
  handle(inlined_loop);
  look_for_unroll_ = true;
  if (!non_trivial_pred_found_) {
    expr_replacement_map_.insert({fl, inlined_loop});
  } else {
    if (!canOmitElseClause(fl)) {
      unroll_ite->elseBody().push_back(inlined_loop);
    }
    expr_replacement_map_.insert({fl, unroll_ite});
  }
}

bool UnrollPass::canOmitElseClause(kir::ForLoop* fl) const {
  kir::ExpressionEvaluator eval;
  std::vector<kir::ForLoop*> loops({fl});

  const auto& pred_map = GpuLower::current()->threadPredMap();

  while (loops.size() > 0) {
    auto loop = loops.back();
    loops.pop_back();

    // If there's any expression that requires barrier
    // synchronization, the else part can't be omitted
    for (auto expr : loop->body().exprs()) {
      if (expr->isA<kir::BroadcastOp>()) {
        const ParallelTypeBitmap domains = pred_map.getParallelBroadcastDomains(
            expr->outputs()[0]->as<kir::TensorView>()->fuserTv());
        if (domains.any()) {
          return false;
        }
      } else if (expr->isA<kir::ReductionOp>() || expr->isA<kir::WelfordOp>()) {
        auto td = ir_utils::getTVOutput(expr)->domain();
        if (td->hasBlockReduction() || td->hasGridReduction()) {
          return false;
        }
      }
    }
    // If the number of visits of the loop body per thread is one, the
    // unswitch predicate is sufficient.
    // When the loop stop is the same as the extent of its IterDomain,
    // the per-thread visit count is guaranteed to be one at most (see
    // CudaKernelGenerator::visit(kir::ForLoop*) as well. Also, when a
    // loop is vectorized (not misaligned), the count must be one at
    // most. Even if not parallelized nor vectoirzed, it is also
    // sufficient if the loop stop is in fact one.
    bool visit_once = false;
    auto id = loop->iter_domain();
    if ((id->isThread() && (loop->stop() == id->extent())) ||
        id->parallelType() == ParallelType::Vectorize) {
      visit_once = true;
    }
    if (!visit_once) {
      const auto result = eval.evaluate(loop->stop());
      if (result.has_value() && result.value() == 1) {
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

// Generate the loop nest structure and place it in lowered_exprs
UnrollPass::UnrollPass(const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnrollPass::computeMap");

  // Run through loop nests and further lower the expressions
  for (auto* expr : exprs) {
    handle(expr);
  }
}

std::vector<kir::Expr*> UnrollPass::runPass(
    Fusion* fusion,
    const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::UnrollPass::runPass");

  UnrollPass unroll_pass(exprs);

  std::vector<kir::Expr*> mutated_exprs;
  mutated_exprs.reserve(exprs.size());
  for (auto expr : exprs) {
    mutated_exprs.push_back(
        ir_utils::applyReplacements(unroll_pass.replacementMap(), expr));
  }

  return mutated_exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
