#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Provide a new for loop matching the one provided
kir::ForLoop* cloneLoopNest(const kir::ForLoop* for_loop, bool unroll = false) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  const auto new_loop = ir_builder.create<kir::ForLoop>(
      for_loop->index(), for_loop->iter_domain(), unroll);
  for (auto expr : for_loop->body().exprs()) {
    if (auto nested_for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      expr = cloneLoopNest(nested_for_loop, unroll);
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

kir::Bool* UnrollPass::getThreadPredicate(const kir::TensorView* tv) {
  // No thread predicate is needed predicate when tv is output of a
  // parallel broadcast expression.
  if (auto bop = dynamic_cast<kir::BroadcastOp*>(tv->definition())) {
    TORCH_INTERNAL_ASSERT(bop->out()->isA<kir::TensorView>());
    const auto out = bop->out()->as<kir::TensorView>()->fuserTv();
    if (ir_utils::getParallelBroadcastDomains(out, thread_predicates_).any()) {
      return kir::IrBuilder(GpuLower::current()->kernel())
          .create<kir::Bool>(true);
    }
  }
  return thread_predicates_.getExpr(tv->fuserTv());
}

void UnrollPass::handle(kir::Expr* expr) {
  if (ir_utils::isTVOp(expr)) {
    // If tv op, predicate it
    const auto out_tv = expr->outputs()[0]->as<kir::TensorView>();
    const bool should_predicate = !for_loops_.empty() ||
        out_tv->memoryType() == MemoryType::Global ||
        out_tv->memoryType() == MemoryType::Shared;
    if (!should_predicate) {
      return;
    }
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());
    const auto thread_pred = isReductionInitExpr(expr)
        ? ir_builder.create<kir::Bool>(true)
        : getThreadPredicate(out_tv);
    const auto pred =
        PredicateCompute::getInlinePredicate(expr, for_loops_, thread_pred);

    TORCH_INTERNAL_ASSERT(pred != nullptr);

    // If we need a predicate, put expr inside an if then else
    if (!pred->isConst() || !(pred->isConst() && pred->value().value())) {
      non_trivial_pred_found_ = true;
      kir::IfThenElse* inline_ite = ir_builder.create<kir::IfThenElse>(pred);
      if (for_loops_.empty()) {
        // Special handling for top level output expressions that still
        // need predicates. One motivating example is a reduction op that
        // reduces to a scalar (issue #491)
        loop_replacement_map_.insert({expr, inline_ite});
      } else {
        for_loops_.back()->body().insert_before(expr, inline_ite);
        for_loops_.back()->body().erase(expr);
      }
      inline_ite->thenBody().push_back(expr);
    }
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
    for (auto expr : exprs_copy) {
      handle(expr);
    }

    for_loops_.pop_back();
    return;
  }

  auto unroll_pred = UnswitchPredicate::get(for_loops_, fl, p2c_root_map_);

  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  kir::IfThenElse* unroll_ite = ir_builder.create<kir::IfThenElse>(unroll_pred);

  // Get the loop nest for the unrolled path
  kir::ForLoop* unrolled_loop_nest = cloneLoopNest(fl, true);

  unroll_ite->thenBody().push_back(unrolled_loop_nest);
  if (fl->iter_domain()->parallelType() == ParallelType::Vectorize) {
    loop_replacement_map_.insert({fl, unroll_ite});
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
    loop_replacement_map_.insert({fl, inlined_loop});
  } else {
    if (!canOmitElseClause(fl)) {
      unroll_ite->elseBody().push_back(inlined_loop);
    }
    loop_replacement_map_.insert({fl, unroll_ite});
  }
}

bool UnrollPass::canOmitElseClause(kir::ForLoop* fl) const {
  kir::ExpressionEvaluator eval;
  std::vector<kir::ForLoop*> loops({fl});
  while (loops.size() > 0) {
    auto loop = loops.back();
    loops.pop_back();
    auto id = loop->iter_domain();
    if (id->isThread() || id->parallelType() == ParallelType::Vectorize) {
      continue;
    }
    const auto result = eval.evaluate(id->rawExtent());
    if (!(result.has_value() && result.value() == 1)) {
      return false;
    }
    for (auto nested_loop :
         ir_utils::filterByType<kir::ForLoop>(loop->body().exprs())) {
      loops.push_back(nested_loop);
    }
  }
  return true;
}

// Generate the loop nest structure and place it in lowered_exprs
void UnrollPass::computeMap(const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("UnrollPass::computeMap");

  // Run through loop nests and further lower the expressions
  for (auto* expr : exprs) {
    handle(expr);
  }
}

// TODO(kir): incorporate this into a new Scope interface
kir::Expr* UnrollPass::applyReplacements(kir::Expr* expr) const {
  auto handle_scope = [this](kir::Scope& scope) {
    for (size_t i = 0; i < scope.size(); ++i) {
      scope[i] = applyReplacements(scope[i]);
    }
  };

  const auto it = loop_replacement_map_.find(expr);
  if (it != loop_replacement_map_.end()) {
    return it->second;
  } else {
    if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle_scope(for_loop->body());
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle_scope(ite->thenBody());
      handle_scope(ite->elseBody());
    }
    return expr;
  }
}

std::vector<kir::Expr*> UnrollPass::runPass(
    Fusion* fusion,
    const std::vector<kir::Expr*>& exprs,
    const ThreadPredicateMap& thread_predicates) {
  FUSER_PERF_SCOPE("UnrollPass::runPass");

  UnrollPass unroll_pass(fusion, thread_predicates);
  unroll_pass.computeMap(exprs);

  std::vector<kir::Expr*> mutated_exprs;
  mutated_exprs.reserve(exprs.size());
  for (auto expr : exprs) {
    mutated_exprs.push_back(unroll_pass.applyReplacements(expr));
  }

  return mutated_exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
