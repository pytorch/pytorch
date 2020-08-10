#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>

namespace torch {
namespace jit {
namespace fuser {

kir::Bool* UnrollPass::getThreadPredicate(TensorView* tv) {
  // No thread predicate is needed predicate when tv is output of a
  // parallel broadcast expression.
  const auto origin = tv->getOrigin();
  if (origin != nullptr && origin->getExprType() == ExprType::BroadcastOp) {
    const auto out = origin->as<BroadcastOp>()->out();
    if (ir_utils::getParallelBroadcastDomains(out, thread_predicates_).any()) {
      return nullptr;
    }
  }

  return thread_predicates_.getExpr(tv);
}

// Custom dispatch for Expr, want to find out of it's a TV op
void UnrollPass::handle(Expr* expr) {
  OptOutDispatch::handle(expr);
}

namespace {

bool initLocalBuffer(
    const std::vector<Expr*>& tv_ops,
    const std::unordered_set<Expr*>& init_exprs) {
  size_t num_init_exprs =
      std::count_if(tv_ops.begin(), tv_ops.end(), [&init_exprs](Expr* tv_op) {
        if (init_exprs.find(tv_op) == init_exprs.end()) {
          return false;
        }
        auto out = ir_utils::getTVOutput(tv_op);
        TORCH_INTERNAL_ASSERT(out != nullptr);
        return out->getMemoryType() == MemoryType::Local;
      });
  TORCH_INTERNAL_ASSERT(
      num_init_exprs == 0 || num_init_exprs == tv_ops.size(),
      "Some are local-buffer initializers but not all of them");
  return num_init_exprs == tv_ops.size();
}

kir::Bool* getPredicate(
    const std::vector<Expr*>& tv_ops,
    const std::vector<Val*>& inds_,
    kir::Bool* thread_pred,
    const std::unordered_set<Expr*>& init_exprs) {
  TORCH_INTERNAL_ASSERT(
      !tv_ops.empty() && !inds_.empty(),
      "Provided empty values to getPredicate.");

  std::vector<kir::Bool*> all_preds;

  if (!initLocalBuffer(tv_ops, init_exprs)) {
    // Need to start with an output to (effectively) grab its root domain size
    std::vector<bool> overall_contiguity =
        ir_utils::getTVOutput(tv_ops[0])->domain()->contiguity();

    // We want to get all the contiguity information from all TensorViews in the
    // exprs provided we need to support checking the predicate with the worst
    // case contiguity information across these TVs.
    for (auto tv_op : tv_ops) {
      TensorView* consumer_tv = nullptr;

      for (auto out : tv_op->outputs()) {
        if (!ir_utils::isTV(out))
          continue;
        consumer_tv = out->as<TensorView>();

        TORCH_INTERNAL_ASSERT(
            inds_.size() == consumer_tv->nDims() ||
                inds_.size() == consumer_tv->domain()->noReductions().size(),
            "Invalid indices vector provided for getPredicate");

        TORCH_INTERNAL_ASSERT(
            consumer_tv->domain()->contiguity().size() ==
                overall_contiguity.size(),
            "Invalid expressions in getPredicate, their out domains don't match up,",
            " they shouldn't be in the same loop nest together.");

        overall_contiguity = IndexCompute::contiguityAnd(
            overall_contiguity, consumer_tv->domain()->contiguity());
      }

      for (auto inp : tv_op->inputs()) {
        if (!ir_utils::isTV(inp))
          continue;
        overall_contiguity = IndexCompute::contiguityAnd(
            overall_contiguity,
            IndexCompute::contiguityPasC(
                inp->as<TensorView>()->domain(), consumer_tv->domain()));
      }
    }

    // Need a tv to base the indexing on, just grab the first.
    auto consumer_tv = ir_utils::getTVOutput(tv_ops[0]);

    // Do we need to adjust for reduction axes?
    const bool reductions = inds_.size() != consumer_tv->nDims();

    // Sanitize the indices
    std::vector<Val*> inds;
    if (reductions) {
      for (size_t ind_i = 0, consumer_tv_i = 0;
           consumer_tv_i < consumer_tv->nDims();) {
        if (consumer_tv->axis(consumer_tv_i++)->isReduction()) {
          inds.push_back(new kir::Int(0));
        } else {
          TORCH_INTERNAL_ASSERT(
              ind_i < inds_.size(),
              "Ran out of indices to generate predicate.");
          inds.push_back(inds_[ind_i++]);
        }
      }
    } else {
      inds = inds_;
    }

    // Compute indices based on consumer_tv and all contiguity information
    // combined
    all_preds = PredicateCompute::computePredicates(
        consumer_tv,
        IndexCompute::get(consumer_tv->domain(), inds, overall_contiguity));
  }

  // If we have thread predicates, add those
  if (thread_pred != nullptr) {
    all_preds.push_back(thread_pred);
  }

  std::vector<kir::Bool*> preds;

  for (auto pred : all_preds)
    if (!(pred->isConst()) || !(pred->isConst() && pred->value().value()))
      preds.push_back(pred);

  if (preds.empty()) {
    return new kir::Bool(true);
  }

  Val* cond = preds[0];

  for (decltype(preds.size()) i{1}; i < preds.size(); i++) {
    cond = kir::andExpr(cond, preds[i]);
  }

  TORCH_INTERNAL_ASSERT(
      cond->getValType().value() == ValType::KirScalar &&
          cond->getDataType().value() == DataType::Bool,
      "Error computing predicate, should be returning a Bool, but returning ",
      cond->getDataType().value());

  return cond->as<kir::Bool>();
}

} // namespace

// This function is one huge mess that should be refactored.
// It handles the unrolling and predicate generation
void UnrollPass::handle(kir::ForLoop* fl) {
  // Setup for loop scoping
  for_loops.push_back(fl);
  bool prev_unroll = within_unroll;
  within_unroll = ir_utils::isUnrolledFor(fl) || within_unroll;

  for (auto expr : fl->body().exprs()) {
    OptOutDispatch::handle(expr);
  }

  std::vector<Expr*> tv_ops;
  for (Expr* expr : fl->body().exprs()) {
    if (ir_utils::isTVOp(expr)) {
      // Predicate determining ops for unroll
      tv_ops.push_back(expr);
    }
  }

  bool has_TV_op = !tv_ops.empty();

  if (within_unroll && has_TV_op) {
    // Setup unrolled loop information:

    // Indices used to detect when we can unroll a loop safely
    // For loops outside the unroll, it's just the index, for loops inside
    // the unroll, if it's a thread it's the thread index, otherwise it's
    // the size-1
    std::vector<Val*> unroll_pred_inds;
    auto it = for_loops.begin();
    while (it != for_loops.end()) {
      if (ir_utils::isUnrolledFor(*it))
        break;
      unroll_pred_inds.push_back((*it)->index());
      it++;
    }

    TORCH_INTERNAL_ASSERT(
        it != for_loops.end(),
        "Error unrolling loops, expected an unrolled loop but wasn't found.");

    // This is the outer most loop that needs to be unrolled
    kir::ForLoop* first_unroll = *it;

    // Indicies inside the unroll
    while (it != for_loops.end()) {
      kir::IterDomain* id = (*it)->iter_domain();
      if (id->isThread())
        unroll_pred_inds.push_back((*it)->index());
      else
        unroll_pred_inds.push_back(kir::subExpr(id->extent(), new kir::Int(1)));
      it++;
    }

    // Make predicates for the unrolling, and the epilogue
    kir::Bool* unroll_predicate = getPredicate(
        tv_ops,
        unroll_pred_inds,
        getThreadPredicate(ir_utils::getTVOutput(tv_ops[0])),
        incoming_init_exprs_);

    // Make the IfThenElse controlling the unrolling
    kir::IfThenElse* unroll_ite = new kir::IfThenElse(
        unroll_predicate, {}, {}, first_unroll->parentScope());

    // Get the loop nest for the unrolled path
    kir::ForLoop* unrolled_loop =
        scope_utils::cloneLoopNest(first_unroll, unroll_ite);
    unroll_ite->body().push_back(unrolled_loop);

    // Loop nest for inlined path
    kir::ForLoop* inlined_loop =
        scope_utils::cloneLoopNest(first_unroll, unroll_ite);
    unroll_ite->elseBody().push_back(inlined_loop);

    // Inner most inlined loop
    Expr* inner_most_inlined_loop =
        scope_utils::firstInnerMostScope(inlined_loop);

    loop_replacement_map.insert({first_unroll, unroll_ite});

    for (auto expr : fl->body().exprs()) {
      if (!ir_utils::isTVOp(expr))
        continue;

      // Setup the expressions that need predicates around them.
      auto inline_predicate = getPredicate(
          {expr},
          ir_utils::indices(for_loops),
          getThreadPredicate(ir_utils::getTVOutput(expr)),
          incoming_init_exprs_);

      kir::IfThenElse* inline_ite = new kir::IfThenElse(
          inline_predicate, {expr}, {}, inner_most_inlined_loop);
      std::unordered_map<Expr*, Expr*> inline_replacement_map;
      inline_replacement_map.emplace(std::pair<Expr*, Expr*>(expr, inline_ite));
      scope_utils::replaceExprsInScope(
          inner_most_inlined_loop, inline_replacement_map);

    } // for expr
  } else { //  if(!within_unroll)
    // modify in place, so grab a copy of exprs first.
    const std::vector<Expr*> exprs = fl->body().exprs();

    for (auto expr : exprs) {
      if (!ir_utils::isTVOp(expr))
        continue;

      TensorView* out = ir_utils::asTV(ir_utils::asExpr(expr)->outputs()[0]);

      auto pred = getPredicate(
          {expr},
          ir_utils::indices(for_loops),
          getThreadPredicate(ir_utils::getTVOutput(expr)),
          incoming_init_exprs_);

      // If we need a predicate, put expr inside an if then else
      if (!(pred->isConst()) || !(pred->isConst() && pred->value().value())) {
        kir::IfThenElse* inline_ite =
            new kir::IfThenElse(pred, {expr}, {}, for_loops.back());
        for_loops.back()->body().insert_before(expr, inline_ite);
        for_loops.back()->body().erase(expr);
      }
    }
  } // else (if(!within_unroll))

  for_loops.pop_back();
  within_unroll = prev_unroll;
}

// Generate the loop nest structure and place it in lowered_exprs
void UnrollPass::computeMap() {
  FusionGuard fg(fusion_);

  // Run through loop nests and further lower the expressions
  for (auto* expr : incoming_exprs_) {
    OptOutDispatch::handle(expr);
  }
}

std::vector<Expr*> UnrollPass::runPass(
    Fusion* fusion,
    const std::vector<Expr*>& exprs,
    const std::unordered_set<Expr*>& init_exprs,
    const ThreadPredicateMap& thread_predicates) {
  FusionGuard fg(fusion);
  UnrollPass up(fusion, exprs, init_exprs, thread_predicates);
  up.computeMap();
  std::vector<Expr*> mutated_exprs;
  for (Expr* expr : exprs) {
    if (up.loop_replacement_map.find(expr) != up.loop_replacement_map.end()) {
      mutated_exprs.push_back(up.loop_replacement_map[expr]);
    } else {
      if (ir_utils::isScope(expr))
        scope_utils::replaceExprsInScope(expr, up.loop_replacement_map);
      mutated_exprs.push_back(expr);
    }
  }
  return mutated_exprs;
}

} // namespace fuser
} // namespace jit
} // namespace torch
