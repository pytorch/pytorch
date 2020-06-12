#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>

namespace torch {
namespace jit {
namespace fuser {

// Custom dispatch for Expr, want to find out of it's a TV op
void UnrollPass::handle(Expr* expr) {
  OptOutDispatch::handle(expr);
}

namespace {
Bool* getPredicate(TensorView* tv, std::vector<Val*> inds_) {
  TORCH_INTERNAL_ASSERT(
      inds_.size() == tv->nDims() ||
      inds_.size() == tv->domain()->noReductions().size());

  std::vector<Val*> inds;
  if (inds_.size() < tv->nDims()) {
    size_t i_ = 0;
    for (size_t i = 0; i < tv->nDims() && i_ < inds_.size(); i++) {
      if (tv->axis(i)->isReduction())
        inds.push_back(new Int(0));
      else
        inds.push_back(inds_[i_++]);
    }
  } else {
    inds = inds_;
  }
  if (tv->nDims() > inds.size()) {
    for (decltype(tv->nDims()) i{0}; i < tv->nDims(); i++) {
      if (tv->axis(i)->isReduction())
        inds.insert(inds.begin() + i, new Int(0));
    }
  }
  std::vector<Bool*> all_preds = PredicateCompute::computePredicates(
      new TensorIndex(tv, IndexCompute::get(tv->domain(), inds)));

  std::vector<Bool*> preds;

  for (Bool* pred : all_preds)
    if (!(pred->isConst()) || !(pred->isConst() && pred->value().value()))
      preds.push_back(pred);

  if (preds.size() == 0)
    return new Bool(true);

  Val* cond = preds[0];

  for (decltype(preds.size()) i{1}; i < preds.size(); i++) {
    cond = andOp(cond, preds[i]);
  }

  TORCH_INTERNAL_ASSERT(
      cond->getValType().value() == ValType::Scalar &&
          cond->getDataType().value() == DataType::Bool,
      "Error computing predicate, should be returning a Bool, but returning ",
      cond->getDataType().value());

  return static_cast<Bool*>(cond);
}
} // namespace

// This function is one huge mess that should be refactored.
// It handles the unrolling and predicate generation
void UnrollPass::handle(ForLoop* fl) {
  // Setup for loop scoping
  for_loops.push_back(fl);
  bool prev_unroll = within_unroll;
  within_unroll = ir_utils::isUnrolledFor(fl) || within_unroll;

  for (auto expr : fl->body().exprs()) {
    OptOutDispatch::handle(expr);
  }

  TensorView* out = nullptr;
  bool has_global = false;
  for (Expr* expr : fl->body().exprs())
    if (ir_utils::isTVOp(expr)) {
      // Predicate determining op for unroll
      out = ir_utils::asTV(expr->output(0));
      has_global = has_global || out->getMemoryType() == MemoryType::Global;
      for (auto inp : expr->inputs())
        if (ir_utils::isTV(inp))
          has_global = has_global ||
              ir_utils::asTV(inp)->getMemoryType() == MemoryType::Global;
    }

  bool has_TV_op = out != nullptr;

  if (within_unroll && has_TV_op && has_global) {
    // Setup unrolled loop information:

    // Indices used to detect when we can unroll a loop safely
    // For loops outside the unroll, it's just he index, for loops inside
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
    ForLoop* first_unroll = *it;

    // Indicies inside the unroll
    while (it != for_loops.end()) {
      IterDomain* id = (*it)->iter_domain();
      if (id->isThread())
        unroll_pred_inds.push_back((*it)->index());
      else
        unroll_pred_inds.push_back(sub(id->extent(), new Int(1)));
      it++;
    }

    // Make predicates for the unrolling, and the epilogue
    Bool* unroll_predicate = getPredicate(out, unroll_pred_inds);
    // Make the IfThenElse controlling the unrolling
    IfThenElse* unroll_ite =
        new IfThenElse(unroll_predicate, {}, {}, first_unroll->parentScope());

    // Get the loop nest for the unrolled path
    ForLoop* unrolled_loop =
        scope_utils::cloneLoopNest(first_unroll, unroll_ite);
    unroll_ite->body().push_back(unrolled_loop);

    // Loop nest for inlined path
    ForLoop* inlined_loop =
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
      Bool* inline_predicate = getPredicate(out, ir_utils::indices(for_loops));

      IfThenElse* inline_ite =
          new IfThenElse(inline_predicate, {expr}, {}, inner_most_inlined_loop);
      std::unordered_map<Expr*, Expr*> inline_replacement_map;
      inline_replacement_map.emplace(std::pair<Expr*, Expr*>(expr, inline_ite));
      scope_utils::replaceExprsInScope(
          inner_most_inlined_loop, inline_replacement_map);

    } // for expr
  } else { //  if(!within_unroll)
    // modify in place, so grab a copy of exprs first.
    std::vector<Expr*> exprs(
        fl->body().exprs().begin(), fl->body().exprs().end());

    for (auto expr : exprs) {
      if (!ir_utils::isTVOp(expr))
        continue;

      TensorView* out = ir_utils::asTV(ir_utils::asExpr(expr)->outputs()[0]);

      Bool* pred = getPredicate(out, ir_utils::indices(for_loops));

      // If we need a predicate, put expr inside an if then else
      if (!(pred->isConst()) || !(pred->isConst() && pred->value().value())) {
        IfThenElse* inline_ite =
            new IfThenElse(pred, {expr}, {}, for_loops.back());
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
    const std::vector<Expr*>& exprs) {
  FusionGuard fg(fusion);
  UnrollPass up(fusion, exprs);
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