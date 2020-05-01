#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

namespace torch {
namespace jit {
namespace fuser {
// all the way in the loop nest, grab predicate
/*
for( i : ceil(I/4) ) {
  for( j : ceil(J/128) ) {

    if( i * 4 + 3 < I && j * 128 + 127 < J ){
      for( k : 4)
        for( l : 128 )
          T0[ ( i * 4 + k ) * J + j * 128 + l ] = …
    } else {
      for( k : 4 )
        for( l : 128 )
          if( i * 4 + k < I && j * 128 + l < J)
             T0[ ( i * 4 + k ) * J + j * 128 + l ] = …
    }

  }
}
*/

// Custom dispatch for Expr, want to find out of it's a TV op
void UnrollPass::handle(Expr* expr) {
  OptOutDispatch::handle(expr);
}

namespace {
Int* getPredicate(const TensorView* const pred_tv, std::vector<Val*> indices) {
  TensorIndex* ti = new TensorIndex(
      pred_tv, IndexCompute::computeIndices(pred_tv, std::move(indices)));
  std::vector<Int*> all_preds = PredicateCompute::computePredicates(ti);

  std::vector<Int*> preds;

  Int* one = new Int(1);

  for (Int* pred : all_preds)
    if (!pred->sameAs(one))
      preds.push_back(pred);

  if (preds.size() == 0) {
    return one;
  } else {
    Int* cond = preds[0];

    for (decltype(preds.size()) i{1}; i < preds.size(); i++)
      cond = static_cast<Int*>(andOp(cond, preds[i]));

    return cond;
  }
}
} // namespace

// Open the for loop.
void UnrollPass::handle(ForLoop* fl) {
  // Setup for loop scoping
  for_loops.push_back(fl);
  bool prev_unroll = within_unroll;
  within_unroll = ir_utils::isUnrolledFor(fl) || within_unroll;

  for (auto expr : fl->body().exprs()) {
    OptOutDispatch::handle(expr);
  }

  TensorView* out;
  bool has_TV_op = false;
  for (Expr* expr : fl->body().exprs())
    if (ir_utils::isTVOp(expr)) {
      // Predicate determining op for unroll
      out = ir_utils::asTV(expr->output(0));
      has_TV_op = true;
      break;
    }

  if (within_unroll && has_TV_op) {
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
    Int* unroll_predicate = getPredicate(out, unroll_pred_inds);

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

    bool first_expr = true;
    for (auto expr : fl->body().exprs()) {
      if (!ir_utils::isTVOp(expr))
        continue;

      // Setup the expressions that need predicates around them.
      Int* inline_predicate =
          getPredicate(out, scope_utils::getLoopIndices(for_loops.back()));
      IfThenElse* inline_ite =
          new IfThenElse(inline_predicate, {expr}, {}, inner_most_inlined_loop);
      std::unordered_map<Expr*, Expr*> inline_replacement_map;
      inline_replacement_map.emplace(std::pair<Expr*, Expr*>(expr, inline_ite));
      scope_utils::replaceExprsInScope(
          inner_most_inlined_loop, inline_replacement_map);

    } // for expr

  } else { //  if(!within_unroll)

    for (auto expr : fl->body().exprs()) {
      if (!ir_utils::isTVOp(expr))
        continue;

      // ! within_unroll
      TensorView* out = ir_utils::asTV(ir_utils::asExpr(expr)->outputs()[0]);
      Int* pred =
          getPredicate(out, scope_utils::getLoopIndices(for_loops.back()));
      if (!pred->isOneInt()) {
        IfThenElse* inline_ite =
            new IfThenElse(pred, {expr}, {}, for_loops.back());
        for_loops.back()->body().insert_before(expr, inline_ite);
        for_loops.back()->body().erase(expr);
      }
    }
  } // else (if(!within_unroll))
  for_loops.pop_back();
  bool within_unroll = prev_unroll;
}

// Generate the loop nest structure and place it in lowered_exprs
void UnrollPass::computeMap() {
  FusionGuard fg(fusion_);

  // Initialize members of the class
  active_view = nullptr;
  active_view_axis = 0;

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

void LoopNestGenerator::pushAlloc(TensorView* tv) {
  TORCH_INTERNAL_ASSERT(
      !(FusionGuard::getCurFusion()->hasInput(tv) ||
        FusionGuard::getCurFusion()->hasOutput(tv)),
      "Tried to allocate an input or output tensor.");

  // Compute at axis can be == tv->nDims() meaning it's inline
  decltype(tv->nDims()) alloc_pos = 0;
  bool reset = true;
  while (alloc_pos <= tv->nDims()) {
    if (tv->hasComputeAt() && alloc_pos == tv->getComputeAtAxis()) {
      reset = false;
      break;
    }
    if (alloc_pos < tv->nDims() &&
        tv->getComputeAtAxis(alloc_pos)->parallel_method() ==
            ParallelType::Unroll) {
      reset = false;
      break;
    }
    alloc_pos++;
  }
  alloc_pos = reset ? 0 : alloc_pos;

  std::vector<Val*> alloc_dims;
  for (auto i = alloc_pos; i < tv->nDims(); i++) {
    IterDomain* dim = tv->getComputeAtAxis(i);
    if (dim->isThreadDim())
      continue;
    // TORCH_INTERNAL_ASSERT()
    alloc_dims.push_back(dim->extent());
  }

  Val* size;
  if (alloc_dims.size() == 0) {
    size = new Int(1);
  } else {
    size = alloc_dims[0];
    for (decltype(alloc_dims.size()) i{1}; i < alloc_dims.size(); i++) {
      size = mul(size, alloc_dims[i]);
    }
  }
  Allocate* alloc = new Allocate(tv, size);
  if (alloc_pos == 0) {
    lowered_exprs.insert(lowered_exprs.begin(), alloc);
  } else if (alloc_pos == for_loops.size()) {
    // inline
    scope_utils::pushBack(for_loops[alloc_pos - 1], alloc);
  } else {
    scope_utils::insertBefore(
        for_loops[alloc_pos - 1], for_loops[alloc_pos], alloc);
  }
}

// Clear out the last recorded computeAtView
void LoopNestGenerator::clearActiveView() {
  active_view_axis = 0;
  active_view = nullptr;
}

// Set active views from computeAtView
void LoopNestGenerator::setActiveView(const TensorView* const tv) {
  active_view_axis = tv->getComputeAtAxis();
  active_view = tv->getComputeAtView();
}

void LoopNestGenerator::openFor(IterDomain* id) {
  if (for_loops.size() > 0) {
    ForLoop* new_scope = scope_utils::openFor(for_loops.back(), id);
    for_loops.push_back(new_scope);
  } else {
    for_loops.push_back(scope_utils::openFor(nullptr, id));
    lowered_exprs.push_back(for_loops.back());
  }
}

void LoopNestGenerator::pushBack(Expr* expr) {
  if (for_loops.size() == 0)
    lowered_exprs.push_back(expr);
  else
    scope_utils::pushBack(for_loops.back(), expr);
}

/*
 *  This is one of the most complex parts of the code lowering logic. what we
 * need to do is: 1) Reduce loop structure
 *    - Reset all loops if active_view == nullptr (I'm not the last in a series
 * of computeAts)
 *    - Else reduce to active_view_axis if loop_depth > active_view_axis
 *  2) Set active_view(_axis)
 *    - If there is a computeAt set for this TV
 *  3) Open to compute At
 *    - If there is a computeAt set for this TV
 *  4) Allocate the output.
 *  5) If this is a reduction, initialize the output (open for loops to inner
 * most, predicate, initialize, close predicate, close to computeAt) 6) Open to
 * inner most loop 7) Open predicate 8) Run operation 9) Close predicate
 */

// Update fors based on tv.
void LoopNestGenerator::updateLoopNest(TensorView* tv) {
  // 1) Reduce loop structure
  if (active_view != nullptr) {
    // - Else reduce to active_view_axis if loop_depth > active_view_axis
    auto depth = for_loops.size();
    for (auto i = depth; i > active_view_axis; i--) {
      for_loops.pop_back();
    }
  }

  if (tv->hasComputeAt()) {
    //  2) Set active_view(_axis)
    //    - If there is a computeAt set for this TV
    setActiveView(tv);

    //  3) Open to compute At
    //    - If there is a computeAt set for this TV
    auto depth = for_loops.size();

    for (auto i = depth; i < tv->getComputeAtAxis(); i++)
      openFor(tv->getComputeAtAxis(i));
  } else {
    if (active_view != nullptr)
      // If we're the last computeAt of a block, active view should match this
      // tv
      TORCH_INTERNAL_ASSERT(
          tv->sameAs(active_view),
          "Error detected in code lowering. Expected ",
          active_view,
          " but recieved ",
          tv);

    clearActiveView();
  }
  //  4) Allocate the output.
  if (!FusionGuard::getCurFusion()->hasInput(tv) &&
      !FusionGuard::getCurFusion()->hasOutput(tv)) {
    pushAlloc(tv);
  }
  // TODO:
  //  5) If this is a reduction, initialize the output (open for loops to inner
  //  most, predicate, initialize, close predicate, close to computeAt)

  //  6) Open to inner most loop
  for (decltype(tv->nDims()) i = for_loops.size(); i < tv->nDims(); i++)
    openFor(tv->getComputeAtAxis(i));
}

// Custom dispatch for Expr, want to find out of it's a TV op
void LoopNestGenerator::handle(Expr* expr) {
  if (!ir_utils::isTVOp(expr))
    return;

  TensorView* out = static_cast<TensorView*>(expr->output(0));
  updateLoopNest(out);

  pushBack(expr);
}

// Generate the loop nest structure and place it in lowered_exprs
void LoopNestGenerator::generate() {
  FusionGuard fg(fusion_);

  // Initialize members of the class
  lowered_exprs = std::vector<Expr*>();
  active_view = nullptr;
  active_view_axis = 0;

  std::vector<Expr*> exprs = fusion_->exprs(true);
  for (auto* expr : exprs)
    handle(expr);
}

} // namespace fuser
} // namespace jit
} // namespace torch
