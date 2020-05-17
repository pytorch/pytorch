#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {

bool operator==(
    const std::pair<IterDomain*, TensorView*>& p1,
    const std::pair<IterDomain*, TensorView*>& p2) {
  return p1.first->sameAs(p2.first) && p1.second == p2.second;
}

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
Bool* getPredicate(TensorView* tv, std::vector<Val*> inds) {
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

      if (has_global) {
        Bool* pred = getPredicate(out, ir_utils::indices(for_loops));

        // If we need a predicate, put expr inside an if then else

        if (!(pred->isConst()) || !(pred->isConst() && pred->value().value())) {
          IfThenElse* inline_ite =
              new IfThenElse(pred, {expr}, {}, for_loops.back());
          for_loops.back()->body().insert_before(expr, inline_ite);
          for_loops.back()->body().erase(expr);
        }
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
        tv->getComputeAtAxis(alloc_pos).first->parallel_method() ==
            ParallelType::Unroll) {
      reset = false;
      break;
    }
    alloc_pos++;
  }
  alloc_pos = reset ? 0 : alloc_pos;

  std::vector<Val*> alloc_dims;
  for (auto i = alloc_pos; i < tv->nDims(); i++) {
    IterDomain* dim = tv->getComputeAtAxis(i).first;
    if (dim->isThreadDim() || dim->isReduction())
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

void LoopNestGenerator::openFor(std::pair<IterDomain*, TensorView*> id_pair) {
  compute_at_scope.push_back(id_pair);
  IterDomain* id = id_pair.first;
  if (for_loops.size() > 0) {
    ForLoop* new_scope = scope_utils::openFor(for_loops.back(), id);
    for_loops.push_back(new_scope);
  } else {
    for_loops.push_back(scope_utils::openFor(nullptr, id));
    lowered_exprs.push_back(for_loops.back());
  }
}

void LoopNestGenerator::popFor() {
  TORCH_INTERNAL_ASSERT(
      !for_loops.empty() && !compute_at_scope.empty(),
      "Can't pop for loop, scope is empty.");
  for_loops.pop_back();
  compute_at_scope.pop_back();
}

void LoopNestGenerator::pushBack(Expr* expr) {
  if (for_loops.size() == 0)
    lowered_exprs.push_back(expr);
  else
    scope_utils::pushBack(for_loops.back(), expr);
}

// Update for loop structure based on this TensorView
void LoopNestGenerator::initReduction(TensorView* tv, Val* init_val) {
  TORCH_INTERNAL_ASSERT(
      tv->getComputeAtAxis() <= for_loops.size(),
      "Initialization of reduction was trying to be placed at the wrong point in the loop nest.");
  int depth = for_loops.size();
  for (decltype(tv->nDims()) i = depth; i < tv->nDims(); i++) {
    if (!tv->axis(i)->isReduction())
      openFor(tv->getComputeAtAxis(i));
  }
  auto clone = tv->unsafeClone();
  pushBack(new UnaryOp(UnaryOpType::Set, clone, init_val));

  while (for_loops.size() > (unsigned int)depth)
    popFor();
}

/*
 *  This is one of the most complex parts of the code lowering logic. what we
 * need to do is:
 *  1) Reduce loop structure if needed
 *  2) Open to compute At
 *    - If there is a computeAt set for this TV
 *  3) Allocate the output.
 *  4) If this is a reduction, initialize the output (open for loops to inner
 *       most, predicate, initialize, close predicate, close to computeAt)
 *  5) Open to inner most loop
 *  6) Run operation
 *  7) Close to computeAt
 */
void LoopNestGenerator::handle(Expr* expr) {
  if (!ir_utils::isTVOp(expr)) {
    for (auto out : expr->outputs())
      pushBack(new Allocate(out, new Int(1)));
    pushBack(expr);
    return;
  }

  TensorView* out = static_cast<TensorView*>(expr->output(0));
  // 1) Reduce loop structure
  while (compute_at_scope.size() > out->getComputeAtAxis() &&
         compute_at_scope.back().second != out &&
         compute_at_scope.back() !=
             out->getComputeAtAxis((int)compute_at_scope.size() - 1)) {
    popFor();
  }

  // 2) Open back up to computeAt
  while (compute_at_scope.size() < out->getComputeAtAxis()) {
    openFor(out->getComputeAtAxis((int)compute_at_scope.size()));
  }

  //  3) Allocate the output.
  if (!FusionGuard::getCurFusion()->hasInput(out) &&
      !FusionGuard::getCurFusion()->hasOutput(out))
    pushAlloc(out);

  //  4) If this is a reduction, initialize the output (open for loops to inner
  //  most, predicate, initialize, F predicate, close to computeAt)
  if (out->hasReduction())
    initReduction(out, static_cast<ReductionOp*>(expr)->init());

  //  5) Open to inner most loop
  for (decltype(out->nDims()) i = for_loops.size(); i < out->nDims(); i++)
    openFor(out->getComputeAtAxis(i));
  //  6) Run expression
  pushBack(expr);

  // 7) Reduce loop structure back to computeAt
  while (!compute_at_scope.empty() &&
         compute_at_scope.size() > out->getComputeAtAxis())
    popFor();
}

// Generate the loop nest structure and place it in lowered_exprs
void LoopNestGenerator::generate(const std::vector<Expr*>& exprs) {
  FusionGuard fg(fusion_);

  // Initialize members of the class
  lowered_exprs = std::vector<Expr*>();

  for (auto* expr : exprs)
    handle(expr);
}

} // namespace fuser
} // namespace jit
} // namespace torch