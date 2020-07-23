#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <numeric>

namespace torch {
namespace jit {
namespace fuser {

// Create, place, and return the allocation for tv
Expr* LoopNestGenerator::pushAlloc(TensorView* tv) {
  TORCH_INTERNAL_ASSERT(
      !(FusionGuard::getCurFusion()->hasInput(tv) ||
        FusionGuard::getCurFusion()->hasOutput(tv)),
      "Tried to allocate an input or output tensor.");

  // First figure out which loop nest this allocation needs to be placed in
  // Do we need to place the allocation at the root?
  size_t alloc_pos = 0;
  // If there's no computeAt, then we want to be allocated at the root
  while (alloc_pos <= tv->nDims() && tv->hasComputeAt()) {
    // If we have a computeAt and we reached computeAt pos that's where it  goes
    if (tv->hasComputeAt() && alloc_pos == tv->getThisComputeAtAxis()) {
      break;
    }
    // If we found an unroll, we want to place the allocation outside the unroll
    if (alloc_pos < tv->nDims() &&
        tv->getComputeAtAxis(alloc_pos).first->getParallelType() ==
            ParallelType::Unroll) {
      break;
    }
    alloc_pos++;
  }

  // Grab the dimensions the allocation will be based on
  std::vector<Val*> alloc_dims;
  for (auto i = alloc_pos; i < tv->nDims(); i++) {
    IterDomain* compute_at_dim = tv->getComputeAtAxis(i).first;
    IterDomain* local_dim = tv->axis(i);
    if (
        // If shared memory, don't use any IDs bound to a grid dimension
        (tv->memory_type_ == MemoryType::Shared &&
         compute_at_dim->isBlockDim()) ||
        // If local memory, don't use any IDs bound to a grid or block dimension
        (tv->memory_type_ == MemoryType::Local && compute_at_dim->isThread()) ||
        // If we're reducing this dimension, don't use it in the allocation
        // computation
        local_dim->isReduction() ||
        // If this is a broadcast dimension, don't use it in the allocation
        // computation
        local_dim->isBroadcast()) {
      continue;
    }
    alloc_dims.push_back(compute_at_dim->extent());
  }

  // Multiply all the dimensions we're going to use for the allocation together
  // to get the total size
  Val* size = nullptr;
  if (alloc_dims.size() == 0) {
    size = new Int(1);
  } else {
    size = alloc_dims[0];
    for (size_t i = 1; i < alloc_dims.size(); i++) {
      size = mul(size, alloc_dims[i]);
    }
  }

  // Create the allocation node
  kir::Allocate* alloc = new kir::Allocate(tv, MemoryType::Local, size);

  // Place the allocation
  if (alloc_pos == 0) {
    // If we allocate at the root, insert at the begining of the lowered
    // expressions
    lowered_exprs.insert(lowered_exprs.begin(), alloc);
  } else if (alloc_pos == for_loops.size()) {
    // If we allocate inline, push to the back of the last for loop
    scope_utils::pushBack(for_loops[for_loops.size() - 1], alloc);
  } else {
    // Otherwise we allocate in some loop nest that is not inline, or root, so
    // insert right before the loop we're just outside of
    scope_utils::insertBefore(
        for_loops[alloc_pos - 1], for_loops[alloc_pos], alloc);
  }

  return alloc;
}

void LoopNestGenerator::openFor(std::pair<IterDomain*, TensorView*> id_pair) {
  compute_at_scope.push_back(id_pair);
  IterDomain* id = id_pair.first;
  if (for_loops.size() > 0) {
    kir::ForLoop* new_scope = scope_utils::openFor(for_loops.back(), id);
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

// Update for loop structure based on this TensorView, if there's an allocation
// stmt, send it in so we can make sure that we insert this initialization after
// it
void LoopNestGenerator::initReduction(
    TensorView* tv,
    Val* init_val,
    Expr* alloc_expr) {
  // This logic was taken from pushAlloc, as the initialization loop nest will
  // go at the same place.

  // First figure out which loop nest this allocation needs to be placed in
  // Do we need to place the allocation at the root?
  size_t alloc_pos = 0;
  // If there's no computeAt, then we want to be allocated at the root
  while (alloc_pos <= tv->nDims() && tv->hasComputeAt()) {
    // If we have a computeAt and we reached computeAt pos that's where it  goes
    if (tv->hasComputeAt() && alloc_pos == tv->getThisComputeAtAxis()) {
      break;
    }

    // If we found an unroll, we want to place the allocation outside the unroll
    if (alloc_pos < tv->nDims() &&
        tv->getComputeAtAxis(alloc_pos).first->getParallelType() ==
            ParallelType::Unroll) {
      break;
    }
    alloc_pos++;
  }

  // Grab the IDs that will be involved in the initialization, ignore reduction
  // dimensions. Everything else will be iterated over to cover the entire
  // buffer. Index compute will ignore [block, grid]Dims depending on buffer
  // memory location
  std::vector<IterDomain*> ids;
  for (auto i = alloc_pos; i < tv->nDims(); i++) {
    IterDomain* dim = tv->getComputeAtAxis(i).first;
    if (dim->isReduction())
      continue;
    ids.push_back(dim);
  }

  // Unsafe clone, as we want an exact replica of tv so we can create a UnaryOp
  // to set the buffer to the init_val.
  auto clone = tv->unsafeClone();
  thread_predicates_.duplicate(clone, tv);
  // The initilization stmt that will be located inside the loop nest (if there
  // is one)
  auto init_stmt = new UnaryOp(UnaryOpType::Set, clone, init_val);

  // Init a pointer that will become the entirety of the initialization
  Expr* init_loop_nest = nullptr;

  // The for loop that we will place the initialization within (alloc_pos - 1),
  // if one exists. Once we're done this inner_fl will be the inner most loop
  // containing the init_stmt
  kir::ForLoop* inner_fl = nullptr;
  if (alloc_pos >= 1)
    inner_fl = for_loops[alloc_pos - 1];

  // Work through the iter domains that we need to initialize on, outside to
  // inside, to construct the loop nest for the initialization.
  for (auto id : ids) {
    kir::ForLoop* new_fl;

    if (id->isThread()) {
      // If based on a thread, make sure we get the named Int right
      std::stringstream ss;
      ss << id->getParallelType();
      new_fl = new kir::ForLoop(
          new NamedScalar(ss.str(), DataType::Int), id, {}, inner_fl);
    } else {
      // Otherwise it's just a new int-
      new_fl = new kir::ForLoop(new Int(), id, {}, inner_fl);
    }

    if (init_loop_nest == nullptr) {
      // If this is our first generated loop, then it will be our outer most
      // loop nest
      init_loop_nest = new_fl;
    } else {
      // Otherwise place it inside the last generated loop
      inner_fl->body().push_back(new_fl);
    }
    // Increment the inner most for loop
    inner_fl = new_fl;
  }

  if (init_loop_nest == nullptr) {
    // If no loops were generated, than our init_stmt is all we need
    init_loop_nest = init_stmt;
  } else {
    // If there were for loops generated, place the init_stmt in the inner most
    // for loop.
    inner_fl->body().push_back(init_stmt);
  }

  // Place the allocation
  if (alloc_pos == 0) {
    // If we allocate at the root, look for the provided allocatoin if it
    // exists, and place after it.
    if (alloc_expr != nullptr) {
      bool found = false;
      for (auto it = lowered_exprs.begin(); it != lowered_exprs.end(); it++) {
        if ((*it) == alloc_expr) {
          lowered_exprs.insert(it + 1, init_loop_nest);
          found = true;
          break;
        }
      }
      TORCH_INTERNAL_ASSERT(
          found,
          "Could not figure out where to initialize the buffer for ",
          tv);
    } else {
      lowered_exprs.insert(lowered_exprs.begin(), init_loop_nest);
    }
  } else if (alloc_pos == for_loops.size()) {
    // If we allocate inline, push to the back of the last for loop
    scope_utils::pushBack(for_loops[for_loops.size() - 1], init_loop_nest);
  } else {
    // Otherwise we allocate in some loop nest that is not inline, or root, so
    // insert right before the loop we're just outside of
    scope_utils::insertBefore(
        for_loops[alloc_pos - 1], for_loops[alloc_pos], init_loop_nest);
  }
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
    for (auto out : expr->outputs()) {
      TORCH_INTERNAL_ASSERT(
          out->getValType().value() == ValType::Scalar,
          "Unrecognized output type found in expr ",
          expr,
          " cannot lower ",
          out->getValType().value());

      pushBack(new kir::Allocate(out, MemoryType::Local, new Int(1)));
    }
    pushBack(expr);
    return;
  }

  TensorView* out = expr->output(0)->as<TensorView>();
  // 1) Reduce loop structure
  while (compute_at_scope.size() > out->getThisComputeAtAxis() &&
         compute_at_scope.back().second != out &&
         compute_at_scope.back() !=
             out->getComputeAtAxis((int)compute_at_scope.size() - 1)) {
    popFor();
  }

  // 2) Open back up to computeAt
  while (compute_at_scope.size() < out->getThisComputeAtAxis()) {
    openFor(out->getComputeAtAxis((int)compute_at_scope.size()));
  }

  Expr* alloc_stmt = nullptr;
  //  3) Allocate the output.
  if (!FusionGuard::getCurFusion()->hasInput(out) &&
      !FusionGuard::getCurFusion()->hasOutput(out)) {
    alloc_stmt = pushAlloc(out);
  }

  //  4) If this is a reduction, initialize the output (open for loops to inner
  //  most, predicate, initialize, place next after allocation if exists, close
  //  to computeAt)
  if (out->hasReduction())
    initReduction(out, expr->as<ReductionOp>()->init(), alloc_stmt);

  //  5) Open to inner most loop
  for (decltype(out->nDims()) i = for_loops.size(); i < out->nDims(); i++)
    openFor(out->getComputeAtAxis(i));
  //  6) Run expression
  pushBack(expr);

  // 7) Reduce loop structure back to computeAt
  while (!compute_at_scope.empty() &&
         compute_at_scope.size() > out->getThisComputeAtAxis())
    popFor();
}

namespace {

TensorView* findOutputTensor(Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      expr->outputs().size() <= 1, "Unexpected number of outputs");
  if (expr->outputs().size() != 1) {
    return nullptr;
  }
  auto out = expr->output(0);
  if (out->getValType() != ValType::TensorView) {
    return nullptr;
  }
  return out->as<TensorView>();
}

void findTargetTensor(Expr* expr, TensorView*& target, unsigned& score) {
  TORCH_INTERNAL_ASSERT(expr->outputs().size() <= 1);

  TensorView* out_tv = findOutputTensor(expr);
  if (out_tv == nullptr) {
    target = nullptr;
    score = 0;
    return;
  }

  if (!out_tv->hasComputeAt()) {
    target = out_tv;
    // No computeAt, so this should come last.
    score = std::numeric_limits<unsigned>::max();
    return;
  }

  auto axis = out_tv->getRelativeComputeAtAxis();
  target = out_tv->getComputeAtView();
  std::tie(axis, target) = target->getComputeAtPos(axis);

  score = axis;
}

// Type definitions for brevity
using ExprListT = std::vector<Expr*>;
using TargetGroupMapT = std::unordered_map<TensorView*, ExprListT>;
using ExprTargetMapT = std::unordered_map<Expr*, TensorView*>;
using ScoreT = unsigned;
using ExprScoreMapT = std::unordered_map<const Expr*, ScoreT>;

void sanityCheck(
    const ExprListT& exprs,
    const ExprListT& reordered_exprs,
    const ExprScoreMapT& scores,
    const ExprTargetMapT& target_map,
    const TargetGroupMapT& computed_at_exprs) {
  const auto num_exprs = exprs.size();
  TORCH_INTERNAL_ASSERT(scores.size() == num_exprs);
  TORCH_INTERNAL_ASSERT(
      reordered_exprs.size() + target_map.size() == num_exprs);
  int num_computed_exprs = std::accumulate(
      computed_at_exprs.begin(),
      computed_at_exprs.end(),
      0,
      [](int acc, const std::pair<TensorView*, ExprListT>& p) {
        return acc + p.second.size();
      });
  TORCH_INTERNAL_ASSERT(num_computed_exprs == target_map.size());
}

// Arrange exprs into loop-nest groups. Loop-nest groups are
// disjoint grouping of expressions based on the expression
// where each expression is computed at.
void groupExpressions(
    Expr* expr,
    ExprListT& reordered_exprs,
    ExprTargetMapT& target_map,
    TargetGroupMapT& computed_at_exprs,
    ExprScoreMapT& scores) {
  TensorView* target_tensor = nullptr;
  ScoreT score;
  findTargetTensor(expr, target_tensor, score);
  scores.emplace(expr, score);
  if (target_tensor == nullptr) {
    reordered_exprs.push_back(expr);
  } else {
    target_map.emplace(expr, target_tensor);
    if (computed_at_exprs.find(target_tensor) == computed_at_exprs.end()) {
      computed_at_exprs.emplace(target_tensor, TargetGroupMapT::mapped_type());
    }
    auto& exprs = computed_at_exprs[target_tensor];
    exprs.push_back(expr);
  }
}

// Sort each loop-nest group based on axis (i.e., score)
void sortGroup(TensorView* target, ExprListT& exprs, ExprScoreMapT& scores) {
  std::stable_sort(
      exprs.begin(),
      exprs.end(),
      [&scores](const Expr* expr1, const Expr* expr2) {
        return scores[expr1] < scores[expr2];
      });
}

void mergeNonRootGroupsIntoRootGroups(
    TargetGroupMapT& computed_at_exprs,
    ExprTargetMapT& target_map) {
  for (auto it = computed_at_exprs.begin(); it != computed_at_exprs.end();) {
    TensorView* target = it->first;
    if (target->hasComputeAt()) {
      Expr* target_expr = target->getOrigin();
      TensorView* target_of_target = target_map.at(target_expr);
      auto& target_group = computed_at_exprs.at(target_of_target);
      auto pos =
          std::find(target_group.begin(), target_group.end(), target_expr);
      TORCH_INTERNAL_ASSERT(pos != target_group.end());
      target_group.insert(pos, it->second.begin(), it->second.end());
      // Upate the target map
      for (auto& inserted_expr : it->second) {
        TORCH_INTERNAL_ASSERT(target_map.at(inserted_expr) == target);
        target_map.at(inserted_expr) = target_of_target;
      }
      it = computed_at_exprs.erase(it);
    } else {
      ++it;
    }
  }
}

// Merge root loop-nests into reordered_exprs
void mergeGroupsIntoSortedList(
    TargetGroupMapT& computed_at_exprs,
    ExprListT& reordered_exprs) {
  while (computed_at_exprs.size() > 0) {
    // Find the root loop-nest that has no dependency with the other
    // loop-nests
    TensorView* cur_target = computed_at_exprs.begin()->first;
    for (auto& group : computed_at_exprs) {
      auto target = group.first;
      if (cur_target == target)
        continue;
      if (DependencyCheck::isDependencyOf(target, cur_target)) {
        cur_target = target;
      }
    }
    // cur_target can be visited
    reordered_exprs.insert(
        reordered_exprs.end(),
        computed_at_exprs.at(cur_target).begin(),
        computed_at_exprs.at(cur_target).end());
    computed_at_exprs.erase(cur_target);
  }
}

// Reorder exprs so that LoopNestGenerator::handle(Expr*) can generate
// correct loop nests. Vector exprs is assumed to be topologically
// sorted, but that is not sufficient as tensors computed at
// outer loops need to be located earlier.
void reorderExprsForComputeAt(std::vector<Expr*>& exprs) {
  ExprListT reordered_exprs;
  // expr -> target
  ExprTargetMapT target_map;
  // target -> [computed at expressions]
  TargetGroupMapT computed_at_exprs;
  // score of each expression that is calculated based on the
  // computeAt axis. A lower score of an expression means it should be
  // placed earlier in the expression list. This is a requirement for
  // the loop-nest generation of this class to work.
  ExprScoreMapT scores;

  // 1. Group expressions by target tensors. Non-grouped expressions
  // are copied into reordered_exprs.
  for (auto& expr : exprs) {
    groupExpressions(
        expr, reordered_exprs, target_map, computed_at_exprs, scores);
  }

  sanityCheck(exprs, reordered_exprs, scores, target_map, computed_at_exprs);

  // If no computeAt found, no need to reorder.
  if (computed_at_exprs.size() == 0) {
    return;
  }

  // 2. Sort each loop-nest group based on axis (i.e., score)
  for (auto& group : computed_at_exprs) {
    sortGroup(group.first, group.second, scores);
  }

  // 3. Merge non-root loop-nests into root loop-nests
  mergeNonRootGroupsIntoRootGroups(computed_at_exprs, target_map);

  // At this point, only root loop-nests (i.e., no computeAt'ed)
  // should exist.
  for (auto& group : computed_at_exprs) {
    // Make usre only root loop-nests exist.
    TensorView* target = group.first;
    TORCH_INTERNAL_ASSERT(!target->hasComputeAt());
  }

  sanityCheck(exprs, reordered_exprs, scores, target_map, computed_at_exprs);

  mergeGroupsIntoSortedList(computed_at_exprs, reordered_exprs);

  // Reordering completed. Reordered exprs exist in reordered_exprs.

  TORCH_INTERNAL_ASSERT(exprs.size() == reordered_exprs.size());
  exprs = std::move(reordered_exprs);
}

} // namespace

// Generate the loop nest structure and place it in lowered_exprs
void LoopNestGenerator::generate(const std::vector<Expr*>& exprs) {
  FusionGuard fg(fusion_);

  // Initialize members of the class
  lowered_exprs = std::vector<Expr*>();

  auto reordered = exprs;
  reorderExprsForComputeAt(reordered);

  for (auto* expr : reordered) {
    handle(expr);
  }
}

} // namespace fuser
} // namespace jit
} // namespace torch
