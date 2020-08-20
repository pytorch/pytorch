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

  const auto alloc_point = loop_utils::getAllocPoint(tv, for_loops);
  const auto alloc_loop = alloc_point.first;
  const auto alloc_pos = alloc_point.second;

  // Grab the dimensions the allocation will be based on to compute a size
  std::vector<Val*> alloc_dims;
  for (size_t i = alloc_pos; i < tv->nDims(); i++) {
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
    size = new kir::Int(1);
  } else {
    size = kir::lowerValue(alloc_dims[0]);
    for (size_t i = 1; i < alloc_dims.size(); i++) {
      size = kir::mulExpr(size, kir::lowerValue(alloc_dims[i]));
    }
  }

  // Create the allocation node
  const auto lowered_tv = new kir::TensorView(tv);
  const auto alloc =
      new kir::Allocate(lowered_tv, lowered_tv->getMemoryType(), size);

  if (alloc_loop != nullptr) {
    alloc_loop->body().insert(0, alloc);
  } else {
    lowered_exprs.insert(lowered_exprs.begin(), alloc);
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
  auto alloc_point = loop_utils::getAllocPoint(tv, for_loops);
  auto alloc_loop = alloc_point.first;
  auto alloc_pos = alloc_point.second;

  // Grab the IDs that will be involved in the initialization, ignore reduction
  // dimensions. Everything else will be iterated over to cover the entire
  // buffer. Index compute will ignore [block, grid]Dims depending on buffer
  // memory location
  std::vector<kir::IterDomain*> ids;
  for (size_t i = alloc_pos; i < tv->nDims(); i++) {
    IterDomain* dim = tv->getComputeAtAxis(i).first;
    if (dim->isReduction())
      continue;
    ids.push_back(kir::lowerValue(dim)->as<kir::IterDomain>());
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
          new kir::NamedScalar(ss.str(), DataType::Int), id, {}, inner_fl);
    } else {
      // Otherwise it's just a new int-
      new_fl = new kir::ForLoop(new kir::Int(c10::nullopt), id, {}, inner_fl);
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

  // If we don't have an alloc_loop defined it means it needs to go in
  // lowered_exprs Make sure to place after the allocation of what we're
  // initializing if there is one.
  if (alloc_loop == nullptr) {
    if (alloc_expr != nullptr) {
      auto it =
          std::find(lowered_exprs.begin(), lowered_exprs.end(), alloc_expr);
      TORCH_INTERNAL_ASSERT(
          it != lowered_exprs.end(),
          "Could not figure out where to initialize the buffer for ",
          tv);
      lowered_exprs.insert(it + 1, init_loop_nest);
    } else {
      lowered_exprs.insert(lowered_exprs.begin(), init_loop_nest);
    }
  } else {
    if (alloc_expr != nullptr) {
      // If there is an allocation for this tensor view place this loop nest
      // after it
      alloc_loop->body().insert_after(alloc_expr, init_loop_nest);
    } else {
      // Otherwise we're allocating a global value
      alloc_loop->body().insert(0, init_loop_nest);
    }
  }
}

void LoopNestGenerator::handle(Expr* expr) {
  // Check if it's a tensor view expression we need to place in the loop nest
  // structure
  if (!ir_utils::isTVOp(expr)) {
    for (auto out : expr->outputs()) {
      TORCH_INTERNAL_ASSERT(
          out->getValType().value() == ValType::Scalar,
          "Unrecognized output type found in expr ",
          expr,
          " cannot lower ",
          out->getValType().value());

      pushBack(new kir::Allocate(
          kir::lowerValue(out), MemoryType::Local, new kir::Int(1)));
    }
    pushBack(expr);
    return;
  }

  //  0) Apply SyncThreads if any shared memory inputs are modified
  bool shared_memory_sync = false;
  for (auto in : expr->inputs()) {
    shared_memory_sync |= isModifiedSharedMemory(in);
  }
  if (shared_memory_sync) {
    // push Sync to the back of the last for loop
    scope_utils::pushBack(for_loops.back(), new kir::Sync());
    cleanSharedMemory();
  }

  TensorView* out = expr->output(0)->as<TensorView>();

  // Figure out what the entire loop structure should look like.
  std::deque<std::pair<IterDomain*, TensorView*>> loop_structure;

  // As we go through iteration domains track the previous view
  TensorView* last_ca_view = nullptr;
  // Check where in the previous view our last axis was in that view
  int64_t last_ca_view_ind = 0;

  // Look at each axis individually in out's domain
  for (int64_t out_i = 0; out_i < (int64_t)out->getThisComputeAtAxis();
       out_i++) {
    // Grab the axis information
    auto ca_point = out->getComputeAtAxis(out_i);
    auto ca_view = ca_point.second;
    auto ca_id = ca_point.first;

    // Figure out if there are axes in the compute at tensor view that aren't
    // in out, make sure to also open them. Check where to start looking for
    // them in the compute at view.
    size_t start = 0;
    if (last_ca_view == nullptr) {
      // Start at the begining, we haven't processed any axes yet.
      start = 0;
    } else if (last_ca_view == ca_view) {
      // This view is the same as the last axis, so start where we left off.
      start = last_ca_view_ind + 1;
    } else {
      // This is a new view, figure out where we are in it, and start from there
      for (start = 0; start < ca_view->nDims(); start++) {
        if (loop_structure.back().first ==
            ca_view->getComputeAtAxis(start).first) {
          break;
        }
      }
      start++;
    }

    // Go from start, and open all loops in the computeAt view until we hit the
    // one associated with out->getComputeAtAxis(out_i)
    for (size_t ca_i = start; ca_i < ca_view->nDims(); ca_i++) {
      // Note that ca_view->getComputeAtAxis(ca_i) is equivalent to
      // std::pair(ca_view->axis(ca_i), ca_view)
      loop_structure.push_back(ca_view->getComputeAtAxis(ca_i));

      // Update the last view processed
      last_ca_view_ind = ca_i;
      last_ca_view = ca_view;
      if (ca_view->getComputeAtAxis(ca_i).first == ca_id) {
        break;
      }
    }

    // Shouldn't ever hit this, but make sure we hit the break above, meaning we
    // added all necessary axes from the compute at view.
    TORCH_INTERNAL_ASSERT(
        ca_view->getComputeAtAxis(last_ca_view_ind).first == ca_id);
  }

  // We're up to the compute at point in loop_structure, grab the remaining
  // axes.
  for (int64_t out_i = (int64_t)out->getThisComputeAtAxis();
       out_i < (int64_t)out->nDims();
       out_i++) {
    // It's actually local, but getComputeAtAxis returns a std::pair, axis
    // doesn't
    loop_structure.push_back(out->getComputeAtAxis(out_i));
  }

  // At this point loop_structure contains our overal target loop nest structure
  // Lets get a copy of the loop structure, and figure out which loops we need
  // to open.
  decltype(loop_structure) loops_to_open(loop_structure);
  // Pop out loops already opened
  for (const auto& existing_loop : for_loops) {
    if (loops_to_open.empty()) {
      // Nothing to open
      break;
    }
    if (kir::lowerValue(loops_to_open.front().first)->as<kir::IterDomain>() ==
        existing_loop->iter_domain()) {
      loops_to_open.pop_front();
    }
  }

  // At this point for_loops + loops_to_open contains our overal target loop
  // nest structure. Open loops in "loops_to_open".
  while (!loops_to_open.empty()) {
    openFor(loops_to_open.front());
    loops_to_open.pop_front();
  }

  Expr* alloc_expr = nullptr;
  // Place the allocation for out
  if (!FusionGuard::getCurFusion()->hasInput(out) &&
      !FusionGuard::getCurFusion()->hasOutput(out)) {
    alloc_expr = pushAlloc(out);
  }

  //  If this is a reduction, initialize the output (open for loops to inner
  //  most, predicate, initialize, place next after allocation if exists, close
  //  to computeAt)
  if (out->hasReduction())
    initReduction(out, expr->as<ReductionOp>()->init(), alloc_expr);

  //  Place the expression
  pushBack(expr);

  // If output is a shared memory buffer, set modified status
  modifySharedMemory(out);

  // Reduce the loop nest structure back to computeAt
  if (out->getThisComputeAtAxis() == 0) {
    while (!for_loops.empty()) {
      popFor();
    }
  } else {
    auto ca_axis = out->getThisComputeAtAxis() - 1;
    while (for_loops.size() > 0 &&
           for_loops.back()->iter_domain() !=
               kir::lowerValue(out->getComputeAtAxis(ca_axis).first)
                   ->as<kir::IterDomain>()) {
      popFor();
    }
  }
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
  while (target->hasComputeAt()) {
    if (target->getThisComputeAtAxis() < axis)
      break;
    TORCH_INTERNAL_ASSERT(target->getThisComputeAtAxis() == axis);
    axis = target->getComputeAtRelPos(axis);
    target = target->getComputeAtView();
  }

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
  TORCH_INTERNAL_ASSERT(num_computed_exprs == (int)target_map.size());
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

  // Identify all shared memory TensorViews
  // Initialize Modified status
  for (auto v : fusion_->vals()) {
    if (v->getValType().value() == ValType::TensorView) {
      if (v->as<TensorView>()->getMemoryType() == MemoryType::Shared) {
        smem_.insert({v, false});
      }
    }
  }

  // Initialize members of the class
  lowered_exprs = std::vector<Expr*>();

  auto reordered = exprs;
  reorderExprsForComputeAt(reordered);

  for (auto* expr : reordered) {
    handle(expr);
  }
}

void LoopNestGenerator::cleanSharedMemory() {
  for (auto& item : smem_) {
    item.second = false;
  }
}

void LoopNestGenerator::modifySharedMemory(Val* key) {
  auto it = smem_.find(key);
  if (it != smem_.end()) {
    it->second = true;
  }
}

bool LoopNestGenerator::isModifiedSharedMemory(Val* key) const {
  auto it = smem_.find(key);
  if (it != smem_.end()) {
    return it->second;
  }
  return false;
}

} // namespace fuser
} // namespace jit
} // namespace torch
