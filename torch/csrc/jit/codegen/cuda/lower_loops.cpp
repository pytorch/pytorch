#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

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
        tv->getComputeAtAxis(alloc_pos).first->parallel_method() ==
            ParallelType::Unroll) {
      break;
    }
    alloc_pos++;
  }

  // Grab the dimensions the allocation will be based on
  std::vector<Val*> alloc_dims;
  for (auto i = alloc_pos; i < tv->nDims(); i++) {
    IterDomain* dim = tv->getComputeAtAxis(i).first;
    if (
        // If shared memory, don't use any IDs bound to a grid dimension
        (tv->memory_type_ == MemoryType::Shared && dim->isBlockDim()) ||
        // If local memory, don't use any IDs bound to a grid or block dimension
        (tv->memory_type_ == MemoryType::Local && dim->isThread()) ||
        // If we're reducing this dimension, don't use it in the allocation
        // computation
        dim->isReduction() ||
        // If this is a broadcast dimension, don't use it in the allocation
        // computation
        dim->isBroadcast())
      continue;
    alloc_dims.push_back(dim->extent());
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
  Allocate* alloc = new Allocate(tv, size);

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
        tv->getComputeAtAxis(alloc_pos).first->parallel_method() ==
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
  if (thread_predicates_.find(tv) != thread_predicates_.end()) {
    thread_predicates_[clone] = thread_predicates_[tv];
  }
  // The initilization stmt that will be located inside the loop nest (if there
  // is one)
  auto init_stmt = new UnaryOp(UnaryOpType::Set, clone, init_val);

  // Init a pointer that will become the entirety of the initialization
  Expr* init_loop_nest = nullptr;

  // The for loop that we will place the initialization within (alloc_pos - 1),
  // if one exists. Once we're done this inner_fl will be the inner most loop
  // containing the init_stmt
  ForLoop* inner_fl = nullptr;
  if (alloc_pos >= 1)
    inner_fl = for_loops[alloc_pos - 1];

  // Work through the iter domains that we need to initialize on, outside to
  // inside, to construct the loop nest for the initialization.
  for (auto id : ids) {
    ForLoop* new_fl;

    if (id->isThread()) {
      // If based on a thread, make sure we get the named Int right
      std::stringstream ss;
      ss << id->parallel_method();
      new_fl = new ForLoop(
          new NamedScalar(ss.str(), DataType::Int), id, {}, inner_fl);
    } else {
      // Otherwise it's just a new int-
      new_fl = new ForLoop(new Int(), id, {}, inner_fl);
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
    for (auto out : expr->outputs())
      pushBack(new Allocate(out, new Int(1)));
    pushBack(expr);
    return;
  }

  TensorView* out = static_cast<TensorView*>(expr->output(0));
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
      !FusionGuard::getCurFusion()->hasOutput(out))
    alloc_stmt = pushAlloc(out);

  //  4) If this is a reduction, initialize the output (open for loops to inner
  //  most, predicate, initialize, place next after allocation if exists, close
  //  to computeAt)
  if (out->hasReduction())
    initReduction(out, static_cast<ReductionOp*>(expr)->init(), alloc_stmt);

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
