#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {

Val* threadPredicate(ParallelType pt) {
  return eq(new NamedScalar(stringifyThread(pt), DataType::Int), new Int(0));
}

Bool* getThreadPredicate(const ir_utils::ParallelTypeBitmap& bits) {
  if (bits.none()) {
    return new Bool(true);
  }

  Val* pred = nullptr;

  for (const auto& pt_bool : bits.getMap()) {
    if (pt_bool.second) {
      if (pred == nullptr) {
        pred = threadPredicate(pt_bool.first);
      } else {
        pred = andOp(pred, threadPredicate(pt_bool.first));
      }
    }
  }

  // Should never be hit.
  TORCH_INTERNAL_ASSERT(pred != nullptr);

  TORCH_INTERNAL_ASSERT(
      pred->getDataType().value() == DataType::Bool,
      "Tried to return a predicate that is not a bool val.");

  return pred->as<Bool>();
}

} // namespace

// Update the reduction_deps bitset based on provided Expr
void ThreadPredicateMap::updateBitSet(Expr* expr) {
  // Which predicates were set for the inputs
  ir_utils::ParallelTypeBitmap input_preds;

  // Which dims are reductions in inputs
  ir_utils::ParallelTypeBitmap input_reductions;

  // Which dims are bcast in inputs
  ir_utils::ParallelTypeBitmap input_bcasts;

  // Run through inputs and update bitsets
  for (const auto* inp : expr->inputs()) {
    if (!ir_utils::isTV(inp))
      continue;

    auto tv_inp = ir_utils::asConstTV(inp);
    TORCH_INTERNAL_ASSERT(
        thread_predicates_.find(tv_inp) != thread_predicates_.end(),
        "Thread predicate map was not initialized, couldn't find ",
        inp);

    input_preds |= thread_predicates_[tv_inp];

    ir_utils::ParallelTypeBitmap id_reductions;
    ir_utils::ParallelTypeBitmap id_bcasts;
    ir_utils::ParallelTypeBitmap id_ptypes;

    for (auto id : tv_inp->domain()->domain()) {
      if (id->isThread()) {
        id_ptypes.set(id->parallel_method(), true);
        if (id->isReduction())
          id_reductions.set(id->parallel_method(), true);
        if (id->isBroadcast())
          id_bcasts.set(id->parallel_method(), true);
      }
    }

    // Validate the combination of ptypes, reductions, bcasts
    for (size_t i = 0; i < ir_utils::ParallelTypeBitmap::num_p_type; i++) {
      if (input_reductions[i]) {
        if (id_ptypes[i]) {
          TORCH_INTERNAL_ASSERT(
              id_reductions[i],
              "Mismatched parallelized reductions found on inputs of epxr: ",
              expr);
          TORCH_CHECK(
              !id_bcasts[i],
              "Invalid broadcast and reduction combination, tried to parallelize both with the same thread dim: ",
              inp);
        }
      }
    }

    // Accumulate
    input_reductions |= id_reductions;
    input_bcasts |= id_bcasts;
  }

  // Update map for this tv, before accumulating to other inputs
  // Add any reductions this id has to any input predicates
  auto output_preds = input_preds | input_reductions;

  // Figure out which dims bcast wants to reset
  auto bcast_reset_map = output_preds & input_bcasts;

  // Flip it to make a bit mask
  bcast_reset_map = ~bcast_reset_map;

  // Get rid of any reductions which are bcasted
  output_preds &= bcast_reset_map;

  // Run through outputs and set bitset predicates
  for (const auto* out : expr->outputs()) {
    if (!ir_utils::isTV(out))
      continue;
    thread_predicates_[ir_utils::asConstTV(out)] = output_preds;
  }
}

ThreadPredicateMap::ThreadPredicateMap(Fusion* _fusion) : fusion_(_fusion) {
  for (auto inp : fusion_->inputs()) {
    if (ir_utils::isTV(inp)) {
      thread_predicates_[ir_utils::asConstTV(inp)] =
          ir_utils::ParallelTypeBitmap();
    }
  }
  for (auto expr : fusion_->exprs(true)) {
    updateBitSet(expr);
  }
}

ThreadPredicateMap::const_iterator ThreadPredicateMap::find(
    const TensorView* tv) const {
  return thread_predicates_.find(tv);
}

ThreadPredicateMap::const_iterator ThreadPredicateMap::end() const {
  return thread_predicates_.end();
}

const ir_utils::ParallelTypeBitmap& ThreadPredicateMap::at(
    const TensorView* tv) const {
  return thread_predicates_.at(tv);
}

ir_utils::ParallelTypeBitmap& ThreadPredicateMap::at(const TensorView* tv) {
  return thread_predicates_.at(tv);
}

ir_utils::ParallelTypeBitmap& ThreadPredicateMap::operator[](
    const TensorView* tv) {
  return thread_predicates_[tv];
}

Bool* ThreadPredicateMap::getExpr(const TensorView* tv) const {
  TORCH_INTERNAL_ASSERT(find(tv) != end(), "Couldn't find ", tv);
  return getThreadPredicate(at(tv));
}

} // namespace fuser
} // namespace jit
} // namespace torch
