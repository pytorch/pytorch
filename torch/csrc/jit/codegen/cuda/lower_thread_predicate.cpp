#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>

namespace torch {
namespace jit {
namespace fuser {

const static std::unordered_map<ParallelType, int, TypeHash> pt_to_offset{
    {ParallelType::BIDx, 0},
    {ParallelType::BIDy, 1},
    {ParallelType::BIDz, 2},
    {ParallelType::TIDx, 3},
    {ParallelType::TIDy, 4},
    {ParallelType::TIDz, 5}};

const static std::unordered_map<int, ParallelType> offset_to_pt{
    {0, ParallelType::BIDx},
    {1, ParallelType::BIDy},
    {2, ParallelType::BIDz},
    {3, ParallelType::TIDx},
    {4, ParallelType::TIDy},
    {5, ParallelType::TIDz}};

static constexpr int num_p_type = 6;

namespace {

void flip_true(std::bitset<num_p_type>& bits, const ParallelType p_type) {
  if (pt_to_offset.find(p_type) == pt_to_offset.end()) {
    TORCH_INTERNAL_ASSERT(false, "Could not recognize parallel type.");
  }
  bits[pt_to_offset.at(p_type)] = true;
}

Val* threadPredicate(int i) {
  if (offset_to_pt.find(i) == offset_to_pt.end()) {
    TORCH_INTERNAL_ASSERT(
        false,
        "Invalid int for predicate computation, should be from [0-5], but recieved, ",
        i,
        ".");
  }
  return eq(
      new NamedScalar(stringifyThread(offset_to_pt.at(i)), DataType::Int),
      new Int(0));
}

Bool* getThreadPredicate(std::bitset<num_p_type> bits) {
  if (bits.none())
    return new Bool(true);

  Val* pred = nullptr;

  for (int i = 0; i < num_p_type; i++) {
    if (bits[i]) {
      if (pred == nullptr) {
        pred = threadPredicate(i);
      } else {
        pred = andOp(pred, threadPredicate(i));
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

std::bitset<num_p_type> ThreadPredicates::getThreadPredicates(
    const TensorView* tv) {
  TORCH_INTERNAL_ASSERT(
      thread_predicates.find(tv) != thread_predicates.end(),
      "Invalid predicate initialization, couldn't find ",
      tv);
  return thread_predicates[tv];
}

// Update the reduction_deps bitset based on provided Expr
void ThreadPredicates::updateBitSet(Expr* expr) {
  // Which predicates were set for the inputs
  std::bitset<num_p_type> input_preds;

  // Which dims are reductions in inputs
  std::bitset<num_p_type> input_reductions;

  // Which dims are bcast in inputs
  std::bitset<num_p_type> input_bcasts;

  // Run through inputs and update bitsets
  for (const auto* inp : expr->inputs()) {
    if (!ir_utils::isTV(inp))
      continue;

    auto tv_inp = ir_utils::asConstTV(inp);
    TORCH_INTERNAL_ASSERT(
        thread_predicates.find(tv_inp) != thread_predicates.end(),
        "Thread predicate map was not initialized, couldn't find ",
        inp);

    input_preds |= thread_predicates[tv_inp];

    std::bitset<num_p_type> id_reductions;
    std::bitset<num_p_type> id_bcasts;
    std::bitset<num_p_type> id_ptypes;

    for (auto id : tv_inp->domain()->domain()) {
      if (id->isThread()) {
        flip_true(id_ptypes, id->parallel_method());
        if (id->isReduction())
          flip_true(id_reductions, id->parallel_method());
        if (id->isBroadcast())
          flip_true(id_bcasts, id->parallel_method());
      }
    }

    // Validate the combination of ptypes, reductions, bcasts
    for (size_t i = 0; i < num_p_type; i++) {
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
    thread_predicates[ir_utils::asConstTV(out)] = output_preds;
  }
}
ThreadPredicates::ThreadPredicates(Fusion* _fusion) : fusion_(_fusion) {
  for (auto inp : fusion_->inputs())
    if (ir_utils::isTV(inp))
      thread_predicates[ir_utils::asConstTV(inp)] = std::bitset<num_p_type>();
}

std::unordered_map<const TensorView*, Bool*> ThreadPredicates::compute(
    Fusion* fusion) {
  ThreadPredicates tp(fusion);
  for (auto expr : fusion->exprs(true))
    tp.updateBitSet(expr);
  std::unordered_map<const TensorView*, Bool*> preds;
  for (auto entry : tp.thread_predicates) {
    preds[entry.first] = getThreadPredicate(entry.second);
  }
  return preds;
}

} // namespace fuser
} // namespace jit
} // namespace torch
