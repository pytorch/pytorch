#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_reductions.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

bool isDerivedFromTrivialReduction(TensorView* tv, IterDomain* id);

bool traverseToRFactorTensor(TensorView* tv, IterDomain* root_id) {
  TORCH_INTERNAL_ASSERT(
      root_id->definition() == nullptr, "Not root IterDomain: ", root_id);

  if (tv->definition() == nullptr) {
    // This is an input tensor, so no rafactor tensor to traverse.
    return false;
  }

  const auto& inputs = tv->definition()->inputs();

  if (inputs.size() != 1 || !inputs[0]->isA<TensorView>() ||
      tv->definition()->getExprType() != ExprType::ReductionOp) {
    // No rfactor producer found
    return false;
  }

  auto producer = inputs[0]->as<TensorView>();

  if (!producer->hasRFactor()) {
    return false;
  }

  auto c2p = PairwiseRootDomainMap(producer, tv)
                 .mapConsumerToProducer(tv->domain(), producer->domain());

  auto producer_id_it = c2p.find(root_id);
  if (producer_id_it == c2p.end()) {
    // No matching producer is found. Stop traversing.
    return false;
  }

  auto producer_root_id = producer_id_it->second;

  return isDerivedFromTrivialReduction(producer, producer_root_id);
}

bool isDerivedFromTrivialReduction(TensorView* tv, IterDomain* id) {
  auto id_inputs = InputsOf::output(id->fusion(), id);
  for (auto root_id : ir_utils::filterByType<IterDomain>(id_inputs)) {
    if (root_id->isReduction() && root_id->rawExtent()->isOneInt()) {
      continue;
    }
    // If not possible to prove the root ID is trivial, see if the ID
    // is derived from a rfactor tensor and, if so, continue the
    // analysis at the rfactor tensor.
    if (!traverseToRFactorTensor(tv, root_id)) {
      return false;
    }
  }
  return true;
}

} // namespace

std::unordered_set<IterDomain*> detectTrivialReductionDerivedDomains(
    Fusion* fusion) {
  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, fusion->outputs());

  std::unordered_set<IterDomain*> trivial_reductions;

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (auto id : tv->domain()->domain()) {
      if (isDerivedFromTrivialReduction(tv, id)) {
        // If id is a trivial reduction, all of its ancestor vals are
        // also trivial reductions.
        for (auto dep_id : DependencyCheck::getAllValsBetween(
                 std::unordered_set<Val*>(
                     tv->getRootDomain().begin(), tv->getRootDomain().end()),
                 {id})) {
          trivial_reductions.insert(dep_id->as<IterDomain>());
        }
      }
    }
  }

  return trivial_reductions;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
