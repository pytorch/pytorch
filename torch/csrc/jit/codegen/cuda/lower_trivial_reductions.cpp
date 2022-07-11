#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_reductions.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

bool analyzeIfDerivedFromTrivialReduction(TensorView* tv, IterDomain* id);

// Checks the producer of tv to see if the
bool traverseToRFactorTensor(TensorView* tv, IterDomain* root_id) {
  TORCH_INTERNAL_ASSERT(
      root_id->definition() == nullptr, "Not root IterDomain: ", root_id);

  if (tv->definition() == nullptr) {
    // This is an input tensor, so no rfactor tensor to traverse.
    return false;
  }

  const auto& inputs = tv->definition()->inputs();

  // Check the reduction expression that produces tv
  if (inputs.size() != 1 || !inputs[0]->isA<TensorView>() ||
      (tv->definition()->getExprType() != ExprType::ReductionOp &&
       tv->definition()->getExprType() != ExprType::WelfordOp)) {
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

  return analyzeIfDerivedFromTrivialReduction(producer, producer_root_id);
}

bool analyzeIfDerivedFromTrivialReduction(TensorView* tv, IterDomain* id) {
  auto id_inputs = InputsOf::output(id->fusion(), id);
  for (auto root_id : ir_utils::filterByType<IterDomain>(id_inputs)) {
    if (root_id->isReduction() && root_id->extent()->isOneInt()) {
      continue;
    }
    // If not possible to prove the root ID is trivial, see if the ID
    // is derived from a rfactor tensor. This may mean that the iteration domain
    // was merged or split in another expression through rfactor. Trace back
    // through rfactor expressions to find original roots and determine there if
    // trivial.
    if (!traverseToRFactorTensor(tv, root_id)) {
      return false;
    }
  }
  return true;
}

} // namespace

void TrivialReductionInfo::build(Fusion* fusion) {
  auto used_vals = fusion->usedMathVals();

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (auto id : tv->domain()->domain()) {
      if (analyzeIfDerivedFromTrivialReduction(tv, id)) {
        // If id is a trivial reduction, all of its ancestor vals are
        // also trivial reductions.
        for (auto dep_id : DependencyCheck::getAllValsBetween(
                 std::unordered_set<Val*>(
                     tv->getRootDomain().begin(), tv->getRootDomain().end()),
                 {id})) {
          domains_.insert(dep_id->as<IterDomain>());
          domains_derived_from_root_.insert(dep_id->as<IterDomain>());
        }
      } else if (id->isReduction() && id->extent()->isOneInt()) {
        // This happens when a leaf domain is trivial but its root
        // axes are not. For example, consider a non-trivial domain
        // split by one. The inner output axis is a trivial domain,
        // whereas the outer output axis is not. Since the root axis
        // is not trivial, a for-loop needs to be generated.
        domains_.insert(id);
      }
    }
  }
}

bool TrivialReductionInfo::isDerived(IterDomain* id) const {
  return domains_.find(id) != domains_.end();
}

bool TrivialReductionInfo::isDerivedFromRoot(IterDomain* id) const {
  return domains_derived_from_root_.find(id) !=
      domains_derived_from_root_.end();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
