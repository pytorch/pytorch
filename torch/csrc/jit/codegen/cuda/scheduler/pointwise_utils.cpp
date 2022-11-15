#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/pointwise_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace pointwise_utils {

DomainMap::DomainMap(Fusion* fusion) : fusion_(fusion), ca_map_(fusion) {
  view_tvs_ = scheduler_utils::getViewTVs(fusion);
  for (auto select : ir_utils::getSelectOps(fusion)) {
    select_ids_.emplace(select->getSelectAxis());
  }
}

// Determine if all IterDomains in input are mapped to the given tensor
bool DomainMap::areAllInputIdsMappedTo(TensorView* input_tv, TensorView* tv)
    const {
  // Get concrete IDs for input root or rfactor domain
  std::unordered_set<IterDomain*> in_concrete_ids;
  for (auto in_id : input_tv->getMaybeRFactorDomain()) {
    // Permissive map is required for the transpose scheduler to support cases
    // like T0[I0, b] + T1[b, I1]
    auto concrete =
        ca_map_.getConcreteMappedID(in_id, IdMappingMode::PERMISSIVE);
    if (!concrete->isBroadcast() && !in_id->isReduction() &&
        !isSelectId(in_id)) {
      in_concrete_ids.insert(concrete);
    }
  }

  // Erase all input concrete IDs mapped to the output domain
  // Ignore unresolved broadcast dimensions
  for (auto id : tv->getMaybeRFactorDomain()) {
    if (!eraseIfMapped(in_concrete_ids, id)) {
      eraseIfInputMappedThroughViewTo(in_concrete_ids, id);
    }
  }
  return in_concrete_ids.empty();
}

// Erase input concrete ID if it is mapped to output ID
bool DomainMap::eraseIfMapped(
    std::unordered_set<IterDomain*>& in_concrete_ids,
    IterDomain* out_id) const {
  auto out_concrete_id =
      ca_map_.getConcreteMappedID(out_id, IdMappingMode::PERMISSIVE);
  auto in_concrete_id_iter = in_concrete_ids.find(out_concrete_id);
  bool found_match = in_concrete_id_iter != in_concrete_ids.end();
  if (found_match) {
    in_concrete_ids.erase(in_concrete_id_iter);
  }
  return found_match;
}

// Check if in_id is mapped to out_id through any view rfactor domain.
// Currently this function only allow having one view on the path from input to
// output. If there are multiple views, then likely the pointwise scheduler will
// reject the fusion because we can not correctly find a reference tensor.
void DomainMap::eraseIfInputMappedThroughViewTo(
    std::unordered_set<IterDomain*>& in_concrete_ids,
    IterDomain* id) const {
  for (auto view : view_tvs_) {
    // Find any ID in view rfactor domain that is mapped to output ID
    auto view_rfactor_id = anyMapped(view->getRFactorDomain(), id);
    if (view_rfactor_id == nullptr) {
      continue;
    }

    if (view_rfactor_id->isRFactorProduct()) {
      // Check if input ID is mapped to any input IDs of the view rfactor ID
      auto root_inputs = InputsOf::outputs(fusion_, {view_rfactor_id});
      auto filtered_root_ids = ir_utils::filterByType<IterDomain>(root_inputs);
      for (auto view_root_id : filtered_root_ids) {
        eraseIfMapped(in_concrete_ids, view_root_id);
      }
    } else {
      // Otherwise, the input ID must map to the view rfactor ID
      eraseIfMapped(in_concrete_ids, view_rfactor_id);
    }
  }
}

// Find any id in domain that maps with target id
IterDomain* DomainMap::anyMapped(
    const std::vector<IterDomain*>& domain,
    IterDomain* target) const {
  for (auto id : domain) {
    if (ca_map_.areMapped(id, target, IdMappingMode::EXACT)) {
      return id;
    }
  }
  return nullptr;
}

// Determine if output TensorView is a valid reference tensor for this fusion.
// The reference tensor must map to all the iterDomains in each input.
bool DomainMap::isValidReference(TensorView* tv) const {
  for (auto input_tv : ir_utils::filterByType<TensorView>(fusion_->inputs())) {
    if (input_tv->uses().empty()) {
      continue;
    }
    if (!areAllInputIdsMappedTo(input_tv, tv)) {
      return false;
    }
  }
  return true;
}

} // namespace pointwise_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
