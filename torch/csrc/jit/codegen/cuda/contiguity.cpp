#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <torch/csrc/jit/codegen/cuda/contiguity.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

ContigIDs::ContigIDs(
    const std::vector<IterDomain*>& ids,
    const std::vector<IterDomain*>& root_domain,
    const std::vector<bool>& root_contiguity,
    std::unordered_map<IterDomain*, IterDomain*> concrete_to_ref,
    std::unordered_map<IterDomain*, IterDomain*> p2c_id_map,
    bool ignore_halo_constraint,
    bool ignore_indexability)
    : root_domain_(root_domain),
      root_contiguity_(root_contiguity),
      concrete_to_ref_(std::move(concrete_to_ref)),
      p2c_id_map_(std::move(p2c_id_map)),
      ignore_indexability_(ignore_indexability) {
  if (ids.empty()) {
    return;
  }

  TORCH_INTERNAL_ASSERT(
      root_domain_.size() == root_contiguity_.size(),
      "Arguments don't match ",
      root_domain_.size(),
      " != ",
      root_contiguity_.size());

  // GpuLower is required to honor halo constraints
  if (!ignore_halo_constraint) {
    TORCH_INTERNAL_ASSERT(GpuLower::hasCurrent(), "GpuLower not found");
  }

  for (const auto i : c10::irange(root_domain_.size())) {
    auto root_domain_i = root_domain_[i]->as<IterDomain>();
    root_to_indexed_id_[root_domain_i] = root_domain_i;
    // Initialize to false
    is_contig_root_[root_domain_i] = false;
    // If a root domain has halo, can't use merged domain even if
    // both inputs are contiguous. HaloInfo is also initialized for
    // rfactor root domains, which should just return "zero"
    // RootAxisInfo. This should be safe as no rfactor tensor should
    // need halo.
    if (root_contiguity_[i] &&
        (ignore_halo_constraint ||
         !GpuLower::current()
              ->haloInfo()
              .getRootAxisInfo(root_domain_i)
              .hasHalo())) {
      contig_ids_.emplace(root_domain_i);
      is_contig_root_[root_domain_i] = true;
      within_contig_ids_[root_domain_i] = std::unordered_set<IterDomain*>();
    }
  }

  if (!contig_ids_.empty()) {
    auto exprs = StmtSort::getExprs(ids[0]->fusion(), {ids.begin(), ids.end()});
    for (auto expr : exprs) {
      handle(expr);
    }
  }
}

void ContigIDs::handle(Merge* merge) {
  // If either input is non-contiguous so is output.
  const auto inner = merge->inner();
  const auto outer = merge->outer();
  const auto out = merge->out();

  if (!isContig(inner) || !isContig(outer)) {
    return;
  }

  // Stop contig merging if the merge output is not indexable.
  if (!ignore_indexability_ && !isIndexable(out)) {
    return;
  }

  // Grab inputs, make sure they're in root domain, check if they're
  // contiguous.

  auto lhs_inputs =
      ir_utils::iterDomainInputsOfOrderedAs({outer}, root_domain_);
  auto rhs_inputs =
      ir_utils::iterDomainInputsOfOrderedAs({inner}, root_domain_);

  TORCH_INTERNAL_ASSERT(
      inRoot(lhs_inputs) && inRoot(rhs_inputs),
      "Found an invalid merge operation, inputs of its arguments are not in the root domain.");

  std::deque<IterDomain*> ordered_inputs(lhs_inputs.begin(), lhs_inputs.end());
  ordered_inputs.insert(
      ordered_inputs.end(), rhs_inputs.begin(), rhs_inputs.end());

  // If any root input is not contig, output is not contig
  if (!(std::all_of(
          ordered_inputs.begin(), ordered_inputs.end(), [this](IterDomain* id) {
            // Allow reduction tensors in contiguity check since we're using
            // this to check contiguous vectors of reference tensors in
            // schedulers (to set vectorization sizes), those reference tensors
            // may have reduction dims, don't bail on contiguity just because
            // it's a reduction dimension.
            return is_contig_root_.at(id);
          }))) {
    return;
  }

  std::deque<IterDomain*> root_copy(root_domain_.begin(), root_domain_.end());

  // Forward to first matching argument
  while (!root_copy.empty() && !ordered_inputs.empty()) {
    if (root_copy.front() != ordered_inputs.front()) {
      root_copy.pop_front();
    } else {
      break;
    }
  }

  // Forward through all matching arguments
  while (!root_copy.empty() && !ordered_inputs.empty()) {
    if (root_copy.front() == ordered_inputs.front()) {
      root_copy.pop_front();
      ordered_inputs.pop_front();
    } else if (
        root_copy.front()->isReduction() || root_copy.front()->isBroadcast()) {
      // This was a cause of an error with
      // ReductionSchedulerMultiDimNonFastest. The test no longer
      // fails.
      root_copy.pop_front();
    } else {
      break;
    }
  }

  // If we matched all inputs, the output is contiguous. Only want to keep the
  // top contig ID, lower ids should be placed in the "within_contig_ids" map
  // of top id.
  if (ordered_inputs.empty()) {
    if (contig_ids_.find(inner) != contig_ids_.end()) {
      contig_ids_.erase(inner);
    }

    if (contig_ids_.find(outer) != contig_ids_.end()) {
      contig_ids_.erase(outer);
    }

    contig_ids_.emplace(out);

    std::unordered_set<IterDomain*> within_out;
    within_out.emplace(inner);
    if (within_contig_ids_.find(inner) != within_contig_ids_.end()) {
      auto in_inner = within_contig_ids_.at(inner);
      within_out.insert(in_inner.begin(), in_inner.end());
      within_contig_ids_.erase(inner);
    }

    within_out.emplace(outer);
    if (within_contig_ids_.find(outer) != within_contig_ids_.end()) {
      auto in_outer = within_contig_ids_.at(outer);
      within_out.insert(in_outer.begin(), in_outer.end());
      within_contig_ids_.erase(outer);
    }

    within_contig_ids_[out] = within_out;

    for (auto root : lhs_inputs) {
      root_to_indexed_id_[root] = out;
    }
    for (auto root : rhs_inputs) {
      root_to_indexed_id_[root] = out;
    }
  }
}

IterDomain* ContigIDs::getMappedId(IterDomain* id) const {
  auto it = p2c_id_map_.find(id);
  if (it != p2c_id_map_.end()) {
    return it->second;
  } else {
    return id;
  }
}

IterDomain* ContigIDs::getCAIndexConcreteId(IterDomain* id) const {
  TORCH_INTERNAL_ASSERT(
      GpuLower::current() != nullptr, "GpuLower is not found");

  auto c_id = GpuLower::current()->caMap()->getConcreteMappedID(
      getMappedId(id), IdMappingMode::EXACT);
  return c_id;
}

bool ContigIDs::isIndexable(IterDomain* id) const {
  // If ID is mapped to consumer through persmissive map but not exact map it
  // will not be mapped through to the exact map through the p2c map. Therefore
  // reject because it involves broadcast resolution.
  if (!GpuLower::current()->caMap()->idExistsInMap(getMappedId(id))) {
    return false;
  }
  auto c_id = getCAIndexConcreteId(id);
  return concrete_to_ref_.find(c_id) != concrete_to_ref_.end();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
