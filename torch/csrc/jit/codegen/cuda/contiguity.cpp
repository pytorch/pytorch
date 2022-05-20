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
    const std::vector<bool>& root_contiguity)
    : root_domain_(root_domain), root_contiguity_(root_contiguity) {
  if (ids.empty()) {
    return;
  }

  TORCH_INTERNAL_ASSERT(
      root_domain_.size() == root_contiguity_.size(),
      "Arguments don't match ",
      root_domain_.size(),
      " != ",
      root_contiguity_.size());

  TORCH_INTERNAL_ASSERT(
      GpuLower::current() != nullptr, "GpuLower is not found");

  for (const auto i : c10::irange(root_domain_.size())) {
    auto root_domain_i = root_domain_[i]->as<IterDomain>();
    // If a root domain has halo, can't use merged domain even if
    // both inputs are contiguous. HaloInfo is also initialized for
    // rfactor root domains, which should just return "zero"
    // RootAxisInfo. This should be safe as no rfactor tensor should
    // need halo.
    if (root_contiguity_[i] &&
        !GpuLower::current()
             ->haloInfo()
             .getRootAxisInfo(root_domain_i)
             .hasHalo()) {
      contig_ids_.emplace(root_domain_i);
      is_contig_root_[root_domain_i] = true;
      within_contig_ids_[root_domain_i] = std::unordered_set<IterDomain*>();
    } else {
      is_contig_root_[root_domain_i] = false;
    }
    root_to_indexed_id_[root_domain_i] = root_domain_i;
  }

  auto exprs = StmtSort::getExprs(ids[0]->fusion(), {ids.begin(), ids.end()});

  for (auto expr : exprs) {
    handle(expr);
  }
}

void ContigIDs::handle(Merge* merge) {
  // If either input is non-contiguous so is output.
  const auto inner = merge->inner();
  const auto outer = merge->outer();

  if (!isContig(inner) || !isContig(outer)) {
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
            return is_contig_root_.at(id) && !id->isBroadcast() &&
                !id->isReduction();
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
      // This is no longer causing an error in:
      // ReductionSchedulerMultiDimNonFastest TODO: test reenablement to make
      // sure it does what's expected
      //  } else if (
      //     root_copy.front()->isReduction() ||
      //     root_copy.front()->isBroadcast()) {
      //   root_copy.pop_front();
    } else {
      break;
    }
  }

  // If we matched all inputs, the output is contiguous. Only want to keep the
  // top contig ID, lower ids should be placed in the "within_contig_ids" map
  // of top id.
  auto out = merge->out()->as<IterDomain>();
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

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
