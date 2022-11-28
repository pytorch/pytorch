#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <torch/csrc/jit/codegen/cuda/contiguity.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

OrderedIdInformation::OrderedIdInformation(
    const std::vector<IterDomain*>& ids,
    const std::vector<IterDomain*>& root_domain,
    std::shared_ptr<const ConcretizedBroadcastDomains> concrete_info)
    : active_ids_(root_domain), concrete_info_(concrete_info) {
  if (ids.empty() || root_domain.empty()) {
    return;
  }

  // Grab root ids and initialize them.
  for (const auto root_i : c10::irange(root_domain.size())) {
    auto root_id = root_domain[root_i]->as<IterDomain>();

    // Initialize id_to_root_ids to map roots to themselves
    id_to_root_ids_[root_id] = {root_id};

    // Initialize roots as being made up of correctly ordered transforms.
    consistently_ordered_ids_.emplace(root_id);

    exclusively_consumes_roots_.emplace(root_id);
  }

  // Iterate from the root domain to the provided ids and fill
  // consistently_ordered_ids_, id_to_root_ids_, and exclusively_consumes_roots_
  // for all the IDs
  auto exprs = StmtSort::getExprsBetween(
      ids[0]->fusion(),
      {root_domain.begin(), root_domain.end()},
      {ids.begin(), ids.end()});

  for (auto expr : exprs) {
    OptInDispatch::handle(expr);
  }
}

bool OrderedIdInformation::checkExclusivelyConsumesRoots(IterDomain* id) {
  TORCH_INTERNAL_ASSERT(
      std::find(active_ids_.begin(), active_ids_.end(), id) !=
          active_ids_.end(),
      "Error replaying transforms in contiguous ID checker, expected ",
      id->toString(),
      " to be in the active ID set.");

  auto root_id_it = id_to_root_ids_.find(id);
  TORCH_INTERNAL_ASSERT(
      root_id_it != id_to_root_ids_.end(),
      "Error replaying transforms in contiguous ID checker, couldn't find mapped roots of ",
      id->toString());

  const auto& root_ids = root_id_it->second;

  // Check all the roots of all other ids, to see if any root_ids in id are also
  // in them.
  for (auto other_active_id : active_ids_) {
    if (other_active_id == id || other_active_id == nullptr) {
      continue;
    }

    auto root_id_it = id_to_root_ids_.find(other_active_id);
    TORCH_INTERNAL_ASSERT(
        root_id_it != id_to_root_ids_.end(),
        "Error replaying transforms in contiguous ID checker, couldn't find mapped roots of ",
        other_active_id->toString());

    const auto& other_root_ids = root_id_it->second;

    for (auto other_root_id : other_root_ids) {
      if (root_ids.has(other_root_id)) {
        return false;
      }
    }
  }
  return true;
}

void OrderedIdInformation::handle(Merge* merge) {
  // Find inputs in the active_ids_ vector
  const auto inner_it =
      std::find(active_ids_.begin(), active_ids_.end(), merge->inner());
  const auto outer_it =
      std::find(active_ids_.begin(), active_ids_.end(), merge->outer());

  // If either aren't in active_ids_ it means the inputs were detected to not be
  // ordered correctly before hitting this expression.
  if (inner_it == active_ids_.end() || outer_it == active_ids_.end()) {
    return;
  }

  auto inner_pos = std::distance(active_ids_.begin(), inner_it);
  auto outer_pos = std::distance(active_ids_.begin(), outer_it);

  // Find inputs in the ordered transforms map
  const auto inner_ordered_it = consistently_ordered_ids_.find(merge->inner());
  const auto outer_ordered_it = consistently_ordered_ids_.find(merge->outer());

  bool inner_ordered = inner_ordered_it != consistently_ordered_ids_.end();
  bool outer_ordered = outer_ordered_it != consistently_ordered_ids_.end();

  // Get root ids of the two inputs
  const auto inner_root_ids_it = id_to_root_ids_.find(merge->inner());
  const auto outer_root_ids_it = id_to_root_ids_.find(merge->outer());

  TORCH_INTERNAL_ASSERT(
      inner_root_ids_it != id_to_root_ids_.end() &&
          outer_root_ids_it != id_to_root_ids_.end(),
      "Error replaying transforms in contiguous ID checker.");

  const auto& inner_root_ids = inner_root_ids_it->second;
  const auto& outer_root_ids = outer_root_ids_it->second;

  // TODO: Concretization may prevent contiguous indexing or vectorization.
  //  It prevents contiguous indexing if the concretization is within the IDs
  //  that are used for indexing.
  //  For vectorization it just means we need to make sure the extents of the
  //  axes to the right of the broadcast root domain in the contigous merge is
  //  bigger than the vectorization dimension. And that the tensor buffer
  //  supports the vector word size (always done).
  bool outer_is_concretized_bcast = merge->outer()->isBroadcast() &&
      concrete_info_->isConcretized(merge->outer());

  bool inner_is_concretized_bcast = merge->inner()->isBroadcast() &&
      concrete_info_->isConcretized(merge->inner());

  // Update maps
  // Find the position inner would have to have to be considered ordered
  auto pos_after_outer = outer_pos + 1;
  for (; pos_after_outer < active_ids_.size(); pos_after_outer++) {
    if (active_ids_[pos_after_outer] == nullptr) {
      // Can't be considered ordered after a nullptr
      break;
    }
    if (active_ids_[pos_after_outer]->isReduction() ||
        ((active_ids_[pos_after_outer]->isBroadcast() &&
          !concrete_info_->isConcretized(active_ids_[pos_after_outer])))) {
      // Skip reduction or broadcast axes that aren't concretized in the fusion
      continue;
    }
    break;
  }

  // The output is ordered as long as the inputs were ordered and outer position
  // is directly left of the inner position.
  bool out_ordered = inner_ordered && outer_ordered;
  out_ordered = out_ordered &&
      // If inner_pos is before outer_pos it's not ordered correctly. If for
      // some reason it's the same, that would be an error.
      inner_pos > outer_pos &&
      // Inner could be a broadcast, so doesn't have to be right on
      // pos_after_outer as that ID (if it exists) should not be a broadcast.
      // However, merging over a broadcast should be fine.
      inner_pos <= pos_after_outer && !inner_is_concretized_bcast &&
      !outer_is_concretized_bcast;

  if (out_ordered) {
    consistently_ordered_ids_.emplace(merge->out());
  }

  // Don't just remove active_ids_, as if we have something like:
  //   [i0, i1, i2, i3]
  //   ->merge(0, 2)
  //   ->merge(1)
  // The latter merge looks like it's ordered correctly, if we update the active
  // map as:
  //   [i0, i1, i2, i3] -> [i0*i2, i1, i3]
  // Hoever if we instead mark it as:
  //   [i0, i1, i2, i3] -> [i0*i2, i1, nullptr, i3]
  // Or:
  //   [i0, i1, i2, i3] -> [nullptr, i1, i0*i2, i3]
  // It's clear the second merge is not ordered correctly. Doesn't matter which
  // direction we put the iter domain in, prefer putting it in outer as we often
  // are looking for inner dimensions that are contiguous. We don't want to
  // always do this, as it could make ordered merges look non-ordered.
  // For exmaple: [i0, i1, i2, i3]
  // ->merge(0)
  // ->merge(1)
  // ->merge(0)
  // If it's updated as:
  // [i0, i1, i2, i3]
  // -> [i0*i1, nullptr, i2, i3]
  // -> [i0*i1, nullptr, i2*i3, nullptr]
  // Now the final merge looks non-ordered but it is. So only insert a nullptr
  // entry if the out is not ordered.
  active_ids_[outer_pos] = merge->out();

  if (!out_ordered) {
    active_ids_[inner_pos] = nullptr;
  } else {
    active_ids_.erase(active_ids_.begin() + inner_pos);
    for (auto i = outer_pos + 1; i < inner_pos; i++) {
      // If there's broadcast axes between outer and inner and the merge was
      // contiguous, there may be broadcasts between outer and inner that cannot
      // be ordered merged anywhere else so remove them.
      active_ids_.erase(active_ids_.begin() + outer_pos + 1);
    }
  }

  // Update the root_id entry for the output.
  VectorOfUniqueEntries<IterDomain*> root_ids = inner_root_ids;
  root_ids.pushBack(outer_root_ids);

  id_to_root_ids_[merge->out()] = root_ids;

  // Need to check this after updating active_ids_ and id_to_root_ids_
  if (checkExclusivelyConsumesRoots(merge->out())) {
    exclusively_consumes_roots_.emplace(merge->out());
  }
}

void OrderedIdInformation::handle(Split* split) {
  // Find the input in the active_ids_ vector
  const auto in_it =
      std::find(active_ids_.begin(), active_ids_.end(), split->in());

  if (in_it == active_ids_.end()) {
    return;
  }

  auto in_pos = std::distance(active_ids_.begin(), in_it);

  // Find the input in the ordered transforms map
  const auto in_ordered_it = consistently_ordered_ids_.find(split->in());

  bool in_ordered = in_ordered_it != consistently_ordered_ids_.end();

  // Get root ids of the input
  const auto in_root_ids_it = id_to_root_ids_.find(split->in());

  TORCH_INTERNAL_ASSERT(
      in_root_ids_it != id_to_root_ids_.end(),
      "Error replaying transforms in contiguous ID checker.");

  VectorOfUniqueEntries<IterDomain*> in_root_ids = in_root_ids_it->second;

  // Update map for outputs
  // Remove inputs from the active_ids_ and insert the output ID
  active_ids_[in_pos] = split->outer();
  active_ids_.insert(active_ids_.begin() + in_pos + 1, split->inner());

  // The outputs are ordered as long as the input is ordered.
  if (in_ordered) {
    consistently_ordered_ids_.emplace(split->outer());
    consistently_ordered_ids_.emplace(split->inner());
  }

  // Update the root_id entry for the outputs.
  id_to_root_ids_[split->outer()] = in_root_ids;
  id_to_root_ids_[split->inner()] = in_root_ids;
}

// Swizzle generally can't be contiguous because of the non-affine nature of it,
// but we can still analyze the operation in the same way as merge/split.
void OrderedIdInformation::handle(Swizzle2D* swizzle) {
  // Find inputs in the active_ids_ vector
  const auto in_x_it =
      std::find(active_ids_.begin(), active_ids_.end(), swizzle->inX());
  const auto in_y_it =
      std::find(active_ids_.begin(), active_ids_.end(), swizzle->inY());

  if (in_x_it == active_ids_.end() || in_y_it == active_ids_.end()) {
    return;
  }

  auto in_x_pos = std::distance(active_ids_.begin(), in_x_it);
  auto in_y_pos = std::distance(active_ids_.begin(), in_y_it);

  // Find inputs in the ordered transforms map
  const auto in_x_ordered_it = consistently_ordered_ids_.find(swizzle->inX());
  const auto in_y_ordered_it = consistently_ordered_ids_.find(swizzle->inY());

  bool in_x_ordered = in_x_ordered_it != consistently_ordered_ids_.end();
  bool in_y_ordered = in_y_ordered_it != consistently_ordered_ids_.end();

  // Get root ids of the two inputs
  const auto in_x_root_ids_it = id_to_root_ids_.find(swizzle->inX());
  const auto in_y_root_ids_it = id_to_root_ids_.find(swizzle->inY());

  TORCH_INTERNAL_ASSERT(
      in_x_root_ids_it != id_to_root_ids_.end() &&
          in_y_root_ids_it != id_to_root_ids_.end(),
      "Error replaying transforms in contiguous ID checker.");

  const auto& in_x_root_ids = in_x_root_ids_it->second;
  const auto& in_y_root_ids = in_y_root_ids_it->second;

  // Update map for outputs
  // Remove inputs from the active_ids_ and insert the output ID
  active_ids_[in_x_pos] = swizzle->outX();
  active_ids_[in_y_pos] = swizzle->outY();

  // In the case of no real swizzle we can forward properties on each domain
  // independently.
  if (swizzle->swizzleType() == Swizzle2DType::NoSwizzle) {
    if (in_x_ordered) {
      consistently_ordered_ids_.emplace(swizzle->outX());
    }

    if (exclusivelyConsumesRoots(swizzle->inX())) {
      exclusively_consumes_roots_.emplace(swizzle->outX());
    }

    if (in_y_ordered) {
      consistently_ordered_ids_.emplace(swizzle->outY());
    }

    if (exclusivelyConsumesRoots(swizzle->inY())) {
      exclusively_consumes_roots_.emplace(swizzle->outY());
    }

    id_to_root_ids_[swizzle->outX()] = in_x_root_ids;
    id_to_root_ids_[swizzle->outY()] = in_y_root_ids;
  } else {
    VectorOfUniqueEntries<IterDomain*> root_ids = in_x_root_ids;
    root_ids.pushBack(in_y_root_ids);
    id_to_root_ids_[swizzle->outX()] = root_ids;
    id_to_root_ids_[swizzle->outY()] = root_ids;
  }
}

NonDivisibleSplitDependencies::NonDivisibleSplitDependencies(
    // TODO: Revisit reduction rfactor axes and propagation. Should probably use
    // ca_map to propogate non divisibility dependencies across exact map. Still
    // need to think through divisible split and non divisible dependencies to
    // see if there's conflicts where a split might look non divisible but
    // actually is divisible and one's overruling the other.
    const std::vector<IterDomain*>& ids,
    const std::vector<IterDomain*>& root_domain,
    const std::unordered_set<Split*>& divisible_splits) {
  if (ids.empty() || root_domain.empty()) {
    return;
  }
  auto transforms = StmtSort::getExprsBetween(
      ids[0]->fusion(),
      {root_domain.begin(), root_domain.end()},
      {ids.begin(), ids.end()});
  for (auto transform : transforms) {
    auto inp_ids = ir_utils::filterByType<IterDomain>(transform->inputs());
    for (auto inp_id : inp_ids) {
      if (std::find(root_domain.begin(), root_domain.end(), inp_id) !=
          root_domain.end()) {
        // This generally shouldn't happen as there shouldn't be
        // transformations before the root ids, but in case for some reason
        // we eventually do have cases like that, we should reset the
        // root_ids if for some reason they've been placed in the non
        // divisible split set.
        depends_on_non_divisible_split.erase(inp_id);
      }
    }

    bool inputs_non_divisible =
        std::any_of(inp_ids.begin(), inp_ids.end(), [this](IterDomain* inp_id) {
          return depends_on_non_divisible_split.find(inp_id) !=
              depends_on_non_divisible_split.end();
        });

    auto out_ids = ir_utils::filterByType<IterDomain>(transform->outputs());

    if (inputs_non_divisible) {
      // If any inputs are known to be dependent on a divisible split
      // Mark outputs as dependent on a non_divisible split
      depends_on_non_divisible_split.insert(out_ids.begin(), out_ids.end());
      continue;
    }

    if (!transform->isA<Split>()) {
      continue;
    }

    auto split = transform->as<Split>();
    // If this transform is a non-divisible split
    if (divisible_splits.find(split) == divisible_splits.end()) {
      // Mark outputs as dependent on a non_divisible split
      auto out_ids = ir_utils::filterByType<IterDomain>(transform->outputs());
      depends_on_non_divisible_split.insert(out_ids.begin(), out_ids.end());
    }
  }
}

ContigIDs::ContigIDs(
    const std::vector<IterDomain*>& ids,
    const std::vector<IterDomain*>& root_domain,
    const std::vector<bool>& root_contiguity,
    const std::unordered_set<IterDomain*>& final_ids,
    const std::unordered_map<IterDomain*, Val*>& index_map,
    const std::unordered_set<Split*>& divisible_splits,
    std::unordered_map<IterDomain*, IterDomain*> p2c_id_map,
    bool ignore_indexability,
    bool ignore_consistent_ordering)
    : root_domain_(root_domain),
      root_contiguity_(root_contiguity),
      final_ids_(final_ids),
      index_map_(index_map),
      divisible_splits_(divisible_splits),
      p2c_id_map_(std::move(p2c_id_map)),
      ignore_indexability_(ignore_indexability),
      ignore_consistent_ordering_(ignore_consistent_ordering),
      non_divisible_id_info_(ids, root_domain_, divisible_splits_) {
  if (ids.size() > 0) {
    // This constructor doesn't provide the following information so it needs to
    // be built.
    ca_map_ = std::make_shared<ComputeAtMap>(ids[0]->fusion());
    halo_info_ = std::make_shared<HaloInfo>(ids[0]->fusion(), ca_map_);
    concrete_info_ =
        std::make_shared<ConcretizedBroadcastDomains>(ids[0]->fusion());

    consistent_transform_info_ = std::make_unique<const OrderedIdInformation>(
        ids, root_domain, concrete_info_);
  }
  build(ids);
}

ContigIDs::ContigIDs(
    const std::vector<IterDomain*>& ids,
    const std::vector<IterDomain*>& root_domain,
    const std::vector<bool>& root_contiguity,
    const std::unordered_set<IterDomain*>& final_ids,
    const std::unordered_map<IterDomain*, Val*>& index_map,
    const std::unordered_set<Split*>& divisible_splits,
    std::shared_ptr<const ComputeAtMap> ca_map,
    std::shared_ptr<const HaloInfo> halo_info,
    std::shared_ptr<const ConcretizedBroadcastDomains> concrete_info,
    std::unordered_map<IterDomain*, IterDomain*> p2c_id_map,
    bool ignore_indexability,
    bool ignore_consistent_ordering)
    : root_domain_(root_domain),
      root_contiguity_(root_contiguity),
      final_ids_(final_ids),
      index_map_(index_map),
      divisible_splits_(divisible_splits),
      ca_map_(ca_map),
      halo_info_(halo_info),
      concrete_info_(concrete_info),
      p2c_id_map_(std::move(p2c_id_map)),
      ignore_indexability_(ignore_indexability),
      ignore_consistent_ordering_(ignore_consistent_ordering),
      consistent_transform_info_(std::make_unique<const OrderedIdInformation>(
          ids,
          root_domain,
          concrete_info_)),
      non_divisible_id_info_(ids, root_domain, divisible_splits_) {
  build(ids);
}

ContigIDs ContigIDs::getNonContigIDs() {
  return ContigIDs({}, {}, {}, {}, {}, {});
}

void ContigIDs::build(const std::vector<IterDomain*>& ids) {
  if (ids.empty() || root_domain_.empty()) {
    return;
  }

  TORCH_INTERNAL_ASSERT(
      root_domain_.size() == root_contiguity_.size(),
      "Arguments don't match ",
      root_domain_.size(),
      " != ",
      root_contiguity_.size());

  for (const auto root_domain_i : c10::irange(root_domain_.size())) {
    auto root_domain_id = root_domain_[root_domain_i]->as<IterDomain>();
    root_to_indexed_id_[root_domain_id] = root_domain_id;
    // Initialize to false
    is_contig_root_[root_domain_id] = false;
    // If a root domain has halo, can't use merged domain even if
    // both inputs are contiguous. HaloInfo is also initialized for
    // rfactor root domains, which should just return "zero"
    // RootAxisInfo. This should be safe as no rfactor tensor should
    // need halo.
    if (root_contiguity_[root_domain_i] &&
        !halo_info_->getRootAxisInfo(root_domain_id).hasHalo()) {
      contig_ids_.emplace(root_domain_id);
      is_contig_root_[root_domain_id] = true;
      within_contig_ids_[root_domain_id] = std::unordered_set<IterDomain*>();
    }
  }

  if (!contig_ids_.empty()) {
    auto exprs = StmtSort::getExprsBetween(
        ids[0]->fusion(),
        {root_domain_.begin(), root_domain_.end()},
        {ids.begin(), ids.end()});
    for (auto expr : exprs) {
      handle(expr);
    }
  }
}

void ContigIDs::handle(Merge* merge) {
  // If output is not consistently ordered or doesn't solely consume all root
  // domains in its dependencies, then it can't be a contiguously indexable
  // iterdomain.
  if (!(ignore_consistent_ordering_ ||
        consistent_transform_info_->isConsistentlyOrdered(merge->out()))) {
    return;
  }

  if (!consistent_transform_info_->exclusivelyConsumesRoots(merge->out())) {
    return;
  }

  // If output is not "directly indexable" then it's definitely not contiguously
  // indexable.
  if (!ignore_indexability_ && !isIndexable(merge->out())) {
    return;
  }

  // If inputs are marked as final, stop
  if (final_ids_.count(merge->inner()) || final_ids_.count(merge->outer())) {
    return;
  }

  // Check root domains for contiguity
  auto root_ids_it =
      consistent_transform_info_->idToRootIds().find(merge->out());

  TORCH_INTERNAL_ASSERT(
      root_ids_it != consistent_transform_info_->idToRootIds().end(),
      "\nError in contiguous analysis, merge info doesn't exist for:\n",
      merge->toString(),
      "\nId: ",
      merge->out()->toString());

  VectorOfUniqueEntries<IterDomain*> root_ids = root_ids_it->second;

  bool is_indexing_pass = !ignore_consistent_ordering_;

  IterDomain* last_root = nullptr;
  for (auto root_id_i : c10::irange(root_domain_.size())) {
    auto root_id = root_domain_[root_id_i];
    if (root_ids.has(root_id)) {
      // ID found, remove it
      root_ids.erase(root_id);
      // If we're indexing:
      // we could still potentially consider this ID linearly indexable, as we
      // could multiple the index by the last root's stride.
      //
      // If we're computing predicates (ignore_consistent_ordering_==true),
      // then we don't have this same constraint, we can just ignore
      // contiguity of the roots all together.
      if (!root_contiguity_[root_id_i] && is_indexing_pass) {
        if (!root_ids.empty()) {
          return;
        }
      }
      last_root = root_id;
    }
  }

  // If there's a non_divisible split in the history of merge->out then it can't
  // be contiguously indexable.
  if (non_divisible_id_info_.dependsOnNonDivisibleSplit(merge->out())) {
    return;
  }

  // Now we know merge->out is a contiguously indexable ID

  TORCH_INTERNAL_ASSERT(
      last_root != nullptr,
      "Issue processing root ids for ",
      merge->out()->toString());

  // Reset root_ids
  root_ids = root_ids_it->second;
  for (auto root_id : root_ids) {
    root_to_indexed_id_[root_id] = merge->out();
  }

  auto all_within_vals = DependencyCheck::getAllValsBetween(
      {root_domain_.begin(), root_domain_.end()}, {merge->out()});
  auto all_within_ids = ir_utils::filterByType<IterDomain>(all_within_vals);

  std::unordered_set<IterDomain*> within_id_set(
      all_within_ids.begin(), all_within_ids.end());

  within_id_set.erase(merge->out());
  within_contig_ids_[merge->out()] = within_id_set;
  for (auto id : all_within_ids) {
    contig_ids_.erase(id);
  }

  contig_ids_.emplace(merge->out());
}

IterDomain* ContigIDs::getMappedId(IterDomain* id) const {
  auto it = p2c_id_map_.find(id);
  if (it != p2c_id_map_.end()) {
    return it->second;
  } else {
    return id;
  }
}

bool ContigIDs::isIndexable(IterDomain* id) const {
  // If ID is mapped to consumer through persmissive map but not exact map it
  // will not be mapped through to the exact map through the p2c map. Therefore
  // reject because it involves broadcast resolution.
  if (!ca_map_->idExistsInMap(getMappedId(id))) {
    return false;
  }
  auto c_id =
      ca_map_->getConcreteMappedID(getMappedId(id), IdMappingMode::EXACT);
  return index_map_.find(c_id) != index_map_.end();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
