#include <scheduler/vectorize_helper.h>

#include <compute_at_map.h>
#include <contiguity.h>
#include <expr_evaluator.h>
#include <ir_builder.h>
#include <iter_visitor.h>
#include <lower_divisible_split.h>
#include <scheduler/registry.h>

#include <c10/util/irange.h>

#include <unordered_set>

namespace nvfuser {
namespace vectorize_helper {

namespace factorization_helpers {

std::multiset<int64_t> computeFactors(int64_t i) {
  TORCH_INTERNAL_ASSERT(
      i > 0, "Only support values for factorization >0 but received, ", i);
  std::multiset<int64_t> factors;
  // Take out multiples of two first as that's easy
  while (i % 2 == 0) {
    factors.emplace(2);
    i /= 2;
  }

  // Brute force factorization:
  int64_t factor = 3;
  while (factor * factor <= i) {
    if (i % factor == 0) {
      factors.emplace(factor);
      i /= factor;
    } else {
      factor += 2;
    }
  }

  if (i != 1) {
    factors.emplace(i);
  }
  return factors;
}

std::multiset<int64_t> getAllFactors(const std::multiset<int64_t>& in_vals) {
  std::multiset<int64_t> factored_vals;
  for (auto val : in_vals) {
    auto factors = computeFactors(val);
    factored_vals.insert(factors.begin(), factors.end());
  }
  return factored_vals;
}

std::pair<std::multiset<int64_t>, std::multiset<int64_t>> removeCommonFactors(
    const std::multiset<int64_t>& set1,
    const std::multiset<int64_t>& set2) {
  // Will iterate over this list, removing entries from the following lists
  auto simplified_set1 = set1;
  auto simplified_set2 = set2;

  simplified_set1.erase(1);
  simplified_set2.erase(1);

  for (auto numerator_factor : set1) {
    auto denominator_factor_it = simplified_set2.find(numerator_factor);
    if (denominator_factor_it != simplified_set2.end()) {
      simplified_set2.erase(denominator_factor_it);
      simplified_set1.erase(simplified_set1.find(numerator_factor));
    }
  }

  return {simplified_set1, simplified_set2};
}

// Given factors made up of a product of integers over a product of integers,
// simplify the factors. This should really likely use a better factorization
// algorithm, however going to start simple here.
std::pair<std::vector<int64_t>, std::vector<int64_t>> removeCommonFactors(
    const std::vector<int64_t>& vec1,
    const std::vector<int64_t>& vec2) {
  std::multiset<int64_t> set1(vec1.begin(), vec1.end());
  std::multiset<int64_t> set2(vec2.begin(), vec2.end());

  set1.erase(1);
  set2.erase(1);

  // Initial removal of common factors
  std::tie(set1, set2) = removeCommonFactors(set1, set2);

  // If denominator is empty no reason to continue trying to remove factors
  if (set2.empty()) {
    return {{set1.begin(), set1.end()}, {}};
  }

  // Break sets up into largest set of smallest factors
  set1 = getAllFactors(set1);
  set2 = getAllFactors(set2);

  // Remove common factors again
  std::tie(set1, set2) = removeCommonFactors(set1, set2);

  return {{set1.begin(), set1.end()}, {set2.begin(), set2.end()}};
}

std::pair<std::multiset<Val*>, std::multiset<Val*>> removeSameVals(
    const std::multiset<Val*>& set1,
    const std::multiset<Val*>& set2) {
  // Will iterate over this list, removing entries from the following lists
  auto simplified_set1 = set1;
  auto simplified_set2 = set2;

  for (auto fact1 : set1) {
    auto fact2_it = simplified_set2.find(fact1);
    if (fact2_it != simplified_set2.end()) {
      simplified_set2.erase(fact2_it);
      simplified_set1.erase(simplified_set1.find(fact1));
    }
  }

  return {simplified_set1, simplified_set2};
}

std::pair<std::vector<Val*>, std::vector<Val*>> removeSameVals(
    const std::vector<Val*>& vec1,
    const std::vector<Val*>& vec2) {
  std::multiset<Val*> set1(vec1.begin(), vec1.end());
  std::multiset<Val*> set2(vec2.begin(), vec2.end());
  std::tie(set1, set2) = removeSameVals(set1, set2);

  return {{set1.begin(), set1.end()}, {set2.begin(), set2.end()}};
}

} // namespace factorization_helpers

namespace {
// Search through the almost exact map to see if there's a compile time factor
// that can be used. Otherwise return the concrete ID extent which makes
// simplification pass easier as the same factor will be used more consistently
// where possible.
//
// TODO: This is generally useful, would be good to add it to compute at map and
// refactor lowering to use it so we consistently use compile time values or the
// same symbolic value consistently.
Val* commonOrConstExtent(
    std::shared_ptr<const ComputeAtMap> ca_map,
    IterDomain* id) {
  auto disjoint_set = ca_map->idGraph().almostExactNodes().getDisjointSetOf(id);
  for (auto entry : disjoint_set) {
    if (entry->extent()->isConstScalar()) {
      return entry->extent();
    }
  }
  return ca_map->getConcreteMappedID(id, IdMappingMode::ALMOSTEXACT)->extent();
}
} // namespace

ContiguousInnerDimensionsMapper::ContiguousInnerDimensionsMapper(
    TensorView* reference,
    const std::vector<IterDomain*>& reference_ids,
    std::shared_ptr<const ComputeAtMap> ca_map,
    const std::unordered_set<Split*>& divisible_splits)
    // Send null info to MaxInfoSpanning tree because we need state to compute
    // the info object for reference. It's not really needed on construction,
    // just before traversal.
    : MaxInfoSpanningTree(reference, std::make_shared<MappedDomain>()),
      ca_map_(ca_map),
      divisible_splits_(divisible_splits) {
  FusionGuard fg(reference->fusion());
  // Check which domain of tensor view we should be looking at. All IDs must be
  // found either in the root domain, or the rfactor domain**.
  bool reference_is_rfactor = reference->hasRFactor() &&
      std::all_of(reference_ids.begin(),
                  reference_ids.end(),
                  [reference](IterDomain* id) {
                    return (
                        std::find(
                            reference->getMaybeRFactorDomain().begin(),
                            reference->getMaybeRFactorDomain().end(),
                            id) != reference->getMaybeRFactorDomain().end());
                  });

  if (!reference_is_rfactor) {
    TORCH_INTERNAL_ASSERT(
        std::all_of(
            reference_ids.begin(),
            reference_ids.end(),
            [reference](IterDomain* id) {
              return (
                  std::find(
                      reference->getRootDomain().begin(),
                      reference->getRootDomain().end(),
                      id) != reference->getRootDomain().end());
            }),
        "\nIterDomains passed in to ContiguousInnerDimensionsMapper passed in to ",
        "ContiguousInnerDimensionsMapper must either all exist in the root domain, or all exist ",
        "in the rfactor domain.\nReference: ",
        reference->toString());
  }

  // Record while processing reference's information
  recording_ = true;
  std::shared_ptr<Information> reference_information;
  // Ordering of dimensions is important in this analysis, if an ordering is
  // contiguous in the reference, but not the target tensor views, then we
  // cannot consider that a contiguous merge dimension for vectorization.
  if (reference_is_rfactor) {
    std::vector<IterDomain*> reordered_rfactor;
    for (auto id : reference->getMaybeRFactorDomain()) {
      if (std::find(reference_ids.begin(), reference_ids.end(), id) !=
          reference_ids.end()) {
        reordered_rfactor.push_back(id);
        // Initiailze the extent for the mapped iter domain
        ProjectedExtent pe;
        pe.multiplyNumeratorValue(commonOrConstExtent(ca_map_, id));
        addProjectedExtent(id, pe);
      } else if (!id->isBroadcast()) {
        // Ignore broadcasts in the reference. Otherwise, remove non-contiguous
        // IDs in the reference tensor as this is the contiguous mapper.
        reordered_rfactor.clear();
      }
    }

    reference_information = MappedDomain::build(
        projectIdToRoot(reference, reordered_rfactor),
        reordered_rfactor,
        true /*shouldn't matter how we initialize this*/);
  } else {
    std::vector<IterDomain*> reordered_root;
    for (auto id : reference->getRootDomain()) {
      if (std::find(reference_ids.begin(), reference_ids.end(), id) !=
          reference_ids.end()) {
        reordered_root.push_back(id);
        // Initiailze the extent for the mapped iter domain
        ProjectedExtent pe;
        pe.multiplyNumeratorValue(commonOrConstExtent(ca_map_, id));
        addProjectedExtent(id, pe);
      } else if (!id->isBroadcast()) {
        // Ignore broadcasts in the reference. Otherwise, remove non-contiguous
        // IDs in the reference tensor as this is the contiguous mapper.
        reordered_root.clear();
      }
    }
    reference_information = MappedDomain::build(
        reordered_root,
        projectIdToRFactor(reference, reordered_root),
        false /*shouldn't matter how we initialize this*/);
  }
  // Stop recording before traversal
  recording_ = false;

  // Set MaxInfoSpanningTree::reference_info_ before traversal
  reference_info_ = reference_information;
  // Set ContiguousInnerDimensionsMapper::tv_infos_ entry for reference
  tv_infos_[reference] = reference_information;

  traverse(this);
}

ContiguousInnerDimensionsMapper ContiguousInnerDimensionsMapper::map(
    TensorView* reference,
    const std::vector<IterDomain*>& reference_ids,
    std::shared_ptr<const ComputeAtMap> ca_map,
    const std::unordered_set<Split*>& divisible_splits) {
  return ContiguousInnerDimensionsMapper(
      reference, reference_ids, ca_map, divisible_splits);
}

void ContiguousInnerDimensionsMapper::propagateExtentSplitBackward(
    Split* split,
    bool outer_maps) {
  auto& inner_mapping = getMappedExtent(split->inner());
  if (inner_mapping.isZero()) {
    return;
  }

  if (outer_maps) {
    // TODO: Fix comment

    // Both dimensions map, inner dimension maps fully. For more context see
    // the comment:
    //  projectIdToRFactor
    //    if (find_outer_it != ids.end() && find_inner_it != ids.end() &&
    //      !hasPartialExtent(merge->inner())) {
    //      Comment here.

    const auto inner_numerator = inner_mapping.getNumerator();
    const auto inner_denominator = inner_mapping.getDenominator();

    // Is the inner dimension is fully mapped
    auto inner_is_fully_mapped = IrBuilder::eqExpr(
        SimplifyingIrBuilder::divExpr(inner_numerator, inner_denominator),
        commonOrConstExtent(ca_map_, split->inner()));

    // Divisibility checks are done later, simply propagate fractional
    // values through the graph.

    // Always propagate inner dimension to in backward through split
    auto in_mapping = inner_mapping;
    // If inner is fully mapped propagate the outer mapping as well
    const auto outer_mapping = getMappedExtent(split->outer());
    in_mapping.maybeMul(inner_is_fully_mapped, outer_mapping);
    addProjectedExtent(split->in(), in_mapping);
  } else {
    // Only inner maps
    addProjectedExtent(split->in(), getMappedExtent(split->inner()));
  }
}

void ContiguousInnerDimensionsMapper::propagateExtentMergeBackward(
    const Merge* merge) {
  auto out_mapping = getMappedExtent(merge->out());
  if (out_mapping.isZero()) {
    return;
  }

  // Map inner and outer dimensions. If outer doesn't map fully this will be
  // found during evaluation. Don't have to worry about it here.

  const auto out_numerator = out_mapping.getNumerator();
  const auto out_denominator = out_mapping.getDenominator();

  auto out_bigger_than_inner = IrBuilder::gtExpr(
      SimplifyingIrBuilder::divExpr(out_numerator, out_denominator),
      commonOrConstExtent(ca_map_, merge->inner()));

  // Divisibility checks are done later, simply propagate fractional
  // values through the graph.

  ProjectedExtent inner_mapping;
  // If out is larger than the inner extent use the inner extent
  inner_mapping.multiplyNumeratorValue(SimplifyingIrBuilder::whereExpr(
      out_bigger_than_inner,
      commonOrConstExtent(ca_map_, merge->inner()),
      FusionGuard::getCurFusion()->oneVal()));
  // otherwise use the out mapping extent
  inner_mapping.maybeMul(
      SimplifyingIrBuilder::notExpr(out_bigger_than_inner), out_mapping);
  addProjectedExtent(merge->inner(), inner_mapping);

  ProjectedExtent outer_mapping;
  // If out mapping is bigger than inner, propagate out divided by the inner
  // mapping.
  outer_mapping.maybeMul(out_bigger_than_inner, out_mapping);
  outer_mapping.multiplyDenominatorValue(SimplifyingIrBuilder::whereExpr(
      out_bigger_than_inner,
      commonOrConstExtent(ca_map_, merge->inner()),
      FusionGuard::getCurFusion()->oneVal()));
  addProjectedExtent(merge->outer(), outer_mapping);
}

void ContiguousInnerDimensionsMapper::propagateExtentMergeForward(
    const Merge* merge,
    bool outer_maps) {
  auto& inner_mapping = getMappedExtent(merge->inner());
  if (inner_mapping.isZero()) {
    return;
  }

  if (outer_maps) {
    // Both dimensions map, inner dimension maps fully. We don't map the
    // outer dimension through if the inner dimension maps partially as we'd
    // have to support mapping a non-continuous dimension. i.e.:
    //
    // merge(I0*I1, I2*I3) -> I0*I1*I2*I3
    //
    // With partial mapping of I1 and I3, then there'd be I2 between them
    // so it wouldn't be a continuous segment that we map through this
    // merge. Therefore we'd only consider I3 partialy mapping through this
    // operation.
    //
    // If we have the same merge
    // merge(I0*I1, I2*I3) -> I0*I1*I2*I3
    // However, I2*I3 completely maps, and I1 partially maps, then we can
    // forward a partially mapped domain to the output of size I1*I2*I3
    const auto inner_numerator = inner_mapping.getNumerator();
    const auto inner_denominator = inner_mapping.getDenominator();

    // Is the inner dimension is fully mapped
    auto inner_is_fully_mapped = IrBuilder::eqExpr(
        SimplifyingIrBuilder::divExpr(inner_numerator, inner_denominator),
        commonOrConstExtent(ca_map_, merge->inner()));

    // Divisibility checks are done later, simply propagate fractional
    // values through the graph.

    // Always propagate inner dimension to in backward through split
    auto in_projected_extent = inner_mapping;
    // If inner is fully mapped propagate the outer mapping as well
    const auto outer_mapping = getMappedExtent(merge->outer());
    in_projected_extent.maybeMul(inner_is_fully_mapped, outer_mapping);
    addProjectedExtent(merge->out(), in_projected_extent);
  } else {
    // Only inner maps
    addProjectedExtent(merge->out(), getMappedExtent(merge->inner()));
  }
}

void ContiguousInnerDimensionsMapper::propagateExtentSplitForward(
    Split* split) {
  auto in_mapping = getMappedExtent(split->in());
  if (in_mapping.isZero()) {
    return;
  }
  const auto in_numerator = in_mapping.getNumerator();
  const auto in_denominator = in_mapping.getDenominator();

  auto in_bigger_than_inner = IrBuilder::gtExpr(
      SimplifyingIrBuilder::divExpr(in_numerator, in_denominator),
      commonOrConstExtent(ca_map_, split->inner()));

  // Divisibility checks are done later, simply propagate fractional
  // values through the graph.

  ProjectedExtent inner_mapping;
  // If in is larger than the inner extent use the inner extent
  inner_mapping.multiplyNumeratorValue(IrBuilder::whereExpr(
      in_bigger_than_inner,
      commonOrConstExtent(ca_map_, split->inner()),
      FusionGuard::getCurFusion()->oneVal()));
  // otherwise use the in mapping extent
  inner_mapping.maybeMul(
      SimplifyingIrBuilder::notExpr(in_bigger_than_inner), in_mapping);
  addProjectedExtent(split->inner(), inner_mapping);

  ProjectedExtent outer_mapping;
  // If out mapping is bigger than inner, propagate out divided by the inner
  // mapping.
  outer_mapping.maybeMul(in_bigger_than_inner, in_mapping);
  outer_mapping.multiplyDenominatorValue(SimplifyingIrBuilder::whereExpr(
      in_bigger_than_inner,
      commonOrConstExtent(ca_map_, split->inner()),
      FusionGuard::getCurFusion()->oneVal()));
  addProjectedExtent(split->outer(), outer_mapping);
}

std::vector<IterDomain*> ContiguousInnerDimensionsMapper::projectIdToRoot(
    TensorView* ref,
    std::vector<IterDomain*> ids) {
  auto transform_exprs = StmtSort::getExprs(
      ref->fusion(),
      {ref->getRFactorDomain().begin(), ref->getRFactorDomain().end()});

  // Mapping from rfactor to root, reverse expressions
  std::reverse(transform_exprs.begin(), transform_exprs.end());

  for (auto* expr : transform_exprs) {
    if (Split* split = dynamic_cast<Split*>(expr)) {
      // Initialize state
      auto find_outer_it = ids.begin();
      auto outer_pos = ids.size();
      auto find_inner_it = ids.begin();
      auto inner_pos = ids.size();

      // Removes all entries to the left of provided `it`, if `it` is not
      // ids.begin(). Updates all state of finding outer and inner in the ids
      // vector after erasing.
      auto clear_left_of = [&find_outer_it,
                            &outer_pos,
                            &find_inner_it,
                            &inner_pos,
                            &ids,
                            &split](decltype(find_outer_it) it) {
        if (it != ids.begin()) {
          ids.erase(ids.begin(), it);
        }

        // Set outer it and position
        find_outer_it = std::find(ids.begin(), ids.end(), split->outer());
        outer_pos = std::distance(ids.begin(), find_outer_it);

        // Set inner it and position
        find_inner_it = std::find(ids.begin(), ids.end(), split->inner());
        inner_pos = std::distance(ids.begin(), find_inner_it);
      };

      // Dry run to fill state
      clear_left_of(ids.begin());

      // Cannot map through non-divisible split
      if (divisible_splits_.find(split) == divisible_splits_.end()) {
        if (find_inner_it != ids.end()) {
          clear_left_of(find_inner_it + 1);
        }
        if (find_outer_it != ids.end()) {
          clear_left_of(find_outer_it + 1);
        }
        continue;
      }

      // Check if the domains out of the split are contiguous in the mapped
      // domain.
      if (find_outer_it == ids.end() && find_inner_it != ids.end()) {
        // Outer dimension was not found, but inner dimension was. Must assume
        // everything to the left of inner is not contiguously merged.
        //
        // Clear left of inner
        clear_left_of(find_inner_it);
      } else if (find_outer_it != ids.end() && find_inner_it == ids.end()) {
        // Inner dimension was not found, outer and anything left of outer are
        // definitely not contiguous.
        //
        // Clear outer and left of outer
        clear_left_of(find_outer_it + 1);
        continue;
      } else if (find_outer_it == ids.end() && find_inner_it == ids.end()) {
        // Nothing mapped, just continue
        continue;
      }

      if (find_outer_it != ids.end() && find_inner_it != ids.end()) {
        // Both outer and inner mapped.
        if (outer_pos >= inner_pos) {
          // Make sure outer is outside inner, otherwise neither could be part
          // of a continuous mapping. There are cases where we could have
          // reversible operations e.g.:
          //    [id{3} id{5} id{6}] -> merge(1, 0)
          // -> [id{5*3} id{6}] -> split(0, 3)
          // -> [id{5} id{3} id{6}] -> transpose(0, 1)
          // -> [id{3} id{5} id{6}]
          // However we don't try and capture cases like this correcly, we'd
          // just reduce this down to only the iter domain of size 6 mapping.
          //
          // Clear outer and left of outer
          clear_left_of(find_outer_it + 1);
          continue;
        }

        // Find the position inner would have to have to be considered ordered
        // relative to outer
        auto pos_after_outer = outer_pos + 1;
        for (; pos_after_outer < ids.size(); pos_after_outer++) {
          if (ids[pos_after_outer]->isBroadcast() &&
              pos_after_outer != inner_pos) {
            // Skip broadcast axes as they must not have been concretized in the
            // reference. We remove dimensions that underwent a concretization
            // as well as the dimensions to the left of that in propagateC2P.
            continue;
          }
          break;
        }

        if (inner_pos != pos_after_outer) {
          // Nothing to the left of inner could be continuous.
          //
          // Clear left of inner
          clear_left_of(find_inner_it);
        }
      }

      ids[inner_pos] = split->in();
      bool outer_mapped = find_outer_it != ids.end();
      if (outer_mapped) {
        // Remove outer
        ids.erase(find_outer_it);
      } else {
        // Clear to the left of inner in since only inner maps
        ids.erase(ids.begin(), ids.begin() + inner_pos);
      }

      propagateExtentSplitBackward(split, outer_mapped);

    } else if (const Merge* merge = dynamic_cast<const Merge*>(expr)) {
      auto find_out_it = std::find(ids.begin(), ids.end(), merge->out());
      if (find_out_it == ids.end()) {
        continue;
      }

      auto out_pos = std::distance(ids.begin(), find_out_it);
      ids[out_pos] = merge->outer();
      ids.insert(ids.begin() + out_pos + 1, merge->inner());

      propagateExtentMergeBackward(merge);
    } else {
      // TODO: I wonder if we should just remove all inputs instead of erroring.
      // Seems that would be safe.
      TORCH_INTERNAL_ASSERT(
          false,
          "ProjectDimensions does not support expr type: ",
          expr->toString());
    } // switch on expr type
  } // For loop on the transform expressions

  return ids;
}

// This function is very similar to projectIdToRoot, we just generally swap the
// logic of split and merge as the reverse mapping of merge looks a lot like
// split and vice versa.
std::vector<IterDomain*> ContiguousInnerDimensionsMapper::projectIdToRFactor(
    TensorView* ref,
    std::vector<IterDomain*> ids) {
  auto transform_exprs = StmtSort::getExprs(
      ref->fusion(),
      {ref->getRFactorDomain().begin(), ref->getRFactorDomain().end()});

  // Map forward through transforms since we're going from root to rfactor
  for (auto* expr : transform_exprs) {
    if (const Merge* merge = dynamic_cast<const Merge*>(expr)) {
      // Initialize state
      auto find_outer_it = ids.begin();
      auto outer_pos = ids.size();
      auto find_inner_it = ids.begin();
      auto inner_pos = ids.size();

      // Removes all entries to the left of provided `it`, if `it` is not
      // ids.begin(). Updates all state of finding outer and inner in the ids
      // vector after erasing.
      auto clear_left_of = [&find_outer_it,
                            &outer_pos,
                            &find_inner_it,
                            &inner_pos,
                            &ids,
                            &merge](decltype(find_outer_it) it) {
        if (it != ids.begin()) {
          ids.erase(ids.begin(), it);
        }
        find_outer_it = std::find(ids.begin(), ids.end(), merge->outer());
        outer_pos = find_outer_it == ids.end()
            ? ids.size()
            : std::distance(ids.begin(), find_outer_it);

        find_inner_it = std::find(ids.begin(), ids.end(), merge->inner());
        inner_pos = find_inner_it == ids.end()
            ? ids.size()
            : std::distance(ids.begin(), find_inner_it);
      };

      // Dry run to fill state
      clear_left_of(ids.begin());

      // Check if the input domains of the merge are contiguous in the mapped
      // domain.
      if (find_outer_it == ids.end() && find_inner_it != ids.end()) {
        // Outer dimension was not found, but inner dimension was. Must assume
        // everything to the left of inner is not contiguously merged.
        //
        // Clear left of inner
        clear_left_of(find_inner_it);
      } else if (find_outer_it != ids.end() && find_inner_it == ids.end()) {
        // Inner dimension was not found, outer and anything left of outer are
        // definitely not contiguous.
        //
        // Clear outer and left of outer
        clear_left_of(find_outer_it + 1);
        continue;
      } else if (find_outer_it == ids.end() && find_inner_it == ids.end()) {
        // Nothing mapped, just continue
        continue;
      }

      if (find_outer_it != ids.end() && find_inner_it != ids.end()) {
        // Both outer and inner mapped.
        if (outer_pos >= inner_pos) {
          // Make sure outer is outside inner, otherwise neither could be part
          // of a continuous mapping. There are cases where we could have
          // reversible operations e.g.:
          //    [id{3} id{5} id{6}] -> merge(1, 0)
          // -> [id{5*3} id{6}] -> split(0, 5)
          // -> [id{5} id{3} id{6}] -> transpose(0, 1)
          // -> [id{3} id{5} id{6}]
          // However we don't try and capture cases like this correcly, we'd
          // just reduce this down to only the iter domain of size 6 mapping.
          //
          // Clear outer and left of outer
          clear_left_of(find_outer_it + 1);
          continue;
        }

        // Find the position inner would have to have to be considered ordered
        // relative to outer
        auto pos_after_outer = outer_pos + 1;
        for (; pos_after_outer < ids.size(); pos_after_outer++) {
          if (ids[pos_after_outer]->isBroadcast() &&
              pos_after_outer != inner_pos) {
            // Skip broadcast axes as they must not have been concretized in the
            // reference. We remove dimensions that underwent a concretization
            // as well as the dimensions to the left of that.
            continue;
          }
          break;
        }

        if (inner_pos != pos_after_outer) {
          // Nothing to the left of inner could be continuous.
          //
          // Clear left of inner
          clear_left_of(find_inner_it);
        }
      }

      ids[inner_pos] = merge->out();
      bool outer_mapped = find_outer_it != ids.end();
      if (outer_mapped) {
        // Remove outer
        ids.erase(find_outer_it);
      } else {
        // Clear to the left of inner in since only inner maps
        ids.erase(ids.begin(), ids.begin() + inner_pos);
      }

      propagateExtentMergeForward(merge, outer_mapped);

    } else if (Split* split = dynamic_cast<Split*>(expr)) {
      auto find_in_it = std::find(ids.begin(), ids.end(), split->in());
      if (find_in_it == ids.end()) {
        continue;
      }

      auto in_pos = std::distance(ids.begin(), find_in_it);

      // Map inner and outer dimensions. If outer doesn't map fully this will be
      // found during evaluation. Don't have to worry about it here.
      ids[in_pos] = split->outer();
      ids.insert(ids.begin() + in_pos + 1, split->inner());

      // Cannot map through non-divisible split
      if (divisible_splits_.find(split) == divisible_splits_.end()) {
        if (find_in_it != ids.end()) {
          ids.erase(ids.begin(), ids.begin() + in_pos + 1);
        }
        continue;
      }

      propagateExtentSplitForward(split);
    } else {
      // TODO: I wonder if we should just remove all inputs instead of erroring.
      // Seems that would be safe.
      TORCH_INTERNAL_ASSERT(
          false,
          "ProjectDimensions does not support expr type: ",
          expr->toString());
    } // switch on expr type
  } // For loop on the transform expressions
  return ids;
}

std::shared_ptr<MaxInfoSpanningTree::Information>
ContiguousInnerDimensionsMapper::computeInfoC2P(
    TensorView* from,
    TensorView* to,
    std::shared_ptr<MaxInfoSpanningTree::Information> from_info) {
  auto from_ids = std::dynamic_pointer_cast<const MappedDomain>(from_info)
                      ->mapped_root_ids_;
  // If we have a case where we have a concretized broadcast that's being
  // tracked in a consumer but not concretized in the producer we should break
  // off the dimensions connected to the left of that dimension. So if we have:
  // T0[i0, i2]
  // T1[i0, b1, i2] = broadcast(T0)
  // T2[i0, i1, i2]
  // T3[i0, i1, i2] = T1 + T2
  // and we're propogating from T3 with {i0, i1, i2}
  // When we go from T3 to T0, we don't have any mechanism to understand that i0
  // and i2 are not contiguous in the original domain of T3. It's not ideal with
  // transpose, but when this happens we'll clear all dimensions mapped left of
  // the concretized broadcast.
  // So if we have:
  // T0[i1, i2]
  // T1[b0, i1, i2] = broadcast(T0)
  // T2[i1, b0, i2] = transpose(T1)
  // T3[i1, i0, i2]
  // T4[i1, i0, i2] = T2 + T3
  // T5[i0, i1, i2] = transpose(T4)
  // Then i1 and i2 are contiguous in both T0 and T5, but due to the realization
  // of the broadcast on T4 we will have removed i1 from the mapped set.
  PairwiseRootDomainMap root_map(to, from);
  auto c2p_map = root_map.mapConsumerToProducer(from->domain(), to->domain());

  // Id's in consumer to clear from the mapped set due to broadcast
  // concretization.
  std::unordered_set<IterDomain*> consumer_ids_to_clear;
  if (to->hasBroadcast()) {
    // Find the last broadcast dimension resolved in consumers root domain
    int clear_pos = -1;
    for (auto i : c10::irange(from->getRootDomain().size())) {
      auto c_id = from->getRootDomain()[i];
      auto c_it = c2p_map.find(c_id);
      if (c_it == c2p_map.end()) {
        continue;
      }
      auto p_id = c_it->second;
      if ((!c_id->isBroadcast()) && p_id->isBroadcast()) {
        clear_pos = i;
      }
    }
    // Clear everything to the left of the inner most resolved broadcast
    // dimension, including the broadcasted domain.
    if (clear_pos >= 0) {
      consumer_ids_to_clear.insert(
          from->getRootDomain().begin(),
          from->getRootDomain().begin() + clear_pos + 1);
    }
  }

  std::vector<IterDomain*> producer_rfactor_ids;
  for (auto from_id : from_ids) {
    auto c2p_it = c2p_map.find(from_id);
    if (c2p_it != c2p_map.end() &&
        consumer_ids_to_clear.find(c2p_it->first) ==
            consumer_ids_to_clear.end()) {
      producer_rfactor_ids.push_back(c2p_it->second);
      if (recording_) {
        addProjectedExtent(c2p_it->second, getMappedExtent(c2p_it->first));
      }
    }
  }
  return MappedDomain::build(
      projectIdToRoot(to, producer_rfactor_ids), producer_rfactor_ids, true);
}

std::shared_ptr<MaxInfoSpanningTree::Information>
ContiguousInnerDimensionsMapper::computeInfoP2C(
    TensorView* from,
    TensorView* to,
    std::shared_ptr<MaxInfoSpanningTree::Information> from_info) {
  auto from_ids = std::dynamic_pointer_cast<const MappedDomain>(from_info)
                      ->mapped_rfactor_ids_;
  // If we have a case where we have a reduction that's being tracked in a
  // producer but not a consumer we should break off the dimensions connected to
  // the left of that reduction. So if we have:
  // T0[i0, i1, i2]
  // T1[i0, r1, i2] = sum(T0)
  // T2[i0, i2] = T1
  // and we're propogating from T0 with {i0, i1, i2}
  // When we go from T1 to T2, we don't have any mechanism to understand that i0
  // and i2 are not contiguous in the original domain of T0. It's not ideal with
  // transpose, but when this happens we'll clear all dimensions mapped left of
  // the reduction.
  // So if we have:
  // T0[i0, i1, i2]
  // T1[i1, i0, i2] = transpose(T0)
  // T2[i1, r0, i2] = sum(T1)
  // T3[i1, i2] = T2
  // Then i1 and i2 are contiguous in both T0 and T3, but due to the sum on T1
  // we will have removed i1.
  PairwiseRootDomainMap root_map(from, to);
  auto p2c_map = root_map.mapProducerToConsumer(from->domain(), to->domain());
  std::vector<IterDomain*> consumer_root_ids;

  // Id's in producer to clear from the mapped set due to reductions.
  std::unordered_set<IterDomain*> producer_ids_to_clear;
  if (from->hasReduction()) {
    // Find the last reduction dimension in the rfactor domain.
    int clear_pos = -1;
    for (auto i : c10::irange(from->getMaybeRFactorDomain().size())) {
      if (from->getMaybeRFactorDomain()[i]->isReduction()) {
        clear_pos = i;
      }
    }
    // Clear everything to the left of the inner most reduction dimension.
    if (clear_pos >= 0) {
      producer_ids_to_clear.insert(
          from->getMaybeRFactorDomain().begin(),
          from->getMaybeRFactorDomain().begin() + clear_pos + 1);
    }
  }

  for (auto from_id : from_ids) {
    auto p2c_it = p2c_map.find(from_id);
    if (p2c_it != p2c_map.end() &&
        producer_ids_to_clear.find(p2c_it->first) ==
            producer_ids_to_clear.end()) {
      consumer_root_ids.push_back(p2c_it->second);
      if (recording_) {
        addProjectedExtent(p2c_it->second, getMappedExtent(p2c_it->first));
      }
    }
  }
  return MappedDomain::build(
      consumer_root_ids, projectIdToRFactor(to, consumer_root_ids), false);
}

std::shared_ptr<MaxInfoSpanningTree::Information>
ContiguousInnerDimensionsMapper::computeInfoSibling(
    TensorView* from,
    TensorView* to,
    std::shared_ptr<MaxInfoSpanningTree::Information> from_info) {
  TORCH_INTERNAL_ASSERT(
      from->getRootDomain().size() == to->getRootDomain().size(),
      "Siblings of different root sizes not supported, but found:\n  ",
      from->toString(),
      "\n  and\n  ",
      to->toString(),
      "\nhave root sizes of ",
      from->getRootDomain().size(),
      " and ",
      to->getRootDomain().size());

  auto from_root_ids = std::dynamic_pointer_cast<const MappedDomain>(from_info)
                           ->mapped_root_ids_;
  std::vector<IterDomain*> sibling_root_ids;

  for (auto from_root_id : from_root_ids) {
    auto from_it = std::find(
        from->getRootDomain().begin(),
        from->getRootDomain().end(),
        from_root_id);
    TORCH_INTERNAL_ASSERT(
        from_it != from->getRootDomain().end(),
        "Expected ",
        from_root_id->toString(),
        " to be in the root of ",
        from->toString());
    auto pos = std::distance(from->getRootDomain().begin(), from_it);
    sibling_root_ids.push_back(to->getRootDomain()[pos]);
    if (recording_) {
      addProjectedExtent(
          to->getRootDomain()[pos],
          getMappedExtent(from->getRootDomain()[pos]));
    }
  }

  if (!from->hasRFactor()) {
    return MappedDomain::build(
        sibling_root_ids,
        sibling_root_ids,
        false /*shouldn't matter how we initialize this*/);
  }

  TORCH_INTERNAL_ASSERT(
      from->getRFactorDomain().size() == to->getRFactorDomain().size(),
      "Siblings of different rfactor sizes not supported, but found:\n  ",
      from->toString(),
      "\n  and\n  ",
      to->toString(),
      "\nhave rfactor sizes of ",
      from->getRFactorDomain().size(),
      " and ",
      to->getRFactorDomain().size());

  auto from_rfactor_ids =
      std::dynamic_pointer_cast<const MappedDomain>(from_info)
          ->mapped_rfactor_ids_;
  std::vector<IterDomain*> sibling_rfactor_ids;

  for (auto from_rfactor_id : from_rfactor_ids) {
    auto from_it = std::find(
        from->getRFactorDomain().begin(),
        from->getRFactorDomain().end(),
        from_rfactor_id);
    TORCH_INTERNAL_ASSERT(
        from_it != from->getRFactorDomain().end(),
        "Expected ",
        from_rfactor_id->toString(),
        " to be in the rfactor of ",
        from->toString());
    auto pos = std::distance(from->getRFactorDomain().begin(), from_it);
    sibling_rfactor_ids.push_back(to->getRFactorDomain()[pos]);
    if (recording_) {
      addProjectedExtent(
          to->getRFactorDomain()[pos],
          getMappedExtent(from->getRFactorDomain()[pos]));
    }
  }

  return MappedDomain::build(sibling_root_ids, sibling_rfactor_ids, false);
}

// MaxInfoSpanningTree functions
void ContiguousInnerDimensionsMapper::propagateC2P(
    TensorView* from,
    TensorView* to) {
  recording_ = true;
  auto from_info = tv_infos_.at(from);
  auto to_info = computeInfoC2P(from, to, from_info);
  tv_infos_[to] = to_info;
}

void ContiguousInnerDimensionsMapper::propagateP2C(
    TensorView* from,
    TensorView* to) {
  recording_ = true;
  auto from_info = tv_infos_.at(from);
  auto to_info = computeInfoP2C(from, to, from_info);
  tv_infos_[to] = to_info;
}

void ContiguousInnerDimensionsMapper::propagateSibling(
    TensorView* from,
    TensorView* to) {
  recording_ = true;
  auto from_info = tv_infos_.at(from);
  auto to_info = computeInfoSibling(from, to, from_info);
  tv_infos_[to] = to_info;
}

// Returns Mappings of all dims in reference starting from inner most position
// to outer most position. e.g. T0[i0, r1, b2] will return 3 Mapper instances
// associated with:
// {{i0, r1, b1}, {r1, b1}, {b1}}
std::vector<ContiguousInnerDimensionsMapper> getAllVectorizedMapsOf(
    TensorView* ref) {
  std::vector<ContiguousInnerDimensionsMapper> mappers;
  auto root_dom = ref->hasReduction() && ref->hasRFactor()
      ? ref->getRootDomain()
      : ref->getMaybeRFactorDomain();
  while (!root_dom.empty()) {
    mappers.push_back(ContiguousInnerDimensionsMapper::map(ref, root_dom));
    root_dom.erase(root_dom.begin());
  }
  return mappers;
}

// Returns ProjectedExtent entires that should be evaluated and multiplied based
// on contiguity of reference, dimensions mapped to ref in mapper, and
// divisibiltiy/partial mapping of the vector.
//
// TODO: Rename, recomment (ProjectedExtent references based on mapper. Lifetime
// has to be managed the same as mapper, or references returned will be
// invalid).
std::vector<std::pair<ProjectedExtent&, IterDomain*>> getContigVectorSizesOf(
    TensorView* of_tv,
    ContiguousInnerDimensionsMapper& mapper) {
  // Logic copied to get root according to scheduler_utils::innerMostRootDim
  // also copied from SchedulerRuntimeInfo::getMaxVectorizableWidth
  bool use_root_dom = of_tv->hasReduction() && of_tv->hasRFactor();
  auto of_tv_root =
      use_root_dom ? of_tv->getRootDomain() : of_tv->getMaybeRFactorDomain();

  if (!mapper.hasMappedDims(of_tv)) {
    return {};
  }

  const std::vector<IterDomain*>& projected_dims = use_root_dom
      ? mapper.mappedRootIds(of_tv)
      : mapper.mappedRFactorIds(of_tv);
  auto of_tv_root_no_reductions = TensorDomain::noReductions(of_tv_root);

  auto contiguity = of_tv->domain()->contiguity();
  // Appears after reductions the reduction domain often has a contiguity entry.
  // This only matters if the result of the reduction is an output
  if (contiguity.size() == of_tv_root.size() &&
      contiguity.size() != of_tv_root_no_reductions.size()) {
    std::vector<bool> new_contiguity;
    for (auto i : c10::irange(of_tv_root.size())) {
      if (!of_tv_root[i]->isReduction()) {
        new_contiguity.push_back(contiguity[i]);
      }
    }
    contiguity = new_contiguity;
  }
  of_tv_root = of_tv_root_no_reductions;

  auto of_tv_root_size = of_tv_root.size();

  // Filter out 0-dim tensors
  if (of_tv_root_size < 1) {
    return {};
  }

  TORCH_INTERNAL_ASSERT(
      of_tv_root_size == contiguity.size(), "Contiguity mismatch found.");

  std::vector<std::pair<ProjectedExtent&, IterDomain*>> vectorizable_dim_sizes;

  // Order is important, need to make sure dimensions match up correctly with
  // what was propogated through the mapper. The mapper's dimensions is
  // propogated in the order of the reference, if that order doesn't match the
  // tensor we're mapping too then a transpose interfered with expanded the
  // vectorize dimension.
  size_t projected_dims_i = projected_dims.size();

  for (auto i : c10::irange(of_tv_root_size)) {
    if (projected_dims_i == 0) {
      break;
    }
    auto root_i = of_tv_root_size - i - 1;
    auto root_id = of_tv_root[root_i];

    if (root_id->extent()->isOneInt() || root_id->isBroadcast()) {
      if (projected_dims[projected_dims_i - 1]->sameAs(root_id)) {
        --projected_dims_i;
      }
      continue;
    }

    // Not contiguous
    if (!contiguity[root_i]) {
      break;
    }

    // Mapping order isn't correct, cannot expand vectorization dimension.
    if (!projected_dims[--projected_dims_i]->sameAs(root_id)) {
      break;
    }

    auto& mapped_extent_PE = mapper.getMappedExtent(root_id);
    if (mapped_extent_PE.isZero()) {
      break;
    }

    vectorizable_dim_sizes.push_back({mapped_extent_PE, root_id});
  }
  return vectorizable_dim_sizes;
}

// ProjectedExtent and ExpressionEvaluation cannot be const since they have lazy
// evaluated values. However, nothing will be modified in this function for
// either object. Max vectorize size returned will be 128.
int64_t getVectorizationSize(
    std::vector<std::pair<ProjectedExtent&, IterDomain*>> dim_info,
    ExpressionEvaluator& expr_eval) {
  if (dim_info.empty()) {
    return 1;
  }

  // Reverse the size vector to traverse from the innermost dimensions
  std::reverse(dim_info.begin(), dim_info.end());
  int64_t vectorize_size = 1;

  for (auto dim : dim_info) {
    if (vectorize_size > 128 && vectorize_size % 128 == 0) {
      return vectorize_size;
    }
    auto& pe = dim.first;
    if (pe.isZero()) {
      continue;
    }
    auto orig_extent_val = dim.second->extent();
    auto numerator_optional = expr_eval.evaluate(pe.getNumerator());
    auto denominator_optional = expr_eval.evaluate(pe.getDenominator());
    auto extent_optional = expr_eval.evaluate(orig_extent_val);
    TORCH_INTERNAL_ASSERT(
        numerator_optional.has_value() && denominator_optional.has_value() &&
            extent_optional.has_value(),
        "Vectorization heuristic could not evaluate required extents.");
    TORCH_INTERNAL_ASSERT(
        numerator_optional->isInt() && denominator_optional->isInt() &&
            extent_optional->isInt(),
        "Vectorization heuristic expects integer values only.");
    auto numerator = numerator_optional->as<int64_t>();
    auto denominator = denominator_optional->as<int64_t>();
    auto extent = extent_optional->as<int64_t>();

    if (denominator != 1) {
      break;
    }

    // Full mapping of numerator
    if (numerator == extent) {
      // Full mappings can continue to the next dimension
      vectorize_size = vectorize_size * extent;
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        numerator < extent,
        "Mapped extent in vectorization analysis should never be greater than the extent but ",
        numerator,
        " > ",
        extent);

    if (extent % numerator) {
      vectorize_size = vectorize_size * numerator;
      // partial mappings cannot continue to the next dimension
      break;
    }

    int64_t greatest_common_factor = 1;
    // Look for a common factors of 3 and 2
    while (extent % 3 == 0 && numerator % 3 == 0) {
      extent /= 3;
      numerator /= 3;
      greatest_common_factor = greatest_common_factor * 3;
    }

    while (extent % 2 == 0 && numerator % 2 == 0) {
      extent /= 2;
      numerator /= 2;
      greatest_common_factor = greatest_common_factor * 2;
    }
    vectorize_size = vectorize_size * greatest_common_factor;
    // partial mappings cannot continue to the next dimension
    break;
  }
  return vectorize_size;
}

size_t getExpandedVectorization(
    const std::vector<ContiguousInnerDimensionsMapper>& reference_maps,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*> vectorizable_inputs_outputs,
    TensorView* reference_tv,
    int break_point,
    size_t default_word_size) {
  if (vectorizable_inputs_outputs.empty()) {
    return 1;
  }

  size_t max_expand_size = SchedulerRuntimeInfo::max_alignment_size_in_byte;
  size_t common_alignment_size =
      SchedulerRuntimeInfo::max_alignment_size_in_byte;

  for (auto inp_or_out : vectorizable_inputs_outputs) {
    auto dtype_size = dataTypeSize(
        inp_or_out->dtype(), indexModeToDtype(runtime_info.getIndexMode()));

    max_expand_size = std::min(
        max_expand_size,
        SchedulerRuntimeInfo::max_alignment_size_in_byte / dtype_size);
    max_expand_size = std::min(
        max_expand_size, runtime_info.getMaxVectorizableWidth(inp_or_out));
    common_alignment_size = std::min(
        common_alignment_size, runtime_info.getAlignmentSize(inp_or_out));
  }

  // If there's no possibility to increase vector size of provided tensors,
  // then don't bother doing a more complex analysis to try and do so, just
  // return early.
  if (max_expand_size == default_word_size) {
    return default_word_size;
  }

  auto reference_map = reference_maps[break_point];
  // Initialize to max the tensors could support.
  size_t max_supported_vector_size = max_expand_size;
  for (auto inp_or_out : vectorizable_inputs_outputs) {
    size_t contig_dim_size = getVectorizationSize(
        getContigVectorSizesOf(inp_or_out, reference_map),
        runtime_info.expressionEvaluator());
    size_t local_max_vec_size = 1;

    while (contig_dim_size > 1 && contig_dim_size % 2 == 0 &&
           local_max_vec_size < max_expand_size) {
      contig_dim_size /= 2;
      local_max_vec_size *= 2;
    }

    max_supported_vector_size =
        std::min(local_max_vec_size, max_supported_vector_size);
  }
  return max_supported_vector_size;
}

size_t getVectorizationFactor(
    SchedulerRuntimeInfo& runtime_info,
    TensorView* reference_tv,
    HeuristicSummary* data_cache,
    int break_point) {
  auto vectorizable_inputs_outputs_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::VectorizableInputsAndOutputs>(
          data_cache, [&reference_tv]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getInputsOutputsWithInnerDim(
                    reference_tv, true, true));
          });

  auto& vectorizable_inputs_outputs = vectorizable_inputs_outputs_entry.get();

  size_t vectorize_factor = std::numeric_limits<size_t>::max();

  for (auto tv : vectorizable_inputs_outputs) {
    const auto tv_vectorize_factor =
        runtime_info.getInnerDimVectorizableWidth(tv);
    vectorize_factor = std::min(vectorize_factor, tv_vectorize_factor);
  }

  if (vectorize_factor == std::numeric_limits<size_t>::max()) {
    vectorize_factor = 1;
  }

  auto vectorize_maps_entry =
      HeuristicSummaryEntry<HeuristicCompileTime::VectorizeMaps>(
          data_cache, [&reference_tv]() {
            return std::make_unique<
                std::vector<vectorize_helper::ContiguousInnerDimensionsMapper>>(
                vectorize_helper::getAllVectorizedMapsOf(reference_tv));
          });

  vectorize_factor = vectorize_helper::getExpandedVectorization(
      vectorize_maps_entry.get(),
      runtime_info,
      vectorizable_inputs_outputs,
      reference_tv,
      break_point,
      vectorize_factor);

  return vectorize_factor;
}

} // namespace vectorize_helper
} // namespace nvfuser
