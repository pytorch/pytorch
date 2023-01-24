#include <torch/csrc/jit/codegen/cuda/scheduler/vectorize_helper.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/contiguity.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower_divisible_split.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>

#include <c10/util/irange.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace vectorize_helper {

// Grab all values and expressions used to make the merged_domain and remove
// them from the fusion
void cleanUpInnermostMergedDomains(
    const std::vector<IterDomain*>& root_domain,
    IterDomain* merged_domain) {
  TORCH_INTERNAL_ASSERT(merged_domain != nullptr);
  TORCH_INTERNAL_ASSERT(!root_domain.empty());

  std::unordered_set<Val*> root_set({root_domain.begin(), root_domain.end()});

  auto vals = DependencyCheck::getAllValsBetween(root_set, {merged_domain});

  for (auto it = vals.rbegin(); it != vals.rend(); ++it) {
    TORCH_INTERNAL_ASSERT((*it)->isA<IterDomain>());
    auto id = (*it)->as<IterDomain>();
    if (root_set.find(id) != root_set.end()) {
      continue;
    }
    Fusion* fusion = id->container()->as<Fusion>();
    auto id_def = id->definition();
    TORCH_INTERNAL_ASSERT(
        id_def->isA<Merge>(),
        "Invalid ID: ",
        id->toString(),
        ". Expected definition of a Merge expression: ",
        (id_def != nullptr ? id_def->toString() : "nullptr"));
    fusion->removeExpr(id_def);
    fusion->removeVal(id);
  }
}

// Merge innermost domains for finding the widest vectorizable
// size. Return the merged domain or nullptr if no merge is done.
IterDomain* mergeInnermostDomains(
    const std::vector<IterDomain*>& domain,
    int num_merged_domains) {
  const auto ndims = domain.size();
  IterDomain* merged_id = nullptr;
  bool is_merge_done = false;
  for (const auto i : c10::irange(num_merged_domains)) {
    auto id = domain.at(ndims - 1 - i);
    // broadcast and trivial reductions are ignored
    if (id->isBroadcast() || id->isTrivialReduction()) {
      continue;
    }
    if (merged_id == nullptr) {
      merged_id = id;
    } else {
      auto id_inner = merged_id;
      auto id_outer = id;
      merged_id = IterDomain::merge(id_outer, id_inner);
      is_merge_done = true;
    }
  }
  return is_merge_done ? merged_id : nullptr;
}

size_t collectMaxVectorizeSizeWithContigMerge(
    TensorView* tv,
    IterDomain* leaf_merged_domain,
    size_t max_vector_size_in_byte,
    ExpressionEvaluator& expression_evaluator,
    DataType index_type) {
  auto dtype_size = dataTypeSize(tv->dtype(), index_type);
  const size_t max_vector_size = max_vector_size_in_byte / dtype_size;

  // Assume no halo-related expression appears in the fusion. No
  // broadcast is merged, so indexability can be assumed to be true.
  // This is expensive, as ContigIDs builds other things like CAMap,
  // HaloInfo, and ConcreteBroadcast info. We should explicitly build and reuse
  // these as they're compile time information.
  ContigIDs contigIds(
      {leaf_merged_domain},
      tv->getMaybeRFactorDomain(),
      tv->domain()->contiguity(),
      {},
      {},
      getAllDivisibleSplits(tv->fusion()),
      {},
      true);

  auto innermost_root_id = tv->getMaybeRFactorDomain().back();
  auto indexed_id = contigIds.rootToIndexedID().at(innermost_root_id);

  size_t merged_size = 1;
  // If the indexed ID is a contig merged domain, i.e., it is
  // different from innermost_root_id, we accumulate the extents of
  // all the root domains covered by the contig indexed ID. Otherwise,
  // just look at the extent of the innermost root ID.
  if (indexed_id != innermost_root_id) {
    const auto& within_root = contigIds.withinContigIDs().at(indexed_id);
    for (auto root_id : tv->getMaybeRFactorDomain()) {
      if (within_root.find(root_id) == within_root.end()) {
        continue;
      }
      auto maybe_dimension_size =
          expression_evaluator.evaluate(root_id->extent());
      TORCH_INTERNAL_ASSERT(
          maybe_dimension_size.has_value(),
          "Unknown extent of tv: ",
          tv->toString(),
          ", id: ",
          root_id->toString());
      merged_size *= maybe_dimension_size->as<int64_t>();
    }
  } else {
    auto maybe_dimension_size =
        expression_evaluator.evaluate(innermost_root_id->extent());
    TORCH_INTERNAL_ASSERT(
        maybe_dimension_size.has_value(),
        "Unknown extent of tv: ",
        tv->toString(),
        ", id: ",
        innermost_root_id->toString());
    merged_size = maybe_dimension_size->as<int64_t>();
  }

  size_t vector_size = 1;
  size_t next_vector_size = vector_size * 2;

  // Try until vector size exceeds the max allowed size
  while (next_vector_size <= max_vector_size) {
    if (merged_size % next_vector_size != 0) {
      break;
    }
    vector_size = next_vector_size;
    next_vector_size *= 2;
  }

  return vector_size;
}

//! Attempt to expand vectorized domains to contig merged domains. Break point
//! identifies the point in which you can't propagate contiguous merges. For
//! example in pointwise this is the point where we want to split the
//! parallelization to take advantage of broadcast, and for reduction
//! schedulers it's the point where we switch from a reduction domain to an
//! iter domain (or vice versa).
size_t expandVectorizationToContigMergedDomains(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*> vectorizable_inputs_outputs,
    TensorView* reference_tv,
    int break_point,
    size_t default_word_size) {
  size_t max_expand_size = SchedulerRuntimeInfo::max_alignment_size_in_byte;
  size_t common_alignment_size =
      SchedulerRuntimeInfo::max_alignment_size_in_byte;

  for (auto inp_out : vectorizable_inputs_outputs) {
    auto dtype_size = dataTypeSize(
        inp_out->dtype(), indexModeToDtype(runtime_info.getIndexMode()));

    max_expand_size = std::min(
        max_expand_size,
        SchedulerRuntimeInfo::max_alignment_size_in_byte / dtype_size);
    max_expand_size = std::min(
        max_expand_size, runtime_info.getMaxVectorizableWidth(inp_out));
    common_alignment_size =
        std::min(common_alignment_size, runtime_info.getAlignmentSize(inp_out));
  }

  // If there's no possibility to increase vector size of provided tensors,
  // then don't bother doing a more complex analysis to try and do so, just
  // return early.
  if (max_expand_size == default_word_size) {
    return default_word_size;
  }

  auto ca_map = ComputeAtMap(fusion);

  // Merge the domains right of the break point
  const auto& ref_root = reference_tv->getMaybeRFactorDomain();
  const int max_num_merged_domains =
      static_cast<int>(ref_root.size()) - static_cast<int>(break_point);
  int64_t num_merged_domains = 0;
  while (num_merged_domains < max_num_merged_domains) {
    auto pos = (int64_t)ref_root.size() - 1 - num_merged_domains;
    if (!reference_tv->domain()->contiguity()[pos]) {
      break;
    }
    num_merged_domains++;
  }

  // No expansion with no merged domain
  if (num_merged_domains == 0) {
    return default_word_size;
  }

  // Merge the domains but don't modify TensorDomain
  auto merged_domain = mergeInnermostDomains(ref_root, num_merged_domains);

  // No expansion is done if no merge is done.
  if (merged_domain == nullptr) {
    return default_word_size;
  }

  // Find the vectorizable word size with the merged domains
  size_t word_size = collectMaxVectorizeSizeWithContigMerge(
      reference_tv,
      merged_domain,
      common_alignment_size,
      runtime_info.expressionEvaluator(),
      indexModeToDtype(runtime_info.getIndexMode()));

  cleanUpInnermostMergedDomains(ref_root, merged_domain);

  // Stop if the reference doesn't get a larger word size.
  if (word_size <= default_word_size) {
    return default_word_size;
  }

  // Check the other TVs and take the minimum of the valid word sizes
  for (const auto tv : vectorizable_inputs_outputs) {
    if (tv == reference_tv) {
      continue;
    }

    const auto& tv_root = tv->getMaybeRFactorDomain();

    int tv_num_merged_domains = 0;
    for (const auto i : c10::irange(max_num_merged_domains)) {
      if (i == tv_root.size()) {
        break;
      }
      auto ref_id = ref_root.at(ref_root.size() - 1 - i);
      auto pos = tv_root.size() - 1 - i;
      IterDomain* tv_id = tv_root.at(pos);
      // If not mapped, stop expanding.
      if (!ca_map.areMapped(ref_id, tv_id, IdMappingMode::EXACT) ||
          !tv->domain()->contiguity()[pos]) {
        break;
      } else {
        ++tv_num_merged_domains;
      }
    }

    size_t tv_word_size = 1;
    if (tv_num_merged_domains > 1) {
      auto tv_merged_domain =
          mergeInnermostDomains(tv_root, tv_num_merged_domains);
      if (tv_merged_domain == nullptr) {
        tv_word_size = runtime_info.getInnerDimVectorizableWidth(tv);
      } else {
        tv_word_size = collectMaxVectorizeSizeWithContigMerge(
            tv,
            tv_merged_domain,
            common_alignment_size,
            runtime_info.expressionEvaluator(),
            indexModeToDtype(runtime_info.getIndexMode()));
        cleanUpInnermostMergedDomains(tv_root, tv_merged_domain);
      }
    } else {
      tv_word_size = runtime_info.getInnerDimVectorizableWidth(tv);
    }

    word_size = std::min(word_size, tv_word_size);
  }

  return word_size;
}

} // namespace vectorize_helper
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
