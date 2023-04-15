
#include <lower_divisible_split.h>

#include <disjoint_set.h>
#include <ir_utils.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::unordered_set<Split*> getAllDivisibleSplits(Fusion* fusion) {
  ComputeAtMap ca_map(fusion);
  return getAllDivisibleSplits(fusion, &ca_map);
}

std::unordered_set<Split*> getAllDivisibleSplits(
    Fusion* fusion,
    const ComputeAtMap* ca_map) {
  std::unordered_set<Split*> all_divisible_splits;

  auto all_tvs = ir_utils::allTvs(fusion);
  // Find all tensor views with a view like rfactor. Splits used in view
  // transformations must be divisible by definition.
  for (auto tv : all_tvs) {
    auto rfactor_dom = tv->getMaybeRFactorDomain();
    // Not view if there's no rfactor axis
    if (!tv->domain()->hasViewLikeRFactor()) {
      continue;
    }

    // Take the view transformations and add all the splits. Those splits are
    // the only divisible splits.
    auto view_exprs =
        StmtSort::getExprs(fusion, {rfactor_dom.begin(), rfactor_dom.end()});
    auto split_exprs = ir_utils::filterByType<Split>(view_exprs);
    all_divisible_splits.insert(split_exprs.begin(), split_exprs.end());
  }

  // Vectorized dimensions are enforced to be a result of divisible splits.
  // Gather vectorized splits.
  for (auto tv : all_tvs) {
    auto vec_id_it = std::find_if(
        tv->domain()->domain().begin(),
        tv->domain()->domain().end(),
        [](IterDomain* id) {
          return isParallelTypeVectorize(id->getParallelType());
        });

    if (vec_id_it == tv->domain()->domain().end()) {
      continue;
    }

    // We could have a case technically like:
    // [8, 2] where we do:
    // split(0, 2)
    // merge(1)
    // so it ends up as [4, 4]
    // split(0, 2) must be divisible, but for now we're not going to capture
    // cases like this. Just look for direct split's producing a vectorize
    // dimension.
    auto vec_id = *vec_id_it;
    if (vec_id->definition() != nullptr && vec_id->definition()->isA<Split>()) {
      all_divisible_splits.emplace(vec_id->definition()->as<Split>());
    }
  }

  // If there's no view like splits, there's nothing to find
  if (all_divisible_splits.empty()) {
    return all_divisible_splits;
  }

  // Track the concrete id in the exact map of the outer output of the split
  // expressions. This is how we'll check if there are matching splits. This
  // also gets rid of any splits that already match (for processing).
  std::unordered_map<IterDomain*, Expr*> outer_concrete_id_to_expr;

  for (auto split : all_divisible_splits) {
    outer_concrete_id_to_expr[ca_map->getConcreteMappedID(
        split->outer(), IdMappingMode::EXACT)] = split;
  }

  std::unordered_set<Expr*> visited(
      all_divisible_splits.begin(), all_divisible_splits.end());

  // Find splits that match what we already have:
  for (auto entry : outer_concrete_id_to_expr) {
    auto concrete_id = entry.first;
    auto original_view_split = entry.second;

    const auto& exact_mapped_ids =
        ca_map->idGraph().exactNodes().getDisjointSetOf(concrete_id).vector();
    for (auto other_id : exact_mapped_ids) {
      if (other_id->definition() == nullptr) {
        continue;
      }

      if (!visited.emplace(other_id->definition()).second) {
        // Already visited
        continue;
      }

      if (IterDomainGraph::exprsMap(
              original_view_split,
              other_id->definition(),
              false,
              ca_map->idGraph().exactNodes())) {
        all_divisible_splits.emplace(other_id->definition()->as<Split>());
      }
    }
  }

  return all_divisible_splits;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
