#include <inlining.h>
#include <ir_utils.h>
#include <root_domain_map.h>
#include <transform_iter.h>

#include <utility>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

MaxPosCalculator::MaxPosCalculator(
    const std::unordered_set<IterDomain*>& uninlinable_ids)
    : uninlinable_ids_(uninlinable_ids) {
  buildUnmappableDims();
}

void MaxPosCalculator::buildUnmappableDims() {
  ComputeAtRootDomainMap root_map;
  root_map.build();
  auto all_tvs = ir_utils::allTvs(FusionGuard::getCurFusion());
  for (auto tv : all_tvs) {
    auto consumers = ir_utils::consumerTvsOf(tv);
    for (auto consumer : consumers) {
      // Grab dimensions in producer and consumer that are mappable to eachother
      // based on the computeAtRootDomainMap. This will tell us which dimensions
      // can be inlined based on avoiding trying to inline non-trivial
      // reduction structures.
      auto mappable_roots =
          root_map.getMappableDims(tv->domain(), consumer->domain());
      for (auto tv_root_id : tv->getMaybeRFactorDomain()) {
        if (mappable_roots.find(tv_root_id) == mappable_roots.end() &&
            !tv_root_id->isTrivialReduction()) {
          unmappable_dims_.emplace(tv_root_id);
        }
      }
    }
  }
}

bool MaxPosCalculator::isAllowedID(
    IterDomain* id,
    TensorView* tv,
    bool best_effort,
    bool allow_reduction,
    bool allow_vectorize,
    bool allow_unmappable) const {
  bool allowed = true;

  if (!allow_reduction) {
    allowed = allowed && !id->isReduction();
  }

  if (uninlinable_ids_.count(id)) {
    return false;
  }

  if (!allow_vectorize) {
    // Avoid inlining if marked as Vectorize or Group. In the case of
    // BestEffort and MostInlined modes, avoid Unroll as well.
    bool is_vectorize = isParallelTypeVectorize(id->getParallelType()) ||
        id->getParallelType() == ParallelType::Group ||
        (best_effort && id->getParallelType() == ParallelType::Unroll);
    allowed = allowed && !is_vectorize;
  }

  if (!allow_unmappable) {
    auto root_dom = tv->getMaybeRFactorDomain();
    std::unordered_set<Val*> root_dom_set(root_dom.begin(), root_dom.end());
    auto all_vals = DependencyCheck::getAllValsBetween(root_dom_set, {id});
    bool is_unmappable = false;
    for (auto val : all_vals) {
      auto id = val->as<IterDomain>();
      if (root_dom_set.count(val) > 0 && unmappable_dims_.count(id) > 0) {
        is_unmappable = true;
        break;
      }
    }
    allowed = allowed && !is_unmappable;
  }

  return allowed;
}

size_t MaxPosCalculator::getMaxPosSelf(
    TensorView* tv,
    bool best_effort,
    bool allow_reduction,
    bool allow_vectorize,
    bool allow_unmappable) const {
  auto dom = tv->domain()->domain();
  auto iter = std::find_if(dom.begin(), dom.end(), [=](IterDomain* id) {
    return !isAllowedID(
        id,
        tv,
        best_effort,
        allow_reduction,
        allow_vectorize,
        allow_unmappable);
  });
  return std::distance(dom.begin(), iter);
}

// Return the max position in producer that can be inlined to consumer
// Cannot inline:
//   Vectorized dimensions in consumer
//   Unrolled dimensions in consumer
size_t MaxPosCalculator::getMaxProducerPosFromConsumer(
    TensorView* producer,
    TensorView* consumer,
    bool best_effort) const {
  auto pairwise_root_map = PairwiseRootDomainMap(producer, consumer);
  auto replay_CasP =
      BestEffortReplay::replayCasP(consumer, producer, -1, pairwise_root_map);
  auto p2c_replay_map = replay_CasP.getReplay();

  for (size_t producer_pos = 0; producer_pos < producer->nDims();
       producer_pos++) {
    // If the producer position is mismatching with the consumer, then we can
    // not inline into this position, otherwise the max producer position of
    // the consumer will become invalid and expression sort will fail.
    if (TransformReplay::getMatchedLeafPosWithoutReplayCasP(
            consumer, producer, producer_pos + 1) < 0) {
      return producer_pos;
    }
    auto map_it = p2c_replay_map.find(producer->axis(producer_pos));
    if (map_it != p2c_replay_map.end()) {
      auto c_id = map_it->second;
      if (!isAllowedID(c_id, consumer, best_effort, true, false, true)) {
        return producer_pos;
      }
    }
  }
  return producer->nDims();
}

size_t MaxPosCalculator::getMaxPosAll(
    TensorView* tv,
    bool best_effort,
    bool check_siblings) {
  auto max_pos = getMaxPosSelf(tv, best_effort, false, false, false);
  for (auto consumer_tv : ir_utils::consumerTvsOf(tv)) {
    max_pos = std::min<size_t>(
        max_pos, getMaxProducerPosFromConsumer(tv, consumer_tv, best_effort));
  }
  if (check_siblings) {
    for (auto sibling_tv : ir_utils::siblingTvsOf(tv)) {
      max_pos = std::min<size_t>(
          max_pos, getMaxPosAll(sibling_tv, best_effort, false));
    }
  }
  return max_pos;
}

void inlineMost(const std::unordered_set<IterDomain*>& uninlinable_ids) {
  inlineMost(ir_utils::allTvs(FusionGuard::getCurFusion()), uninlinable_ids);
}

void inlineMost(
    const std::vector<TensorView*>& tvs,
    const std::unordered_set<IterDomain*>& uninlinable_ids) {
  if (tvs.empty()) {
    return;
  }
  MaxPosCalculator calc(uninlinable_ids);
  for (auto tv : tvs) {
    tv->inlineAt(-1, true, &calc);
  }
}

void inlineMost(
    const std::unordered_set<TensorView*>& tvs,
    const std::unordered_set<IterDomain*>& uninlinable_ids) {
  if (tvs.empty()) {
    return;
  }
  MaxPosCalculator calc(uninlinable_ids);
  for (auto tv : tvs) {
    tv->inlineAt(-1, true, &calc);
  }
}

namespace {

// Find the positions of `selected` tensors that is mapped to the given position
// in the reference tensor.
class FindMappedPositions : public MaxInfoSpanningTree::Propagator {
  std::unordered_map<TensorView*, size_t>& output_;

 public:
  FindMappedPositions(
      std::unordered_map<TensorView*, size_t>& output,
      TensorView* reference,
      int64_t reference_pos);

  ~FindMappedPositions() override = default;

  virtual void propagateC2P(TensorView* from, TensorView* to) override;
  virtual void propagateP2C(TensorView* from, TensorView* to) override;
  virtual void propagateSibling(TensorView* from, TensorView* to) override;
};

FindMappedPositions::FindMappedPositions(
    std::unordered_map<TensorView*, size_t>& output,
    TensorView* reference,
    int64_t reference_pos)
    : output_(output) {
  if (reference_pos < 0) {
    reference_pos += int64_t(reference->nDims()) + 1;
  }
  TORCH_CHECK(
      reference_pos >= 0 && reference_pos <= int64_t(reference->nDims()),
      "Invalid axis received ",
      reference_pos,
      " but should be > -",
      reference->nDims(),
      " and <= ",
      reference->nDims(),
      ".");
  output_[reference] = reference_pos;
}

void FindMappedPositions::propagateC2P(TensorView* from, TensorView* to) {
  int from_pos = output_.at(from);
  auto to_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(to, from, from_pos);
  // If there is no matching position found, we compute the highest matched
  // position as the closest approximation
  while (to_pos < 0) {
    from_pos--;
    to_pos =
        TransformReplay::getMatchedLeafPosWithoutReplayPasC(to, from, from_pos);
  }
  output_[to] = to_pos;
}

void FindMappedPositions::propagateP2C(TensorView* from, TensorView* to) {
  int from_pos = output_.at(from);
  auto to_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(to, from, from_pos);
  // If there is no matching position found, we compute the highest matched
  // position as the closest approximation
  while (to_pos < 0) {
    from_pos--;
    to_pos =
        TransformReplay::getMatchedLeafPosWithoutReplayCasP(to, from, from_pos);
  }
  output_[to] = to_pos;
}

void FindMappedPositions::propagateSibling(TensorView* from, TensorView* to) {
  auto from_pos = output_.at(from);
  TORCH_CHECK(
      TransformReplay::fullSelfMatching(to, from),
      "Transformations in siblings ",
      from,
      " and ",
      to,
      " does not match with each other.");
  output_[to] = from_pos;
}

std::unordered_map<TensorView*, size_t> getPositionsMappedTo(
    TensorView* reference_tv,
    int64_t reference_pos) {
  std::unordered_map<TensorView*, size_t> mapped_positions;
  MaxRootDomainInfoSpanningTree tree(reference_tv, reference_pos);
  FindMappedPositions propagator(mapped_positions, reference_tv, reference_pos);
  tree.traverse(&propagator);
  return mapped_positions;
}

} // namespace

void inlineAllAt(
    TensorView* reference_tv,
    int64_t reference_pos,
    bool best_effort,
    const std::unordered_set<IterDomain*>& uninlinable_ids) {
  auto mapped_positions = getPositionsMappedTo(reference_tv, reference_pos);
  MaxPosCalculator calc(uninlinable_ids);
  for (auto pair : mapped_positions) {
    pair.first->inlineAt(pair.second, best_effort, &calc);
  }
}

void inlineSelectedAt(
    const std::unordered_set<TensorView*>& selected,
    TensorView* reference_tv,
    int64_t reference_pos,
    bool best_effort,
    const std::unordered_set<IterDomain*>& uninlinable_ids) {
  auto mapped_positions = getPositionsMappedTo(reference_tv, reference_pos);
  MaxPosCalculator calc(uninlinable_ids);
  for (auto pair : mapped_positions) {
    if (selected.count(pair.first) > 0) {
      pair.first->inlineAt(pair.second, best_effort, &calc);
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
