#include <torch/csrc/jit/codegen/cuda/compute_at.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Simple selector that only propagates across tensor views in the provided
// unordered_set. Will also propagate to all consumers of those tensors, and the
// siblings of those tensors.
class ComputeAtSelector : public MaxInfoSpanningTree::Selector {
  std::unordered_set<TensorView*> selected_;

 public:
  virtual bool allowC2P(TensorView* from, TensorView* to) override {
    return selected_.count(to) > 0;
  }

  virtual bool allowP2C(TensorView* from, TensorView* to) override {
    // If the producer is in the selected set, then the consumer must also be
    // replayed to obtain a compatible loop structure so that this producer
    // can be consumed in this loop.
    return selected_.count(from) > 0 || selected_.count(to) > 0;
  }

  virtual bool allowSibling(TensorView* from, TensorView* to) override {
    return true;
  }

  ComputeAtSelector(std::unordered_set<TensorView*> selected)
      : selected_(std::move(selected)) {}
  const std::unordered_set<TensorView*>& selected() const {
    return selected_;
  }
};

namespace {

// Wrapper around set_intersection
template <typename T>
std::set<T> set_intersection(const std::set<T>& set1, const std::set<T>& set2) {
  std::set<T> intersection;
  std::set_intersection(
      set1.begin(),
      set1.end(),
      set2.begin(),
      set2.end(),
      std::inserter(intersection, intersection.begin()));
  return intersection;
}

std::deque<std::deque<TensorView*>> tvChains(
    std::deque<std::deque<Val*>> val_chains) {
  std::deque<std::deque<TensorView*>> tv_chains(val_chains.size());
  for (const auto i : c10::irange(val_chains.size())) {
    auto tv_iterable = ir_utils::filterByType<TensorView>(val_chains[i]);
    tv_chains[i] =
        std::deque<TensorView*>(tv_iterable.begin(), tv_iterable.end());
  }
  return tv_chains;
}

std::unordered_set<TensorView*> getAllTVsBetween(
    TensorView* producer,
    TensorView* consumer) {
  TORCH_CHECK(
      DependencyCheck::isDependencyOf(producer, consumer),
      "Compute At expects ",
      producer->name(),
      " is a dependency of ",
      consumer->name(),
      ", however it is not.");
  auto between_vals =
      DependencyCheck::getAllValsBetween({producer}, {consumer});
  auto between_tvs = ir_utils::filterByType<TensorView>(between_vals);
  std::unordered_set<TensorView*> result(
      between_tvs.begin(), between_tvs.end());
  result.erase(consumer);
  return result;
}

TensorView* getCommonConsumer(TensorView* producer, TensorView* consumer) {
  FUSER_PERF_SCOPE("ComputeAt::setCommonConsumer");
  auto producer_use_chains_ =
      tvChains(DependencyCheck::getAllUseChains(producer));

  // Convert the first chain to a set.
  std::set<TensorView*> common_consumers(
      producer_use_chains_.front().begin(), producer_use_chains_.front().end());

  // Run through all use chains of producer, and intersect them to find common
  // TVs
  for (auto tv_chain : producer_use_chains_) {
    common_consumers = set_intersection(
        common_consumers,
        std::set<TensorView*>(tv_chain.begin(), tv_chain.end()));
  }

  auto all_chains =
      tvChains(DependencyCheck::getAllDependencyChains(producer, consumer));

  // Right now we only support compute at if at some point in the graph consumer
  // is dependent on producer.
  TORCH_CHECK(
      !all_chains.empty(),
      "Compute At expects ",
      producer->name(),
      " is a dependency of ",
      consumer->name(),
      ", however it is not.");

  // Remove all TVs from producer to consumer as common consumer must be at or
  // after consumer
  for (const auto& tv_chain : all_chains) {
    for (auto tv : tv_chain) {
      if (tv != consumer)
        common_consumers.erase(tv);
    }
  }

  // If there is a common consumer, grab the first one at or after consumer
  TensorView* common_consumer = nullptr;
  if (!common_consumers.empty()) {
    for (auto tv : producer_use_chains_.front()) {
      if (common_consumers.find(tv) != common_consumers.end()) {
        common_consumer = tv;
        break;
      }
    }
    TORCH_INTERNAL_ASSERT(
        common_consumer != nullptr,
        "Hit a logical inconsistency in the computeAt pass.");
  }
  return common_consumer;
}

void pullInSiblings(std::unordered_set<TensorView*>& s) {
  for (auto tv : s) {
    for (auto sibling_tv : ir_utils::siblingTvsOf(tv)) {
      if (sibling_tv == tv) {
        continue;
      }
      s.emplace(sibling_tv);
    }
  }
}

// I am just trying to get the same set of tensors being transformed matching
// the previous behavior of ComputeAt. The algorithm to compute this set is
// horrible, but I don't care because I will eventually completely remove
// ComputeAt, and this algorihtm is not worse than the pervious ComputeAt. :)
std::unordered_set<TensorView*> getPropagationSubgraph(
    TensorView* producer,
    TensorView* consumer) {
  TORCH_CHECK(
      DependencyCheck::isDependencyOf(producer, consumer),
      "Compute At expects ",
      producer->name(),
      " is a dependency of ",
      consumer->name(),
      ", however it is not.");
  TensorView* common_consumer = getCommonConsumer(producer, consumer);
  if (common_consumer != nullptr) {
    auto result = getAllTVsBetween(producer, common_consumer);
    pullInSiblings(result);
    return result;
  }
  auto result_vals = DependencyCheck::getAllDependentVals({producer});
  result_vals.emplace(producer);
  auto result_tvs = ir_utils::filterByType<TensorView>(result_vals);
  std::unordered_set<TensorView*> result;
  std::copy_if(
      result_tvs.begin(),
      result_tvs.end(),
      std::inserter(result, result.begin()),
      [](TensorView* tv) { return !tv->uses().empty(); });
  pullInSiblings(result);
  return result;
}

} // namespace

void ComputeAt::runAt(
    TensorView* producer,
    TensorView* consumer,
    int64_t consumer_position,
    ComputeAtMode mode) {
  FUSER_PERF_SCOPE("ComputeAt::runAt");

  // Make sure the correct fusion is setup between this and consumer.
  TORCH_CHECK(
      producer->fusion() == consumer->fusion(),
      producer,
      " and ",
      consumer,
      " are not in the same fusion.");

  if (mode == ComputeAtMode::MostInlined) {
    consumer_position = -1;
  }

  FusionGuard fg(producer->fusion());

  auto selected = getPropagationSubgraph(producer, consumer);
  ComputeAtSelector selector(selected);

  MaxRootDomainInfoSpanningTree path(consumer, consumer_position, &selector);

  if (mode == ComputeAtMode::MostInlined) {
    MostInlinedTransformPropagator propagator;
    path.traverse(&propagator);
    inlineMost(selected);
  } else {
    TransformPropagator propagator(consumer, consumer_position);
    path.traverse(&propagator);
    inlineSelectedAt(
        selected,
        consumer,
        consumer_position,
        mode == ComputeAtMode::BestEffort);
  }
}

void ComputeAt::runWith(
    TensorView* producer,
    TensorView* consumer,
    int64_t producer_position,
    ComputeAtMode mode) {
  FUSER_PERF_SCOPE("ComputeAt::runWith");

  // Make sure the correct fusion is setup between this and consumer.
  TORCH_CHECK(
      producer->fusion() == consumer->fusion(),
      producer,
      " and ",
      consumer,
      " are not in the same fusion.");

  if (mode == ComputeAtMode::MostInlined) {
    producer_position = -1;
  }

  FusionGuard fg(producer->fusion());

  auto selected = getPropagationSubgraph(producer, consumer);
  ComputeAtSelector selector(selected);

  MaxRootDomainInfoSpanningTree path(producer, producer_position, &selector);

  if (mode == ComputeAtMode::MostInlined) {
    MostInlinedTransformPropagator propagator;
    path.traverse(&propagator);
    inlineMost(selected);
  } else {
    TransformPropagator propagator(producer, producer_position);
    path.traverse(&propagator);
    inlineSelectedAt(
        selected,
        producer,
        producer_position,
        mode == ComputeAtMode::BestEffort);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
