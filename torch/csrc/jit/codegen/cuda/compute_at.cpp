#include <torch/csrc/jit/codegen/cuda/compute_at.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

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

// convert an iterable of Val* to be an iterable of TensorView*
template <typename T1, typename T2>
T1 tvIterable(const T2& val_iterable) {
  T1 tv_iterable = T1();
  std::transform(
      val_iterable.begin(),
      val_iterable.end(),
      std::back_inserter(tv_iterable),
      [](Val* v) {
        TORCH_INTERNAL_ASSERT(
            v->getValType().value() == ValType::TensorView,
            "When following the computeAt dependency chain, a non TensorView value was found.");
        return v->as<TensorView>();
      });
  return tv_iterable;
}

std::deque<std::deque<TensorView*>> tvChains(
    std::deque<std::deque<Val*>> val_chains) {
  std::deque<std::deque<TensorView*>> tv_chains(val_chains.size());
  for (size_t i = 0; i < val_chains.size(); i++) {
    tv_chains[i] = tvIterable<std::deque<TensorView*>>(val_chains[i]);
  }
  return tv_chains;
}

bool validateDomain(TensorView* tv, TensorDomain* new_td) {
  auto first_mismatch =
      BestEffortReplay::findFirstMismatchedID(tv->domain(), new_td);
  return first_mismatch >= (int)tv->getMaxProducerPosition() &&
      first_mismatch >= (int)tv->getComputeAtPosition();
}

unsigned int getReplayablePosPasC(
    TensorView* producer,
    TensorView* consumer,
    const ComputeAtRootDomainMap& root_map_) {
  auto mappable_roots =
      root_map_.getMappableDims(producer->domain(), consumer->domain(), true);

  for (size_t consumer_pos = consumer->nDims(); consumer_pos > 0;
       consumer_pos--) {
    auto root_dim_vals = IterVisitor::getInputsTo(
        {consumer->domain()->domain().begin(),
         consumer->domain()->domain().begin() + consumer_pos});
    auto root_dim = ir_utils::filterByType<IterDomain>(root_dim_vals);
    if (std::any_of(
            root_dim.begin(),
            root_dim.end(),
            [&mappable_roots](IterDomain* root_id) {
              return mappable_roots.find(root_id) == mappable_roots.end();
            })) {
      continue;
    }
    return consumer_pos;
  }
  return 0;
}

unsigned int getReplayablePosCasP(
    TensorView* consumer,
    TensorView* producer,
    const ComputeAtRootDomainMap& root_map_) {
  auto mappable_roots =
      root_map_.getMappableDims(producer->domain(), consumer->domain(), false);

  auto p_dom = producer->domain()->domain();
  auto first_reduction =
      std::find_if(p_dom.begin(), p_dom.end(), [](IterDomain* id) {
        return id->isReduction();
      });

  auto max_producer_pos = std::distance(p_dom.begin(), first_reduction);

  for (size_t producer_pos = max_producer_pos; producer_pos > 0;
       producer_pos--) {
    auto all_vals = DependencyCheck::getAllValsBetween(
        {producer->getMaybeRFactorDomain().begin(),
         producer->getMaybeRFactorDomain().end()},
        {p_dom.begin(), p_dom.begin() + producer_pos});

    if (std::any_of(
            producer->getMaybeRFactorDomain().begin(),
            producer->getMaybeRFactorDomain().end(),
            [&mappable_roots, &all_vals](IterDomain* root_id) {
              return std::find(all_vals.begin(), all_vals.end(), root_id) !=
                  all_vals.end() &&
                  mappable_roots.find(root_id) == mappable_roots.end();
            })) {
      continue;
    }
    return producer_pos;
  }
  return 0;
}

} // namespace

void ComputeAt::runAt(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_position,
    ComputeAtMode mode) {
  FUSER_PERF_SCOPE("ComputeAt::run");

  // Make sure the correct fusion is setup between this and consumer.
  TORCH_CHECK(
      producer->fusion() == consumer->fusion(),
      producer,
      " and ",
      consumer,
      " are not in the same fusion.");

  // Make sure Fusion Guard is set appropriately
  FusionGuard fg(producer->fusion());

  TORCH_CHECK(
      DependencyCheck::isDependencyOf(producer, consumer),
      "Compute At expects ",
      producer->name(),
      " is a dependency of ",
      consumer->name(),
      ", however it is not.");

  // Run computeAt on our potentially modified producer(s)
  ComputeAt ca(producer, consumer, consumer, consumer_position, mode);
  ca.runPass();
}

void ComputeAt::runWith(
    TensorView* producer,
    TensorView* consumer,
    unsigned int producer_position,
    ComputeAtMode mode) {
  FUSER_PERF_SCOPE("ComputeAt::runWith");

  // Make sure the correct fusion is setup between this and consumer.
  TORCH_CHECK(
      producer->fusion() == consumer->fusion(),
      producer,
      " and ",
      consumer,
      " are not in the same fusion.");

  TORCH_CHECK(
      DependencyCheck::isDependencyOf(producer, consumer),
      "Compute At expects ",
      producer->name(),
      " is a dependency of ",
      consumer->name(),
      ", however it is not.");

  // Make sure Fusion Guard is set appropriately
  FusionGuard fg(producer->fusion());

  ComputeAt ca(producer, consumer, producer, producer_position, mode);
  ca.runPass();
}

// Actually applies transformation
unsigned int ComputeAt::backwardComputeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_compute_at_pos) {
  FUSER_PERF_SCOPE("backwardComputeAt_impl");

  if (mode_ == ComputeAtMode::BestEffort) {
    consumer_compute_at_pos = std::min(
        consumer_compute_at_pos,
        getReplayablePosPasC(producer, consumer, root_map_));
  } else if (mode_ == ComputeAtMode::MostInlined) {
    consumer_compute_at_pos =
        getReplayablePosPasC(producer, consumer, root_map_);
  }

  auto replay = TransformReplay::replayPasC(
      producer->domain(),
      consumer->domain(),
      (int)consumer_compute_at_pos,
      root_map_);

  if (replay.second == 0) {
    return 0;
  }

  if (replay.second >= producer->getComputeAtPosition()) {
    const TensorDomain* current_domain = producer->domain();
    TensorDomain* new_domain = replay.first;

    TORCH_INTERNAL_ASSERT(
        validateDomain(producer, new_domain),
        "Tried to set the domain of ",
        producer,
        " to ",
        new_domain,
        " but that would invalidate previously compute at position or max producer position.");

    producer->setDomain(new_domain);
    if (!producer->isFusionInput()) {
      producer->setComputeAt(replay.second);
    }
    consumer->setMaxProducer(consumer_compute_at_pos);
    root_map_.setAlias(current_domain, new_domain);
  }

  return replay.second;
}

// Actually applies transformation, replay consumer based on producer, set
// compute at of producer, set pass position of consumer, return position
// relative to consumer
unsigned int ComputeAt::forwardComputeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int producer_compute_at_pos) {
  FUSER_PERF_SCOPE("forwardComputeAt_impl");

  // Can get into a situation where we inlined into a reduction, but then would
  // try to traverse forward at that position but wouldn't be valid.
  // Reduce position to be inside first reduction
  unsigned int first_red_pos = producer->nDims();
  for (unsigned int i = 0;
       i < (unsigned int)producer->domain()->domain().size();
       i++) {
    if (producer->axis((int)i)->isReduction()) {
      first_red_pos = i;
      break;
    }
  }
  producer_compute_at_pos = std::min(first_red_pos, producer_compute_at_pos);
  if (producer_compute_at_pos == 0) {
    return 0;
  }

  if (mode_ == ComputeAtMode::BestEffort) {
    producer_compute_at_pos = std::min(
        producer_compute_at_pos,
        getReplayablePosCasP(consumer, producer, root_map_));
  } else if (mode_ == ComputeAtMode::MostInlined) {
    producer_compute_at_pos =
        getReplayablePosCasP(consumer, producer, root_map_);
  }
  auto replay = TransformReplay::replayCasP(
      consumer->domain(),
      producer->domain(),
      (int)producer_compute_at_pos,
      root_map_);

  if (producer_compute_at_pos > producer->getComputeAtPosition()) {
    if (!producer->isFusionInput()) {
      producer->setComputeAt((int)producer_compute_at_pos);
    }
  }

  if (replay.second > consumer->getMaxProducerPosition()) {
    const TensorDomain* current_domain = consumer->domain();
    TensorDomain* new_domain = replay.first;

    TORCH_INTERNAL_ASSERT(
        validateDomain(consumer, new_domain),
        "Tried to set the domain of ",
        producer,
        " to ",
        new_domain,
        " but that would invalidate previously compute at position or max producer position.");

    consumer->setDomain(new_domain);
    consumer->setMaxProducer(replay.second);
    root_map_.setAlias(current_domain, new_domain);
  }

  return replay.second;
}

void ComputeAt::setCommonConsumer() {
  FUSER_PERF_SCOPE("ComputeAt::setCommonConsumer");

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
      tvChains(DependencyCheck::getAllDependencyChains(producer_, consumer_));

  // Right now we only support compute at if at some point in the graph consumer
  // is dependent on producer.
  TORCH_CHECK(
      !all_chains.empty(),
      "Compute At expects ",
      producer_->name(),
      " is a dependency of ",
      consumer_->name(),
      ", however it is not.");

  // Remove all TVs from producer to consumer as common consumer must be at or
  // after consumer
  for (const auto& tv_chain : all_chains) {
    for (auto tv : tv_chain) {
      if (tv != consumer_)
        common_consumers.erase(tv);
    }
  }

  // If there is a common consumer, grab the first one at or after consumer
  common_consumer_ = nullptr;
  if (!common_consumers.empty()) {
    for (auto tv : producer_use_chains_.front()) {
      if (common_consumers.find(tv) != common_consumers.end()) {
        common_consumer_ = tv;
        break;
      }
    }
    TORCH_INTERNAL_ASSERT(
        common_consumer_ != nullptr,
        "Hit a logical inconsistency in the computeAt pass.");
  }
}

// Similar to backward traversal in traverseAllKnown but we should only apply
// computeAt if it will increase computeAt positions.
void ComputeAt::traverseBackward() {
  FUSER_PERF_SCOPE("ComputeAt::traverseBackward");
  if (reference_ == producer_) {
    // Forward compute at don't need to run backward traversal
    producer_position_ = reference_position_;
    return;
  }

  // propagate *backward* through all *producer* use_chains or from *producer*
  // to common_consumer if common_consumer exists. Only apply transform if
  // increases computeAt position.
  auto chains =
      tvChains(DependencyCheck::getAllDependencyChains(producer_, consumer_));

  for (auto tv_chain : chains) {
    TensorView* running_producer = tv_chain.back();
    TensorView* running_consumer = nullptr;
    unsigned int running_consumer_pos = reference_position_;
    tv_chain.pop_back();

    TORCH_INTERNAL_ASSERT(running_producer == consumer_);

    while (!tv_chain.empty()) {
      running_consumer = running_producer;
      running_producer = tv_chain.back();
      tv_chain.pop_back();

      running_consumer_pos = backwardComputeAt_impl(
          running_producer, running_consumer, running_consumer_pos);
    }

    TORCH_INTERNAL_ASSERT(
        running_producer == producer_,
        "Compute at backward traversal ended up on something other than the producer.");
    producer_position_ = running_consumer_pos;
  }
}

void ComputeAt::traverseForward() {
  FUSER_PERF_SCOPE("ComputeAt::traverseForward");

  // propagate forward through all *producer* use_chains or from *producer* to
  // common_consumer if common_consumer exists.
  auto chains = producer_use_chains_;
  if (common_consumer_ != nullptr) {
    chains = tvChains(
        DependencyCheck::getAllDependencyChains(producer_, common_consumer_));
  }

  // propagate forward through all chains
  for (auto tv_dep_chain : chains) {
    TensorView* running_producer = nullptr;
    TensorView* running_consumer = tv_dep_chain.front();
    tv_dep_chain.pop_front();
    unsigned int running_producer_pos = producer_position_;

    TORCH_INTERNAL_ASSERT(running_consumer == producer_);

    while (!tv_dep_chain.empty()) {
      running_producer = running_consumer;
      running_consumer = tv_dep_chain.front();
      tv_dep_chain.pop_front();
      running_producer_pos = forwardComputeAt_impl(
          running_producer, running_consumer, running_producer_pos);
    }
  }
}

void ComputeAt::runPass() {
  FUSER_PERF_SCOPE("ComputeAt::runPass");

  // Traverse backward through all dep chains from producer to consumer
  traverseBackward();

  // Start at producer and traverse forward through all chains
  traverseForward();
}

ComputeAt::ComputeAt(
    TensorView* _producer,
    TensorView* _consumer,
    TensorView* _reference,
    unsigned int _reference_position,
    ComputeAtMode _mode)
    : producer_(_producer),
      consumer_(_consumer),
      reference_(_reference),
      reference_position_(_reference_position),
      mode_(_mode) {
  TORCH_INTERNAL_ASSERT(
      reference_ == producer_ || reference_ == consumer_,
      "For compute at reference must be producer or consumer, it's neither.",
      " reference: ",
      reference_,
      " consumer: ",
      consumer_,
      " producer: ",
      producer_);
  TORCH_INTERNAL_ASSERT(
      reference_position_ >= 0 && reference_position_ <= reference_->nDims(),
      "Invalid computeAt axis, received ",
      reference_position_,
      " but should be > -",
      reference_->nDims(),
      " and <= ",
      reference_->nDims(),
      ".");

  producer_use_chains_ = tvChains(DependencyCheck::getAllUseChains(producer_));

  // Look through all the use chains of producer. Check if there's a single
  // consumer for all chains at or after the consumer specified in the computeAt
  // call.
  setCommonConsumer();

  root_map_.build();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
