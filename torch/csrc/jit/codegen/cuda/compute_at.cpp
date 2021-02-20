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

ComputeAtData::ComputeAtData(TensorView* tv)
    : tv_ref_(tv), original_compute_at_position(tv->getComputeAtPosition()) {}

// Clear pass based data
void ComputeAtData::clearPass() {
  current_traversal_position_set = false;
  current_traversal_position = 0;
}

void ComputeAtData::setPassPosition(unsigned int pos) {
  if (current_traversal_position_set) {
    // A single traversal cannot try to enforce more than one position on a
    // TensorView as it would produce in incorrect code. If this is hit, then
    // the given tensor and its production should be duplicated.
    TORCH_CHECK(
        pos == current_traversal_position,
        "Error during computeAt. ComputeAt pass wanted to set position of TensorView: ",
        tv_ref_->name(),
        " at position ",
        pos,
        " but was already set to position ",
        current_traversal_position,
        ". This tensor would have to be recomputed to satsify the selected computeAt position.");
  }

  if (pos > original_compute_at_position) {
    current_traversal_position = pos;
    current_traversal_position_set = true;
  }
}

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

} // namespace

void ComputeAt::runAt(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_position) {
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

  std::vector<TensorView*> producers;

  // It doesn't make sense to set computeAt on an input as it's not generated,
  // it's provided. If this was called, move the computeAt to users of the
  // producer that are in a dependency between prodcer and consumer.
  if (producer->fusion()->hasInput(producer)) {
    auto all_chains =
        tvChains(DependencyCheck::getAllDependencyChains(producer, consumer));

    TORCH_CHECK(
        !all_chains.empty(),
        "Compute At expects ",
        producer->name(),
        " is a dependency of ",
        consumer->name(),
        ", however it is not.");

    std::unordered_set<TensorView*> added_producers;

    // Check all dependency chains, select the next TV after producer towards
    // consumer. These are the TVs we're going to actually call computeAt on.
    for (const auto& tv_chain : all_chains) {
      // When a chain only has two tensors, they must be the producer,
      // which is an input, and the consumer. There is nothing we need
      // to do for such chains.
      if (tv_chain.size() > 2) {
        // Make sure we only add once, but we want to add in a determinsitic
        // order
        if (added_producers.find(tv_chain[1]) == added_producers.end()) {
          producers.push_back(tv_chain[1]);
          added_producers.emplace(tv_chain[1]);
        }
      }
    }
  } else {
    // If producer is not an input, it's the only one.
    producers.push_back(producer);
  }

  // Run computeAt on our potentially modified producer(s)
  if (!producers.empty()) {
    for (auto producer_to_run : producers) {
      ComputeAt ca(producer_to_run, consumer, consumer, consumer_position);
      ca.runPass();
    }
  }
}

void ComputeAt::runWith(
    TensorView* producer,
    TensorView* consumer,
    unsigned int producer_position) {
  FUSER_PERF_SCOPE("ComputeAt::runWith");

  // Make sure the correct fusion is setup between this and consumer.
  TORCH_CHECK(
      producer->fusion() == consumer->fusion(),
      producer,
      " and ",
      consumer,
      " are not in the same fusion.");

  // Make sure Fusion Guard is set appropriately
  FusionGuard fg(producer->fusion());

  ComputeAt ca(producer, consumer, producer, producer_position);
  ca.runPass();
}

// Actually applies transformation
unsigned int ComputeAt::backwardComputeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_compute_at_pos) {
  FUSER_PERF_SCOPE("backwardComputeAt_impl");

  auto& producer_entry = tv_data.at(producer);

  auto replay = TransformReplay::replayPasC(
      producer->domain(),
      consumer->domain(),
      (int)consumer_compute_at_pos,
      root_map_);

  if (replay.second == 0) {
    return 0;
  }

  producer_entry.setPassPosition(replay.second);

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
    producer->setComputeAt(replay.second);
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

  auto& consumer_entry = tv_data.at(consumer);
  const auto& producer_entry = tv_data.at(producer);

  auto replay = TransformReplay::replayCasP(
      consumer->domain(),
      producer->domain(),
      (int)producer_compute_at_pos,
      root_map_);

  consumer_entry.setPassPosition(replay.second);

  if (producer_compute_at_pos > producer->getComputeAtPosition()) {
    producer->setComputeAt((int)producer_compute_at_pos);
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

  unsigned int producer_pos = reference_ == producer_
      ? reference_position_
      : producer_->getComputeAtPosition();

  // propagate forward through all chains
  for (auto tv_dep_chain : chains) {
    TensorView* running_producer = nullptr;
    TensorView* running_consumer = tv_dep_chain.front();
    tv_dep_chain.pop_front();
    unsigned int running_producer_pos = producer_pos;

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

  // Initialize tv_data for all TensorViews we may modify
  auto chains = producer_use_chains_;
  if (common_consumer_ != nullptr) {
    chains = tvChains(
        DependencyCheck::getAllDependencyChains(producer_, common_consumer_));
  }

  for (const auto& tv_chain : chains) {
    for (auto tv : tv_chain) {
      if (tv_data.find(tv) == tv_data.end()) {
        tv_data[tv] = ComputeAtData(tv);
      }
    }
  }

  // Traverse backward through all dep chains from producer to consumer
  traverseBackward();

  // Clear data from backward traversal:
  for (auto& entry : tv_data) {
    entry.second.clearPass();
  }

  // Start at producer and traverse forward through all chains
  traverseForward();
}

ComputeAt::ComputeAt(
    TensorView* _producer,
    TensorView* _consumer,
    TensorView* _reference,
    unsigned int _reference_position)
    : producer_(_producer),
      consumer_(_consumer),
      reference_(_reference),
      reference_position_(_reference_position) {
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
