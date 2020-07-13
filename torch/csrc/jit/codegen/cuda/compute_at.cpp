#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <torch/csrc/jit/codegen/cuda/compute_at.h>

namespace torch {
namespace jit {
namespace fuser {

// Actually applies transformation
void ComputeAt::computeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_compute_at_axis) {
  // Reset view otherwise will conflict with replay.
  producer->clearComputeAt();
  // replay this as consumer / producer as consumer
  auto replay = TransformReplay::replayPasC(
      producer, consumer, (int)consumer_compute_at_axis);
  producer->setComputeAt(consumer, replay.second);
}

// Runs replay, and checks computeAt position. If higher than that provided,
// actually applies.
void ComputeAt::maybe_computeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_compute_at_axis) {
  unsigned int prev_pos = 0;
  if (producer->hasComputeAt())
    prev_pos = producer->getThisComputeAtAxis();

  auto replay = TransformReplay::replayPasC(
      producer->domain(), consumer->domain(), (int)consumer_compute_at_axis);

  if (replay.second > prev_pos) {
    producer->setDomain(replay.first);
    producer->setComputeAt(consumer, replay.second);
  }
}

// Actually applies transformation
void ComputeAt::forwardComputeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int producer_compute_at_axis) {
  // Reset view otherwise will conflict with replay. Don't think this is true
  // anymore.
  producer->clearComputeAt();
  auto replay = TransformReplay::replayCasP(
      consumer, producer, (int)producer_compute_at_axis);
  producer->setComputeAt(consumer, replay.second);
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
T1 tv_iterable(const T2& val_iterable) {
  T1 tv_iterable = T1();
  std::transform(
      val_iterable.begin(),
      val_iterable.end(),
      std::back_inserter(tv_iterable),
      [](Val* v) {
        TORCH_INTERNAL_ASSERT(
            v->getValType().value() == ValType::TensorView,
            "When following the computeAt dependency chain, a non TensorView value was found.");
        return static_cast<TensorView*>(v);
      });
  return tv_iterable;
}

std::deque<std::deque<TensorView*>> getAllTVUseChains(TensorView* tv) {
  // Grab all paths from producer to  of producer in fusion.
  auto val_all_use_chains = DependencyCheck::getAllUseChains(tv);

  // Convert dep chains to tensor view chains.
  std::deque<std::deque<TensorView*>> producer_use_chains_;
  for (const auto& val_dep_chain : val_all_use_chains)
    producer_use_chains_.push_back(
        tv_iterable<std::deque<TensorView*>>(val_dep_chain));
  return producer_use_chains_;
}
} // namespace

void ComputeAt::setCommonConsumer() {
  // Convert the first chain to a set.
  std::set<TensorView*> common_consumers(
      producer_use_chains_.front().begin(), producer_use_chains_.front().end());

  // Run through all use chains of producer, and intersect them to find common
  // TVs
  for (auto dep_chain : producer_use_chains_)
    common_consumers = set_intersection(
        common_consumers,
        std::set<TensorView*>(dep_chain.begin(), dep_chain.end()));

  auto all_chains =
      DependencyCheck::getAllDependencyChains(producer_, consumer_);

  // Right now we only support compute at if at some point in the graph consumer
  // is dependent on producer.
  TORCH_CHECK(
      !all_chains.empty(),
      "Compute At expects ",
      producer_,
      " is a dependency of ",
      consumer_,
      ", however it is not.");

  // Remove all TVs from producer to consumer as common consumer must be at or
  // after consumer
  for (const auto& dep_chain : all_chains) {
    auto tv_chain = tv_iterable<std::deque<TensorView*>>(dep_chain);
    for (auto tv : tv_chain) {
      if (tv != consumer_)
        common_consumers.erase(tv);
    }
  }

  // If there is a common consumer, grab the first one at or after consumer
  common_consumer_ = nullptr;
  if (!common_consumers.empty()) {
    for (TensorView* tv : producer_use_chains_.front())
      if (common_consumers.find(tv) != common_consumers.end()) {
        common_consumer_ = tv;
        break;
      }
    TORCH_INTERNAL_ASSERT(
        common_consumer_ != nullptr,
        "Hit a logical inconsistency in the computeAt pass.");
  }
}

void ComputeAt::traverseAllKnown() {
  std::deque<std::deque<Val*>> chains;

  // propagate backwards through all dep chains from producer to consumer

  // Grab all chains from common_consumer to producer
  chains = DependencyCheck::getAllDependencyChains(producer_, consumer_);

  TORCH_CHECK(
      !chains.empty(),
      "Producer and consumer in a computeAt call must have a dependency between them even if indirect.");

  for (const auto& val_chain : chains) {
    auto tv_chain = tv_iterable<std::deque<TensorView*>>(val_chain);
    TensorView* running_consumer = nullptr;
    TensorView* running_producer = tv_chain.back();
    unsigned int running_consumer_pos = consumer_position_;

    tv_chain.pop_back();

    while (!tv_chain.empty()) {
      running_consumer = running_producer;
      running_producer = tv_chain.back();
      tv_chain.pop_back();

      if (compute_at_ed.find(running_producer) != compute_at_ed.end() &&
          known_positions.find(running_producer) != known_positions.end()) {
        running_consumer_pos = known_positions.at(running_producer);
        continue;
      }

      computeAt_impl(running_producer, running_consumer, running_consumer_pos);
      running_consumer_pos = running_producer->getThisComputeAtAxis();

      // Update both compute_at_ed and compute_at_axis_lookup
      compute_at_ed.emplace(running_producer);

      if (known_positions.find(running_producer) != known_positions.end()) {
        TORCH_INTERNAL_ASSERT(
            known_positions.at(running_producer) ==
                running_producer->getThisComputeAtAxis(),
            "Hit a logical inconsistency in the computeAt pass.");
      } else {
        known_positions[running_producer] =
            running_producer->getThisComputeAtAxis();
      }
    }
  }

  // propagate forward through all consumer use_chains or from consumer to
  // common_consumer if common_consumer exists, mark as finished.

  if (common_consumer_ == nullptr) {
    chains = DependencyCheck::getAllUseChains(consumer_);
  } else if (common_consumer_ != consumer_) {
    chains =
        DependencyCheck::getAllDependencyChains(consumer_, common_consumer_);
  }

  // propagate forward through all chains
  unsigned int running_producer_compute_at = consumer_position_;

  for (const auto& dep_chain : chains) {
    TORCH_INTERNAL_ASSERT(
        !dep_chain.empty(), "Computed an invalid common_consumer.");

    std::deque<TensorView*> tv_dep_chain =
        tv_iterable<std::deque<TensorView*>>(dep_chain);

    TensorView* running_consumer = tv_dep_chain.front();
    tv_dep_chain.pop_front();

    TensorView* running_producer = nullptr;

    while (!tv_dep_chain.empty()) {
      running_producer = running_consumer;
      running_consumer = tv_dep_chain.front();
      tv_dep_chain.pop_front();

      if (compute_at_ed.find(running_producer) != compute_at_ed.end() &&
          known_positions.find(running_consumer) != known_positions.end()) {
        running_producer_compute_at = known_positions.at(running_consumer);
        continue;
      }

      forwardComputeAt_impl(
          running_producer, running_consumer, running_producer_compute_at);

      compute_at_ed.emplace(running_producer);

      if (known_positions.find(running_consumer) != known_positions.end()) {
        TORCH_INTERNAL_ASSERT(
            known_positions.at(running_consumer) ==
                running_producer->getRelativeComputeAtAxis(),
            "Hit a logical inconsistency in computeAt pass.");
      } else {
        known_positions[running_consumer] =
            running_producer->getRelativeComputeAtAxis();
      }
    }
  }
}

// Similar to forward traversal in traverseAllKnown but we don't know if the
// positions are actually correct
void ComputeAt::traverseForward() {
  // propagate forward through all *producer* use_chains or from *producer* to
  // common_consumer if common_consumer exists.
  std::deque<std::deque<Val*>> chains;
  if (common_consumer_ == nullptr) {
    chains = DependencyCheck::getAllUseChains(producer_);
  } else if (common_consumer_ != consumer_) {
    chains =
        DependencyCheck::getAllDependencyChains(producer_, common_consumer_);
  }

  // propagate forward through all chains
  for (const auto& dep_chain : chains) {
    int running_producer_compute_at = known_positions.at(producer_);
    TORCH_INTERNAL_ASSERT(
        !dep_chain.empty(), "Computed an invalid common_consumer.");

    std::deque<TensorView*> tv_dep_chain =
        tv_iterable<std::deque<TensorView*>>(dep_chain);

    TensorView* running_consumer = tv_dep_chain.front();
    tv_dep_chain.pop_front();

    TensorView* running_producer = nullptr;

    while (!tv_dep_chain.empty()) {
      running_producer = running_consumer;
      running_consumer = tv_dep_chain.front();
      tv_dep_chain.pop_front();

      if (compute_at_ed.find(running_producer) != compute_at_ed.end() &&
          known_positions.find(running_consumer) != known_positions.end()) {
        running_producer_compute_at = known_positions.at(running_consumer);
        continue;
      }

      forwardComputeAt_impl(
          running_producer, running_consumer, running_producer_compute_at);

      compute_at_ed.emplace(running_producer);

      if (known_positions.find(running_consumer) != known_positions.end()) {
        TORCH_INTERNAL_ASSERT(
            known_positions.at(running_consumer) ==
                running_producer->getRelativeComputeAtAxis(),
            "Hit a logical inconsistency in computeAt pass.");
      }
    }
  }
}

// Similar to backward traversal in traverseAllKnown but we should only apply
// computeAt if it will increase computeAt positions.
void ComputeAt::traverseBackward() {
  // propagate *backward* through all *producer* use_chains or from *producer*
  // to common_consumer if common_consumer exists. Only apply transform if
  // increases computeAt position.
  std::deque<std::deque<Val*>> chains;
  if (common_consumer_ == nullptr) {
    chains = DependencyCheck::getAllUseChains(producer_);
  } else if (common_consumer_ != consumer_) {
    chains =
        DependencyCheck::getAllDependencyChains(producer_, common_consumer_);
  }

  for (const auto& val_chain : chains) {
    auto tv_chain = tv_iterable<std::deque<TensorView*>>(val_chain);
    TensorView* running_consumer = nullptr;
    TensorView* running_producer = tv_chain.back();
    auto it = known_positions.find(running_producer);

    if (it == known_positions.end()) {
      TORCH_INTERNAL_ASSERT(
          common_consumer_ == nullptr,
          "Hit a logical inconsistency in computeAt pass.");
      continue;
    }

    unsigned int running_consumer_pos = it->second;

    tv_chain.pop_back();

    while (!tv_chain.empty()) {
      running_consumer = running_producer;
      running_producer = tv_chain.back();
      tv_chain.pop_back();

      if (compute_at_ed.find(running_producer) != compute_at_ed.end() &&
          known_positions.find(running_producer) != known_positions.end()) {
        running_consumer_pos = known_positions.at(running_producer);
        continue;
      }

      // If we're already at consumer_position_ that's the max position we could
      // hope for, don't bother running again.
      if (running_producer->getThisComputeAtAxis() != consumer_position_) {
        maybe_computeAt_impl(
            running_producer, running_consumer, running_consumer_pos);
      }
      running_consumer_pos = running_producer->getThisComputeAtAxis();

      if (known_positions.find(running_producer) != known_positions.end()) {
        TORCH_INTERNAL_ASSERT(
            known_positions.at(running_producer) ==
                running_producer->getThisComputeAtAxis(),
            "Hit a logical inconsistency in the computeAt pass.");
      }
    }
  }
}

void ComputeAt::runPass() {
  // Make sure the correct fusion is setup between this and consumer.
  TORCH_CHECK(
      producer_->fusion() == consumer_->fusion(),
      producer_,
      " and ",
      consumer_,
      " are not in the same fusion.");

  // Make sure Fusion Guard is set appropriately
  FusionGuard fg(producer_->fusion());

  // Look through all the use chains of producer. Check if there's a single
  // consumer for all chains at or after the consumer specified in the computeAt
  // call.
  setCommonConsumer();

  // Propagate in a way we know result will be correct, which is forward from
  // consumer and backward from consumer to producer
  traverseAllKnown();

  TORCH_INTERNAL_ASSERT(
      producer_->hasComputeAt(),
      "Hit a logical inconsistency in the computeAt pass.");

  // Start at producer and traverse forward
  traverseForward();

  // Propagate backward from consumer or common consumer, check if it increase
  // computeAt position on tensors, if so take it!
  traverseBackward();
}

void ComputeAt::setupOutputs() {
  if (common_consumer_ != nullptr)
    return;

  // output and its compute at position
  std::unordered_map<TensorView*, int> touched_outputs;
  for (auto tv : compute_at_ed) {
    TORCH_INTERNAL_ASSERT(
        tv->hasComputeAt(),
        "Hit a logical inconsistency in the computeAt pass.");
    auto ca_view = tv->getComputeAtView();
    if (FusionGuard::getCurFusion()->hasOutput(ca_view)) {
      touched_outputs[ca_view] = tv->getRelativeComputeAtAxis();
    }
  }

  std::vector<TensorView*> touched_output_order(touched_outputs.size());

  {
    size_t i = 0;
    for (auto out : FusionGuard::getCurFusion()->outputs()) {
      if (out->getValType() == ValType::TensorView) {
        if (touched_outputs.find(out->as<TensorView>()) !=
            touched_outputs.end()) {
          touched_output_order[i++] = out->as<TensorView>();
        }
      }
    }
    TORCH_INTERNAL_ASSERT(
        i == touched_output_order.size(),
        "Hit a logical inconsistency in the computeAt pass.");
  }

  for (size_t i = 0; i < touched_output_order.size() - 1; i++) {
    touched_output_order[i]->setComputeAt(
        touched_output_order[i + 1],
        touched_outputs.at(touched_output_order[i]),
        touched_outputs.at(touched_output_order[i + 1]));
  }
}

ComputeAt::ComputeAt(
    TensorView* _producer,
    TensorView* _consumer,
    unsigned int _consumer_position)
    : producer_(_producer),
      consumer_(_consumer),
      consumer_position_(_consumer_position) {}

void ComputeAt::run(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_position) {
  ComputeAt ca(producer, consumer, consumer_position);
  ca.producer_use_chains_ = getAllTVUseChains(ca.producer_);
  ca.setCommonConsumer();
  ca.runPass();
  ca.setupOutputs();
}

} // namespace fuser
} // namespace jit
} // namespace torch
