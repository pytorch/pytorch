#include <torch/csrc/jit/codegen/cuda/inline_propagator.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <utility>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

bool InlinePropagatorSelector::allowPasC(TensorView* from, TensorView* to) {
  return selected_.count(to) > 0;
}

bool InlinePropagatorSelector::allowCasP(TensorView* from, TensorView* to) {
  // If the producer is in the selected set, then the consumer must also be
  // replayed to obtain a compatible loop structure so that this producer
  // can be consumed in this loop.
  return selected_.count(from) > 0 || selected_.count(to) > 0;
}

bool InlinePropagatorSelector::allowSibling(TensorView* from, TensorView* to) {
  return true;
}

MaxPosCalculator::MaxPosCalculator(ComputeAtMode mode) : mode_(mode) {
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
    bool allow_reduction,
    bool allow_vectorize,
    bool allow_unmappable) const {
  bool allowed = true;

  if (!allow_reduction) {
    allowed = allowed && !id->isReduction();
  }

  if (!allow_vectorize) {
    bool is_vectorize = isParallelTypeVectorize(id->getParallelType()) ||
        ((mode_ == ComputeAtMode::BestEffort ||
          mode_ == ComputeAtMode::MostInlined) &&
         id->getParallelType() == ParallelType::Unroll);
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
    bool allow_reduction,
    bool allow_vectorize,
    bool allow_unmappable) const {
  auto dom = tv->domain()->domain();
  auto iter = std::find_if(dom.begin(), dom.end(), [=](IterDomain* id) {
    return !isAllowedID(
        id, tv, allow_reduction, allow_vectorize, allow_unmappable);
  });
  return std::distance(dom.begin(), iter);
}

// Return the max position in consumer that producer can be inlined to
// Cannot inline:
//   Reduction dimensions in producer
//   Block broadcast dimensions in producer
//   Vectorized dimensions in producer or consumer
//   Unrolled dimensions in producer or consumer
//   Dimensions derived from root dimensions that exist in both but are
//   unmappable
size_t MaxPosCalculator::getMaxPosPasC(
    TensorView* producer,
    TensorView* consumer) const {
  // Limit max position based on vectorized dims in consumer.
  auto max_consumer_pos = getMaxPosSelf(consumer, true, false, true);

  auto pairwise_root_map = PairwiseRootDomainMap(producer, consumer);
  auto replay_PasC =
      BestEffortReplay::replayPasC(producer, consumer, -1, pairwise_root_map);
  auto c2p_replay_map = replay_PasC.getReplay();

  for (size_t consumer_pos = max_consumer_pos; consumer_pos > 0;
       consumer_pos--) {
    auto map_it = c2p_replay_map.find(consumer->axis((int)consumer_pos - 1));
    if (map_it != c2p_replay_map.end()) {
      auto p_id = map_it->second;
      if (!isAllowedID(p_id, producer, true, false, false)) {
        max_consumer_pos = consumer_pos - 1;
      }
    }
  }

  return max_consumer_pos;
}

// Return the max position in producer that can be inlined to consumer
// Cannot inline:
//   Reduction dimensions in producer
//   Vectorized dimensions in producer or consumer
//   Unrolled dimensions in producer or consumer
//   Dimensions derived from root dimensions that exist in both but are
//   unmappable
size_t MaxPosCalculator::getMaxPosCasP(
    TensorView* consumer,
    TensorView* producer) const {
  auto max_producer_pos = getMaxPosSelf(producer, false, false, false);

  auto pairwise_root_map = PairwiseRootDomainMap(producer, consumer);
  auto replay_CasP =
      BestEffortReplay::replayCasP(consumer, producer, -1, pairwise_root_map);
  auto p2c_replay_map = replay_CasP.getReplay();

  for (size_t producer_pos = max_producer_pos; producer_pos > 0;
       producer_pos--) {
    auto map_it = p2c_replay_map.find(producer->axis((int)producer_pos - 1));
    if (map_it != p2c_replay_map.end()) {
      auto c_id = map_it->second;
      if (!isAllowedID(c_id, consumer, true, false, true)) {
        max_producer_pos = producer_pos - 1;
      }
    }
  }

  return max_producer_pos;
}

size_t InlinePropagator::getMaxPosAll(TensorView* tv) {
  auto max_pos = max_pos_calc.getMaxPosSelf(tv, false, false, false);
  for (auto consumer_tv : ir_utils::consumerTvsOf(tv)) {
    // consumers are always replayed consistently
    max_pos =
        std::min<size_t>(max_pos, max_pos_calc.getMaxPosCasP(consumer_tv, tv));
  }
  return max_pos;
}

size_t InlinePropagator::adjustComputeAtPos(TensorView* tv, size_t pos) {
  pos = std::min<size_t>(pos, getMaxPosAll(tv));

  // hoist inner most broadcast
  while (pos > 0 && tv->axis(pos - 1)->isBroadcast()) {
    pos--;
  }

  return pos;
}

size_t InlinePropagator::getReplayPosPasC(
    TensorView* producer,
    TensorView* consumer) {
  size_t max_pos = max_pos_calc.getMaxPosPasC(producer, consumer);
  size_t pos = retrieveReplayedPos(consumer);

  if (mode_ == ComputeAtMode::BestEffort) {
    return std::min(pos, max_pos);
  } else if (mode_ == ComputeAtMode::MostInlined) {
    return max_pos;
  }

  TORCH_INTERNAL_ASSERT(
      pos <= max_pos,
      "Invalid compute at position detected in compute at when trying to replay producer: ",
      producer,
      " as consumer: ",
      consumer,
      " tried to do this at position: ",
      pos,
      " but max position that's allowed is ",
      max_pos);
  return pos;
}

size_t InlinePropagator::getReplayPosCasP(
    TensorView* consumer,
    TensorView* producer) {
  size_t max_pos = max_pos_calc.getMaxPosCasP(consumer, producer);
  size_t pos = retrieveReplayedPos(producer);

  if (mode_ == ComputeAtMode::BestEffort) {
    return std::min(pos, max_pos);
  } else if (mode_ == ComputeAtMode::MostInlined) {
    return max_pos;
  }

  TORCH_INTERNAL_ASSERT(
      pos <= max_pos,
      "Invalid compute at position detected in compute at when trying to replay consumer: ",
      consumer,
      " as producer: ",
      producer,
      " tried to do this at position: ",
      pos,
      " but max position that's allowed is ",
      max_pos);
  return pos;
}

void InlinePropagator::recordReplayedPos(TensorView* tv, size_t pos) {
  if (selected_.count(tv)) {
    auto new_pos = adjustComputeAtPos(tv, pos);
    if (pos != new_pos) {
      replayed_pos_[tv] = pos;
      pos = new_pos;
    }
    if (!tv->isFusionInput()) {
      tv->setComputeAt(pos);
    } else {
      replayed_pos_[tv] = pos;
    }
  } else {
    replayed_pos_[tv] = pos;
  }
}

size_t InlinePropagator::retrieveReplayedPos(TensorView* tv) {
  auto it = replayed_pos_.find(tv);
  if (it != replayed_pos_.end()) {
    return it->second;
  }
  return tv->getComputeAtPosition();
}

InlinePropagator::InlinePropagator(
    std::unordered_set<TensorView*> selected,
    TensorView* reference,
    int64_t reference_pos,
    ComputeAtMode mode)
    : max_pos_calc(mode),
      selected_(std::move(selected)),
      reference_(reference),
      reference_pos_(reference_pos),
      mode_(mode) {
  if (reference_pos < 0) {
    reference_pos += int64_t(reference->nDims()) + 1;
  }
  TORCH_INTERNAL_ASSERT(
      reference_pos >= 0 && reference_pos <= reference->nDims(),
      "Invalid computeAt axis, received ",
      reference_pos,
      " but should be > -",
      reference->nDims(),
      " and <= ",
      reference->nDims(),
      ".");
}

namespace {

// Make sure if tv is set to new_td it doesn't violate set compute at and max
// produce at positions.
bool validateDomain(TensorView* tv, TensorDomain* new_td) {
  auto first_mismatch =
      BestEffortReplay::findFirstMismatchedID(tv->domain(), new_td);
  return first_mismatch >= (int)tv->getMaxProducerPosition() &&
      first_mismatch >= (int)tv->getComputeAtPosition();
}

} // namespace

void InlinePropagator::propagateTvPasC(TensorView* from, TensorView* to) {
  if (is_first_) {
    is_first_ = false;
    recordReplayedPos(reference_, reference_pos_);
  }
  int pos = getReplayPosPasC(to, from);
  auto to_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(to, from, pos);
  // TODO: Can we make TransformPropagator do the transformation, and
  // InlinePropagator only set the CA positions?
  //   TORCH_CHECK(to_pos >= 0);
  if (to_pos < 0) {
    auto replay = TransformReplay::replayPasC(to, from, pos);
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay.first);
    to_pos = replay.second;
  }
  recordReplayedPos(to, to_pos);
}

void InlinePropagator::propagateTvCasP(TensorView* from, TensorView* to) {
  if (is_first_) {
    is_first_ = false;
    recordReplayedPos(reference_, reference_pos_);
  }
  int pos = getReplayPosCasP(to, from);
  auto to_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(to, from, pos);
  // TODO: Can we make TransformPropagator do the transformation, and
  // InlinePropagator only set the CA positions?
  //   TORCH_CHECK(to_pos >= 0);
  if (to_pos < 0) {
    auto replay = TransformReplay::replayCasP(to, from, pos);
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay.first);
    to_pos = replay.second;
  }
  recordReplayedPos(to, to_pos);
}

void InlinePropagator::propagateTvSibling(TensorView* from, TensorView* to) {
  if (is_first_) {
    is_first_ = false;
    recordReplayedPos(reference_, reference_pos_);
  }
  auto from_pos = retrieveReplayedPos(from);
  if (!TransformReplay::fullSelfMatching(to, from)) {
    auto replay = TransformReplay::fullSelfReplay(to->domain(), from->domain());
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay),
        "Tried to set the domain of ",
        to,
        " to ",
        replay,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay);
  }
  recordReplayedPos(to, from_pos);
}

// Try to find the aligned position on consumer's domain corresponding to the
// compute at position of producer domain.
void MaxProducerPosUpdater::handle(TensorView* consumer) {
  unsigned int consumer_pos = consumer->nDims();
  while (consumer_pos > 0) {
    for (auto producer : ir_utils::producerTvsOf(consumer)) {
      auto producer_pos = TransformReplay::getMatchedLeafPosWithoutReplayPasC(
          producer, consumer, consumer_pos);
      if (producer_pos >= 0 &&
          producer_pos <= producer->getComputeAtPosition()) {
        goto finished;
      }
    }
    consumer_pos--;
  }
finished:
  consumer->setMaxProducer(consumer_pos, true);
}

void MaxProducerPosUpdater::propagateTvPasC(TensorView* from, TensorView* to) {
  if (updated_.empty()) {
    // handle the reference tensor
    updated_.insert(nullptr);
    propagateTvPasC(nullptr, from);
  }
  for (auto consumer_tv : ir_utils::consumerTvsOf(to)) {
    if (updated_.count(consumer_tv) > 0) {
      continue;
    }
    handle(consumer_tv);
    updated_.insert(consumer_tv);
  }
}

void MaxProducerPosUpdater::propagateTvCasP(TensorView* from, TensorView* to) {
  propagateTvPasC(from, to);
}

void MaxProducerPosUpdater::propagateTvSibling(
    TensorView* from,
    TensorView* to) {
  propagateTvPasC(from, to);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
