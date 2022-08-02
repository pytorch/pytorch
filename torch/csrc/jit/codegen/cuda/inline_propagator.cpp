#include <torch/csrc/jit/codegen/cuda/inline_propagator.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <utility>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

bool InlinePropagatorSelector::allowC2P(TensorView* from, TensorView* to) {
  return selected_.count(to) > 0;
}

bool InlinePropagatorSelector::allowP2C(TensorView* from, TensorView* to) {
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
    // Avoid inlining if marked as Vectorize or Group. In the case of
    // BestEffort and MostInlined modes, avoid Unroll as well.
    bool is_vectorize = isParallelTypeVectorize(id->getParallelType()) ||
        id->getParallelType() == ParallelType::Group ||
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

// Return the max position in producer that can be inlined to consumer
// Cannot inline:
//   Vectorized dimensions in consumer
//   Unrolled dimensions in consumer
size_t MaxPosCalculator::getMaxProducerPosFromConsumer(
    TensorView* producer,
    TensorView* consumer) const {
  auto pairwise_root_map = PairwiseRootDomainMap(producer, consumer);
  auto replay_CasP =
      BestEffortReplay::replayCasP(consumer, producer, -1, pairwise_root_map);
  auto p2c_replay_map = replay_CasP.getReplay();

  for (size_t producer_pos = 0; producer_pos < producer->nDims();
       producer_pos++) {
    auto map_it = p2c_replay_map.find(producer->axis(producer_pos));
    if (map_it != p2c_replay_map.end()) {
      auto c_id = map_it->second;
      if (!isAllowedID(c_id, consumer, true, false, true)) {
        return producer_pos;
      }
    }
  }
  return producer->nDims();
}

size_t InlinePropagator::getMaxPosAll(TensorView* tv, bool check_siblings) {
  auto max_pos = max_pos_calc.getMaxPosSelf(tv, false, false, false);
  for (auto consumer_tv : ir_utils::consumerTvsOf(tv)) {
    max_pos = std::min<size_t>(
        max_pos, max_pos_calc.getMaxProducerPosFromConsumer(tv, consumer_tv));
  }
  if (check_siblings) {
    for (auto sibling_tv : ir_utils::siblingTvsOf(tv)) {
      max_pos = std::min<size_t>(max_pos, getMaxPosAll(sibling_tv, false));
    }
  }
  return max_pos;
}

void InlinePropagator::setCAPos(TensorView* tv) {
  size_t pos = mapped_reference_pos_.at(tv);
  if ((selected_.empty() || selected_.count(tv)) && !tv->isFusionInput()) {
    auto max_pos = getMaxPosAll(tv);
    if (mode_ == ComputeAtMode::Standard) {
      TORCH_INTERNAL_ASSERT(
          pos <= max_pos,
          "Invalid compute at position detected in InlinePropagator when trying to set the CA position of: ",
          tv,
          " to ",
          pos,
          ",  max position that's allowed is ",
          max_pos);
    } else {
      pos = std::min<size_t>(pos, max_pos);
    }
    // hoist inner most broadcast
    while (pos > 0 && tv->axis(pos - 1)->isBroadcast()) {
      pos--;
    }
    tv->setComputeAt(pos);
  }
}

InlinePropagator::InlinePropagator(
    TensorView* reference,
    int64_t reference_pos,
    ComputeAtMode mode,
    std::unordered_set<TensorView*> selected)
    : max_pos_calc(mode),
      selected_(std::move(selected)),
      reference_(reference),
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
  reference_pos_ = reference_pos;
}

void InlinePropagator::propagateC2P(TensorView* from, TensorView* to) {
  if (is_first_) {
    is_first_ = false;
    mapped_reference_pos_[reference_] = reference_pos_;
    setCAPos(reference_);
  }
  // Step 1: find mapped_reference_pos_[to]
  int from_pos;
  if (mode_ != ComputeAtMode::MostInlined) {
    from_pos = mapped_reference_pos_.at(from);
  } else {
    from_pos = from->nDims();
  }
  auto to_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(to, from, from_pos);
  TORCH_CHECK(
      to_pos >= 0,
      "Unable to propagate CA position from consumer ",
      from,
      " at ",
      from_pos,
      " to producer ",
      to,
      " because this would require replay.");
  mapped_reference_pos_[to] = to_pos;
  // Step 2: set CA position of `to`
  setCAPos(to);
}

void InlinePropagator::propagateP2C(TensorView* from, TensorView* to) {
  if (is_first_) {
    is_first_ = false;
    mapped_reference_pos_[reference_] = reference_pos_;
    setCAPos(reference_);
  }
  // Step 1: find mapped_reference_pos_[to]
  int from_pos;
  if (mode_ != ComputeAtMode::MostInlined) {
    from_pos = mapped_reference_pos_.at(from);
  } else {
    from_pos = from->nDims();
  }
  auto to_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(to, from, from_pos);
  TORCH_CHECK(
      to_pos >= 0,
      "Unable to propagate CA position from producer ",
      from,
      " at ",
      from_pos,
      " to consumer ",
      to,
      " because this would require replay.");
  mapped_reference_pos_[to] = to_pos;
  // Step 2: set CA position of `to`
  setCAPos(to);
}

void InlinePropagator::propagateSibling(TensorView* from, TensorView* to) {
  if (is_first_) {
    is_first_ = false;
    mapped_reference_pos_[reference_] = reference_pos_;
    setCAPos(reference_);
  }
  // Step 1: find mapped_reference_pos_[to]
  auto from_pos = mapped_reference_pos_.at(from);
  TORCH_CHECK(
      TransformReplay::fullSelfMatching(to, from),
      "Unable to propagate CA position from ",
      from,
      " to sibling ",
      to,
      " because this would require replay.");
  mapped_reference_pos_[to] = from_pos;
  // Step 2: set CA position of `to`
  setCAPos(to);
}

namespace {

// Try to find the aligned position on consumer's domain corresponding to the
//  compute at position of producer domain. Used in computeAt pass only. No
//  checking on actual producer-consumer relationship.
unsigned int getConsumerPosAlignedToProducerCA(
    TensorView* consumer,
    TensorView* producer) {
  // Locate consumer's position that aligns with
  //  the producer's new compute at axis. We need broadcast axes forwarded so we
  //  need to replay PasC as CasP will not forward braodcast dims. For example
  //  if we have:
  // T2[ iS22{( 3 * 1 )} ] ca_pos( 1 ) = broadcast( T1[ iS1{3} ] ca_pos( 1 )
  // produce_pos( 1) ) CasP will have the mapping iS1{3} -> iS2{3} and PasC will
  // have the mapping iS22{( 3 * 1 )} <- iS1{3} We need the latter. Refer to
  // NVFuserTest.FusionComplexBCast1_CUDA

  auto c2p_map =
      BestEffortReplay::replayPasC(
          producer,
          consumer,
          -1,
          // Compute at root domain may not be valid here, as all
          // producers don't have to be able to map into consumer at
          // max producer position. Since computeAt should be valid
          // and this mechanism is only intended to lower produce
          // position of consumer, we can simply use the pairwise map.
          PairwiseRootDomainMap(producer, consumer))
          .getReplay();

  // Find the innermost position of consumer that has
  //  been mapped within the producer ca axis.
  unsigned int consumer_pos = consumer->nDims();
  while (consumer_pos > 0) {
    auto consumer_id = consumer->axis((int)consumer_pos - 1);
    auto p_dom = producer->domain()->domain();
    if (std::any_of(
            p_dom.begin(),
            p_dom.begin() + producer->getComputeAtPosition(),
            [&consumer_id, &c2p_map](IterDomain* p_id) {
              auto c_id_it = c2p_map.find(consumer_id);
              if (c_id_it != c2p_map.end()) {
                return c_id_it->second == p_id;
              }
              return false;
            })) {
      break;
    }
    consumer_pos--;
  }

  return consumer_pos;
}

} // namespace

// Try to find the aligned position on consumer's domain corresponding to the
// compute at position of producer domain.
void MaxProducerPosUpdater::handle(TensorView* consumer) {
  unsigned int consumer_pos = 0;
  for (auto producer : ir_utils::producerTvsOf(consumer)) {
    consumer_pos = std::max(
        consumer_pos, getConsumerPosAlignedToProducerCA(consumer, producer));
  }
  consumer->setMaxProducer(consumer_pos);
}

void MaxProducerPosUpdater::propagateC2P(TensorView* from, TensorView* to) {
  if (updated_.empty()) {
    // handle the reference tensor
    updated_.insert(nullptr);
    propagateC2P(nullptr, from);
  }
  for (auto consumer_tv : ir_utils::consumerTvsOf(to)) {
    if (updated_.count(consumer_tv) > 0) {
      continue;
    }
    handle(consumer_tv);
    updated_.insert(consumer_tv);
  }
}

void MaxProducerPosUpdater::propagateP2C(TensorView* from, TensorView* to) {
  propagateC2P(from, to);
}

void MaxProducerPosUpdater::propagateSibling(TensorView* from, TensorView* to) {
  propagateC2P(from, to);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
