#include <torch/csrc/jit/codegen/cuda/compute_at.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <c10/util/irange.h>

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

bool validateDomain(TensorView* tv, TensorDomain* new_td) {
  auto first_mismatch =
      BestEffortReplay::findFirstMismatchedID(tv->domain(), new_td);
  return first_mismatch >= (int)tv->getMaxProducerPosition() &&
      first_mismatch >= (int)tv->getComputeAtPosition();
}

// Return the max position in consumer that producer can be inlined to
// Cannot inline:
//   Reduction dimensions in producer
//   Block broadcast dimensions in producer
//   Vectorized dimensions in producer or consumer
//   Unrolled dimensions in producer or consumer
//   Dimensions derived from root dimensions that exist in both but are
//   unmappable
unsigned int getReplayablePosPasC(
    TensorView* producer,
    TensorView* consumer,
    const std::unordered_set<IterDomain*>& unmappable_producer_dims,
    ComputeAtMode mode) {
  // Check if any consumer dimensions are marked as vectorize as producer can
  // not be inlined to vectorized dimensions in consumer.
  auto c_dom = consumer->domain()->domain();
  auto vector_dim_it =
      std::find_if(c_dom.begin(), c_dom.end(), [&mode](IterDomain* id) {
        return isParallelTypeVectorize(id->getParallelType()) ||
            ((mode == ComputeAtMode::BestEffort ||
              mode == ComputeAtMode::MostInlined) &&
             id->getParallelType() == ParallelType::Unroll);
      });

  // Limit max position based on vectorized dims in consumer.
  auto max_consumer_pos = std::distance(c_dom.begin(), vector_dim_it);

  auto pairwise_root_map = PairwiseRootDomainMap(producer, consumer);
  auto c2p_root_map =
      PairwiseRootDomainMap(producer, consumer)
          .mapConsumerToProducer(consumer->domain(), producer->domain());

  auto replay_PasC =
      BestEffortReplay::replayPasC(producer, consumer, -1, pairwise_root_map);

  // Look for id's that map to a consumer id that's vectorized
  auto c2p_replay_map = replay_PasC.getReplay();

  for (size_t consumer_pos = max_consumer_pos; consumer_pos > 0;
       consumer_pos--) {
    auto map_it = c2p_replay_map.find(consumer->axis((int)consumer_pos - 1));
    if (map_it != c2p_replay_map.end()) {
      auto p_id = map_it->second;
      // If we find a consumer dim that maps to a producer dim that's
      // vectorized or unrolled limit max compute at by it.
      if (isParallelTypeVectorize(p_id->getParallelType()) ||
          ((mode == ComputeAtMode::BestEffort ||
            mode == ComputeAtMode::MostInlined) &&
           p_id->getParallelType() == ParallelType::Unroll)) {
        max_consumer_pos = consumer_pos - 1;
      }
    }
  }

  // Start at max position and work backwards,  try to find a location where
  // producer can be inlined.
  for (size_t consumer_pos = max_consumer_pos; consumer_pos > 0;
       consumer_pos--) {
    // Grab all root dimensions of consumer as roots must be used to understand
    // inlining potential.
    auto consumer_root_dim_vals =
        IterVisitor::getInputsTo({c_dom.begin(), c_dom.begin() + consumer_pos});
    // convert to iter domains
    auto consumer_root_dim_ids =
        ir_utils::filterByType<IterDomain>(consumer_root_dim_vals);
    // If any root dimensions cannot be mapped to producer we can't inline. If
    // any root dimension
    if (std::any_of(
            consumer_root_dim_ids.begin(),
            consumer_root_dim_ids.end(),
            [&unmappable_producer_dims, &c2p_root_map](IterDomain* c_root_id) {
              auto p_root_id_it = c2p_root_map.find(c_root_id);
              if (p_root_id_it == c2p_root_map.end()) {
                return false;
              }
              auto p_id = p_root_id_it->second;
              return unmappable_producer_dims.find(p_id) !=
                  unmappable_producer_dims.end();
            })) {
      continue;
    }
    return consumer_pos;
  }

  return 0;
}

// Return the max position in producer that can be inlined to consumer
// Cannot inline:
//   Reduction dimensions in producer
//   Vectorized dimensions in producer or consumer
//   Unrolled dimensions in producer or consumer
//   Dimensions derived from root dimensions that exist in both but are
//   unmappable
unsigned int getReplayablePosCasP(
    TensorView* consumer,
    TensorView* producer,
    const std::unordered_set<IterDomain*>& unmappable_producer_dims,
    ComputeAtMode mode) {
  auto p_dom = producer->domain()->domain();
  auto first_reduction =
      std::find_if(p_dom.begin(), p_dom.end(), [](IterDomain* id) {
        return id->isReduction();
      });

  auto first_vectorized_axis =
      std::find_if(p_dom.begin(), first_reduction, [&mode](IterDomain* id) {
        return isParallelTypeVectorize(id->getParallelType()) ||
            ((mode == ComputeAtMode::BestEffort ||
              mode == ComputeAtMode::MostInlined) &&
             id->getParallelType() == ParallelType::Unroll);
      });

  auto max_producer_pos = std::distance(p_dom.begin(), first_vectorized_axis);

  auto pairwise_root_map = PairwiseRootDomainMap(producer, consumer);
  auto p2c_root_map = pairwise_root_map.mapProducerToConsumer(
      producer->domain(), consumer->domain());

  auto replay_CasP =
      BestEffortReplay::replayCasP(consumer, producer, -1, pairwise_root_map);

  // Look for id's that map to a consumer id that's vectorized
  auto p2c_replay_map = replay_CasP.getReplay();

  for (size_t producer_pos = max_producer_pos; producer_pos > 0;
       producer_pos--) {
    auto map_it = p2c_replay_map.find(producer->axis((int)producer_pos - 1));
    if (map_it != p2c_replay_map.end()) {
      auto c_id = map_it->second;
      // If we find a producer dim that maps to a consumer vectorized or
      // unrolled dim, limit max compute at by it
      if (isParallelTypeVectorize(c_id->getParallelType()) ||
          ((mode == ComputeAtMode::BestEffort ||
            mode == ComputeAtMode::MostInlined) &&
           c_id->getParallelType() == ParallelType::Unroll)) {
        max_producer_pos = producer_pos - 1;
      }
    }
  }

  for (size_t producer_pos = max_producer_pos; producer_pos > 0;
       producer_pos--) {
    auto all_vals = DependencyCheck::getAllValsBetween(
        {producer->getMaybeRFactorDomain().begin(),
         producer->getMaybeRFactorDomain().end()},
        {p_dom.begin(), p_dom.begin() + producer_pos});

    // If any root dims could have mapped to consumer, but don't, then we can't
    // compute at this point
    if (std::any_of(
            producer->getMaybeRFactorDomain().begin(),
            producer->getMaybeRFactorDomain().end(),
            [&unmappable_producer_dims, &all_vals](IterDomain* p_root_id) {
              return std::find(all_vals.begin(), all_vals.end(), p_root_id) !=
                  all_vals.end() &&
                  unmappable_producer_dims.find(p_root_id) !=
                  unmappable_producer_dims.end();
            })) {
      continue;
    }

    return producer_pos;
  }
  return 0;
}

unsigned int getInnermostNonBroadcastIdFrom(TensorView* tv) {
  unsigned int ret = tv->getComputeAtPosition();

  // Still assuming we only have block broadcast for now.
  //  This part may change
  while (ret > 0 && tv->axis((int)ret - 1)->isBroadcast()) {
    ret--;
  }

  return ret;
}

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

namespace {

// Checks if producer and consumer are transformed consistently so that to
// satisfy the provided compute at position. This means no replay is actually
// necessary for the compute at requested. If consumer_pos then
// consumer_or_producer_pos is relative to the consumer and skipReplay returns
// the associated position in producer.
//
// If producer and consumer are not transformed consistently with provided
// postition, returns -1.
int skipReplay(
    const TensorView* producer,
    const TensorView* consumer,
    int consumer_or_producer_pos,
    bool consumer_pos = true) {
  FUSER_PERF_SCOPE("transform_replay.cpp::skipReplay");

  const auto c2p_root_map =
      PairwiseRootDomainMap(producer, consumer)
          .mapConsumerToProducer(consumer->domain(), producer->domain());

  // IterDomains in consumer root also in producer root
  std::unordered_set<Val*> mapped_consumer_roots;
  for (auto entry : c2p_root_map) {
    mapped_consumer_roots.emplace(entry.first);
  }

  const auto consumer_domain = consumer->domain()->domain();

  auto mapped_consumer_domain_ids_vec = DependencyCheck::getAllValsBetween(
      mapped_consumer_roots, {consumer_domain.begin(), consumer_domain.end()});

  std::unordered_set<Val*> mapped_consumer_domain_ids(
      mapped_consumer_domain_ids_vec.begin(),
      mapped_consumer_domain_ids_vec.end());

  const auto producer_domain = producer->domain()->domain();

  auto it_consumer = consumer_domain.begin();
  auto it_producer = producer_domain.begin();

  auto best_effort_PasC = BestEffortReplay::replayPasC(
      producer, consumer, -1, PairwiseRootDomainMap(producer, consumer));

  auto c2p_map = best_effort_PasC.getReplay();

  int mismatched_consumer_pos = 0;
  int mismatched_producer_pos = 0;
  while (it_consumer != consumer_domain.end()) {
    auto consumer_id = *it_consumer;
    if (!mapped_consumer_domain_ids.count(consumer_id)) {
      ++it_consumer;
      mismatched_consumer_pos++;
      continue;
    }

    auto c2p_it = c2p_map.find(consumer_id);
    if (c2p_it == c2p_map.end()) {
      break;
    }

    if (it_producer == producer_domain.end()) {
      break;
    }

    auto producer_id = *it_producer;

    if (c2p_it->second == producer_id) {
      ++mismatched_consumer_pos;
      ++mismatched_producer_pos;
      ++it_consumer;
      ++it_producer;
      if (consumer_pos) {
        if (consumer_or_producer_pos == mismatched_consumer_pos) {
          return mismatched_producer_pos;
        }
      } else {
        if (consumer_or_producer_pos == mismatched_producer_pos) {
          return mismatched_consumer_pos;
        }
      }
    } else {
      break;
    }
  }
  return -1;
}

} // namespace

// Actually applies transformation
unsigned int ComputeAt::backwardComputeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int consumer_compute_at_pos) {
  FUSER_PERF_SCOPE("backwardComputeAt_impl");

  auto max_consumer_compute_at_pos =
      getReplayablePosPasC(producer, consumer, unmappable_dims_, mode_);

  if (mode_ == ComputeAtMode::BestEffort) {
    consumer_compute_at_pos =
        std::min(consumer_compute_at_pos, max_consumer_compute_at_pos);
  } else if (mode_ == ComputeAtMode::MostInlined) {
    consumer_compute_at_pos = max_consumer_compute_at_pos;
  } else {
    TORCH_INTERNAL_ASSERT(
        consumer_compute_at_pos <= max_consumer_compute_at_pos,
        "Invalid compute at position detected in compute at when trying to replay producer: ",
        producer,
        " as consumer: ",
        consumer,
        " tried to do this at position: ",
        consumer_compute_at_pos,
        " but max position that's allowed is ",
        max_consumer_compute_at_pos);
  }

  // Short cut if no replay is necessary
  auto maybe_producer_pos =
      skipReplay(producer, consumer, (int)consumer_compute_at_pos, true);
  if (maybe_producer_pos >= 0) {
    if (!producer->isFusionInput()) {
      producer->setComputeAt((unsigned int)maybe_producer_pos);
    }
    consumer->setMaxProducer(consumer_compute_at_pos);
    return (unsigned int)maybe_producer_pos;
  }

  auto replay_producer_pair = TransformReplay::replayPasC(
      producer, consumer, (int)consumer_compute_at_pos, root_map_);

  if (replay_producer_pair.second == 0) {
    return 0;
  }

  if (replay_producer_pair.second >= producer->getComputeAtPosition()) {
    const TensorDomain* current_domain = producer->domain();
    TensorDomain* new_domain = replay_producer_pair.first;

    TORCH_INTERNAL_ASSERT(
        validateDomain(producer, new_domain),
        "Tried to set the domain of ",
        producer,
        " to ",
        new_domain,
        " but that would invalidate previously compute at position or max producer position.");

    producer->setDomain(new_domain);
    if (!producer->isFusionInput()) {
      producer->setComputeAt(replay_producer_pair.second);
    }

    consumer->setMaxProducer(consumer_compute_at_pos);
    root_map_.setAlias(current_domain, new_domain);
  }

  return replay_producer_pair.second;
}

// Actually applies transformation, replay consumer based on producer, set
// compute at of producer, set pass position of consumer, return position
// relative to consumer
unsigned int ComputeAt::forwardComputeAt_impl(
    TensorView* producer,
    TensorView* consumer,
    unsigned int producer_compute_at_pos) {
  FUSER_PERF_SCOPE("forwardComputeAt_impl");

  auto max_producer_compute_at_pos =
      getReplayablePosCasP(consumer, producer, unmappable_dims_, mode_);

  if (mode_ == ComputeAtMode::BestEffort) {
    producer_compute_at_pos =
        std::min(producer_compute_at_pos, max_producer_compute_at_pos);
  } else if (mode_ == ComputeAtMode::MostInlined) {
    producer_compute_at_pos = max_producer_compute_at_pos;
  } else {
    TORCH_INTERNAL_ASSERT(
        producer_compute_at_pos <= max_producer_compute_at_pos,
        "Invalid compute at position detected in compute at when trying to replay consumer: ",
        consumer,
        " as producer: ",
        producer,
        " tried to do this at position: ",
        producer_compute_at_pos,
        " but max position that's allowed is ",
        max_producer_compute_at_pos);
  }

  // Short cut if no replay is necessary
  auto maybe_consumer_pos =
      skipReplay(producer, consumer, (int)producer_compute_at_pos, false);
  if (maybe_consumer_pos > -1) {
    if (!producer->isFusionInput()) {
      producer->setComputeAt(producer_compute_at_pos);
    }
    consumer->setMaxProducer((unsigned int)maybe_consumer_pos);
    return (unsigned int)maybe_consumer_pos;
  }

  auto replay_consumer_pair = TransformReplay::replayCasP(
      consumer, producer, (int)producer_compute_at_pos, root_map_);

  if (producer_compute_at_pos > producer->getComputeAtPosition()) {
    if (!producer->isFusionInput()) {
      producer->setComputeAt((int)producer_compute_at_pos);
    }
  }

  if (replay_consumer_pair.second > consumer->getMaxProducerPosition()) {
    const TensorDomain* current_domain = consumer->domain();
    TensorDomain* new_domain = replay_consumer_pair.first;

    TORCH_INTERNAL_ASSERT(
        validateDomain(consumer, new_domain),
        "Tried to set the domain of ",
        consumer,
        " to ",
        new_domain,
        " but that would invalidate previously compute at position or max producer position.");

    consumer->setDomain(new_domain);
    consumer->setMaxProducer(replay_consumer_pair.second);
    root_map_.setAlias(current_domain, new_domain);
  }

  return replay_consumer_pair.second;
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

void ComputeAt::resetMaxProducerPos(TensorView* consumer_tv) {
  if (consumer_tv->definition() == nullptr) {
    consumer_tv->setMaxProducer(0, true);
  }

  unsigned int new_consummer_pa_pos = 0;

  // Re-compute the max producer position as one or more
  //  of the producers of this consumer have updated their
  //  compute at position.
  for (auto inp : ir_utils::producerTvsOf(consumer_tv)) {
    if (!inp->isFusionInput()) {
      // Locate consumer's position that aligns with
      //  the producer's new compute at axis.
      unsigned int inp_ca_pos_to_consumer =
          getConsumerPosAlignedToProducerCA(consumer_tv, inp);

      // Populate the max consumer position required by
      //  producer compute at.
      new_consummer_pa_pos =
          std::max(new_consummer_pa_pos, inp_ca_pos_to_consumer);
    }
  }

  consumer_tv->setMaxProducer(new_consummer_pa_pos, true);
}

void ComputeAt::hoistInnermostBroadcast() {
  auto fusion = producer_->fusion();

  std::unordered_set<TensorView*> consumers_to_update;

  auto all_vals = fusion->usedMathVals();
  auto all_tvs = ir_utils::filterByType<TensorView>(all_vals);

  for (auto running_producer : all_tvs) {
    if (!running_producer->isFusionInput()) {
      auto producer_ca_pos = running_producer->getComputeAtPosition();
      // Find the innermost iterdomain that is not a broadcast
      auto new_ca_pos = getInnermostNonBroadcastIdFrom(running_producer);
      // Update the compute at pos of this producer if the original
      //  compute at is within inner most broadcast axes
      if (new_ca_pos < producer_ca_pos) {
        running_producer->setComputeAt(new_ca_pos, true);
      }
      // Mark all consumers of this producer for later produce
      //  position update.
      // This is safe with segmented fusion. TV uses will reset
      //  when FusionSegmentGuard try to change the IO.
      auto tv_consumers = ir_utils::consumerTvsOf(running_producer);
      consumers_to_update.insert(tv_consumers.begin(), tv_consumers.end());
    }
  }
}

void ComputeAt::updateSiblings() {
  // Track which consumers may have a wrong produce at position to update
  // later
  auto updateSiblingsOfTv = [&](TensorView* tv) {
    if (tv->definition() == nullptr) {
      return;
    }

    std::unordered_set<TensorView*> consumers_to_update;

    if (tv->definition()->outputs().size() > 1) {
      auto outs = tv->definition()->outputs();
      auto out_tvs = ir_utils::filterByType<TensorView>(outs);
      for (auto sibling_tv : out_tvs) {
        if (sibling_tv == tv) {
          continue;
        }

        std::unordered_map<IterDomain*, IterDomain*> tv_to_sibling_map;
        TORCH_INTERNAL_ASSERT(
            tv->getRootDomain().size() == sibling_tv->getRootDomain().size(),
            "Error replaying multiple output expressions in computeAt.");

        // Propagate any root parallelization as fullSelfReplay expects it.
        for (const auto i : c10::irange(sibling_tv->getRootDomain().size())) {
          auto id = tv->getRootDomain()[i];
          auto sibling_id = sibling_tv->getRootDomain()[i];
          if (id->getParallelType() != ParallelType::Serial &&
              sibling_id->getParallelType() == ParallelType::Serial) {
            sibling_id->parallelize(id->getParallelType());
          } else if (
              id->getParallelType() == ParallelType::Serial &&
              sibling_id->getParallelType() != ParallelType::Serial) {
            id->parallelize(sibling_id->getParallelType());
          }
        }
        auto sibling_domain =
            TransformReplay::fullSelfReplay(sibling_tv->domain(), tv->domain());
        validateDomain(sibling_tv, sibling_domain);
        sibling_tv->setDomain(sibling_domain);
        sibling_tv->setComputeAt(tv->getComputeAtPosition());
        sibling_tv->setMaxProducer(tv->getMaxProducerPosition());
        auto consumer_tvs = ir_utils::consumerTvsOf(sibling_tv);
        consumers_to_update.insert(consumer_tvs.begin(), consumer_tvs.end());
      }
    }

    // Update sibling consumer tv's max producer position
    for (auto consumer : consumers_to_update) {
      this->resetMaxProducerPos(consumer);
    }
  };

  // Find all tensor views that may have been modified
  auto chains = producer_use_chains_;
  if (common_consumer_ != nullptr) {
    chains = tvChains(
        DependencyCheck::getAllDependencyChains(producer_, common_consumer_));
  }

  std::unordered_set<TensorView*> participating_tvs;
  for (auto chain : chains) {
    participating_tvs.insert(chain.begin(), chain.end());
  }

  for (auto tv : participating_tvs) {
    updateSiblingsOfTv(tv);
  }
}

void ComputeAt::runPass() {
  FUSER_PERF_SCOPE("ComputeAt::runPass");

  // Traverse backward through all dep chains from producer to consumer
  traverseBackward();

  // Start at producer and traverse forward through all chains
  traverseForward();

  // Back off on inlining the inner broadcast axes
  hoistInnermostBroadcast();

  // Update siblings of multi output expressions
  updateSiblings();

  // Update the compute at position of all consumers, this used to be done
  // during the compute at pass itself, but its cleaner to do this as a cleanup
  // pass similar to hoistInnermostBroadcast and updateSiblings.
  std::unordered_set<TensorView*> all_consumers;

  // Find all tensor views that may have been modified
  auto chains = producer_use_chains_;
  if (common_consumer_ != nullptr) {
    chains = tvChains(
        DependencyCheck::getAllDependencyChains(producer_, common_consumer_));
  }

  for (const auto& chain : chains) {
    for (auto tv : chain) {
      all_consumers.emplace(tv);
    }
  }

  // Reset max producer position of all tensor views.
  for (auto tv : all_consumers) {
    resetMaxProducerPos(tv);
  }
}

void ComputeAt::buildUnmappableDims() {
  auto all_tvs = ir_utils::allTvs(producer_->fusion());
  for (auto tv : all_tvs) {
    auto consumers = ir_utils::consumerTvsOf(tv);
    for (auto consumer : consumers) {
      // Grab dimensions in producer and consumer that are mappable to eachother
      // based on the computeAtRootDomainMap. This will tell us which dimensions
      // can be inlined based on avoiding trying to inline reduction structures.
      auto mappable_roots =
          root_map_.getMappableDims(tv->domain(), consumer->domain());
      for (auto tv_root_id : tv->getMaybeRFactorDomain()) {
        if (mappable_roots.find(tv_root_id) == mappable_roots.end()) {
          unmappable_dims_.emplace(tv_root_id);
        }
      }
    }
  }
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

  buildUnmappableDims();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
