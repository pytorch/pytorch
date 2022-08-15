#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/maxinfo_propagator.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <deque>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

using id_map = std::unordered_map<IterDomain*, IterDomain*>;

namespace {

class ReplaySelf : public ReplayTransformations {
 private:
  // Took a good bit of this from ReplayTransformations::handle(Split...)
  void handle(Split* s) override {
    // Grab input to the split operation
    auto id_in = s->in();

    // Grab our mapping of that ID to the one we're replaying
    auto it = id_map_.find(id_in);

    // Make sure it exists in the map
    TORCH_INTERNAL_ASSERT(
        it != id_map_.end(),
        "Transform traversal failed, dependencies not met.");
    // Grab the ID we're going to replay on
    auto mapped = it->second;

    // This ID should be a leaf ID (meaning it has no uses we generated)
    TORCH_INTERNAL_ASSERT(
        leaf_ids_.find(mapped) != leaf_ids_.end(),
        "Transform traversal failed, modified a node but it was not a leaf node.");

    // outer loop size
    Val* remainder = ceilDiv(
        Split::extent(mapped->extent(), s->startOffset(), s->stopOffset()),
        s->factor());

    // Manually replay the split, following the output of the operations.
    // This is so rfactor ops are replayed correctly.
    IterDomain* ido =
        IterDomainBuilder(s->outer())
            .start(s->container()->zeroVal())
            .extent(s->innerSplit() ? remainder->as<Int>() : s->factor())
            .build();

    // inner IterDomain
    IterDomain* idi =
        IterDomainBuilder(s->inner())
            .start(s->container()->zeroVal())
            .extent(s->innerSplit() ? s->factor() : remainder->as<Int>())
            .build();

    // Generate the split node
    IrBuilder::create<Split>(
        s->container(),
        ido,
        idi,
        mapped,
        s->factor(),
        s->innerSplit(),
        s->startOffset(),
        s->stopOffset());

    // Remove mapped id from leaf IDs
    leaf_ids_.erase(mapped);

    // Add outputs to leaf IDs
    leaf_ids_[ido] = counter++;
    leaf_ids_[idi] = counter++;

    // Update our ID map to include these outputs
    id_map_[s->outer()] = ido;
    id_map_[s->inner()] = idi;
  }

  void handle(Merge* m) override {
    auto id_outer = m->outer();
    auto id_inner = m->inner();

    auto it_outer = id_map_.find(id_outer);
    auto it_inner = id_map_.find(id_inner);

    TORCH_INTERNAL_ASSERT(
        it_outer != id_map_.end() && it_inner != id_map_.end(),
        "Transform traversal failed, dependencies not met.");

    auto id_outer_mapped = it_outer->second;
    auto id_inner_mapped = it_inner->second;

    TORCH_INTERNAL_ASSERT(
        leaf_ids_.find(id_outer_mapped) != leaf_ids_.end() &&
            leaf_ids_.find(id_inner_mapped) != leaf_ids_.end(),
        "Transform traversal failed, modified ",
        id_outer_mapped,
        " and ",
        id_inner_mapped,
        " however one or both are not leaf nodes.");

    Val* merged_id_size =
        mul(id_outer_mapped->extent(), id_inner_mapped->extent());

    IterDomain* merged_id = IterDomainBuilder(m->out())
                                .start(m->container()->zeroVal())
                                .extent(merged_id_size->as<Int>())
                                .build();

    IrBuilder::create<Merge>(
        m->container(), merged_id, id_outer_mapped, id_inner_mapped);

    // Remove inputs from the leaf IDs
    leaf_ids_.erase(id_outer_mapped);
    leaf_ids_.erase(id_inner_mapped);

    // Add the output to the leaf IDs
    leaf_ids_[merged_id] = counter++;

    id_map_[m->out()] = merged_id;
  }

 public:
  ReplaySelf(const std::vector<IterDomain*>& _target_domain, id_map _id_map)
      : ReplayTransformations(_target_domain, std::move(_id_map), false) {}
};

} // namespace

// Self replay.
TensorDomain* TransformReplay::fullSelfReplay(
    const TensorDomain* new_self_root,
    const TensorDomain* self) {
  FUSER_PERF_SCOPE("TransformReplay::fullSelfReplay");

  TORCH_INTERNAL_ASSERT(
      new_self_root->getRootDomain().size() == self->getRootDomain().size(),
      "Invalid number of IterDomains provided.");

  // Map for replay, should be pretty simple.
  id_map axis_map;
  {
    size_t i = 0;
    for (auto id : self->getRootDomain()) {
      TORCH_INTERNAL_ASSERT(
          new_self_root->getRootDomain()[i]->isReduction() ==
                  id->isReduction() &&
              new_self_root->getRootDomain()[i]->isRFactorProduct() ==
                  id->isRFactorProduct() &&
              new_self_root->getRootDomain()[i]->isBroadcast() ==
                  id->isBroadcast(),
          "Axes ",
          id,
          " and ",
          new_self_root->getRootDomain()[i],
          " do not match for self replay.");
      axis_map[id] = new_self_root->getRootDomain()[i];
      i++;
    }
  }

  // Replay producer dimensions.
  ReplaySelf replay(self->domain(), axis_map);
  std::vector<IterDomain*> new_domain(self->nDims(), nullptr);

  {
    size_t i = 0;
    for (auto id : self->domain()) {
      auto it = replay.getReplay().find(id);
      TORCH_INTERNAL_ASSERT(
          it != replay.getReplay().end(),
          "Error during replay, didn't replay an axis.");
      new_domain[i++] = it->second;
    }

    if (self->hasRFactor()) {
      std::vector<IterDomain*> new_rfactor_domain(
          self->getMaybeRFactorDomain().size(), nullptr);
      size_t i = 0;
      for (auto id : self->getMaybeRFactorDomain()) {
        auto it = replay.getReplay().find(id);
        TORCH_INTERNAL_ASSERT(
            it != replay.getReplay().end(),
            "Error during replay, didn't replay an axis.");
        new_rfactor_domain[i++] = it->second;
      }
      return IrBuilder::create<TensorDomain>(
          self->container(),
          new_self_root->getRootDomain(),
          new_rfactor_domain,
          new_domain,
          self->contiguity());
    }
  }

  return IrBuilder::create<TensorDomain>(
      self->container(),
      new_self_root->getRootDomain(),
      new_domain,
      new_self_root->contiguity());
}

// Producer could have rfactor axes which consumer may want replayed. We can
// "replay" them as long as it doesn't modify the root rfactor axes. What we
// really want to do is validate if we replayed these axes to the ones they
// mapped to in the consumer the operations would all be the same. then we want
// to start the replay of the producer from the rfactor root axes, not the root.
std::pair<TensorDomain*, unsigned int> TransformReplay::replayPasC(
    const TensorView* producer,
    const TensorView* consumer,
    int consumer_compute_at_axis,
    const RootDomainMap& root_map,
    bool replay_swizzle) {
  FUSER_PERF_SCOPE("TransformReplay::replayPasC");

  // If this is a reduction operation, we may call transform_replay on the
  // tensor view. When this happens, just return thet target view.
  if (producer == consumer)
    return {producer->domain(), producer->nDims()};

  if (consumer_compute_at_axis < 0)
    consumer_compute_at_axis += (int)consumer->nDims() + 1;
  TORCH_INTERNAL_ASSERT(
      consumer_compute_at_axis >= 0 &&
          (unsigned int)consumer_compute_at_axis <= consumer->nDims(),
      "Invalid axis in transform replayPasC.");

  // consumer ids we need to match in producer
  std::vector<IterDomain*> consumer_CA_ids(
      consumer->domain()->domain().begin(),
      consumer->domain()->domain().begin() + consumer_compute_at_axis);

  // Instead of replaying from the root, lets try to play forward the history of
  // producer if they match ops on consumer. Enforce if we modify an rfactor
  // axis that those ops must match.
  auto forward_replay = BestEffortReplay::replayPasC(
      producer, consumer, consumer_compute_at_axis, root_map);

  // Make a new map based on all the leaves resulting from best effort replay
  id_map forwarded_replay_map;
  auto forward_dangling_leaves = forward_replay.getUnorderedLeafIDs();
  for (auto entry : forward_replay.getReplay()) {
    if (forward_dangling_leaves.find(entry.second) !=
        forward_dangling_leaves.end()) {
      forwarded_replay_map[entry.first] = entry.second;
      forward_dangling_leaves.erase(entry.second);
    }
  }

  // Replay producer dimensions.
  ReplayTransformations replay_PasC(
      consumer_CA_ids, forwarded_replay_map, false, replay_swizzle);

  auto leaf_ids(replay_PasC.getUnorderedLeafIDs());

  // Remove all ids that map to the compute at axis, we're going to replay the
  // rest, track all dims needed to match consumer CA dims
  std::vector<IterDomain*> needed_dims;
  for (auto c_id : consumer_CA_ids) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it == replay_PasC.getReplay().end()) {
      TORCH_INTERNAL_ASSERT(
          c_id->isBroadcast() || c_id->isGather() || c_id->isVectorComponent(),
          "Could not find axis, ",
          c_id,
          ", requested in replay.");
      continue;
    }
    TORCH_INTERNAL_ASSERT(
        leaf_ids.find(it->second) != leaf_ids.end(),
        "Replayed id to match consumer id ",
        c_id,
        " should be a leaf in replay map.");
    leaf_ids.erase(it->second);
    needed_dims.push_back(it->second);
  }

  // leaf_ids now contains all producer ID products that are not used to satisfy
  // the computeAt Turn into a  map so we can play forward these IDs in producer
  // (if possible):
  id_map producer_self_replay_map;
  for (auto entry : leaf_ids) {
    producer_self_replay_map[entry.first] = entry.first;
  }

  for (auto entry : forward_dangling_leaves) {
    producer_self_replay_map[entry.first] = entry.first;
  }

  // Check which root domains were used to produce the leaf_ids. We may have
  // picked up extra roots in consumer because of broadcast forwarding.
  std::vector<Val*> unordered_non_root_leaf_vals;
  for (auto leaf_id : replay_PasC.getUnorderedLeafIDs()) {
    if (leaf_id.first->definition() == nullptr) {
      continue;
    } else {
      unordered_non_root_leaf_vals.emplace_back(leaf_id.first);
    }
  }

  auto producer_root = producer->getMaybeRFactorDomain();

  // Figure out all id's that have been processed to generate the
  // unordered_non_root_leaf_vals. This needs to be done because we want to
  // match on producer's rfactor domain, not root domain.
  std::unordered_set<IterDomain*> all_processed_ids;
  {
    auto all_processed_vals_vec = DependencyCheck::getAllValsBetween(
        {producer_root.begin(), producer_root.end()},
        unordered_non_root_leaf_vals);
    auto all_processed_ids_vec =
        ir_utils::filterByType<IterDomain>(all_processed_vals_vec);
    all_processed_ids.insert(
        all_processed_ids_vec.begin(), all_processed_ids_vec.end());
  }

  // Any root domain that was not used to generate computeIDs we can also put in
  // the map to forward their transformations.
  for (auto producer_root_id : producer_root) {
    if (all_processed_ids.find(producer_root_id) == all_processed_ids.end() &&
        std::find(needed_dims.begin(), needed_dims.end(), producer_root_id) ==
            needed_dims.end()) {
      producer_self_replay_map[producer_root_id] = producer_root_id;
    }
  }

  // Play forward transformations all producer IDs we can
  auto producer_replayed_leaves = BestEffortReplay(
      producer->domain()->domain(),
      producer->domain()->domain(),
      producer_self_replay_map);

  /*
   * Accumulate axes in to the new domain in the following order, making sure to
   * avoid any duplicates:
   *
   * (1) replay_PasC.getReplay holds mappings from axes in consumer compute at
   * axes -> corresponding generated axes in producer
   *
   * (2) Any axes that were not added, that can be mapped directly from an ID in
   * consumer->domain(). These are axes that were "fully replayed" relative to
   * the consumer, even though it wasn't in the computeAt range.
   *
   * producer_replayed_leaves now contain ids that we tried to forward
   * back to what they were in producer. If they couldn't be forwarded they're
   * left in their "most forwarded" form which may be just a remainder of the
   * transformation required to generate the computeAt axes.
   *
   * (3) Axes in producer->domain() that are in producer_replayed_leaves
   *
   * (4) Axes not in producer->domain() that are in producer_replayed_leaves
   *
   */

  std::vector<IterDomain*> new_IDs;
  std::unordered_set<IterDomain*> used_IDs;
  // Add axes in (1)
  for (auto c_id : consumer_CA_ids) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it == replay_PasC.getReplay().end()) {
      TORCH_INTERNAL_ASSERT(
          c_id->isBroadcast() || c_id->isGather() || c_id->isVectorComponent(),
          "Could not find axis, ",
          c_id,
          ", requested in replay.");
      continue;
    }
    new_IDs.push_back(it->second);
    used_IDs.emplace(it->second);
  }

  unsigned int producer_compute_at_axis = new_IDs.size();

  // Add axes in (2)
  for (auto c_id : consumer->domain()->domain()) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it != replay_PasC.getReplay().end()) {
      auto id = it->second;
      // If the leaf id from ReplayTransformations is used to move
      // forward in BestEffortReplay, it is not a final ID.
      if (producer_replayed_leaves.getUnorderedLeafIDs().find(id) ==
          producer_replayed_leaves.getUnorderedLeafIDs().end()) {
        continue;
      }
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (3)
  for (auto id : producer->domain()->domain()) {
    if (producer_replayed_leaves.getUnorderedLeafIDs().find(id) !=
        producer_replayed_leaves.getUnorderedLeafIDs().end()) {
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (4)
  for (auto id : producer_replayed_leaves.getLeafIDs()) {
    if (used_IDs.find(id) == used_IDs.end()) {
      new_IDs.push_back(id);
    }
  }
  TensorDomain* replayed = IrBuilder::create<TensorDomain>(
      producer->container(),
      producer->getRootDomain(),
      producer->getRFactorDomain(),
      new_IDs,
      producer->domain()->contiguity());

  return {replayed, producer_compute_at_axis};
}

std::pair<TensorDomain*, unsigned int> TransformReplay::replayCasP(
    const TensorView* consumer,
    const TensorView* producer,
    int producer_compute_at_axis,
    const RootDomainMap& root_map) {
  FUSER_PERF_SCOPE("TransformReplay::replayCasP");

  // If this is a reduction operation, we may call transform_replay on the same
  // tensor view. When this happens, just return thet target view.
  if (consumer == producer)
    return {consumer->domain(), consumer->nDims()};

  if (producer_compute_at_axis < 0)
    producer_compute_at_axis += (int)producer->nDims() + 1;

  TORCH_INTERNAL_ASSERT(
      producer_compute_at_axis >= 0 &&
          (unsigned int)producer_compute_at_axis <= producer->nDims(),
      "Invalid axis in transform replayCasP.");

  // producer ids we need to match in consumer
  std::vector<IterDomain*> producer_CA_ids(
      producer->domain()->domain().begin(),
      producer->domain()->domain().begin() + producer_compute_at_axis);
  producer_CA_ids = TensorDomain::noReductions(producer_CA_ids);

  // Instead of replaying from the root, lets try to forward the history of
  // consumer if they match ops on producer. Enforce if we modify an rfactor
  // axis that those ops match.
  BestEffortReplay forward_replay = BestEffortReplay::replayCasP(
      consumer, producer, producer_compute_at_axis, root_map);

  // Track dangling leaves which can be produced in
  // BestEffortReplay::replayCasP these don't have any equivalent in producer
  // so they're not in the map. We will simply map them to themselves so we
  // don't lose them.
  id_map forwarded_replay_map;
  auto forward_dangling_leaves = forward_replay.getUnorderedLeafIDs();
  for (auto entry : forward_replay.getReplay()) {
    if (forward_dangling_leaves.find(entry.second) !=
        forward_dangling_leaves.end()) {
      forwarded_replay_map[entry.first] = entry.second;
      forward_dangling_leaves.erase(entry.second);
    }
  }

  // Replay producer dimensions.
  ReplayTransformations replay_CasP(
      producer_CA_ids, forwarded_replay_map, false);

  auto leaf_ids(replay_CasP.getUnorderedLeafIDs());

  // Remove all ids that map to the compute at axis, we're going to replay the
  // rest, track all dims that are needed to match producer CA dims
  std::vector<IterDomain*> needed_dims;
  for (auto p_id : producer_CA_ids) {
    auto it = replay_CasP.getReplay().find(p_id);
    TORCH_INTERNAL_ASSERT(
        it != replay_CasP.getReplay().end(),
        "Could not find axis, ",
        p_id,
        ", requested in replay.");
    TORCH_INTERNAL_ASSERT(
        leaf_ids.find(it->second) != leaf_ids.end(),
        "Replayed id to match producer id ",
        p_id,
        " should be a leaf in replay map.");
    leaf_ids.erase(it->second);
    needed_dims.push_back(it->second);
  }

  // leaf_ids now contains all consumer ID products that are not used to satisfy
  // the computeAt. Turn into a  map so we can play forward these IDs in
  // consumer (if possible):
  id_map consumer_self_replay_map;
  for (auto entry : leaf_ids) {
    consumer_self_replay_map[entry.first] = entry.first;
  }

  for (auto entry : forward_dangling_leaves) {
    consumer_self_replay_map[entry.first] = entry.first;
  }

  // Check which root domains were used to produce the leaf_ids. We may have
  // picked up extra roots in consumer because of broadcast forwarding.
  std::vector<Val*> unordered_non_root_leaf_vals;
  for (auto leaf_id : replay_CasP.getUnorderedLeafIDs()) {
    if (leaf_id.first->definition() == nullptr) {
      continue;
    } else {
      unordered_non_root_leaf_vals.emplace_back(leaf_id.first);
    }
  }

  auto processed_roots = IterVisitor::getInputsTo(unordered_non_root_leaf_vals);

  std::vector<IterDomain*> consumer_root = consumer->getRootDomain();

  // Any root domain that was not used to generate computeIDs we can also put in
  // the map to forward their transformations.
  for (auto consumer_root_id : consumer_root) {
    if (std::find(
            processed_roots.begin(), processed_roots.end(), consumer_root_id) ==
            processed_roots.end() &&
        // Don't re-add roots that may have directly mapped in the replay
        std::find(needed_dims.begin(), needed_dims.end(), consumer_root_id) ==
            needed_dims.end()) {
      consumer_self_replay_map[consumer_root_id] = consumer_root_id;
    }
  }

  // Play forward transformations all consumer IDs we can
  auto consumer_replayed_leaves = BestEffortReplay(
      consumer->domain()->domain(),
      consumer->domain()->domain(),
      consumer_self_replay_map);

  /*
   * Accumulate axes in to the new domain in the following order, making sure to
   * avoid any duplicates:
   *
   * (1) replay_PasC.getReplay holds mappings from axes in consumer compute at
   * axes -> corresponding generated axes in producer
   *
   * (2) Any axes that were not added, that can be mapped directly from an ID in
   * producer->domain(). These are axes that were "fully replayed" relative to
   * the producer, even though it wasn't in the computeAt range.
   *
   * producer_replayed_leaves now contain ids that we tried to forward
   * back to what they were in producer. If they couldn't be forwarded they're
   * left in their "most forwarded" form which may be just a remainder of the
   * transformation required to generate the computeAt axes.
   *
   * (3) Axes in producer->domain() that are in producer_replayed_leaves
   *
   * (4) Axes not in producer->domain() that are in producer_replayed_leaves
   *
   * TODO: Should (2) and (3) be swapped?
   */

  std::vector<IterDomain*> new_IDs;
  std::unordered_set<IterDomain*> used_IDs;
  // Add axes in (1)
  for (auto p_id : producer_CA_ids) {
    auto it = replay_CasP.getReplay().find(p_id);
    TORCH_INTERNAL_ASSERT(
        it != replay_CasP.getReplay().end(),
        "Could not find axis, ",
        p_id,
        ", requested in replay.");
    new_IDs.push_back(it->second);
    used_IDs.emplace(it->second);
  }

  // Add axes in (2)
  for (auto p_id : producer->domain()->domain()) {
    auto it = replay_CasP.getReplay().find(p_id);
    if (it != replay_CasP.getReplay().end()) {
      auto id = it->second;
      // If the leaf id from ReplayTransformations is used to move
      // forward in BestEffortReplay, it is not a final ID.
      if (consumer_replayed_leaves.getUnorderedLeafIDs().find(id) ==
          consumer_replayed_leaves.getUnorderedLeafIDs().end()) {
        continue;
      }
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (3)
  for (auto id : consumer->domain()->domain()) {
    if (consumer_replayed_leaves.getUnorderedLeafIDs().find(id) !=
        consumer_replayed_leaves.getUnorderedLeafIDs().end()) {
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (4)
  for (auto id : consumer_replayed_leaves.getLeafIDs())
    if (used_IDs.find(id) == used_IDs.end())
      new_IDs.push_back(id);

  TensorDomain* replayed = IrBuilder::create<TensorDomain>(
      consumer->container(),
      consumer->getRootDomain(),
      consumer->getRFactorDomain(),
      new_IDs,
      consumer->domain()->contiguity());

  return {replayed, producer_CA_ids.size()};
}

// replay Producer as Consumer
std::pair<TensorDomain*, unsigned int> TransformReplay::replayPasC(
    const TensorView* producer,
    const TensorView* consumer,
    int compute_at_axis,
    bool replay_swizzle) {
  // Use the pairwise root map as a default mapper
  PairwiseRootDomainMap root_map(producer, consumer);
  return replayPasC(
      producer, consumer, compute_at_axis, root_map, replay_swizzle);
}

std::pair<TensorDomain*, unsigned int> TransformReplay::replayCasP(
    const TensorView* consumer,
    const TensorView* producer,
    int compute_at_axis) {
  // Use the pairwise root map as a default mapper
  PairwiseRootDomainMap root_map(producer, consumer);
  return replayCasP(consumer, producer, compute_at_axis, root_map);
}

// In a PasC replay, we want the producer to exactly match the consumer:
// all the beginning axes in the producer should be mapped to the consumer in
// the same order. Reductions in the producer needs to be in the back of the
// producer.
int TransformReplay::getMatchedLeafPosWithoutReplayPasC(
    const TensorView* producer,
    const TensorView* consumer,
    int consumer_pos) {
  FUSER_PERF_SCOPE("transform_replay.cpp::getMatchedLeafPosWithoutReplayPasC");

  const auto pairwise_map = PairwiseRootDomainMap(producer, consumer);
  id_map c2p_root_map = pairwise_map.mapConsumerToProducer(
      consumer->domain(), producer->domain());

  // IterDomains in `consumer` root also in `producer` root
  const auto consumer_domain = consumer->domain()->domain();

  std::unordered_set<Val*> mapped_consumer_roots;
  for (auto entry : c2p_root_map) {
    mapped_consumer_roots.emplace(entry.first);
  }

  auto unskippable_consumer_ids_vec = DependencyCheck::getAllValsBetween(
      mapped_consumer_roots, {consumer_domain.begin(), consumer_domain.end()});

  std::unordered_set<Val*> unskippable_consumer_ids(
      unskippable_consumer_ids_vec.begin(), unskippable_consumer_ids_vec.end());

  // IterDomains in `producer` root also in `consumer` root
  const auto producer_domain = producer->domain()->domain();

  auto it_consumer = consumer_domain.begin();
  auto it_producer = producer_domain.begin();

  auto disjoint_sets =
      BestEffortReplay::replayPasC(producer, consumer, -1, pairwise_map)
          .getDisjointSets();

  int mismatched_consumer_pos = 0;
  int mismatched_producer_pos = 0;
  while (it_consumer != consumer_domain.end()) {
    if (consumer_pos == mismatched_consumer_pos) {
      return mismatched_producer_pos;
    }

    auto consumer_id = *it_consumer;
    if (unskippable_consumer_ids.count(consumer_id) == 0) {
      ++it_consumer;
      ++mismatched_consumer_pos;
      continue;
    }

    if (it_producer == producer_domain.end()) {
      return -1;
    }

    auto producer_id = *it_producer;
    if (disjoint_sets.permissiveAreMapped(producer_id, consumer_id)) {
      ++mismatched_consumer_pos;
      ++mismatched_producer_pos;
      ++it_consumer;
      ++it_producer;
    } else {
      return -1;
    }
  }
  if (consumer_pos == mismatched_consumer_pos) {
    return mismatched_producer_pos;
  }
  return -1;
}

// We want to ignore reductions in the producer in a CasP replay.
int TransformReplay::getMatchedLeafPosWithoutReplayCasP(
    const TensorView* consumer,
    const TensorView* producer,
    int producer_pos) {
  FUSER_PERF_SCOPE("transform_replay.cpp::getMatchedLeafPosWithoutReplayCasP");

  const auto pairwise_map = PairwiseRootDomainMap(producer, consumer);
  id_map p2c_root_map = pairwise_map.mapProducerToConsumer(
      producer->domain(), consumer->domain());

  // IterDomains in `producer` root that are not reduction
  const auto producer_domain = producer->domain()->domain();
  auto unskippable_producer_ids_vec =
      TensorDomain::noReductions(producer_domain);
  std::unordered_set<IterDomain*> unskippable_producer_ids(
      unskippable_producer_ids_vec.begin(), unskippable_producer_ids_vec.end());

  // IterDomains in `consumer` root also in `producer` root
  const auto consumer_domain = consumer->domain()->domain();

  std::unordered_set<Val*> mapped_consumer_roots;
  for (auto entry : p2c_root_map) {
    mapped_consumer_roots.emplace(entry.second);
  }

  auto unskippable_consumer_ids_vec = DependencyCheck::getAllValsBetween(
      mapped_consumer_roots, {consumer_domain.begin(), consumer_domain.end()});

  std::unordered_set<Val*> unskippable_consumer_ids(
      unskippable_consumer_ids_vec.begin(), unskippable_consumer_ids_vec.end());

  auto it_producer = producer_domain.begin();
  auto it_consumer = consumer_domain.begin();

  auto disjoint_sets =
      BestEffortReplay::replayPasC(producer, consumer, -1, pairwise_map)
          .getDisjointSets();

  int mismatched_producer_pos = 0;
  int mismatched_consumer_pos = 0;
  while (it_producer != producer_domain.end()) {
    if (producer_pos == mismatched_producer_pos) {
      return mismatched_consumer_pos;
    }

    auto producer_id = *it_producer;
    if (unskippable_producer_ids.count(producer_id) == 0) {
      ++it_producer;
      ++mismatched_producer_pos;
      continue;
    }

    if (it_consumer == consumer_domain.end()) {
      return -1;
    }

    auto consumer_id = *it_consumer;
    if (unskippable_consumer_ids.count(consumer_id) == 0) {
      ++it_consumer;
      ++mismatched_consumer_pos;
      continue;
    }

    if (disjoint_sets.permissiveAreMapped(producer_id, consumer_id)) {
      ++mismatched_producer_pos;
      ++mismatched_consumer_pos;
      ++it_producer;
      ++it_consumer;
    } else {
      return -1;
    }
  }
  if (producer_pos == mismatched_producer_pos) {
    return mismatched_consumer_pos;
  }
  return -1;
}

bool TransformReplay::fullSelfMatching(
    const TensorView* replay,
    const TensorView* target) {
  auto replay_root = replay->getRootDomain();
  auto replay_dom = replay->domain()->domain();
  auto target_root = target->getRootDomain();
  auto target_dom = target->domain()->domain();
  std::unordered_map<IterDomain*, IterDomain*> target2replay_map;
  if (replay_root.size() != target_root.size()) {
    return false;
  }
  target2replay_map.reserve(replay_root.size());
  std::transform(
      target_root.begin(),
      target_root.end(),
      replay_root.begin(),
      std::inserter(target2replay_map, target2replay_map.begin()),
      [](auto a, auto b) { return std::make_pair(a, b); });
  BestEffortReplay replay_(replay_dom, target_dom, target2replay_map);
  auto r = replay_.getReplay();
  for (int64_t i = 0; i < replay_dom.size(); i++) {
    auto target_id = target_dom[i];
    auto replay_it = r.find(target_id);
    if (replay_it == r.end() || replay_it->second != replay_dom[i]) {
      return false;
    }
  }
  return true;
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

void TransformPropagator::propagateC2P(TensorView* from, TensorView* to) {
  int pos = replayed_pos_.at(from);
  // Note: [Using multiple TransformPropagators]
  // There are cases that we use multiple TransformPropagators along different
  // spanning trees with different references in the same fusion. Some of these
  // spanning trees could overlap. In cases when there are overlapping nodes,
  // TransformPropagator needs to respect the replay of others, because the
  // current TransformPropagator might not contain the most amount of
  // information on how to do the correct transformation. The logic below tells
  // TransformPropagator to skip the replay when not necessary.
  int new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(to, from, pos);
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "TransformPropagator::propagateC2P" << std::endl;
    std::cout << "  from: " << from << " @ " << pos << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayPasC(to, from, pos);
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay.first);
    new_pos = replay.second;
    if (debug) {
      std::cout << "  replayed: " << to << " @ " << new_pos << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped. result position: " << new_pos << std::endl;
  }
  replayed_pos_[to] = new_pos;
}

void TransformPropagator::propagateP2C(TensorView* from, TensorView* to) {
  int pos = replayed_pos_.at(from);
  // See note [Using multiple TransformPropagators]
  int new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(to, from, pos);
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "TransformPropagator::propagateP2C" << std::endl;
    std::cout << "  from: " << from << " @ " << pos << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayCasP(to, from, pos);
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay.first);
    new_pos = replay.second;
    if (debug) {
      std::cout << "  replayed: " << to << " @ " << new_pos << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped. result position: " << new_pos << std::endl;
  }
  replayed_pos_[to] = new_pos;
}

void TransformPropagator::propagateSibling(TensorView* from, TensorView* to) {
  int pos = replayed_pos_.at(from);
  // See note [Using multiple TransformPropagators]
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "TransformPropagator::propagateSibling" << std::endl;
    std::cout << "  from: " << from << " @ " << pos << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
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
    if (debug) {
      std::cout << "  replayed: " << to << " @ " << pos << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped. result position: " << pos << std::endl;
  }
  replayed_pos_[to] = pos;
}

TransformPropagator::TransformPropagator(TensorView* from, int64_t pos) {
  if (pos < 0) {
    pos += int64_t(from->nDims()) + 1;
  }
  TORCH_CHECK(
      pos >= 0 && pos <= from->nDims(),
      "TransformPropagator called on an pos outside valid range.");
  replayed_pos_[from] = pos;
}

void MostInlinedTransformPropagator::propagateC2P(
    TensorView* from,
    TensorView* to) {
  int pos = from->nDims();
  // See note [Using multiple TransformPropagators]
  int new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(to, from, pos);
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "MostInlinedTransformPropagator::propagateC2P" << std::endl;
    std::cout << "  from: " << from << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayPasC(to, from, pos);
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay.first);
    if (debug) {
      std::cout << "  replayed: " << to << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped" << std::endl;
  }
}

void MostInlinedTransformPropagator::propagateP2C(
    TensorView* from,
    TensorView* to) {
  int pos = from->nDims();
  // See note [Using multiple TransformPropagators]
  int new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(to, from, pos);
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "MostInlinedTransformPropagator::propagateP2C" << std::endl;
    std::cout << "  from: " << from << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayCasP(to, from, pos);
    TORCH_INTERNAL_ASSERT(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max producer position.");
    to->setDomain(replay.first);
    if (debug) {
      std::cout << "  replayed: " << to << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped" << std::endl;
  }
}

void MostInlinedTransformPropagator::propagateSibling(
    TensorView* from,
    TensorView* to) {
  // See note [Using multiple TransformPropagators]
  bool debug = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug) {
    std::cout << "MostInlinedTransformPropagator::propagateSibling"
              << std::endl;
    std::cout << "  from: " << from << std::endl;
    std::cout << "  to: " << to << std::endl;
  }
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
    if (debug) {
      std::cout << "  replayed: " << to << std::endl;
    }
  } else if (debug) {
    std::cout << "  replay skipped" << std::endl;
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
