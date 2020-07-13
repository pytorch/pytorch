#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

using id_map = std::unordered_map<IterDomain*, IterDomain*>;

namespace {

struct ReplaySelf : public ReplayTransformations {
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
    Val* oe = ceilDiv(mapped->extent(), s->factor());

    // Manually replay the split, following the output of the operations.
    // This is so rfactor ops are replayed correctly.
    IterDomain* ido = new IterDomain(
        new Int(0),
        static_cast<Int*>(oe),
        s->outer()->parallel_method(),
        s->outer()->isReduction(),
        s->outer()->isRFactorProduct(),
        s->outer()->isBroadcast());

    // inner IterDomain
    IterDomain* idi = new IterDomain(
        new Int(0),
        s->factor(),
        s->inner()->parallel_method(),
        s->inner()->isReduction(),
        s->inner()->isRFactorProduct(),
        s->inner()->isBroadcast());

    // Generate the split node
    new Split(ido, idi, mapped, s->factor());

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

    IterDomain* merged_id = new IterDomain(
        new Int(0),
        static_cast<Int*>(merged_id_size),
        m->out()->parallel_method(),
        m->out()->isReduction(),
        m->out()->isRFactorProduct(),
        m->out()->isBroadcast());

    new Merge(merged_id, id_outer_mapped, id_inner_mapped);

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
    TensorDomain* new_self_root,
    TensorDomain* self) {
  TORCH_INTERNAL_ASSERT(
      new_self_root->nDims() == self->rootDomain().size(),
      "Invalid number of IterDomains provided.");

  // Map for replay, should be pretty simple.
  id_map axis_map;
  {
    size_t i = 0;
    for (auto id : self->rootDomain()) {
      TORCH_INTERNAL_ASSERT(
          new_self_root->axis(i)->start() == id->start(),
          "Replay does not support IterDomains that do not start at 0.");

      TORCH_INTERNAL_ASSERT(
          new_self_root->axis(i)->parallel_method() == id->parallel_method() &&
              new_self_root->axis(i)->isReduction() == id->isReduction() &&
              new_self_root->axis(i)->isRFactorProduct() ==
                  id->isRFactorProduct() &&
              new_self_root->axis(i)->isBroadcast() == id->isBroadcast(),
          "Axes do not match for self replay.");
      axis_map[id] = new_self_root->axis(i);
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
  }

  return new TensorDomain(new_self_root->domain(), new_domain);
}

// Replay producer as consumer.
// Producer could have rfactor axes which consumer may want replayed. We can
// "replay" them as long as it doesn't modify the root rfactor axes. What we
// really want to do is validate if we replayed these axes to the ones they
// mapped to in the consumer the operations would all be the same. then we want
// to start the replay of the producer from the rfactor root axes, not the root.
TensorDomain* TransformReplay::replayPasC(
    TensorDomain* producer,
    TensorDomain* consumer,
    int consumer_compute_at_axis) {
  if (consumer_compute_at_axis < 0)
    consumer_compute_at_axis += (int)consumer->nDims() + 1;
  TORCH_INTERNAL_ASSERT(
      consumer_compute_at_axis >= 0 &&
          (unsigned int)consumer_compute_at_axis <= consumer->nDims(),
      "Invalid axis in transform replayPasC.");

  // consumer ids we need to match in producer
  std::vector<IterDomain*> consumer_CA_ids;
  {
    int itc = 0;
    while (itc < consumer_compute_at_axis) {
      if (consumer->axis(itc)->isBroadcast()) {
        itc++;
      } else {
        consumer_CA_ids.emplace_back(consumer->axis(itc++));
      }
    }
  }

  // Figure out all inputs required to generate the compute_at dimensions
  std::unordered_set<Val*> consumer_CA_root_ids = IterVisitor::getInputsTo(
      std::vector<Val*>(consumer_CA_ids.begin(), consumer_CA_ids.end()));

  // Map of consumer_CA_root_ids to related producer_CA_ids
  id_map replay_root_map;

  // Grab root domains of producer and consumer
  std::vector<IterDomain*> consumer_root = consumer->rootDomain();
  std::vector<IterDomain*> producer_root = producer->rootDomain();

  // If producer has an rfactor root, that's what will match with consumer,
  // as it means the consumer was a result of the rfactor operation.
  if (producer->hasRFactor())
    producer_root = producer->rfactorDomain();

  // Track which root axes in producer we will send to replay
  std::unordered_set<IterDomain*> producer_roots4replay;

  // Map related axes from producer and consumer roots. Make sure we go to the
  // end of both.
  {
    size_t itc = 0, itp = 0;
    while (itc < consumer_root.size() || itp < producer_root.size()) {
      if (itc < consumer_root.size() && consumer_root[itc]->isBroadcast()) {
        itc++;
        continue;
      }
      if (itp < producer_root.size() && producer_root[itp]->isReduction()) {
        itp++;
        continue;
      }
      TORCH_INTERNAL_ASSERT(
          itc < consumer_root.size() && itp < producer_root.size(),
          "Error during replay, wanted to keep going, but ran out of root dimensions.");

      if (consumer_CA_root_ids.find(consumer_root[itc]) !=
          consumer_CA_root_ids.end()) {
        replay_root_map[consumer_root[itc]] = producer_root[itp];
        producer_roots4replay.emplace(producer_root[itp]);
      }
      itc++;
      itp++;
    }
  }

  // Instead of replaying from the root, lets try to play forward the history of
  // producer if they match ops on consumer. Enforce if we modify an rfactor
  // axis that those ops must match.
  BestEffortReplay forward_replay(
      producer->domain(), consumer_CA_ids, replay_root_map);

  id_map forwarded_replay_map;
  for (auto entry : forward_replay.getReplay()) {
    if (forward_replay.getUnorderedLeafIDs().find(entry.second) !=
        forward_replay.getUnorderedLeafIDs().end())
      forwarded_replay_map[entry.first] = entry.second;
  }

  // Replay producer dimensions.
  ReplayTransformations replay_PasC(
      consumer_CA_ids, forwarded_replay_map, false);

  auto leaf_ids(replay_PasC.getUnorderedLeafIDs());

  // Remove all ids that map to the compute at axis, we're going to replay the
  // rest
  for (auto c_id : consumer_CA_ids) {
    auto it = replay_PasC.getReplay().find(c_id);
    TORCH_INTERNAL_ASSERT(
        it != replay_PasC.getReplay().end(),
        "Could not find axis, ",
        c_id,
        ", requested in replay.");
    if (leaf_ids.find(it->second) != leaf_ids.end())
      leaf_ids.erase(it->second);
  }

  // leaf_ids now contains all producer ID products that are not used to satisfy
  // the computeAt Turn into a  map so we can play forward these IDs in producer
  // (if possible):
  id_map producer_self_replay_map;
  for (auto entry : leaf_ids)
    producer_self_replay_map[entry.first] = entry.first;

  // Any root domain that was not used to generate computeIDs we can also put in
  // the map to forward their transformations.
  for (auto producer_root_id : producer_root)
    if (producer_roots4replay.find(producer_root_id) ==
        producer_roots4replay.end()) {
      producer_self_replay_map[producer_root_id] = producer_root_id;
    }

  // Play forward transformations all producer IDs we can
  auto producer_replayed_leaves = BestEffortReplay(
      producer->domain(), producer->domain(), producer_self_replay_map);

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
    TORCH_INTERNAL_ASSERT(
        it != replay_PasC.getReplay().end(),
        "Could not find axis, ",
        c_id,
        ", requested in replay.");
    new_IDs.push_back(it->second);
    used_IDs.emplace(it->second);
  }

  // Add axes in (2)
  std::unordered_set<IterDomain*> consumer_CA_ids_set(
      consumer_CA_ids.begin(), consumer_CA_ids.end());
  for (auto c_id : consumer->domain()) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it != replay_PasC.getReplay().end()) {
      auto id = it->second;
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (3)
  for (auto id : producer->domain()) {
    if (producer_replayed_leaves.getUnorderedLeafIDs().find(id) !=
        producer_replayed_leaves.getUnorderedLeafIDs().end()) {
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (4)
  for (auto id : producer_replayed_leaves.getLeafIDs())
    if (used_IDs.find(id) == used_IDs.end())
      new_IDs.push_back(id);

  TensorDomain* replayed = new TensorDomain(
      producer->rootDomain(), producer->rfactorDomain(), new_IDs);
  return replayed;
}

// Replay consumer as producer.
TensorDomain* TransformReplay::replayCasP(
    TensorDomain* consumer,
    TensorDomain* producer,
    int producer_compute_at_axis) {
  if (producer_compute_at_axis < 0)
    producer_compute_at_axis += (int)producer->nDims() + 1;
  TORCH_INTERNAL_ASSERT(
      producer_compute_at_axis >= 0 &&
          (unsigned int)producer_compute_at_axis <= producer->nDims(),
      "Invalid axis in transform replayCasP.");

  // producer ids we need to match in consumer
  std::vector<IterDomain*> producer_CA_ids;
  {
    int itp = 0;
    while (itp < producer_compute_at_axis) {
      if (producer->axis(itp)->isReduction()) {
        itp++;
      } else {
        producer_CA_ids.emplace_back(producer->axis(itp++));
      }
    }
  }

  // Figure out all inputs required to generate the compute_at dimensions
  std::unordered_set<Val*> producer_CA_root_ids = IterVisitor::getInputsTo(
      std::vector<Val*>(producer_CA_ids.begin(), producer_CA_ids.end()));

  // Map of producer_CA_root_ids to related producer_CA_ids
  id_map replay_root_map;

  // Grab root domains of producer and consumer
  std::vector<IterDomain*> consumer_root = consumer->rootDomain();
  std::vector<IterDomain*> producer_root = producer->rootDomain();
  // If producer has an rfactor root, that's the one that will match the
  // consumer
  if (producer->hasRFactor())
    producer_root = producer->rfactorDomain();

  // Track which root axes in consumer we send to replay
  std::unordered_set<IterDomain*> consumer_roots4replay;
  // Map related axes from producer and consumer roots. Make sure we go to the
  // end of both.
  {
    size_t itc = 0, itp = 0;
    while (itc < consumer_root.size() || itp < producer_root.size()) {
      if (itc < consumer_root.size() && consumer_root[itc]->isBroadcast()) {
        itc++;
        continue;
      }
      if (itp < producer_root.size() && producer_root[itp]->isReduction()) {
        itp++;
        continue;
      }
      TORCH_INTERNAL_ASSERT(
          itc < consumer_root.size() && itp < producer_root.size(),
          "Error during replay, wanted to keep going, but ran out of root dimensions.");

      if (producer_CA_root_ids.find(producer_root[itp]) !=
          producer_CA_root_ids.end()) {
        replay_root_map[producer_root[itp]] = consumer_root[itc];
        consumer_roots4replay.emplace(consumer_root[itc]);
      }
      itc++;
      itp++;
    }
  }

  // Instead of replaying from the root, lets try to forward the history of
  // consumer if they match ops on producer. Enforce if we modify an rfactor
  // axis that those ops match.
  BestEffortReplay forward_replay(
      consumer->domain(), producer_CA_ids, replay_root_map);

  id_map forwarded_replay_map;
  for (auto entry : forward_replay.getReplay()) {
    if (forward_replay.getUnorderedLeafIDs().find(entry.second) !=
        forward_replay.getUnorderedLeafIDs().end())
      forwarded_replay_map[entry.first] = entry.second;
  }

  // Replay producer dimensions.
  ReplayTransformations replay_CasP(
      producer_CA_ids, forwarded_replay_map, false);

  auto leaf_ids(replay_CasP.getUnorderedLeafIDs());

  // Remove all ids that map to the compute at axis, we're going to replay the
  // rest
  for (auto p_id : producer_CA_ids) {
    auto it = replay_CasP.getReplay().find(p_id);
    TORCH_INTERNAL_ASSERT(
        it != replay_CasP.getReplay().end(),
        "Could not find axis, ",
        p_id,
        ", requested in replay.");
    if (leaf_ids.find(it->second) != leaf_ids.end())
      leaf_ids.erase(it->second);
  }

  // leaf_ids now contains all consumer ID products that are not used to satisfy
  // the computeAt Turn into a  map so we can play forward these IDs in consumer
  // (if possible):
  id_map consumer_self_replay_map;
  for (auto entry : leaf_ids)
    consumer_self_replay_map[entry.first] = entry.first;

  // Any root domain that was not used to generate computeIDs we can also put in
  // the map to forward their transformations.
  for (auto consumer_root_id : consumer_root)
    if (consumer_roots4replay.find(consumer_root_id) ==
        consumer_roots4replay.end())
      consumer_self_replay_map[consumer_root_id] = consumer_root_id;

  // Play forward transformations all consumer IDs we can
  auto consumer_replayed_leaves = BestEffortReplay(
      consumer->domain(), consumer->domain(), consumer_self_replay_map);

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
  std::unordered_set<IterDomain*> consumer_CA_ids_set(
      producer_CA_ids.begin(), producer_CA_ids.end());
  for (auto p_id : producer->domain()) {
    auto it = replay_CasP.getReplay().find(p_id);
    if (it != replay_CasP.getReplay().end()) {
      auto id = it->second;
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (3)
  for (auto id : consumer->domain()) {
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

  TensorDomain* replayed = new TensorDomain(
      consumer->rootDomain(), consumer->rfactorDomain(), new_IDs);

  return replayed;
}

// replay Producer as Consumer
TensorView* TransformReplay::replayPasC(
    TensorView* producer,
    TensorView* consumer,
    int compute_at_axis) {
  // If this is a reduction operation, we may call transform_replay on the

  // tensor view. When this happens, just return thet target view.
  if (producer == consumer)
    return producer;

  TensorDomain* td =
      replayPasC(producer->domain(), consumer->domain(), compute_at_axis);
  producer->setDomain(td);
  return producer;
}

TensorView* TransformReplay::replayCasP(
    TensorView* consumer,
    TensorView* producer,
    int compute_at_axis) {
  // If this is a reduction operation, we may call transform_replay on the same
  // tensor view. When this happens, just return thet target view.
  if (consumer == producer)
    return consumer;
  TensorDomain* td =
      replayCasP(consumer->domain(), producer->domain(), compute_at_axis);
  consumer->setDomain(td);
  return consumer;
}

} // namespace fuser
} // namespace jit
} // namespace torch
