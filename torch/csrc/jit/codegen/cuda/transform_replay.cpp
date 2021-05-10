#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <vector>

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
    Val* oe = ceilDiv(mapped->extent(), s->factor());

    // Manually replay the split, following the output of the operations.
    // This is so rfactor ops are replayed correctly.
    IterDomain* ido = new IterDomain(
        new Int(0),
        oe->as<Int>(),
        s->outer()->getParallelType(),
        s->outer()->getIterType(),
        s->outer()->isRFactorProduct());

    // inner IterDomain
    IterDomain* idi = new IterDomain(
        new Int(0),
        s->factor(),
        s->inner()->getParallelType(),
        s->outer()->getIterType(),
        s->inner()->isRFactorProduct());

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
        merged_id_size->as<Int>(),
        m->out()->getParallelType(),
        m->outer()->getIterType(),
        m->out()->isRFactorProduct());

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
    const TensorDomain* new_self_root,
    const TensorDomain* self) {
  FUSER_PERF_SCOPE("fullSelfReplay");

  TORCH_INTERNAL_ASSERT(
      new_self_root->nDims() == self->getRootDomain().size(),
      "Invalid number of IterDomains provided.");

  // Map for replay, should be pretty simple.
  id_map axis_map;
  {
    size_t i = 0;
    for (auto id : self->getRootDomain()) {
      TORCH_INTERNAL_ASSERT(
          new_self_root->axis(i)->start() == id->start(),
          "Replay does not support IterDomains that do not start at 0.");

      TORCH_INTERNAL_ASSERT(
          new_self_root->axis(i)->getParallelType() == id->getParallelType() &&
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

  return new TensorDomain(
      new_self_root->domain(), new_domain, self->contiguity());
}

// Producer could have rfactor axes which consumer may want replayed. We can
// "replay" them as long as it doesn't modify the root rfactor axes. What we
// really want to do is validate if we replayed these axes to the ones they
// mapped to in the consumer the operations would all be the same. then we want
// to start the replay of the producer from the rfactor root axes, not the root.
std::pair<TensorDomain*, unsigned int> TransformReplay::replayPasC(
    const TensorDomain* producer,
    const TensorDomain* consumer,
    int consumer_compute_at_axis) {
  FUSER_PERF_SCOPE("replayPasC");

  if (consumer_compute_at_axis < 0)
    consumer_compute_at_axis += (int)consumer->nDims() + 1;
  TORCH_INTERNAL_ASSERT(
      consumer_compute_at_axis >= 0 &&
          (unsigned int)consumer_compute_at_axis <= consumer->nDims(),
      "Invalid axis in transform replayPasC.");

  // consumer ids we need to match in producer
  std::vector<IterDomain*> consumer_CA_ids(
      consumer->domain().begin(),
      consumer->domain().begin() + consumer_compute_at_axis);

  // Figure out all inputs required to generate the compute_at dimensions
  std::unordered_set<Val*> consumer_CA_root_vals = IterVisitor::getInputsTo(
      std::vector<Val*>(consumer_CA_ids.begin(), consumer_CA_ids.end()));

  std::unordered_set<IterDomain*> consumer_CA_root_ids;
  for (auto val : consumer_CA_root_vals) {
    if (val->getValType().value() == ValType::IterDomain) {
      consumer_CA_root_ids.emplace(val->as<IterDomain>());
    }
  }

  // Map of consumer_CA_root_ids to related producer_CA_ids
  auto replay_root_map =
      TensorDomain::mapRootCtoP(consumer, producer, consumer_CA_root_ids);

  // Track which root axes in producer we will send to replay
  std::unordered_set<IterDomain*> producer_roots4replay;
  for (auto entry : replay_root_map) {
    producer_roots4replay.emplace(entry.second);
  }

  // Instead of replaying from the root, lets try to play forward the history of
  // producer if they match ops on consumer. Enforce if we modify an rfactor
  // axis that those ops must match.
  BestEffortReplay forward_replay(
      producer->domain(), consumer_CA_ids, replay_root_map);

  // Make a new map based on all the leaves resulting from best effort replay
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
    if (it == replay_PasC.getReplay().end()) {
      TORCH_INTERNAL_ASSERT(
          c_id->isBroadcast(),
          "Could not find axis, ",
          c_id,
          ", requested in replay.");
      continue;
    }
    if (leaf_ids.find(it->second) != leaf_ids.end())
      leaf_ids.erase(it->second);
  }

  // leaf_ids now contains all producer ID products that are not used to satisfy
  // the computeAt Turn into a  map so we can play forward these IDs in producer
  // (if possible):
  id_map producer_self_replay_map;
  for (auto entry : leaf_ids)
    producer_self_replay_map[entry.first] = entry.first;

  auto producer_root = producer->getMaybeRFactorDomain();

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
    if (it == replay_PasC.getReplay().end()) {
      TORCH_INTERNAL_ASSERT(
          c_id->isBroadcast(),
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
  for (auto c_id : consumer->domain()) {
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
      producer->getRootDomain(),
      producer->getRFactorDomain(),
      new_IDs,
      producer->contiguity());
  return {replayed, producer_compute_at_axis};
}

std::pair<TensorDomain*, unsigned int> TransformReplay::replayCasP(
    const TensorDomain* consumer,
    const TensorDomain* producer,
    int producer_compute_at_axis) {
  FUSER_PERF_SCOPE("replayCasP");

  if (producer_compute_at_axis < 0)
    producer_compute_at_axis += (int)producer->nDims() + 1;

  TORCH_INTERNAL_ASSERT(
      producer_compute_at_axis >= 0 &&
          (unsigned int)producer_compute_at_axis <= producer->nDims(),
      "Invalid axis in transform replayCasP.");

  // producer ids we need to match in consumer
  std::vector<IterDomain*> producer_CA_ids(
      producer->domain().begin(),
      producer->domain().begin() + producer_compute_at_axis);
  producer_CA_ids = TensorDomain::noReductions(producer_CA_ids);

  // Grab root domains of producer and consumer
  std::vector<IterDomain*> consumer_root = consumer->getRootDomain();

  // If producer has an rfactor root, that's what will match the consumer
  std::vector<IterDomain*> producer_root = producer->getMaybeRFactorDomain();

  // Figure out all inputs required to generate the compute_at dimensions. We
  // need all deps because inputs on producer may be in getRootDomain, but we
  // may need in rFactorDomain
  std::unordered_set<Val*> all_CA_id_deps = DependencyCheck::getAllValsBetween(
      {producer_root.begin(), producer_root.end()},
      {producer_CA_ids.begin(), producer_CA_ids.end()});

  // Figure out which root IDs we need:
  std::unordered_set<IterDomain*> producer_CA_root_ids;
  for (IterDomain* id : producer_root) {
    if (all_CA_id_deps.find(id) != all_CA_id_deps.end())
      producer_CA_root_ids.emplace(id);
  }

  auto replay_root_map =
      TensorDomain::mapRootPtoC(producer, consumer, producer_CA_root_ids);

  // Track which root axes in producer we will send to replay
  std::unordered_set<IterDomain*> consumer_roots4replay;
  for (auto entry : replay_root_map) {
    consumer_roots4replay.emplace(entry.second);
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
  for (auto p_id : producer->domain()) {
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
      consumer->getRootDomain(),
      consumer->getRFactorDomain(),
      new_IDs,
      consumer->contiguity());

  return {replayed, producer_CA_ids.size()};
}

// replay Producer as Consumer
std::pair<TensorView*, unsigned int> TransformReplay::replayPasC(
    TensorView* producer,
    TensorView* consumer,
    int compute_at_axis) {
  // If this is a reduction operation, we may call transform_replay on the

  // tensor view. When this happens, just return thet target view.
  if (producer == consumer)
    return {producer, 0};

  std::pair<TensorDomain*, unsigned int> replay =
      replayPasC(producer->domain(), consumer->domain(), compute_at_axis);
  producer->setDomain(replay.first);
  return {producer, replay.second};
}

std::pair<TensorView*, unsigned int> TransformReplay::replayCasP(
    TensorView* consumer,
    TensorView* producer,
    int compute_at_axis) {
  // If this is a reduction operation, we may call transform_replay on the same
  // tensor view. When this happens, just return thet target view.
  if (consumer == producer)
    return {consumer, 0};
  std::pair<TensorDomain*, unsigned int> replay =
      replayCasP(consumer->domain(), producer->domain(), compute_at_axis);
  consumer->setDomain(replay.first);
  return {consumer, replay.second};
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
