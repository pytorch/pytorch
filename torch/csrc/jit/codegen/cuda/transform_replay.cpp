#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
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
          new_self_root->getRootDomain()[i]->getParallelType() ==
                  id->getParallelType() &&
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
    const RootDomainMap& root_map) {
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
      consumer_CA_ids, forwarded_replay_map, false);

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
    int compute_at_axis) {
  // Use the pairwise root map as a default mapper
  PairwiseRootDomainMap root_map(producer, consumer);
  return replayPasC(producer, consumer, compute_at_axis, root_map);
}

std::pair<TensorDomain*, unsigned int> TransformReplay::replayCasP(
    const TensorView* consumer,
    const TensorView* producer,
    int compute_at_axis) {
  // Use the pairwise root map as a default mapper
  PairwiseRootDomainMap root_map(producer, consumer);
  return replayCasP(consumer, producer, compute_at_axis, root_map);
}

namespace {

std::deque<TensorView*> deduplicate(const std::deque<TensorView*>& tv_deuqe) {
  std::deque<TensorView*> deduplicated;
  std::unordered_set<TensorView*> inserted;
  for (auto tv_entry : tv_deuqe) {
    if (inserted.find(tv_entry) == inserted.end()) {
      deduplicated.emplace_back(tv_entry);
      inserted.emplace(tv_entry);
    }
  }
  return deduplicated;
}

std::deque<TensorView*> tvInputs(Expr* expr) {
  auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());
  return std::deque<TensorView*>(tv_inputs.begin(), tv_inputs.end());
}

std::deque<TensorView*> tvOutputs(Expr* expr) {
  auto tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());
  return std::deque<TensorView*>(tv_outputs.begin(), tv_outputs.end());
}

std::deque<TensorView*> consumersOf(TensorView* tv) {
  std::deque<TensorView*> consumer_tvs;
  for (auto def : tv->uses()) {
    auto outs = tvOutputs(def);
    consumer_tvs.insert(consumer_tvs.end(), outs.begin(), outs.end());
  }
  return deduplicate(consumer_tvs);
}

std::deque<TensorView*> producersFor(TensorView* tv) {
  auto def = tv->definition();
  if (def == nullptr) {
    return {};
  }

  return deduplicate(tvInputs(def));
}

// This is a struct storing how the information about a root ID in the
// starting tensor is preserved during propagation. If during propagation, we
// reached a tensor called the "current" tensor, we are interested in the
// following information:
// - Which reference tensor's root ID's information does the current tensor
//   contains? Each RootIDInfo object should correspond to one reference
//   tensor's root ID, but we don't need to store this ID explicitly.
// - For this reference tensor's root ID, what are its corresponding IDs in
//   the current tensor's root/rfactor domain?
// - Is the current tensor's information about this reference tensor's root ID
//   complete?
struct RootIDInfo {
  // Each object of this class correspond to one root ID in the reference
  // tensor, but we do not need to explicitly store this ID.

  // The IDs in the current tensor's root or rfactor domain that contains
  // information of the corresponding reference tensor's root ID. Whether we
  // are using root domain or rfactor domain depends on how we reached the
  // current tensor during propagation. `is_rfactor` tells us whether the IDs
  // contained in `mapped_ids` are from the root domain or the rfactor domain.
  std::unordered_set<IterDomain*> mapped_ids;

  // Does `mapped_ids` contain all the IDs required to recompute the
  // corresponding reference tensor's root ID? For example, if we have
  //   t1 = input tensor of shape (20,)
  //   t2 = view(t1, {4, 5})
  //   t3 = sum(t2, {1})
  //   t4 = set(t3)
  // and we start the propagation from t1, then t2 and t3's information about
  // t1 is complete, but t4 is not because one axis is missing.
  bool is_complete;

  // Is `mapped_ids` from the root domain or rfactor domain of the current
  // tensor? We only store IDs from one of them, depending on how we reach the
  // current tensor during propagation. If we reached the current tensor from
  // a consumer, then `mapped_ids` containes IDs in the current tensor's
  // rfactor domain because the rfactor domain contains raw information. If we
  // reached the current tensor from a producer, then `mapped_ids` containes
  // IDs in the current tensor's root domain because the root domain contains
  // raw information.
  bool is_rfactor;

  RootIDInfo() = default;

  // This constructor is only used on the reference tensor where the
  // propagation starts, so the mapped_ids are just the starting_root_id.
  RootIDInfo(IterDomain* starting_root_id)
      : mapped_ids{starting_root_id}, is_complete(true), is_rfactor(false) {}
};

enum class NextHopType {
  C_AS_P,
  P_AS_C,
};

// This is a helper struct that contains all the information about the next
// step in the Dijkstra algorithm
struct NextHopInfo {
  NextHopType type;
  TensorView* from = nullptr;
  TensorView* to;

  std::vector<RootIDInfo> root_id_info_from;
  std::vector<RootIDInfo> root_id_info_to;
};

// l < r means l contains a smaller amount of information about the starting
// tensor than r.
bool operator<(const NextHopInfo& l, const NextHopInfo& r) {
  if (l.root_id_info_to.size() != r.root_id_info_to.size()) {
    return l.root_id_info_to.size() < r.root_id_info_to.size();
  }
  size_t l_complete = std::count_if(
      l.root_id_info_to.begin(),
      l.root_id_info_to.end(),
      [](const RootIDInfo& i) { return i.is_complete; });
  size_t r_complete = std::count_if(
      r.root_id_info_to.begin(),
      r.root_id_info_to.end(),
      [](const RootIDInfo& i) { return i.is_complete; });
  return l_complete < r_complete;
}

// l > r means l contains a bigger amount of information about the starting
// tensor than r.
bool operator>(const NextHopInfo& l, const NextHopInfo& r) {
  return r < l;
}

// l == r means it is hard to tell which one of then contains more information
bool operator==(const NextHopInfo& l, const NextHopInfo& r) {
  return !(r < l) && !(l < r);
}

std::vector<RootIDInfo> getStartingRootIDInfo(TensorView* tv) {
  std::vector<RootIDInfo> result;
  const auto& root_domain = tv->getRootDomain();
  result.reserve(root_domain.size());
  for (auto id : root_domain) {
    result.emplace_back(id);
  }
  return result;
}

// Infer the compute-at position from the information of preserved reference
// root ID.
//
// TODO:
// I think I need to modify TransformReplay to add a new interface to specify
// the root domains, instead of a position in the leaf domain. With the new
// interface, this function will not be needed.
size_t getReplayPos(const NextHopInfo& next_hop) {
  auto& root_id_info = next_hop.root_id_info_from;
  auto from_tv = next_hop.from;
  // Flatten `root_id_info_from` to get the list of ids in the `from_tv`'s
  // root/rfactor domain that contains information about the reference tensor.
  std::unordered_set<IterDomain*> from_ids;
  from_ids.reserve(root_id_info.size());
  for (auto info : root_id_info) {
    for (auto id : info.mapped_ids) {
      from_ids.insert(id);
    }
  }
  // Get leaf IDs that contain information of `from_ids`
  std::unordered_set<IterDomain*> relevant_leaves;
  std::vector<IterDomain*> to_visit(from_ids.begin(), from_ids.end());
  while (!to_visit.empty()) {
    auto front = to_visit.back();
    to_visit.pop_back();
    if (front->uses().empty()) {
      relevant_leaves.emplace(front);
    } else {
      for (auto def : front->uses()) {
        auto outs = ir_utils::filterByType<IterDomain>(def->outputs());
        to_visit.insert(to_visit.end(), outs.begin(), outs.end());
      }
    }
  }
  // Find the pos where all leaf IDs at <= pos contains
  // information about the starting root domain
  //
  // TODO: should I change to the following behavior?
  //
  // Find the smallest pos where all leaf IDs containing
  // information about the starting root domain are <= pos
  //
  // For example, if I have
  //   preserved root domain: [I1, I2, I3, I4]
  //   leaf domain: [I5, I6, I7, I8]
  // where
  //   I5 = merge(I1, I2)
  //   I6 = something unrelated
  //   I7 = merge(I3, I4)
  //   I8 = something unrelated
  // should I return 1, or 3 ?

  // size_t i;
  // for (i = from_tv->nDims() - 1; i >= 0; i--) {
  //   if (relevant_leaves.count(from_tv->axis(i)) > 0) {
  //     break;
  //   }
  // }
  // return i + 1;

  for (size_t i = 0; i < from_tv->nDims(); i++) {
    if (relevant_leaves.count(from_tv->axis(i)) == 0) {
      return i;
    }
  }
  return from_tv->nDims();
}

// Given `root_ids`, a list of IDs in the root domain of `tv`, find their
// corresponding IDs in the rfactor domain of `tv`.
std::unordered_set<IterDomain*> mapRootToRFactor(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& root_ids) {
  std::unordered_set<IterDomain*> mapped_rfactor_ids;
  const auto& rfactor_dom = tv->getMaybeRFactorDomain();
  for (auto id : rfactor_dom) {
    if (root_ids.count(id) > 0) {
      mapped_rfactor_ids.emplace(id);
      continue;
    }
    for (auto root_id : root_ids) {
      if (id == root_id || DependencyCheck::isDependencyOf(root_id, id)) {
        mapped_rfactor_ids.emplace(id);
        break;
      }
    }
  }
  return mapped_rfactor_ids;
}

// Given `rfactor_ids`, a list of IDs in the rfactor domain of `tv`, find their
// corresponding IDs in the root domain of `tv`.
std::unordered_set<IterDomain*> mapRFactorToRoot(
    TensorView* tv,
    const std::unordered_set<IterDomain*>& rfactor_ids) {
  std::unordered_set<IterDomain*> mapped_root_ids;
  for (auto id : tv->getRootDomain()) {
    if (rfactor_ids.count(id) > 0) {
      mapped_root_ids.emplace(id);
      continue;
    }
    for (auto rfactor_id : rfactor_ids) {
      if (DependencyCheck::isDependencyOf(id, rfactor_id)) {
        mapped_root_ids.emplace(id);
        break;
      }
    }
  }
  return mapped_root_ids;
}

// Given the preserved reference root ID info of a producer, compute
// the corresponding info in consumer. The given info may be represented by
// producer's root domain, or rfactor domain, depending on how we reached the
// producer during propagation. If the given info is already represented with
// producer's rfactor domain, then we directly map it to the consumer's root
// domain. If the given info is represented with producer's root domain, we need
// to first map it to the rfactor domain of the producer, then we can map it to
// the consumer's root domain. The computed info will be represented by root
// domain as root domain contains the raw information.
std::vector<RootIDInfo> computeNextRootIDInfoCasP(
    TensorView* producer,
    TensorView* consumer,
    const std::vector<RootIDInfo>& producer_root_id_info) {
  std::vector<RootIDInfo> result;

  auto pairwise_map = PairwiseRootDomainMap(producer, consumer);
  auto p2c_map = pairwise_map.mapProducerToConsumer(
      producer->domain(), consumer->domain());

  for (auto& info : producer_root_id_info) {
    RootIDInfo consumer_info;
    consumer_info.is_complete = info.is_complete;
    consumer_info.is_rfactor = false;

    // mapped root ids in producer -> mapped rfactor ids in producer
    std::unordered_set<IterDomain*> producer_mapped_rfactor_ids;
    if (producer->hasRFactor() && !info.is_rfactor) {
      producer_mapped_rfactor_ids = mapRootToRFactor(producer, info.mapped_ids);
    } else {
      producer_mapped_rfactor_ids = info.mapped_ids;
    }

    // mapped rfactor ids in producer -> mapped root ids in consumer
    for (auto producer_id : producer_mapped_rfactor_ids) {
      auto it = p2c_map.find(producer_id);
      if (it != p2c_map.end()) {
        consumer_info.mapped_ids.insert(it->second);
      } else {
        consumer_info.is_complete = false;
      }
    }

    // If at least one root id in the consumer contains information
    // of this starting root id, then keep this record
    if (!consumer_info.mapped_ids.empty()) {
      result.push_back(consumer_info);
    }
  }
  return result;
}

// Given the preserved reference root ID info of a consumer, compute
// the corresponding info in producer. The given info may be represented by
// consumer's root domain, or rfactor domain, depending on how we reached the
// consumer during propagation. If the given info is already represented with
// consumer's root domain, then we directly map it to the producer's rfactor
// domain. If the given info is represented with consumer's rfactor domain, we
// need to first map it to the root domain of the consumer, then we can map it
// to the producer's rfactor domain. The computed info will be represented by
// rfactor domain as rfactor domain contains the raw information.
std::vector<RootIDInfo> computeNextRootIDInfoPasC(
    TensorView* producer,
    TensorView* consumer,
    const std::vector<RootIDInfo>& consumer_root_id_info) {
  std::vector<RootIDInfo> result;
  auto pairwise_map = PairwiseRootDomainMap(producer, consumer);
  auto c2p_map = pairwise_map.mapConsumerToProducer(
      consumer->domain(), producer->domain());

  for (auto& info : consumer_root_id_info) {
    RootIDInfo producer_info;
    producer_info.is_complete = info.is_complete;
    producer_info.is_rfactor = true;

    // mapped rfactor ids in consumer -> mapped root ids in consumer
    std::unordered_set<IterDomain*> consumer_mapped_root_ids;
    if (info.is_rfactor && consumer->hasRFactor()) {
      consumer_mapped_root_ids = mapRFactorToRoot(consumer, info.mapped_ids);
    } else {
      consumer_mapped_root_ids = info.mapped_ids;
    }

    // mapped root ids in consumer -> mapped rfactor ids in producer
    for (auto consumer_id : consumer_mapped_root_ids) {
      auto it = c2p_map.find(consumer_id);
      if (it != c2p_map.end()) {
        producer_info.mapped_ids.insert(it->second);
      } else {
        producer_info.is_complete = false;
      }
    }

    // We will stop at the rfactor ids in producer, and will not further map
    // them into root ids in producer. This means, we only keep the unprocessed
    // raw information of a tensor. This behavior is important to make sure that
    // info is as accurate as possible throughout the propagation.
    //
    // For example, if we do a C->P->C' propagation, we want to do
    //   C(root) -> P(rfactor) -> C'(root)
    // instead of
    //   C(root) -> P(rfactor) -> P(root) -> P(rfactor) -> C'(root)
    //
    // and the above two paths do lead to different results:
    //
    // For example if you have a producer tensor
    //   root domain: [I1, I2]
    //   rfactor domain: [I3, I5]
    // where I3, I4 = split(I1), I5 = merge(I4, I2)
    // Then the P(rfactor) -> P(root) -> P(rfactor) could lead to
    // P(rfactor: {I5}) -> P(root: {I1, I2}) -> P(rfactor: {I3, I5})
    // which is not correct

    // If at least one root id in the producer contains information
    // of this starting root id, then keep this record
    if (!producer_info.mapped_ids.empty()) {
      result.push_back(producer_info);
    }
  }
  return result;
}

}; // namespace

unsigned int TransformPropagator::replay(const NextHopInfo& next_hop) {
  if (next_hop.from == nullptr) {
    // nullptr used to start from starting_tv
    return next_hop.to->nDims();
  }
  // TODO: why does TransformReplay require specifying a position in the
  // leaf domain? I want to change the interface to allow specifying
  // the starting root domains instead of leaf position.
  int pos = getReplayPos(next_hop);
  std::pair<TensorDomain*, unsigned int> replay;
  switch (next_hop.type) {
    case NextHopType::P_AS_C: {
      auto pairwiseMap = PairwiseRootDomainMap(next_hop.to, next_hop.from);
      replay = TransformReplay::replayPasC(
          next_hop.to, next_hop.from, pos, pairwiseMap);
      break;
    }
    case NextHopType::C_AS_P: {
      auto pairwiseMap = PairwiseRootDomainMap(next_hop.from, next_hop.to);
      replay = TransformReplay::replayCasP(
          next_hop.to, next_hop.from, pos, pairwiseMap);
      break;
    }
  }
  next_hop.to->setDomain(replay.first);
  return replay.second;
}

// Dijkstra
TransformPropagator::TransformPropagator(TensorView* from) : starting_tv(from) {
  // A set that allows us to quickly tell if a tensor has been replayed. If yes,
  // then we will not bother computing if a new path to this tensor is worth
  // taking (because the answer is always not worth)
  std::unordered_set<TensorView*> replayed;

  // A sorted list of possible next steps. The list is sorted in the order of
  // ascending amount of preserved information about the reference tensor. The
  // back of the list preserves the most amount of information about the
  // reference tensor, and should always be the next step to take. We use
  // std::list instead of std::priority_queue because C++'s
  // std::priority_queue does not support increase-key, and might not be
  // deterministic either.
  std::list<NextHopInfo> propagation(1);
  propagation.back().from = nullptr;
  propagation.back().to = starting_tv;
  propagation.back().root_id_info_to = getStartingRootIDInfo(starting_tv);

  // Insert the given next hop the correct position in `propagation`. If there
  // is an existing next hop that preserves more information, then we will just
  // discard `info`.
  auto insertNextHopInfo = [&](const NextHopInfo& info) {
    if (info.root_id_info_from.empty()) {
      // When there is no more information about the starting tensor,
      // we are not interested in continuing the propagation.
      return;
    }
    // Find if there is already a path to the dest tensor
    auto existing = std::find_if(
        propagation.begin(), propagation.end(), [&](const NextHopInfo& i) {
          return i.to == info.to;
        });
    // Only insert if there is no existing path to the dest tensor, or the new
    // path preserves more information about the starting tensor.
    if (existing == propagation.end() || *existing < info) {
      if (existing != propagation.end()) {
        propagation.erase(existing);
      }
      auto pos = std::upper_bound(propagation.begin(), propagation.end(), info);
      propagation.insert(pos, info);
    }
  };

  while (!propagation.empty()) {
    auto next_hop = propagation.back();
    propagation.pop_back();

    replay(next_hop);
    replayed.emplace(next_hop.to);

    for (auto consumer_tv : consumersOf(next_hop.to)) {
      if (replayed.count(consumer_tv)) {
        continue;
      }
      insertNextHopInfo(
          {.type = NextHopType::C_AS_P,
           .from = next_hop.to,
           .to = consumer_tv,
           .root_id_info_from = next_hop.root_id_info_to,
           .root_id_info_to = computeNextRootIDInfoCasP(
               next_hop.to, consumer_tv, next_hop.root_id_info_to)});
    }

    for (auto producer_tv : producersFor(next_hop.to)) {
      if (replayed.count(producer_tv)) {
        continue;
      }
      insertNextHopInfo(
          {.type = NextHopType::P_AS_C,
           .from = next_hop.to,
           .to = producer_tv,
           .root_id_info_from = next_hop.root_id_info_to,
           .root_id_info_to = computeNextRootIDInfoPasC(
               producer_tv, next_hop.to, next_hop.root_id_info_to)});
    }
  }
}

void TransformPropagator::from(TensorView* tv) {
  TransformPropagator propagate(tv);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
