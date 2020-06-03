#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// Replay producer as consumer.
TensorDomain* TransformReplay::fullSelfReplay(
    TensorDomain* self,
    TensorDomain* self_copy) {
  // Want producer root with no reductions, rfactor included
  TensorDomain* self_root = self->rootDomain();

  // Want full consumer root, even before rfactor
  TensorDomain* self_copy_root = self_copy->rootDomain();

  TORCH_INTERNAL_ASSERT(
      self_root->nDims(), self_copy_root->nDims(), "Invalid self replay.");

  for (decltype(self_root->nDims()) i{0}; i < self_root->nDims(); i++)
    TORCH_INTERNAL_ASSERT(
        self_root->axis(i)->parallel_method() ==
                self_copy_root->axis(i)->parallel_method() &&
            self_root->axis(i)->isReduction() ==
                self_copy_root->axis(i)->isReduction() &&
            self_root->axis(i)->start() == self_copy_root->axis(i)->start(),
        "Invalid self replay detected, root domain does not match.");

  std::vector<int> axis_map(self_root->nDims());
  std::iota(axis_map.begin(), axis_map.end(), 0);

  // Finally replay producer as consumer on marked axes

  auto replayed = TransformIter::replay(
      self_copy_root, TransformIter::getHistory(self), axis_map);

  return replayed;
}

// Replay producer as consumer.
TensorDomain* TransformReplay::replayPasC(
    TensorDomain* producer,
    TensorDomain* consumer,
    int compute_at_axis) {
  if (compute_at_axis < 0)
    compute_at_axis += (int)consumer->nDims() + 1;
  TORCH_INTERNAL_ASSERT(
      compute_at_axis >= 0 &&
          (unsigned int)compute_at_axis <= consumer->nDims(),
      "Invalid axis in transform replayPasC.");

  // Consumer in rfactor cases is based off producer's rfactor root, not
  // producer's root
  TensorDomain* producer_rfactor_root = TransformIter::getRFactorRoot(producer);

  // Want full consumer root, even before rfactor
  TensorDomain* consumer_root = TransformIter::getRoot(consumer);

  // We want to see which axes in the consumer root were modified to create axes
  // < compute_at_axis
  std::vector<bool> consumer_influence(consumer->nDims(), false);
  for (int i = 0; i < compute_at_axis; i++)
    consumer_influence[i] = true;

  // Check which axes in ref_root need to be modified to honor transformations
  // to compute at axis
  std::vector<bool> consumer_root_influence =
      TransformIter::getRootInfluence(consumer, consumer_influence);

  // We have the influence on the consumer root, we need it on producer, we
  // want to keep those axes that don't need to be modified by the replay
  std::vector<bool> producer_rfactor_root_influence(
      producer_rfactor_root->nDims(), false);

  // Map is based on producer
  std::vector<int> replay_axis_map(consumer_root->nDims(), -1);
  // Setup producer_rfactor_root_influence vector on root for replay
  decltype(producer_rfactor_root_influence.size()) ip = 0, ic = 0;

  while (ip < producer_rfactor_root_influence.size() &&
         ic < consumer_root->nDims()) {
    bool is_reduction = producer_rfactor_root->axis(ip)->isReduction();
    if (is_reduction) {
      producer_rfactor_root_influence[ip++] = false;
    } else {
      if (consumer_root_influence[ic]) {
        replay_axis_map[ic] = ip;
      } else {
        replay_axis_map[ic] = -1;
      }
      producer_rfactor_root_influence[ip++] = consumer_root_influence[ic++];
    }
  }

  for (decltype(producer_rfactor_root->nDims()) i{0};
       i < producer_rfactor_root->nDims();
       i++)
    TORCH_INTERNAL_ASSERT(
        !(producer_rfactor_root_influence[i] &&
          producer_rfactor_root->axis(i)->isRFactorProduct()),
        "An illegal attempt to modify an rfactor axis detected.");

  // We should have hit the end of the consumer root domain
  TORCH_INTERNAL_ASSERT(
      ic == consumer_root->nDims(),
      "Error when trying to run replay, didn't reach end of consumer/target root.");

  TORCH_INTERNAL_ASSERT(
      producer_rfactor_root_influence.size() == producer_rfactor_root->nDims(),
      "Error detected during replay, expected matching sizes of influence map to root dimensions.");

  auto producer_root_influence = TransformIter::getRootInfluence(
      producer_rfactor_root, producer_rfactor_root_influence);

  TensorDomain* producer_root = TransformIter::getRoot(producer_rfactor_root);

  std::vector<int> producer_replay_map(producer_root->nDims());
  for (decltype(producer_replay_map.size()) i{0};
       i < producer_replay_map.size();
       i++) {
    if (producer_root->axis(i)->isRFactorProduct()) {
      producer_replay_map[i] = i;
    } else {
      producer_replay_map[i] = producer_root_influence[i] ? -1 : i;
    }
  }

  // Replay axes that won't be modified by transform replay
  TensorDomain* producer_replay_root = TransformIter::replaySelf(
      producer, TransformIter::getHistory(producer), producer_replay_map);

  // Record axes positions.
  std::unordered_map<IterDomain*, int> new_position;
  for (decltype(producer_replay_root->nDims()) i{0};
       i < producer_replay_root->nDims();
       i++)
    new_position[producer_replay_root->axis(i)] = i;

  std::unordered_map<int, int> root_axis_map;
  // reorder producer_replay_root to respect replay_axis_map
  for (decltype(replay_axis_map.size()) i{0}; i < replay_axis_map.size(); i++) {
    if (replay_axis_map[i] == -1)
      continue;
    auto ax = producer_root->axis(replay_axis_map[i]);
    TORCH_INTERNAL_ASSERT(
        new_position.find(ax) != new_position.end(),
        "Error hit during transform replay, could not find ",
        ax,
        " expected in root domain.");
    root_axis_map[new_position[ax]] = replay_axis_map[i];
  }

  // root_axis_map is now mapping from producer_replay_root -> consumer_root
  // Take producer_replay_root transform for all modified axes are in correct
  // relative order, matching how it was in replay_axis_map
  producer_replay_root = producer_replay_root->reorder(root_axis_map);

  // Finally replay producer as consumer on marked axes
  TensorDomain* replayed = TransformIter::replay(
      producer_replay_root,
      TransformIter::getHistory(consumer),
      replay_axis_map);

  TORCH_INTERNAL_ASSERT(
      std::none_of(
          replayed->domain().begin(),
          replayed->domain().begin() + compute_at_axis,
          [](IterDomain* id) { return id->isReduction(); }),
      "Reduction axes found within compute_at_axis in replay of producer.");
  return replayed;
}

// Replay consumer as producer.
TensorDomain* TransformReplay::replayCasP(
    TensorDomain* consumer,
    TensorDomain* producer,
    int compute_at_axis) {
  if (compute_at_axis < 0)
    compute_at_axis += (int)producer->nDims() + 1;
  TORCH_INTERNAL_ASSERT(
      compute_at_axis >= 0 &&
          (unsigned int)compute_at_axis <= producer->nDims(),
      "Invalid axis in transform replayPasC.");

  // Want producer root with no reductions, rfactor included
  TensorDomain* producer_rfactor_root = TransformIter::getRFactorRoot(producer);
  TensorDomain* producer_root = TransformIter::getRoot(producer);
  // Producer root still has reductions

  // Want full consumer root, even before rfactor
  TensorDomain* consumer_root = TransformIter::getRoot(consumer);

  // We want to see which axes in the producer root were modified to create axes
  // < compute_at_axis
  std::vector<bool> producer_influence(producer->nDims(), false);
  for (int i = 0; i < compute_at_axis; i++)
    producer_influence[i] = true;

  // Check which axes in ref_root need to be modified to honor transformations
  // to compute at axis
  std::vector<bool> producer_root_influence =
      TransformIter::getRootInfluence(producer, producer_influence);

  for (decltype(producer_root->nDims()) i{0}; i < producer_root->nDims(); i++) {
    TORCH_INTERNAL_ASSERT(
        !(producer_root_influence[i] && producer_root->axis(i)->isReduction()),
        "Error during replay, likely due to an illegal bad computeAt.");
  }

  std::vector<bool> producer_rfactor_root_influence =
      TransformIter::replayInfluence(
          TransformIter::getHistory(producer_rfactor_root),
          producer_root_influence);

  // We have the influence on the producer root, we need it on consumer, we
  // want to keep those axes that don't need to be modified by the replay
  std::vector<bool> consumer_root_influence(
      consumer->rootDomain()->nDims(), false);

  // Producer -> consumer axis map
  std::vector<int> replay_axis_map(producer_rfactor_root->nDims(), -1);

  // Setup consumer_root_influence vector on root for replay
  decltype(consumer_root_influence.size()) ip = 0, ic = 0;
  while (ic < consumer_root_influence.size() &&
         ip < producer_rfactor_root->nDims()) {
    bool is_reduction = producer_rfactor_root->axis(ip)->isReduction();
    if (is_reduction) {
      replay_axis_map[ip++] = -1;
      continue;
    }
    if (producer_rfactor_root_influence[ip]) {
      replay_axis_map[ip] = ic;
    } else {
      replay_axis_map[ip] = -1;
    }
    consumer_root_influence[ic++] = producer_rfactor_root_influence[ip++];
  }

  // Unlike PasC if last axes in producer_rfactor_root is a reduction we won't
  // be guarneteed that ip == producer_rfactor_root->nDims(), that's why we
  // initialize replay_axis_map with -1

  TORCH_INTERNAL_ASSERT(
      consumer_root_influence.size() == consumer_root->nDims(),
      "Error detected during replay, expected matching sizes of influence map to root dimensions.");

  std::vector<int> consumer_replay_map(consumer_root->nDims());
  for (decltype(consumer_replay_map.size()) i{0};
       i < consumer_replay_map.size();
       i++) {
    if (consumer_root->axis(i)->isRFactorProduct()) {
      consumer_replay_map[i] = i;
    } else {
      consumer_replay_map[i] = consumer_root_influence[i] ? -1 : i;
    }
  }

  // Replay axes that won't be modified by transform replay
  TensorDomain* consumer_replay_root = TransformIter::replaySelf(
      consumer, TransformIter::getHistory(consumer), consumer_replay_map);

  // Record axes positions.
  std::unordered_map<IterDomain*, int> new_position;
  for (decltype(consumer_replay_root->nDims()) i{0};
       i < consumer_replay_root->nDims();
       i++)
    new_position[consumer_replay_root->axis(i)] = i;

  std::unordered_map<int, int> root_axis_map;
  // reorder consumer_replay_root to respect replay_axis_map
  for (decltype(replay_axis_map.size()) i{0}; i < replay_axis_map.size(); i++) {
    if (replay_axis_map[i] == -1)
      continue;
    auto ax = consumer_root->axis(replay_axis_map[i]);
    TORCH_INTERNAL_ASSERT(
        new_position.find(ax) != new_position.end(),
        "Error hit during transform replay, could not find ",
        ax,
        " expected in root domain.");
    root_axis_map[new_position[ax]] = replay_axis_map[i];
  }

  auto replay_history = TransformIter::getHistory(producer);
  auto rfactor_history = TransformIter::getHistory(producer_rfactor_root);
  replay_history.erase(
      replay_history.begin(), replay_history.begin() + rfactor_history.size());

  consumer_replay_root = consumer_replay_root->reorder(root_axis_map);
  // Finally replay consumer as producer on marked axes

  auto replayed = TransformIter::replay(
      consumer_replay_root, replay_history, replay_axis_map);

  return replayed;
}

// replay Producer as Consumer
TensorView* TransformReplay::replayPasC(
    TensorView* producer,
    TensorView* consumer,
    int compute_at_axis) {
  // If this is a reduction operation, we may call transform_replay on the same
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
