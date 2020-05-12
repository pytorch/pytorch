#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

namespace torch {
namespace jit {
namespace fuser {

/*
 * Functions to backward propagate influence from split/merge/reorder
 */
void TransformReplay::replayBackward(Split* expr) {
  int axis = expr->axis();
  TORCH_INTERNAL_ASSERT(
      axis + 1 < influence.size(),
      "Error during replay backwards, influence is not sized correctly.");
  influence[axis] = influence[axis] | influence[axis + 1];
  influence.erase(influence.begin() + axis + 1);
}

void TransformReplay::replayBackward(Merge* expr) {
  int axis = expr->axis();
  TORCH_INTERNAL_ASSERT(
      axis < influence.size(),
      "Error during replay backwards, influence is not sized correctly.");
  influence.insert(influence.begin() + axis + 1, influence[axis]);
}

void TransformReplay::replayBackward(Reorder* expr) {
  // pos2axis[new_pos] = old_pos Generate new axis2pos map
  const std::vector<int>& pos2axis = expr->pos2axis();

  std::vector<bool> reorder_influence(influence.size(), false);
  for (decltype(pos2axis.size()) i = 0; i < pos2axis.size(); i++) {
    int new_pos = i;
    int old_pos = pos2axis[i];
    TORCH_INTERNAL_ASSERT(
        new_pos < influence.size() && old_pos < reorder_influence.size(),
        "Error during replay backwards, influence is not sized correctly.");
    reorder_influence[old_pos] = influence[new_pos];
  }

  influence = reorder_influence;
}

// Entry for backward influence propagation on td following record
TensorDomain* TransformReplay::replayBackward(
    TensorDomain* td,
    bool create_record) {
  influence = std::vector<bool>(td->nDims(), false);
  for (int i = 0; i < compute_at_axis; i++)
    influence[i] = true;
  return TransformIter::runBackward(td, create_record);
}

/*
 * Replay functions, takes a TensorDomain and steps through the operations in
 * "record" based on influence axes. Will also update influence and propagate
 * it forward.
 */
TensorDomain* TransformReplay::replay(Split* expr, TensorDomain* td) {
  int axis = expr->axis();
  bool run_split = influence[axis];

  // Propagate influence
  influence.insert(influence.begin() + axis + 1, influence[axis]);

  // Forward prop influence
  if (run_split) {
    // Make sure split axis is real.
    int real_axis = axis_map[expr->axis()];
    TORCH_INTERNAL_ASSERT(
        real_axis != -1,
        "During transformation replay attempted to split an imaginary axis.");
    TORCH_INTERNAL_ASSERT(
        td->axis(real_axis)->start()->isZeroInt(),
        "Transform Replay tried to split an IterDomain with a start value that is not 0,",
        " this is not currently supported.");
    // Inserted a real axis, push everything in axis_map over to the right
    // after this inserted axis
    for (decltype(axis_map.size()) i{0}; i < axis_map.size(); i++)
      if (axis_map[i] > real_axis)
        axis_map[i] = axis_map[i] + 1;

    axis_map.insert(
        axis_map.begin() + expr->axis() + 1,
        real_axis + 1); // insert axis at position axis.

    // Replay split
    return td->split(real_axis, *(expr->factor()->value()));
  } else {
    // Fake it
    axis_map.insert(axis_map.begin() + expr->axis() + 1, -1);
  }

  return td;
}

TensorDomain* TransformReplay::replay(Merge* expr, TensorDomain* td) {
  int axis = expr->axis();
  bool merge = influence[axis] || influence[axis + 1];
  axis_map.erase(axis_map.begin() + expr->axis() + 1);

  for (decltype(axis_map.size()) i = expr->axis() + 1; i < axis_map.size(); i++)
    if (axis_map[i] != -1)
      axis_map[i]--;

  // Forward prop influence
  influence[axis] = influence[axis] || influence[axis + 1];
  influence.erase(influence.begin() + axis + 1);

  if (merge) {
    // Make sure both merge axes are real.
    TORCH_INTERNAL_ASSERT(
        axis_map[axis] != -1 && axis_map[axis + 1] != -1,
        "During transformation replay attempted to merge an imaginary axis.");
    // Replay merge
    TORCH_INTERNAL_ASSERT(
        td->axis(axis)->start()->isZeroInt() &&
            td->axis(axis + 1)->start()->isZeroInt(),
        "Transform Replay tried to Merge IterDomains with a start value that is not 0,",
        " this is not currently supported.");
    return td->merge(axis_map[axis]);
  } else {
    // If we aren't applying the merge, we won't change any following axis
    // Doesn't matter which axis we propagate for the merge in the axis_map
    assert(axis_map[axis + 1] == -1);
    return td;
  }
}

TensorDomain* TransformReplay::replay(Reorder* expr, TensorDomain* td) {
  // axis2pos[old_pos] = new_pos is sent to reorder, Reorder holds
  // pos2axis[new_pos] = old_pos Generate new axis2pos map
  std::unordered_map<int, int> axis2pos;
  const std::vector<int>& pos2axis = expr->pos2axis();

  std::vector<int> reordered_axis_map(axis_map.size(), -1);
  std::vector<bool> reordered_influence(pos2axis.size(), false);

  // We have
  // axis_map[old_fake_pos] -> old_real_pos
  // pos2axis[new_fake_pos] -> old_fake_pos
  // f2r[new_fake_pos] -> new_real_pos
  //
  // We want:
  // axis2pos[old_real_pos] -> new_real_pos
  // axis_map[new_fake_pos] -> new_real_pos

  std::vector<std::pair<int, int>> needed_real_reorder;
  for (decltype(pos2axis.size()) i{0}; i < pos2axis.size(); i++) {
    int new_fake_axis = i;
    int old_fake_axis = pos2axis[i];
    int old_real_axis = axis_map[old_fake_axis];
    bool is_real_axis = old_real_axis != -1;
    // If a real axis
    if (is_real_axis)
      if (influence[old_fake_axis]) {
        needed_real_reorder.push_back({old_real_axis, new_fake_axis});
      }
  }

  // Sort needed_real_reorder by new_fake_axis.
  std::sort(
      needed_real_reorder.begin(),
      needed_real_reorder.end(),
      [](std::pair<int, int> a, std::pair<int, int> b) -> bool {
        return a.second < b.second;
      });

  // axis2pos[old_real_axis] -> new_real_axis
  int axis = 0;
  for (auto entry : needed_real_reorder) {
    axis2pos[entry.first] = axis++;
  }

  for (decltype(td->nDims()) i{0}; i < td->nDims(); i++) {
    if (axis2pos.find(i) == axis2pos.end())
      axis2pos[i] = axis++;
  }

  // replay reorder
  TensorDomain* reordered_td = td->reorder(axis2pos);

  // Fake transform:
  for (decltype(pos2axis.size()) i = 0; i < pos2axis.size(); i++) {
    int new_pos = i;
    int old_pos = pos2axis[i];
    // Forward prop influence
    reordered_influence[new_pos] = influence[old_pos];
    if (axis_map[old_pos] != -1)
      reordered_axis_map[new_pos] = axis2pos[axis_map[old_pos]];
  }
  influence = reordered_influence;
  axis_map = reordered_axis_map;

  return reordered_td;
}

/*
 * TODO: When we compare root axes, we should ignore reduction axes in the
 * producer. Reduction axes are owned by a consumer.
 *
 * TODO: We should be able to relax the constraints of replay a bit. Right now
 * it requires that the root domain of the target and replay are completely
 * the same. However, we should only require that the root derived from the
 * axes < compute_at_axis match. We could even go further and look for those
 * matching axes as they don't necessairly need to be in the same order.
 * However, once they're replayed they should be.
 *
 * 1) Take the reference, trace back its domain history to get all the
 * split/merge/reorder calls, as well as its original domain. Get the
 * original domain of the target as well.
 *
 * 2) We only need compute_at_axis and earlier dimensions to match for
 * compute_at. Therefore, we want to find all original axes that must have
 * been modified in order to produce the axes below compute_at_axis. We take a
 * bool vector called influence, and mark axes below compute_at_axis as true,
 * and all others as false. This vector is propagated up through
 * split/merge/reorder if split/merge/reorder output a marked axis, their
 * input will be marked as well. This marks all original axes required to be
 * modified to produce the axes below compute_at_axis.
 *
 * 3) We take the ordered list of split/merge/reorder and the influence vector
 * on the inputs and we apply split/merge/reorder operations on the
 * replay_target. We also forward propagate the influence vector again (as this
 * time it could be different than originally marked), a map from "fake axes"
 * (refrence axes corresponding to the full replay) to real axes (axes produced
 * by running the selected split/merge/reorder operations). Reorder replay's can
 * actually be partial and non-equivelent to the original, as some axes may
 * never have been produced based on split, and we don't want to reorder axes
 * outside of compute_at_axis.
 *
 */
TensorDomain* TransformReplay::runReplay(
    TensorDomain* replay_ref,
    TensorDomain* replay_target,
    int compute_at_axis) {
  if (compute_at_axis < 0)
    compute_at_axis += int(replay_ref->nDims()) + 1;

  TORCH_CHECK(
      compute_at_axis >= 0 && compute_at_axis < int(replay_ref->nDims()) + 1,
      "Transform replay cannot be performed as the compute_at_axis is not in the valid range, it should be 0 or greater, and less than ",
      int(replay_ref->nDims()) + 1,
      ".");

  this->compute_at_axis = compute_at_axis;

  /* STEP 1 */
  // Reset the tensor domain of the target, this is the only way we can be
  // certain That we can actually replay the ops of ref.
  // Trace back to the root TensorDomain's of ref and target
  replay_target = replay_target->rootDomain();

  /* STEP 2 */
  // Mark compute_at_axis and below as "influenced", trace back through
  // operations, and map these axes to the ref root axis that were modified to
  // produce these axis
  // As we trace the ref, record the operations to go from replay_ref ->
  // ref_root, save in "record"
  TensorDomain* ref_root = replayBackward(replay_ref, true);
  // We're going to save a copy of this vector, class member influnce will be
  // used during replay to forward propagate influence.
  std::vector<bool> root_influence_vector = influence;

  // Remove isReduction from the axis_map of a producer
  // isReduction is only impactful when its on a consumer
  auto init_size = replay_target->nDims();
  for (decltype(init_size) i = 0; i < init_size; i++)
    if (!replay_target->axis(i)->isReduction())
      axis_map.push_back(i);

  // Domain sizes must match at root for replay.
  if (axis_map.size() != ref_root->nDims()) {
    std::stringstream err_msg;
    err_msg
        << "Transforms cannot be replayed as source and destinations do not have the same root sizes."
        << " " << ref_root << " vs " << replay_target->domain() << std::endl;
    TORCH_CHECK(false, err_msg.str());
  }

  /*
   * TODO: The JIT graph has symbolic sizes, so inputs may actually have the
   * same sizes (assuming no broadcasts/reductions), we at some point want to
   * have some size matching, and sizes should actually match at this point, but
   * the check below won't work.
   */

  // for (decltype(axis_map.size()) i{0}; i < axis_map.size(); i++) {
  //   TORCH_CHECK(
  //       ref_root->axis(i)->size()->same_as(
  //           target_root->axis(axis_map[i])->size()),
  //       "Transforms cannot be replayed as source and destinations do not have
  //       the same root sizes.");
  // }

  /* STEP 3 */
  // Replay operations while forward propagating influence. The resulting
  // influence can be different in forward propagation, than in backward
  // propagation depending on the combination of merge/split/reorder nodes
  // There are multiple things we have to track here. We need to track
  // the propagation of axes for all operations, though we only want to
  // actually execute those based on influence. If we didn't track all
  // axes, we wouldn't know what axis split/merge/reorder are referencing
  // as they're relative to the "full" replay that produced the reference.
  TensorDomain* replayed = TransformIter::runReplay(replay_target);

  for (decltype(replayed->nDims()) i{0}; i < compute_at_axis; i++)
    if (replayed->axis(i)->isReduction())
      TORCH_CHECK(
          false,
          "Generated a compute_at dependency where a reduction would be used before computed.");

  return replayed;
}

TensorView* TransformReplay::runReplay(
    TensorView* replay_ref,
    TensorView* replay_target,
    int compute_at_axis) {
  TensorDomain* td =
      runReplay(replay_ref->domain(), replay_target->domain(), compute_at_axis);
  replay_target->setDomain(td);
  return replay_target;
}

TensorView* TransformReplay::replay(
    TensorView* replay_ref,
    TensorView* replay_target,
    int compute_at_axis) {
  TransformReplay tr;
  tr.runReplay(replay_ref, replay_target, compute_at_axis);
  return replay_target;
}

TensorView* TransformReplay::fullReplay(
    TensorView* replay_ref,
    TensorView* replay_target) {
  TransformReplay tr;
  return tr.runReplay(replay_ref, replay_target, -1);
}

TensorDomain* TransformReplay::fullReplay(
    TensorDomain* replay_ref,
    TensorDomain* replay_target) {
  TransformReplay tr;
  return tr.runReplay(replay_ref, replay_target, -1);
}

} // namespace fuser
} // namespace jit
} // namespace torch
