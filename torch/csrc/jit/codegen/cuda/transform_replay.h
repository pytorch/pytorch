#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <algorithm>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

/*
 * compute_at is a relative property between two TensorViews which marks at what
 * iteration domain we're going to generate a tensor to be consumed by another.
 * For example if we have: T2[I, J, K] = T1[I, J, K] * 2.0 and then we call
 * T2.split(axis = 0, factor = ...): T2[Io, Ii, J, K] = T1[I, J, K] * 2.0 where
 * Io is the outer axes from the split, and Ii is the inner axes from the split.
 * then we call T1.compute_at(T2, axis=1) we would expect to have:
 * T2[Io, Ii, J, K] = T1[Io, Ii, J, K] * 2.0
 * which would produce the following loop nest structure:
 *
 * for(io : Io)
 *  for(ii : Ii)
 *   for(j : J)
 *    for(k : K)
 *     //produce T1:
 *     T1[io, ii, j, k] = ...
 *  for(ii : Ii)
 *   for(j : J)
 *    for(k : K)
 *     //consume T1, produce T2
 *     T2[io, ii, j, k] = T1[io, ii, j, k] * 2.0
 *
 * This file provides the replay function that allows us to construct T1's
 * domain from T2 at a desired level (compute_at_axis) without modifying any
 * unnecessary parts of the domain.
 *
 * EXAMPLES:
 *
 * ANOTHER ITER EXAMPLE:
 *   T2[I, J, K] = T1[I, J, K] * 2.0
 * T2.split(axis = 0, factor = ...)
 *   T2[Io, Ii, J, K] = T1[I, J, K] * 2.0
 * T2.split(axis = 2, factor = ...)
 *   T2[Io, Ii, Jo, Ji, K] = T1[I, J, K] * 2.0
 * T1.compute_at(T2, axis=1)
 *   T2[Io, Ii, Jo, Ji, K] = T1[Io, Ii, J, K] * 2.0
 *
 * Note: compute_at axis:
 * T2[ 0 Io, 1 Ii, 2 Jo, 3 Ji, 4 K 5 ] //5 is inline, 0 is at "root" which means
 * completely separate loop nests.
 *
 * for(io : Io)
 *  for(ii : Ii)
 *   for(j : J)
 *    for(k : K)
 *     //produce T1, this is the view that replay generates:
 *     T1[io, ii, j, k] = ...
 *  for(ii : Ii)
 *   for(jo : Jo)
 *     for(ji : Ji)
 *      for(k : K)
 *       //consume T1, produce T2
 *       T2[io, ii, jo, ji, k] = T1[io, ii, jo, ji, k] * 2.0
 *       //consumer view on T1 will be produced at a later stage.
 *
 *
 * SIMPLE REDUCTION EXAMPLE:
 *   T1[I, J, K] = ...
 *   T2[I, R, K] = T1[I, J, K] //.sum(axis = 1), we reduce on R/J to produce
 * T2[I, K] T2.split(axis = 0, factor = ...) T2[Io, Ii, R, K] = T1[I, J, K]
 * T1.compute_at(T2, axis=3)
 *   T2[Io, Ii, R, K] = T1[Io, Ii, J, K]
 *
 * for(io : Io)
 *  for(ii : Ii)
 *   for(k : K)
 *    T2[io, ii, k] = init
 *   for(r : R)
 *    for(k : K)
 *     //produce T1:
 *     T1[io, ii, r, k] = ...
 *     //consume T1 produce T2:
 *     T2[io, ii, k] += T1[io, ii, r, k]
 *
 *
 * REDUCTION EXAMPLE RESULTING IN AN ERROR:
 *   T1[I, R, K] = ... //R is reduction domain, we reduce on R to produce T1[I,
 * K] T2[I, K] = T1[I, K]
 *
 * for(i : I)
 *   for(k : K)
 *     T1[i, k] = init
 *   for(r : R)
 *     for(k : K)
 *       T1[i, k] += ...[i, r, k]
 * for(i : I)
 *   for(k : K)
 *     T2[i, k] = T1[i, k]
 *
 * T1.compute_at(T2, axis=2)
 * This should be an error, or a warning and changed to:
 * T1.compute_at(T2, axis=1)
 * The error is because the kernel would have to be:
 *
 * for(i : I)
 *   T1[i, k] = init
 *   for(r : R)
 *     for(k : K)
 *       T1[i, k] += ...[i, r, k]
 *   for(k : K)
 *     T2[i, k] = T1[i, k]
 *
 * Otherwise we would produce incorrect results.
 *
 */

struct TORCH_CUDA_API TransformReplay : public TransformIter {
 private:
  /*
   * Functions to backward propagate influence from split/merge/reorder
   */
  void replayBackward(Split* expr);
  void replayBackward(Merge* expr);
  void replayBackward(Reorder* expr);

  // Entry for backward influence propagation on td following record
  TensorDomain* replayBackward(TensorDomain* td, bool generate_record = false);

  /*
   * Replay functions, takes a TensorView and steps through the operations in
   * "record" based on influence axes. Will also update influence and propagate
   * it forward.
   */
  TensorDomain* replay(Split* expr, TensorDomain* tv);
  TensorDomain* replay(Merge* expr, TensorDomain* tv);
  TensorDomain* replay(Reorder* expr, TensorDomain* tv);

  /*
   * Takes replay_ref and replays its transformations on replay_target
   * Replays from begining of both TensorDomains. could be more efficient to try
   * and find a common ancestor to start from, but likely not a worthwhile
   * optimization.
   */
  TensorDomain* runReplay(
      TensorDomain* replay_ref,
      TensorDomain* replay_target,
      int compute_at_axis);

  /*
   * Takes replay_ref and replays its transformations on replay_target
   * Replays from begining of both TensorDomains. could be more efficient to try
   * and find a common ancestor to start from, but likely not a worthwhile
   * optimization.
   */
  TensorView* runReplay(
      TensorView* replay_ref,
      TensorView* replay_target,
      int compute_at_axis);

  // Running influence vector
  std::vector<bool> influence;

  // compute_at_axis
  int compute_at_axis;

  // In the replay we won't apply all transformations, but will need relative
  // axes for later transformations. axis_map[full transform position] = partial
  // transform position Full transform position is relative to if we played all
  // transformations if full transform position is not in partial transform
  // position it will return -1
  // axis_map[fake_pos] = real_pos
  std::vector<int> axis_map;

 public:
  static TensorView* replay(
      TensorView* replay_ref,
      TensorView* replay_target,
      int compute_at_axis);

  static TensorView* fullReplay(
      TensorView* replay_ref,
      TensorView* replay_target);

  static TensorDomain* fullReplay(
      TensorDomain* replay_ref,
      TensorDomain* replay_target);
};

} // namespace fuser
} // namespace jit
} // namespace torch
