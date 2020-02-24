#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/iter_visitor.h>
#include <torch/csrc/jit/fuser/common/tensor.h>

#include <algorithm>
#include <vector>

// For debug:
/*
#include <torch/csrc/jit/fuser/common/iriostream.h>
*/
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
 *   T1[I, J, K] = ...
 *   T2[I, R, K] = T1[I, J, K] //.sum(axis = 1)   //Where R is a reduction axis
 * T2.split(axis = 0, factor = ...)
 * //R == J except R.isReduction() == true
 *   T2[Io, Ii, R, K] = T1[I, J, K]
 * T1.compute_at(T2, axis=2)
 *   T2[Io, Ii, R, K] = T1[Io, Ii, J, K]
 *
 * for(io : Io)
 *  for(ii : Ii)
 *   for(j : J)
 *    for(k : K)
 *     //produce T1:
 *     T1[io, ii, j, k] = ...
 *   for(k: K)
 *    //Must be before all reduction axes
 *    T2[io, ii, k] = 0.0
 *   for(r : R)
 *    for(k : K)
 *     //consume T1, produce T2
 *     T2[io, ii, k] += T1[io, ii, r, k]
 *
 *   T1[I, J, K] = ...
 *   T2[I, R, K] = T1[I, J, K] //.sum(axis = 1)
 * T2.split(axis = 0, factor = ...)
 *   T2[Io, Ii, R, K] = T1[I, J, K]
 * T1.compute_at(T2, axis=3)
 *   T2[Io, Ii, R, K] = T1[Io, Ii, J, K]
 *
 * Is this an error? We're using a value before we reduce it.
 * But we're only using it to reduce it. I'm pretty sure TE/Halide
 * would not work with this, but it seems like we could generate
 * reasonable code from this...
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
 */

struct TORCH_API TransformReplay : public IterVisitor {
 private:
  /*
   * Functions to backward propagate influence from split/merge/reorder
   */
  void compute_influence(Split* expr);
  void compute_influence(Merge* expr);
  void compute_influence(Reorder* expr);

  // Backward influence propagate dispatch
  void compute_influence(Expr* expr);

  // Entry for backward influence propagation on td following record
  void compute_influence(TensorDomain* td);

  /*
   * Replay functions, takes a TensorView and steps through the operations in
   * "record" based on influence axes. Will also update influence and propagate
   * it forward.
   */
  TensorView* replay(Split* expr, TensorView* tv);
  TensorView* replay(Merge* expr, TensorView* tv);
  TensorView* replay(Reorder* expr, TensorView* tv);

  // Dispatch for replay functions
  TensorView* replay(Expr* expr, TensorView* tv);

  // Entry point for replay on a TensorView, will relpay all ops from "replay"
  TensorView* replay(TensorView* target);

  /*
   * Takes replay_ref and replays its transformations on replay_target
   * Replays from begining of both TensorDomains. could be more efficient to try
   * and find a common ancestor to start from, but that's outside the scope of
   * this work for now.
   *
   */
  TensorView* runReplay(
      TensorView* replay_ref,
      TensorView* replay_target,
      int compute_at_axis);

  // Trace back the history of td, record the Expr's that made this td (split,
  // merge, reorder)
  TensorDomain* get_root(TensorDomain* td, bool create_record = false);

  // Forward record from root, to replay_ref/ref_root
  std::vector<Expr*> record;

  // Running influence vector
  std::vector<bool> influence;

  // compute_at_axis
  int compute_at_axis;

  // In the replay we won't apply all transformations, but will need relative
  // axes for later transformations. axis_map[full transform position] = partial
  // transform position Full transform position is relative to if we played all
  // transformations if full transform position is not in partial transform
  // position it will return -1
  //axis_map[fake_pos] = real_pos
  std::vector<int> axis_map;

 public:
  static TensorDomain* getRoot(TensorDomain* td) {
    TransformReplay tr;
    return tr.get_root(td);
  }

  static TensorView* replay(
      TensorView* replay_ref,
      TensorView* replay_target,
      int compute_at_axis);
};

} // namespace fuser
} // namespace jit
} // namespace torch