#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <ir_internal_nodes.h>
#include <maxinfo_propagator.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

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

class TensorDomain;
class TensorView;
class RootDomainMap;

class TORCH_CUDA_CU_API TransformReplay {
 public:
  // Replay producer as consumer, returns {producer, producer_compute_at_axis}.
  static std::pair<TensorDomain*, unsigned int> replayPasC(
      const TensorView* producer,
      const TensorView* consumer,
      int consumer_compute_at_axis,
      bool replay_swizzle = false);
  static std::pair<TensorDomain*, unsigned int> replayPasC(
      const TensorView* producer,
      const TensorView* consumer,
      int consumer_compute_at_axis,
      const RootDomainMap& root_map,
      bool replay_swizzle = false);

  // Replay producer as consumer, returns {replayed_consumer_domain,
  // consumer_compute_at_axis}.
  static std::pair<TensorDomain*, unsigned int> replayCasP(
      const TensorView* consumer,
      const TensorView* producer,
      int producer_compute_at_axis);
  static std::pair<TensorDomain*, unsigned int> replayCasP(
      const TensorView* consumer,
      const TensorView* producer,
      int producer_compute_at_axis,
      const RootDomainMap& root_map);

  // Self replay.
  static TensorDomain* fullSelfReplay(
      const TensorDomain* new_self_root,
      const TensorDomain* self);

  // Returns the leaf position in producer that matches with `consumer_pos` in
  // consumer. Returns -1 if matching is impossible. This function can be used
  // to test if replay is needed for getting matching outer dims. This function
  // should be consistent with `replayPasC`: if you pass the tensors just
  // replayed by replayPasC as inputs, you should return exactly the same
  // position as `replayPasC`. However, this function is more tolerant than
  // fully matching `replayPasC`: if in the consumer, there are unmappable
  // dimensions, these dimensions are just ignored.
  static int getMatchedLeafPosWithoutReplayPasC(
      const TensorView* producer,
      const TensorView* consumer,
      int consumer_pos);

  // Returns the leaf position in consumer that matches with `producer_pos` in
  // producer. Behavior similar to getMatchedLeafPosWithoutReplayPasC, except
  // that we are also ignoring reductions in the producer.
  static int getMatchedLeafPosWithoutReplayCasP(
      const TensorView* consumer,
      const TensorView* producer,
      int producer_pos);

  // tests if two tensors has fully matching transformations
  static bool fullSelfMatching(
      const TensorView* replay,
      const TensorView* target);
};

class TORCH_CUDA_CU_API TransformPropagator
    : public MaxRootDomainInfoSpanningTree::Propagator {
 protected:
  std::unordered_map<TensorView*, size_t> replayed_pos_;

 public:
  virtual void propagateC2P(TensorView* from, TensorView* to) override;
  virtual void propagateP2C(TensorView* from, TensorView* to) override;
  virtual void propagateSibling(TensorView* from, TensorView* to) override;
  TransformPropagator(TensorView* from, int64_t pos = -1);
};

struct TORCH_CUDA_CU_API MostInlinedTransformPropagator
    : public MaxRootDomainInfoSpanningTree::Propagator {
  virtual void propagateC2P(TensorView* from, TensorView* to) override;
  virtual void propagateP2C(TensorView* from, TensorView* to) override;
  virtual void propagateSibling(TensorView* from, TensorView* to) override;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
