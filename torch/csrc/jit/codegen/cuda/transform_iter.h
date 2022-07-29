#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Enable pair<IterDomain*, size_t> in a set, size_t must be unique in set
struct id_int_lt {
  bool operator()(
      const std::pair<IterDomain*, size_t>& first,
      const std::pair<IterDomain*, size_t>& second) const {
    return first.second < second.second;
  }
};

} // namespace

// Uses the history of _target_domain, and replays that history using the
// provided map.
//
// target_domain contains the history we want replayed.
//
// id_map maps IterDomains in that history to the IterDomains we want it
// replayed on.
//
// error_on_failure = true will cause the replay to error if we can't replay any
// operation in target_domain's history due to missing IDs in the id_map.
//
// If error_on_failure = false, replay will replay everything it can, and ignore
// operations it can't.
class TORCH_CUDA_CU_API ReplayTransformations : public IterVisitor {
 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const std::vector<IterDomain*>& target_domain_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_map<IterDomain*, IterDomain*> id_map_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_map<IterDomain*, size_t> leaf_ids_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<IterDomain*> leaf_vec_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  size_t counter = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool error_on_failure_ = true;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool ran_replay = false; // Mark if replay has been run
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool replay_swizzle_ = false;
  using IterVisitor::handle;

  // Transform dispatch
  void handle(Expr* e) override;

  // We're going to replay this split operation on the corresponding ID
  void handle(Split* s) override;

  // We're going to replay this merge operation on the corresponding IDs
  void handle(Merge* m) override;

  // We're going to replay this swizzle operation on the corresponding IDs
  //  if replaying swizzle is enabled.
  void handle(Swizzle2D* m) override;

 public:
  ReplayTransformations(
      const std::vector<IterDomain*>& _target_domain,
      std::unordered_map<IterDomain*, IterDomain*> _id_map,
      bool _error_on_failure = true,

      // Indicates if we want to replay swizzle ops on the replayed
      //  tensor.
      // The swizzle op will be replayed if true,
      // The swizzle inputs will be directly forwarded, and therefore skipping
      //  the swizzle op if false.
      // Currently this options should always be off but
      //  later we may have cases in scheduling large fusions where
      //  this functionality could be useful.
      bool replay_swizzle = false);

  // Replays outputs that were generated from ids.first on ids.second
  void runReplay();

  // Returns map from provided target domain to their corresponding IDs
  const std::unordered_map<IterDomain*, IterDomain*>& getReplay() {
    if (!ran_replay)
      runReplay();
    return id_map_;
  }

  // Returns leaf_ids_ the size_t marks the order in which they were put into
  // the map, this is part of the structure because it's used to generate the
  // order from 'getLeafIDs'
  const std::unordered_map<IterDomain*, size_t>& getUnorderedLeafIDs() {
    if (!ran_replay)
      runReplay();
    return leaf_ids_;
  }

  // Returns all terminating IDs that resulted from the replay. Leaf IDs are run
  // to run deterministic, but otherwise in no specific order.
  const std::vector<IterDomain*>& getLeafIDs() {
    if (!ran_replay)
      runReplay();
    return leaf_vec_;
  }
};

/*
 * Motivation:
 *
 * Consider the following program:
 *
 * T1[I0, R1] = T0[I0, I1]
 * T2[I0] = T1[I0, R1i]
 *
 * T1->split(1, factor)
 * T1->rFactor(2)
 *
 * T4[I0, R1orf, I1irf] = T0[I0, I1]
 * T1[I0, R1i] = T4[I0, R1orf, I1irf]
 * T2[I0] = T1[I0, R1i]
 *
 * There's an issue when we call replayCasP on
 * T4[I0, R1o, I1i] = T0[I0, I1]
 *
 * This would try to replay T4 as T0, and it could include the rfactor domains.
 * For example we compute T0 inline with T4. The way computeAt is setup this
 * would call replayPasC(T0, T4, -1) then repalyCasP(T4, T0, -1)
 *
 * We might assume that the only way we will hit this is if we call
 * T4->computeAt(T0...) so it might be safe to assume that the right
 * transformations would be replayed. However, we want to preserve the rfactor
 * domain, so since it would replay T4 at root, it would produce iterdomains
 * that wouldn't corresopnd to those in rfactor. Also, I don't know if this
 * assumption is correct.
 *
 * Therefore, we will assume it is not correct, and we will validate here that
 * if we replay a domain that it would transform it in a way consistent with
 * any defined RFactor domains, then we will update the replay map so that
 * RFactor roots are mapped to intermediate IterDomains  in the target and start
 * replay from there.
 *
 *
 * SHORT DESCRIPTION:
 *
 * This class will validate/do the above. It will also run through
 * transformations in target according to replay_map. If equal transformations
 * already exist in replay_domain history, we will not redo those
 * transformations, but instead update replay_map to reflect forwarding the
 * existing transformations. This later part is the "best effort" replay. Though
 * we include rfactor replay and validation here.
 *
 * Given an Expr in target_domain, check if its inputs are in replay_map. If so,
 * check if the mapped domain in replay_map are recorded to be transformed by an
 * equivelent operation in replay_domain's history. If so, "forward" the
 * operation and update replay_map to the outputs of target_domain's output(s),
 * to the output of the equivlent expr's outputs in relpay_domain's history.
 *
 * replay_map maps root IDs in the history of target_domain to root IDs in the
 * history replay_domain
 */

class TORCH_CUDA_CU_API BestEffortReplay {
 private:
  std::unordered_map<IterDomain*, IterDomain*> target2replay_id_map_;
  std::unordered_map<IterDomain*, IterDomain*> replay_forward_id_map_;
  std::unordered_map<IterDomain*, IterDomain*> target_forward_id_map_;
  std::unordered_map<IterDomain*, size_t> leaf_ids_;
  std::vector<IterDomain*> forwarded_ids_;

  // Need to track which id's have been forwarded. Later need to make sure leaf
  // nodes to produce compliment axes are properly tracked. i.e.
  // T[i0, b1, b2, i3]
  // -> T[i0, b1o, b1i, b2o, b2i, i3]
  // -> T[i0*b1i*b2o, b1o, b2i, i3]
  // -> T[i0*b1i*b2o*i3, b1o, b2i]
  // If we forwarded i0 -> i0*b1i*b2o*i3, we need to know that b1o and b2i
  // are leaf nodes even though their split wasn't part of targets replay.

  // Counter to make sure best effort replay leaf_ids can be grabbed
  // deterministicly
  size_t counter = 0;

  // Determine if current replay will ignore swizzle ops.
  // When not skipping swizzles, swizzle ops will have to be matched
  //  same way as split and merge to progress forward on the mapping.
  //
  // When skipping swizzles, mismatched swizzle ops will not stop matching
  //  further down the tensor domains but only the swizzle outputs will be on
  //  the target to replay map, since we only generate one-to-one maps in
  //  BestEffortReplay and the swizzle outputs is just picked as a convention
  //  for simpler and uniform mapping behavior. The swizzle op inputs will be
  //  added by the disjoint set passes when building the iterdomain graph.
  //
  // Example:
  //   Target:
  //     I0o, I0i   = split I0
  //     Ix0o, Ix0i = swizzle I0o, I0i
  //     I02        = merge Ix0o, Ix0i
  //   Replay:
  //     I1o, I1i = split I1
  //     I12      = merge I1o, I1i
  //
  //   BestEffortReplay **no** skip swizzle gives:
  //  {
  //   I0->I1,
  //   I0o->I1o,
  //   I0i->I1i,
  //  }
  //
  //   BestEffortReplay skip swizzle gives:
  //  {
  //    I0->I1,
  //    Ix0o->I1o,
  //    Ix0i->I1i,
  //    I02->I12
  //  }
  //
  bool skip_swizzle_ = true;

  bool inReplayForwardMap(IterDomain* id) const {
    return replay_forward_id_map_.find(id) != replay_forward_id_map_.end();
  }

  bool inTargetForwardMap(IterDomain* id) const {
    return target_forward_id_map_.find(id) != target_forward_id_map_.end();
  }

  IterDomain* getReplayForwardedId(IterDomain* id) const {
    auto forwarded_id_it = replay_forward_id_map_.find(id);
    if (forwarded_id_it == replay_forward_id_map_.end()) {
      return id;
    } else {
      return getReplayForwardedId(forwarded_id_it->second);
    }
  }

  IterDomain* getTargetForwardedId(IterDomain* id) const {
    auto forwarded_id_it = target_forward_id_map_.find(id);
    if (forwarded_id_it == target_forward_id_map_.end()) {
      return id;
    } else {
      return getTargetForwardedId(forwarded_id_it->second);
    }
  }

  //! Adds complimenting IDs of forwarded IDs to the leaf map
  void addComplimentLeafIDs(
      const std::unordered_map<IterDomain*, IterDomain*>& forwarding_map,
      const std::unordered_map<IterDomain*, std::vector<IterDomain*>>&
          compliment_map);

  // Skip swizzle step to make sure both target and
  //  replay swizzles are skipped while the mapping
  //  makes progress. This makes sure that, for example
  //  different tensors can still be inlined despite
  //  different local swizzle patterns.
  void skipSwizzles(
      const std::unordered_map<IterDomain*, Expr*>& target_id2expr,
      const std::unordered_map<IterDomain*, Expr*>& replay_id2expr);

 public:
  BestEffortReplay(
      const std::vector<IterDomain*>& replay_domain,
      const std::vector<IterDomain*>& target_domain,
      std::unordered_map<IterDomain*, IterDomain*> target2replay_map,
      std::unordered_map<IterDomain*, IterDomain*> replay_forward_id_map = {},
      std::unordered_map<IterDomain*, IterDomain*> target_forward_id_map = {},
      bool skip_swizzle = true);

  // Return iter domain map from target_domain IDs to their "replayed"
  // replay_domain IDs. If not in map, was not replayed.
  const std::unordered_map<IterDomain*, IterDomain*>& getReplay() const {
    return target2replay_id_map_;
  }

  // ids in replay that did not have matching transforms in target_domain
  const std::unordered_map<IterDomain*, size_t>& getUnorderedLeafIDs() {
    return leaf_ids_;
  }

  // Returned ordered set of IDs in getUnorderedLeafIDs
  std::vector<IterDomain*> getLeafIDs() {
    std::set<std::pair<IterDomain*, size_t>, id_int_lt> ordered_set;
    for (auto entry : leaf_ids_)
      ordered_set.emplace(entry);

    std::vector<IterDomain*> leaf_vec_;
    leaf_vec_.resize(ordered_set.size());
    std::transform(
        ordered_set.begin(),
        ordered_set.end(),
        leaf_vec_.begin(),
        [](std::pair<IterDomain*, size_t> entry) { return entry.first; });
    return leaf_vec_;
  }

  DisjointSets<IterDomain*> getDisjointSets();

  // Runs a best effort replay that ignores broadcast axes that appear in
  // consumer that are not mapped to producer in root_map.
  static BestEffortReplay replayCasP(
      const TensorView* consumer,
      const TensorView* producer,
      int producer_compute_at_axis,
      const RootDomainMap& root_map);

  // Runs a best effort replay that ignores broadcast axes that appear in
  // consumer that are not mapped to producer in root_map.
  static BestEffortReplay replayPasC(
      const TensorView* producer,
      const TensorView* consumer,
      int consumer_compute_at_axis,
      const RootDomainMap& root_map);

  // Find the first position i where td1[i] is not the same as td2[i]. "Same"
  // means the DAG and input IDs to generate td1[i] and td2[i] are the same.
  // td1 and td2 are assumed to have some matching iter domains, as this is a
  // strict same-ness check.
  static int findFirstMismatchedID(
      const TensorDomain* td1,
      const TensorDomain* td2);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
