#pragma once

#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_reductions.h>

#include <deque>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// There's three modes of these iter domain mappings all uniquely important in
// the lowering process.
//
// For EXACT/PERMISSIVE mode consider:
//
// consumer[i0, b1] = producer[i0]
// consumer->merge(0) (consumer will now be [i0 * b1])
// When producer is replayed as consumer (the direction we use for mapping)
// with BestEffortReplay forward_bcast_mismatch = True the producer to
// consumer map will have both a mapping of consumer(i0) to producer(i0) as
// well as consumer(i0*b1) to producer(i0). This latter mapping is important
// for loop nest mappings as the consumer will generate a loop based on i0*b1
// and the producer may be computeAt inside this loop nest. However, for
// indexing we do not want these two maps as producer may be indexed as i0*i1
// depending on the loop nest structure and how it was built. Therefore we
// really need to carry (at least) two sets of maps around for lowering.
//
// LOOP mode is important if we have something like:
// consumer[i0o, threadIdx.x{i0i}] = producer[i0o, threadIdx.y{i0i}](computeAt
// = 1) which can easily happen when using shared memory. We want to make sure
// that the iteration domain used for loop construction (concreteId) has the
// proper parallelization strategy. In parallel mode we do typical iteration
// domain mapping, however we remove from it any iteration domains outside the
// computeAt of producer when mapping. This guarentees we won't map
// IterDomains that could have different parallelization strategies. We also
// propagate the parallel strategy in parallel mode so all mapped IDs that
// must have the same parallel type, do.
//
// IdMappingMode::LOOP
//   Only maps leaf axes to left of compute at
//   Forward broadcast axes in replay
// IdMappingMode::PERMISSIVE
//   Forward broadcast axes in replay
//   Map all iteration domains
//   Always contain root mappings (otherwise they could have been forwarded in
//   broadcast)
// IdMappingMode::EXACT
//   Don't map any broadcast axes to non-broadcast axes
//   Do not forward through any broadcast IDs
class TORCH_CUDA_CU_API IterDomainGraph {
 public:
  IterDomainGraph(Fusion* fusion);

  const DisjointSets<IterDomain*>& permissiveNodes() const {
    return permissive_nodes_;
  }
  const DisjointSets<IterDomain*>& exactNodes() const {
    return exact_nodes_;
  }
  const DisjointSets<IterDomain*>& loopNodes() const {
    return loop_nodes_;
  }

  // Consumers and producers is not symmetric like the other sets
  const std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>&
  consumers() const {
    return consumers_;
  }
  const std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>&
  producers() const {
    return producers_;
  }

  const DisjointSets<IterDomain*>& siblings() const {
    return sibling_sets_;
  }

  const VectorOfUniqueEntries<IterDomain*>& allIds() const {
    return all_ids_;
  }

  const std::unordered_set<IterDomain*>& viewRfactorIds() const {
    return view_rfactor_ids_;
  }

 private:
  void build(Fusion* fusion);

  void initializeId(IterDomain* id, bool is_view_rfactor_id, bool is_leaf_id);

  DisjointSets<IterDomain*> permissive_nodes_;
  DisjointSets<IterDomain*> exact_nodes_;
  DisjointSets<IterDomain*> loop_nodes_;

  // Consumers and producers is not symmetric like the other sets
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      consumers_;
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      producers_;

  DisjointSets<IterDomain*> sibling_sets_;

  VectorOfUniqueEntries<IterDomain*> all_ids_;

  std::unordered_set<IterDomain*> view_rfactor_ids_;
};

class TrivialReductionInfo;

class TORCH_CUDA_CU_API ComputeAtMap {
 public:
  ComputeAtMap() = delete;
  ComputeAtMap(Fusion* fusion);

  //! Run through disjoint sets in the LOOP map, make sure there's only one
  //! non-serial parallel type in each disjoint set, set the parallel type of
  //! all IterDomains in the disjoint set to that PType.
  void validateAndPropagatePType();

  //! Returns if id0 and id1 are mapped to eachother with provided IdMappingMode
  bool areMapped(IterDomain* id0, IterDomain* id1, IdMappingMode mode) const;

  //! Returns an iter domain that is the maximum expanded size of all iter
  //! domains the one provided maps to. Useful for opening loops to the correct
  //! iteration size. Not guarenteed to return the same ID every call, but is
  //! guarenteed to return iter domains in the same disjoint set.
  IterDomain* getConcreteMappedID(IterDomain* id, IdMappingMode mode) const;

  // Prints mapping information, forwards to an internal IterDomainGraph
  std::string toString() const;

  // Returns if the provided ID is a view like rfactor id
  bool isViewRfactor(IterDomain* ref_id) const;

  // Returns all rfactor domains in rfactor_concrete_count_reset_domains_ that
  // are in the disjoint set of the provided IterDomain. This will be every view
  // like rfactor ID the provided ID "depends" on in the map.
  std::vector<IterDomain*> getViewRfactorDomainsOfIdGroup(
      IterDomain* ref_id,
      IdMappingMode mode) const;

  const IterDomainGraph& idGraph() const {
    return id_graph_;
  }

  //! Get the ID sets for a provided IdMappingMode
  const DisjointSets<IterDomain*>& getIdSets(IdMappingMode mode) const;

 private:
  // Build id_graph_
  void build(Fusion* fusion);

  // Build concrete_id_cache_
  // Build a single entry in  concrete_cache_id_
  IterDomain* computeConcreteId(IterDomain* id, IdMappingMode mode);
  void buildConcreteIds();

  // Produce the disjoint set containing provided id with mapping mode.
  const std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>& disjointSetOf(
      IterDomain* id,
      IdMappingMode mode) const;

  // Should be built once and never modified again.
  const IterDomainGraph id_graph_;
  TrivialReductionInfo trivial_reduction_info_;

  // Prevent needing to recompute concrete_id's in compute at map.
  // VectorOfUniqueEntries is unique across mapping modes, so don't need to use
  // mapping mode directly in this cache. const
  // VectorOfUniqueEntries<IterDomain*>& is what's returned by
  // ComputeAtMap::disjointSetOf which can be used directly.
  std::unordered_map<
      std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>,
      IterDomain*>
      concrete_id_cache_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
