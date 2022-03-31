#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <deque>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TrivialReductionInfo;

class TORCH_CUDA_CU_API ComputeAtMap {
 public:
  // There's three modes of these iter domain mappings. For indexing, for loop
  // nest mapping/generation, and to figure out the parallelization strategy.
  //
  // For index/loop mode consider:
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
  // really need to carry two sets of maps around for lowering.
  //
  // Parallel mode is important if we have something like:
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
  // MappingMode::PARALLEL
  //   Only maps leaf axes to left of compute at
  //   Forward broadcast axes in replay
  // MappingMode::LOOP
  //   Forward broadcast axes in replay
  //   Map all iteration domains
  //   Always contain root mappings (otherwise they could have been forwarded in
  //   broadcast)
  // MappingMode::INDEX
  //   Don't map any broadcast axes to non-broadcast axes
  //   Do not forward through any broadcast IDs
  enum class MappingMode { PARALLEL, LOOP, INDEX };

  // Would be nice to be able to remove this constructor, should be able to do
  // so if we wrap it in a unique pointer in GPULower
  ComputeAtMap() = default;

  // Passing in a trivial reduction info pointer will prevent compute at map
  // from generating its own. This is available during lowering so it can be
  // used, however for uses outside of lowering compute at map will simply
  // generate this information from the provided fusion.
  ComputeAtMap(
      Fusion* fusion,
      MappingMode mapping_mode,
      const TrivialReductionInfo* _trivial_reduction_info = nullptr);
  //! Returns if id0 and id1 are mapped to eachother, meaning they represent the
  //! same loop nest in the lowered code
  bool areMapped(IterDomain* id0, IterDomain* id1) const;

  //! Returns an iter domain that is the maximum expanded size of all iter
  //! domains the one provided maps to. Useful for opening loops to the correct
  //! iteration size. Not guarenteed to return the same ID every call, but is
  //! guarenteed to return iter domains in the same disjoint set.
  IterDomain* getConcreteMappedID(IterDomain* id) const;

  // Prints mapping information via Fusion IR
  std::string toString() const;

  // If Index mapping mode is selected will return the entry of id in
  // concrete_id_count_map_ otherwise will throw. Pair is count of concrete then
  // count of broadcast dims as inputs to id.
  std::pair<int, int> getConcreteIdCountOf(IterDomain* id) const {
    auto concrete_count_it = concrete_id_count_map_.find(id);
    TORCH_INTERNAL_ASSERT(
        concrete_count_it != concrete_id_count_map_.end(),
        "Could not find concrete counts for id: ",
        id->toString());
    return concrete_count_it->second;
  }

 private:
  void mapIds(IterDomain* id0, IterDomain* id1);

 private:
  //! Builds all valid mappings for fusion in provided mapping mode. If
  //! trivial_reduction_info is not passed in it will be built.
  void build(
      Fusion* fusion,
      const TrivialReductionInfo* trivial_reduction_info = nullptr);

  // Detects id's that need to be added to rfactor_concrete_count_reset_domains_
  // from tv and adds them, returns true if anything was added.
  bool pullConcreteCountResetIds(
      const torch::jit::fuser::cuda::TrivialReductionInfo*
          trivial_reduction_info,
      TensorView* tv);

  // Should be run on consumer id after producer id and consumer id are mapped
  // together, will place all id's in consumer_id's disjoint set into
  // count_one_concrete_dims if consumer_id is in
  // rfactor_concrete_count_reset_domains_
  void maybePropagateConcreteCountOne(IterDomain* consumer_id);

  MappingMode mapping_mode_ = MappingMode::LOOP;

  // This is actually only used when mapping mode == LOOP. Only used in expr
  // sorting, it's actually maximum position where a loop is shared across any
  // neighbor.
  std::unordered_map<TensorView*, unsigned int> produce_at_map_;

  // Disjoint sets of iter domains, only defined if iter domain is within
  // compute at of a tensor view. Maps these iter domains to a set containing
  // all other iter domains in the fusion that map to the same loop nest.
  std::unordered_map<IterDomain*, std::shared_ptr<std::deque<IterDomain*>>>
      disjoint_iter_set_maps_;

  // Keep a list of disjoint_iter_sets that's deterministic to iterate over
  std::deque<std::shared_ptr<std::deque<IterDomain*>>> disjoint_iter_sets_;

  // Tracks if there's a parallel iter domain associated a disjoint iter domain
  // set
  std::unordered_map<std::shared_ptr<std::deque<IterDomain*>>, ParallelType>
      parallel_type_map_;

  // One iteration domain with the largest count in concrete_id_count_map_
  // within a disjoint set will be selected as the "concrete_id" of that set
  std::unordered_map<IterDomain*, IterDomain*> concrete_id_map_;

  // Track how many concrete IDs compose each iteration domain. If in the
  // rfactor domain of a view operation this value must be maximum 1 (can be 0
  // if it's a broadcast). concrete_id_count_map_ is only important for parallel
  // and loop maps as in index map any ID in the disjoint set is a valid
  // concrete ID by definition of the index map.
  std::unordered_map<IterDomain*, std::pair<int, int>> concrete_id_count_map_;

  // Track all domains that are the result of a view operation and consumed in a
  // subsequent expression. When resolving concrete IDs we want to set these
  // domains to count of concrete domains = 1 so they won't be favored over post
  // view iteration domains when resolving the parallel and loop map concrete
  // IDs.
  std::unordered_set<IterDomain*> rfactor_concrete_count_reset_domains_;

  // Gather all dimensions that should be considered having one concrete
  // dimension, this is propagated from the domains in
  // rfactor_concrete_count_reset_domains_ as we're building the map. This will
  // reset all domains topologically before the domains in
  // rfactor_concrete_count_reset_domains_ to have count of 1 concrete domain.
  // This prevents pre-view ID's from being involved in concrete resolution when
  // mapped to post view domains in the current map (Parallel and Loop maps
  // IndexMap is valid with any domain in any disjoint set being the concrete
  // ID).
  std::unordered_set<IterDomain*> count_one_concrete_dims_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
