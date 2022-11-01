#pragma once

#include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_broadcast.h>

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
// IdMappingMode::AlmostExact
//   Forward through broadcast axes, but not through to a non-broadcast axis
//     i.e. id{b1*i0}, id{i0} are mapped
//          id{i1*i0}, id{i0} are not mapped (this part is the difference from
//          PERMISSIVE)
//   Forward through split one axes, i.e. id{ceilDiv(i0, 1)}, id{i0} are mapped
//
class TORCH_CUDA_CU_API IterDomainGraph {
 public:
  IterDomainGraph(Fusion* fusion, bool allow_self_mapping = false);

  const DisjointSets<IterDomain*>& permissiveNodes() const {
    return permissive_nodes_;
  }
  const DisjointSets<IterDomain*>& exactNodes() const {
    return exact_nodes_;
  }
  const DisjointSets<IterDomain*>& almostExactNodes() const {
    return almost_exact_nodes_;
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

  // Returns if first and second are expressions through which the provided
  // id_map have matching inputs (if forward), or outputs (if not forward).
  // Returning true means the expressions are "the same", in terms they modify
  // matching original extents, by the same amount.
  static bool exprsMap(
      Expr* first,
      Expr* second,
      bool forward,
      const DisjointSets<IterDomain*>& id_map);

  bool hasSelfMapping() const {
    return self_mapping_info_.has_value();
  }

 private:
  void build(Fusion* fusion);

  void initializeId(IterDomain* id, bool is_view_rfactor_id, bool is_leaf_id);

  // Checks if exprsMap then if forward will map outputs else inputs in exact
  // and permissive map.
  void mapThroughExpr(Expr* first, Expr* second, bool forward);

  DisjointSets<IterDomain*> permissive_nodes_;
  DisjointSets<IterDomain*> exact_nodes_;
  DisjointSets<IterDomain*> almost_exact_nodes_;
  DisjointSets<IterDomain*> loop_nodes_;

  // Consumers and producers is not symmetric like the other sets
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      consumers_;
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<IterDomain*>>
      producers_;

  DisjointSets<IterDomain*> sibling_sets_;

  VectorOfUniqueEntries<IterDomain*> all_ids_;

  std::unordered_set<IterDomain*> view_rfactor_ids_;

  c10::optional<std::tuple<TensorView*, IterDomain*, IterDomain*, std::string>>
      self_mapping_info_ = c10::nullopt;
};

using DoubleBufferIndices = std::unordered_map<DoubleBufferLoopStage, Int*>;

class TORCH_CUDA_CU_API ComputeAtMap {
 public:
  ComputeAtMap() = delete;
  ComputeAtMap(const ComputeAtMap&) = delete;
  ComputeAtMap& operator=(const ComputeAtMap&) = delete;
  ComputeAtMap(ComputeAtMap&&) = default;
  ComputeAtMap& operator=(ComputeAtMap&&) = default;
  ComputeAtMap(Fusion* fusion);

  //! Run through disjoint sets in the LOOP map, make sure there's only one
  //! non-serial parallel type in each disjoint set, set the parallel type of
  //! all IterDomains in the disjoint set to that PType.
  void validateAndPropagatePType();

  //! Run through disjoint sets in the LOOP map and allocate the index
  //!  variable for the associated for loop that will be generated
  //!  for each disjoint sets in the loop map. This pre-allocation makes
  //!  2 key assumptions about computeAt map that would very likely be
  //!  long term invariant:
  //!    1. All kir::forloop created in the lowering pass should belong
  //!  to one of the disjoint sets in loop map.
  //!    2. The lowering pass will *never* create a loop nest with 2
  //!  different nesting levels mapped together, i.e. the case below
  //!  never occurs:
  //!   for i in IterDomain1
  //!    for j in IterDomain2
  //!     ...
  //!   With loop_map.areMapped(IterDomain1, IterDomain2) == true.
  //! Under this condition, we can pre-allocate all required index
  //!  variable integers before creating any kir::forloop, and this
  //!  would help optimizing the generated integer math for indexing.
  void allocateIndexVariables();

  //! Returns if id0 and id1 are mapped to eachother with provided IdMappingMode
  bool areMapped(IterDomain* id0, IterDomain* id1, IdMappingMode mode) const;

  //! Returns an iter domain that is the maximum expanded size of all iter
  //! domains the one provided maps to. Useful for opening loops to the correct
  //! iteration size. Not guarenteed to return the same ID every call, but is
  //! guarenteed to return iter domains in the same disjoint set.
  IterDomain* getConcreteMappedID(IterDomain* id, IdMappingMode mode) const;

  //! Returns a list of expressions that produce the iter domains of all exact
  //! mapped id's to 'id'. Expressions that are the same exact transformations
  //! are deduplicated in the returned expressions.
  std::vector<Expr*> uniqueExactDefinitions(IterDomain* id) const {
    auto disjoint_set = disjointSetOf(id, IdMappingMode::EXACT);
    auto unique_exact_definition_it =
        unique_exact_definitions_.find(disjoint_set);
    if (unique_exact_definition_it == unique_exact_definitions_.end()) {
      return {};
    }
    return unique_exact_definition_it->second;
  }

  //! Returns a list of expressions that *use* the iter domains of all exact
  //! mapped id's to 'id'. Expressions that are the same exact transformations
  //! are deduplicated in the returned expressions.
  std::vector<Expr*> uniqueExactUses(IterDomain* id) const {
    auto disjoint_set = disjointSetOf(id, IdMappingMode::EXACT);
    auto unique_exact_use_it = unique_exact_uses_.find(disjoint_set);
    if (unique_exact_use_it == unique_exact_uses_.end()) {
      return {};
    }
    return unique_exact_use_it->second;
  }

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

  // Returns if the ID actually has a disjoint set meaning it has been processed
  // in the creation of the compute at map.
  bool idExistsInMap(IterDomain* id) const;

  //! Returns the pre-allocated index variable integer used in
  //!  the kir::ForLoop corresponding to the given IterDomain.
  //!  this interface is only valid if the ID has a loop mapping,
  //!  ca_map will throw exceptions if given iterdomain doesn't
  //!  have a loop map entry.
  Val* getIndexVariable(
      IterDomain* id,
      DoubleBufferLoopStage double_buffer_loop_stage =
          DoubleBufferLoopStage::NotApplicable) const;

  // Returns if expr_1 and expr_2 have exact mapped IterDomains in
  // inputs/outputs (order matters) and if the expressions have matching
  // parameters.
  bool areExactExprs(Expr* expr_1, Expr* expr_2);

  // Produce the disjoint set containing provided id with mapping mode.
  const std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>& disjointSetOf(
      IterDomain* id,
      IdMappingMode mode) const;

 private:
  // Traverses through definitions of exact maps (unique_exact_definitions_) to
  // input ID's from provided ID. Returns all the exact map concrete IDs of the
  // exact sets that are inputs required to construct the exact concrete id of
  // of_id.
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
  getInputDisjointSetsOf(IterDomain* of_id, bool stop_at_rfactor = true);

  // Traverses through definitions of exact maps (unique_exact_definitions_) to
  // all input ID's from provided exact_sets. Returns all the exact map concrete
  // IDs of all the exact sets that on the path to and including the inputs
  // required to construct the exact concrete id of of_id.
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
  getAllDisjointSetProducers(
      const VectorOfUniqueEntries<
          std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>& exact_sets);

  // Traverses through uses of exact maps (unique_exact_uses_) to
  // all input ID's from provided exact_sets. Returns all the exact map concrete
  // IDs of all the exact sets that on the path to and including the inputs
  // required to construct the exact concrete id of of_id.
  VectorOfUniqueEntries<std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>
  getAllDisjointSetConsumers(
      const VectorOfUniqueEntries<
          std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>>& exact_sets);

  // Build id_graph_
  void build(Fusion* fusion);

  // Build concrete_id_cache_
  // Build a single entry in  concrete_cache_id_
  IterDomain* computeConcreteId(IterDomain* id, IdMappingMode mode);
  void buildConcreteIds();

  // Relies on concrete_id_cache_, buildConcreteIds() must be run before this.
  void buildUniqueExactExprMaps();

  // Should be built once and never modified again.
  IterDomainGraph id_graph_;

  // Used specifically for concrete ID computation
  ConcretizedBroadcastDomains concretized_bcasts_;

  // Prevent needing to recompute concrete_id's in compute at map.
  // VectorOfUniqueEntries is unique across mapping modes, so don't need to use
  // mapping mode directly in this cache. const
  // VectorOfUniqueEntries<IterDomain*>& is what's returned by
  // ComputeAtMap::disjointSetOf which can be used directly.
  std::unordered_map<
      std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>,
      IterDomain*>
      concrete_id_cache_;

  // Unique expressions operating on exact disjoint set. For each IterDomain in
  // each exact disjoint set will log its definition in the std::vector<Expr*>.
  // If another expression is already in the set where inputs and outputs
  // exactly match with the expression to add along with the other parameters of
  // the transformation (like split's factor, or swizzles types) then the
  // expression will not be added as it would be an "duplicate" transformation.
  std::unordered_map<
      std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>,
      std::vector<Expr*>>
      unique_exact_definitions_;

  // Same as unique_exact_definitions_ but for uses instead of definitions
  std::unordered_map<
      std::shared_ptr<VectorOfUniqueEntries<IterDomain*>>,
      std::vector<Expr*>>
      unique_exact_uses_;

  //! Allocated Loop index variable through the CA map.
  //!   only valid for disjoint sets on the loop ca map.
  std::unordered_map<const VectorOfUniqueEntries<IterDomain*>*, Val*>
      loop_index_variable_map_;

  //! Allocated loop indices for double buffer loop.
  //!  only valid for disjoint sets on the loop ca map
  //!  that have double buffer-ed iterdomains.
  using DoubleBufferIndicesPtr = std::unique_ptr<DoubleBufferIndices>;
  std::unordered_map<
      const VectorOfUniqueEntries<IterDomain*>*,
      DoubleBufferIndicesPtr>
      double_buffered_loop_index_variable_map_;

  // Shortcut to access the fusion this computeAt map was
  //  built from.
  Fusion* fusion_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
