#pragma once

#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

/*
 * Index compute takes in a list of indices typically generated from the
 * surrounding for loop nest. The number of indicies are intended to match the
 * number of dimensions of the incomming TensorView which may have less or more
 * dimensions than its root due to split/merge operations.
 * Split/merge operations are then replayed backwards produce resulting
 * indices (based on input indices) that match the root dimension.
 *
 * For example with GLOBAL tensor:
 * TV[I, K]
 * TV[Io, Ii{4}, K] = TV.split(I, factor=4)
 * ALLOC: NONE
 * INDEX: indexCompute {i, j, k} -> {i * 4 + j, k}
 * FLATTENED_INDEX: {i * 4 + j, k} -> {(i * 4 + j) * K + k}
 * PREDICATE: {i * 4 + j, k} -> i * 4 + j < I
 *
 *
 * For example with SHARED tensor:
 *
 * global_TV[I, K]
 * global_TV[Io, Ii{4}, K] = global_TV.split(I, factor=4)
 * smem_TV.compute_at(global_TV, 1)
 * global_TV.parallelize(1, threadIDx.x)
 *
 * ALLOC: alloc(smem_TV, 4 x K)
 * INDEX: indexCompute(smem_TV, {threadIdx.x, k}) -> {threadIdx.x, k}
 * FLATTENED_INDEX: {threadIdx.x * 4 + j, k} -> {(threadIdx.x * 4 + j) * K + k}
 * PREDICATE: {threadIdx.x * 4 + j, k} -> threadIdx.x * 4 + j < I // Same as if
 * global
 *
 *
 * For example with LOCAL tensor:
 * global_TV[I, K, L]
 * global_TV[Io, Ii{4}, K, L] = global_TV.split(I, factor=4)
 * reg_TV.compute_at(global_TV, 2)
 * global_TV.parallelize(1, threadIDx.x)
 * global_TV{i, j, k, l} -> { i * 4 + j, k, l }
 * global_TV{ i * 4 + j, k, l } -> { (i * 4 + j) * K * L  +  k * L  +  l}
 *
 * ALLOC: alloc(reg_TV, K x L)
 * INDEX: {k, l} -> {k, l}
 * FLATTENED_INDEX: {k, l} -> {k * L + l}
 * PREDICATE: i * 4 + j < I && k < K && l < L ->  // Same as if global
 *
 * These indices can then be flattened later based on strides.
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class ContigIDs;
class LoopIndexing;
struct IndexFromIdGraph;

class IndexCompute : public BackwardVisitor {
 protected:
  using BackwardVisitor::handle;

  void handle(Split*) override;
  void handle(Merge*) override;
  void handle(Expr*) override;
  void handle(Swizzle2D*) override;

  // return extent_map_[id] if exists, else return id->extent()
  Val* getExtent(IterDomain* id) const;

  //! True if a domain is not used to index
  bool isZero(IterDomain* id) const;
  //! True if any dependent of a domain is not used to index
  bool hasZeroMerged(IterDomain* id) const;

  //! Returns the concrete ID from the compute at EXACT mode map if
  //! concrete_id_pass == true, otherwise returns id passed in.
  //! Helps unify the expr handling logic in reference domain and concrete id
  //! based traversal.
  IterDomain* maybeGetExactMapConcreteID(IterDomain* id);

  //! (Concrete indexing pass only)
  //!  Collect permissive index binding from the given expression.
  //! See also permissive_map_ and LoopIndexing::getBackwardOutOfLineExprList.
  void collectIndexIntoPermissiveMap(const LoopIndexing& loop_indexing);

  //! (Concrete indexing pass only)
  //!  Iterate through id_expr's input and pull index vals from permissive
  //! map, when both of the following are true:
  //!    1. the output id is missing in index_map_.
  //!    2. the output id is found in permissive map.
  void updateIndexMapFromPermissiveMap(const Expr* id_expr);

  // Tensor domain we're mapping back to root
  const TensorDomain* td_; // NOLINT

  // Map we update as we propagate backward, containing all IDs in the
  // propagation. Initial indices are mapped with this map at tv->domain()
  // and are back propagated to tv->getRootDomain(). This index_map_ keeps the
  // indices at intermediate IterDomain's in that back propagation.
  std::unordered_map<IterDomain*, Val*> index_map_; // NOLINT

  // Map from IterDomain to their broadcasted extent. If a TV has I0*I1 but its
  // producer has B0*I1 this map will contain a mapping from the ID{B0*I1} to
  // the extent I0*I1. Also contains updated extents if we merge in a 0 index.
  // See zero_merged_in_.
  std::unordered_map<IterDomain*, Val*> extent_map_; // NOLINT

  // Keeps track of domains that do not contribute to indexing
  std::unordered_set<IterDomain*> zero_domains_; // NOLINT

  // This set keeps track of IterDomain's that have had a zero index merged into
  // them. This happens if we do something like tv->axis(0)->split(4) then
  // tv->computeAt(1, ...) if this tensor is in smem or lmem the backward
  // indexing would be (0, i) then when we do the backward computation that zero
  // and i would attempt to be merged together. We handle indices like these
  // specially.
  std::unordered_set<IterDomain*> zero_merged_in_;

  // IDs that are a result of contiguous merges
  std::unordered_set<IterDomain*> contig_ids_;

  // Map from root to indexed domains
  std::unordered_map<IterDomain*, IterDomain*> root_to_indexed_id_;

  // Mentions if we should propagate an index down a particular IterDomain path
  // if there's an option
  std::unordered_set<IterDomain*> preferred_paths_;

  // Map from IterDomains to halo-extended extents
  std::unordered_map<IterDomain*, Val*> halo_extent_map_;

  // Temporary flag which tells IndexCompute to use concrete id's from the exact
  // map rather than the actual IDs used in the ID expressions.
  bool concrete_id_pass_ = false;

  // Mode of swizzle that are activated in this index compute
  //  instance. Will treat swizzles of different mode as no-op.
  // Currently data mode swizzles are handled same as before in IndexSwizzle
  //  pass, while loop mode swizzles are handled early on in concrete indexing
  //  pass. See also [Note on swizzle mode]
  SwizzleMode swizzle_mode_ = SwizzleMode::NoSwizzle;

  // (Concrete id pass only)
  // Contains the indexing math that could be resolved with only the
  //  iterdomains on the right of the consumer_tv's ca axis, i.e. the
  //  ones that corresponding to the loops that consumer_tv would not
  //  share with any of its consumers.
  // These indexing vals should be kept separate from index_map_ and
  //  should only be used when the indexing traversal follows the
  //  order defined in LoopIndexingAnalysis::traverseFromDomainVals.
  std::unordered_map<IterDomain*, Val*> permissive_index_map_;

 public:
  const std::unordered_map<IterDomain*, Val*>& indexMap() const {
    return index_map_;
  }

  const std::unordered_map<IterDomain*, Val*>& extentMap() const {
    return extent_map_;
  }

  const std::unordered_set<IterDomain*>& zeroDomains() const {
    return zero_domains_;
  }

  const std::unordered_set<IterDomain*>& zeroMergedIn() const {
    return zero_merged_in_;
  }

  const std::unordered_map<IterDomain*, IterDomain*>& rootToContigID() const {
    return root_to_indexed_id_;
  }

  // Propagate back from _td using initial_index_map
  IndexCompute(
      const TensorDomain* _td,
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_map<IterDomain*, Val*> _extent_map,
      std::unordered_set<IterDomain*> zero_domains,
      std::unordered_set<IterDomain*> _zero_merged_in,
      std::unordered_set<IterDomain*> preferred_paths = {},
      std::unordered_map<IterDomain*, Val*> halo_extent_map = {});

  IndexCompute(
      const TensorDomain* _td,
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_map<IterDomain*, Val*> _extent_map,
      std::unordered_set<IterDomain*> zero_domains,
      std::unordered_set<IterDomain*> _zero_merged_in,
      const ContigIDs& contig_finder,
      std::unordered_set<IterDomain*> preferred_paths = {},
      std::unordered_map<IterDomain*, Val*> halo_extent_map = {});

  // Entry point used for using concrete id based traversal. This traversal is
  // assumed to start at leaf IDs provided by initial_index_map.
  IndexCompute(
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_set<IterDomain*> zero_domains,
      std::unordered_set<IterDomain*> preferred_paths,
      std::unordered_map<IterDomain*, Val*> concrete_halo_extent_map);

  // Updates index_map, extent_map, and zero_merged_in based on id_map and
  // returns a new IndexCompute ready to be used.
  IndexCompute updateIndexCompute(
      const TensorDomain* new_td,
      const std::unordered_map<IterDomain*, IterDomain*>& id_map,
      const ContigIDs& contig_finder) const;

  // Interface to run index traversal through loop indexing analysis result to
  // be used with the entry point for concrete id based traversal.
  void run(const LoopIndexing& loop_indexing);

  virtual void run();
};

//! Apply swizzle and update root indices accordingly
class IndexSwizzle : public IndexCompute {
 public:
  IndexSwizzle(
      const TensorView* tv,
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_map<IterDomain*, Val*> extent_map,
      std::unordered_set<IterDomain*> zero_domains,
      std::unordered_set<IterDomain*> zero_merged_in);

  IndexSwizzle(
      const TensorView* tv,
      const TensorDomain* domain,
      std::unordered_map<IterDomain*, Val*> initial_index_map,
      std::unordered_map<IterDomain*, Val*> extent_map,
      std::unordered_set<IterDomain*> zero_domains,
      std::unordered_set<IterDomain*> zero_merged_in);

  void run() override;

 protected:
  using IndexCompute::handle;

  void handle(Expr* e) override;

  void handle(Swizzle2D* swizzle_2d) override;

 private:
  const TensorView* tv_ = nullptr;
  SwizzleType swizzle_type_ = SwizzleType::NoSwizzle;
  std::vector<IterDomain*> ids_to_swizzle_;
  std::unordered_set<IterDomain*> swizzled_ids_;
};

//! Predicate information of a root or contiguous merged domain
class RootPredicateInfo {
  friend class Index;

 public:
  const auto& startPredicate() const {
    return start_predicate_;
  }

  auto& startPredicate() {
    return start_predicate_;
  }

  const auto& startOffset() const {
    return start_offset_;
  }

  const auto& stopPredicate() const {
    return stop_predicate_;
  }

  const auto& stopOffset() const {
    return stop_offset_;
  }

  const auto& rootIds() const {
    return root_ids_;
  }

  //! Return a false RootPredicateInfo, i.e., both start and stop
  //! predicates are false.
  static RootPredicateInfo getFalseInfo();

 private:
  // prdicate for lower end
  Bool* start_predicate_ = nullptr;
  // prdicate for upper end
  Bool* stop_predicate_ = nullptr;
  // Offset of the start predicate
  Val* start_offset_ = nullptr;
  // Offset of the stop predicate
  Val* stop_offset_ = nullptr;
  // Track which roots have been handled by the generated predicates
  std::unordered_set<IterDomain*> root_ids_;
};

// Simple interface for IndexCompute
// If getComputeAtAxis and more generally TensorView const model is fixed, we
// can make the below tensorviews const.
class Index {
 private:
  // Producer indexing if it's in shared or local memory
  static std::vector<Val*> getNonGlobalProducerStridedIndices(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Consumer indexing if it's in shared or local memory
  static std::vector<Val*> getNonGlobalConsumerStridedIndices(
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Producer if it's in global memory
  static std::vector<Val*> getGlobalProducerStridedIndices(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Consumer indexing if it's in global memory
  static std::vector<Val*> getGlobalConsumerStridedIndices(
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // get the strides of a tensor used for the index lowering
  static std::vector<Val*> getStrides(const TensorView* tv);

  // get the root indices of a tensor used for the index lowering
  static std::vector<Val*> getRootIndices(
      const TensorView* tv,
      const std::vector<kir::ForLoop*>& loops,
      const IndexFromIdGraph& index_from_id_graph);

 public:
  // Indexing functions
  // Consumer = Producer
  // i.e. T0 = T1... -> T0 is the consumer, T1 is the producer
  // Producer indexing dispatch
  static kir::TensorIndex* getProducerIndex(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  // Consumer index dispatch
  static kir::TensorIndex* getConsumerIndex(
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  //! Returns a vector of strided indices mapped onto the (rfactor)
  //! root domain of a producer tensor. The size of the returned
  //! vector is guaranteed to be equal to the number of axes of the
  //! indexing root domain.
  static std::vector<Val*> getProducerStridedIndices(
      TensorView* producer,
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  //! Returns a vector of strided indices mapped onto the (rfactor)
  //! root domain of a consumer tensor. The size of the returned
  //! vector is guaranteed to be equal to the number of axes of the
  //! indexing root domain.
  static std::vector<Val*> getConsumerStridedIndices(
      const TensorView* consumer,
      const std::vector<kir::ForLoop*>& loops);

  //! Returns the logical index linearized from a multi-dimension address into a
  //! linear memory address a consumer tensor. The returned index is intended to
  //! be used for the computation of some tensor factories, such as: arange and
  //! rand (for Philox pseudo random sequences)
  static std::vector<Val*> getLinearLogicalIndex(
      TensorView* consumer_tv,
      const std::vector<kir::ForLoop*>& loops);

  //! Returns a vector of logical indices mapped onto the (rfactor)
  //! root domain of a consumer tensor. The returned index is intended
  //! to be used for the computation of some tensor factories, such as:
  //! eye
  static std::vector<Val*> getPerDimLogicalIndex(
      TensorView* consumer_tv,
      const std::vector<kir::ForLoop*>& loops);

  //! Take a consumer tensorview and loop nest and generates predicates
  //! associated with the concrete roots of the loop nest. Returns a list of
  //! predicates, and a list of concrete roots they're associated with. It
  //! is assumed that no predicate is required if index[i] is an index
  //! directly from a for loop. This will not catch all cases if we actually
  //! have static size information for example:
  //!
  //! TV[I].split(4)
  //! would produce the code:
  //! for(i : I/4)
  //!   for(j : 4)
  //!     if( i * 4 + j < TV.size(0))
  //!       TV[i * 4 + j]...
  //!
  //! However if we had TV.size[0] = 16 at "compile time" then we wouldn't
  //! need the predicate. This will be caught by canOmitPredicate in the
  //! predicate lowering
  //!
  //! unswitch_or_vec_loop is the for loop to start the unswitch like
  //! predicate, this is not a bool value as if we have an unswitch loop
  //! with a vectorized loop inside, we only want to base the "unswitch"
  //! like predicate on the vectorized loop.
  static std::vector<RootPredicateInfo> getReferenceRootPredicates(
      TensorView* consumer_tv,
      const std::vector<kir::ForLoop*>& loops,
      kir::ForLoop* unswitch_or_vec_loop,
      bool padding_predicate);
};

// Used for local and shared index mapping. Returns a map from loops
// to loop indices as well as a set of loops that do not contribute to
// indexing.
// TODO: could be cleaned up further.
std::pair<
    std::unordered_map<kir::ForLoop*, Val*>,
    std::unordered_set<kir::ForLoop*>>
indexMapFromTV(
    const TensorView* tv,
    const std::vector<kir::ForLoop*>& loops,
    kir::ForLoop* alloc_loop,
    bool as_consumer,
    kir::ForLoop* double_buffer_loop = nullptr);

//! Set "pragma unroll" required for loops that indexing of Local
//! tensors depends on.
//!
//! \param tv Indexed tensor
//! \param alloc_loop Allocation loop of tv
//! \param loops The current loop structure
//! \param id_map Producer-to-consumer map in case of indexing as producer
void ensureStaticIndexing(
    const TensorView* tv,
    kir::ForLoop* alloc_loop,
    const std::vector<kir::ForLoop*>& loops,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map = {});

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
