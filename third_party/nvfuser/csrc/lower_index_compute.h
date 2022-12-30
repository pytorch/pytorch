#pragma once

#include <fusion.h>
#include <index_compute.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Struct to hold useful information from an index pass on iterdomain graph.
// Used to return the IndexCompute structure back to the indexing calls in
// index_compute.cpp. Other structurs are required to resolve the actual
// indexing math there.
struct IndexFromIdGraph {
  IndexCompute index;
  IndexCompute concrete_index;
  std::unordered_map<IterDomain*, Val*> initial_concrete_index_map;
  std::vector<IterDomain*> resolved_loop_domains;

  explicit IndexFromIdGraph(
      IndexCompute index,
      IndexCompute concrete_index,
      std::unordered_map<IterDomain*, Val*> initial_concrete_index_map,
      std::vector<IterDomain*> loop_domains);
};

//! Indexing interface, returns IndexFromIdGraph which the IndexCompute object
//! can be queried from directly for the produced indexing. If producer_tv !=
//! nullptr producer will be indexed, if producer_tv == nullptr consumer will be
//! indexed. If is_global global indexing will be done, else shared memory or
//! local indexing will be performed.
IndexFromIdGraph getTensorIndexFromIdGraph(
    const std::vector<kir::ForLoop*>& loops,
    const TensorView* consumer_tv,
    const TensorView* producer_tv = nullptr,
    bool is_global = true,
    std::unordered_map<IterDomain*, IterDomain*> c2p_map = {});

//! Indexing interface for calculating predicate index returns IndexFromIdGraph
//! which the IndexCompute object can be queried from directly for the produced
//! indexing If is_start_predicate, will produce indexing math for the start
//! predicates.
IndexFromIdGraph getPredicateIndexingFromIdGraph(
    const std::vector<kir::ForLoop*>& loops,
    TensorView* consumer_tv,
    kir::ForLoop* unswitch_or_vec_loop,
    IterDomain* double_buffer_axis,
    bool is_start_predicate);

//! getTensorIndexFromIdGraph is the function that index_compute will call very
//! straightforwardly. However, for implementing the new indexing logic that
//! starts to abstract some of the indexing away from index_compute we need to
//! move quite a bit of the intertwined indexing logic away from the
//! index_compute file and the index_reference_replay file. This is because we
//! want to separate out what has to be done on the fly, from what analysis we
//! can do early on with the iter domain graph and associated properties.
//!
//! getTensorIndexFromIdGraph places this analysis internally in
//! LoopIndexingAnalysis. LoopIndexingAnalysis though has to communicate to:
//!   1) index_compute.cpp::IndexCompute to tell IndexCompute which expressions
//!   it needs to traverse to compute the indexing math.
//!   2) lower_shift.cpp::HaloInfo::buildConcreteHaloExtentMap to build the halo
//!   extent map used in indexing.
//!
//! LoopIndexing is nothing but a mechanism for this communication.
//!
//! Holds information needed to produce indexing math. In the current version of
//! indexing pass, the iter domains combined with the loop nests are the source
//! of truth in terms of resolving the actual integer indexing math from the
//! sequence of iterdomain transforms.
//!
//! This information is crtiical in resolving indexing associated with complex
//! broadcast patterns. Check FusionComplexBCast* test cases as well as
//! Indexing* tests for examples where resolving indices from IterDomain
//! transformations can be challenging.
//!
//! The source of this challenge is due to inling patterns where the IterDomains
//! responsible for control flow are not local to a particular TensorView.
//! Broadcast, operations like view/reshape, and gather/shift can make indexing
//! local buffers complex because of the complex effects inlining into other
//! TensorViews produce.
//!
//! TODO:
//!   The first iteration tries to match the semantics of reference
//!     replay without any new logic. In a follow up iteration will
//!     need to revisit a few further pathological patterns.
//!
//! Note:
//!   The current implementation of loop indexing pass works on
//! equivalent classes defined by ComputeAt exact map. The
//! list of expressions stored in this class form a "reference", graph of
//! iterdomain expressions when all of their inputs and outputs are replaced
//! with their exact concrete mapped id's.
//!
//!   Here an invariant in a graph of iterdomain expressions is that
//! each iterdomain is produced exactly once and is either a leaf domain
//! or has been consumed exactly once by another expression. This makes sure
//! that a well defined indexing can be generated for each of the concrete ids
//! whenever we either forward or backward traverse the graph.
class LoopIndexing {
 public:
  //! Returns the original loop nest.
  const auto& loops() const {
    return loops_;
  }

  //! Returns the vector of Iterdomains
  //!  that match the original loop pattern.
  const auto& loopDomains() const {
    return loop_domains_;
  }

  //! Returns the consumer tv that the view info
  //!  was derived from.
  auto consumerTv() const {
    return consumer_tv_;
  }

  //! Returns the set of Iterdomain transforms that
  //!  define the correct indexing path, in forward
  //!  topological order.
  std::vector<Expr*> getForwardExprList() const;

  //! Returns the set of Iterdomain transforms that
  //!  define the correct indexing path, in backward
  //!  topological order.
  std::vector<Expr*> getBackwardExprList() const;

  //! Returns the set of out of line expressions in
  //!  reverse topological order.
  const std::vector<Expr*>& getBackwardOutOfLineExprList() const {
    return out_of_line_exprs_;
  }

  //! Returns all exact concrete id's that were produced
  //!  or consumed in the selected indexing expressions
  std::unordered_set<IterDomain*> getAllExactConcreteIdSet() const;

 private:
  friend class LoopIndexingAnalysis;

  //! The loop nest that this loop indexing is derived from.
  std::vector<kir::ForLoop*> loops_;

  //! Consumer tv, where the view related info was derived from.
  const TensorView* consumer_tv_;

  //! The source iterdomains that all the Iterdomain transforms
  //!   in this loop nest originated from.
  std::vector<IterDomain*> loop_root_;

  //! The leaf iterdomains that the original loop nests correspond
  //!  to. May be longer than loops_ with the dangling iterdomains
  //!  appended towards the end.
  std::vector<IterDomain*> loop_domains_;

  //! The selected sequence of expressions that should represent
  //!  the correct indexing math from the given loop nest.
  std::vector<Expr*> index_exprs_;

  //! The subset of sequence of expressions that can be resolved
  //!  with only the iterdomains on the right of consumer tv's ca
  //!  axis.
  //! Expressions are ordered in reverse topological order.
  std::vector<Expr*> out_of_line_exprs_;
};

class LoopIndexingAnalysis {
 public:
  static LoopIndexing fromLoopAndConsumer(
      const std::vector<kir::ForLoop*>& loops,
      const TensorView* consumer_tv);

  //! Return all concrete IDs that can be reachable from a given list
  //! of consumer leaf IDs. Reachability is defined as the existence
  //! an indexing path from the the leaf IDs
  static VectorOfUniqueEntries<IterDomain*> getReplayableConcreteIDs(
      const std::vector<IterDomain*>& consumer_leaf_ids,
      const TensorView* consumer_tv);

 private:
  explicit LoopIndexingAnalysis(
      const std::vector<kir::ForLoop*>& loops,
      const TensorView* consumer_tv);

  explicit LoopIndexingAnalysis(
      const std::vector<IterDomain*>& consumer_leaf_ids,
      const TensorView* consumer_tv);

  void run();

  //! Populate derived information into a LoopIndexing
  //!  data structure.
  LoopIndexing getLoopIndexing(const std::vector<kir::ForLoop*>& loops) {
    LoopIndexing indexing;
    indexing.loops_ = loops;
    indexing.consumer_tv_ = consumer_tv_;
    indexing.loop_root_ = loop_root_domains_;
    indexing.loop_domains_ = loop_domains_.vector();
    indexing.index_exprs_ = replayed_exprs_;
    indexing.out_of_line_exprs_ = out_of_line_exprs_;
    return indexing;
  }

  //! Validates that the current loop structure is well formed, in the sense
  //! that ca_map would not map any two loops in the loop nest together.
  void validateLoopStructure(const std::vector<kir::ForLoop*>& loops);

  //! Start at the loop iter domains, and traverse back into history on the
  //! concrete IDs in the exact map calling "visitExpr" expressions through the
  //! history.
  void traverseFromDomainVals();

  //! Concretize the given iterdomain and record the visit (in deterministic
  //! order) in terms of the exact mapped concrete id. Marks the mapping of the
  //! id to the concrete id in "concrete_to_original_id_" and returns the
  //! concrete id.
  IterDomain* concretizeAndVisitId(IterDomain* id);

  //! If an equivalent expression has already been processed this function
  //! simply returns. Otherwise puts the exact concrete IDs of inputs in
  //! consumed_concrete_, and concrete IDs of outputs in produced_concrete_.
  //! Then adds the expression to replayed_exprs_.
  void visitExpr(Expr* expr);

  //! Iterates through provided vals, calls concretizeAndVisitId on them, and
  //! returns if any of the returned vals are in existing_ids. This is used to
  //! check if inputs or outputs of ID expressions have already been
  //! produced/consumed in the traversal. Indexing only needs to consume/produce
  //! one IterDomain per exact disjoint set.
  bool visitIdsAndCheckDuplication(
      const std::vector<Val*>& vals,
      const std::unordered_set<IterDomain*>& existing_ids);

  //! Fills loop_domains_ with the corresponding replayed_concrete_id mapping to
  //! the provided loops. Must be done after the exact iterdomain "replay"
  //! (traverseFromDomainVals). loop_domains_ are the original_id not the
  //! concrete_id (translated with concrete_to_original_id). These iter domains
  //! are used to grab the history that will be replayed in IndexCompute. We're
  //! looking for "new" root domains and subsequent transformations, filling in
  //! any missing "outputs" (or inputs for backward traversal). Then fills
  //! loop_domains_ with all of these iter domains.
  void constructLoopDomains();

  //! Fills out_of_line_exprs_ by traversing the selected list of
  //!  expressions in reverse topological order and collect iterdomains
  //!  on the indexing paths that only involves leaf id's on the right
  //!  of consumer's ca axis.
  void collectOutOfLineExprs();

 private:
  //! Original consumer tv to derive view info from.
  const TensorView* consumer_tv_ = nullptr;

  // Exact concrete domains that has been used
  //  in the traversal connection.
  std::unordered_set<IterDomain*> produced_concrete_;
  std::unordered_set<IterDomain*> consumed_concrete_;

  //! Iterdomains that the corresponding loops are generated from.
  std::vector<IterDomain*> initial_loop_domain_ids_;

  //! All Id's in consumer's transform history
  std::vector<Val*> all_consumer_id_vals_;

  //! Concrete iterdomains visited in the domain traversal,
  //!  in the order they are visited in traverseFromDomainVals.
  VectorOfUniqueEntries<IterDomain*> replayed_concrete_ids_;

  //! Keeping track of the original visited id's before they
  //!  were concretized.
  std::unordered_map<IterDomain*, IterDomain*> concrete_to_original_id_;

  //! Map from concrete id to its single consumer on the selected
  //!  iterdomain expression list.
  std::unordered_map<IterDomain*, Expr*> concrete_id_to_consumer_;

  //! Source domains that all the Iterdomain transforms
  //!  in the loop nest originated from.
  std::vector<IterDomain*> loop_root_domains_;

  //! Leaf domains representing the original loop structure
  VectorOfUniqueEntries<IterDomain*> loop_domains_;

  //! Selected list of exprs that will produce and consume each
  //!  of the exact concrete ids from the loop nest exactly once.
  std::vector<Expr*> replayed_exprs_;

  //! Set of expressions from the selected list that can be
  //!  resolved from axes on the right of ca axes.
  std::vector<Expr*> out_of_line_exprs_;
};

// When indexing there are sometimes an option to propagate an index down
// multiple paths. This will return the IterDomains in the history of the
// reference domain and mark which paths should be taken (if there's a
// preference) to reach the roots provided in preferred_roots.
std::unordered_set<IterDomain*> buildLoopIndexingPreferredPath(
    const TensorView* original_tv,
    const LoopIndexing& loop_indexing,
    bool use_replay_map = false,
    std::unordered_map<IterDomain*, IterDomain*> p2c_map = {});

// Get an rfactor IterDomain that is mapped with an IterDomain. If
// multiple such IDs exist, select one whose input IDs are mapped with
// the consumer IDs. This is to ensure the path from the leaf
// IterDomains to the root matches with the consumer tensor.
IterDomain* getRfactorIDToTraverse(
    IterDomain* id,
    const std::vector<Val*>& consumer_all_ids);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
