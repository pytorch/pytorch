#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>

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
//! FusionAdvancedIndexing* for examples where resolving indices from IterDomain
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
