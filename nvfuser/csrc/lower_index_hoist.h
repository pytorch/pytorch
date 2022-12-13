#pragma once

#include <ir_all_nodes.h>

#include <functional>
#include <unordered_map>
#include <vector>

// Hoisting common index subexpressions
//
// Class CommonIndexMap is updated during the lowering as new indices
// are inserted. An index is uniquely identified with CommonIndexKey,
// which consists of the concrete ID of the indexed/predicated domain,
// the for-loops used in the index, and the index vals of the use
// for-loops.
//
// Once all indices are inserted to CommonIndexMap, allocations of the
// the hoisted indices are inserted by allocateCommonIndices. Note
// that this assumes that the CUDA code generator does not inline a
// scalar Val with allocation (PR #1434).

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Class to represent unique indexed domains for index
//! hoisting. Uniquenesss is determined with the indexed domain
//! itself, the for-loops and their index values.
class TORCH_CUDA_CU_API CommonIndexKey {
  friend struct CommonIndexKeyHash;

 public:
  //! \param consumer_indexed_id Indexed consumer domain
  //! \param consumer_td TensorDomain of consumer_indexed_id
  //! \param ref_td Reference domain at the time of indexing
  //! \param ref_index_map Index map of the reference domain
  //! \param loops Loop structure where this id is indexed
  CommonIndexKey(
      IterDomain* consumer_indexed_id,
      TensorDomain* consumer_td,
      TensorDomain* ref_td,
      const std::unordered_map<IterDomain*, Val*>& ref_index_map,
      const std::vector<kir::ForLoop*>& loops);

  //! \param consumer_indexed_id Indexed consumer domain
  //! \param consumer_td TensorDomain of consumer_indexed_id
  //! \param loop_domains Resolved vector of iterdomain corresponding to loops
  //! \param loop_index_map Index mapping generated from the loop nest.
  //! \param loops Loop structure where this id is indexed
  //! Duplicate of above, but without a reference domain. TODO: Remove other
  //! implementation.
  CommonIndexKey(
      IterDomain* consumer_indexed_id,
      TensorDomain* consumer_td,
      const std::vector<IterDomain*>& loop_domains,
      const std::unordered_map<IterDomain*, Val*>& loop_index_map,
      const std::vector<kir::ForLoop*>& loops);

  const IterDomain* concreteIndexedId() const {
    return concrete_indexed_id_;
  }

  const std::vector<kir::ForLoop*>& usedLoops() const {
    return used_loops_;
  }

  const std::vector<Val*>& loopIndexVals() const {
    return loop_index_vals_;
  }

  bool operator==(const CommonIndexKey& other) const;

  std::string toString() const;

 private:
  //! Concrete domain of indexed domain
  IterDomain* concrete_indexed_id_ = nullptr;
  //! Loops used for the index
  std::vector<kir::ForLoop*> used_loops_;
  //! Loop index vals for the used loops
  std::vector<Val*> loop_index_vals_;
};

struct CommonIndexKeyHash {
  std::size_t operator()(const CommonIndexKey& key) const {
    auto h = std::hash<const IterDomain*>{}(key.concrete_indexed_id_);
    // NOTE: do not use other fields as the pointers can be different
    // even when two keys can share the same index
    return h;
  }
};

//! Map to hold hoisted common indices
class TORCH_CUDA_CU_API CommonIndexMap {
 public:
  //! Register an indexd consumer domain to hoist
  //!
  //! Returns a corresponding hoisted index and a flag indicating if a
  //! new index is inserted.
  //!
  //! Consumer domains are used even for producer indexing since
  //! producer domains in producer indexing are temporary replay
  //! domains.
  std::pair<Val*, bool> insert(
      IterDomain* indexed_consumer_id,
      TensorDomain* consumer_td,
      TensorDomain* ref_td,
      const std::unordered_map<IterDomain*, Val*>& ref_index_map,
      const std::vector<kir::ForLoop*>& loops,
      Val* index);

  //! Duplicate of above, but without a reference domain. TODO: Remove other
  //! implementation.
  std::pair<Val*, bool> insert(
      IterDomain* indexed_consumer_id,
      TensorDomain* consumer_td,
      const std::vector<IterDomain*>& loop_domains,
      const std::unordered_map<IterDomain*, Val*>& loop_index_map,
      const std::vector<kir::ForLoop*>& loops,
      Val* index);

  const auto& commonIndexMap() const {
    return common_index_map_;
  }

  const auto& useCounts() const {
    return use_counts_;
  }

 private:
  //! Utility method to insert a key into common index
  //!  map. Returns a pair of an IR node and a boolean value.
  //! The IR node will be the previously inserted index if
  //!  the key found a match, or will be the original index
  //!  if this is new key and the key will be stored.
  //! The boolean value will be true if the key is stored,
  //!  i.e. first time it is inserted.
  std::pair<Val*, bool> tryInsertNewIndex(CommonIndexKey key, Val* index);

 private:
  //! Map to hold hoisted common indices
  std::unordered_map<CommonIndexKey, Val*, CommonIndexKeyHash>
      common_index_map_;
  std::unordered_map<CommonIndexKey, int, CommonIndexKeyHash> use_counts_;
};

//! Insert allocations of hoisted indices. Must be called after
//! collecting all common indices.
std::vector<Expr*> allocateCommonIndices(const std::vector<Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
