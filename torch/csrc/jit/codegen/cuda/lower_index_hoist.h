#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <list>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Hoisting common index subexpressions
//
// Class CommonIndexMap is updated during the lowering as new indices
// are inserted.
//
// Once all indices are inserted to CommonIndexMap, allocations of the
// the hoisted indices are inserted by allocateCommonIndices. Note
// that this assumes that the CUDA code generator does not inline a
// scalar Val with allocation (PR #1434).

class TORCH_CUDA_CU_API CommonIndexMap {
 public:
  //! For the given index, insert its subexpressions to the loop that has
  //! minimum amount of computation. For example, if I have a loop
  //! FOR i1
  //!   FOR i2
  //!     FOR i3
  //!       FOR i4
  //!         index = ((i1*1 + i2*2) + i3*3) + i4*4
  //! then this function will try insert i1*1 to common_index_map_[FOR i1],
  //! try insert i1*1 + i2*2 to common_index_map_[FOR i2],
  //! try insert ((i1*1 + i2*2) + i3*3) to common_index_map_[FOR i3],
  //! try insert ((i1*1 + i2*2) + i3*3) + i4*4 to common_index_map_[FOR i4],
  //! Before insertion, this function recursively uses
  //! reuseIndexIfAlreadyComputed to find existing expressions/subexpressions in
  //! common_index_map_ that can be reused. If a reuse oppportunity is found,
  //! then this function will modify the definition of `index` to use the
  //! existing subexpression. This function returns the modified value whose
  //! definition reuses other expressions in the list.
  Val* hoistIndex(Val* index, const std::vector<kir::ForLoop*>& loops);

  //! common_index_map_ stores all seen indices in a given loop, however, we
  //! don't want to create separate allocation for all of them. We are only
  //! interested in allocating for the indices that is actually hoisted, or used
  //! more than once. This method returns the Vals that will get its separate
  //! allocation.
  std::vector<Val*> getHoistedIndices(kir::ForLoop* loop) const;

 private:
  //! This is the underlying implementation of the public hoistIndex, with some
  //! additional arguments and return values.
  //! Returns (hoisted index, has tensor dependency)
  std::pair<Val*, bool> hoistIndexImpl(
      Val* index,
      const std::vector<kir::ForLoop*>& loops,
      int64_t position, // if `index` is given to `hoistIndex` (i.e., is_give ==
                        // true), then this is the position of the outer-most
                        // loop nest that contains all the dependencies of
                        // `index`. if `index` is a subexpression of the index
                        // given to `hoistIndex`, then this is the position of
                        // the outer-most loop nest that contains all the
                        // dependencies of its parent.
      bool is_given = false // true for the given index from the public
                            // `hoistIndex`, false otherwise.
  );

  //! If there is already an expression in common_index_map_[loop] which is
  //! sameAs `index`, then just return that expression. Otherwise, if there is a
  //! subexpression of an existing expression sameAs `index`, then that
  //! subexpression will be split out as a separate item in the mapped list, and
  //! that subexpression will be returned. If nothing sameAs `index`, then
  //! return nullptr.
  Val* reuseIndexIfAlreadyComputed(Val* index, kir::ForLoop* loop);

 private:
  //! Map to hold hoisted common indices. The order matters and indicates data
  //! dependency. For example, my list might have [i1*4, i1*4+2, i1*4/16]
  std::unordered_map<kir::ForLoop*, std::list<Val*>> common_index_map_;

  //! A set to identify that if a val is hoisted (an expression used in the
  //! inner loop, but its value only depend on outer loop variables, so the
  //! computation of this expression is hoisted to an outer loop) or reused (one
  //! expression is used in multiple indices/predicates).
  std::unordered_set<Val*> hoisted_or_reused_;
};

//! Insert allocations of hoisted indices. Must be called after
//! collecting all common indices.
std::vector<Expr*> allocateCommonIndices(const std::vector<Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
