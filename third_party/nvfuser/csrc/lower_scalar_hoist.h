#pragma once

#include <ir_all_nodes.h>

#include <list>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Hoisting common subexpressions for scalar expressions, including indices,
// predicates, tensor factories, etc.
//
// Class CommonScalarMap is updated during the lowering as new scalar
// are inserted.
//
// Once all scalars are inserted to CommonScalarMap, allocations of the
// the hoisted scalars are inserted by allocateCommonScalars. Note
// that this assumes that the CUDA code generator does not inline a
// scalar Val with allocation (PR #1434).

class TORCH_CUDA_CU_API CommonScalarMap {
 public:
  //! For the given scalar, insert the subexpressions in its definition to the
  //! loop that has minimum amount of computation. For example, if I have a loop
  //! FOR i1
  //!   FOR i2
  //!     FOR i3
  //!       FOR i4
  //!         index = ((i1*1 + i2*2) + i3*3) + i4*4
  //! and I want to hoist `index`. Then this function will try insert i1*1 to
  //! common_scalar_map_[FOR i1], try insert i1*1 + i2*2 to
  //! common_scalar_map_[FOR i2], try insert ((i1*1 + i2*2) + i3*3) to
  //! common_scalar_map_[FOR i3], try insert ((i1*1 + i2*2) + i3*3) + i4*4 to
  //! common_scalar_map_[FOR i4], Before insertion, this function recursively
  //! uses reuseScalarIfAlreadyComputed to find existing
  //! expressions/subexpressions in common_scalar_map_ that can be reused. If a
  //! reuse oppportunity is found, then this function will modify the definition
  //! of `value` to use the existing subexpression. This function returns the
  //! modified value whose definition reuses other expressions in the list.
  Val* hoistScalar(Val* value, const std::vector<kir::ForLoop*>& loops);

  //! common_scalar_map_ stores all seen indices in a given loop, however, we
  //! don't want to create separate allocation for all of them. We are only
  //! interested in allocating for the indices that is actually hoisted, or used
  //! more than once. This method returns the Vals that will get its separate
  //! allocation.
  std::vector<Val*> getHoistedScalars(kir::ForLoop* loop) const;

  //! Initialize the common_scalar_map_ with lowered exprs. If some scalar is
  //! already computed in these lowered exprs and is recomputed in indexing or
  //! predicate math, then we should reuse these existing computation.
  void initialize(const std::vector<Expr*> exprs);

 private:
  //! This is the underlying implementation of the public hoistScalar, with some
  //! additional arguments and return values.
  //! Returns (hoisted value, has tensor index dependency)
  std::pair<Val*, bool> hoistScalarImpl(
      Val* value,
      const std::vector<kir::ForLoop*>& loops,
      std::vector<Val*>&
          seen_subexprs, // Stores the subexpressions that has already been seen
                         // during the recursion. This is used to detect
                         // self-reuse. For example, if I have
                         // i3 = i1 * i2 + i1 * i2
                         // when visiting the second i1 * i2, I will have the
                         // first i1 * i2 in this vector, so that we know we can
                         // reuse that i1 * i2.
      int64_t position, // if `value` is given to `hoistScalar` (i.e., is_give
                        // == true), then this is the position of the outer-most
                        // loop nest that contains all the dependencies of
                        // `value`. if `value` is a subexpression of the value
                        // given to `hoistScalar`, then this is the position of
                        // the outer-most loop nest that contains all the
                        // dependencies of its parent.
      bool is_given = false // true for the given index from the public
                            // `hoistScalar`, false otherwise.
  );

  //! If there is already an expression in common_scalar_map_[loop] which is
  //! sameAs `value`, then just return that expression. Otherwise, if there is a
  //! subexpression of an existing expression sameAs `value`, then that
  //! subexpression will be split out as a separate item in the mapped list, and
  //! that subexpression will be returned. If nothing sameAs `value`, then
  //! return nullptr.
  Val* reuseScalarIfAlreadyComputed(Val* value, kir::ForLoop* loop);

 private:
  //! Map to hold hoisted common indices. The order matters and indicates data
  //! dependency. For example, my list might have [i1*4, i1*4+2, i1*4/16]
  std::unordered_map<kir::ForLoop*, std::list<Val*>> common_scalar_map_;

  //! A set to identify that if a val is hoisted (an expression used in the
  //! inner loop, but its value only depend on outer loop variables, so the
  //! computation of this expression is hoisted to an outer loop) or reused (one
  //! expression is used in multiple indices/predicates).
  std::unordered_set<Val*> hoisted_or_reused_;
};

//! Insert allocations of hoisted indices. Must be called after
//! collecting all common indices.
std::vector<Expr*> allocateCommonScalars(const std::vector<Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
