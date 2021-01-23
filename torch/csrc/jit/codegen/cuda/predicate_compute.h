#pragma once

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

/*
 * Predicate compute takes a TensorView and set of indices. The number of
 * indices and the root of the TensorView are required to have the same number
 * of dimensions. Predicate compute should be run after index compute, and the
 * result of index compute should be used for the indices entry.
 *
 * A vector of Int values are returned which are the output of the operation
 * index[i] < get_root(TV)->domain()->axis(i)->size()
 *
 * It is assumed that no predicate is required if index[i] is an index directly
 * from a for loop. This will not catch all cases if we actually have static
 * size information for example:
 *
 * TV[I].split(4)
 * would produce the code:
 * for(i : I/4)
 *   for(j : 4)
 *     if( i * 4 + j < TV.size(0))
 *       TV[i * 4 + j]...
 *
 * However if we had TV.size[0] = 16 at "compile time" then we wouldn't need the
 * predicate. However we will still generate: for(i : 4) for(j : 4) if( i * 4 +
 * j < TV.size(0)) TV[i * 4 + j]...
 *
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class PredicateCompute {
 public:
  // Return the series of predicates, if an axis doesn't have a predicate
  // reutrns 1
  static std::vector<kir::Bool*> computePredicates(
      const TensorView* tv,
      const std::vector<Val*>& indices,
      bool use_rfactor);

  static kir::Bool* getInlinePredicate(
      Expr* expr,
      const std::vector<kir::ForLoop*>& loops,
      kir::Bool* thread_pred,
      bool ignore_block_grid_reductions = true);
};

class TORCH_CUDA_CU_API UnrollPredicate {
 public:
  static kir::Bool* get(
      const std::vector<kir::ForLoop*>& outer_loops,
      kir::ForLoop* unrolled_loop,
      const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map);

 private:
  UnrollPredicate(
      std::vector<kir::ForLoop*> outer_loops,
      kir::ForLoop* unrolled_loop,
      const std::unordered_map<IterDomain*, IterDomain*>& _p2c_root_map);

  void predicateOn(Expr*);

  void openLoop(kir::ForLoop*);

 private:
  std::unordered_map<IterDomain*, kir::Bool*> predicates_;
  std::vector<kir::ForLoop*> for_loops_;

  const std::unordered_map<IterDomain*, IterDomain*>& p2c_root_map_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
