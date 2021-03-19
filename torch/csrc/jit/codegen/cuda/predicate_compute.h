
#pragma once

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Predicate compute takes a TensorView and set of indices. The number of
//! indices and the root of the TensorView are required to have the same number
//! of dimensions. Predicate compute should be run after index compute, and the
//! result of index compute should be used for the indices entry.
//!
//! A vector of Int values are returned which are the output of the operation
//! index[i] < get_root(TV)->domain()->axis(i)->size()
//!
//! It is assumed that no predicate is required if index[i] is an index directly
//! from a for loop. This will not catch all cases if we actually have static
//! size information for example:
//!
//! TV[I].split(4)
//! would produce the code:
//! for(i : I/4)
//!   for(j : 4)
//!     if( i * 4 + j < TV.size(0))
//!       TV[i * 4 + j]...
//!
//! However if we had TV.size[0] = 16 at "compile time" then we wouldn't need
//! the predicate. However we will still generate: for(i : 4) for(j : 4) if( i *
//! 4 + j < TV.size(0)) TV[i * 4 + j]...
//!
class PredicateCompute {
 public:
  //! Return the series of predicates (or 1 if an axis doesn't have a predicate)
  static std::vector<kir::Bool*> computePredicates(
      const kir::TensorView* tv,
      const std::vector<kir::Val*>& indices,
      bool buffer_init);

  static kir::Bool* getInlinePredicate(
      const kir::Expr* expr,
      const std::vector<kir::ForLoop*>& loops,
      kir::Bool* thread_pred,
      bool ignore_block_grid_external_ops = true);
};

class TORCH_CUDA_CU_API UnswitchPredicate {
 public:
  static kir::Bool* get(
      const std::vector<kir::ForLoop*>& outer_loops,
      kir::ForLoop* unrolled_loop,
      const IterDomainMap& p2c_root_map);

 private:
  UnswitchPredicate(
      std::vector<kir::ForLoop*> outer_loops,
      kir::ForLoop* unrolled_loop,
      const IterDomainMap& _p2c_root_map);

  void predicateOn(kir::Expr*);

  void openLoop(kir::ForLoop*);

 private:
  std::unordered_map<kir::IterDomain*, kir::Bool*> predicates_;
  std::vector<kir::ForLoop*> for_loops_;

  const IterDomainMap& p2c_root_map_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
