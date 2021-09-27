#pragma once

#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class PredicateCompute {
 public:
  // ignore_internal_syncthread_ops will prevent creation of predicates on
  // block/grid broadcast/reduce as these have syncthread calls within them
  // so all threads need to execute the function.
  static kir::Bool* getInlinePredicate(
      const kir::Expr* expr,
      const std::vector<kir::ForLoop*>& loops,
      kir::Bool* thread_pred,
      PredicateType pred_type);
};

class TORCH_CUDA_CU_API UnswitchPredicate {
 public:
  static kir::Bool* get(
      const std::vector<kir::ForLoop*>& outer_loops,
      kir::ForLoop* unrolled_loop);

 private:
  UnswitchPredicate(
      std::vector<kir::ForLoop*> outer_loops,
      kir::ForLoop* unrolled_loop);

  void predicateOn(kir::Expr*);

  void openLoop(kir::ForLoop*);

  void openIte(kir::IfThenElse*);

 private:
  // Track which iter domains have been predicated, uses concrete_id from
  // caLoopMap.
  std::vector<kir::IterDomain*> predicated_iter_dom_;

  // The predicates that have been generated.
  std::vector<kir::Bool*> predicates_;

  std::vector<kir::ForLoop*> for_loops_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
