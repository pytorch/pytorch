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

//! Keys to identify unique unswitch predicates. Just consists of a
//! predicated concrete domain if not parallelized. If parallelized,
//! pick one for each different parallelization. When the same
//! parallel type is used for different concrete domains, they are
//! considered different predicates and are included in the unswitch
//! condition lists.
class UnswitchPredicateKey {
 public:
  UnswitchPredicateKey();

  UnswitchPredicateKey(
      IterDomain* predicated_concrete_id,
      const ReferenceTensor& reference);

  bool operator==(const UnswitchPredicateKey& other) const {
    return predicated_concrete_id_ == other.predicated_concrete_id_ &&
        parallel_concrete_ids_ == other.parallel_concrete_ids_;
  }

  const auto& predicatedId() const {
    return predicated_concrete_id_;
  }

  const auto& parallelConcreteIds() const {
    return parallel_concrete_ids_;
  }

  IterDomain* parallelId(ParallelType pt) const {
    auto it = parallelConcreteIds().find(pt);
    if (it == parallelConcreteIds().end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }

  std::string toString() const;

 private:
  //! Predicated concrete domain
  IterDomain* predicated_concrete_id_ = nullptr;
  //! Store parallelized concrete domains
  std::unordered_map<ParallelType, IterDomain*, TypeHash>
      parallel_concrete_ids_;
};

struct UnswitchPredicateKeyHash {
  std::size_t operator()(const UnswitchPredicateKey& key) const;
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
  // Track which iter domains have been predicated
  std::unordered_set<UnswitchPredicateKey, UnswitchPredicateKeyHash>
      predicated_keys_;

  // The predicates that have been generated.
  std::vector<kir::Bool*> predicates_;

  std::vector<kir::ForLoop*> for_loops_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
