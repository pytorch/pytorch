
#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/parallel_type_bitmap.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Maps TensorViews to a { ParallelTypeBitmap, SourceMap } pair
//!
//! Map from TensorView to bit set represnting <BIDx, BIDy, BIDz, TIDx, TIDy,
//! TIDz> If any dependency of TV had a parallelized reduction, we will track
//! it here. This will be used for predicate generation to prevent
//! parallelization on that axis. This is important if we have a reduction on
//! for example TIDx, as the reduced value is only valid on threadIdx.x == 0
//! therefore if we use that value later in the kernel we have that predicate.
//! If we follow a reduction parallelized on TIDx with a broadcast on TIDx we
//! no longer need the predicate and can reset the bit accordingly
//!
//! In addition, if a parallel thread type is not used, it is
//! redundant to use all threads/blocks. That isn't a problem
//! generally although it can be inefficient, but when an aliased smem
//! buffer is used as an output, redundant writes can be invalid (see issue
//! #1110). PredicateInfo::redundant_types track which parallel types
//! are redundant for each tensor and is used to let only one
//! thread/block of a redundant type execute the expression for a
//! tensor.
class TORCH_CUDA_CU_API ThreadPredicateMap {
 public:
  using SourceMap = std::unordered_map<
      ParallelType,
      std::unordered_set<const TensorView*>,
      TypeHash>;

  //! Thread predicate information for each tensor
  struct PredicateInfo {
    // Parallel types where only one thread/block is valid.
    ParallelTypeBitmap limited_types;
    // Parallel types where only one thread/block is enough.
    ParallelTypeBitmap redundant_types;
    // Tracking use chain of redundant writes:
    //  [Redundant use chain]
    //  a parallel type is a `redundant_consumer_type` only
    //    if all of its propagation use chains terminate with
    //    a redundant write of this type.
    //  A propagation use chain is currently either a reg-to-reg
    //   chain for a shared mem tv, or a reg/smem-to-reg/smem chain
    //   for a global tv.
    // This is complementary information to `redundant_types`.
    //  If a tensor view is redundantly written and not redundantly
    //  used by all consumers, see FusionRedundantPredSync3,
    //  a RAW sync will need to be inserted before reading
    //  this redundantly written tensor.
    ParallelTypeBitmap redundant_use_types;
    bool operator==(const PredicateInfo& other) const {
      return limited_types == other.limited_types &&
          redundant_types == other.redundant_types &&
          redundant_use_types == other.redundant_use_types;
    }
  };

  using MapType = std::unordered_map<const TensorView*, PredicateInfo>;

  using const_iterator = MapType::const_iterator;

  //! Build a map from each tensor to PredicateInfo.
  void build(Fusion* fusion);

  //! Get a PredicateInfo for a given tensor. If it's an output of
  //! a parallel broadcast, unmask the limited_types_ bit of the
  //! corresponding parallel type since it must join the broadcast
  //! operation although the valid input is only available at one of
  //! the threads/blocks.
  PredicateInfo getPredicateInfo(const TensorView* tv) const;

  //! Returns a flag set that indicates which parallel types should be
  //! predicated.
  ParallelTypeBitmap getPredicatedParallelTypes(const TensorView* tv) const;

  //! Returns a Bool predicate for a given TensorView.
  Bool* getPredicate(const TensorView* tv) const;

  //! Returns a ParallelTypeBitmap representing which domain needs
  //! blockBroadcast.
  //!
  //! Even when a domain is broadcast and parallelized, it does not need
  //! blockBroadcast unless it is predicated by limited_types_
  ParallelTypeBitmap getParallelBroadcastDomains(const TensorView* tv) const;

  //! Mark tv as updated so that rebuilding the map should recompute
  //! its predicates and those of its dependents.
  void markAsUpdated(const TensorView* tv);

  void print() const;

  //! Generate a Bool value from PredicateInfo.
  static Bool* getPredicateFromPredicateInfo(
      const ThreadPredicateMap::PredicateInfo& pred_info);

  //! Get the redundant use types of the given expr, see [Redundant use chain]
  ParallelTypeBitmap getRedundantConsumerType(Expr* expr) const;

 private:
  // Update the thread_predicates bitset based on provided Expr
  void updateBitSet(const Expr*);

  const_iterator find(const TensorView* tv) const;
  const_iterator end() const;

  const PredicateInfo& at(const TensorView* tv) const;
  PredicateInfo& at(const TensorView* tv);

  //! Update a mapping
  bool update(
      const TensorView* tv,
      const ParallelTypeBitmap& limited_types,
      const ParallelTypeBitmap& redundant_types);

  //! Update a mapping
  bool update(const TensorView* tv, const PredicateInfo& pred_and_src);

  //! Backward populate redundant use chain info once the redundant
  //!  parallel writes have been identified.
  void populateRedundantUseMap(Fusion* fusion);

 private:
  MapType thread_predicates_;
  //! Keep track of updated tensors that need predicates to be computed
  std::unordered_set<const TensorView*> updated_tvs_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
