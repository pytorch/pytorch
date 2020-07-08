#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <bitset>

namespace torch {
namespace jit {
namespace fuser {

/*
 * Map from tensorview to bit set represnting <BIDx, BIDy, BIDz, TIDx, TIDy,
 * TIDz> If any dependency of TV had a parallelized reduction, we will track
 * it here. This will be used for predicate generation to prevent
 * parallelization on that axis. This is important if we have a reduction on
 * for example TIDx, as the reduced value is only valid on threadIdx.x == 0
 * therefore if we use that value later in the kernel we have that predicate.
 * If we follow a reduction parallelized on TIDx with a broadcast on TIDx we
 * no longer need the predicate and can reset the bit accordingly
 */
class TORCH_CUDA_API ThreadPredicateMap {
 public:
  using MapType =
      std::unordered_map<const TensorView*, ir_utils::ParallelTypeBitmap>;
  using const_iterator = MapType::const_iterator;

  explicit ThreadPredicateMap(Fusion* _fusion);

  const_iterator find(const TensorView* tv) const;
  const_iterator end() const;
  const ir_utils::ParallelTypeBitmap& at(const TensorView* tv) const;
  ir_utils::ParallelTypeBitmap& at(const TensorView* tv);
  ir_utils::ParallelTypeBitmap& operator[](const TensorView* tv);

  // Returns a Bool predicate expression for a given TensorView.
  Bool* getExpr(const TensorView* tv) const;

 private:
  Fusion* fusion_;
  MapType thread_predicates_;

  // Update the thread_predicates bitset based on provided Expr
  void updateBitSet(Expr*);
};

} // namespace fuser
} // namespace jit
} // namespace torch
