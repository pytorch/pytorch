#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <bitset>

namespace torch {
namespace jit {
namespace fuser {

class TORCH_CUDA_API ThreadPredicates {
 private:
  Fusion* fusion_;

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
  std::unordered_map<const TensorView*, std::bitset<6>> thread_predicates;

  // Update the thread_predicates bitset based on provided Expr
  void updateBitSet(Expr*);

  // Safety wrapper to access thread_predicates
  std::bitset<6> getThreadPredicates(const TensorView*);

  ThreadPredicates(Fusion* _fusion);

 public:
  // Computes any thread predicates that need to be applied when computing a
  // TensorView.
  static std::unordered_map<const TensorView*, Bool*> compute(Fusion* fusion);
};

} // namespace fuser
} // namespace jit
} // namespace torch
