#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// return true or false on whether given fusion could be scheduled;
TORCH_CUDA_API bool scheduleFusion(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue> inputs);

/**
 * This struct communicates whether a launch param
 * should be calculated per kernel instance if a new set of elements
 * matches the kernel.
 *
 * @param value_ : -1 = symbolic, > 0 = valid constant launch param
 * @param symbolic_value_ : a pointer to a symbolic val of param
 */
struct LaunchParam {
  bool mutable_ = false;
  int value_ = 1;
};

// Parameters the Reduction Heuristic Generates to describe
// the optimial schedule
struct ReductionParams {
  // Reduction Blocking
  LaunchParam gdimx_;
  LaunchParam gdimy_;
  LaunchParam bdimx_;
  LaunchParam bdimy_;

  // Reduction Attributes
  bool fastest_dim = true;
  bool cross_block = false;
  bool cross_grid = false;
  bool mul_reds_per_blk = false;

  bool operator==(const ReductionParams& other) const {
    bool lp_equal = other.gdimx == gdimx && other.gdimy == gdimy &&
        other.bdimx == bdimx && other.bdimy == bdimy;
    bool attr_equal = other.fastest_dim == fastest_dim &&
        other.cross_block == cross_block && other.cross_grid == cross_grid &&
        other.mul_reds_per_blk == mul_reds_per_blk;
    return attr_equal && lp_equal;
  }
};

class ReductionParamsHash {
 public:
  size_t operator()(const ReductionParams& rp) const {
    size_t lp_hash = (rp.gdimx.is_mutable ? 0 : rp.gdimx.value) ^
        (rp.gdimy.is_mutable ? 0 : rp.gdimy.value) ^
        (rp.bdimx.is_mutable ? 0 : rp.bdimx.value) ^
        (rp.bdimy.is_mutable ? 0 : rp.bdimy.value);
    constexpr size_t bits = sizeof(std::size_t) * 8;
    size_t attr_hash = static_cast<size_t>(rp.fastest_dim) << (bits - 1) |
        static_cast<size_t>(rp.cross_block) << (bits - 2) |
        static_cast<size_t>(rp.cross_grid) << (bits - 3) |
        static_cast<size_t>(rp.mul_reds_per_blk) << (bits - 4);
    return lp_hash | attr_hash;
  }
};

// TODO: This function is currently a redundant API as I populate a more
// substantial reduction heuristic
// fusion is the input IR that will be modified by this function
TORCH_CUDA_API c10::optional<ReductionParams> scheduleReduction(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    TensorView* red_tv);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
