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

  bool operator==(const LaunchParam& other) const {
    return other.value_ == value_ && other.mutable_ == mutable_;
  }
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
  bool fastest_dim_ = true;
  bool cross_block_ = false;
  bool cross_grid_ = false;
  bool mul_reds_per_blk_ = false;

  bool operator==(const ReductionParams& other) const {
    bool lp_equal = other.gdimx_ == gdimx_ && other.gdimy_ == gdimy_ &&
        other.bdimx_ == bdimx_ && other.bdimy_ == bdimy_;
    bool attr_equal = other.fastest_dim_ == fastest_dim_ &&
        other.cross_block_ == cross_block_ &&
        other.cross_grid_ == cross_grid_ &&
        other.mul_reds_per_blk_ == mul_reds_per_blk_;
    return attr_equal && lp_equal;
  }
};

class ReductionParamsHash {
 public:
  size_t operator()(const ReductionParams& rp) const {
    size_t lp_hash = (rp.gdimx_.mutable_ ? 0 : rp.gdimx_.value_) ^
        (rp.gdimy_.mutable_ ? 0 : rp.gdimy_.value_) ^
        (rp.bdimx_.mutable_ ? 0 : rp.bdimx_.value_) ^
        (rp.bdimy_.mutable_ ? 0 : rp.bdimy_.value_);
    constexpr size_t bits = sizeof(std::size_t) * 8;
    size_t attr_hash = static_cast<size_t>(rp.fastest_dim_) << (bits - 1) |
        static_cast<size_t>(rp.cross_block_) << (bits - 2) |
        static_cast<size_t>(rp.cross_grid_) << (bits - 3) |
        static_cast<size_t>(rp.mul_reds_per_blk_) << (bits - 4);
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
