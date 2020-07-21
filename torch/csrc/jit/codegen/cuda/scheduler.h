#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// return true or false on whether given fusion could be scheduled;
TORCH_CUDA_API bool scheduleFusion(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue> inputs);

// Parameters the Reduction Heuristic Generates to describe
// the optimial schedule
struct ReductionParams {
  // Reduction Blocking
  int grid_dim_x_ = 1;
  int grid_dim_y_ = 1;
  int block_dim_x_ = 1;
  int block_dim_y_ = 1;

  // Reduction Attributes
  bool fastest_dim_ = true;
  bool cross_warp_ = false;
  bool cross_block_ = false;
  bool mul_reds_per_blk_ = false;
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
