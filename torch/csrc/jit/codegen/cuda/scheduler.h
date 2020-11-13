#pragma once

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>
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

// Parameters the Reduction Heuristic Generates to describe the optimial
// schedule. Warning: equal operator is intended for use in caching the kernel
// associated with these reduction parameters. It does not check if the launch
// parameters are equivelent!
struct ReductionParams {
  // Reducing inner most dimension?
  bool fastest_dim = true;
  // Reduce across the block?
  bool cross_block = false;
  // Reduce across the grid?
  bool cross_grid = false;
  // Perform multiple reductions per block?
  bool multiple_reds_per_blk = false;
  // Unrolling factor
  int64_t loop_unroll = 4;
  // Number of batches for each block
  int64_t batches_per_block = 1;
  // Number of warps per block
  int64_t num_warps = 1;
  // Store input in shared memory or registers to reduce global memory reads
  bool persistent_kernel = false;

  LaunchParams lparams;

  // Warning: Does not check launch parameters!
  bool operator==(const ReductionParams& other) const {
    bool attr_equal = other.fastest_dim == fastest_dim &&
        other.cross_block == cross_block && other.cross_grid == cross_grid &&
        other.multiple_reds_per_blk == multiple_reds_per_blk &&
        other.loop_unroll == loop_unroll &&
        other.batches_per_block == batches_per_block &&
        other.num_warps == num_warps &&
        other.persistent_kernel == persistent_kernel;
    return attr_equal;
  }
};

// Warning: Hash is not based on launch parameters!
class ReductionParamsHash {
 public:
  size_t operator()(const ReductionParams& rp) const {
    constexpr size_t bits = sizeof(std::size_t) * 8;
    size_t attr_hash = static_cast<size_t>(rp.fastest_dim) << (bits - 1) |
        static_cast<size_t>(rp.cross_block) << (bits - 2) |
        static_cast<size_t>(rp.cross_grid) << (bits - 3) |
        static_cast<size_t>(rp.multiple_reds_per_blk) << (bits - 4) |
        static_cast<size_t>(rp.batches_per_block) << (bits - 5) |
        static_cast<size_t>(rp.num_warps) << (bits - 6) |
        static_cast<size_t>(rp.persistent_kernel) << (bits - 7);
    return attr_hash;
  }
};

TORCH_CUDA_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    TensorView* red_tv);

TORCH_CUDA_API void scheduleReduction(
    Fusion* fusion,
    const ReductionParams& rparams,
    TensorView* red_tv,
    std::vector<TensorView*> outs_of_red);

TORCH_CUDA_API c10::optional<ReductionParams> getMultipleReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    const std::vector<TensorView*>& reduction_tv);

TORCH_CUDA_API void scheduleMultipleReduction(
    Fusion* fusion,
    const ReductionParams& rparams,
    const std::vector<TensorView*>& reduction_tv,
    std::vector<TensorView*>& other_tv);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
