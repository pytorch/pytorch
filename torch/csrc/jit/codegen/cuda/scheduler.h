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

// Parameters the Reduction Heuristic Generates to describe
// the optimial schedule
struct ReductionParams {
  // Reduction Attributes
  bool fastest_dim = true;
  bool cross_block = false;
  bool cross_grid = false;
  bool mul_reds_per_blk = false;

  LaunchParams lparams;

  bool operator==(const ReductionParams& other) const {
    bool attr_equal = other.fastest_dim == fastest_dim &&
        other.cross_block == cross_block && other.cross_grid == cross_grid &&
        other.mul_reds_per_blk == mul_reds_per_blk;
    return attr_equal && lparams == other.lparams;
  }
};

class ReductionParamsHash {
 public:
  size_t operator()(const ReductionParams& rp) const {
    size_t lp_hash = rp.lparams.gdimx() ^ rp.lparams.gdimy() ^
        rp.lparams.bdimx() ^ rp.lparams.bdimy();
    constexpr size_t bits = sizeof(std::size_t) * 8;
    size_t attr_hash = static_cast<size_t>(rp.fastest_dim) << (bits - 1) |
        static_cast<size_t>(rp.cross_block) << (bits - 2) |
        static_cast<size_t>(rp.cross_grid) << (bits - 3) |
        static_cast<size_t>(rp.mul_reds_per_blk) << (bits - 4);
    return lp_hash | attr_hash;
  }
};

TORCH_CUDA_API c10::optional<ReductionParams> scheduleReduction(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    TensorView* red_tv);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
