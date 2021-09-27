#pragma once

#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Parameters the Reduction Heuristic Generates to describe the optimial
// schedule. Warning: equal operator is intended for use in caching the kernel
// associated with these reduction parameters. It does not check if the launch
// parameters are equivelent!
class ReductionParams {
 public:
  // Reducing inner most dimension?
  bool fastest_dim = true;
  // Reduce across the block?
  bool cross_block = false;
  // Reduce across the grid?
  bool cross_grid = false;
  // Perform multiple reductions per block?
  bool multiple_reds_per_blk = false;
  // Unrolling factor
  int64_t loop_unroll = 1;
  // Should unrolling be done on reduction dimension
  bool reduction_unroll = true;
  // vectorize instead of unroll
  bool vectorize = false;
  // Number of batches for each block
  int64_t batches_per_block = 1;
  // Number of warps per block
  // TODO: Remove or repurpose
  int64_t num_warps = 1;
  // Store input in shared memory or registers to reduce global memory reads
  bool persistent_kernel = false;

  // Split grid dim in case it's too large for cuda
  bool split_grid_dim = false;

  std::string tag = "";

  LaunchParams lparams;

 public:
  // Warning: Does not check launch parameters!
  bool operator==(const ReductionParams& other) const {
    bool attr_equal = other.fastest_dim == fastest_dim &&
        other.cross_block == cross_block && other.cross_grid == cross_grid &&
        other.multiple_reds_per_blk == multiple_reds_per_blk &&
        other.loop_unroll == loop_unroll && other.vectorize == vectorize &&
        other.batches_per_block == batches_per_block &&
        other.num_warps == num_warps &&
        other.persistent_kernel == persistent_kernel &&
        other.reduction_unroll == reduction_unroll &&
        other.split_grid_dim == split_grid_dim;
    return attr_equal;
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "\n===== Reduction Parameters ========\n"
       << (tag == "" ? "" : "Tag: ") << tag
       << (fastest_dim ? "Red On Fastest Dim\n" : "Red On Slow Dim\n")
       << "Reduction Characteristics:\n"
       << (multiple_reds_per_blk ? "Multiple Reds Per Block\n" : "")
       << (cross_block ? "Cross block reduction\n" : "")
       << (cross_grid ? "Cross grid reduction\n" : "");
    if (persistent_kernel) {
      ss << "Persistent Kernel\n"
         << "Batches per block: " << batches_per_block << "\n";
    }
    ss << "Blocking:\n"
       << " GridY: " << lparams.gdimy() << " BlckY: " << lparams.bdimy()
       << " BlckX: " << lparams.bdimx() << "\n";
    if (loop_unroll > 1) {
      ss << (vectorize ? "Vectorize " : "Unroll ")
         << (reduction_unroll ? " reduction dim, " : " iter dim, ")
         << "Factor: " << loop_unroll << "\n";
    }
    ss << "====================================\n";
    return ss.str();
  }
};

// Warning: Hash is not based on launch parameters!
class ReductionParamsHash {
 public:
  size_t operator()(const ReductionParams& rp) const {
    constexpr size_t bits = sizeof(std::size_t) * 8;
    size_t attr_hash = static_cast<size_t>(rp.fastest_dim) << (bits - 1) ^
        static_cast<size_t>(rp.cross_block) << (bits - 2) ^
        static_cast<size_t>(rp.cross_grid) << (bits - 3) ^
        static_cast<size_t>(rp.multiple_reds_per_blk) << (bits - 4) ^
        static_cast<size_t>(rp.loop_unroll) ^
        static_cast<size_t>(rp.reduction_unroll) << (bits - 5) ^
        static_cast<size_t>(rp.vectorize) << (bits - 6) ^
        static_cast<size_t>(rp.batches_per_block) ^
        static_cast<size_t>(rp.num_warps) ^
        static_cast<size_t>(rp.persistent_kernel) << (bits - 7) ^
        static_cast<size_t>(rp.split_grid_dim) << (bits - 8);
    return attr_hash;
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
