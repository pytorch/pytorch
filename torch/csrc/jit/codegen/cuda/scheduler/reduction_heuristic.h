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
  bool fastest_dim = false;

  // Store input in shared memory or registers to reduce global memory reads
  bool persistent_kernel = false;

  // Number of batches for each block
  int64_t batches_per_block = 1;

  // Are we treating the scheduling as 3 dimensional, can be useful for patterns
  // like [reduction, iteration, reduction].
  bool schedule_3D = false;

  // Inner Reduction Domain:

  // Reduce across the block?
  bool cross_block_inner_reduce = false;
  // Reduce across the grid?
  bool cross_grid_inner_reduce = false;
  // Inner reduction unroll/vectorize
  bool unroll_inner_reduction = false;
  // Unrolling factor
  int64_t unroll_factor_inner_reduction = 1;
  // vectorize instead of unroll
  bool vectorize_inner_reduction = false;
  // Split grid dim for iteration axis in case it's too large for cuda
  bool split_grid_dim_inner_reduction = false;

  // Which block parallel dimension should be used for the inner reduction.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType block_dim_inner_reduction = ParallelType::Serial;
  // Which grid parallel dimension should be used for the inner reduction.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType grid_dim_inner_reduction = ParallelType::Serial;

  // Iteration Domain:

  // Perform multiple reductions per block?
  bool multiple_reds_per_blk = false;
  // Iteration dimension unroll/vectorize
  bool unroll_iter_dom = false;
  // Unrolling factor
  int64_t unroll_factor_iter_dom = 1;
  // vectorize instead of unroll
  bool vectorize_iter_dom = false;
  // Split grid dim for iteration axis in case it's too large for cuda
  bool split_grid_dim_iter_dom = false;

  // Which block parallel dimension should be used for the iter domain.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType block_dim_iter_dom = ParallelType::Serial;
  // Which grid parallel dimension should be used for the iter domain.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType grid_dim_iter_dom = ParallelType::Serial;

  // Outer Reduction Domain if 3D Scheduled:

  // Reduce across the block?
  bool cross_block_outer_reduce = false;
  // Reduce across the grid?
  bool cross_grid_outer_reduce = false;
  // Split grid dim for iteration axis in case it's too large for cuda
  bool split_grid_dim_outer_reduction = false;

  // Which block parallel dimension should be used for the outer reduction.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType block_dim_outer_reduction = ParallelType::Serial;
  // Which grid parallel dimension should be used for the outer reduction.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType grid_dim_outer_reduction = ParallelType::Serial;

  std::string tag = "";

  LaunchParams lparams;

 public:
  // Warning: Does not check launch parameters!
  bool operator==(const ReductionParams& other) const {
    bool attr_equal = other.fastest_dim == fastest_dim &&
        other.batches_per_block == batches_per_block &&
        other.persistent_kernel == persistent_kernel &&
        other.schedule_3D == schedule_3D &&
        other.cross_block_inner_reduce == cross_block_inner_reduce &&
        other.cross_grid_inner_reduce == cross_grid_inner_reduce &&
        other.unroll_inner_reduction == unroll_inner_reduction &&
        other.unroll_factor_inner_reduction == unroll_factor_inner_reduction &&
        other.vectorize_inner_reduction == vectorize_inner_reduction &&
        other.split_grid_dim_inner_reduction ==
            split_grid_dim_inner_reduction &&
        other.multiple_reds_per_blk == multiple_reds_per_blk &&
        other.unroll_iter_dom == unroll_iter_dom &&
        other.unroll_factor_iter_dom == unroll_factor_iter_dom &&
        other.vectorize_iter_dom == vectorize_iter_dom &&
        other.split_grid_dim_iter_dom == split_grid_dim_iter_dom &&
        other.cross_block_outer_reduce == cross_block_outer_reduce &&
        other.cross_grid_outer_reduce == cross_grid_outer_reduce &&
        other.split_grid_dim_outer_reduction == split_grid_dim_outer_reduction;
    return attr_equal;
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "\n===== Reduction Parameters ========\n"
       << (tag == "" ? "" : "Tag: ") << tag << "\n"
       << (fastest_dim ? "Red On Fastest Dim\n" : "Red On Slow Dim\n")
       << (persistent_kernel ? "Persistent Kernel\n" : "");
    if (batches_per_block > 1 || persistent_kernel) {
      ss << "Batches per block: " << batches_per_block << "\n";
    }

    if (schedule_3D) {
      ss << "3D Schedule\n"
         << "Outer Reduction: ";
      if (cross_block_outer_reduce) {
        ss << "cross block - " << block_dim_outer_reduction << " / ";
      }
      if (cross_grid_outer_reduce) {
        ss << "cross grid - " << grid_dim_outer_reduction << " / ";
        ss << (split_grid_dim_outer_reduction ? "split grid dim / " : "");
      }
    }

    ss << "\nIteration Domain: ";

    if (grid_dim_iter_dom != ParallelType::Serial) {
      ss << grid_dim_iter_dom << " / "
         << (split_grid_dim_iter_dom ? "split grid dimension / " : "");
    }
    if (block_dim_iter_dom != ParallelType::Serial) {
      ss << block_dim_iter_dom << " / ";
    }
    ss << (multiple_reds_per_blk ? "multiple reductions per block / " : "")
       << (vectorize_iter_dom ? "vectorize / " : "")
       << (unroll_iter_dom && !vectorize_iter_dom ? "unroll / " : "");
    if (unroll_iter_dom || vectorize_iter_dom) {
      ss << "factor " << unroll_factor_iter_dom;
    }

    ss << "\nInner Reduction Domain: ";

    if (cross_block_inner_reduce) {
      ss << "cross block - " << block_dim_inner_reduction << " / ";
    }
    if (cross_grid_inner_reduce) {
      ss << "cross grid - " << grid_dim_inner_reduction << " / ";
      ss << (split_grid_dim_inner_reduction ? "split grid dim / " : "");
    }
    ss << (cross_grid_inner_reduce && split_grid_dim_inner_reduction
               ? "split grid dimension / "
               : "")
       << (vectorize_inner_reduction ? "vectorize / " : "")
       << (unroll_inner_reduction && !vectorize_inner_reduction ? "unroll / "
                                                                : "");
    if (unroll_inner_reduction || vectorize_inner_reduction) {
      ss << "factor " << unroll_factor_inner_reduction;
    }

    ss << "\n" << lparams.toString() << "\n";
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
        static_cast<size_t>(rp.batches_per_block) ^
        static_cast<size_t>(rp.persistent_kernel) << (bits - 2) ^
        static_cast<size_t>(rp.schedule_3D) << (bits - 3) ^
        static_cast<size_t>(rp.cross_block_inner_reduce) << (bits - 4) ^
        static_cast<size_t>(rp.cross_grid_inner_reduce) << (bits - 5) ^
        static_cast<size_t>(rp.unroll_inner_reduction) << (bits - 6) ^
        static_cast<size_t>(rp.unroll_factor_inner_reduction) ^
        static_cast<size_t>(rp.vectorize_inner_reduction) << (bits - 7) ^
        static_cast<size_t>(rp.split_grid_dim_inner_reduction) << (bits - 8) ^
        static_cast<size_t>(rp.multiple_reds_per_blk) << (bits - 9) ^
        static_cast<size_t>(rp.unroll_iter_dom) << (bits - 10) ^
        static_cast<size_t>(rp.unroll_factor_iter_dom) ^
        static_cast<size_t>(rp.vectorize_iter_dom) << (bits - 11) ^
        static_cast<size_t>(rp.split_grid_dim_iter_dom) << (bits - 12) ^
        static_cast<size_t>(rp.cross_block_outer_reduce) << (bits - 13) ^
        static_cast<size_t>(rp.cross_grid_outer_reduce) << (bits - 14) ^
        static_cast<size_t>(rp.split_grid_dim_outer_reduction) << (bits - 15);
    return attr_hash;
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
