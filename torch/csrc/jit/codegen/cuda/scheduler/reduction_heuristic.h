#pragma once

#include <torch/csrc/jit/codegen/cuda/scheduler/heuristic.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Parameters of the reduction heuristic to describe the optimial schedule.
// Warning: equal operator is intended for use in caching the kernel associated
// with these reduction parameters. It does not check if the launch parameters
// are equivelent!
class ReductionParams : public HeuristicParams {
 public:
  // Reducing inner most dimension?
  bool fastest_dim = false;

  // Store input in shared memory or registers to reduce global memory reads
  bool persistent_kernel = false;

  // Project persistent buffers back to inputs to reduce persistent buffer size
  bool project_persistent_buffers = false;

  // Are we treating the scheduling as 3 dimensional, can be useful for patterns
  // like [reduction, iteration, reduction].
  bool schedule_3D = false;

  // For outer reductions we may want to swap the gdimx and gdimy bindings to
  // amortize the cost of the final cleanup in grid reductions.
  bool flip_grid = false;

  // Inner Reduction Domain:

  // Reduce across the block?
  bool cross_block_inner_reduction = false;
  // Reduce across the grid?
  bool cross_grid_inner_reduction = false;
  // Unrolling/Vectorization factor for inner reduction dimension
  int64_t unroll_factor_inner_reduction = 1;
  // vectorize instead of unroll
  bool vectorize_inner_reduction = false;
  // Split grid dim for iteration axis in case it's too large for cuda
  bool split_grid_dim_inner_reduction = false;
  // Pad inner dimension to nearest warp
  bool pad_inner_reduction_to_warp = false;
  // Register persistent buffer size in inner dimension
  int64_t batches_per_block_inner_reduction = 1;

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
  // Unrolling/Vectorization factor for iteration dimension
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
  bool cross_block_outer_reduction = false;
  // Reduce across the grid?
  bool cross_grid_outer_reduction = false;
  // Split grid dim for iteration axis in case it's too large for cuda
  bool split_grid_dim_outer_reduction = false;
  // Register persistent buffer size in outer dimension
  int64_t batches_per_block_outer_reduction = 1;
  // Unrolling/Vectorization factor for outer reduction factor
  int64_t unroll_factor_outer_reduction = 1;

  // Which block parallel dimension should be used for the outer reduction.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType block_dim_outer_reduction = ParallelType::Serial;
  // Which grid parallel dimension should be used for the outer reduction.
  // !!WARNING!! Convenience method, this be unique based on non-parallel type
  // parameters, not used for equivalence/hashing.
  ParallelType grid_dim_outer_reduction = ParallelType::Serial;

  bool isUnrolled() const {
    return unroll_factor_inner_reduction > 1 || unroll_factor_iter_dom > 1 ||
        unroll_factor_outer_reduction > 1;
  }

 public:
  using HeuristicParams::HeuristicParams;

  // Warning: Does not check launch parameters!
  bool sameAs(
      const std::shared_ptr<HeuristicParams>& other_base) const override {
    auto other_casted = std::dynamic_pointer_cast<ReductionParams>(other_base);
    if (other_casted == nullptr) {
      return false;
    }
    const ReductionParams& other = *other_casted;
    bool attr_equal = other.fastest_dim == fastest_dim &&
        other.persistent_kernel == persistent_kernel &&
        other.project_persistent_buffers == project_persistent_buffers &&
        other.schedule_3D == schedule_3D && other.flip_grid == flip_grid &&
        other.cross_block_inner_reduction == cross_block_inner_reduction &&
        other.cross_grid_inner_reduction == cross_grid_inner_reduction &&
        other.unroll_factor_inner_reduction == unroll_factor_inner_reduction &&
        other.vectorize_inner_reduction == vectorize_inner_reduction &&
        other.split_grid_dim_inner_reduction ==
            split_grid_dim_inner_reduction &&
        other.pad_inner_reduction_to_warp == pad_inner_reduction_to_warp &&
        other.batches_per_block_inner_reduction ==
            batches_per_block_inner_reduction &&
        other.multiple_reds_per_blk == multiple_reds_per_blk &&
        other.unroll_factor_iter_dom == unroll_factor_iter_dom &&
        other.vectorize_iter_dom == vectorize_iter_dom &&
        other.split_grid_dim_iter_dom == split_grid_dim_iter_dom &&
        other.cross_block_outer_reduction == cross_block_outer_reduction &&
        other.cross_grid_outer_reduction == cross_grid_outer_reduction &&
        other.unroll_factor_outer_reduction == unroll_factor_outer_reduction &&
        other.split_grid_dim_outer_reduction ==
            split_grid_dim_outer_reduction &&
        other.batches_per_block_outer_reduction ==
            batches_per_block_outer_reduction;
    return attr_equal;
  }

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Reduction Parameters ========\n"
       << (tag == "" ? "" : "Tag: ") << tag << "\n"
       << (fastest_dim ? "Red On Fastest Dim\n" : "Red On Slow Dim\n")
       << (persistent_kernel ? "Persistent Kernel\n" : "")
       << (project_persistent_buffers ? "Project Persistent Buffers\n" : "");
    if (batches_per_block_inner_reduction > 1 || persistent_kernel) {
      ss << "Batches per block: " << batches_per_block_inner_reduction << "\n";
    }

    if (schedule_3D) {
      ss << "3D Schedule\n"
         << "Outer Reduction: ";
      if (cross_block_outer_reduction) {
        ss << "cross block - " << block_dim_outer_reduction << " / ";
      }
      if (cross_grid_outer_reduction) {
        ss << "cross grid - " << grid_dim_outer_reduction << " / ";
        ss << (split_grid_dim_outer_reduction ? "split grid dim / " : "");
      }

      ss << (unroll_factor_outer_reduction > 1 ? "unroll / " : "");
      if (unroll_factor_outer_reduction > 1) {
        ss << "factor " << unroll_factor_outer_reduction << " ";
      }

      if (batches_per_block_outer_reduction > 1 || persistent_kernel) {
        ss << "persistent batch - " << batches_per_block_outer_reduction;
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
       << (unroll_factor_iter_dom > 1 && !vectorize_iter_dom ? "unroll / "
                                                             : "");
    if (unroll_factor_iter_dom > 1) {
      ss << "factor " << unroll_factor_iter_dom;
    }

    ss << "\nInner Reduction Domain: ";

    if (cross_block_inner_reduction) {
      ss << "cross block - " << block_dim_inner_reduction << " / ";
      ss << (pad_inner_reduction_to_warp ? " pad to warp / " : "");
    }
    if (cross_grid_inner_reduction) {
      ss << "cross grid - " << grid_dim_inner_reduction << " / ";
      ss << (split_grid_dim_inner_reduction ? "split grid dim / " : "");
    }
    if (batches_per_block_inner_reduction > 1 || persistent_kernel) {
      ss << "persistent batch - " << batches_per_block_inner_reduction << " / ";
    }
    ss << (cross_grid_inner_reduction && split_grid_dim_inner_reduction
               ? "split grid dimension / "
               : "")
       << (vectorize_inner_reduction ? "vectorize / " : "")
       << (unroll_factor_inner_reduction > 1 && !vectorize_inner_reduction
               ? "unroll / "
               : "");
    if (unroll_factor_inner_reduction > 1) {
      ss << "factor " << unroll_factor_inner_reduction;
    }

    ss << "\n" << lparams.toString() << "\n";
    ss << "====================================\n";
    return ss.str();
  }

  // Warning: Hash is not based on launch parameters!
  size_t hash() const override {
    constexpr size_t bits = sizeof(std::size_t) * 8;
    size_t attr_hash = static_cast<size_t>(fastest_dim) << (bits - 1) ^
        static_cast<size_t>(persistent_kernel) << (bits - 2) ^
        static_cast<size_t>(project_persistent_buffers) << (bits - 3) ^
        static_cast<size_t>(schedule_3D) << (bits - 4) ^
        static_cast<size_t>(flip_grid) << (bits - 5) ^
        static_cast<size_t>(cross_block_inner_reduction) << (bits - 6) ^
        static_cast<size_t>(cross_grid_inner_reduction) << (bits - 7) ^
        static_cast<size_t>(unroll_factor_inner_reduction) << (bits - 8) ^
        static_cast<size_t>(vectorize_inner_reduction) << (bits - 9) ^
        static_cast<size_t>(split_grid_dim_inner_reduction) << (bits - 10) ^
        static_cast<size_t>(pad_inner_reduction_to_warp) << (bits - 11) ^
        static_cast<size_t>(batches_per_block_inner_reduction) << (bits - 12) ^
        static_cast<size_t>(multiple_reds_per_blk) << (bits - 13) ^
        static_cast<size_t>(unroll_factor_iter_dom) << (bits - 14) ^
        static_cast<size_t>(vectorize_iter_dom) << (bits - 15) ^
        static_cast<size_t>(split_grid_dim_iter_dom) << (bits - 16) ^
        static_cast<size_t>(cross_block_outer_reduction) << (bits - 17) ^
        static_cast<size_t>(cross_grid_outer_reduction) << (bits - 18) ^
        static_cast<size_t>(split_grid_dim_outer_reduction) << (bits - 19) ^
        static_cast<size_t>(batches_per_block_outer_reduction) << (bits - 20) ^
        static_cast<size_t>(unroll_factor_outer_reduction) << (bits - 21);
    return attr_hash;
  }

  std::shared_ptr<HeuristicParams> clone() const override {
    return std::make_shared<ReductionParams>(*this);
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
