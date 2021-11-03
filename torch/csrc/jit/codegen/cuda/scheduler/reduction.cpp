#include <torch/csrc/jit/codegen/cuda/scheduler/reduction.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

ReductionParams innerReductionHeuristic(
    const int64_t num_elems_in_reduction,
    const int64_t num_outputs_for_reduction,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size,
    size_t vectorize_factor) {
  // Set some targets for parallelization

  const int64_t n_elems = num_elems_in_reduction * num_outputs_for_reduction;

  // WARNING: Current device for codegen may not be the target device
  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)16 / (int64_t)max_input_dtype_size,
      // Reduce unrolling if we have many inputs, start reduction at 4 inputs
      std::max(
          (scheduler_utils::lastPow2((int64_t)n_tensor_inputs) >> 2),
          (int64_t)1));

  // Conservative value, could be set to larger based on arch if necessary.
  constexpr int64_t l1_cache = 32 * 1024;
  // Could change per generation, but for l1 we want to consider active threads,
  // not resident
  constexpr int64_t active_threads = 1024;

  // if data fits in l2 and we need more parallelization in the reduction dim,
  // we can use a smaller warp size. While thread local data fits in l1, and
  // reduction dim is really small, we can use <32 threads per warp.
  const bool fits_in_l2 = n_elems * max_input_dtype_size * n_tensor_inputs <
      at::cuda::getCurrentDeviceProperties()->l2CacheSize;

  // If it fits in l2, we just want to make sure each thread uses 32Bytes.
  const int64_t warp_size_based_on_l2 =
      fits_in_l2 ? (int64_t)32 / max_input_dtype_size : 32;

  // Check how many elements it would take per thread to start thrashing l1
  // set that to minimum number we want to reduce per thread.
  const int64_t warp_size_based_on_l1 = std::min(
      ceilDiv(
          num_elems_in_reduction,
          std::max(
              l1_cache /
                  (n_tensor_inputs * max_input_dtype_size * active_threads),
              (int64_t)1)),
      (int64_t)16);

  // Take the smaller
  const int64_t warp_size =
      std::min(warp_size_based_on_l1, warp_size_based_on_l2);

  // Initialization
  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t target_iterations = 1;

  // Try to set a minmum amount of work for each thread, as cross thread
  // communication is slow so it shouldn't be done for every element in the
  // reduction.
  int64_t min_target_iterations =
      std::max((int64_t)32 / (int64_t)max_input_dtype_size, (int64_t)1);

  // Start trying to break parallelization up across threads,
  // unrolling/iterations, and blocks.

  // max_threads_in_block is the cap on a thread block, the minimum is based on
  // warp_size
  int64_t max_threads_in_block = std::max(
      warp_size, ceilDiv(num_elems_in_reduction, min_target_iterations));

  // If we have one warp per block, check if that's enough to saturate the SMs
  target_blocks = ceilDiv(n_elems, warp_size);

  // If we have more than a wave of blocks, put parallelism into unrolling and
  // target iterations
  if (target_blocks > device_multiprocessor_count) {
    auto available_unroll = std::max(
        n_elems / (warp_size * device_multiprocessor_count), (int64_t)1);

    // Spread across unrolling and iterations, want a balance of the two so flip
    // back and forth to alternate adding to them.
    bool flip = true;

    while (available_unroll > 1 &&
           (target_unroll < max_unroll ||
            // Prefer unrolling
            target_iterations < ceilDiv(min_target_iterations, max_unroll))) {
      if (target_unroll * 2 <= max_unroll && flip) {
        target_unroll *= 2;
      }

      if (target_iterations * 2 <= ceilDiv(min_target_iterations, max_unroll) &&
          !flip) {
        target_iterations *= 2;
      }

      available_unroll = std::max(
          n_elems /
              (warp_size * device_multiprocessor_count * target_unroll *
               target_iterations),
          (int64_t)1);

      flip = !flip;
    }

    // Recompute target blocks
    target_blocks =
        ceilDiv(n_elems, warp_size * target_unroll * target_iterations);
  }

  // Cap target blocks to 4 waves
  target_blocks = std::min(target_blocks, device_multiprocessor_count * 4);

  if (target_blocks * target_unroll * target_iterations < n_elems) {
    // targeting 4 waves, so try to use a quarter of available threads
    max_threads_in_block = std::min(
        ceilDiv(n_elems, target_blocks * target_unroll),
        ceilDiv(device_max_threads_per_multiprocessor, (int64_t)4));
  }

  // To get to target threads:
  // Prioritize
  // (1) x dim in reduction
  // (2) unrolling in reduction
  // (3) y in output
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions

  // Blocks for reductions
  int64_t grdim = 1;
  // Blocks for outputs
  int64_t godim = 1;

  // Threads for outputs
  int64_t bdimy = 1;
  // Threads for reduction
  int64_t bdimx = 1;

  // Should we unroll from reduction axis, or outs axis
  bool unroll_reduction = true;

  // Unroll amount
  int64_t unroll_factor = 1;

  // Grab what we can out of reduction domain, but don't go over a warp size yet
  bdimx = std::min(num_elems_in_reduction, (int64_t)warp_size);
  // Put everything else in bdimy for now
  bdimy = std::max(max_threads_in_block / bdimx, (int64_t)1);
  int64_t remainder_in_reduction = ceilDiv(num_elems_in_reduction, bdimx);
  int64_t remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimy);

  // Adjust blocking and setup unrolling
  if (remainder_in_reduction == 1) {
    // Small number of reduction elements, try unrolling output dimension
    unroll_factor = std::min(target_unroll, remainder_in_output);
    if (unroll_factor > 1) {
      unroll_reduction = false;
      remainder_in_output =
          ceilDiv(num_outputs_for_reduction, unroll_factor * bdimy);
    }
  } else {
    // If there are reduction elements left after unrolling a warp, re-adjust
    // the block dims to put more threads into the reduction
    bdimx = std::min(
        std::max(
            ceilDiv(num_elems_in_reduction, target_iterations * target_unroll),
            warp_size),
        max_threads_in_block);

    // Don't exceed target.
    bdimy = std::max(max_threads_in_block / bdimx, (int64_t)1);
    remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimy);

    remainder_in_reduction = ceilDiv(num_elems_in_reduction, bdimx);
    unroll_factor = std::min(remainder_in_reduction, target_unroll);
    if (unroll_factor == 1) {
      // If we can't unroll reduction dim, unroll output dim
      unroll_factor = std::min(remainder_in_output, target_unroll);
      if (unroll_factor > 1) {
        unroll_reduction = false;
      }
      remainder_in_output =
          ceilDiv(num_outputs_for_reduction, bdimy * unroll_factor);
      remainder_in_reduction =
          ceilDiv(num_elems_in_reduction, bdimx * target_iterations);
    } else {
      remainder_in_reduction = ceilDiv(
          num_elems_in_reduction,
          bdimx * std::max(unroll_factor, target_iterations));
    }
  }

  godim = remainder_in_output;

  // Clang tidy
  constexpr int64_t kEight = 8;
  constexpr int64_t kThirtyTwo = 32;

  // Cross grid reduction if we haven't hit our target blocks, and we have many
  // reduction elements.
  if ((godim < target_blocks && remainder_in_reduction > kEight &&
       remainder_in_reduction < kThirtyTwo) ||
      (remainder_in_reduction >= kThirtyTwo)) {
    // Grid reductions do not support unrolling iteration dimension, revert if
    // set.
    if (!unroll_reduction) {
      unroll_reduction = true;
      unroll_factor = 1;
      remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimy);
      remainder_in_reduction =
          ceilDiv(num_elems_in_reduction, bdimx * target_iterations);
    }
    if (remainder_in_reduction >= kThirtyTwo) {
      // Do at least 2 iterations of unrolling per thread before we go cross
      // grid. Limit cross grid to a multiple of the block size so cleanup on
      // the last block doesn't take too long.
      grdim = std::min(
          ceilDiv(remainder_in_reduction, (int64_t)2), bdimx * bdimy * kEight);
      // Clang tidy
      // remainder_in_reduction = ceilDiv(remainder_in_reduction, grdim);
    } else {
      grdim = ceilDiv(remainder_in_reduction, (int64_t)4);
    }
    // Clang tidy
    //
    // remainder_in_reduction = ceilDiv(
    //     num_elems_in_reduction,
    //     bdimx *
    //         std::max(
    //             unroll_reduction ? unroll_factor : 1,
    //             min_red_elems_per_thread) *
    //         grdim);
  }

  // Try to do some cleanup of ragged waves on device
  // godim is a remainder of a split, so can only control bdimy
  if (
      // If we have less than 8 waves of blocks
      grdim * godim < device_multiprocessor_count * kEight &&
      // And we don't have an even divisible number of blocks
      (grdim * godim) % device_multiprocessor_count != 0 &&
      // And we have more than one wave
      grdim * godim > device_multiprocessor_count) {
    // round waves down
    auto waves =
        std::max((godim * grdim) / device_multiprocessor_count, (int64_t)1);
    auto new_grdim =
        std::max((waves * device_multiprocessor_count) / godim, (int64_t)1);
    if (
        // If difference is less than 25% of the original grdim
        (new_grdim - grdim) * 4 < grdim &&
        // and difference is less than 25% of the original number of blocks
        ((new_grdim * godim) - (grdim * godim)) * 4 < grdim * godim) {
      grdim = new_grdim;
    }
  }

  bool vectorize = false;

  if (vectorize_factor > 1 && unroll_factor > 1 && unroll_reduction) {
    vectorize = true;
    unroll_factor = std::min(
        scheduler_utils::lastPow2(unroll_factor), (int64_t)vectorize_factor);
  }

  ReductionParams rparams;
  rparams.fastest_dim = true;
  rparams.cross_block = true;
  rparams.cross_grid = grdim > 1;
  rparams.multiple_reds_per_blk = bdimy > 1;
  rparams.loop_unroll = unroll_factor;
  rparams.vectorize = vectorize;
  rparams.reduction_unroll = unroll_reduction;

  // If we have a cross grid case we want to have gdimy assigned to godim and
  // gdimx assigned to grdim. Otherwise it's helpful to pull godim into gdimx in
  // case it's larger than gdimy can hold, as not doing so can thrash the cache.
  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;

  if (rparams.cross_grid) {
    gdimx = grdim;
    rparams.split_grid_dim = gdimy > scheduler_utils::y_grid_limit;
  } else {
    rparams.split_grid_dim = gdimx > scheduler_utils::x_grid_limit;
  }

  rparams.lparams = LaunchParams(
      gdimx,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);
  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    std::cerr << "\n===== Reduction Stats ========\n"
              << "num_elems_in_reduction: " << num_elems_in_reduction << "\n"
              << "num_outputs_for_reduction: " << num_outputs_for_reduction
              << "\n"
              << "n_tensor_inputs: " << n_tensor_inputs << "\n"
              << "max_input_dtype_size: " << max_input_dtype_size << std::endl;
    std::cerr << rparams.toString() << std::endl;
  }

  return rparams;
}

ReductionParams OuterReductionHeuristic(
    const int64_t num_elems_in_reduction,
    const int64_t num_outputs_for_reduction,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size,
    size_t vectorize_factor) {
  // Set some targets for parallelization

  const int64_t n_elems = num_elems_in_reduction * num_outputs_for_reduction;
  const int64_t l2_cache_size =
      at::cuda::getCurrentDeviceProperties()->l2CacheSize;

  const int64_t warp_size =
      n_elems * max_input_dtype_size * n_tensor_inputs < l2_cache_size
      ? (int64_t)32 / max_input_dtype_size
      : 32;

  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t max_threads_in_block = warp_size;

  // WARNING: Current device for codegen may not be the target device
  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)16 / (int64_t)max_input_dtype_size,
      // Reduce unrolling if we have many inputs, start reduction at 4 inputs
      std::max(
          (scheduler_utils::lastPow2((int64_t)n_tensor_inputs) >> 2),
          (int64_t)1));

  // If we have one warp per block, how many blocks would that be?
  target_blocks = ceilDiv(n_elems, (int64_t)warp_size);

  // If we have more than a wave, put parallelism into unrolling
  if (target_blocks > device_multiprocessor_count) {
    target_unroll = std::min(
        max_unroll, ceilDiv(target_blocks, device_multiprocessor_count));
    target_blocks = ceilDiv(target_blocks, target_unroll);
  }

  // Cap target blocks to 4 waves
  target_blocks = std::min(target_blocks, device_multiprocessor_count * 4);

  if (target_blocks * target_unroll * max_threads_in_block < n_elems) {
    // targeting 4 waves, so try to use a quarter of available threads
    max_threads_in_block = std::min(
        ceilDiv(n_elems, target_blocks * target_unroll),
        ceilDiv(device_max_threads_per_multiprocessor, (int64_t)4));
  }

  // To get to target threads:
  // Prioritize
  // (1) x dim in iter domain
  // (2) unrolling in iter domain
  // (3) y in reduction domain
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions - need to flip unrolling to reduction
  // domain for this

  // Blocks for reductions
  int64_t gdimy = 1;
  // Blocks for outputs
  int64_t gdimx = 1;

  // Threads for reduction
  int64_t bdimy = 1;
  // Threads for output
  int64_t bdimx = 1;

  // Should we unroll from reduction axis, or outs axis
  bool unroll_reduction = false;

  // Unroll amount
  int64_t unroll_factor = 1;

  int64_t remainder_in_reduction = num_elems_in_reduction;
  int64_t remainder_in_output = num_outputs_for_reduction;

  if (ceilDiv(num_outputs_for_reduction, warp_size) <
      device_multiprocessor_count) {
    // If we can't hit a full wave, leave bdimx as warp_size, and prioritize
    // bdimy. TODO: Re-evaluate, should it be bdimx = warp_size?
    bdimx = std::min(num_outputs_for_reduction, warp_size);
  } else {
    bdimx = std::min(
        max_threads_in_block,
        ceilDiv(num_outputs_for_reduction, target_blocks));
    bdimx = std::max(bdimx, warp_size);
  }

  bdimy = std::min(
      std::max(max_threads_in_block / bdimx, (int64_t)1),
      num_elems_in_reduction);

  // Clang tidy
  // remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimx);
  remainder_in_reduction = ceilDiv(remainder_in_reduction, bdimy);

  if (num_outputs_for_reduction >=
      device_multiprocessor_count * max_threads_in_block) {
    // If we easily saturate the GPU, don't use block dim y and unroll output
    // dimension, this could be a more gentle transition starting earlier
    bdimx = max_threads_in_block;
    remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimx);

    bdimy = 1;
    remainder_in_reduction = num_elems_in_reduction;

    // Assume unroll in output, switch to remainder if cross grid
    // Don't unroll if we don't have 2 full waves
    unroll_factor = std::min(
        ceilDiv(remainder_in_output, device_multiprocessor_count * 2),
        target_unroll);

    if (unroll_factor == 1 && remainder_in_reduction > 1) {
      // Try unrolling in reduction dimension
      unroll_factor = std::min(remainder_in_reduction, unroll_factor);
      // Clang tidy
      // remainder_in_reduction = ceilDiv(remainder_in_reduction,
      // unroll_factor);
      if (unroll_factor > 1) {
        unroll_reduction = true;
      }
    }
    // Clang tidy
    // else {
    //   remainder_in_output =
    //       ceilDiv(num_outputs_for_reduction, bdimx * unroll_factor);
    // }
  } else {
    // Not many output elements, so we want to try expand grid level parallelism
    // first go after unrolling
    unroll_factor = std::min(max_unroll, remainder_in_reduction);
    if (unroll_factor > 1) {
      unroll_reduction = true;
    }

    remainder_in_reduction =
        ceilDiv(num_elems_in_reduction, bdimy * unroll_factor);

    // Go cross grid
    gdimy = ceilDiv(remainder_in_reduction, (int64_t)4);
    // Clang tidy
    // remainder_in_reduction =
    //     ceilDiv(num_elems_in_reduction, bdimy * unroll_factor * gdimy);
  }

  // Clang tidy
  constexpr int64_t kEight = 8;
  constexpr int64_t kSixteen = 16;
  constexpr int64_t kThirtyTwo = 32;

  if (ceilDiv(num_elems_in_reduction, bdimy * unroll_factor) >= kThirtyTwo) {
    // Many reduction elements, go cross grid
    int64_t min_gdimy = 1;
    if (gdimy > 1) {
      // already cross grid, don't go below target or what was already set
      min_gdimy = std::min(gdimy, ceilDiv(target_blocks, gdimx));
    }
    gdimy = std::max(
        min_gdimy,
        ceilDiv(
            ceilDiv(num_elems_in_reduction, bdimy * unroll_factor),
            (int64_t)kSixteen));
    // Don't go too far above number of threads in a block since that's how many
    // threads are available to do final reduction iteration
    // This is good!
    gdimy = std::min(gdimy, bdimx * bdimy * kEight);
  }

  // Try to do some cleanup of ragged waves on device
  if (
      // If we have less than 8 waves of blocks
      gdimy * gdimx < device_multiprocessor_count * kEight &&
      // And we don't have an even divisible number of blocks
      (gdimy * gdimx) % device_multiprocessor_count != 0 &&
      // And we have more than one wave
      gdimy * gdimx > device_multiprocessor_count) {
    // round waves down
    auto waves =
        std::max((gdimx * gdimy) / device_multiprocessor_count, (int64_t)1);
    auto new_gdimy =
        std::max((waves * device_multiprocessor_count) / gdimx, (int64_t)1);
    if (
        // If difference is less than 25% of the original gdimy
        (new_gdimy - gdimy) * 4 < gdimy &&
        // and difference is less than 25% of the original number of blocks
        ((new_gdimy * gdimx) - (gdimy * gdimx)) * 4 < gdimy * gdimx) {
      gdimy = new_gdimy;
    }
  }

  // Cannot unroll with cross grid reductions
  if (gdimy > 1 && !unroll_reduction) {
    unroll_reduction = true;
    unroll_factor = 1;
  }

  bool vectorize = false;

  if (vectorize_factor > 1 && unroll_factor > 1 && !unroll_reduction) {
    vectorize = true;
    unroll_factor = std::min(
        scheduler_utils::lastPow2(unroll_factor), (int64_t)vectorize_factor);
  }

  ReductionParams rparams;
  rparams.fastest_dim = false;
  // cross grid implies cross block
  rparams.cross_block = bdimy > 1 || gdimy > 1;
  rparams.cross_grid = gdimy > 1;
  rparams.multiple_reds_per_blk = bdimx > 1;
  rparams.loop_unroll = unroll_factor;
  rparams.vectorize = vectorize;
  rparams.reduction_unroll = unroll_reduction;

  rparams.lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    std::cerr << "\n===== Reduction Stats ========\n"
              << "num_elems_in_reduction: " << num_elems_in_reduction << "\n"
              << "num_outputs_for_reduction: " << num_outputs_for_reduction
              << "\n"
              << "n_tensor_inputs: " << n_tensor_inputs << "\n"
              << "max_input_dtype_size: " << max_input_dtype_size << std::endl;
    std::cerr << rparams.toString() << std::endl;
  }
  return rparams;
}

} // namespace

ReductionParams reductionHeuristic(
    int64_t num_elems_in_reduction,
    int64_t num_outputs_for_reduction,
    bool fastest_dim_reduction,
    size_t n_tensor_inputs,
    size_t max_input_dtype_size,
    size_t vectorize_factor) {
  if (fastest_dim_reduction) {
    return innerReductionHeuristic(
        num_elems_in_reduction,
        num_outputs_for_reduction,
        n_tensor_inputs,
        max_input_dtype_size,
        vectorize_factor);
  } else {
    return OuterReductionHeuristic(
        num_elems_in_reduction,
        num_outputs_for_reduction,
        n_tensor_inputs,
        max_input_dtype_size,
        vectorize_factor);
  }
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getReductionHeuristics");

  SchedulerRuntimeInfo runtime_info(fusion, runtime_inputs, true);

  return getReductionHeuristics(fusion, runtime_info, data_cache);
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache) {
  FUSER_PERF_SCOPE("getReductionHeuristics");

  FusionGuard fg(fusion);

  HeuristicCacheAccessor<std::vector<TensorView*>> reduction_tv_data;
  // TODO: move all these boilerplate code into the accessor class
  // (follow up)
  if (data_cache && !data_cache->isRecording()) {
    reduction_tv_data.writeTemporary(data_cache->getReductionTVs());
  } else {
    reduction_tv_data.writeNew(scheduler_utils::getReductionTvs(fusion));
    if (data_cache && data_cache->isRecording()) {
      data_cache->setReductionTVs(reduction_tv_data.read());
    }
  }

  auto& reduction_tvs = reduction_tv_data.read();

  TORCH_INTERNAL_ASSERT(
      reduction_tvs.size() == 1, "Need reduction tensor views to schedule.");

  auto reduction_tv = reduction_tvs[0];

  TORCH_INTERNAL_ASSERT(reduction_tv != nullptr);

  auto red_root_dom = reduction_tv->getRootDomain();
  bool fastest_dim_reduction = true;
  for (size_t i = red_root_dom.size(); i > 0; i--) {
    if (red_root_dom[i - 1]->isBroadcast() ||
        red_root_dom[i - 1]->isTrivialReduction()) {
      continue;
    } else if (red_root_dom[i - 1]->isReduction()) {
      fastest_dim_reduction = true;
      break;
    } else {
      fastest_dim_reduction = false;
      break;
    }
  }

  TORCH_INTERNAL_ASSERT(
      reduction_tv != nullptr, "Reduction TensorView wasn't found.");

  TORCH_INTERNAL_ASSERT(
      reduction_tv->hasReduction(), "TensorView doesn't have a reduction.");
  const auto red_expr = reduction_tv->definition();

  TORCH_INTERNAL_ASSERT(
      red_expr->getExprType() != c10::nullopt &&
          (red_expr->getExprType().value() == ExprType::ReductionOp ||
           red_expr->getExprType().value() == ExprType::WelfordOp),
      "TensorView doesn't have a reduction.");

  int64_t num_outputs_for_reduction = 1;
  int64_t red_elements = 1;

  for (auto id : reduction_tv->getRootDomain()) {
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(id->extent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(), "Error inferring reduction size.");
    if (id->isReduction()) {
      red_elements *= inferred_val.value();
    } else {
      num_outputs_for_reduction *= inferred_val.value();
    }
  }

  size_t max_dtype_size = 1;
  size_t n_tensor_inputs = 0;
  for (auto inp : fusion->inputs()) {
    if (inp->isA<TensorView>()) {
      max_dtype_size =
          std::max(max_dtype_size, dataTypeSize(inp->getDataType().value()));
      n_tensor_inputs++;
    }
  }

  TORCH_INTERNAL_ASSERT(
      n_tensor_inputs > 0,
      "Tried to schedule a fusion with no tensor inputs, currently not supported.");

  auto vectorizable_inputs_outputs =
      scheduler_utils::getVectorizableInputsOutputs(reduction_tv);

  // Vectorize as much as we can
  size_t vectorize_factor = std::numeric_limits<size_t>::max();

  for (auto tv : vectorizable_inputs_outputs) {
    const auto tv_vectorize_factor = runtime_info.getVectorizableWidth(tv);
    vectorize_factor = std::min(vectorize_factor, tv_vectorize_factor);
  }

  if (vectorize_factor == std::numeric_limits<size_t>::max()) {
    vectorize_factor = 1;
  }

  return reductionHeuristic(
      red_elements,
      num_outputs_for_reduction,
      fastest_dim_reduction,
      n_tensor_inputs,
      max_dtype_size,
      vectorize_factor);
}

// fusion is the input IR that will be modified by this function
void scheduleReduction(Fusion* fusion, const ReductionParams& rparams) {
  FUSER_PERF_SCOPE("scheduleReduction");
  FusionGuard fg(fusion);

  // Cache inputs if unrolled
  auto cached_inputs =
      scheduler_utils::cacheInputs(fusion, rparams.loop_unroll > 1);

  // Cache and fork  outputs
  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs =
      scheduler_utils::cacheAndForkOutputs(fusion, rparams.loop_unroll > 1);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  TORCH_INTERNAL_ASSERT(
      reduction_tvs.size() <= 1,
      "Found multiple reductions sent to reduction heuristics",
      " (and reductions are not from a multi-output expr).");
  TORCH_INTERNAL_ASSERT(reduction_tvs.size());

  auto reduction_tv = reduction_tvs[0];

  auto dim_analysis =
      scheduler_utils::canonicalDimReduction(fusion, reduction_tv);
  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;

  TORCH_INTERNAL_ASSERT(
      has_red_axis,
      "Could not find reduction axis in tensor used for reduction scheduler.");

  if (!has_iter_axis) {
    TORCH_INTERNAL_ASSERT(
        rparams.fastest_dim,
        "If all dims are reduction, should be sending it to fastest dim scheduler.");
  }

  TensorView* reference_tv = scheduler_utils::scheduleReductionTV(
      rparams, reduction_tv, has_iter_axis);

  // Reduction tensor views and rfactor tensor views are setup. Let's finish off
  // the scheduling, particularly inlining and unrolling.
  TORCH_INTERNAL_ASSERT(
      reference_tv != nullptr && reduction_tv != nullptr,
      "Need these two tensor views to finish the scheduling.");
  scheduler_utils::multiReductionInliner(
      fusion,
      rparams,
      reduction_tv,
      reference_tv,
      reduction_tvs,
      cached_inputs,
      cached_outputs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
