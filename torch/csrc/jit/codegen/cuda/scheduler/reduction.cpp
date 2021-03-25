#include <torch/csrc/jit/codegen/cuda/scheduler/reduction.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
constexpr int64_t x_grid_limit = ((int64_t)1 << (int64_t)31) - (int64_t)1;
constexpr int64_t y_grid_limit = 65535;
// Largest Power of 2 less-than n
constexpr int64_t lastPow2(int64_t n) {
  TORCH_INTERNAL_ASSERT(n >= 0);
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 16); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 32); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  return std::max((int64_t)1, n - (n >> 1));
}

ReductionParams innerReductionHeuristic(
    const int64_t num_elems_in_reduction,
    const int64_t num_outputs_for_reduction,
    const int64_t n_input_tensors,
    const int64_t max_input_dtype_size) {
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
      (int64_t)16 / max_input_dtype_size,
      // Reduce unrolling if we have many inputs, start reduction at 2 inputs
      std::max((lastPow2((int64_t)n_input_tensors) >> 1), (int64_t)1));

  // Conservative value, could be set to larger based on arch if necessary.
  constexpr int64_t l1_cache = 32 * 1024;
  // Could change per generation, but for l1 we want to consider active threads,
  // not resident
  constexpr int64_t active_threads = 1024;
  // Check how many elements it would take per thread to start thrashing l1
  // set that to minimum number we want to reduce per thread.
  int64_t min_red_elems_per_thread = std::max(
      l1_cache / (n_input_tensors * max_input_dtype_size * active_threads),
      (int64_t)1);

  // if data fits in l2 and we need more parallelization in the reduction dim,
  // we can use a smaller warp size. While thread local data fits in l1, and
  // reduction dim is really small, we can use <32 threads per warp.
  const bool fits_in_l2 = n_elems * max_input_dtype_size * n_input_tensors <
      at::cuda::getCurrentDeviceProperties()->l2CacheSize;

  // If it fits in l2, we just want to make sure each thread uses 32Bytes.
  const int64_t warp_size_based_on_l2 =
      fits_in_l2 ? (int64_t)32 / max_input_dtype_size : 32;

  const int64_t warp_size_based_on_l1 = std::min(
      ceilDiv(num_elems_in_reduction, min_red_elems_per_thread), (int64_t)32);

  // Take the smaller
  const int64_t warp_size =
      std::min(warp_size_based_on_l1, warp_size_based_on_l2);

  // Initialization
  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t max_threads_in_block = std::min(
      warp_size, ceilDiv(num_elems_in_reduction, min_red_elems_per_thread));

  // If we have one warp per block, how many blocks would that be?
  target_blocks = ceilDiv(n_elems, warp_size * min_red_elems_per_thread);

  // If we have more than a wave, put parallelism into unrolling
  if (target_blocks > device_multiprocessor_count) {
    target_unroll = std::min(
        max_unroll, ceilDiv(target_blocks, device_multiprocessor_count));
    target_blocks = ceilDiv(
        n_elems, warp_size * std::max(target_unroll, min_red_elems_per_thread));
  } else {
    // Steal reduction elements from threads if it helps us get a wave of blocks
    min_red_elems_per_thread = std::min(
        min_red_elems_per_thread,
        ceilDiv(
            num_elems_in_reduction * num_outputs_for_reduction,
            warp_size * device_multiprocessor_count));
  }

  // Cap target blocks to 4 waves
  target_blocks = std::min(target_blocks, device_multiprocessor_count * 4);

  if (target_blocks * target_unroll *
          std::max(target_unroll, min_red_elems_per_thread) <
      n_elems) {
    // targetting 4 waves, so try to use a quarter of available threads
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

  // TODO: Flip block y and x
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
    // Small number of reduction elements, don't try to unroll the reduction dim
    unroll_reduction = false;
    // Try unrolling output dimension
    unroll_factor = std::min(target_unroll, remainder_in_output);
    remainder_in_output =
        ceilDiv(num_outputs_for_reduction, unroll_factor * bdimy);
  } else {
    // If we have reduction elements left, re-adjust the block dims
    bdimx = std::min(
        ceilDiv(num_elems_in_reduction, min_red_elems_per_thread),
        max_threads_in_block);

    // Don't exceed target.
    bdimy = std::max(max_threads_in_block / bdimx, (int64_t)1);
    remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimy);

    remainder_in_reduction = ceilDiv(num_elems_in_reduction, bdimx);
    unroll_factor = std::min(remainder_in_reduction, target_unroll);
    if (unroll_factor == 1) {
      // If we can't unroll reduction dim, unroll output dim
      unroll_reduction = false;
      unroll_factor = std::min(remainder_in_output, target_unroll);
      remainder_in_output =
          ceilDiv(num_outputs_for_reduction, bdimy * unroll_factor);
      remainder_in_reduction =
          ceilDiv(num_elems_in_reduction, bdimx * min_red_elems_per_thread);
    } else {
      remainder_in_reduction = ceilDiv(
          num_elems_in_reduction,
          bdimx * std::max(unroll_factor, min_red_elems_per_thread));
    }
  }

  godim = remainder_in_output;

  // Clang tidy
  constexpr int64_t kEight = 8;
  constexpr int64_t kThirtyTwo = 32;

  // Cross grid reduction if we haven't hit our target blocks, and we have many
  // reduction elements.
  if (godim < target_blocks && remainder_in_reduction > kEight &&
      remainder_in_reduction < kThirtyTwo) {
    grdim = ceilDiv(remainder_in_reduction, (int64_t)4);
    // Clang tidy
    //
    // remainder_in_reduction = ceilDiv(
    //     num_elems_in_reduction,
    //     bdimx *
    //         std::max(
    //             unroll_reduction ? unroll_factor : 1,
    //             min_red_elems_per_thread) *
    //         grdim);
  } else if (remainder_in_reduction >= kThirtyTwo) {
    // Do at least 2 iterations of unrolling per thread before we go cross grid.
    // Limit cross grid to a multiple of the block size so cleanup on the last
    // block doesn't take too long.
    grdim = std::min(
        ceilDiv(remainder_in_reduction, (int64_t)2), bdimx * bdimy * kEight);
    // Clang tidy
    // remainder_in_reduction = ceilDiv(remainder_in_reduction, grdim);
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

  ReductionParams rparams;
  rparams.fastest_dim = true;
  rparams.cross_block = true;
  rparams.cross_grid = grdim > 1;
  rparams.multiple_reds_per_blk = bdimy > 1;
  rparams.loop_unroll = unroll_factor;
  rparams.reduction_unroll = unroll_reduction;

  // If we have a cross grid case we want to have gdimy assigned to godim and
  // gdimx assigned to grdim. Otherwise it's helpful to pull godim into gdimx in
  // case it's larger than gdimy can hold, as not doing so can thrash the cache.
  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;

  if (rparams.cross_grid) {
    gdimx = grdim;
    rparams.split_grid_dim = gdimy > y_grid_limit;
  } else {
    rparams.split_grid_dim = gdimx > x_grid_limit;
  }

  rparams.lparams = LaunchParams(
      gdimx,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cerr << rparams.toString() << std::endl;
  }

  return rparams;
}

ReductionParams OuterReductionHeuristic(
    const int64_t num_elems_in_reduction,
    const int64_t num_outputs_for_reduction,
    const int64_t n_input_tensors,
    const int64_t max_input_dtype_size) {
  // Set some targets for parallelization

  const int64_t n_elems = num_elems_in_reduction * num_outputs_for_reduction;
  const int64_t l2_cache_size =
      at::cuda::getCurrentDeviceProperties()->l2CacheSize;

  const int64_t warp_size =
      n_elems * max_input_dtype_size * n_input_tensors < l2_cache_size
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
      // Reduce unrolling if we have many inputs, start reduction at 2 inputs
      std::max((lastPow2((int64_t)n_input_tensors) >> 1), (int64_t)1));

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
    // targetting 4 waves, so try to use a quarter of available threads
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
    // bdimy
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

  ReductionParams rparams;
  rparams.fastest_dim = false;
  // cross grid implies cross block
  rparams.cross_block = bdimy > 1 || gdimy > 1;
  rparams.cross_grid = gdimy > 1;
  rparams.multiple_reds_per_blk = bdimx > 1;
  rparams.loop_unroll = unroll_factor;
  rparams.reduction_unroll = unroll_reduction;

  // WAR as it seems nvcc is doing some strange unrolling behavior in
  // this scenario for fp16 small reduction dim large iter dim. Needs more
  // investigation.
  if (!rparams.cross_block && !rparams.cross_grid) {
    rparams.loop_unroll = 1;
    rparams.reduction_unroll = true;
  }

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cerr << rparams.toString() << std::endl;
  }

  rparams.lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);

  return rparams;
}

} // namespace

ReductionParams reductionHeuristic(
    int64_t num_elems_in_reduction,
    int64_t num_outputs_for_reduction,
    bool fastest_dim_reduction,
    size_t n_input_tensors,
    size_t max_input_dtype_size) {
  if (fastest_dim_reduction) {
    return innerReductionHeuristic(
        num_elems_in_reduction,
        num_outputs_for_reduction,
        n_input_tensors,
        max_input_dtype_size);
  } else {
    return OuterReductionHeuristic(
        num_elems_in_reduction,
        num_outputs_for_reduction,
        n_input_tensors,
        max_input_dtype_size);
  }
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    TensorView* red_tv) {
  FUSER_PERF_SCOPE("getReductionHeuristics");

  auto evaluator = executor_utils::bindFusionInputs(fusion_inputs, fusion);

  return getReductionHeuristics(fusion, evaluator, red_tv);
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    ExpressionEvaluator& evaluator,
    TensorView* red_tv) {
  FUSER_PERF_SCOPE("getReductionHeuristics");

  FusionGuard fg(fusion);

  auto red_root_dom = red_tv->getRootDomain();
  bool fastest_dim_reduction = true;
  for (size_t i = red_root_dom.size(); i > 0; i--) {
    if (red_root_dom[i - 1]->isBroadcast()) {
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
      red_tv != nullptr, "Reduction TensorView wasn't found.");

  TORCH_INTERNAL_ASSERT(
      red_tv->hasReduction(), "TensorView doesn't have a reduction.");
  const auto red_expr = red_tv->definition();

  TORCH_INTERNAL_ASSERT(
      red_expr->getExprType() != c10::nullopt &&
          (red_expr->getExprType().value() == ExprType::ReductionOp ||
           red_expr->getExprType().value() == ExprType::WelfordOp),
      "TensorView doesn't have a reduction.");

  int64_t num_outputs_for_reduction = 1;
  int64_t red_elements = 1;

  for (auto id : red_tv->getRootDomain()) {
    auto inferred_val = evaluator.evaluate(id->rawExtent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(), "Error inferring reduction size.");
    if (id->isReduction()) {
      red_elements *= inferred_val.value();
    } else {
      num_outputs_for_reduction *= inferred_val.value();
    }
  }

  size_t max_dtype_size = 1;
  size_t n_input_tensors = 0;
  for (auto inp : fusion->inputs()) {
    if (inp->isA<TensorView>()) {
      max_dtype_size =
          std::max(max_dtype_size, dataTypeSize(inp->getDataType().value()));
      n_input_tensors++;
    }
  }

  TORCH_INTERNAL_ASSERT(
      n_input_tensors > 0,
      "Tried to schedule a fusion with no tensor inputs, currently not supported.");

  return reductionHeuristic(
      red_elements,
      num_outputs_for_reduction,
      fastest_dim_reduction,
      n_input_tensors,
      max_dtype_size);
}

// fusion is the input IR that will be modified by this function
void scheduleReduction(
    Fusion* fusion,
    const ReductionParams& rparams,
    TensorView* red_tv,
    const std::vector<TensorView*>& outs_of_red) {
  FUSER_PERF_SCOPE("scheduleReduction");
  FusionGuard fg(fusion);
  constexpr int kLoopUnrollSplit = 4;

  // If either of these are nullptr at the end of this function don't do
  // anything. Otherwise Transform and parallize entire fusion based on
  // reference_tv and compute at most inlined from reduction_tv to inputs and
  // outputs.
  TensorView* reference_tv = nullptr;
  TensorView* reduction_tv = nullptr;

  // We coalesce all reduction axes to the right;
  scheduler_utils::mergeReduction(red_tv);

  // Merge all iteration dimensions
  if (red_tv->domain()->domain().size() > 1) {
    scheduler_utils::mergeNonReduction(red_tv);
  }

  // Evaluate Dimensions of Reduction TensorView
  auto red_ids = red_tv->domain()->domain();

  TORCH_INTERNAL_ASSERT(
      red_ids.size() == 1 || red_ids.size() == 2,
      "Error coalesing dimensions.");

  if (red_ids.size() == 1) {
    TORCH_INTERNAL_ASSERT(
        rparams.fastest_dim,
        "If all dims are reduction, should be sending it to fastest dim scheduler.");
  }

  std::vector<TensorView*> cached_inputs;
  // If we're going to unroll, make a cache of the inputs
  if (rparams.loop_unroll > 1) {
    auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    for (auto tv : in_tvs) {
      auto cached_tv = tv->cache_after();
      cached_inputs.emplace_back(cached_tv);
    }
  }

  // Scheduling the Reduction
  if (rparams.fastest_dim) {
    const bool has_iter_axis = red_ids.size() == 2;
    const int iter_axis = 0;
    const int reduce_axis = red_ids.size() == 2 ? 1 : 0;

    // Do multiple reductions per block
    if (rparams.multiple_reds_per_blk) {
      if (rparams.reduction_unroll) {
        // Fastest dim, multiple reductions per block
        // Output Dimensions
        // [x-BIDx, x-TIDy
        //  0       1
        //
        //  Reduction Dimensions
        //  rF-Remain, rf-Unswitch, rf-Unroll, X-TIDx]
        //  2 (-4)     3 (-3)       4 (-2)     5 (-1)

        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
        red_tv->split(reduce_axis, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        red_tv->split(reduce_axis, 1);

        auto red_tv_rf = scheduler_utils::rfactorHelper(red_tv, {-4, -3, -2});

        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(-3)->parallelize(ParallelType::Unswitch);

        if (has_iter_axis) {
          red_tv_rf->split(
              iter_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
          red_tv_rf->axis(iter_axis + 1)->parallelize(ParallelType::TIDy);
          if (rparams.split_grid_dim) {
            red_tv_rf->split(iter_axis, x_grid_limit);
            red_tv_rf->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
          } else {
            red_tv_rf->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }
        }
        reference_tv = red_tv_rf;
        reduction_tv = red_tv;
      } else {
        TORCH_INTERNAL_ASSERT(
            has_iter_axis,
            "This scheduler requires an outer dim to the reduction.");
        // Fastest dim, Multiple reductions per block iter unroll
        // Output Dimensions
        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDy
        //  0       1           2         3
        //
        //  Reduction Dimensions
        //  rF-Remain, r-TIDx]
        //  4 (-2)     5 (-1)
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));

        auto red_tv_rf = scheduler_utils::rfactorHelper(red_tv, {-2});
        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);

        if (has_iter_axis) {
          red_tv_rf->split(
              iter_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
          red_tv_rf->split(iter_axis, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          red_tv_rf->split(iter_axis, 1);

          red_tv_rf->axis(3)->parallelize(ParallelType::TIDy);
          // TODO: Re-enable unswitch in this case:
          // https://github.com/csarofeen/pytorch/issues/748
          // red_tv_rf->axis(1)->parallelize(ParallelType::Unswitch);

          // [BIDx, 1, 8, TIDy, rf-outer, r-TIDx]

          if (rparams.split_grid_dim) {
            red_tv_rf->split(iter_axis, x_grid_limit);
            red_tv_rf->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
          } else {
            red_tv_rf->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }

          reference_tv = red_tv_rf;
          reduction_tv = red_tv;
        }
      }
    } else {
      if (rparams.cross_grid) {
        // Fastest dim, cross grid, cross block
        //      [outputs,
        // Idx:     0
        //   | rf-Remain, r-BIDx, r-TIDy, r-Unswitch, rf-Unroll, r-TIDx]
        //     1(-6)      2(-5)   3(-4)   4(-3)       5(-2)      6(-1)|
        //                Reduction Dimensions
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
        red_tv->split(reduce_axis, rparams.loop_unroll);
        red_tv->split(reduce_axis, 1);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::BIDx));

        // Clang tidy
        constexpr int kNegFive = -5;
        constexpr int kNegSix = -6;
        auto red_tv_rf =
            scheduler_utils::rfactorHelper(red_tv, {kNegSix, -3, -2});

        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(-3)->parallelize(ParallelType::Unswitch);
        red_tv_rf->axis(-4)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(kNegFive)->parallelize(ParallelType::BIDx);

        if (has_iter_axis) {
          if (rparams.split_grid_dim) {
            red_tv_rf->split(iter_axis, y_grid_limit);
            red_tv_rf->axis(iter_axis + 1)->parallelize(ParallelType::BIDy);
          } else {
            red_tv_rf->axis(iter_axis)->parallelize(ParallelType::BIDy);
          }
        }

        reference_tv = red_tv_rf;
        reduction_tv = red_tv;

      } else {
        // Fastest dim, Reduction Splits
        // Output Dimensions
        // [BIDx
        //  0
        //
        // Reduction Dimensions
        // rF-Remain, rf-Unswitch, rf-Unroll, r-TIDx]
        // 1(-4)      2(-3)        3(-2)      4(-1)
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
        red_tv->split(reduce_axis, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        red_tv->split(reduce_axis, 1);

        auto red_tv_rf = scheduler_utils::rfactorHelper(red_tv, {-4, -3, -2});

        red_tv_rf->axis(-1)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(-3)->parallelize(ParallelType::Unswitch);

        if (has_iter_axis) {
          if (rparams.split_grid_dim) {
            red_tv_rf->split(iter_axis, x_grid_limit);
            red_tv_rf->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
          } else {
            red_tv_rf->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }
        }

        reference_tv = red_tv_rf;
        reduction_tv = red_tv;
      }
    }
  } else {
    if (rparams.cross_block) {
      if (rparams.cross_grid) {
        // Outer Dim, cross grid, cross block

        // Unrolling in this case can only be applied to the reduction dimension
        // since currently, grid reductions cannot be called multiple times
        //
        // Output Dimensions
        // [x-BIDx, x-TIDx,
        //  0         1
        //
        // Reduction Dimensions
        // rF-Leftover, r-BIDy, r-TIDy, rf-Unswitch, rf-Unroll]
        // 2(-5)        3(-4)   4(-3)   5(-2)        6(-1)
        red_tv->split(1, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        red_tv->split(1, 1);
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::BIDy));

        red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));

        auto red_tv_rf = scheduler_utils::rfactorHelper(
            red_tv,
            {-5, -2, -1}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

        red_tv_rf->axis(-2)->parallelize(ParallelType::Unswitch);
        red_tv_rf->axis(-3)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(-4)->parallelize(ParallelType::BIDy);
        red_tv_rf->axis(1)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(0)->parallelize(ParallelType::BIDx);

        reference_tv = red_tv_rf;
        reduction_tv = red_tv;

      } else {
        if (rparams.reduction_unroll || rparams.loop_unroll == 1) {
          // Outer Dim, cross block, unroll reduction dimension

          // Reduction Splits
          // Output Dimensions
          // [x-BIDx, x-TIDx
          //  0       1
          //
          // Reduction Dimensions
          // rF-Leftover, r-TIDy, rf-Unswitch, rf-Unroll]
          // 2(-4)        3(-3)   4(-2)       5(-1)
          red_tv->split(1, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          red_tv->split(1, 1);
          red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
          red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));

          auto red_tv_rf = scheduler_utils::rfactorHelper(
              red_tv,
              {-4, -2, -1}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

          red_tv_rf->axis(-2)->parallelize(ParallelType::Unswitch);
          red_tv_rf->axis(-3)->parallelize(ParallelType::TIDy);
          red_tv_rf->axis(1)->parallelize(ParallelType::TIDx);
          red_tv_rf->axis(0)->parallelize(ParallelType::BIDx);

          reference_tv = red_tv_rf;
          reduction_tv = red_tv;

        } else {
          // Outer Dim, cross block, unroll iter dimension

          // Output Dimensions
          // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx
          //  0       1           2         3
          //
          // Reduction Dimensions
          // rF-Leftover, r-TIDy]
          // 4(-2)        5(-1)

          red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
          red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
          red_tv->split(0, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          red_tv->split(0, 1);

          auto red_tv_rf = scheduler_utils::rfactorHelper(
              red_tv, {-2}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

          red_tv_rf->axis(-1)->parallelize(ParallelType::TIDy);
          red_tv_rf->axis(3)->parallelize(ParallelType::TIDx);
          red_tv_rf->axis(1)->parallelize(ParallelType::Unswitch);
          red_tv_rf->axis(0)->parallelize(ParallelType::BIDx);

          red_tv_rf->reorder({{-2, 0}});

          reference_tv = red_tv_rf;
          reduction_tv = red_tv;
        }
      }
    } else {
      if (rparams.reduction_unroll) {
        // Outer Dim, no parallelization on reduction, unroll reduction axis
        // Output Dimensions
        // [x-BIDx, x-TIDx
        //  0       1
        //
        // Reduction Dimensions
        // rf-Leftover, rf-Unswitch, r-Unroll]
        // 2(-3)        3(-2)        4(-1)
        red_tv->split(1, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        red_tv->split(1, 1);
        red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));

        auto red_tv_rf = scheduler_utils::rfactorHelper(red_tv, {-3, -2});

        red_tv_rf->axis(0)->parallelize(ParallelType::BIDx);
        red_tv_rf->axis(1)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(-2)->parallelize(ParallelType::Unswitch);

        reference_tv = red_tv_rf;
        reduction_tv = red_tv;
      } else {
        // No parallelization on reduction, unroll iter axis
        // Output Dimensions
        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx
        //  0       1           2         3
        //
        // Reduction Dimensions
        // r-Leftover]
        // 4(-1)
        red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
        red_tv->split(0, rparams.loop_unroll);
        red_tv->split(0, 1);

        red_tv->axis(0)->parallelize(ParallelType::BIDx);
        red_tv->axis(1)->parallelize(ParallelType::Unswitch);
        red_tv->axis(3)->parallelize(ParallelType::TIDx);
        red_tv->reorder({{-1, 0}});

        reference_tv = red_tv;
        reduction_tv = red_tv;
      }
    }
  }

  // Reduction tensor views and rfactor tensor views are setup. Let's finish off
  // the scheduling, particularly inlining and unrolling.
  TORCH_INTERNAL_ASSERT(
      reference_tv != nullptr && reduction_tv != nullptr,
      "Need these two tensor views to finish the scheduling.");

  if (rparams.loop_unroll > 1) {
    // Schedule unrolling on inputs

    TransformPropagator::from(reference_tv);

    // Inline rfactor into reduction
    if (reference_tv != reduction_tv) {
      reference_tv->computeWith(reduction_tv, -1, ComputeAtMode::BestEffort);
    }

    // Find unswitch position
    int unswitch_axis = -1;
    for (int i = 0; i < (int)reference_tv->nDims(); i++) {
      if (reference_tv->axis(i)->getParallelType() == ParallelType::Unswitch) {
        unswitch_axis = i;
      }
    }

    unswitch_axis++;
    // Input to cahced_input we want outside unswitched position
    // Cached input to rfactor we want inlined
    for (auto cached_input : cached_inputs) {
      auto consumers_of_input_cache =
          scheduler_utils::consumerTvsOf(cached_input);
      for (auto consumer : consumers_of_input_cache) {
        if (consumer != reference_tv) {
          // consumer->computeAt(reference_tv, -1, ComputeAtMode::MostInlined);
          scheduler_utils::computeWithOutputs(
              consumer, -1, ComputeAtMode::MostInlined);
        }
        // TODO: Re-evaluate this based on SegmentReducePointwise, and other
        // more complex reduction fusions
        cached_input->computeAt(
            consumer, unswitch_axis, ComputeAtMode::BestEffort);
      }
    }

    scheduler_utils::computeWithOutputs(
        reduction_tv, -1, ComputeAtMode::MostInlined);

    scheduler_utils::parallelizeAllLike(
        reference_tv, scheduler_utils::allTvs(fusion));

    // Nasty gotcha which we don't have a better mechanism to fix yet
    if (
        // Have an unswitch in the reduction
        std::any_of(
            reduction_tv->domain()->domain().begin(),
            reduction_tv->domain()->domain().end(),
            [](IterDomain* id) {
              return id->getParallelType() == ParallelType::Unswitch;
            }) &&
        // Have a parallelized reduction
        std::any_of(
            reduction_tv->domain()->domain().begin(),
            reduction_tv->domain()->domain().end(),
            [](IterDomain* id) {
              return id->isReduction() && id->isThread();
            })) {
      // If we leave unswitch on we could get a predicate around block/grid
      // reduce which produces wrong result.
      auto vals_post_reduction = DependencyCheck::getAllUseChains(red_tv);
      for (const auto& chain : vals_post_reduction) {
        auto tvs_post_reduction = ir_utils::filterByType<TensorView>(chain);
        for (auto tv : tvs_post_reduction) {
          for (auto id : tv->domain()->domain()) {
            if (id->getParallelType() == ParallelType::Unswitch) {
              id->parallelize(ParallelType::Serial);
            }
          }
        }
      }
    }
  } else {
    // Inline and parallelize
    TransformPropagator::from(reference_tv);
    // Want to inline, especially backwards based on reduction_tv, otherwise
    // rfactor tv may not be inlined correctly
    scheduler_utils::computeAtInputs(
        reduction_tv, -1, ComputeAtMode::MostInlined);
    scheduler_utils::computeWithOutputs(
        reduction_tv, -1, ComputeAtMode::MostInlined);
    scheduler_utils::parallelizeAllLike(
        reference_tv, scheduler_utils::allTvs(fusion));
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
