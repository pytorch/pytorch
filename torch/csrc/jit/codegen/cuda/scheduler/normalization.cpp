#include <torch/csrc/jit/codegen/cuda/scheduler/reduction.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO: Fork outputs

namespace {
constexpr int64_t x_grid_limit = ((int64_t)1 << (int64_t)31) - (int64_t)1;
// constexpr int64_t y_grid_limit = 65535; // unused at this time
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

// Copied from reduction scheduler, should generalize. Simply needed to take out
// grid reductions.
ReductionParams innerNormalizationHeuristic(
    const int64_t num_elems_in_reduction,
    const int64_t num_outputs_for_reduction,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size,
    bool persistence_required) {
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
      // Reduce unrolling if we have many inputs, start reduction at 2 inputs
      std::max((lastPow2((int64_t)n_tensor_inputs) >> 1), (int64_t)1));

  // Conservative value, could be set to larger based on arch if necessary.
  constexpr int64_t l1_cache = 32 * 1024;
  // Could change per generation, but for l1 we want to consider active threads,
  // not resident
  constexpr int64_t active_threads = 1024;
  // Check how many elements it would take per thread to start thrashing l1
  // set that to minimum number we want to reduce per thread.
  int64_t min_red_elems_per_thread = std::max(
      l1_cache / (n_tensor_inputs * max_input_dtype_size * active_threads),
      (int64_t)1);

  // if data fits in l2 and we need more parallelization in the reduction dim,
  // we can use a smaller warp size. While thread local data fits in l1, and
  // reduction dim is really small, we can use <32 threads per warp.
  const bool fits_in_l2 = n_elems * max_input_dtype_size * n_tensor_inputs <
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
      // remainder_in_reduction =
      //     ceilDiv(num_elems_in_reduction, bdimx * min_red_elems_per_thread);
      // Leave this commented for clang, still think it's important to have
      // though
    }
    //  else {
    // remainder_in_reduction = ceilDiv(
    //     num_elems_in_reduction,
    //     bdimx * std::max(unroll_factor, min_red_elems_per_thread));
    // Leave this commented for clang, still think it's important to have though
    // }
  }

  godim = remainder_in_output;

  // Persistence size from buffers
  int64_t batches_per_block = ceilDiv(
      num_elems_in_reduction,
      bdimx * (unroll_reduction ? unroll_factor : (int64_t)1));
  // round up to multiple of 8 or pow2 whichever smaller
  auto round_up_pow2 = lastPow2(batches_per_block);
  if (round_up_pow2 < batches_per_block) {
    round_up_pow2 *= 2;
  }

  constexpr int64_t kEight = 8; // clang tidy

  auto round_up_8 = batches_per_block % kEight == 0
      ? batches_per_block
      : batches_per_block + (kEight - batches_per_block % kEight);

  batches_per_block = std::min(round_up_8, round_up_pow2);

  ReductionParams rparams;
  rparams.fastest_dim = true;
  rparams.cross_block = true;
  rparams.cross_grid = false;
  rparams.multiple_reds_per_blk = bdimy > 1;
  rparams.loop_unroll = unroll_factor;
  rparams.reduction_unroll = unroll_reduction;
  rparams.batches_per_block = batches_per_block;
  rparams.persistent_kernel = persistence_required;

  // If we have a cross grid case we want to have gdimy assigned to godim and
  // gdimx assigned to grdim. Otherwise it's helpful to pull godim into gdimx in
  // case it's larger than gdimy can hold, as not doing so can thrash the cache.

  rparams.split_grid_dim = godim > x_grid_limit;

  rparams.lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      persistence_required ? LaunchParams::UNINITIALIZED_VAL : bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);

  rparams.tag = persistence_required ? "Inner normalization heuristic.\n"
                                     : "Multi inner reduction (norm heuristic)";

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cerr << rparams.toString() << std::endl;
  }

  return rparams;
}

// Copied from reduction scheduler, should generalize. Simply needed to take out
// grid reductions.
ReductionParams OuterNormalizationHeuristic(
    const int64_t num_elems_in_reduction,
    const int64_t num_outputs_for_reduction,
    const int64_t n_tensor_inputs,
    const int64_t max_input_dtype_size,
    bool persistence_required) {
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
      // Reduce unrolling if we have many inputs, start reduction at 2 inputs
      std::max((lastPow2((int64_t)n_tensor_inputs) >> 1), (int64_t)1));

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

  // Blocks for outputs
  // int64_t gdimx = 1; // unused at this time, comment for clang tidy

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
    // If we can't hit a full wave, reduce the warp_size to increase
    // the number of blocks.  The warp should be reduced at a minimum
    // to the granularity that an SM would pull a unique portion of a
    // cacheline from the memory system or else there is no
    // benefit from spreading the work to a different block.
    // This is dependent on the data size of elements.
    const int64_t cache_sector_bytes = 32;
    int64_t min_outputs_per_block =
        std::max(cache_sector_bytes / max_input_dtype_size, (int64_t)1);
    bdimx =
        std::min(
            std::max(
                ceilDiv(
                    num_outputs_for_reduction, device_multiprocessor_count) /
                    min_outputs_per_block,
                (int64_t)1),
            (int64_t)1) *
        min_outputs_per_block;
  } else {
    bdimx = std::min(
        max_threads_in_block,
        ceilDiv(num_outputs_for_reduction, target_blocks));
    bdimx = std::max(bdimx, warp_size);
  }

  bdimy = std::min(
      std::max(max_threads_in_block / bdimx, (int64_t)1),
      num_elems_in_reduction);

  // remainder_in_output = ceilDiv(num_outputs_for_reduction, bdimx);
  // unused, but only commenting for clang-tidy
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
      // remainder_in_reduction = ceilDiv(remainder_in_reduction,
      // unroll_factor); Unused, comment for clang tidy.
      if (unroll_factor > 1) {
        unroll_reduction = true;
      }
    }
    //  else {
    // remainder_in_output =
    //     ceilDiv(num_outputs_for_reduction, bdimx * unroll_factor);
    // unused, comment for clang tidy
    // }
  } else {
    // Not many output elements, try unrolling reduction dimension
    unroll_factor = std::min(max_unroll, remainder_in_reduction);
    if (unroll_factor > 1) {
      unroll_reduction = true;
    }
  }

  // Persistence size from buffers
  int64_t batches_per_block = 1;
  if (persistence_required) {
    batches_per_block = ceilDiv(
        num_elems_in_reduction,
        bdimy * (unroll_reduction ? unroll_factor : (int64_t)1));
    // round up to multiple of 8 or pow2 whichever smaller
  }

  auto round_up_pow2 = lastPow2(batches_per_block);
  if (round_up_pow2 < batches_per_block) {
    round_up_pow2 *= 2;
  }

  constexpr int64_t kEight = 8; // clang tidy

  auto round_up_8 = batches_per_block % kEight == 0
      ? batches_per_block
      : batches_per_block + (kEight - batches_per_block % kEight);

  batches_per_block = std::min(round_up_8, round_up_pow2);

  ReductionParams rparams;
  rparams.fastest_dim = false;
  rparams.cross_block = true;
  rparams.cross_grid = false;
  rparams.multiple_reds_per_blk = bdimx > 1;
  rparams.loop_unroll = unroll_factor;
  rparams.reduction_unroll = unroll_reduction;
  rparams.batches_per_block = batches_per_block;
  rparams.persistent_kernel = persistence_required;

  // WAR as it seems nvcc is doing some strange unrolling behavior in
  // this scenario for fp16 small reduction dim large iter dim. Needs more
  // investigation.
  if (!rparams.cross_block) {
    rparams.loop_unroll = 1;
    rparams.reduction_unroll = true;
  }

  rparams.lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      persistence_required ? LaunchParams::UNINITIALIZED_VAL : bdimy,
      LaunchParams::UNINITIALIZED_VAL);

  rparams.tag = persistence_required ? "Outer normalization heuristic.\n"
                                     : "Multi outer reduction (norm heuristic)";

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cerr << rparams.toString() << std::endl;
  }

  return rparams;
}

} // namespace

ReductionParams NormalizationHeuristic(
    int64_t num_elems_in_reduction,
    int64_t num_outputs_for_reduction,
    bool fastest_dim_reduction,
    size_t n_tensor_inputs,
    size_t max_input_dtype_size,
    bool persistence_required) {
  if (fastest_dim_reduction) {
    return innerNormalizationHeuristic(
        num_elems_in_reduction,
        num_outputs_for_reduction,
        n_tensor_inputs,
        max_input_dtype_size,
        persistence_required);
  } else {
    return OuterNormalizationHeuristic(
        num_elems_in_reduction,
        num_outputs_for_reduction,
        n_tensor_inputs,
        max_input_dtype_size,
        persistence_required);
  }
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getNormalizationHeuristics(
    Fusion* fusion,
    ExpressionEvaluator& evaluator) {
  FUSER_PERF_SCOPE("getNormalizationHeuristics");

  FusionGuard fg(fusion);

  std::vector<TensorView*> reduction_tvs;
  for (auto tv : scheduler_utils::allTvs(fusion)) {
    if (tv->hasReduction() && !fusion->hasInput(tv)) {
      reduction_tvs.push_back(tv);
    }
  }

  TORCH_INTERNAL_ASSERT(
      !reduction_tvs.empty(), "Need reduction tensor views to schedule.");

  auto first_red_tv = reduction_tvs[0];

  TORCH_INTERNAL_ASSERT(
      first_red_tv != nullptr, "Reduction TensorView wasn't found.");

  TORCH_INTERNAL_ASSERT(
      first_red_tv->hasReduction(), "TensorView doesn't have a reduction.");
  const auto red_expr = first_red_tv->definition();

  TORCH_INTERNAL_ASSERT(
      red_expr->getExprType() != c10::nullopt &&
          (red_expr->getExprType().value() == ExprType::ReductionOp ||
           red_expr->getExprType().value() == ExprType::WelfordOp),
      "TensorView doesn't have a reduction.");

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

  auto persistent_buffers = scheduler_utils::persistentBuffers(fusion);
  bool requires_persistence = !persistent_buffers.buffers.empty();

  auto properties =
      scheduler_utils::getProperties(fusion, evaluator, first_red_tv);

  return NormalizationHeuristic(
      properties.reduction_numel,
      properties.iteration_numel,
      properties.fastest_dim_reduction,
      n_tensor_inputs,
      max_dtype_size,
      requires_persistence);
}

TORCH_CUDA_CU_API c10::optional<ReductionParams> getNormalizationHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs) {
  FUSER_PERF_SCOPE("getNormalizationHeuristics");

  auto evaluator = executor_utils::bindFusionInputs(fusion_inputs, fusion);

  return getNormalizationHeuristics(fusion, evaluator);
}
namespace {

void schedulePersistentNormalization(
    Fusion* fusion,
    const ReductionParams& rparams) {
  FUSER_PERF_SCOPE("schedulePersistentNormalization");

  FusionGuard fg(fusion);

  std::vector<TensorView*> reduction_tvs;
  for (auto tv : scheduler_utils::allTvs(fusion)) {
    if (tv->hasReduction() && !fusion->hasInput(tv)) {
      if (auto welford_op = dynamic_cast<WelfordOp*>(tv->definition())) {
        if (tv == welford_op->out()) {
          reduction_tvs.push_back(tv);
        }
      } else {
        reduction_tvs.push_back(tv);
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      !reduction_tvs.empty(), "Need reduction tensor views to schedule.");

  auto reduction_tv = reduction_tvs[0];
  TensorView* rfactor_tv = nullptr;

  scheduler_utils::mergeReduction(reduction_tv);

  // Merge all iteration dimensions
  if (reduction_tv->nDims() > 1) {
    scheduler_utils::mergeNonReduction(reduction_tv);
  }

  // Evaluate Dimensions of Reduction TensorView
  TORCH_INTERNAL_ASSERT(
      reduction_tv->nDims() == 1 || reduction_tv->nDims() == 2,
      "Error coalesing dimensions.");

  if (reduction_tv->domain()->domain().size() == 1) {
    TORCH_INTERNAL_ASSERT(
        rparams.fastest_dim,
        "If all dims are reduction, should be sending it to fastest dim scheduler.");
  }

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  for (auto tv : scheduler_utils::allTvs(fusion)) {
    if (tv->isFusionInput() || tv->isFusionOutput()) {
      tv->setMemoryType(MemoryType::Global);
    } else {
      tv->setMemoryType(MemoryType::Local);
    }
  }

  std::vector<TensorView*> cached_inputs;

  if (rparams.loop_unroll > 1) {
    auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    for (auto tv : in_tvs) {
      auto cached_tv = tv->cache_after();
      cached_inputs.emplace_back(cached_tv);
    }
  } else {
    auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    for (auto tv : in_tvs) {
      if (tv->uses().size() > 1) {
        auto cached_tv = tv->cache_after();
        cached_inputs.emplace_back(cached_tv);
      }
    }
  }

  std::vector<int> rfactor_axes;

  // Scheduling the Reduction
  if (rparams.fastest_dim) {
    const bool has_iter_axis = reduction_tv->nDims() == 2;
    const int iter_axis = 0;
    const int reduce_axis = reduction_tv->nDims() == 2 ? 1 : 0;

    // Do multiple reductions per block
    if (rparams.multiple_reds_per_blk) {
      if (rparams.reduction_unroll) {
        // Fastest dim, multiple reductions per block
        // Output Dimensions
        // [x-BIDx, x-TIDy
        //  0       1
        //
        //  Reduction Dimensions
        //  rF-persistent, rf-Unswitch, rf-Unroll, X-TIDx]
        //  2 (-4)     3 (-3)       4 (-2)     5 (-1)

        //  X-TIDx, rF-persistent, rf-Unswitch, rf-Unroll]
        //  2 (-4)  3 (-3)         4 (-2)       5 (-1)
        reduction_tv->split(
            reduce_axis,
            rparams.batches_per_block * rparams.loop_unroll,
            false);
        reduction_tv->split(reduce_axis, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reduction_tv->split(reduce_axis, 1);
        reduction_tv->reorder({{-1, -4}, {-4, -3}, {-3, -2}, {-2, -1}});
        rfactor_axes = {-3, -2, -1};
        rfactor_tv = scheduler_utils::rfactorHelper(reduction_tv, rfactor_axes);

        rfactor_tv->axis(-4)->parallelize(ParallelType::TIDx);
        rfactor_tv->axis(-3)->parallelize(ParallelType::Unswitch);

        if (has_iter_axis) {
          rfactor_tv->split(
              iter_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
          rfactor_tv->axis(iter_axis + 1)->parallelize(ParallelType::TIDy);
          if (rparams.split_grid_dim) {
            rfactor_tv->split(iter_axis, x_grid_limit);
            rfactor_tv->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
          } else {
            rfactor_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }
        }
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
        //  rF-persistent, r-TIDx]
        //  4 (-2)     5 (-1)

        reduction_tv->split(reduce_axis, rparams.batches_per_block, false);

        rfactor_axes = {-2};
        rfactor_tv = scheduler_utils::rfactorHelper(reduction_tv, rfactor_axes);

        rfactor_tv->axis(-1)->parallelize(ParallelType::TIDx);

        if (has_iter_axis) {
          rfactor_tv->split(
              iter_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
          rfactor_tv->split(iter_axis, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          rfactor_tv->split(iter_axis, 1);

          rfactor_tv->axis(3)->parallelize(ParallelType::TIDy);
          // TODO: Re-enable unswitch in this case:
          // https://github.com/csarofeen/pytorch/issues/748
          // rfactor_tv->axis(1)->parallelize(ParallelType::Unswitch);

          // [BIDx, 1, 8, TIDy, rf-outer, r-TIDx]

          if (rparams.split_grid_dim) {
            rfactor_tv->split(iter_axis, x_grid_limit);
            rfactor_tv->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
          } else {
            rfactor_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }
        }
      }
    } else {
      // Fastest dim, Reduction Splits
      // Output Dimensions
      // [BIDx
      //  0
      //
      //  Reduction Dimensions
      //  rF-persistent, rf-Unswitch, rf-Unroll, X-TIDx]
      //  1 (-4)         2 (-3)       3 (-2)     4 (-1)

      //  X-TIDx, rF-persistent, rf-Unswitch, rf-Unroll]
      //  1 (-4)  2 (-3)         3 (-2)       4 (-1)

      reduction_tv->split(
          reduce_axis, rparams.batches_per_block * rparams.loop_unroll, false);
      reduction_tv->split(reduce_axis, rparams.loop_unroll);
      // Unswitch axis which gives us finer control on allocations with
      // unrolling
      reduction_tv->split(reduce_axis, 1);

      reduction_tv->reorder({{-1, -4}, {-4, -3}, {-3, -2}, {-2, -1}});

      rfactor_axes = {-3, -2, -1};
      rfactor_tv = scheduler_utils::rfactorHelper(reduction_tv, rfactor_axes);

      rfactor_tv->axis(-4)->parallelize(ParallelType::TIDx);
      rfactor_tv->axis(-2)->parallelize(ParallelType::Unswitch);

      if (has_iter_axis) {
        if (rparams.split_grid_dim) {
          rfactor_tv->split(iter_axis, x_grid_limit);
          rfactor_tv->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
        } else {
          rfactor_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
        }
      }
    }
  } else {
    if (rparams.cross_block) {
      if (rparams.reduction_unroll || rparams.loop_unroll == 1) {
        // Outer Dim, cross block, unroll reduction dimension

        // Reduction Splits
        // Output Dimensions
        // [x-BIDx, x-TIDx
        //  0       1
        //
        // Reduction Dimensions
        // rF-Persistent, r-TIDy, rf-Unswitch, rf-Unroll]
        // 2(-4)        3(-3)   4(-2)       5(-1)
        reduction_tv->split(-1, rparams.batches_per_block, false);
        reduction_tv->split(-1, rparams.loop_unroll);
        reduction_tv->split(-2, 1);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reduction_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
        rfactor_axes = {-4, -2, -1};
        rfactor_tv = scheduler_utils::rfactorHelper(reduction_tv, rfactor_axes);

        rfactor_tv->axis(-2)->parallelize(ParallelType::Unswitch);
        rfactor_tv->axis(-3)->parallelize(ParallelType::TIDy);
        rfactor_tv->axis(1)->parallelize(ParallelType::TIDx);
        rfactor_tv->axis(0)->parallelize(ParallelType::BIDx);
      } else {
        // Outer Dim, cross block, unroll iter dimension

        // Output Dimensions
        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx
        //  0       1           2         3
        //
        // Reduction Dimensions
        // rF-Leftover, r-TIDy]
        // 4(-2)        5(-1)

        reduction_tv->split(-1, rparams.batches_per_block, false);
        reduction_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
        reduction_tv->split(0, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reduction_tv->split(0, 1);
        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx, rF-Leftover, r-TIDy]
        reduction_tv->reorder({{-2, 0}});
        // [rF-Leftover, x-BIDx, x-Unswitch, x-Unroll, x-TIDx, r-TIDy]
        rfactor_axes = {0};
        rfactor_tv = scheduler_utils::rfactorHelper(reduction_tv, rfactor_axes);

        rfactor_tv->axis(-1)->parallelize(ParallelType::TIDy);
        rfactor_tv->axis(4)->parallelize(ParallelType::TIDx);
        rfactor_tv->axis(2)->parallelize(ParallelType::Unswitch);
        rfactor_tv->axis(1)->parallelize(ParallelType::BIDx);
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "Need to bind thread dimension for persistent kernels.");
    }
  }

  // For intermediate outputs, apply cache_fork
  for (const auto output : fusion->outputs()) {
    if (!output->uses().empty()) {
      if (output->getValType().value() == ValType::TensorView) {
        output->as<TensorView>()->cache_fork();
      }
    }
  }

  bool rfactor = rfactor_tv != nullptr;
  auto reference_tv = rfactor ? rfactor_tv : reduction_tv;
  std::vector<TensorView*> rfactor_tvs;

  // Make everything look like reference tv
  TransformPropagator::from(reference_tv);

  for (auto reduction_tv_ : reduction_tvs) {
    if (reduction_tv_ == reduction_tv) {
      // The reduction tv
      rfactor_tvs.push_back(rfactor_tv);
      continue;
    } else {
      // other reduction tvs
      rfactor_tvs.push_back(
          scheduler_utils::rfactorHelper(reduction_tv_, rfactor_axes));
    }
  }

  scheduler_utils::parallelizeAllLike(
      reference_tv, scheduler_utils::allTvs(fusion));

  if (rparams.loop_unroll > 1) {
    // Schedule unrolling on inputs

    // Find unswitch position
    int unswitch_axis = -1;
    for (int i = 0; i < (int)reference_tv->nDims(); i++) {
      if (reference_tv->axis(i)->getParallelType() == ParallelType::Unswitch) {
        unswitch_axis = i;
      }
    }
    unswitch_axis++;

    // Input to cached we want outside unswitched position
    // Cached input to rfactor we want inlined
    std::unordered_set<TensorView*> reference_tvs;
    {
      auto ref_tvs = rfactor ? rfactor_tvs : reduction_tvs;
      std::transform(
          ref_tvs.begin(),
          ref_tvs.end(),
          std::inserter(reference_tvs, reference_tvs.end()),
          [](TensorView* tv) { return tv; });
    }
    for (auto cached_input : cached_inputs) {
      auto consumers_of_input_cache =
          scheduler_utils::consumerTvsOf(cached_input);
      for (auto consumer : consumers_of_input_cache) {
        scheduler_utils::computeWithOutputs(
            consumer, -1, ComputeAtMode::MostInlined);
        cached_input->computeAt(
            consumer, unswitch_axis, ComputeAtMode::BestEffort);
      }
    }

    // These are lined up, inline rfactor tv's into reduction tvs.
    for (size_t red_i = 0;
         red_i < reduction_tvs.size() && red_i < rfactor_tvs.size();
         red_i++) {
      rfactor_tvs[red_i]->computeWith(
          reduction_tvs[red_i], -1, ComputeAtMode::BestEffort);
    }

    for (auto red_tv : reduction_tvs) {
      // TODO: Should reduction also be best effort here? We already tried to
      // inline based on input caches. Can we just remove this?
      scheduler_utils::computeWithOutputs(
          red_tv, -1, ComputeAtMode::BestEffort);
    }

    // Compute at should not remove parallelization scheme, but let's just make
    // sure everything is set properly
    scheduler_utils::parallelizeAllLike(
        reference_tv, scheduler_utils::allTvs(fusion));
  } else {
    // Want to inline, especially backwards based on reduction_tv, otherwise
    // rfactor tv may not be inlined correctly
    for (auto cur_red_it = reduction_tvs.begin();
         cur_red_it != reduction_tvs.end();
         cur_red_it++) {
      if (std::any_of(
              cur_red_it + 1,
              reduction_tvs.end(),
              [&cur_red_it](TensorView* following_red_it) {
                return DependencyCheck::isDependencyOf(
                    *cur_red_it, following_red_it);
              })) {
        // if this reduction is a producer of another, don't compute at from it,
        // as the consumer reduction will cover all tensors that this one would
        // have
        continue;
      }

      scheduler_utils::computeAtInputs(
          *cur_red_it, -1, ComputeAtMode::MostInlined);
      scheduler_utils::computeWithOutputs(
          *cur_red_it, -1, ComputeAtMode::MostInlined);
    }

    scheduler_utils::parallelizeAllLike(
        reference_tv, scheduler_utils::allTvs(fusion));
  }
}

// TODO: This is really similar to persistent normalization except splits that
// are not on inner most dimension. We should probably unify the
// implementations.
void scheduleMultiReduction(Fusion* fusion, const ReductionParams& rparams) {
  FUSER_PERF_SCOPE("scheduleMultiReduction");

  FusionGuard fg(fusion);

  std::vector<TensorView*> reduction_tvs;
  for (auto tv : scheduler_utils::allTvs(fusion)) {
    if (tv->hasReduction() && !fusion->hasInput(tv)) {
      if (auto welford_op = dynamic_cast<WelfordOp*>(tv->definition())) {
        if (tv == welford_op->out()) {
          reduction_tvs.push_back(tv);
        }
      } else {
        reduction_tvs.push_back(tv);
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      !reduction_tvs.empty(), "Need reduction tensor views to schedule.");

  auto reduction_tv = reduction_tvs[0];
  TensorView* rfactor_tv = nullptr;

  scheduler_utils::mergeReduction(reduction_tv);

  // Merge all iteration dimensions
  if (reduction_tv->nDims() > 1) {
    scheduler_utils::mergeNonReduction(reduction_tv);
  }

  // Evaluate Dimensions of Reduction TensorView
  TORCH_INTERNAL_ASSERT(
      reduction_tv->nDims() == 1 || reduction_tv->nDims() == 2,
      "Error coalesing dimensions.");

  if (reduction_tv->domain()->domain().size() == 1) {
    TORCH_INTERNAL_ASSERT(
        rparams.fastest_dim,
        "If all dims are reduction, should be sending it to fastest dim scheduler.");
  }

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  for (auto tv : scheduler_utils::allTvs(fusion)) {
    if (tv->isFusionInput() || tv->isFusionOutput()) {
      tv->setMemoryType(MemoryType::Global);
    } else {
      tv->setMemoryType(MemoryType::Local);
    }
  }

  std::vector<TensorView*> cached_inputs;
  // If we're going to unroll, make a cache of the inputs
  if (rparams.loop_unroll > 1) {
    auto persistent_buffers =
        scheduler_utils::persistentBuffers(fusion).buffers;
    TORCH_INTERNAL_ASSERT(
        persistent_buffers.empty(),
        "Cannot schedule fusions that can produce persistent buffers in multi reduction scheduler.");

    auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    for (auto tv : in_tvs) {
      auto cached_tv = tv->cache_after();
      cached_inputs.emplace_back(cached_tv);
    }
  } else {
    auto in_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    for (auto tv : in_tvs) {
      if (tv->uses().size() > 1) {
        auto cached_tv = tv->cache_after();
        cached_inputs.emplace_back(cached_tv);
      }
    }
  }

  std::vector<int> rfactor_axes;

  // Scheduling the Reduction
  if (rparams.fastest_dim) {
    const bool has_iter_axis = reduction_tv->nDims() == 2;
    const int iter_axis = 0;
    const int reduce_axis = reduction_tv->nDims() == 2 ? 1 : 0;

    // Do multiple reductions per block
    if (rparams.multiple_reds_per_blk) {
      if (rparams.reduction_unroll) {
        // Fastest dim, multiple reductions per block
        // Output Dimensions
        // [x-BIDx, x-TIDy
        //  0       1
        //
        //  Reduction Dimensions
        //  rF-leftover, rf-Unswitch, rf-Unroll, X-TIDx]
        //  2 (-4)     3 (-3)       4 (-2)     5 (-1)

        //  X-TIDx, rF-leftover, rf-Unswitch, rf-Unroll]
        //  2 (-4)  3 (-3)         4 (-2)       5 (-1)
        reduction_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));

        reduction_tv->split(reduce_axis, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reduction_tv->split(reduce_axis, 1);

        reduction_tv->reorder({{-1, -4}, {-4, -3}, {-3, -2}, {-2, -1}});

        rfactor_axes = {-3, -2, -1};
        rfactor_tv = scheduler_utils::rfactorHelper(reduction_tv, rfactor_axes);

        rfactor_tv->axis(-4)->parallelize(ParallelType::TIDx);
        rfactor_tv->axis(-2)->parallelize(ParallelType::Unswitch);

        if (has_iter_axis) {
          rfactor_tv->split(
              iter_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
          rfactor_tv->axis(iter_axis + 1)->parallelize(ParallelType::TIDy);
          if (rparams.split_grid_dim) {
            rfactor_tv->split(iter_axis, x_grid_limit);
            rfactor_tv->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
          } else {
            rfactor_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }
        }
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
        //  rF-persistent, r-TIDx]
        //  4 (-2)     5 (-1)

        reduction_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));

        rfactor_axes = {-2};
        rfactor_tv = scheduler_utils::rfactorHelper(reduction_tv, rfactor_axes);

        rfactor_tv->axis(-1)->parallelize(ParallelType::TIDx);

        if (has_iter_axis) {
          rfactor_tv->split(
              iter_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
          rfactor_tv->split(iter_axis, rparams.loop_unroll);
          // Unswitch axis which gives us finer control on allocations with
          // unrolling
          rfactor_tv->split(iter_axis, 1);

          rfactor_tv->axis(3)->parallelize(ParallelType::TIDy);
          // TODO: Re-enable unswitch in this case:
          // https://github.com/csarofeen/pytorch/issues/748
          // rfactor_tv->axis(1)->parallelize(ParallelType::Unswitch);

          // [BIDx, 1, 8, TIDy, rf-outer, r-TIDx]

          if (rparams.split_grid_dim) {
            rfactor_tv->split(iter_axis, x_grid_limit);
            rfactor_tv->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
          } else {
            rfactor_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }
        }
      }
    } else {
      // Fastest dim, Reduction Splits
      // Output Dimensions
      // [BIDx
      //  0
      //
      //  Reduction Dimensions
      //  rF-Leftover, rf-Unswitch, rf-Unroll, X-TIDx]
      //  1 (-4)         2 (-3)       3 (-2)     4 (-1)

      //  X-TIDx, rF-Leftover, rf-Unswitch, rf-Unroll]
      //  1 (-4)  2 (-3)         3 (-2)       4 (-1)

      reduction_tv->split(
          reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
      reduction_tv->split(reduce_axis, rparams.loop_unroll);
      // Unswitch axis which gives us finer control on allocations with
      // unrolling
      reduction_tv->split(reduce_axis, 1);

      reduction_tv->reorder({{-1, -4}, {-4, -3}, {-3, -2}, {-2, -1}});

      rfactor_axes = {-3, -2, -1};
      rfactor_tv = scheduler_utils::rfactorHelper(reduction_tv, rfactor_axes);

      rfactor_tv->axis(-4)->parallelize(ParallelType::TIDx);
      rfactor_tv->axis(-2)->parallelize(ParallelType::Unswitch);

      if (has_iter_axis) {
        if (rparams.split_grid_dim) {
          rfactor_tv->split(iter_axis, x_grid_limit);
          rfactor_tv->axis(iter_axis + 1)->parallelize(ParallelType::BIDx);
        } else {
          rfactor_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
        }
      }
    }
  } else {
    if (rparams.cross_block) {
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
        reduction_tv->split(1, rparams.loop_unroll);
        reduction_tv->split(1, 1);
        reduction_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));

        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reduction_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
        rfactor_axes = {-4, -2, -1};
        rfactor_tv = scheduler_utils::rfactorHelper(reduction_tv, rfactor_axes);

        rfactor_tv->axis(-2)->parallelize(ParallelType::Unswitch);
        rfactor_tv->axis(-3)->parallelize(ParallelType::TIDy);
        rfactor_tv->axis(1)->parallelize(ParallelType::TIDx);
        rfactor_tv->axis(0)->parallelize(ParallelType::BIDx);
      } else {
        // Outer Dim, cross block, unroll iter dimension

        // Output Dimensions
        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx
        //  0       1           2         3
        //
        // Reduction Dimensions
        // rF-Leftover, r-TIDy]
        // 4(-2)        5(-1)

        reduction_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
        reduction_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
        reduction_tv->split(0, rparams.loop_unroll);
        // Unswitch axis which gives us finer control on allocations with
        // unrolling
        reduction_tv->split(0, 1);
        // [x-BIDx, x-Unswitch, x-Unroll, x-TIDx, rF-Leftover, r-TIDy]
        reduction_tv->reorder({{-2, 0}});
        // [rF-Leftover, x-BIDx, x-Unswitch, x-Unroll, x-TIDx, r-TIDy]
        rfactor_axes = {0};
        rfactor_tv = scheduler_utils::rfactorHelper(reduction_tv, rfactor_axes);

        rfactor_tv->axis(-1)->parallelize(ParallelType::TIDy);
        rfactor_tv->axis(4)->parallelize(ParallelType::TIDx);
        rfactor_tv->axis(2)->parallelize(ParallelType::Unswitch);
        rfactor_tv->axis(1)->parallelize(ParallelType::BIDx);
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "Need to bind thread dimension for persistent kernels.");
    }
  }

  // For intermediate outputs, apply cache_fork
  for (const auto output : fusion->outputs()) {
    if (!output->uses().empty()) {
      if (output->getValType().value() == ValType::TensorView) {
        output->as<TensorView>()->cache_fork();
      }
    }
  }

  bool rfactor = rfactor_tv != nullptr;
  auto reference_tv = rfactor ? rfactor_tv : reduction_tv;
  std::vector<TensorView*> rfactor_tvs;

  // Make everything look like reference tv
  TransformPropagator::from(reference_tv);

  for (auto reduction_tv_ : reduction_tvs) {
    if (reduction_tv_ == reduction_tv) {
      // The reduction tv
      rfactor_tvs.push_back(rfactor_tv);
      continue;
    } else {
      // other reduction tvs
      rfactor_tvs.push_back(
          scheduler_utils::rfactorHelper(reduction_tv_, rfactor_axes));
    }
  }

  scheduler_utils::parallelizeAllLike(
      reference_tv, scheduler_utils::allTvs(fusion));

  if (rparams.loop_unroll > 1) {
    // Schedule unrolling on inputs

    // Find unswitch position
    int unswitch_axis = -1;
    for (int i = 0; i < (int)reference_tv->nDims(); i++) {
      if (reference_tv->axis(i)->getParallelType() == ParallelType::Unswitch) {
        unswitch_axis = i;
      }
    }
    unswitch_axis++;

    // Input to cached we want outside unswitched position
    // Cached input to rfactor we want inlined
    std::unordered_set<TensorView*> reference_tvs;
    {
      auto ref_tvs = rfactor ? rfactor_tvs : reduction_tvs;
      std::transform(
          ref_tvs.begin(),
          ref_tvs.end(),
          std::inserter(reference_tvs, reference_tvs.end()),
          [](TensorView* tv) { return tv; });
    }
    for (auto cached_input : cached_inputs) {
      auto consumers_of_input_cache =
          scheduler_utils::consumerTvsOf(cached_input);
      for (auto consumer : consumers_of_input_cache) {
        scheduler_utils::computeWithOutputs(
            consumer, -1, ComputeAtMode::MostInlined);
        cached_input->computeAt(
            consumer, unswitch_axis, ComputeAtMode::BestEffort);
      }
    }

    // These are lined up, inline rfactor tv's into reduction tvs.
    for (size_t red_i = 0;
         red_i < reduction_tvs.size() && red_i < rfactor_tvs.size();
         red_i++) {
      rfactor_tvs[red_i]->computeWith(
          reduction_tvs[red_i], -1, ComputeAtMode::BestEffort);
    }

    for (auto red_tv : reduction_tvs) {
      // TODO: Should reduction also be best effort here? We already tried to
      // inline based on input caches. Can we just remove this?
      scheduler_utils::computeWithOutputs(
          red_tv, -1, ComputeAtMode::BestEffort);
    }

    scheduler_utils::parallelizeAllLike(
        reference_tv, scheduler_utils::allTvs(fusion));

  } else {
    // Want to inline, especially backwards based on reduction_tv, otherwise
    // rfactor tv may not be inlined correctly

    for (auto red_tv : reduction_tvs) {
      scheduler_utils::computeAtInputs(red_tv, -1, ComputeAtMode::MostInlined);
      scheduler_utils::computeWithOutputs(
          red_tv, -1, ComputeAtMode::MostInlined);
    }

    scheduler_utils::parallelizeAllLike(
        reference_tv, scheduler_utils::allTvs(fusion));
  }
}
} // namespace

// fusion is the input IR that will be modified by this function
TORCH_CUDA_CU_API void scheduleNormalization(
    Fusion* fusion,
    const ReductionParams& rparams) {
  if (rparams.persistent_kernel) {
    schedulePersistentNormalization(fusion, rparams);
  } else {
    scheduleMultiReduction(fusion, rparams);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
