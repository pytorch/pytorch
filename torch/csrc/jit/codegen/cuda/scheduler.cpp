#include <torch/csrc/jit/codegen/cuda/scheduler.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

constexpr int kUnrollFactor = 1;

namespace {

std::vector<int> reductionAxes(TensorView* tv) {
  size_t n_dims = tv->nDims();
  std::vector<int> reduction_axes;
  for (size_t i = 0; i < n_dims; i++) {
    if (tv->axis(i)->isReduction()) {
      reduction_axes.emplace_back(i);
    }
  }
  return reduction_axes;
}

// Merge all reduction to the right side and returns total number of
// reduction axes
size_t mergeReduction(TensorView* tv) {
  int prev_i = -1;
  size_t num_merged = 0;
  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (!tv->axis(i)->isReduction()) {
      continue;
    }
    if (prev_i == -1) {
      prev_i = i;
    } else {
      tv->merge(i, prev_i);
      prev_i = i;
      num_merged++;
    }
  }
  if (prev_i == 0) {
    tv->reorder({{prev_i, -1}});
  }

  return prev_i == -1 ? 0 : num_merged + 1;
}

// merge all non-reduction axes to the left side and returns total number of
// iteration axes
size_t mergeNonReduction(TensorView* tv) {
  int prev_i = -1;
  size_t num_merged = 0;
  for (int i = static_cast<int>(tv->nDims()) - 1; i >= 0; i--) {
    if (tv->axis(i)->isReduction()) {
      continue;
    }
    if (prev_i == -1) {
      prev_i = i;
    } else {
      tv->merge(i, prev_i);
      prev_i = i;
      num_merged++;
    }
  }
  if (prev_i != 0) {
    tv->reorder({{prev_i, 0}});
  }

  return prev_i == -1 ? 0 : num_merged + 1;
}

} // namespace

// This one is a total mess and it should go.
bool scheduleFusion(Fusion* fusion, const at::ArrayRef<c10::IValue> inputs) {
  FUSER_PERF_SCOPE("scheduleFusion");

  FusionGuard fg(fusion);
  // maybe has_reduction for scheduling should be done on a per output tensor
  // basis.
  TORCH_INTERNAL_ASSERT(
      !fusion->hasReduction(), "This scheduler only handles pointwise ops.");
  const bool disable_unroll = fusion->isStochastic();

  for (auto out_val : fusion->outputs()) {
    auto out = out_val->as<TensorView>();

    // Merge all dimensions because we're only supporting pointwise
    // Real reductions aren't supposed to reach here
    // This is a workaround to handle trivial reductions, i.e. size-1 reductions
    mergeNonReduction(out);
  }

  // Run through outputs, grab all inputs of outputs
  // squeeze with computeAt to set overall structure.
  for (auto output : fusion->outputs()) {
    if (output->getValType() != ValType::TensorView)
      continue;
    TensorView* out_tv = output->as<TensorView>();

    // Split into 128 which will be bockDim.x
    out_tv->split(0, kPwThreadX);
    // Split by another 4 which will be our unroll factor
    auto ur_factor = disable_unroll ? 1 : kUnrollFactor;
    out_tv->split(0, ur_factor);
  }

  for (auto output : fusion->outputs()) {
    if (output->getValType() != ValType::TensorView)
      continue;
    TensorView* out_tv = output->as<TensorView>();
    for (Val* inp : fusion->inputsOf(output)) {
      if (inp->getValType().value() == ValType::TensorView)
        inp->as<TensorView>()->computeAt(out_tv, -1);
    }
    out_tv->axis(0)->parallelize(ParallelType::BIDx);
    out_tv->axis(1)->parallelize(ParallelType::Unroll);
    out_tv->axis(2)->parallelize(ParallelType::TIDx);
  }

  return true;
}

namespace {
// Largest Power of 2 less-than n
constexpr int lastPow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  n |= (n >> 16); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  return std::max(1, n - (n >> 1));
}

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) {
    ++log2_value;
  }
  return log2_value;
}

ReductionParams multipleReductionHeuristic(
    int64_t reduction_dim_size,
    int64_t outer_dim_size,
    int64_t inner_dim_size,
    bool fastest_dim_reduction) {
  if (fastest_dim_reduction) {
    TORCH_INTERNAL_ASSERT(reduction_dim_size > 0);
  } else {
    TORCH_INTERNAL_ASSERT(
        reduction_dim_size > 0 && (outer_dim_size > 0 || inner_dim_size > 0));
  }

  const int64_t kMaxThreadsPerCTA =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

  const int64_t kBlockThresholdNotFastestDim = 64;
  const int64_t kBlockThresholdFastestDim = 512;

  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;
  int64_t bdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t bdimy = LaunchParams::UNINITIALIZED_VAL;

  ReductionParams rparams;
  rparams.fastest_dim = fastest_dim_reduction;
  rparams.multiple_reds_per_blk = true;
  rparams.cross_block = false;
  rparams.cross_grid = false;

  // Is fastest dimension a reduction dimension?
  if (rparams.fastest_dim) {
    if (reduction_dim_size <= kMaxThreadsPerCTA) {
      rparams.persistent_kernel = true;

      if (reduction_dim_size <= kBlockThresholdFastestDim) {
        // const int log2_elements = log2_ceil(reduction_dim_size);
        // const int next_power_of_two = 1 << log2_elements;
        // const int kBatchesPerWarp = (next_power_of_two <= 128) ? 2 : 1;
        // rparams.num_warps = 4;

        // TODO: multiple batches per warp causes layer-norm errors
        const int kBatchesPerWarp = 1;
        rparams.batches_per_block = rparams.num_warps * kBatchesPerWarp;
        gdimx = std::max(
            ceilDiv(outer_dim_size, rparams.batches_per_block), (int64_t)1);
        bdimx = at::cuda::warp_size();
      } else {
        // rparams.num_warps = 1;
        // rparams.batches_per_block = 1;
        gdimx = std::max(outer_dim_size, (int64_t)1);
        bdimx = std::min(reduction_dim_size, kMaxThreadsPerCTA);
      }
      // bdimy is the number of warps per block
      bdimy = rparams.num_warps;
      rparams.loop_unroll = ceilDiv(reduction_dim_size, bdimx);
    } else {
      // ILP = sizeof(float4) / sizeof(float)
      const int64_t ILP = 4;
      rparams.loop_unroll = ILP;
      int64_t max_block_size =
          std::min(reduction_dim_size / ILP, kMaxThreadsPerCTA);

      // Combine vectorization while maximizing GPU utilisation
      if (ILP > 1) {
        max_block_size /= 2;
      }

      bdimx = 1;
      while (bdimx < max_block_size) {
        bdimx *= 2;
      }

      // Launch at least a single warp - the kernel assumes that.
      bdimx = std::max(bdimx, (int64_t)at::cuda::warp_size());
      gdimx = std::max(outer_dim_size, (int64_t)1);
    }
  } else {
    rparams.persistent_kernel = false;

    // Warning: Reduce Maximum Threads Per CTA for FP16
    // Register usage exceeds maximum registers per CTA
    const int64_t kFP16MaxThreadsPerCTA = 896;

    // Setup Block Size
    bdimy = std::min(inner_dim_size, kFP16MaxThreadsPerCTA);
    bdimx = 1;
    if (bdimy <= kBlockThresholdNotFastestDim &&
        reduction_dim_size >= kBlockThresholdNotFastestDim) {
      while (bdimy * bdimx <= kMaxThreadsPerCTA &&
             bdimx <= reduction_dim_size) {
        bdimx *= 2;
      }
      bdimx /= 2;
    }
    bdimx = std::max(bdimx, (int64_t)1);

    // Setup Grid Size
    // Estimate maximum number of active blocks
    const int64_t kMaxThreadsPerSM =
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
    const int64_t kSMCount =
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t kNumThreads = bdimx * bdimy;
    const int64_t kActiveBlocks = kMaxThreadsPerSM / kNumThreads;
    const int64_t kMaxActiveBlocks = kActiveBlocks * kSMCount;

    // First, tile blocks over the y-axis
    gdimy = std::min(ceilDiv(inner_dim_size, bdimy), kMaxActiveBlocks);
    // Then, fill the x-axis with remaining blocks
    gdimx = std::min(ceilDiv(kMaxActiveBlocks, gdimy), outer_dim_size);
    gdimx = std::max(gdimx, (int64_t)1);
  }

  const char* debug_env = getenv("PYTORCH_CUDA_FUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n===== Multiple Reduction Parameters ========" << std::endl
              << "Inputs:" << std::endl
              << "\tRed Elems: " << reduction_dim_size
              << " Red Outer: " << outer_dim_size
              << " Red Inner: " << inner_dim_size << " Red On Fastest Dim? "
              << fastest_dim_reduction << std::endl
              << "Reduction Characteristics:" << std::endl
              << "\tMultiple Reds Per Block? " << rparams.multiple_reds_per_blk
              << " Cross Block? " << rparams.cross_block << " Cross Grid? "
              << rparams.cross_grid << std::endl
              << "Recommended Blocking:" << std::endl
              << "\tGridX: " << gdimx << " GridY: " << gdimy << std::endl
              << "\tBlckX: " << bdimx << " BlckY: " << bdimy << std::endl
              << "====================================" << std::endl;
  }

  // Infer BDIMx to avoid conflicts with computeLaunchParams for fastest
  // dimension reduction
  rparams.lparams = LaunchParams(
      gdimx,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      (rparams.fastest_dim && rparams.persistent_kernel)
          ? LaunchParams::UNINITIALIZED_VAL
          : bdimx,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);
  return rparams;
}

ReductionParams reductionHeuristic(
    int num_elems_in_reduction,
    int num_outputs_for_reduction,
    bool fastest_dim_reduction) {
  ReductionParams rparams;
  rparams.fastest_dim = fastest_dim_reduction;

  int gdimx = LaunchParams::UNINITIALIZED_VAL;
  int gdimy = LaunchParams::UNINITIALIZED_VAL;
  int bdimx = LaunchParams::UNINITIALIZED_VAL;
  int bdimy = LaunchParams::UNINITIALIZED_VAL;

  // 1. Initial Assumptions

  // Evaluate Dimensions of Reduction TensorView
  TORCH_INTERNAL_ASSERT(
      num_elems_in_reduction > 0 && num_outputs_for_reduction > 0);

  // 2. Initial Definition of Block Dimensions

  // Is fastest dimension a reduction dimension?
  if (rparams.fastest_dim) {
    if (num_elems_in_reduction < rparams.loop_unroll) {
      rparams.loop_unroll = 1;
    }
    bdimx = ceilDiv(num_elems_in_reduction, rparams.loop_unroll);
    bdimy = num_outputs_for_reduction;
  } else {
    bdimx = num_outputs_for_reduction;
    bdimy = num_elems_in_reduction;
  }

  // 3. Applying Power of 2 Blocking based on the Maximum Number of threads

  constexpr int kMaxNumThreads = 512;
  int num_threads = kMaxNumThreads;
  int device_warp_size = at::cuda::warp_size();

  if (bdimx < num_threads) {
    bdimx = lastPow2(bdimx);
  } else {
    bdimx = num_threads;
  }

  if (bdimy < num_threads) {
    bdimy = lastPow2(bdimy);
  } else {
    bdimy = num_threads;
  }

  int bdimx_prev = bdimx;
  bdimx = std::min(bdimx, device_warp_size);
  bdimy = std::min(bdimy, num_threads / bdimx);
  bdimx = std::min(bdimx_prev, num_threads / bdimy);

  // 4. Distributing work across a block

  // Magic numbers of calculations allowed per thread.
  constexpr int kMinValuesPerThread = 16;
  constexpr int kMaxValuesPerThread = 256;

  int inputs_consumed_per_block_iter = 1;
  int red_elems_per_thread = num_elems_in_reduction;

  int outputs_produced_per_block_iter = 1;

  // Reduction is performed across warp threads (cross-thread reduction)
  if (rparams.fastest_dim) {
    inputs_consumed_per_block_iter *= bdimx;
    red_elems_per_thread =
        ceilDiv(red_elems_per_thread, inputs_consumed_per_block_iter);
    // Warp threads are applied across the output
  } else {
    outputs_produced_per_block_iter *= bdimx;
  }

  // Decision to do a cross-warp reduction per block
  if (red_elems_per_thread >= (bdimy * kMinValuesPerThread) ||
      red_elems_per_thread >= kMaxValuesPerThread || !rparams.fastest_dim) {
    inputs_consumed_per_block_iter *= bdimy;
    red_elems_per_thread = ceilDiv(red_elems_per_thread, bdimy);
    rparams.cross_block = true;
    rparams.multiple_reds_per_blk = false;
    // Do multiple reductions per block
  } else {
    rparams.cross_block = false;
    rparams.multiple_reds_per_blk = true;
    outputs_produced_per_block_iter *= bdimy;
  }

  // 5. Distributing work across blocks

  // WARNING: Current device for codegen may not be the target device
  int device_max_threads_per_multiprocessor =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
  int device_multiprocessor_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  int blocks_per_sm = device_max_threads_per_multiprocessor / (bdimx * bdimy);
  int target_grid_size = device_multiprocessor_count * blocks_per_sm;

  // Setting the number of blocks based on the number of outputs
  gdimx = ceilDiv(num_outputs_for_reduction, outputs_produced_per_block_iter);

  // Cross-block reductions (if necessary)
  if (rparams.cross_block && red_elems_per_thread >= kMaxValuesPerThread &&
      gdimx <= target_grid_size) {
    int blks_per_out_1 = ceilDiv(target_grid_size, gdimx);
    int blks_per_out_2 = ceilDiv(red_elems_per_thread, kMinValuesPerThread);
    int blks_per_out_3 = ceilDiv(red_elems_per_thread, kMaxValuesPerThread);
    int blks_per_output =
        std::max(std::min(blks_per_out_1, blks_per_out_2), blks_per_out_3);

    gdimy = std::max(1, blks_per_output);
    // If a cross-block reduction was generated
    if (blks_per_output > 1) {
      rparams.cross_grid = true;
    }
  }

  const char* debug_env = getenv("PYTORCH_CUDA_FUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n===== Reduction Parameters ========" << std::endl
              << "Inputs:" << std::endl
              << "\tRed Elems: " << num_elems_in_reduction
              << " Red Outputs: " << num_outputs_for_reduction
              << " Red On Fastest Dim? " << fastest_dim_reduction << std::endl
              << "Reduction Characteristics:" << std::endl
              << "\tMultiple Reds Per Block? " << rparams.multiple_reds_per_blk
              << " Cross Block? " << rparams.cross_block << " Cross Grid? "
              << rparams.cross_grid << std::endl
              << "Recommended Blocking:" << std::endl
              << "\tGridX: " << gdimx << " GridY: " << gdimy
              << " BlckX: " << bdimx << " BlckY: " << bdimy << std::endl
              << "====================================" << std::endl;
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
} // anonymous namespace

TORCH_CUDA_API c10::optional<ReductionParams> getMultipleReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    const std::vector<TensorView*>& reduction_tv) {
  FUSER_PERF_SCOPE("scheduleMultipleReduction");
  FusionGuard fg(fusion);
  if (!fusion->hasReduction()) {
    return c10::nullopt;
  }

  TORCH_INTERNAL_ASSERT(
      reduction_tv.size() > 1,
      "A single reduction tv was detected. Use getReductionHeuristics.");

  // Check Reduction Invariants
  for (auto tv : reduction_tv) {
    TORCH_INTERNAL_ASSERT(tv != nullptr, "Reduction TensorView wasn't found.");
    TORCH_INTERNAL_ASSERT(
        tv->hasReduction(), "TensorView doesn't have a reduction.");
    const auto reduction_origin_expr = fusion->origin(tv);
    TORCH_INTERNAL_ASSERT(
        reduction_origin_expr->getExprType() != c10::nullopt &&
            reduction_origin_expr->getExprType().value() ==
                ExprType::ReductionOp,
        "TensorView doesn't have a reduction.");
  }

  auto evaluator = executor_utils::bindFusionInputs(fusion_inputs, fusion);

  std::vector<int> reduction_elements;
  std::vector<int> reduction_outer;
  std::vector<int> reduction_inner;
  std::vector<bool> fastest_dim_reduction;

  for (auto tv : reduction_tv) {
    bool has_outer = false;
    bool has_inner = false;
    int this_outer_size = 1;
    int this_inner_size = 1;
    int this_reduction_size = 1;
    bool this_fastest_dim_reduction = false;

    bool before_reduction = true;
    for (auto id : tv->getRootDomain()) {
      auto inferred_dim_size = evaluator.evaluate(id->rawExtent());
      TORCH_INTERNAL_ASSERT(
          inferred_dim_size.has_value(), "Error inferring dimension size.");

      if (id->isReduction()) {
        this_reduction_size *= inferred_dim_size.value();
        before_reduction = false;
      } else if (before_reduction) {
        has_outer = true;
        this_outer_size *= inferred_dim_size.value();
      } else {
        has_inner = true;
        this_inner_size *= inferred_dim_size.value();
      }
    }

    if (!has_outer) {
      this_outer_size = 0;
    }
    if (!has_inner) {
      this_inner_size = 0;
    }

    reduction_elements.push_back(this_reduction_size);
    reduction_outer.push_back(this_outer_size);
    reduction_inner.push_back(this_inner_size);
    fastest_dim_reduction.push_back(!has_inner);
  }

  // Check that the dimensions of the reductions are equal
  for (size_t idx = 1; idx < fastest_dim_reduction.size(); ++idx) {
    TORCH_INTERNAL_ASSERT(
        reduction_elements[idx] == reduction_elements[idx - 1]);
    TORCH_INTERNAL_ASSERT(reduction_outer[idx] == reduction_outer[idx - 1]);
    TORCH_INTERNAL_ASSERT(reduction_inner[idx] == reduction_inner[idx - 1]);
    TORCH_INTERNAL_ASSERT(
        fastest_dim_reduction[idx] == fastest_dim_reduction[idx - 1]);
  }

  return multipleReductionHeuristic(
      reduction_elements.front(),
      reduction_outer.front(),
      reduction_inner.front(),
      fastest_dim_reduction.front());
}

TORCH_CUDA_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    TensorView* red_tv) {
  FUSER_PERF_SCOPE("getReductionHeuristics");

  FusionGuard fg(fusion);

  auto red_root_dom = red_tv->getRootDomain();
  const bool fastest_dim_reduction =
      red_root_dom[red_root_dom.size() - 1]->isReduction();

  TORCH_INTERNAL_ASSERT(
      red_tv != nullptr, "Reduction TensorView wasn't found.");

  TORCH_INTERNAL_ASSERT(
      red_tv->hasReduction(), "TensorView doesn't have a reduction.");
  const auto red_expr = fusion->origin(red_tv);

  TORCH_INTERNAL_ASSERT(
      red_expr->getExprType() != c10::nullopt &&
          red_expr->getExprType().value() == ExprType::ReductionOp,
      "TensorView doesn't have a reduction.");

  auto evaluator = executor_utils::bindFusionInputs(fusion_inputs, fusion);

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

  return reductionHeuristic(
      red_elements, num_outputs_for_reduction, fastest_dim_reduction);
}

namespace {

void scheduleReductionComputeAt(
    TensorView* red_tv,
    TensorView* red_tv_rf,
    const std::vector<TensorView*>& outs_of_red) {
  if (!outs_of_red.empty()) {
    red_tv->computeAt(outs_of_red[0], -1);
  }
  if (red_tv_rf != nullptr) {
    red_tv_rf->computeAt(red_tv, -1);
  }
}

} // namespace

// fusion is the input IR that will be modified by this function
void scheduleReduction(
    Fusion* fusion,
    const ReductionParams& rparams,
    TensorView* red_tv,
    std::vector<TensorView*> outs_of_red) {
  FUSER_PERF_SCOPE("scheduleReduction");
  FusionGuard fg(fusion);

  // We coalesce all reduction axes to the right;
  mergeReduction(red_tv);

  // Merge all iteration dimensions
  if (red_tv->domain()->domain().size() > 1) {
    mergeNonReduction(red_tv);
    for (auto iter_tv : outs_of_red) {
      mergeNonReduction(iter_tv);
    }
  }

  // Evaluate Dimensions of Reduction TensorView
  auto red_ids = red_tv->domain()->domain();

  TORCH_INTERNAL_ASSERT(
      red_ids.size() == 1 || red_ids.size() == 2,
      "We coalesced all dimensions into 1 or 2 previously.");

  if (red_ids.size() == 1) {
    TORCH_INTERNAL_ASSERT(
        rparams.fastest_dim,
        "If all dims are reduction, so should the fastest dim.");
  }

  constexpr int kLoopUnrollSplit = 4;

  // Scheduling the Reduction
  if (rparams.fastest_dim) {
    const bool has_iter_axis = red_ids.size() == 2;
    const int iter_axis = 0;
    const int reduce_axis = red_ids.size() == 2 ? 1 : 0;

    // Do multiple reductions per block
    if (rparams.multiple_reds_per_blk) {
      // Reduction Splits
      //      [outputs, |rF-Leftover, X-Warp, rf-Unroll|]
      // Idx:     0     |   1(-1)      2(-2)     3(-1) |
      //                --------------------------------
      //                Reduction Dimensions
      red_tv->split(reduce_axis, rparams.loop_unroll);
      red_tv->split(
          reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));

      // Output Splits
      //      [|Out-Leftover, Out-PerBlock|, <Reduction Dims>]
      // Idx:  |     0             1      |   2(-2) -- 3(-1)
      //       ----------------------------
      //       Output Dimensions
      if (has_iter_axis) {
        red_tv->split(
            iter_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
        for (auto iter_tv : outs_of_red) {
          iter_tv->split(
              iter_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
        }
      }

      auto red_tv_rf = red_tv->rFactor({-3, -1});

      scheduleReductionComputeAt(red_tv, red_tv_rf, outs_of_red);

      red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

      if (has_iter_axis) {
        red_tv->axis(0)->parallelize(ParallelType::BIDx);
        for (auto iter_tv : outs_of_red) {
          iter_tv->axis(0)->parallelize(ParallelType::BIDx);
        }
        red_tv->axis(1)->parallelize(ParallelType::TIDy);
        for (auto iter_tv : outs_of_red) {
          iter_tv->axis(1)->parallelize(ParallelType::TIDy);
        }
      }

      red_tv->axis(-1)->parallelize(ParallelType::TIDx);

      // Bind Inputs to Reduction
      for (auto input : fusion->inputsOf(red_tv_rf)) {
        if (input->getValType().value() == ValType::TensorView) {
          input->as<TensorView>()->computeAt(red_tv_rf, -1);
        }
      }
      // Do a cross-warp reduction per block
    } else {
      if (rparams.cross_grid) {
        // Reduction Splits
        //      [outputs, |rF-Leftover, X-Grid, X-Block, X-Warp, rf-Unroll|]
        // Idx:     0     |   1(-5)      2(-4)    3(-3)   4(-2)     5(-1) |
        //                -------------------------------------------------
        //                Reduction Dimensions
        red_tv->split(reduce_axis, rparams.loop_unroll);
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::BIDy));

        auto red_tv_rf = red_tv->rFactor(
            {-5, -1}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

        scheduleReductionComputeAt(red_tv, red_tv_rf, outs_of_red);

        red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

        if (has_iter_axis) {
          red_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
          for (auto iter_tv : outs_of_red) {
            iter_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }
        }
        red_tv->axis(-1)->parallelize(ParallelType::TIDx);
        red_tv->axis(-2)->parallelize(ParallelType::TIDy);
        red_tv->axis(-3)->parallelize(ParallelType::BIDy);

        // Bind Inputs to Reduction
        for (auto input : fusion->inputsOf(red_tv_rf)) {
          if (input->getValType().value() == ValType::TensorView) {
            input->as<TensorView>()->computeAt(red_tv_rf, -1);
          }
        }
      } else {
        // Reduction Splits
        //      [outputs, |rF-Leftover, X-Block, X-Warp, rf-Unroll|]
        // Idx:     0     |   1(-4)       2(-3)   3(-2)     4(-1) |
        //                -----------------------------------------
        //                Reduction Dimensions
        red_tv->split(reduce_axis, rparams.loop_unroll);
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
        red_tv->split(
            reduce_axis, NamedScalar::getParallelDim(ParallelType::TIDy));

        auto red_tv_rf = red_tv->rFactor({-4, -1});

        scheduleReductionComputeAt(red_tv, red_tv_rf, outs_of_red);

        red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

        if (has_iter_axis) {
          red_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
          for (auto iter_tv : outs_of_red) {
            iter_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);
          }
        }

        red_tv->axis(-1)->parallelize(ParallelType::TIDx);
        red_tv->axis(-2)->parallelize(ParallelType::TIDy);

        // Bind Inputs to Reduction
        for (auto input : fusion->inputsOf(red_tv_rf)) {
          if (input->getValType().value() == ValType::TensorView) {
            input->as<TensorView>()->computeAt(red_tv_rf, -1);
          }
        }
      }
    }
  } else {
    if (rparams.cross_block) {
      if (rparams.cross_grid) {
        // Reduction Splits
        //      [outputs, |rF-Leftover, rf-Unroll, X-Grid, X-Block|]
        // Idx:     0     |   1(-4)       2(-3)     3(-2)   4(-1) |
        //                -----------------------------------------
        //                Reduction Dimensions
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::BIDy));
        red_tv->split(1, kLoopUnrollSplit);

        // Reordering the Unroll dimension eases applying computeAt()
        // for preceeding operations and the rFactored Tensor.
        //                                 |--- Reordered ----|
        //                                 V                  V
        //      [outputs, |rF-Leftover, X-Block, X-Grid, rF-Unroll|]
        // Idx:     0     |   1(-4)      2(-3)   3(-2)     4(-1)  |
        //                -----------------------------------------
        //                Reduction Dimensions
        red_tv->reorder({{-1, -3}, {-3, -1}});

        // Output Splits
        //      [|Out-Leftover, Out-PerBlock|, <Reduction Dims>]
        // Idx:  |     0             1      |   2(-4) -- 5(-1)
        //       ----------------------------
        //       Output Dimensions
        red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
        for (auto iter_tv : outs_of_red) {
          iter_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
        }

        auto red_tv_rf = red_tv->rFactor({-4, -1});

        scheduleReductionComputeAt(red_tv, red_tv_rf, outs_of_red);

        red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

        red_tv->axis(0)->parallelize(ParallelType::BIDx);
        for (auto iter_tv : outs_of_red) {
          iter_tv->axis(0)->parallelize(ParallelType::BIDx);
          iter_tv->axis(1)->parallelize(ParallelType::TIDx);
        }

        red_tv->axis(-3)->parallelize(ParallelType::TIDx);
        red_tv->axis(-2)->parallelize(ParallelType::TIDy);
        red_tv->axis(-1)->parallelize(ParallelType::BIDy);

        // Bind Inputs to Reduction
        for (auto input : fusion->inputsOf(red_tv_rf)) {
          if (input->getValType().value() == ValType::TensorView) {
            input->as<TensorView>()->computeAt(red_tv_rf, -1);
          }
        }
      } else {
        // Reduction Splits
        //      [outputs, |rF-Leftover, rf-Unroll, X-Block|]
        // Idx:     0     |   1(-3)       2(-2)     3(-1) |
        //                ---------------------------------
        //                Reduction Dimensions
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
        red_tv->split(1, kLoopUnrollSplit);

        // Reordering the Unroll dimension eases applying computeAt()
        // for preceeding operations and the rFactored Tensor.
        //                               |- Reordered -|
        //                               V             V
        //      [outputs, |rF-Leftover, X-Block, rF-Unroll|]
        // Idx:     0     |   1(-3)      2(-2)     3(-1)  |
        //                ---------------------------------
        //                Reduction Dimensions
        red_tv->reorder({{-1, -2}, {-2, -1}});

        // Output Splits
        //      [|Out-Leftover, Out-PerBlock|, <Reduction Dims>]
        // Idx:  |     0             1      |   2(-3) -- 4(-1)
        //       ----------------------------
        //       Output Dimensions
        red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
        for (auto iter_tv : outs_of_red) {
          iter_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
        }

        auto red_tv_rf = red_tv->rFactor({-3, -1});

        scheduleReductionComputeAt(red_tv, red_tv_rf, outs_of_red);

        red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

        red_tv->axis(0)->parallelize(ParallelType::BIDx);
        for (auto iter_tv : outs_of_red) {
          iter_tv->axis(0)->parallelize(ParallelType::BIDx);
          iter_tv->axis(1)->parallelize(ParallelType::TIDx);
        }
        red_tv->axis(-2)->parallelize(ParallelType::TIDx);
        red_tv->axis(-1)->parallelize(ParallelType::TIDy);

        // Bind Inputs to Reduction
        for (auto input : fusion->inputsOf(red_tv_rf)) {
          if (input->getValType().value() == ValType::TensorView) {
            input->as<TensorView>()->computeAt(red_tv_rf, -1);
          }
        }
      }
    } else {
      red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
      for (auto iter_tv : outs_of_red) {
        iter_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDx));
      }

      scheduleReductionComputeAt(red_tv, nullptr, outs_of_red);

      red_tv->axis(0)->parallelize(ParallelType::BIDx);
      red_tv->axis(1)->parallelize(ParallelType::TIDx);
      for (auto iter_tv : outs_of_red) {
        iter_tv->axis(0)->parallelize(ParallelType::BIDx);
        iter_tv->axis(1)->parallelize(ParallelType::TIDx);
      }

      for (auto input : fusion->inputsOf(red_tv)) {
        if (input->getValType().value() == ValType::TensorView) {
          input->as<TensorView>()->computeAt(red_tv, -1);
        }
      }
    }
  }
}

namespace {

bool isPointwiseOp(const Expr* expr) {
  return expr->outputs().size() == 1 && ir_utils::isTV(expr->output(0)) &&
      (expr->getExprType().value() == ExprType::BinaryOp ||
       expr->getExprType().value() == ExprType::UnaryOp ||
       expr->getExprType().value() == ExprType::TernaryOp);
}

bool isConstantAllocation(const TensorView* tv) {
  if (!tv->hasComputeAt()) {
    // We cannot determine allocation size without computeAt structure.
    // Assume Non-Constant Allocation
    return false;
  }

  bool constant_allocation = true;
  auto domain = tv->domain()->domain();
  for (size_t axis = tv->getThisComputeAtAxis(); axis < domain.size(); ++axis) {
    if (!domain[axis]->isBroadcast() && !domain[axis]->isReduction()) {
      constant_allocation &= domain[axis]->isConstScalar();
    }
  }
  return constant_allocation;
}

//! Find all TensorViews that require duplication to avoid recompute
//! computeAt error when applying inline ComputeAt
std::vector<TensorView*> findTensorViewsToDuplicate(
    Fusion* fusion,
    const std::vector<TensorView*>& other_tv) {
  std::vector<TensorView*> duplicate_tv;
  // Initialize stack with any pointwise op with multiple usages
  // Find any pointwise origin expressions via depth-first search (DFS)
  std::vector<TensorView*> stack;
  for (auto tensor : other_tv) {
    if (fusion->unordered_uses(tensor).size() > 1) {
      stack.push_back(tensor);
    }
  }

  std::unordered_set<StmtNameType> visited;
  while (!stack.empty()) {
    auto tensor = stack.back();
    stack.pop_back();

    if (visited.find(tensor->name()) == visited.end()) {
      auto origin_expr = tensor->getOrigin();
      if (isPointwiseOp(origin_expr)) {
        duplicate_tv.push_back(tensor);

        for (auto input_tv :
             ir_utils::filterByType<TensorView>(origin_expr->inputs())) {
          if (!fusion->hasInput(input_tv) && !isConstantAllocation(input_tv)) {
            stack.push_back(input_tv);
          }
        }
      }
    }
    visited.insert(tensor->name());
  }

  // sort TensorViews in descending order
  std::sort(
      duplicate_tv.begin(),
      duplicate_tv.end(),
      [](TensorView* left, TensorView* right) {
        return left->name() > right->name();
      });
  return duplicate_tv;
}

//! Find all TensorViews that require inline ComputeAt
//! to avoid non-static allocation error
std::vector<TensorView*> findTensorViewsToComputeAtInline(
    Fusion* fusion,
    const std::vector<TensorView*>& other_tv) {
  std::vector<TensorView*> computeAt_inline_tv;
  for (auto tv : other_tv) {
    if (!fusion->hasInput(tv) && !fusion->hasOutput(tv)) {
      if (!isConstantAllocation(tv) &&
          tv->getMemoryType() == MemoryType::Local) {
        computeAt_inline_tv.push_back(tv);
      }
    }
  }
  return computeAt_inline_tv;
}

//! Place all cache TensorViews in Shared Memory
//! All point-wise TensorViews inherit shared memory from their parents
void setupSharedMemory(
    Fusion* fusion,
    const std::vector<TensorView*>& cache_tv) {
  std::vector<TensorView*> stack(cache_tv.begin(), cache_tv.end());
  while (!stack.empty()) {
    auto tensor = stack.back();
    stack.pop_back();
    if (!fusion->hasOutput(tensor) && !fusion->hasInput(tensor)) {
      tensor->setMemoryType(MemoryType::Shared);
      for (auto expr : fusion->unordered_uses(tensor)) {
        if (isPointwiseOp(expr)) {
          auto output = expr->output(0)->as<TensorView>();
          stack.push_back(output);
        }
      }
    }
  }
}

void organizeAxes(
    const std::vector<TensorView*>& reduction_tv,
    const std::vector<TensorView*>& all_tv) {
  // Determine merged reduction axis position
  auto findMergedReductionAxis = [](TensorView* reduction_tv) {
    int merged_reduction_axis = -1;
    auto domain = reduction_tv->domain()->domain();
    for (size_t axis = 0; axis < domain.size(); ++axis) {
      if (domain[axis]->isReduction()) {
        TORCH_INTERNAL_ASSERT(merged_reduction_axis == -1);
        merged_reduction_axis = axis;
      }
    }
    return merged_reduction_axis;
  };

  auto first_reduction_tv = reduction_tv.front();
  auto root_domain = first_reduction_tv->getRootDomain();
  int merged_reduction_axis = -1;

  // Find reduction axes positions
  std::vector<int> reduction_axes;
  for (size_t axis = 0; axis < root_domain.size(); ++axis) {
    if (root_domain[axis]->isReduction()) {
      reduction_axes.push_back(axis);
    }
  }

  // Coalese reduction axes together
  for (auto tv : all_tv) {
    const int kOuterAxis = reduction_axes.front();
    for (int idx = 0; idx < reduction_axes.size() - 1; ++idx) {
      int inner_axis = reduction_axes[idx + 1] - idx;
      tv->merge(kOuterAxis, inner_axis);
    }
  }

  // Coalese non-reduction axes together divided by merged reduction axis
  // Flatten input into [Outer, Reduction, Inner]
  merged_reduction_axis = findMergedReductionAxis(first_reduction_tv);
  const int kBeforeReductionAxis = merged_reduction_axis - 1;
  const int kAfterReductionAxis = merged_reduction_axis + 1;
  const int kNumberOfDims = first_reduction_tv->nDims();
  for (auto tv : all_tv) {
    for (int idx = 0; idx < kBeforeReductionAxis; ++idx) {
      tv->merge(0, 1);
    }
    for (int idx = kAfterReductionAxis; idx < kNumberOfDims - 1; ++idx) {
      tv->merge(kAfterReductionAxis, kAfterReductionAxis + 1);
    }
  }

  // Move reduction axes to the inner-most position
  merged_reduction_axis = findMergedReductionAxis(first_reduction_tv);
  const size_t kInnerMostAxis = first_reduction_tv->domain()->nDims() - 1;
  if (merged_reduction_axis != kInnerMostAxis) {
    for (auto tv : all_tv) {
      tv->reorder({{merged_reduction_axis, kInnerMostAxis},
                   {kInnerMostAxis, merged_reduction_axis}});
    }
  }
}

} // namespace

void scheduleMultipleReduction(
    Fusion* fusion,
    const ReductionParams& rparams,
    const std::vector<TensorView*>& reduction_tv,
    std::vector<TensorView*>& other_tv) {
  FusionGuard fg(fusion);

  const auto& in_tv = ir_utils::filterByType<TensorView>(fusion->inputs());
  const auto& out_tv = ir_utils::filterByType<TensorView>(fusion->outputs());

  std::vector<TensorView*> all_tv;
  for (auto input : in_tv) {
    if (input->getRootDomain().size() ==
        reduction_tv.front()->getRootDomain().size()) {
      all_tv.push_back(input);
    }
  }
  all_tv.insert(all_tv.end(), reduction_tv.begin(), reduction_tv.end());
  all_tv.insert(all_tv.end(), other_tv.begin(), other_tv.end());

  organizeAxes(reduction_tv, all_tv);

  // Determine if there are any casts on fusion inputs
  bool has_input_casts = false;
  for (auto tv : other_tv) {
    const auto kOriginExpr = tv->getOrigin();
    const bool kIsCastOp = kOriginExpr->getExprType() == ExprType::UnaryOp &&
        kOriginExpr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Cast;
    has_input_casts |= kIsCastOp;
  }

  // Scheduling the Reduction
  if (rparams.fastest_dim) {
    const bool kHasOuterAxis = reduction_tv.front()->nDims() > 1;
    if (rparams.persistent_kernel) {
      // 1) Apply heuristics to each reduction
      std::vector<TensorView*> rfactor_tv;
      for (auto tv : reduction_tv) {
        if (kHasOuterAxis && rparams.batches_per_block > 1 &&
            rparams.num_warps > 1) {
          // Output Splits
          //      [Out-Lft, Out-PerBlock?, Out-NumWarps>|, <Reduction Dims>]
          // Idx: |     0             1             2   |
          //      ---------------------------------------
          //       Output Dimensions
          tv->split(0, rparams.batches_per_block);
          tv->split(1, rparams.num_warps);
        }

        // Reduction Split
        //      [outer,   |rF-Leftover, rf-Unroll|]
        // Idx:     0     |   (-2)       (-1)    |
        //                ----------------------
        //                Reduction Dimensions
        tv->split(-1, rparams.loop_unroll);

        auto reduction_tv_rf = tv->rFactor({-1});
        rfactor_tv.push_back(reduction_tv_rf);
      }

      // 3) Split the other TensorViews
      for (auto tv : other_tv) {
        if (kHasOuterAxis && rparams.batches_per_block > 1 &&
            rparams.num_warps > 1) {
          tv->split(0, rparams.batches_per_block);
          tv->split(1, rparams.num_warps);
        }
        tv->split(-1, rparams.loop_unroll);
      }

      if (kHasOuterAxis) {
        // 4) ComputeAt Structure
        const int kComputeAtAxis = 1;
        for (auto input : in_tv) {
          for (auto output : out_tv) {
            if (input->getRootDomain().size() ==
                output->getRootDomain().size()) {
              input->computeAt(output, kComputeAtAxis);
            }
          }
        }

        // 5) Handle Inline-ComputeAt
        // Fusion input castOp replaces cache_after
        if (!has_input_casts) {
          for (const auto input : in_tv) {
            other_tv.push_back(input->cache_after());
          }
        }
      }

      // 6) Parallel Binding
      //      [Out-Lft, Out-PerBlock?, Out-NumWarps>|, rF-Lft,  rf-Unroll]
      // Idx: [   0        1              2         |    3          4    ]
      //      [  BIDx      1             TIDy       |   TIDx        4    ]
      //      |-------------------------------------|--------------------]
      //                    Outer                         Reduction
      // For all TensorViews
      for (auto tv : other_tv) {
        if (kHasOuterAxis) {
          tv->axis(0)->parallelize(ParallelType::BIDx);
          if (rparams.num_warps > 1) {
            tv->axis(2)->parallelize(ParallelType::TIDy);
          }
        }
        tv->axis(-2)->parallelize(ParallelType::TIDx);
      }

      // Reduction TensorViews
      for (auto tv : reduction_tv) {
        if (kHasOuterAxis) {
          tv->axis(0)->parallelize(ParallelType::BIDx);
          if (rparams.num_warps > 1) {
            tv->axis(2)->parallelize(ParallelType::TIDy);
          }
        }
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }

      // rFactor TensorViews
      for (auto tv : rfactor_tv) {
        if (kHasOuterAxis) {
          tv->axis(0)->parallelize(ParallelType::BIDx);
          if (rparams.num_warps > 1) {
            tv->axis(2)->parallelize(ParallelType::TIDy);
          }
        }
        tv->axis(-2)->parallelize(ParallelType::TIDx);
      }
      // end persistent kernel
    } else {
      // 1) Apply heuristics to each reduction
      std::vector<TensorView*> rfactor_tv;
      for (auto tv : reduction_tv) {
        // Reduction Splits
        //      [ Outer  |, rF-Leftover, rf-TDX, rf-Unroll|]
        // Idx:     0    |     1         2         3      |
        //               ----------------------------------
        //                       Reduction Dimensions
        tv->split(-1, rparams.loop_unroll);
        tv->split(-2, rparams.lparams.bdimx());

        auto reduction_tv_rf = tv->rFactor({-3, -1});
        rfactor_tv.push_back(reduction_tv_rf);
      }

      // 2) Split the other TensorViews
      for (auto tv : other_tv) {
        tv->split(-1, rparams.loop_unroll);
        tv->split(-2, rparams.lparams.bdimx());
      }

      if (kHasOuterAxis) {
        // 3) ComputeAt Structure
        const int kComputeAtAxis = 1;
        for (auto input : in_tv) {
          for (auto output : out_tv) {
            if (input->getRootDomain().size() ==
                output->getRootDomain().size()) {
              input->computeAt(output, kComputeAtAxis);
            }
          }
        }

        // 4) Find TensorViews to duplicate
        auto duplicate_tv = findTensorViewsToDuplicate(fusion, other_tv);

        // Any TVs with multiple uses and dependencies with same IterDomain
        // Order of Duplication is necessary for correctness
        for (auto tensor : duplicate_tv) {
          auto result = tensor->duplicate();
          other_tv.insert(other_tv.end(), result.begin(), result.end());
        }

        // 5) Handle Inline-ComputeAt
        auto compute_inline_tv =
            findTensorViewsToComputeAtInline(fusion, other_tv);
        for (auto tensor : compute_inline_tv) {
          auto uses = fusion->unordered_uses(tensor);
          TORCH_INTERNAL_ASSERT(
              uses.size() == 1,
              "This inline-computeAt TensorView ",
              tensor->name(),
              " is used multiple times.")
          Expr* expr = *uses.begin();
          TensorView* consumer = expr->output(0)->as<TensorView>();
          tensor->computeAt(consumer, -1);
        }
      }

      // 6) Parallel Binding
      //      [ outer |, rF-Leftover, rf-TDX, rf-Unroll]
      // Idx: [  BIDx |     1          TIDx       3    ]
      //      |-------|--------------------------------]
      //        Outer             Reduction
      // For all TensorViews
      for (auto tv : other_tv) {
        if (kHasOuterAxis) {
          tv->axis(0)->parallelize(ParallelType::BIDx);
        }
        tv->axis(-2)->parallelize(ParallelType::TIDx);
      }

      // Reduction TensorViews
      for (auto tv : reduction_tv) {
        if (kHasOuterAxis) {
          tv->axis(0)->parallelize(ParallelType::BIDx);
        }
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }

      // rFactor TensorViews
      for (auto tv : rfactor_tv) {
        if (kHasOuterAxis) {
          tv->axis(0)->parallelize(ParallelType::BIDx);
        }
        tv->axis(-2)->parallelize(ParallelType::TIDx);
      }
    } // end non-persistent
    // end fastest_dim logic
  } else {
    // non_fastest_dim logic
    const bool outer_axis_exists = reduction_tv.front()->nDims() > 2;
    const int reduction_axis =
        reduction_tv.front()->domain()->getReductionAxis().value();
    const int inner_axis = reduction_axis - 1;
    TORCH_INTERNAL_ASSERT(!outer_axis_exists || (inner_axis != 0));

    // 1) For each reduction, apply reduction heuristics
    std::vector<TensorView*> rfactor_tv;
    for (auto tv : reduction_tv) {
      bool rfactor_axis = false;

      // Reduction Splits - [outer, inner, reduction-Leftover, TDX?]
      if (rparams.lparams.bdimx() > 1) {
        // Reduction Split
        //      [outer, inner, | rF-Leftover, rf-TIDx  ]
        // Idx:     0     1    |   (-2)       (-1)     |
        //                     -------------------------
        //                        Reduction Dimensions
        rfactor_axis = true;
        tv->split(
            reduction_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
      }

      // Inner Splits
      //      [Outer, |Inner-Lft, Inner-BIDy, Inner-TIDy|, <Reduction Dims>]
      // Idx:         |     0        1             2    |
      //              ---------------------------------------
      //                          Inner Dimensions
      tv->split(inner_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
      tv->split(inner_axis, NamedScalar::getParallelDim(ParallelType::BIDy));

      // Outer Splits
      //      [Outer-Leftover, Outer-BIDx |, Inner, <Reduction Dims>]
      // Idx: |     0             1       |
      //      -----------------------------
      //             Outer Dimensions
      if (outer_axis_exists && rparams.lparams.gdimx() > 1) {
        tv->split(0, NamedScalar::getParallelDim(ParallelType::BIDx));
      }

      if (rfactor_axis) {
        auto reduction_tv_rf = tv->rFactor({-2});
        rfactor_tv.push_back(reduction_tv_rf);
      }
    }

    // 2) Other Tensor Splits
    for (auto tv : other_tv) {
      // Reduction Splits - [outer, inner, reduction-Leftover, TDX?]
      if (rparams.lparams.bdimx() > 1) {
        tv->split(
            reduction_axis, NamedScalar::getParallelDim(ParallelType::TIDx));
      }

      // Inner Splits - [outer, inner-Leftover, BDY, TDY, reduction]
      tv->split(inner_axis, NamedScalar::getParallelDim(ParallelType::TIDy));
      tv->split(inner_axis, NamedScalar::getParallelDim(ParallelType::BIDy));

      // Outer Splits
      // [outer-Leftover, BDX?, inner-Leftover, BDY, TDY, reduction]
      if (outer_axis_exists && rparams.lparams.gdimx() > 1) {
        tv->split(0, NamedScalar::getParallelDim(ParallelType::BIDx));
      }
    }

    int kBIDyAxis = -1;
    if (outer_axis_exists) {
      if (rparams.lparams.gdimx() > 1) {
        kBIDyAxis = 3;
      } else {
        kBIDyAxis = 2;
      }
    } else {
      kBIDyAxis = 1;
    }
    TORCH_INTERNAL_ASSERT(kBIDyAxis > 0);
    const int kTIDyAxis = kBIDyAxis + 1;

    // 3) ComputeAt structure
    // [outer-lft, BDX?, inner-lft, BDY, TDY, reduction-lft, TDX?]
    const int kComputeAtAxis = kTIDyAxis + 1;
    for (auto input : in_tv) {
      for (auto output : out_tv) {
        if (input->getRootDomain().size() == output->getRootDomain().size()) {
          input->computeAt(output, kComputeAtAxis);
        }
      }
    }

    // 4) Find TensorViews to duplicate and computeAt inline
    auto duplicate_tv = findTensorViewsToDuplicate(fusion, other_tv);

    // Any TVs with multiple uses and dependencies with same IterDomain
    // Order of Duplication is necessary for correctness
    for (auto tensor : duplicate_tv) {
      auto result = tensor->duplicate();
      // Add duplicated TVs to Other TVs
      other_tv.insert(other_tv.end(), result.begin(), result.end());
    }

    // 5) Handle Inline-ComputeAt
    auto compute_inline_tv = findTensorViewsToComputeAtInline(fusion, other_tv);
    for (auto tensor : compute_inline_tv) {
      auto uses = fusion->unordered_uses(tensor);
      TORCH_INTERNAL_ASSERT(
          uses.size() == 1,
          "This inline-computeAt TensorView ",
          tensor->name(),
          " is used multiple times.")
      Expr* expr = *uses.begin();
      TensorView* consumer = expr->output(0)->as<TensorView>();
      tensor->computeAt(consumer, -1);
    }

    // 6) Parallel Bindings
    for (auto tv : other_tv) {
      if (outer_axis_exists && rparams.lparams.gdimx() > 1) {
        tv->axis(1)->parallelize(ParallelType::BIDx);
      }

      tv->axis(kBIDyAxis)->parallelize(ParallelType::BIDy);
      tv->axis(kTIDyAxis)->parallelize(ParallelType::TIDy);

      if (tv->nDims() > kComputeAtAxis && rparams.lparams.bdimx() > 1) {
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }

    for (auto tv : reduction_tv) {
      if (outer_axis_exists && rparams.lparams.gdimx() > 1) {
        tv->axis(1)->parallelize(ParallelType::BIDx);
      }

      tv->axis(kBIDyAxis)->parallelize(ParallelType::BIDy);
      tv->axis(kTIDyAxis)->parallelize(ParallelType::TIDy);

      if (tv->nDims() > kComputeAtAxis && rparams.lparams.bdimx() > 1) {
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }

    for (auto tv : rfactor_tv) {
      if (outer_axis_exists && rparams.lparams.gdimx() > 1) {
        tv->axis(1)->parallelize(ParallelType::BIDx);
      }

      tv->axis(kBIDyAxis)->parallelize(ParallelType::BIDy);
      tv->axis(kTIDyAxis)->parallelize(ParallelType::TIDy);

      if (tv->nDims() > kComputeAtAxis && rparams.lparams.bdimx() > 1) {
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }
  } // end non_fastest_dim logic
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
