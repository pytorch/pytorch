#include <torch/csrc/jit/codegen/cuda/scheduler/reduction.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

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
} // namespace

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

  int red_elems_per_thread = num_elems_in_reduction;

  int outputs_produced_per_block_iter = 1;

  // Reduction is performed across warp threads (cross-thread reduction)
  if (rparams.fastest_dim) {
    red_elems_per_thread = ceilDiv(red_elems_per_thread, bdimx);
    // Warp threads are applied across the output
  } else {
    outputs_produced_per_block_iter *= bdimx;
  }

  // Decision to do a cross-warp reduction per block
  if (red_elems_per_thread >= (bdimy * kMinValuesPerThread) ||
      red_elems_per_thread >= kMaxValuesPerThread || !rparams.fastest_dim) {
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

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
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

TORCH_CUDA_CU_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    TensorView* red_tv) {
  FUSER_PERF_SCOPE("getReductionHeuristics");

  auto evaluator = executor_utils::bindFusionInputs(fusion_inputs, fusion);

  return getReductionHeuristics(fusion, evaluator, red_tv);
}

TORCH_CUDA_API c10::optional<ReductionParams> getReductionHeuristics(
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

  return reductionHeuristic(
      red_elements, num_outputs_for_reduction, fastest_dim_reduction);
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

  // We coalesce all reduction axes to the right;
  scheduler_utils::mergeReduction(red_tv);

  // Merge all iteration dimensions
  if (red_tv->domain()->domain().size() > 1) {
    scheduler_utils::mergeNonReduction(red_tv);
    for (auto iter_tv : outs_of_red) {
      scheduler_utils::mergeNonReduction(iter_tv);
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

      auto red_tv_rf = scheduler_utils::rfactorHelper(red_tv, {-3, -1});

      scheduler_utils::scheduleReductionComputeAt(
          red_tv, red_tv_rf, outs_of_red);

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

        auto red_tv_rf = scheduler_utils::rfactorHelper(
            red_tv, {-5, -1}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

        scheduler_utils::scheduleReductionComputeAt(
            red_tv, red_tv_rf, outs_of_red);

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

        auto red_tv_rf = scheduler_utils::rfactorHelper(
            red_tv, {-4, -1}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

        scheduler_utils::scheduleReductionComputeAt(
            red_tv, red_tv_rf, outs_of_red);

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

        auto red_tv_rf = scheduler_utils::rfactorHelper(
            red_tv, {-4, -1}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

        scheduler_utils::scheduleReductionComputeAt(
            red_tv, red_tv_rf, outs_of_red);

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

        auto red_tv_rf = scheduler_utils::rfactorHelper(
            red_tv, {-3, -1}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

        scheduler_utils::scheduleReductionComputeAt(
            red_tv, red_tv_rf, outs_of_red);

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

      scheduler_utils::scheduleReductionComputeAt(red_tv, nullptr, outs_of_red);

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

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
