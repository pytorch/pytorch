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
  // maybe has_reduction for scheudling should be done on a per output tensor
  // basis.
  TORCH_INTERNAL_ASSERT(
      !fusion->hasReduction(), "This scheduler only handles pointwise ops.");
  const bool disable_unroll = fusion->isStochastic();

  for (auto out_val : fusion->outputs()) {
    auto out = out_val->as<TensorView>();

    // Merge all dimensions because we're only supporting pointwise
    while (out->nDims() > 1) {
      out->merge(-2, -1);
    }
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

ReductionParams reductionHeuristic(
    int red_elems,
    int red_outputs,
    bool red_on_fastest_dim) {
  ReductionParams rparams;
  rparams.fastest_dim = red_on_fastest_dim;

  int gdimx = LaunchParams::UNINITIALIZED_VAL;
  int gdimy = LaunchParams::UNINITIALIZED_VAL;
  int bdimx = LaunchParams::UNINITIALIZED_VAL;
  int bdimy = LaunchParams::UNINITIALIZED_VAL;

  // 1. Initial Assumptions

  // Evaluate Dimensions of Reduction TensorView
  TORCH_INTERNAL_ASSERT(red_elems > 0 && red_outputs > 0);

  // 2. Initial Definition of Block Dimensions

  // Is fastest dimension a reduction dimension?
  if (rparams.fastest_dim) {
    if (red_elems < rparams.loop_unroll) {
      rparams.loop_unroll = 1;
    }
    bdimx = ceilDiv(red_elems, rparams.loop_unroll);
    bdimy = red_outputs;
  } else {
    bdimx = red_outputs;
    bdimy = red_elems;
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
  int red_elems_per_thread = red_elems;

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
    rparams.mul_reds_per_blk = false;
    // Do multiple reductions per block
  } else {
    rparams.cross_block = false;
    rparams.mul_reds_per_blk = true;
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
  gdimx = ceilDiv(red_outputs, outputs_produced_per_block_iter);

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
              << "\tRed Elems: " << red_elems << " Red Outputs: " << red_outputs
              << " Red On Fastest Dim? " << red_on_fastest_dim << std::endl
              << "Reduction Characteristics:" << std::endl
              << "\tMultiple Reds Per Block? " << rparams.mul_reds_per_blk
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

TORCH_CUDA_CU_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    TensorView* red_tv) {
  FUSER_PERF_SCOPE("scheduleReduction");

  FusionGuard fg(fusion);

  if (!fusion->hasReduction()) {
    return c10::nullopt;
  }

  auto red_root_dom = red_tv->getRootDomain();
  const bool red_on_fastest_dim =
      red_root_dom[red_root_dom.size() - 1]->isReduction();

  TORCH_INTERNAL_ASSERT(
      red_tv != nullptr, "Reduction TensorView wasn't found.");

  if (!fusion->hasReduction()) {
    return c10::nullopt;
  }

  TORCH_INTERNAL_ASSERT(
      red_tv->hasReduction(), "TensorView doesn't have a reduction.");
  const auto red_expr = fusion->origin(red_tv);

  TORCH_INTERNAL_ASSERT(
      red_expr->getExprType() != c10::nullopt &&
          red_expr->getExprType().value() == ExprType::ReductionOp,
      "TensorView doesn't have a reduction.");

  StatefulExpressionEvaluator evaluator(
      executor_utils::statefulBindInputs(fusion_inputs, fusion));

  int64_t red_outputs = 1;
  int64_t red_elements = 1;

  for (auto id : red_tv->getRootDomain()) {
    auto inferred_val = evaluator.inferValue(id->rawExtent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(), "Error inferring reduction size.");
    if (id->isReduction()) {
      red_elements *= inferred_val.value();
    } else {
      red_outputs *= inferred_val.value();
    }
  }

  return reductionHeuristic(red_elements, red_outputs, red_on_fastest_dim);
}

// fusion is the input IR that will be modified by this function
void scheduleReduction(
    Fusion* fusion,
    const ReductionParams& rparams,
    TensorView* red_tv,
    std::vector<TensorView*> outs_of_red) {
  FusionGuard fg(fusion);

  // We coalesc all reduction axes to the right;
  mergeReduction(red_tv);

  // Merge all iteration dimensions
  mergeNonReduction(red_tv);
  for (auto iter_tv : outs_of_red) {
    mergeNonReduction(iter_tv);
  }

  // Evaluate Dimensions of Reduction TensorView
  auto red_ids = red_tv->domain()->domain();

  TORCH_INTERNAL_ASSERT(
      red_ids.size() == 2, "We coalesced all dimensions into 2 previously.");

  constexpr int kLoopUnrollSplit = 4;

  // Scheduling the Reduction
  if (rparams.fastest_dim) {
    // Do multiple reductions per block
    if (rparams.mul_reds_per_blk) {
      // Reduction Splits
      //      [outputs, |rF-Leftover, X-Warp, rf-Unroll|]
      // Idx:     0     |   1(-1)      2(-2)     3(-1) |
      //                --------------------------------
      //                Reduction Dimensions
      red_tv->split(1, rparams.loop_unroll);
      red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));

      // Output Splits
      //      [|Out-Leftover, Out-PerBlock|, <Reduction Dims>]
      // Idx:  |     0             1      |   2(-2) -- 3(-1)
      //       ----------------------------
      //       Output Dimensions
      red_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDy));
      for (auto iter_tv : outs_of_red) {
        iter_tv->split(0, NamedScalar::getParallelDim(ParallelType::TIDy));
      }

      auto red_tv_rf = red_tv->rFactor({-3, -1});

      // WARNING: computeAt will coalesce the rFactored dimensions
      // rFactored Reduction Tensor after computeAt():
      //      [<output dims>, | rF-Leftover, X-Warp, rF-Unroll|]
      // Idx:      0 -- 1     |    2(-3)      3(-2)     4(-1)  |
      //                      ---------------------------------
      //                      Reduction Dimensions
      red_tv_rf->computeAt(red_tv, -1);

      // After the Reduction Tensor has rFactoring applied
      // Reduction Output Tensor:
      //      [Out-Leftover, Out-PerBlock, X-Warp]
      // Idx:       0              1       2(-1)
      if (!outs_of_red.empty()) {
        red_tv->computeAt(outs_of_red[0], -1);
      }

      red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

      red_tv->axis(0)->parallelize(ParallelType::BIDx);
      for (auto iter_tv : outs_of_red) {
        iter_tv->axis(0)->parallelize(ParallelType::BIDx);
      }
      red_tv->axis(1)->parallelize(ParallelType::TIDy);
      for (auto iter_tv : outs_of_red) {
        iter_tv->axis(1)->parallelize(ParallelType::TIDy);
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
        red_tv->split(1, rparams.loop_unroll);
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::BIDy));

        auto red_tv_rf = red_tv->rFactor(
            {-5, -1}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)

        // WARNING: computeAt will coalesce the rFactored dimensions
        // rFactored Reduction Tensor after computeAt():
        //      [Outputs, |X-Grid, X-Block, X-Warp, rF-Leftover, rF-Unroll|]
        // Idx:     0     | 1(-5)    2(-4)   3(-3)      4(-2)      5(-1)  |
        //                -------------------------------------------------
        //                Reduction Dimensions
        red_tv_rf->computeAt(red_tv, -1);

        // After the Reduction Tensor has rFactoring applied
        // Reduction Output Tensor:
        //      [Outputs, X-Grid, X-Block, X-Warp]
        // Idx:     0      1(-3)   2(-2)    3(-1)

        if (!outs_of_red.empty()) {
          red_tv->computeAt(outs_of_red[0], -1);
        }

        red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

        red_tv->axis(0)->parallelize(ParallelType::BIDx);
        for (auto iter_tv : outs_of_red) {
          iter_tv->axis(0)->parallelize(ParallelType::BIDx);
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
        red_tv->split(1, rparams.loop_unroll);
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
        red_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));

        auto red_tv_rf = red_tv->rFactor({-4, -1});

        // WARNING: computeAt will coalesce the rFactored dimensions
        // rFactored Reduction Tensor after computeAt():
        //      [Outputs, |X-Block, X-Warp, rF-Leftover, rF-Unroll|]
        // Idx:     0     | 1(-4)   2(-3)      3(-2)       4(-1)  |
        //                -----------------------------------------
        //                Reduction Dimensions
        red_tv_rf->computeAt(red_tv, -1);

        // After the Reduction Tensor has rFactoring applied
        // Reduction Output Tensor:
        //      [Outputs, X-Block, X-Warp]
        // Idx:     0      1(-2)    2(-1)

        if (!outs_of_red.empty()) {
          red_tv->computeAt(outs_of_red[0], -1);
        }

        red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

        red_tv->axis(0)->parallelize(ParallelType::BIDx);
        for (auto iter_tv : outs_of_red) {
          iter_tv->axis(0)->parallelize(ParallelType::BIDx);
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

        // WARNING: computeAt will coalesce the rFactored dimensions
        // rFactored Reduction Tensor after computeAt():
        //      [<output dims>, |X-Block, X-Grid, rF-Leftover, rF-Unroll|]
        // Idx:      0 -- 1     | 2(-4)   3(-3)      4(-2)       5(-1)  |
        //                      -----------------------------------------
        //                      Reduction Dimensions
        red_tv_rf->computeAt(red_tv, -1);

        // After the Reduction Tensor has rFactoring applied
        // Reduction Output Tensor:
        //      [Out-Leftover, Out-PerBlock, X-Block, X-Grid]
        // Idx:       0              1        2(-2)   3(-1)

        if (!outs_of_red.empty()) {
          red_tv->computeAt(outs_of_red[0], -1);
        }

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

        // WARNING: computeAt will coalesce the rFactored dimensions
        // rFactored Reduction Tensor after computeAt():
        //      [<output dims>, |X-Block, rF-Leftover, rF-Unroll|]
        // Idx:      0 -- 1     | 2(-3)      3(-2)       4(-1)  |
        //                      ---------------------------------
        //                      Reduction Dimensions
        red_tv_rf->computeAt(red_tv, -1);

        // After the Reduction Tensor has rFactoring applied
        // Reduction Output Tensor:
        //      [Out-Leftover, Out-PerBlock, X-Block]
        // Idx:       0              1        2(-1)

        if (!outs_of_red.empty()) {
          red_tv->computeAt(outs_of_red[0], -1);
        }

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

      if (!outs_of_red.empty()) {
        red_tv->computeAt(outs_of_red[0], -1);
      }

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
