#include <torch/csrc/jit/codegen/cuda/scheduler/normalization.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

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
    const int64_t kMaxThreadsPerCTA =
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

    const int64_t kBlockThresholdFastestDim = 1024;
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
    // Ampere - 896
    // Volta - 768
    const int64_t kMaxThreadsPerCTA = 512;
    const int64_t kBlockThresholdNotFastestDim = 64;

    // Setup Block Size
    bdimy = std::min(inner_dim_size, kMaxThreadsPerCTA);
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

  const char* debug_env = getenv("PYTORCH_NVFUSER_RED_SCHED_DEBUG");
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

TORCH_CUDA_CU_API c10::optional<ReductionParams> getNormalizationHeuristics(
    Fusion* fusion,
    ExpressionEvaluator& evaluator,
    const std::vector<TensorView*>& reduction_tv) {
  FusionGuard fg(fusion);
  if (!fusion->hasReduction()) {
    return c10::nullopt;
  }

  // Check Reduction Invariants
  for (auto tv : reduction_tv) {
    TORCH_INTERNAL_ASSERT(tv != nullptr, "Reduction TensorView wasn't found.");
    TORCH_INTERNAL_ASSERT(
        tv->hasReduction(), "TensorView doesn't have a reduction.");
    TORCH_INTERNAL_ASSERT(
        tv->definition()->getExprType() != c10::nullopt &&
            tv->definition()->getExprType().value() == ExprType::ReductionOp,
        "TensorView doesn't have a reduction.");
  }

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

TORCH_CUDA_CU_API c10::optional<ReductionParams> getNormalizationHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    const std::vector<TensorView*>& reduction_tv) {
  FUSER_PERF_SCOPE("scheduleNormalization");

  auto evaluator = executor_utils::bindFusionInputs(fusion_inputs, fusion);

  return getNormalizationHeuristics(fusion, evaluator, reduction_tv);
}

void scheduleNormalization(
    Fusion* fusion,
    const ReductionParams& rparams,
    const std::vector<TensorView*>& reduction_tv,
    std::vector<TensorView*>& other_tv) {
  FusionGuard fg(fusion);

  auto first_reduction_tv = reduction_tv.front();
  const size_t kReductionRootDims = first_reduction_tv->getRootDomain().size();

  const auto& in_tv = ir_utils::filterByType<TensorView>(fusion->inputs());
  const auto& out_tv = ir_utils::filterByType<TensorView>(fusion->outputs());

  if (rparams.fastest_dim && rparams.persistent_kernel) {
    scheduler_utils::cacheInputs(fusion, rparams, reduction_tv, other_tv);
  }

  std::vector<TensorView*> all_tv;
  for (auto input : in_tv) {
    if (input->getRootDomain().size() ==
        reduction_tv.front()->getRootDomain().size()) {
      all_tv.push_back(input);
    }
  }
  all_tv.insert(all_tv.end(), reduction_tv.begin(), reduction_tv.end());
  all_tv.insert(all_tv.end(), other_tv.begin(), other_tv.end());

  scheduler_utils::organizeAxes(reduction_tv, all_tv);

  // For intermediate outputs, apply cache_fork
  for (const auto output : fusion->outputs()) {
    if (!output->uses().empty()) {
      if (output->getValType().value() == ValType::TensorView) {
        other_tv.push_back(output->as<TensorView>()->cache_fork());
      }
    }
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
        //      [outer,   |rf-Unroll, rF-Leftover|]
        // Idx:     0     |   (-2)       (-1)    |
        //                ----------------------
        //                Reduction Dimensions
        tv->split(-1, rparams.loop_unroll, false);

        auto reduction_tv_rf = tv->rFactor({-2});
        rfactor_tv.push_back(reduction_tv_rf);
      }

      // 3) Split the other TensorViews
      for (auto tv : other_tv) {
        if (tv->getRootDomain().size() == kReductionRootDims) {
          if (kHasOuterAxis && rparams.batches_per_block > 1 &&
              rparams.num_warps > 1) {
            tv->split(0, rparams.batches_per_block);
            tv->split(1, rparams.num_warps);
          }
          tv->split(-1, rparams.loop_unroll, false);
        }
      }

      if (kHasOuterAxis) {
        // 4) ComputeAt Structure
        const int kComputeAtAxis = 1;
        for (auto output : out_tv) {
          auto inputs_for_output = fusion->inputsOf(output);
          for (auto input : in_tv) {
            if (inputs_for_output.find(input) != inputs_for_output.end()) {
              input->computeAt(output, kComputeAtAxis);
            }
          }
        }
      }

      // 6) Parallel Binding
      //      [Out-Lft, Out-PerBlock?, Out-NumWarps>|, rf-Unroll,  rF-Lft]
      // Idx: [   0        1              2         |      3         4   ]
      //      [  BIDx      1             TIDy       |      3        TIDx ]
      //      |-------------------------------------|--------------------]
      //                    Outer                         Reduction
      // For all TensorViews
      for (auto tv : other_tv) {
        if (tv->getRootDomain().size() == kReductionRootDims) {
          if (kHasOuterAxis) {
            tv->axis(0)->parallelize(ParallelType::BIDx);
            if (rparams.num_warps > 1) {
              tv->axis(2)->parallelize(ParallelType::TIDy);
            }
          }
          tv->axis(-1)->parallelize(ParallelType::TIDx);
        }
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
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
      // end persistent kernel
    } else {
      // 1) Apply heuristics to each reduction
      std::vector<TensorView*> rfactor_tv;
      for (auto tv : reduction_tv) {
        // Reduction Splits
        //      [ Outer  |, rF-Leftover, rf-Unroll, rf-TDX|]
        // Idx:     0    |     1             2         3  |
        //               ----------------------------------
        //                       Reduction Dimensions
        tv->split(-1, rparams.lparams.bdimx());
        tv->split(-2, rparams.loop_unroll);

        auto reduction_tv_rf = tv->rFactor({-3, -2});
        rfactor_tv.push_back(reduction_tv_rf);
      }

      // 2) Split the other TensorViews
      for (auto tv : other_tv) {
        if (tv->getRootDomain().size() == kReductionRootDims) {
          tv->split(-1, rparams.lparams.bdimx());
          tv->split(-2, rparams.loop_unroll);
        }
      }

      if (kHasOuterAxis) {
        // 3) ComputeAt Structure
        const int kComputeAtAxis = 1;
        for (auto output : out_tv) {
          auto inputs_for_output = fusion->inputsOf(output);
          for (auto input : in_tv) {
            if (inputs_for_output.find(input) != inputs_for_output.end()) {
              input->computeAt(output, kComputeAtAxis);
            }
          }
        }

        // 4) Find TensorViews to duplicate
        auto duplicate_tv =
            scheduler_utils::findTensorViewsToDuplicate(fusion, other_tv);

        // Any TVs with multiple uses and dependencies with same IterDomain
        // Order of Duplication is necessary for correctness
        for (auto tensor : duplicate_tv) {
          auto result = tensor->duplicate();
          other_tv.insert(other_tv.end(), result.begin(), result.end());
        }

        // 5) Handle Inline-ComputeAt
        auto compute_inline_tv =
            scheduler_utils::findTensorViewsToComputeAtInline(fusion, other_tv);
        for (auto tensor : compute_inline_tv) {
          auto uses = tensor->uses();
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
      //      [ outer |, rF-Leftover, rf-Unroll, rf-TDX]
      // Idx: [  BIDx |     1           2         TIDx ]
      //      |-------|--------------------------------]
      //        Outer             Reduction
      // For all TensorViews
      for (auto tv : other_tv) {
        if (tv->getRootDomain().size() == kReductionRootDims) {
          if (kHasOuterAxis) {
            tv->axis(0)->parallelize(ParallelType::BIDx);
          }
          tv->axis(-1)->parallelize(ParallelType::TIDx);
        }
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
        tv->axis(-1)->parallelize(ParallelType::TIDx);
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
      if (tv->getRootDomain().size() == kReductionRootDims) {
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
    const size_t kComputeAtAxis = kTIDyAxis + 1;
    for (auto output : out_tv) {
      auto inputs_for_output = fusion->inputsOf(output);
      for (auto input : in_tv) {
        if (inputs_for_output.find(input) != inputs_for_output.end()) {
          input->computeAt(output, kComputeAtAxis);
        }
      }
    }

    // 4) Find TensorViews to duplicate and computeAt inline
    auto duplicate_tv =
        scheduler_utils::findTensorViewsToDuplicate(fusion, other_tv);

    // Any TVs with multiple uses and dependencies with same IterDomain
    // Order of Duplication is necessary for correctness
    for (auto tensor : duplicate_tv) {
      auto result = tensor->duplicate();
      // Add duplicated TVs to Other TVs
      other_tv.insert(other_tv.end(), result.begin(), result.end());
    }

    // 5) Handle Inline-ComputeAt
    auto compute_inline_tv =
        scheduler_utils::findTensorViewsToComputeAtInline(fusion, other_tv);
    for (auto tensor : compute_inline_tv) {
      auto uses = tensor->uses();
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
      if (tv->getRootDomain().size() == kReductionRootDims) {
        if (outer_axis_exists && rparams.lparams.gdimx() > 1) {
          tv->axis(1)->parallelize(ParallelType::BIDx);
        }

        tv->axis(kBIDyAxis)->parallelize(ParallelType::BIDy);
        tv->axis(kTIDyAxis)->parallelize(ParallelType::TIDy);

        if (tv->nDims() > kComputeAtAxis && rparams.lparams.bdimx() > 1) {
          tv->axis(-1)->parallelize(ParallelType::TIDx);
        }
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

  // If castOp then Broadcast, inline computeAt castOp with BroadcastOp
  for (const auto input : in_tv) {
    if (input->getRootDomain().size() != kReductionRootDims) {
      scheduler_utils::handleCastBroadcastInput(fusion, input);
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
