#include <torch/csrc/jit/codegen/cuda/scheduler.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

constexpr int kUnrollFactor = 4;

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

// coalesces all reduction to the right side and returns total number of
// reduction axes
size_t coalescReduction(TensorView* tv) {
  auto reduction_axes = reductionAxes(tv);
  size_t n_dims = tv->nDims();
  std::unordered_map<int, int> coalesc_permute;
  for (size_t i = 0; i < reduction_axes.size(); i++) {
    size_t new_pos = i + n_dims - reduction_axes.size();
    if (new_pos == reduction_axes[i]) {
      break;
    } else {
      coalesc_permute[reduction_axes[i]] = new_pos;
    }
  }
  if (!coalesc_permute.empty()) {
    tv->reorder(coalesc_permute);
  }
  return reduction_axes.size();
}

} // namespace

// This one is a total mess and it should go.
bool scheduleFusion(Fusion* fusion, const at::ArrayRef<c10::IValue> inputs) {
  FusionGuard fg(fusion);
  // maybe has_reduction for scheudling should be done on a per output tensor
  // basis.
  const bool has_reduction = fusion->hasReduction();
  const bool disable_unroll = fusion->hasRNG();
  bool fcd_reduction = false;

  for (auto out_val : fusion->outputs()) {
    auto out = out_val->as<TensorView>();
    if (has_reduction) {
      // TODO: this scheduling only works for a single reduction operation in
      //       the fusion, in this case we can coalesc all reduction axes and
      //       merge them together. (same applies to iteration axes)
      // TODO: does this work for multiple outputs?

      // query if fastest changing dimension (FCD) is a reduction
      fcd_reduction = out->axis((int)out->nDims() - 1)->isReduction();

      // We coalesc all reduction axes to the right;
      size_t num_reduction_axes = coalescReduction(out);

      // Merge all iteration dimensions
      while (out->nDims() > num_reduction_axes + 1) {
        out->merge(0, 1);
      }
      // Merge all reduction dimensions
      while (out->nDims() > 2) {
        out->merge(1, 2);
      }
    } else {
      // Merge all dimensions because we're only supporting pointwise
      while (out->nDims() > 1)
        out->merge(0, 1);
    }
  }

  if (has_reduction) {
    // Run through outputs, grab all inputs of outputs
    // squeeze with computeAt to set overall structure.
    for (auto output : fusion->outputs()) {
      if (output->getValType() != ValType::TensorView)
        continue;
      TensorView* out_tv = output->as<TensorView>();

      // launch configuratoin.
      TensorView* intermediate = nullptr;
      if (fcd_reduction) {
        out_tv->split(-1, kFcdReductionThreadX);
        // necessary to avoid dynamic allocation on intermediates;
        intermediate = out_tv->rFactor({-2});
      } else {
        // TODO: we don't need a full warp here, this should be determined by
        //       element data type
        out_tv->split(0, kNonFcdReductionThreadX);
        out_tv->split(
            -1, kNonFcdReductionThreadY); // necessary to avoid dynamic
                                          // allocation on intermediates;
        intermediate = out_tv->rFactor({-2});
      }
      for (Val* inp : fusion->inputsOf(output)) {
        // scheduling of inputs shouldn't change with different fcd_reduction
        if (inp->getValType().value() == ValType::TensorView) {
          inp->as<TensorView>()->computeAt(intermediate, -1);
        }
      }
      // scheduling of inputs shouldn't change with different fcd_reduction
      intermediate->computeAt(out_tv, -2);
      if (fcd_reduction) {
        out_tv->axis(0)->parallelize(ParallelType::BIDx);
      } else {
        out_tv->axis(0)->parallelize(ParallelType::BIDx);
        out_tv->axis(1)->parallelize(ParallelType::TIDx);
      }
    }
    // Run through all values, unroll, and bind their axes
    for (auto val : fusion->vals()) {
      if (val->getValType().value() != ValType::TensorView ||
          fusion->hasInput(val))
        continue;
      TensorView* tv = val->as<TensorView>();
      if (fcd_reduction) {
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      } else {
        tv->axis(-1)->parallelize(ParallelType::TIDy);
      }
    }

    TensorView* out0 = fusion->outputs()[0]->as<TensorView>();
    int ndim = (int)out0->nDims();
    Val* numel = new Int(1);
    for (int i = 0; i < ndim; i++) {
      if (out0->axis(i)->isBlockDim()) {
        numel = mul(numel, out0->axis(i)->rawExtent());
      }
    }
  } else {
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
      if (!disable_unroll) {
        out_tv->split(0, ur_factor);
      }
    }

    for (auto output : fusion->outputs()) {
      if (output->getValType() != ValType::TensorView)
        continue;
      TensorView* out_tv = output->as<TensorView>();
      for (Val* inp : fusion->inputsOf(output)) {
        if (inp->getValType().value() == ValType::TensorView)
          inp->as<TensorView>()->computeAt(out_tv, 1);
      }
      out_tv->axis(0)->parallelize(ParallelType::BIDx);
    }

    // Run through all values, unroll, and bind their axes
    for (auto val : fusion->vals()) {
      if (val->getValType().value() != ValType::TensorView ||
          fusion->hasInput(val))
        continue;
      TensorView* tv = val->as<TensorView>();

      // Should be true for all intermediates, but if one isn't hooked
      // up right, skip it and hope for the best for now
      if (!disable_unroll && tv->nDims() == 3) {
        tv->axis(-2)->parallelize(ParallelType::Unroll);
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      } else {
        if (tv->nDims() == 2)
          tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }
    TensorView* out0 = fusion->outputs()[0]->as<TensorView>();
    int ndim = (int)out0->nDims();
    Val* numel = new Int(1);
    for (int i = 0; i < ndim; i++) {
      if (out0->axis(i)->isBlockDim()) {
        numel = mul(numel, out0->axis(i)->rawExtent());
      }
    }
  }
  return true;
}

namespace {
constexpr int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

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
  rparams.fastest_dim_ = red_on_fastest_dim;

  // 1. Initial Assumptions

  // Evaluate Dimensions of Reduction TensorView
  TORCH_INTERNAL_ASSERT(red_elems > 0 && red_outputs > 0);
  int red_inputs = red_elems * red_outputs;

  // 2. Initial Definition of Block Dimensions

  // Is fastest dimension a reduction dimension?
  if (rparams.fastest_dim_) {
    rparams.block_dim_x_ = red_elems;
    rparams.block_dim_y_ = red_outputs;
  } else {
    rparams.block_dim_x_ = red_outputs;
    rparams.block_dim_y_ = red_elems;
  }

  // 3. Applying Power of 2 Blocking based on the Maximum Number of threads

  constexpr int kMaxNumThreads = 512;
  int num_threads = kMaxNumThreads;
  int device_warp_size = at::cuda::warp_size();

  if (rparams.block_dim_x_ < num_threads) {
    rparams.block_dim_x_ = lastPow2(rparams.block_dim_x_);
  } else {
    rparams.block_dim_x_ = num_threads;
  }

  if (rparams.block_dim_y_ < num_threads) {
    rparams.block_dim_y_ = lastPow2(rparams.block_dim_y_);
  } else {
    rparams.block_dim_y_ = num_threads;
  }

  int block_dim_x_prev = rparams.block_dim_x_;
  rparams.block_dim_x_ = std::min(rparams.block_dim_x_, device_warp_size);
  rparams.block_dim_y_ =
      std::min(rparams.block_dim_y_, num_threads / rparams.block_dim_x_);
  rparams.block_dim_x_ =
      std::min(block_dim_x_prev, num_threads / rparams.block_dim_y_);

  // 4. Distributing work across a block

  // Magic numbers of calculations allowed per thread.
  constexpr int kMinValuesPerThread = 16;
  constexpr int kMaxValuesPerThread = 256;

  int inputs_consumed_per_block_iter = 1;
  int red_elems_per_thread = red_elems;

  int outputs_produced_per_block_iter = 1;

  // Reduction is performed across warp threads (cross-thread reduction)
  if (rparams.fastest_dim_) {
    inputs_consumed_per_block_iter *= rparams.block_dim_x_;
    red_elems_per_thread =
        ceilDiv(red_elems_per_thread, inputs_consumed_per_block_iter);
    // Warp threads are applied across the output
  } else {
    outputs_produced_per_block_iter *= rparams.block_dim_x_;
  }

  // Decision to do a cross-warp reduction per block
  if (red_elems_per_thread >= (rparams.block_dim_y_ * kMinValuesPerThread) ||
      red_elems_per_thread >= kMaxValuesPerThread || !rparams.fastest_dim_) {
    inputs_consumed_per_block_iter *= rparams.block_dim_y_;
    red_elems_per_thread = ceilDiv(red_elems_per_thread, rparams.block_dim_y_);
    rparams.cross_warp_ = true;
    rparams.mul_reds_per_blk_ = false;
    // Do multiple reductions per block
  } else {
    rparams.cross_warp_ = false;
    rparams.mul_reds_per_blk_ = true;
    outputs_produced_per_block_iter *= rparams.block_dim_y_;
  }

  // 5. Distributing work across blocks

  // WARNING: Current device for codegen may not be the target device
  int device_max_threads_per_multiprocessor =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
  int device_multiprocessor_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  int blocks_per_sm = device_max_threads_per_multiprocessor /
      (rparams.block_dim_x_ * rparams.block_dim_y_);
  int target_grid_size = device_multiprocessor_count * blocks_per_sm;

  // Setting the number of blocks based on the number of outputs
  rparams.grid_dim_x_ = ceilDiv(red_outputs, outputs_produced_per_block_iter);

  // Cross-block reductions (if necessary)
  if (rparams.cross_warp_ && red_elems_per_thread >= kMaxValuesPerThread &&
      rparams.grid_dim_x_ <= target_grid_size) {
    int blks_per_out_1 = ceilDiv(target_grid_size, rparams.grid_dim_x_);
    int blks_per_out_2 = ceilDiv(red_elems_per_thread, kMinValuesPerThread);
    int blks_per_out_3 = ceilDiv(red_elems_per_thread, kMaxValuesPerThread);
    int blks_per_output =
        std::max(std::min(blks_per_out_1, blks_per_out_2), blks_per_out_3);

    rparams.grid_dim_y_ = std::max(1, blks_per_output);
    // If a cross-block reduction was generated
    if (blks_per_output > 1) {
      rparams.cross_block_ = true;
    }
  }

  const char* debug_env = getenv("PYTORCH_CUDA_FUSER_RED_SCHED_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n===== Reduction Parameters ========" << std::endl
              << "Inputs:" << std::endl
              << "\tRed Elems: " << red_elems << " Red Outputs: " << red_outputs
              << " Red On Fastest Dim? " << red_on_fastest_dim << std::endl
              << "Reduction Characteristics:" << std::endl
              << "\tMultiple Reds Per Block? " << rparams.mul_reds_per_blk_
              << " Cross Warp? " << rparams.cross_warp_ << " Cross Block? "
              << rparams.cross_block_ << std::endl
              << "Recommended Blocking:" << std::endl
              << "\tGridX: " << rparams.grid_dim_x_
              << " GridY: " << rparams.grid_dim_y_
              << " BlckX: " << rparams.block_dim_x_
              << " BlckY: " << rparams.block_dim_y_ << std::endl
              << "====================================" << std::endl;
  }

  return rparams;
}
} // anonymous namespace

// fusion is the input IR that will be modified by this function
c10::optional<ReductionParams> scheduleReduction(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    TensorView* red_tv) {
  FusionGuard fg(fusion);

  if (!fusion->hasReduction()) {
    return c10::nullopt;
  }
  TORCH_INTERNAL_ASSERT(
      red_tv != nullptr, "Reduction TensorView wasn't found.");
  TORCH_INTERNAL_ASSERT(
      red_tv->hasReduction(), "TensorView doesn't have a reduction.");

  const auto red_expr = fusion->origin(red_tv);

  TORCH_INTERNAL_ASSERT(
      red_expr->getExprType() != c10::nullopt &&
          red_expr->getExprType().value() == ExprType::ReductionOp,
      "TensorView doesn't have a reduction.");

  const bool red_on_fastest_dim =
      red_tv->axis(static_cast<int>(red_tv->nDims()) - 1)->isReduction();

  // We coalesc all reduction axes to the right;
  const size_t num_reduction_axes = coalescReduction(red_tv);

  // Merge all iteration dimensions
  while (red_tv->nDims() > num_reduction_axes + 1) {
    red_tv->merge(0, 1);
  }
  // Merge all reduction dimensions
  while (red_tv->nDims() > 2) {
    red_tv->merge(1, 2);
  }

  EvaluationContext eval_context(
      std::move(executor_utils::bindInputs(fusion_inputs, fusion)));

  // Evaluate Dimensions of Reduction TensorView
  auto red_ids = red_tv->domain()->domain();
  TORCH_INTERNAL_ASSERT(
      red_ids.size() == 2, "We coalesced all dimensions into 2 previously.");
  const auto red_outputs =
      ExpressionEvaluator::evaluate(red_ids[0]->extent(), &eval_context);
  const auto red_elems =
      ExpressionEvaluator::evaluate(red_ids[1]->extent(), &eval_context);
  TORCH_INTERNAL_ASSERT(
      red_outputs != c10::nullopt,
      "The number of reduction outputs is expected.");
  TORCH_INTERNAL_ASSERT(
      red_elems != c10::nullopt,
      "The number of reduction elements is expected.");

  ReductionParams rparams = reductionHeuristic(
      red_elems.value(), red_outputs.value(), red_on_fastest_dim);

  // Heuristic Definition
  if (rparams.fastest_dim_) {
    // Do multiple reductions per block
    if (rparams.mul_reds_per_blk_) {
      // Unroll a certain number of rFactored elements
      red_tv->split(1, 4);
      red_tv->split(1, rparams.block_dim_x_);
      // Unroll a certain number of rFactored elements
      // Split Grid dimension to get multiple reds per block
      red_tv->split(0, rparams.block_dim_y_);

      auto red_tv_rf = red_tv->rFactor({-3, -1});
      red_tv_rf->computeAt(red_tv, 1);

      red_tv->axis(0)->parallelize(ParallelType::BIDx);
      red_tv->axis(1)->parallelize(ParallelType::TIDy);
      red_tv->axis(-1)->parallelize(ParallelType::TIDx);

      red_tv_rf->axis(1)->parallelize(ParallelType::TIDy);
      red_tv_rf->axis(-2)->parallelize(ParallelType::TIDx);
      red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

      Val* input = fusion->origin(red_tv_rf)->as<ReductionOp>()->in();
      if (!fusion->hasInput(input)) {
        input->as<TensorView>()->computeAt(red_tv_rf, -2);
        input->as<TensorView>()->axis(-1)->parallelize(ParallelType::Unroll);
      }
      // Do a cross-warp reduction per block
    } else {
      if (rparams.cross_block_) {
        red_tv->split(1, 4);
        red_tv->split(1, rparams.block_dim_x_);
        // Split up rFactor to reduce across warps
        red_tv->split(1, rparams.grid_dim_y_);
        red_tv->split(1, rparams.block_dim_y_);

        auto red_tv_rf = red_tv->rFactor(
            {-5, -1}); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
        red_tv_rf->computeAt(red_tv, 1);

        red_tv->axis(0)->parallelize(ParallelType::BIDx);

        // Cross-block reduction binding
        red_tv_rf->axis(-4)->parallelize(ParallelType::BIDy);
        red_tv_rf->axis(-3)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(-2)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

        red_tv->axis(-3)->parallelize(ParallelType::BIDy);
        red_tv->axis(-2)->parallelize(ParallelType::TIDy);
        red_tv->axis(-1)->parallelize(ParallelType::TIDx);

        Val* input = fusion->origin(red_tv_rf)->as<ReductionOp>()->in();
        if (!fusion->hasInput(input)) {
          input->as<TensorView>()->computeAt(red_tv_rf, -2);
          input->as<TensorView>()->axis(-1)->parallelize(ParallelType::Unroll);
        }
      } else {
        red_tv->split(1, 4);
        red_tv->split(1, rparams.block_dim_x_);
        // Split up rFactor to reduce across warps
        red_tv->split(1, rparams.block_dim_y_);

        auto red_tv_rf = red_tv->rFactor({-4, -1});
        red_tv_rf->computeAt(red_tv, 1);

        red_tv->axis(0)->parallelize(ParallelType::BIDx);

        red_tv_rf->axis(-3)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(-2)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

        red_tv->axis(-2)->parallelize(ParallelType::TIDy);
        red_tv->axis(-1)->parallelize(ParallelType::TIDx);

        Val* input = fusion->origin(red_tv_rf)->as<ReductionOp>()->in();
        if (!fusion->hasInput(input)) {
          input->as<TensorView>()->computeAt(red_tv_rf, -2);
          input->as<TensorView>()->axis(-1)->parallelize(ParallelType::Unroll);
        }
      }
    }
  } else {
    if (rparams.cross_warp_) {
      if (rparams.cross_block_) {
        red_tv->split(1, 4);
        red_tv->split(1, rparams.grid_dim_y_);
        red_tv->split(1, rparams.block_dim_y_);
        red_tv->split(0, rparams.block_dim_x_);
        auto red_tv_rf = red_tv->rFactor({-4, -1});

        // Bindings
        red_tv_rf->axis(1)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(0)->parallelize(ParallelType::BIDx);
        red_tv_rf->axis(-3)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(-2)->parallelize(ParallelType::BIDy);
        red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

        red_tv->axis(1)->parallelize(ParallelType::TIDx);
        red_tv->axis(0)->parallelize(ParallelType::BIDx);
        red_tv->axis(-1)->parallelize(ParallelType::BIDy);
        red_tv->axis(-2)->parallelize(ParallelType::TIDy);

        Val* input = fusion->origin(red_tv_rf)->as<ReductionOp>()->in();
        if (!fusion->hasInput(input)) {
          input->as<TensorView>()->computeAt(red_tv_rf, -2);
          input->as<TensorView>()->axis(-1)->parallelize(ParallelType::Unroll);
        }
      } else {
        red_tv->split(1, 4);
        red_tv->split(1, rparams.block_dim_y_);
        red_tv->split(0, rparams.block_dim_x_);
        auto red_tv_rf = red_tv->rFactor({-3, -1});

        // Bindings
        red_tv_rf->axis(1)->parallelize(ParallelType::TIDx);
        red_tv_rf->axis(0)->parallelize(ParallelType::BIDx);
        red_tv_rf->axis(-2)->parallelize(ParallelType::TIDy);
        red_tv_rf->axis(-1)->parallelize(ParallelType::Unroll);

        red_tv->axis(1)->parallelize(ParallelType::TIDx);
        red_tv->axis(0)->parallelize(ParallelType::BIDx);
        red_tv->axis(-1)->parallelize(ParallelType::TIDy);

        Val* input = fusion->origin(red_tv_rf)->as<ReductionOp>()->in();
        if (!fusion->hasInput(input)) {
          input->as<TensorView>()->computeAt(red_tv_rf, -2);
          input->as<TensorView>()->axis(-1)->parallelize(ParallelType::Unroll);
        }
      }
    } else {
      red_tv->split(0, rparams.block_dim_x_);
      red_tv->axis(0)->parallelize(ParallelType::TIDx);
      red_tv->axis(1)->parallelize(ParallelType::BIDx);
    }
  }

  return rparams;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
