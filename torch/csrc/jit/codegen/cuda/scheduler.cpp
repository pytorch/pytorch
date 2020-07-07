#include <torch/csrc/jit/codegen/cuda/scheduler.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>

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

      // TODO: could really use evaluation here. Launch configuration is
      //       imposed by transformation and the information should be
      //       embedded in codegen IR.
      // cuda_kernel_->reduction_axes_ = reductionAxes(out);

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
      TensorView* out_tv = static_cast<TensorView*>(output);

      // fcd_reduction could be queried later via
      // cuda_kernel_->reduction_axes_, which would ensure we have proper
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
          static_cast<TensorView*>(inp)->computeAt(intermediate, -1);
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
      if (val->getValType().value() != ValType::TensorView)
        continue;
      TensorView* tv = static_cast<TensorView*>(val);
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
    if (fcd_reduction) {
      // assuming all output to be the same shape;
      fusion->setLaunchConfig(
          LaunchConfigType::TIDx, new Int(kFcdReductionThreadX));
      fusion->setLaunchConfig(LaunchConfigType::TIDy, new Int(1));
      fusion->setLaunchConfig(LaunchConfigType::TIDz, new Int(1));
      fusion->setLaunchConfig(LaunchConfigType::BIDx, numel);
      fusion->setLaunchConfig(LaunchConfigType::BIDy, new Int(1));
      fusion->setLaunchConfig(LaunchConfigType::BIDz, new Int(1));
    } else {
      fusion->setLaunchConfig(
          LaunchConfigType::TIDx, new Int(kNonFcdReductionThreadX));
      fusion->setLaunchConfig(
          LaunchConfigType::TIDy, new Int(kNonFcdReductionThreadY));
      fusion->setLaunchConfig(LaunchConfigType::TIDz, new Int(1));
      fusion->setLaunchConfig(LaunchConfigType::BIDx, numel);
      fusion->setLaunchConfig(LaunchConfigType::BIDy, new Int(1));
      fusion->setLaunchConfig(LaunchConfigType::BIDz, new Int(1));
    }
    fusion->setLaunchConfig(LaunchConfigType::Compatible, new Int(1));
    fusion->setLaunchConfig(LaunchConfigType::SharedMemory, new Int(0));
  } else {
    // Run through outputs, grab all inputs of outputs
    // squeeze with computeAt to set overall structure.
    for (auto output : fusion->outputs()) {
      if (output->getValType() != ValType::TensorView)
        continue;
      TensorView* out_tv = static_cast<TensorView*>(output);

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
      TensorView* out_tv = static_cast<TensorView*>(output);
      for (Val* inp : fusion->inputsOf(output)) {
        if (inp->getValType().value() == ValType::TensorView)
          static_cast<TensorView*>(inp)->computeAt(out_tv, 1);
      }
      out_tv->axis(0)->parallelize(ParallelType::BIDx);
    }

    // Run through all values, unroll, and bind their axes
    for (auto val : fusion->vals()) {
      if (val->getValType().value() != ValType::TensorView)
        continue;
      TensorView* tv = static_cast<TensorView*>(val);

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
    Val* tid_x = new Int(kPwThreadX);
    Val* bid_x = numel;
    fusion->setLaunchConfig(LaunchConfigType::TIDx, tid_x);
    fusion->setLaunchConfig(LaunchConfigType::TIDy, new Int(1));
    fusion->setLaunchConfig(LaunchConfigType::TIDz, new Int(1));
    fusion->setLaunchConfig(LaunchConfigType::BIDx, bid_x);
    fusion->setLaunchConfig(LaunchConfigType::BIDy, new Int(1));
    fusion->setLaunchConfig(LaunchConfigType::BIDz, new Int(1));
    fusion->setLaunchConfig(LaunchConfigType::Compatible, new Int(1));
    fusion->setLaunchConfig(LaunchConfigType::SharedMemory, new Int(0));
  }
  return true;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
