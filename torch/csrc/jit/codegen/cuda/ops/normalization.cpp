#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ops/normalization.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TensorView* softmax(TensorView* x, int dim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");

  const int kNumberOfDims =
      TensorDomain::noReductions(x->getRootDomain()).size();
  const int kReductionAxis = (dim < 0) ? dim + kNumberOfDims : dim;
  TORCH_INTERNAL_ASSERT(kReductionAxis >= 0 && kReductionAxis < kNumberOfDims);

  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  broadcast_mask[kReductionAxis] = true;

  auto max_val = max(x, {kReductionAxis});
  auto bcast_max = broadcast(max_val, broadcast_mask);
  auto x_max_sub = sub(x, bcast_max);
  auto exp = unaryOp(UnaryOpType::Exp, x_max_sub);
  auto sum_exp = sum(exp, {kReductionAxis});
  auto bcast_sum = broadcast(sum_exp, broadcast_mask);
  auto y = div(exp, bcast_sum);

  return y;
}

TensorView* softmax_backward(
    TensorView* dy,
    TensorView* y,
    int dim,
    TensorView* x) {
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(y != nullptr, "Output is invalid.");
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");

  const int kNumberOfDims =
      TensorDomain::noReductions(x->getRootDomain()).size();
  const int kReductionAxis = (dim < 0) ? dim + kNumberOfDims : dim;
  TORCH_INTERNAL_ASSERT(kReductionAxis >= 0 && kReductionAxis < kNumberOfDims);

  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  broadcast_mask[kReductionAxis] = true;

  auto new_grad = mul(dy, y);
  auto sum_new_grad = sum(new_grad, {kReductionAxis});
  auto bcast_sum = broadcast(sum_new_grad, broadcast_mask);
  auto output_sum_mul = mul(y, bcast_sum);
  auto dx = sub(new_grad, output_sum_mul);

  return dx;
}

ForwardNormResult layer_norm(
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* weight,
    TensorView* bias,
    Val* eps) {
  return layer_norm(x, norm_shape.size(), weight, bias, eps);
}

ForwardNormResult layer_norm(
    TensorView* x,
    const size_t kNormShapeNumDims,
    TensorView* weight,
    TensorView* bias,
    Val* eps) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(
      eps != nullptr && eps->getDataType().has_value() &&
          eps->getDataType().value() == DataType::Double,
      "Epsilon (eps) is not a valid Double.");

  // (B, C, H, W, D) tensor
  // norm_shape = [H, W, D]
  // M = outer = product of remaining dimensions = B * C
  // N = reduction = product of norm_shape = H * W * D
  // weight = bias = norm_shape tensor
  const size_t kNumberOfDims =
      TensorDomain::noReductions(x->getRootDomain()).size();
  const size_t kOuterNumDims = kNumberOfDims - kNormShapeNumDims;

  std::vector<int> outer_reduction_axes(kOuterNumDims);
  std::vector<bool> outer_broadcast_mask(kNumberOfDims, false);
  for (size_t idx = 0; idx < kOuterNumDims; ++idx) {
    outer_reduction_axes[idx] = idx;
    outer_broadcast_mask[idx] = true;
  }

  std::vector<int> inner_reduction_axes(kNormShapeNumDims);
  std::vector<bool> inner_broadcast_mask(kNumberOfDims, false);
  Val* num_features = new Double(1);
  for (size_t idx = 0; idx < kNormShapeNumDims; ++idx) {
    const size_t axis = kNumberOfDims - 1 - idx;
    inner_reduction_axes[idx] = axis;
    inner_broadcast_mask[axis] = true;
    num_features = mul(num_features, x->domain()->domain()[axis]->extent());
  }

  // Main algorithm
  auto welford_out = Welford(x, inner_reduction_axes);
  auto mean_bcast = broadcast(welford_out.avg, inner_broadcast_mask);
  auto x_sub_mean = sub(x, mean_bcast);

  auto var_sum_bcast = broadcast(welford_out.var_sum, inner_broadcast_mask);
  auto var = div(var_sum_bcast, num_features);
  auto var_eps = add(var, eps);
  auto invstd = unaryOp(UnaryOpType::Rsqrt, var_eps);

  auto y = mul(x_sub_mean, invstd);

  // Optional: norm * weight
  if (weight != nullptr) {
    auto weight_bcast = broadcast(weight, outer_broadcast_mask);
    y = mul(y, weight_bcast);
  }

  // Optional: norm * weight + bias
  if (bias != nullptr) {
    auto bias_bcast = broadcast(bias, outer_broadcast_mask);
    y = add(y, bias_bcast);
  }

  return {y, mean_bcast, invstd};
}

BackwardNormResult layer_norm_backward(
    TensorView* dy,
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* mean,
    TensorView* invstd,
    TensorView* weight,
    TensorView* bias,
    const std::vector<bool>& output_mask) {
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(mean != nullptr, "Mean is invalid.");
  TORCH_INTERNAL_ASSERT(invstd != nullptr, "Inv std is invalid.");

  // (B, C, H, W, D) tensor
  // norm_shape = [H, W, D]
  // M = outer = product of remaining dimensions = B * C
  // N = reduction = product of norm_shape = H * W * D
  // weight = bias = norm_shape tensor
  const size_t kNumberOfDims =
      TensorDomain::noReductions(x->getRootDomain()).size();
  const size_t kNormShapeNumDims = norm_shape.size();
  const size_t kOuterNumDims = kNumberOfDims - kNormShapeNumDims;

  std::vector<int> outer_reduction_axes(kOuterNumDims);
  std::vector<bool> outer_broadcast_mask(kNumberOfDims, false);
  for (size_t idx = 0; idx < kOuterNumDims; ++idx) {
    outer_reduction_axes[idx] = idx;
    outer_broadcast_mask[idx] = true;
  }

  std::vector<int> inner_reduction_axes(kNormShapeNumDims);
  std::vector<bool> inner_broadcast_mask(kNumberOfDims, false);
  Val* num_features = new Double(1);
  for (size_t idx = 0; idx < kNormShapeNumDims; ++idx) {
    const size_t axis = kNumberOfDims - 1 - idx;
    inner_reduction_axes[idx] = axis;
    inner_broadcast_mask[axis] = true;
    num_features = mul(num_features, x->domain()->domain()[axis]->extent());
  }

  auto x_hat = mul(sub(x, mean), invstd);

  TensorView* grad_x_hat = nullptr;
  if (weight != nullptr) {
    auto* bcast_weight = broadcast(weight, outer_broadcast_mask);
    grad_x_hat = mul(dy, bcast_weight);
  } else {
    grad_x_hat = dy;
  }

  auto a = mul(num_features, grad_x_hat);

  auto b = sum(grad_x_hat, inner_reduction_axes);
  auto bcast_b = broadcast(b, inner_broadcast_mask);

  auto c1 = mul(grad_x_hat, x_hat);
  auto c2 = sum(c1, inner_reduction_axes);
  auto bcast_c2 = broadcast(c2, inner_broadcast_mask);
  auto c3 = mul(x_hat, bcast_c2);

  auto inner = sub(sub(a, bcast_b), c3);
  auto reciprocal_size = unaryOp(UnaryOpType::Reciprocal, num_features);

  TensorView* dx = nullptr;
  if (output_mask[0]) {
    dx = mul(mul(reciprocal_size, invstd), inner);
  }

  TensorView* dw = nullptr;
  if (output_mask[1] && weight != nullptr) {
    dw = sum(mul(dy, x_hat), outer_reduction_axes);
  }

  TensorView* db = nullptr;
  if (output_mask[2] && bias != nullptr) {
    db = sum(dy, outer_reduction_axes);
  }
  return {dx, dw, db};
}

ForwardNormResult batch_norm(
    TensorView* x,
    TensorView* weight,
    TensorView* bias,
    TensorView* running_mean,
    TensorView* running_var,
    const bool kTraining,
    Val* momentum,
    Val* eps) {
  auto fusion = FusionGuard::getCurFusion();

  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");

  TORCH_INTERNAL_ASSERT(
      !((running_var == nullptr) ^ (running_mean == nullptr)),
      "running stats should comes in pairs");

  TORCH_INTERNAL_ASSERT(
      momentum != nullptr && momentum->getDataType().has_value() &&
          momentum->getDataType().value() == DataType::Double,
      "Momentum is not a valid Double.");

  TORCH_INTERNAL_ASSERT(
      eps != nullptr && eps->getDataType().has_value() &&
          eps->getDataType().value() == DataType::Double,
      "Epsilon (eps) is not a valid Double.");

  // (B, C, H, W, D) tensor
  // M = outer = channels
  // N = reduction = B * H * W * D
  // weight = bias = (C) tensor
  // const size_t kChannelsDim = 1;
  const size_t kNumberOfDims =
      TensorDomain::noReductions(x->getRootDomain()).size();

  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  Val* num_features = new Double(1);
  for (size_t axis = 0; axis < kNumberOfDims; ++axis) {
    if (axis != 1) {
      reduction_axes.push_back(axis);
      broadcast_mask[axis] = true;
      num_features = mul(num_features, x->domain()->domain()[axis]->extent());
    }
  }

  TensorView* y = nullptr;
  TensorView* mean = nullptr;
  TensorView* invstd = nullptr;
  if (kTraining || running_mean == nullptr) {
    // Algorithm
    auto welford_out = Welford(x, reduction_axes);

    // updating running mean and running var
    if (running_mean != nullptr && running_var != nullptr) {
      auto rev_momentum = sub(new Double(1.0), momentum);
      auto current_mean_hat = mul(welford_out.avg, momentum);
      auto mean_hat = mul(running_mean, rev_momentum);
      auto new_mean_hat = add(mean_hat, current_mean_hat);
      fusion->addOutput(new_mean_hat);
      fusion->aliasOutputToInput(new_mean_hat, running_mean);

      auto num_feature_decrement = sub(num_features, new Int(1));
      auto unbiased_var = div(welford_out.var_sum, num_feature_decrement);
      auto current_var_hat = mul(unbiased_var, momentum);
      auto var_hat = mul(running_var, rev_momentum);
      auto new_var_hat = add(var_hat, current_var_hat);
      fusion->addOutput(new_var_hat);
      fusion->aliasOutputToInput(new_var_hat, running_var);
    }

    mean = welford_out.avg;
    auto mean_bcast = broadcast(mean, broadcast_mask);
    auto x_sub_mean = sub(x, mean_bcast);

    auto var = div(welford_out.var_sum, num_features);
    auto var_eps = add(var, eps);
    invstd = unaryOp(UnaryOpType::Rsqrt, var_eps);
    auto invstd_bcast = broadcast(invstd, broadcast_mask);

    y = mul(x_sub_mean, invstd_bcast);
  } else {
    // This is inference mode with running stats
    auto r_mean_bcasted = broadcast(running_mean, broadcast_mask);
    auto x_sub_mean = sub(x, r_mean_bcasted);

    auto var_eps = add(running_var, eps);
    auto unbiased_invstd = unaryOp(UnaryOpType::Rsqrt, var_eps);
    auto invstd_bcast = broadcast(unbiased_invstd, broadcast_mask);

    // During inference, mean/invstd output are empty tensors
    mean = TensorViewBuilder().shape({0}).build();
    invstd = TensorViewBuilder().shape({0}).build();
    y = mul(x_sub_mean, invstd_bcast);
  }

  // Optional: norm * weight
  if (weight) {
    auto weight_bcast = broadcast(weight, broadcast_mask);
    y = mul(y, weight_bcast);
  }

  // Optional: norm * weight + bias
  if (bias) {
    auto bias_bcast = broadcast(bias, broadcast_mask);
    y = add(y, bias_bcast);
  }
  return {y, mean, invstd};
}

BackwardNormResult batch_norm_backward(
    TensorView* x,
    TensorView* dy,
    TensorView* weight,
    TensorView* running_mean,
    TensorView* running_var,
    TensorView* save_mean,
    TensorView* save_invstd,
    const bool kTraining,
    Val* eps,
    const std::vector<bool>& output_mask) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(
      eps != nullptr && eps->getDataType().has_value() &&
          eps->getDataType().value() == DataType::Double,
      "Epsilon (eps) is not a valid Double.");

  // (B, C, H, W, D) tensor
  // M = outer = channels
  // N = reduction = B * H * W * D
  // weight = bias = (C) tensor
  const size_t kChannelsDim = 1;
  const size_t kNumberOfDims =
      TensorDomain::noReductions(x->getRootDomain()).size();

  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  Val* num_features = new Double(1);
  for (size_t axis = 0; axis < kNumberOfDims; ++axis) {
    if (axis != kChannelsDim) {
      reduction_axes.push_back(axis);
      broadcast_mask[axis] = true;
      num_features = mul(num_features, x->domain()->domain()[axis]->extent());
    }
  }

  Val* bcast_weight = nullptr;
  if (weight != nullptr) {
    bcast_weight = broadcast(weight, broadcast_mask);
  } else {
    bcast_weight = new Double(1);
  }

  TensorView* dx = nullptr;
  TensorView* dw = nullptr;
  TensorView* db = nullptr;
  if (kTraining) {
    TORCH_INTERNAL_ASSERT(
        save_mean != nullptr && save_invstd != nullptr,
        "When training=True, save_mean and save_invstd are required.");

    auto bcast_rstd = broadcast(save_invstd, broadcast_mask);
    auto bcast_mean = broadcast(save_mean, broadcast_mask);
    auto x_hat = mul(sub(x, bcast_mean), bcast_rstd);
    auto grad_x_hat = mul(dy, bcast_weight);

    auto a = mul(num_features, grad_x_hat);

    auto b = sum(grad_x_hat, reduction_axes);
    auto bcast_b = broadcast(b, broadcast_mask);

    auto c1 = mul(grad_x_hat, x_hat);
    auto c2 = sum(c1, reduction_axes);
    auto bcast_c2 = broadcast(c2, broadcast_mask);
    auto c3 = mul(x_hat, bcast_c2);

    auto inner = sub(sub(a, bcast_b), c3);

    auto reciprocal_size = unaryOp(UnaryOpType::Reciprocal, num_features);

    if (output_mask[0]) {
      dx = mul(mul(reciprocal_size, bcast_rstd), inner);
    }

    if (output_mask[1]) {
      dw = sum(mul(dy, x_hat), reduction_axes);
    }
  } else {
    // TODO: this is not a legit assumption? Can't we run with
    // track_running_stats == false && training == false
    // which should just run through the case above.
    TORCH_INTERNAL_ASSERT(
        running_mean != nullptr && running_var != nullptr,
        "When training=False, running_mean and running_invstd are required.");

    auto bcast_var = broadcast(running_var, broadcast_mask);
    auto var_eps = add(bcast_var, eps);
    auto bcast_rstd = unaryOp(UnaryOpType::Rsqrt, var_eps);
    auto bcast_mean = broadcast(running_mean, broadcast_mask);

    if (output_mask[0]) {
      dx = mul(mul(dy, bcast_rstd), bcast_weight);
    }

    if (output_mask[1]) {
      auto x_hat = mul(sub(x, bcast_mean), bcast_rstd);
      dw = sum(mul(dy, x_hat), reduction_axes);
    }
  }

  if (output_mask[2]) {
    db = sum(dy, reduction_axes);
  }

  return {dx, dw, db};
}

ForwardNormResult instance_norm(
    TensorView* x,
    TensorView* weight,
    TensorView* bias,
    TensorView* running_mean,
    TensorView* running_var,
    const bool kUseInputStats,
    Val* momentum,
    Val* eps) {
  auto fusion = FusionGuard::getCurFusion();

  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");

  TORCH_INTERNAL_ASSERT(
      !((running_var == nullptr) ^ (running_mean == nullptr)),
      "running stats should comes in pairs");

  TORCH_INTERNAL_ASSERT(
      momentum != nullptr && momentum->getDataType().has_value() &&
          momentum->getDataType().value() == DataType::Double,
      "Momentum is not a valid Double.");

  TORCH_INTERNAL_ASSERT(
      eps != nullptr && eps->getDataType().has_value() &&
          eps->getDataType().value() == DataType::Double,
      "Epsilon (eps) is not a valid Double.");

  // (B, C, H, W, D) tensor
  // M = outer = B * C
  // N = reduction = H * W * D
  // weight = bias = C tensor
  const size_t kBatchDim = 0;
  const size_t kChannelsDim = 1;
  const size_t kNumberOfDims =
      TensorDomain::noReductions(x->getRootDomain()).size();

  std::vector<int> x_reduction_axes;
  std::vector<bool> x_broadcast_mask(kNumberOfDims, false);
  Val* N = new Double(1);
  for (size_t axis = 0; axis < kNumberOfDims; ++axis) {
    if (axis != kBatchDim && axis != kChannelsDim) {
      x_reduction_axes.push_back(axis);
      x_broadcast_mask[axis] = true;
      N = mul(N, x->domain()->domain()[axis]->extent());
    }
  }
  Val* B = new Double(1);
  B = mul(B, x->domain()->domain()[kBatchDim]->extent());

  std::vector<bool> channels_only_broadcast_mask(kNumberOfDims, false);
  for (size_t axis = 0; axis < kNumberOfDims; ++axis) {
    if (axis != kChannelsDim) {
      channels_only_broadcast_mask[axis] = true;
    }
  }

  TensorView* y = nullptr;
  TensorView* mean = nullptr;
  TensorView* invstd = nullptr;
  if (kUseInputStats || running_mean == nullptr) {
    // Algorithm
    auto welford_out = Welford(x, x_reduction_axes);

    // updating running mean and running var
    if (running_mean != nullptr && running_var != nullptr) {
      auto rev_momentum = sub(new Double(1.0), momentum);
      auto current_mean_hat = mul(welford_out.avg, momentum);
      auto mean_hat = mul(running_mean, rev_momentum);
      auto new_mean_hat = add(mean_hat, current_mean_hat);

      // NS: static_cast to workaround VC++ error, see https://godbolt.org/z/6Prd77xYs
      auto new_mean_sum = sum(new_mean_hat, {static_cast<int>(kBatchDim)});
      auto new_mean_channels_only = div(new_mean_sum, B);
      fusion->addOutput(new_mean_channels_only);
      fusion->aliasOutputToInput(new_mean_channels_only, running_mean);

      auto num_feature_decrement = sub(N, new Int(1));
      auto unbiased_var = div(welford_out.var_sum, num_feature_decrement);
      auto current_var_hat = mul(unbiased_var, momentum);
      auto var_hat = mul(running_var, rev_momentum);
      auto new_var_hat = add(var_hat, current_var_hat);

      auto new_var_sum = sum(new_var_hat, {kBatchDim});
      auto new_var_channels_only = div(new_var_sum, B);
      fusion->addOutput(new_var_channels_only);
      fusion->aliasOutputToInput(new_var_channels_only, running_var);
    }

    mean = welford_out.avg;
    auto mean_bcast = broadcast(mean, x_broadcast_mask);
    auto x_sub_mean = sub(x, mean_bcast);

    auto var = div(welford_out.var_sum, N);
    auto var_eps = add(var, eps);
    invstd = unaryOp(UnaryOpType::Rsqrt, var_eps);
    auto invstd_bcast = broadcast(invstd, x_broadcast_mask);

    y = mul(x_sub_mean, invstd_bcast);
  } else {
    // This is inference mode with running stats
    auto r_mean_bcasted = broadcast(running_mean, channels_only_broadcast_mask);
    auto x_sub_mean = sub(x, r_mean_bcasted);

    auto var_eps = add(running_var, eps);
    auto unbiased_invstd = unaryOp(UnaryOpType::Rsqrt, var_eps);
    auto invstd_bcast =
        broadcast(unbiased_invstd, channels_only_broadcast_mask);

    // During inference, mean/invstd output are empty tensors
    mean = TensorViewBuilder().shape({0}).build();
    invstd = TensorViewBuilder().shape({0}).build();
    y = mul(x_sub_mean, invstd_bcast);
  }

  // Optional: norm * weight
  if (weight) {
    auto weight_bcast = broadcast(weight, channels_only_broadcast_mask);
    y = mul(y, weight_bcast);
  }

  // Optional: norm * weight + bias
  if (bias) {
    auto bias_bcast = broadcast(bias, channels_only_broadcast_mask);
    y = add(y, bias_bcast);
  }
  return {y, mean, invstd};
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
