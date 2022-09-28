#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ops/normalization.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

int nonNegativeAxis(int axis, int ndims) {
  return (axis >= 0) ? axis : (ndims + axis);
}

Val* numFeatures(TensorView* x, const std::vector<int>& dims, int ndims) {
  Val* num_features = IrBuilder::create<Double>(x->container(), 1);
  for (const auto dim : dims) {
    const int axis = nonNegativeAxis(dim, ndims);
    num_features = mul(num_features, x->domain()->domain()[axis]->extent());
  }
  return num_features;
}

TensorView* mean(TensorView* x, const std::vector<int>& dims, bool keepdim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");

  const int kNumberOfDims =
      TensorDomain::noReductions(x->getMaybeRFactorDomain()).size();

  auto sum_x = sum(x, dims, keepdim);
  auto y = div(sum_x, numFeatures(x, dims, kNumberOfDims));
  return y;
}

TensorView* variance(
    TensorView* x,
    const std::vector<int>& dims,
    bool unbiased,
    bool keepdim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  int64_t correction = unbiased ? 1 : 0;
  return variance(x, dims, correction, keepdim);
}

TensorView* variance(
    TensorView* x,
    const std::vector<int>& dims,
    int64_t correction,
    bool keepdim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");

  TORCH_CHECK(
      correction >= 0, "correction must be non-negative, but got ", correction);

  const int kNumberOfDims =
      TensorDomain::noReductions(x->getMaybeRFactorDomain()).size();

  auto bcast_mean = mean(x, dims, true /* keepdim */);
  auto x_mean_sub = sub(x, bcast_mean);
  auto x_mean_sub_sq = mul(x_mean_sub, x_mean_sub);
  auto sum_x_mean_sub_sq = sum(x_mean_sub_sq, dims, keepdim);

  auto num_features = numFeatures(x, dims, kNumberOfDims);
  if (correction > 0) {
    num_features =
        sub(num_features, IrBuilder::create<Int>(x->container(), correction));
  }
  auto y = div(sum_x_mean_sub_sq, num_features);

  return y;
}

TORCH_CUDA_CU_API VarMeanResult variance_mean(
    TensorView* x,
    const std::vector<int>& dims,
    int64_t correction,
    bool keepdim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");

  TORCH_CHECK(
      correction >= 0, "correction must be non-negative, but got ", correction);

  // There are compilation errors for half precision
  auto dtype = x->getDataType().value();
  TORCH_CHECK(
      !(dtype == DataType::Half || dtype == DataType::BFloat16),
      "variance_mean is not supported for ",
      dtype,
      " please upcast to float");

  if (isComplexType(x->getDataType().value())) {
    // There are compilation errors:
    // __tmp_kernel1.cu(6727): error: namespace "CudaCodeGen::std" has no member
    // "imagf"
    // __tmp_kernel1.cu(6753): error: namespace "CudaCodeGen::std" has no member
    // "realf"
    TORCH_CHECK(false, "var_mean is not supported for complex types.");
    auto out_real = variance_mean(real(x), dims, correction, keepdim);
    auto out_imag = variance_mean(imag(x), dims, correction, keepdim);
    // variance of a complex tensor is the sum of real and imaginary variances
    // and is real mean of a complex tensor is complex complex(out_real.mean,
    // out_imag.mean) It seems construction of a complex tensor from two real
    // tensors is not supported yet
    return {add(out_real.var, out_imag.var), nullptr};
  }

  const int kNumberOfDims =
      TensorDomain::noReductions(x->getMaybeRFactorDomain()).size();
  auto num_features = numFeatures(x, dims, kNumberOfDims);
  if (correction > 0) {
    num_features =
        sub(num_features, IrBuilder::create<Int>(x->container(), correction));
  }

  auto welford_out = Welford(x, dims);
  auto mean = welford_out.avg;
  auto var = mul(welford_out.var_sum, reciprocal(num_features));

  if (keepdim) {
    std::vector<bool> is_broadcast(kNumberOfDims, false);
    for (auto dim : dims) {
      is_broadcast[dim] = true;
    }
    var = broadcast(var, is_broadcast);
    mean = broadcast(mean, is_broadcast);
  }

  return {var, mean};
}

TensorView* standard_deviation(
    TensorView* x,
    const std::vector<int>& dims,
    bool unbiased,
    bool keepdim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  return sqrt(variance(x, dims, unbiased, keepdim));
}

TensorView* softmax(TensorView* x, int dim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");

  const int kNumberOfDims =
      TensorDomain::noReductions(x->getMaybeRFactorDomain()).size();
  const int kReductionAxis = (dim < 0) ? dim + kNumberOfDims : dim;
  TORCH_INTERNAL_ASSERT(kReductionAxis >= 0 && kReductionAxis < kNumberOfDims);

  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  broadcast_mask[kReductionAxis] = true;

  auto max_val = max(x, {kReductionAxis});
  auto bcast_max = broadcast(max_val, broadcast_mask);
  auto x_max_sub = sub(x, bcast_max);
  auto exp_val = exp(x_max_sub);
  auto sum_exp = sum(exp_val, {kReductionAxis});
  auto bcast_sum = broadcast(sum_exp, broadcast_mask);
  auto y = mul(exp_val, reciprocal(bcast_sum));

  return y;
}

TensorView* softmax_backward(TensorView* dy, TensorView* y, int dim) {
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(y != nullptr, "Output is invalid.");

  const int kNumberOfDims =
      TensorDomain::noReductions(y->getMaybeRFactorDomain()).size();
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

TensorView* log_softmax(TensorView* x, int dim) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");

  const int kNumberOfDims =
      TensorDomain::noReductions(x->getMaybeRFactorDomain()).size();
  const int kReductionAxis = (dim < 0) ? dim + kNumberOfDims : dim;
  TORCH_INTERNAL_ASSERT(kReductionAxis >= 0 && kReductionAxis < kNumberOfDims);

  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  broadcast_mask[kReductionAxis] = true;

  auto max_val = max(x, {kReductionAxis});
  auto bcast_max = broadcast(max_val, broadcast_mask);
  auto x_max_sub = sub(x, bcast_max);
  auto exp_val = exp(x_max_sub);
  auto bcast_sum = sum(exp_val, {kReductionAxis}, true /* keepdim */);
  auto log_sum_exp = log(bcast_sum);
  auto y = sub(x_max_sub, log_sum_exp);

  return y;
}

TensorView* log_softmax_backward(TensorView* dy, TensorView* y, int dim) {
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(y != nullptr, "Output is invalid.");

  const int kNumberOfDims =
      TensorDomain::noReductions(y->getMaybeRFactorDomain()).size();
  const int kReductionAxis = (dim < 0) ? dim + kNumberOfDims : dim;
  TORCH_INTERNAL_ASSERT(kReductionAxis >= 0 && kReductionAxis < kNumberOfDims);

  auto bcast_sum_grad = sum(dy, {kReductionAxis}, true /* keepdim */);
  auto softmax = exp(y);
  auto softmax_sum_mul = mul(softmax, bcast_sum_grad);
  auto dx = sub(dy, softmax_sum_mul);

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

auto norm_properties_from_num_dims(
    const TensorView* x,
    const size_t kNormShapeNumDims) {
  // (B, C, H, W, D) tensor
  // norm_shape = [H, W, D]
  // M = outer = product of remaining dimensions = B * C
  // N = reduction = product of norm_shape = H * W * D
  // weight = bias = norm_shape tensor
  const size_t kNumberOfDims =
      TensorDomain::noReductions(x->getMaybeRFactorDomain()).size();
  const size_t kOuterNumDims = kNumberOfDims - kNormShapeNumDims;

  std::vector<int> outer_reduction_axes(kOuterNumDims);
  std::vector<bool> outer_broadcast_mask(kNumberOfDims, false);
  std::vector<int> inner_reduction_axes(kNormShapeNumDims);
  std::vector<bool> inner_broadcast_mask(kNumberOfDims, false);

  for (const auto idx : c10::irange(kOuterNumDims)) {
    outer_reduction_axes[idx] = idx;
    outer_broadcast_mask[idx] = true;
  }

  Val* num_features = IrBuilder::create<Double>(x->container(), 1);
  for (const auto idx : c10::irange(kNormShapeNumDims)) {
    const size_t axis = kNumberOfDims - 1 - idx;
    inner_reduction_axes[idx] = axis;
    inner_broadcast_mask[axis] = true;
    num_features = mul(num_features, x->domain()->domain()[axis]->extent());
  }
  struct result {
    std::vector<int> outer_reduction_axes;
    std::vector<bool> outer_broadcast_mask;
    std::vector<int> inner_reduction_axes;
    std::vector<bool> inner_broadcast_mask;
    Val* num_features = nullptr;
  } r;
  r.outer_reduction_axes = outer_reduction_axes;
  r.outer_broadcast_mask = outer_broadcast_mask;
  r.inner_reduction_axes = inner_reduction_axes;
  r.inner_broadcast_mask = inner_broadcast_mask;
  r.num_features = num_features;
  return r;
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

  auto r = norm_properties_from_num_dims(x, kNormShapeNumDims);

  // Main algorithm
  auto welford_out = Welford(x, r.inner_reduction_axes);
  auto mean_bcast = broadcast(welford_out.avg, r.inner_broadcast_mask);
  auto x_sub_mean = sub(x, mean_bcast);

  auto var_sum_bcast = broadcast(welford_out.var_sum, r.inner_broadcast_mask);
  auto var = mul(var_sum_bcast, reciprocal(r.num_features));
  auto var_eps = add(var, eps);
  auto invstd = rsqrt(var_eps);

  auto y = mul(x_sub_mean, invstd);

  // Optional: norm * weight
  if (weight != nullptr) {
    auto weight_bcast = broadcast(weight, r.outer_broadcast_mask);
    y = mul(y, weight_bcast);
  }

  // Optional: norm * weight + bias
  if (bias != nullptr) {
    auto bias_bcast = broadcast(bias, r.outer_broadcast_mask);
    y = add(y, bias_bcast);
  }

  return {y, mean_bcast, invstd};
}

ForwardRMSNormResult rms_norm(
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* weight,
    Val* eps) {
  return rms_norm(x, norm_shape.size(), weight, eps);
}

ForwardRMSNormResult rms_norm(
    TensorView* x,
    const size_t kNormShapeNumDims,
    TensorView* weight,
    Val* eps) {
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(
      eps != nullptr && eps->getDataType().has_value() &&
          eps->getDataType().value() == DataType::Double,
      "Epsilon (eps) is not a valid Double.");

  auto r = norm_properties_from_num_dims(x, kNormShapeNumDims);

  // Main algorithm
  auto var_sum = sum(mul(x, x), r.inner_reduction_axes);
  auto var_sum_bcast = broadcast(var_sum, r.inner_broadcast_mask);
  auto var = mul(var_sum_bcast, reciprocal(r.num_features));
  auto var_eps = add(var, eps);
  auto invstd = rsqrt(var_eps);

  auto y = mul(x, invstd);

  // Optional: norm * weight
  if (weight != nullptr) {
    auto weight_bcast = broadcast(weight, r.outer_broadcast_mask);
    y = mul(y, weight_bcast);
  }

  return {y, invstd};
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

  auto r = norm_properties_from_num_dims(x, norm_shape.size());

  auto x_hat = mul(sub(x, mean), invstd);

  TensorView* grad_x_hat = nullptr;
  if (weight != nullptr) {
    auto* bcast_weight = broadcast(weight, r.outer_broadcast_mask);
    grad_x_hat = mul(dy, bcast_weight);
  } else {
    grad_x_hat = dy;
  }

  auto a = mul(r.num_features, grad_x_hat);

  auto b = sum(grad_x_hat, r.inner_reduction_axes);
  auto bcast_b = broadcast(b, r.inner_broadcast_mask);

  auto c1 = mul(grad_x_hat, x_hat);
  auto c2 = sum(c1, r.inner_reduction_axes);
  auto bcast_c2 = broadcast(c2, r.inner_broadcast_mask);
  auto c3 = mul(x_hat, bcast_c2);

  auto inner = sub(sub(a, bcast_b), c3);
  auto reciprocal_size = reciprocal(r.num_features);

  TensorView* dx = nullptr;
  if (output_mask[0]) {
    dx = mul(mul(reciprocal_size, invstd), inner);
  }

  TensorView* dw = nullptr;
  if (output_mask[1] && weight != nullptr) {
    dw = sum(mul(dy, x_hat), r.outer_reduction_axes);
  }

  TensorView* db = nullptr;
  if (output_mask[2] && bias != nullptr) {
    db = sum(dy, r.outer_reduction_axes);
  }
  return {dx, dw, db};
}

BackwardRMSNormResult rms_norm_backward(
    TensorView* dy,
    TensorView* x,
    const std::vector<int64_t>& norm_shape,
    TensorView* invstd,
    TensorView* weight,
    const std::vector<bool>& output_mask) {
  TORCH_INTERNAL_ASSERT(dy != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(x != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(invstd != nullptr, "Inv std is invalid.");

  auto r = norm_properties_from_num_dims(x, norm_shape.size());

  auto x_hat = mul(x, invstd);

  TensorView* grad_x_hat = nullptr;
  if (weight != nullptr) {
    auto* bcast_weight = broadcast(weight, r.outer_broadcast_mask);
    grad_x_hat = mul(dy, bcast_weight);
  } else {
    grad_x_hat = dy;
  }

  auto a = mul(r.num_features, grad_x_hat);

  auto b = sum(grad_x_hat, r.inner_reduction_axes);
  auto bcast_b = broadcast(b, r.inner_broadcast_mask);

  auto c1 = mul(grad_x_hat, x_hat);
  auto c2 = sum(c1, r.inner_reduction_axes);
  auto bcast_c2 = broadcast(c2, r.inner_broadcast_mask);
  auto c3 = mul(x_hat, bcast_c2);

  auto inner = sub(sub(a, bcast_b), c3);
  auto reciprocal_size = reciprocal(r.num_features);

  TensorView* dx = nullptr;
  if (output_mask[0]) {
    dx = mul(mul(reciprocal_size, invstd), inner);
  }

  TensorView* dw = nullptr;
  if (output_mask[1] && weight != nullptr) {
    dw = sum(mul(dy, x_hat), r.outer_reduction_axes);
  }

  return {dx, dw};
}

ForwardNormResult batch_norm(
    TensorView* x,
    TensorView* weight,
    TensorView* bias,
    TensorView* running_mean,
    TensorView* running_var,
    const bool kTraining,
    Val* momentum,
    Val* eps,
    bool channels_last) {
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
  const size_t kNumberOfDims =
      TensorDomain::noReductions(x->getMaybeRFactorDomain()).size();
  // channels last format means C dimension is at axis kNumberOfDims-1 at x
  size_t c_axis = channels_last ? kNumberOfDims - 1 : 1;

  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  Val* num_features = IrBuilder::create<Double>(x->container(), 1);

  for (const auto axis : c10::irange(kNumberOfDims)) {
    if (axis != c_axis) {
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
      // Note: kTraining is true here!
      TORCH_INTERNAL_ASSERT(
          kTraining,
          "When running stats are provided, batch stats should only be computed during training");

      auto rev_momentum =
          sub(IrBuilder::create<Double>(x->container(), 1.0), momentum);
      auto current_mean_hat = mul(welford_out.avg, momentum);
      auto mean_hat = mul(running_mean, rev_momentum);
      auto new_mean_hat = add(mean_hat, current_mean_hat);

      auto num_feature_decrement = sub(num_features, x->container()->oneVal());
      auto unbiased_var =
          mul(welford_out.var_sum, reciprocal(num_feature_decrement));
      auto current_var_hat = mul(unbiased_var, momentum);
      auto var_hat = mul(running_var, rev_momentum);
      auto new_var_hat = add(var_hat, current_var_hat);

      // when inputs have been cast by parser. We want to alias the output to
      // the pre-cast input, so we can still update running stats
      auto cast_to_input_dtype = [fusion](
                                     Val* cast_input, Val* aliased_output) {
        auto unary_op = cast_input->definition();
        TORCH_INTERNAL_ASSERT(
            unary_op->isA<UnaryOp>() &&
                unary_op->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Cast,
            "check for cast op");
        auto input_to_cast = unary_op->input(0);
        TORCH_INTERNAL_ASSERT(
            input_to_cast->isFusionInput(),
            "IO_tensor batch_norm::running_stats can only updating input tensor to fusion");
        auto rm_dtype = input_to_cast->getDataType();
        TORCH_INTERNAL_ASSERT(
            rm_dtype.has_value(),
            "Input running stats must have dtype defined");
        auto cast_output = castOp(*rm_dtype, aliased_output);

        fusion->aliasOutputToInput(cast_output, input_to_cast);
      };

      if (running_mean->isFusionInput()) {
        fusion->aliasOutputToInput(new_mean_hat, running_mean);
      } else {
        cast_to_input_dtype(running_mean, new_mean_hat);
      }

      if (running_var->isFusionInput()) {
        fusion->aliasOutputToInput(new_var_hat, running_var);
      } else {
        cast_to_input_dtype(running_var, new_var_hat);
      }
    }

    mean = welford_out.avg;
    auto mean_bcast = broadcast(mean, broadcast_mask);
    auto x_sub_mean = sub(x, mean_bcast);

    auto var = mul(welford_out.var_sum, reciprocal(num_features));
    auto var_eps = add(var, eps);
    invstd = rsqrt(var_eps);
    auto invstd_bcast = broadcast(invstd, broadcast_mask);

    y = mul(x_sub_mean, invstd_bcast);
  } else {
    // This is inference mode with running stats
    auto r_mean_bcasted = broadcast(running_mean, broadcast_mask);
    auto x_sub_mean = sub(x, r_mean_bcasted);

    auto var_eps = add(running_var, eps);
    auto unbiased_invstd = rsqrt(var_eps);
    auto invstd_bcast = broadcast(unbiased_invstd, broadcast_mask);

    // During inference, mean/invstd output are empty tensors
    // on CPU, but not on CUDA. We need to make sure we have the same
    // behavior as with eager mode on CUDA.
    mean = set(running_mean); // use set to avoid "trivial input forwarding NOT
                              // IMPLEMENTED" error
    invstd = unbiased_invstd;
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
    TensorView* input,
    TensorView* grad_output,
    TensorView* weight,
    TensorView* running_mean,
    TensorView* running_var,
    TensorView* save_mean,
    TensorView* save_invstd,
    const bool kTraining,
    Val* eps,
    const std::vector<bool>& output_mask,
    bool channels_last) {
  TORCH_INTERNAL_ASSERT(input != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(grad_output != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(
      eps != nullptr && eps->getDataType().has_value() &&
          eps->getDataType().value() == DataType::Double,
      "Epsilon (eps) is not a valid Double.");

  // (B, C, H, W, D) tensor
  // M = outer = channels
  // N = reduction = B * H * W * D
  // weight = bias = (C) tensor
  const size_t kNumberOfDims =
      TensorDomain::noReductions(input->getMaybeRFactorDomain()).size();
  // channels last format means C dimension is at axis kNumberOfDims-1 at x /
  // grad_out
  size_t c_axis = channels_last ? kNumberOfDims - 1 : 1;

  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  Val* num_features = nullptr;
  for (const auto axis : c10::irange(kNumberOfDims)) {
    if (axis != c_axis) {
      reduction_axes.push_back(axis);
      broadcast_mask[axis] = true;
      if (num_features == nullptr) {
        num_features =
            castOp(DataType::Double, input->domain()->domain()[axis]->extent());
      } else {
        num_features =
            mul(num_features, input->domain()->domain()[axis]->extent());
      }
    }
  }

  auto mean = save_mean;
  auto invstd = save_invstd;
  if (kTraining) {
    TORCH_INTERNAL_ASSERT(
        save_mean != nullptr && save_invstd != nullptr,
        "When training=True, save_mean and save_invstd are required.");
  } else {
    mean = running_mean;
    invstd = rsqrt(add(running_var, eps));
  }

  mean = broadcast(mean, broadcast_mask);

  auto norm = reciprocal(num_features);

  auto grad_output_sum = sum(grad_output, reduction_axes);
  auto dot_p = sum(mul(grad_output, sub(input, mean)), reduction_axes);

  auto grad_mean = broadcast(mul(grad_output_sum, norm), broadcast_mask);
  auto proj_scale =
      broadcast(mul(mul(dot_p, norm), mul(invstd, invstd)), broadcast_mask);
  TensorView* grad_scale = nullptr;

  if (weight == nullptr) {
    grad_scale =
        mul(broadcast(invstd, broadcast_mask),
            IrBuilder::create<Double>(input->container(), 1));
  } else {
    grad_scale = mul(
        broadcast(invstd, broadcast_mask), broadcast(weight, broadcast_mask));
  }

  TensorView* grad_input = nullptr;
  if (kTraining) {
    auto proj = mul(sub(input, mean), proj_scale);
    grad_input = mul(sub(sub(grad_output, proj), grad_mean), grad_scale);
  } else {
    grad_input = mul(grad_output, grad_scale);
  }

  TensorView* grad_weight = nullptr;
  if (output_mask[1]) {
    grad_weight = mul(dot_p, invstd);
  }

  TensorView* grad_bias = nullptr;
  if (output_mask[2]) {
    grad_bias = grad_output_sum;
  }

  return {grad_input, grad_weight, grad_bias};
}

ForwardNormResult instance_norm(
    TensorView* x,
    TensorView* weight,
    TensorView* bias,
    TensorView* running_mean,
    TensorView* running_var,
    const bool kUseInputStats,
    Val* momentum,
    Val* eps,
    bool channels_last) {
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
  const size_t kNumberOfDims =
      TensorDomain::noReductions(x->getMaybeRFactorDomain()).size();
  const size_t kChannelsDim = channels_last ? kNumberOfDims - 1 : 1;

  std::vector<int> x_reduction_axes;
  std::vector<bool> x_broadcast_mask(kNumberOfDims, false);
  Val* N = IrBuilder::create<Double>(x->container(), 1);
  for (const auto axis : c10::irange(kNumberOfDims)) {
    if (axis != kBatchDim && axis != kChannelsDim) {
      x_reduction_axes.push_back(axis);
      x_broadcast_mask[axis] = true;
      N = mul(N, x->domain()->domain()[axis]->extent());
    }
  }
  Val* B = IrBuilder::create<Double>(x->container(), 1);
  B = mul(B, x->domain()->domain()[kBatchDim]->extent());

  std::vector<bool> channels_only_broadcast_mask(kNumberOfDims, false);
  for (const auto axis : c10::irange(kNumberOfDims)) {
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
      auto _running_mean = running_mean;
      auto _running_var = running_var;
      if (_running_mean->getDataType().value() == DataType::Half ||
          _running_mean->getDataType().value() == DataType::BFloat16) {
        _running_mean = castOp(DataType::Float, _running_mean);
      }
      if (_running_var->getDataType().value() == DataType::Half ||
          _running_var->getDataType().value() == DataType::BFloat16) {
        _running_var = castOp(DataType::Float, running_var);
      }
      auto rev_momentum =
          sub(IrBuilder::create<Double>(x->container(), 1.0), momentum);
      auto current_mean_hat = mul(welford_out.avg, momentum);
      auto mean_hat = mul(_running_mean, rev_momentum);
      auto new_mean_hat = add(mean_hat, current_mean_hat);

      // NS: static_cast to workaround VC++ error, see
      // https://godbolt.org/z/6Prd77xYs
      auto new_mean_sum = sum(new_mean_hat, {static_cast<int>(kBatchDim)});
      auto new_mean_channels_only = mul(new_mean_sum, reciprocal(B));
      if (running_mean->getDataType().value() == DataType::Half ||
          running_mean->getDataType().value() == DataType::BFloat16) {
        new_mean_channels_only =
            castOp(running_mean->getDataType().value(), new_mean_channels_only);
      }
      // fusion->addOutput(new_mean_channels_only);
      fusion->aliasOutputToInput(new_mean_channels_only, running_mean);

      auto num_feature_decrement = sub(N, x->container()->oneVal());
      auto unbiased_var =
          mul(welford_out.var_sum, reciprocal(num_feature_decrement));
      auto current_var_hat = mul(unbiased_var, momentum);
      auto var_hat = mul(_running_var, rev_momentum);
      auto new_var_hat = add(var_hat, current_var_hat);

      // NS: static_cast to workaround VC++ error, see
      // https://godbolt.org/z/6Prd77xYs
      auto new_var_sum = sum(new_var_hat, {static_cast<int>(kBatchDim)});
      auto new_var_channels_only = mul(new_var_sum, reciprocal(B));
      if (running_var->getDataType().value() == DataType::Half ||
          running_var->getDataType().value() == DataType::BFloat16) {
        new_var_channels_only =
            castOp(running_var->getDataType().value(), new_var_channels_only);
      }
      // fusion->addOutput(new_var_channels_only);
      fusion->aliasOutputToInput(new_var_channels_only, running_var);
    }

    mean = welford_out.avg;
    auto mean_bcast = broadcast(mean, x_broadcast_mask);
    auto x_sub_mean = sub(x, mean_bcast);

    auto var = mul(welford_out.var_sum, reciprocal(N));
    auto var_eps = add(var, eps);
    invstd = rsqrt(var_eps);
    auto invstd_bcast = broadcast(invstd, x_broadcast_mask);

    y = mul(x_sub_mean, invstd_bcast);
  } else {
    // This is inference mode with running stats
    auto r_mean_bcasted = broadcast(running_mean, channels_only_broadcast_mask);
    auto x_sub_mean = sub(x, r_mean_bcasted);

    auto var_eps = add(running_var, eps);
    auto unbiased_invstd = rsqrt(var_eps);
    auto invstd_bcast =
        broadcast(unbiased_invstd, channels_only_broadcast_mask);

    // During inference, mean/invstd output are empty tensors
    mean = TensorViewBuilder().shape(std::vector<int64_t>{0}).build();
    invstd = TensorViewBuilder().shape(std::vector<int64_t>{0}).build();
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

BackwardNormResult instance_norm_backward(
    TensorView* input,
    TensorView* grad_output,
    TensorView* weight,
    TensorView* running_mean,
    TensorView* running_var,
    TensorView* save_mean,
    TensorView* save_invstd,
    const bool kTraining,
    Val* eps,
    const std::vector<bool>& output_mask,
    bool channels_last) {
  TORCH_INTERNAL_ASSERT(input != nullptr, "Input is invalid.");
  TORCH_INTERNAL_ASSERT(grad_output != nullptr, "Grad Output is invalid.");
  TORCH_INTERNAL_ASSERT(
      eps != nullptr && eps->getDataType().has_value() &&
          eps->getDataType().value() == DataType::Double,
      "Epsilon (eps) is not a valid Double.");

  // (B, C, H, W, D) tensor
  // M = outer = channels
  // N = reduction = B * H * W * D
  // weight = bias = (C) tensor
  const size_t kNumberOfDims =
      TensorDomain::noReductions(input->getMaybeRFactorDomain()).size();
  // channels last format means C dimension is at axis kNumberOfDims-1 at x /
  // grad_out
  const size_t b_axis = 0; // for clarity
  const size_t c_axis = channels_last ? kNumberOfDims - 1 : 1;

  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  // weight has its own broadcast mask as it is broadcast for the batch unlike
  // mean/var
  std::vector<bool> weight_broadcast_mask(kNumberOfDims, false);
  Val* num_features = nullptr;
  for (const auto axis : c10::irange(kNumberOfDims)) {
    if (axis != c_axis) {
      weight_broadcast_mask[axis] = true;
      if (axis != b_axis) {
        reduction_axes.push_back(axis);
        broadcast_mask[axis] = true;
        if (num_features == nullptr) {
          num_features = castOp(
              DataType::Double, input->domain()->domain()[axis]->extent());
        } else {
          num_features =
              mul(num_features, input->domain()->domain()[axis]->extent());
        }
      }
    }
  }

  auto mean = save_mean;
  auto invstd = save_invstd;
  if (kTraining) {
    TORCH_INTERNAL_ASSERT(
        save_mean != nullptr && save_invstd != nullptr,
        "When training=True, save_mean and save_invstd are required.");
  } else {
    mean = running_mean;
    invstd = rsqrt(add(running_var, eps));
  }
  mean = broadcast(mean, broadcast_mask);

  auto norm = reciprocal(num_features);

  auto grad_output_sum = sum(grad_output, reduction_axes);
  auto dot_p = sum(mul(grad_output, sub(input, mean)), reduction_axes);

  auto grad_mean = broadcast(mul(grad_output_sum, norm), broadcast_mask);

  auto proj_scale =
      broadcast(mul(mul(dot_p, norm), mul(invstd, invstd)), broadcast_mask);

  TensorView* grad_scale = nullptr;

  if (weight == nullptr) {
    grad_scale =
        mul(broadcast(invstd, broadcast_mask),
            IrBuilder::create<Double>(input->container(), 1));
  } else {
    grad_scale =
        mul(broadcast(invstd, broadcast_mask),
            broadcast(weight, weight_broadcast_mask));
  }

  TensorView* grad_input = nullptr;
  if (kTraining) {
    auto proj = mul(sub(input, mean), proj_scale);
    grad_input = mul(sub(sub(grad_output, proj), grad_mean), grad_scale);
  } else {
    grad_input = mul(grad_output, grad_scale);
  }

  TensorView* grad_weight = nullptr;
  TensorView* grad_weight_reduced = nullptr;
  if (output_mask[1]) {
    grad_weight = mul(dot_p, invstd);
    // TODO: grad weight needs to be reduced across batch-dim but is this the
    // most efficient place or can reduction happen earlier?
    grad_weight_reduced = sum(grad_weight, {0});
  }

  TensorView* grad_bias = nullptr;
  TensorView* grad_bias_reduced = nullptr;
  if (output_mask[2]) {
    grad_bias = grad_output_sum;
    // TODO: same as above for grad weight
    grad_bias_reduced = sum(grad_bias, {0});
  }

  return {grad_input, grad_weight_reduced, grad_bias_reduced};
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
