#include <ATen/native/layer_norm.h>

#include <array>
#include <functional>
#include <numeric>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> layer_norm_cpu(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    double eps) {

  auto inputs = _prepare_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto X = std::get<0>(inputs);
  auto gamma = std::get<1>(inputs);
  auto beta = std::get<2>(inputs);
  auto M = std::get<3>(inputs);
  auto N = std::get<4>(inputs);

  Tensor Y = at::native::empty_like(X, at::MemoryFormat::Contiguous);
  Tensor mean = at::empty({M}, X.options());
  Tensor rstd = at::empty({M}, X.options());
  if (M > 0) {
    LayerNormKernel(kCPU, X, gamma, beta, M, N, eps, &Y, &mean, &rstd);

    const auto input_shape = input.sizes();
    const size_t axis = input.dim() - normalized_shape.size();

    std::vector<int64_t> stat_shape;
    for (size_t idx = 0; idx < axis; ++idx) {
      stat_shape.push_back(input_shape[idx]);
    }
    for (size_t idx = axis; idx < input.dim(); ++idx) {
      stat_shape.push_back(1);
    }

    mean = mean.view(stat_shape);
    rstd = rstd.view(stat_shape);
  }
  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cpu(
    const Tensor& dY,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    std::array<bool, 3> grad_input_mask) {

    auto inputs = _prepare_layer_norm_inputs(input, normalized_shape, weight, bias);
    auto X = std::get<0>(inputs);
    auto gamma = std::get<1>(inputs);
    auto beta = std::get<2>(inputs);
    auto M = std::get<3>(inputs);
    auto N = std::get<4>(inputs);

    Tensor dX;
    Tensor dgamma;
    Tensor dbeta;
    if (grad_input_mask[0]) {
      dX = at::native::empty_like(X, at::MemoryFormat::Contiguous);
    }
    if (grad_input_mask[1]) {
      dgamma = M > 0 ? at::native::empty_like(gamma, at::MemoryFormat::Contiguous) : at::native::zeros_like(gamma, at::MemoryFormat::Contiguous);
    }
    if (grad_input_mask[2]) {
      dbeta = M > 0 ? at::native::empty_like(beta, at::MemoryFormat::Contiguous) : at::native::zeros_like(beta, at::MemoryFormat::Contiguous);
    }
    if (M > 0) {
      LayerNormBackwardKernel(
          kCPU, dY, X, mean, rstd, gamma, M, N, &dX, &dgamma, &dbeta);
    }
    return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}

Tensor layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    double eps,
    bool /* cudnn_enable, deprecated */) {

  return std::get<0>(at::native_layer_norm(input, normalized_shape, weight, bias, eps));
}

DEFINE_DISPATCH(LayerNormKernel);
DEFINE_DISPATCH(LayerNormBackwardKernel);

// Ported from pytorch/xla repo
std::tuple<Tensor, Tensor, Tensor> math_native_layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double eps) {
  auto inputs = _prepare_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto X = std::get<0>(inputs);
  auto gamma = std::get<1>(inputs);
  auto beta = std::get<2>(inputs);
  auto M = std::get<3>(inputs);
  auto N = std::get<4>(inputs);
  auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  const int normalized_ndim = normalized_shape.size();
  const int axis = input_ndim - normalized_ndim;
  at::Tensor input_reshaped = input.view({1, M, -1});
  // Unlike Batch Normalization, which applies scalar scale and bias for each
  // entire channel/plane with the affine option, Layer Normalization applies
  // per-element scale and bias. E.g. For input {N, C, H, W}, weight for
  // batchnorm has shape {C} while weight for layernorm has shape {H, W} or {W}.
  auto outputs = at::native_batch_norm(
      input_reshaped, /*weight=*/{}, /*bias=*/{}, /*running_mean=*/{},
      /*running_var=*/{}, /*training=*/true, /*momentum=*/0, eps);
  at::Tensor out = std::get<0>(outputs);
  out = out.view(input_shape);
  if (weight.defined() && bias.defined()) {
    out = bias.addcmul(out, weight, 1);
  } else if (weight.defined()) {
    out = out.mul(weight);
  } else if (bias.defined()) {
    out = out.add(bias);
  }
  at::Tensor mean = std::get<1>(outputs);
  at::Tensor rstd = std::get<2>(outputs);
  std::vector<int64_t> stat_shape;
  for (size_t idx = 0; idx < axis; ++idx) {
    stat_shape.push_back(input_shape[idx]);
  }
  for (size_t idx = axis; idx < input.dim(); ++idx) {
    stat_shape.push_back(1);
  }
  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);
  return std::make_tuple(out, mean, rstd);
}
} // namespace native
} // namespace at
