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

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> layer_norm_cpu(
    const Tensor& X,
    const Tensor& gamma /* optional */,
    const Tensor& beta /* optional */,
    int64_t M,
    int64_t N,
    double eps) {
  Tensor Y = at::native::empty_like(X, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor mean = at::empty({M}, X.options());
  Tensor rstd = at::empty({M}, X.options());
  if (M > 0) {
    LayerNormKernel(kCPU, X, gamma, beta, M, N, eps, &Y, &mean, &rstd);
  }
  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cpu(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {
  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(X, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[1]) {
    dgamma = M > 0 ? at::native::empty_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : at::native::zeros_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    dbeta = M > 0 ? at::native::empty_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : at::native::zeros_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
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
  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias.defined() || bias.sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias.sizes(),
      " and normalized_shape = ",
      normalized_shape);

  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int axis = input_ndim - normalized_ndim;
  const int64_t M = std::accumulate(
      input_shape.cbegin(),
      input_shape.cbegin() + axis,
      1LL,
      std::multiplies<int64_t>());
  const int64_t N = std::accumulate(
      input_shape.cbegin() + axis,
      input_shape.cend(),
      1LL,
      std::multiplies<int64_t>());

  const auto& X = input.is_contiguous() ? input : input.contiguous();
  const auto& gamma = weight.is_contiguous() ? weight : weight.contiguous();
  const auto& beta = bias.is_contiguous() ? bias : bias.contiguous();
  return std::get<0>(at::native_layer_norm(X, gamma, beta, M, N, eps));
}

DEFINE_DISPATCH(LayerNormKernel);
DEFINE_DISPATCH(LayerNormBackwardKernel);

} // namespace native
} // namespace at
