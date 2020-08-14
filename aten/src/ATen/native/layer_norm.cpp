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

std::tuple<Tensor, Tensor, Tensor> native_layer_norm(
    const Tensor& X,
    const Tensor& gamma /* optional */,
    const Tensor& beta /* optional */,
    int64_t M,
    int64_t N,
    double eps) {
  Tensor Y = at::native::empty_like(X, at::MemoryFormat::Contiguous);
  Tensor mean = at::empty({M}, X.options());
  Tensor rstd = at::empty({M}, X.options());
  LayerNormKernel(
      X.device().type(), X, gamma, beta, M, N, eps, &Y, &mean, &rstd);
  return std::forward_as_tuple(Y, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward(
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
    dX = at::native::empty_like(X, at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[1]) {
    dgamma = M > 0
        ? at::native::empty_like(gamma, at::MemoryFormat::Contiguous)
        : at::native::zeros_like(gamma, at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    dbeta = M > 0 ? at::native::empty_like(gamma, at::MemoryFormat::Contiguous)
                  : at::native::zeros_like(gamma, at::MemoryFormat::Contiguous);
  }
  if (M > 0) {
    LayerNormBackwardKernel(
        X.device().type(),
        dY,
        X,
        mean,
        rstd,
        gamma,
        M,
        N,
        &dX,
        &dgamma,
        &dbeta);
  }
  return std::forward_as_tuple(dX, dgamma, dbeta);
}

Tensor layer_norm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  Tensor X;
  Tensor gamma;
  Tensor beta;
  int64_t M;
  int64_t N;
  std::tie(X, gamma, beta, M, N) =
      _prepare_layer_norm_inputs(input, normalized_shape, weight, bias);
  return std::get<0>(at::native_layer_norm(X, gamma, beta, M, N, eps));
}

DEFINE_DISPATCH(LayerNormKernel);
DEFINE_DISPATCH(LayerNormBackwardKernel);

} // namespace native
} // namespace at
