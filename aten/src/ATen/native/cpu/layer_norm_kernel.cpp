#include <ATen/native/cpu/layer_norm_kernel.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>

namespace at {
namespace native {

namespace {

template <typename T>
void LayerNormKernelImplInternal(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t M,
    int64_t N,
    T eps,
    Tensor* out,
    Tensor* mean,
    Tensor* rstd)
{
  DCHECK_EQ(input.numel(), M * N);
  DCHECK(!weight.defined() || weight.numel() == N);
  DCHECK(!bias.defined() || bias.numel() == N);
  const T* input_data = input.data_ptr<T>();
  const T* weight_data = weight.defined() ? weight.data_ptr<T>() : nullptr;
  const T* bias_data = bias.defined() ? bias.data_ptr<T>() : nullptr;
  T* out_data = out->data_ptr<T>();
  T* mean_data = mean->data_ptr<T>();
  T* rstd_data = rstd->data_ptr<T>();
  const T c = T(1) / static_cast<T>(N);
  const bool weight_null = weight_data == nullptr;
  const bool bias_null = bias_data == nullptr;
  for (int64_t i = 0; i < M; ++i) {
    const T* input_ptr = input_data + i * N;
    T* out_ptr = out_data + i * N;
    T mean_val = T(0);
    T rstd_val = T(0);
    for (int64_t j = 0; j < N; ++j) {
      mean_val += input_ptr[j];
      rstd_val += input_ptr[j] * input_ptr[j];
    }
    mean_val *= c;
    rstd_val = std::max(rstd_val * c - mean_val * mean_val, T(0));
    rstd_val = T(1) / std::sqrt(rstd_val + eps);
    const T scale = rstd_val;
    const T bias = -rstd_val * mean_val;
    for (int64_t j = 0; j < N; ++j) {
      const T weight_v = weight_null ? T(1) : weight_data[j];
      const T bias_v = bias_null ? T(0) : bias_data[j];
      out_ptr[j] = (input_ptr[j] * scale + bias) * weight_v + bias_v;
    }
    mean_data[i] = mean_val;
    rstd_data[i] = rstd_val;
  }
}

void LayerNormKernelImpl(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t M,
    int64_t N,
    double eps,
    Tensor* out,
    Tensor* mean,
    Tensor* rstd)
{
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "LayerNormKernelImpl", [&]() {
    LayerNormKernelImplInternal<scalar_t>(
        input, weight, bias, M, N, static_cast<scalar_t>(eps), out, mean, rstd);
  });
}

template <typename T>
void LayerNormBackwardKernelImplInternal(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    int64_t M,
    int64_t N,
    Tensor* grad_input,
    Tensor* grad_weight,
    Tensor* grad_bias)
{
  DCHECK_EQ(grad_out.numel(), M * N);
  DCHECK_EQ(input.numel(), M * N);
  DCHECK_EQ(mean.numel(), M);
  DCHECK_EQ(rstd.numel(), M);
  DCHECK(!weight.defined() || weight.numel() == N);
  const T* grad_out_data = grad_out.template data_ptr<T>();
  const T* input_data = input.template data_ptr<T>();
  const T* mean_data = mean.template data_ptr<T>();
  const T* rstd_data = rstd.template data_ptr<T>();
  const T* weight_data = weight.defined() ? weight.template data_ptr<T>() : nullptr;
  T* grad_input_data = grad_input->defined() ? grad_input->template data_ptr<T>() : nullptr;
  T* grad_weight_data = grad_weight->defined() ? grad_weight->template data_ptr<T>() : nullptr;
  if (grad_weight_data != nullptr) {
    std::memset(grad_weight_data, 0, N * sizeof(T));
  }
  T* grad_bias_data = grad_bias->defined() ? grad_bias->template data_ptr<T>() : nullptr;
  if (grad_bias_data != nullptr) {
    std::memset(grad_bias_data, 0, N * sizeof(T));
  }
  const T scale = T(1) / static_cast<T>(N);
  const bool weight_null = weight_data == nullptr;
  for (int64_t i = 0; i < M; ++i) {
    const T* grad_out_ptr = grad_out_data + i * N;
    const T* input_ptr = input_data + i * N;
    if (grad_input_data != nullptr) {
      T* grad_input_ptr = grad_input_data + i * N;
      T ds = 0;
      T db = 0;
      for (int64_t j = 0; j < N; ++j) {
        const T weight_v = weight_null ? T(1) : weight_data[j];
        ds += grad_out_ptr[j] * input_ptr[j] * weight_v;
        db += grad_out_ptr[j] * weight_v;
      }
      const T a = rstd_data[i];
      const T b = (db * mean_data[i] - ds) * a * a * a * scale;
      const T c = -b * mean_data[i] - db * a * scale;
      for (int64_t j = 0; j < N; ++j) {
        const T weight_v = weight_null ? T(1) : weight_data[j];
        grad_input_ptr[j] = a * grad_out_ptr[j] * weight_v + b * input_ptr[j] + c;
      }
    }
    if (grad_weight_data != nullptr) {
      const T a = rstd_data[i];
      const T b = -a * mean_data[i];
      for (int64_t j = 0; j < N; ++j) {
        grad_weight_data[j] += grad_out_ptr[j] * (a * input_ptr[j] + b);
      }
    }
    if (grad_bias_data != nullptr) {
      for (int64_t j = 0; j < N; ++j) {
        grad_bias_data[j] += grad_out_ptr[j];
      }
    }
  }
}

void LayerNormBackwardKernelImpl(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    int64_t M,
    int64_t N,
    Tensor* grad_input,
    Tensor* grad_weight,
    Tensor* grad_bias)
{
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "LayerNormBackwardKernelImpl", [&]() {
        LayerNormBackwardKernelImplInternal<scalar_t>(
            grad_out, input, mean, rstd, weight, M, N, grad_input, grad_weight, grad_bias);
      });
}

} // namespace

REGISTER_DISPATCH(LayerNormKernel, &LayerNormKernelImpl);
REGISTER_DISPATCH(LayerNormBackwardKernel, &LayerNormBackwardKernelImpl);

} // namespace native
} // namespace at
