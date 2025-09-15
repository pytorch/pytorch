#pragma once
#include <ATen/native/zendnn/ZenDNN_utils.hpp>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#endif
#include <c10/util/Logging.h>
#include <cstdint>
#include <functional> // For std::reference_wrapper, std::ref, std::cref
#include <iostream>
#include <optional> // For std::optional, std::nullopt
#include <unordered_map>

#if AT_ZENDNN_ENABLED()
namespace at::native {
using namespace zendnnl::interface;

inline std::vector<int64_t> get_2d_size_for_tensor(
    const at::Tensor& inp_tensor) {
  const int64_t dim = inp_tensor.dim();
  std::vector<int64_t> output_size(2);
  output_size[0] = inp_tensor.numel() / inp_tensor.size(dim - 1);
  output_size[1] = inp_tensor.size(dim - 1);
  return output_size;
}

inline at::Tensor get_2d_view(const at::Tensor& tensor) {
  auto stride = tensor.strides();
  if (!std::is_sorted(stride.begin(), stride.end(), std::greater<int64_t>())) {
    auto new_tensor = tensor.clone(at::MemoryFormat::Contiguous)
                          .view(get_2d_size_for_tensor(tensor));
    return new_tensor;
  }
  return tensor.view(get_2d_size_for_tensor(tensor));
}

inline std::vector<int64_t> compute_linear_output_sizes(
    const at::Tensor& input,
    const at::Tensor& weights) {
  auto input_size = input.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  auto weights_last_dim_size = weights.size(weights.dim() - 1);
  output_size.emplace_back(weights_last_dim_size);
  return output_size;
}
// Returns output strides for linear (input @ weights) and linear operations
inline std::vector<int64_t> compute_linear_output_strides(
    const std::vector<int64_t>& output_size) {
  std::vector<int64_t> output_strides(output_size.size(), 1);
  for (int i = output_size.size() - 2; i >= 0; --i) {
    output_strides[i] = output_strides[i + 1] * output_size[i + 1];
  }
  return output_strides;
}

inline at::Tensor create_linear_output_tensor(
    const at::Tensor input,
    const at::Tensor weight) {
  auto output_size = compute_linear_output_sizes(input, weight.t());
  auto output_strides = compute_linear_output_strides(output_size);
  at::Tensor result = at::detail::empty_strided_cpu(
      output_size, output_strides, input.options());
  return result.is_contiguous() ? result : result.contiguous();
}

inline void check_args_for_linear(
    const at::Tensor& input,
    const at::Tensor& weights) {
  TORCH_CHECK(
      (input.dim() != 1 && weights.dim() != 1),
      "1d dims are not supported yet.");
  get_zendnn_dtype(input);
}

inline void check_tensor_sizes_for_linear(
    const at::Tensor& input,
    const at::Tensor& weights,
    const at::Tensor& bias,
    const at::Tensor& result) {
  const int input_dim = input.dim();
  const int weights_dim = weights.dim();
  TORCH_CHECK(
      (input_dim == 2 && weights_dim == 2),
      "unsupported dims for input and weights");
  const auto input_sizes = input.sizes();
  const auto weights_sizes = weights.sizes();
  TORCH_CHECK(
      input_sizes[input_dim - 1] == weights_sizes[input_dim - 2],
      "Tensor shapes incompatible for linear");
  if (bias.defined()) {
    TORCH_CHECK(
        bias.dim() == 1 && bias.size(0) == weights_sizes[1],
        "bias shape incompatible with linear");
  }
}

inline void check_tensor_dtypes_for_linear(
    const at::Tensor& input,
    const at::Tensor& weights,
    const at::Tensor& bias,
    const at::Tensor& result) {
  auto is_fp32 = [](const at::Tensor& t) {
    return t.scalar_type() == c10::ScalarType::Float;
  };
  auto is_bf16 = [](const at::Tensor& t) {
    return t.scalar_type() == c10::ScalarType::BFloat16;
  };
  bool all_fp32 = is_fp32(input) && is_fp32(weights) && is_fp32(result) &&
      (!bias.defined() || is_fp32(bias));
  bool all_bf16 = is_bf16(input) && is_bf16(weights) && is_bf16(result) &&
      (!bias.defined() || is_bf16(bias));
  TORCH_CHECK(
      all_fp32 ^ all_bf16,
      "All tensors must have consistent dtype and zendnn linear only supports Float and BFloat16");
  if (all_bf16) {
    TORCH_CHECK(
        zendnn_bf16_device_check(),
        "zendnn linear bf16 path needs cpu support avx512bf16");
  }
}

inline void set_linear_context_attributes(
    matmul_context_t& matmul_context,
    tensor_t& weights,
    std::optional<std::reference_wrapper<tensor_t>> bias_opt_ref =
        std::nullopt) {
  matmul_context.set_param("weights", weights);
  if (bias_opt_ref.has_value()) {
    tensor_t& bias = bias_opt_ref->get();
    matmul_context.set_param("bias", bias);
  }
}
} // namespace at::native
#endif // AT_ZENDNN_ENABLED()
