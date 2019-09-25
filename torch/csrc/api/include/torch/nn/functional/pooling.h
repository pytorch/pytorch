#pragma once

#include <torch/nn/options/pooling.h>

namespace torch {
namespace nn{
namespace functional {

inline Tensor avg_pool1d(const Tensor& input, const AvgPool1dOptions& options) {
  return torch::avg_pool1d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad());
}

inline Tensor avg_pool2d(const Tensor& input, const AvgPool2dOptions& options) {
  return torch::avg_pool2d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad(),
      options.divisor_override());
}

inline Tensor avg_pool3d(const Tensor& input, const AvgPool3dOptions& options) {
  return torch::avg_pool3d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad(),
      options.divisor_override());
}

// ============================================================================

inline Tensor max_pool1d(const Tensor& input, const MaxPool1dOptions& options) {
   return torch::max_pool1d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

inline Tensor max_pool2d(const Tensor& input, const MaxPool2dOptions& options) {
  return torch::max_pool2d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

inline Tensor max_pool3d(const Tensor& input, const MaxPool3dOptions& options) {
  return torch::max_pool3d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

// ============================================================================

inline Tensor adaptive_max_pool1d(const Tensor& input,
  const AdaptiveMaxPool1dOptions& options) {
   return std::get<0>(torch::adaptive_max_pool1d(input, options.output_size()));
}

inline std::tuple<Tensor, Tensor> adaptive_max_pool1d_with_indices(
  const Tensor& input, const AdaptiveMaxPool1dOptions& options) {
   return torch::adaptive_max_pool1d(input, options.output_size());
}

inline Tensor adaptive_max_pool2d(const Tensor& input,
  const AdaptiveMaxPool2dOptions& options) {
   return std::get<0>(torch::adaptive_max_pool2d(input, options.output_size()));
}

inline std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices(
  const Tensor& input, const AdaptiveMaxPool2dOptions& options) {
   return torch::adaptive_max_pool2d(input, options.output_size());
}

inline Tensor adaptive_max_pool3d(const Tensor& input,
  const AdaptiveMaxPool3dOptions& options) {
   return std::get<0>(torch::adaptive_max_pool3d(input, options.output_size()));
}

inline std::tuple<Tensor, Tensor> adaptive_max_pool3d_with_indices(
  const Tensor& input, const AdaptiveMaxPool3dOptions& options) {
   return torch::adaptive_max_pool3d(input, options.output_size());
}

// ============================================================================

inline Tensor adaptive_avg_pool1d(const Tensor& input,
  const AdaptiveAvgPool1dOptions& options) {
   return torch::adaptive_avg_pool1d(input, options.output_size());
}

inline Tensor adaptive_avg_pool2d(const Tensor& input,
  const AdaptiveAvgPool2dOptions& options) {
   return torch::adaptive_avg_pool2d(input, options.output_size());
}

inline Tensor adaptive_avg_pool3d(const Tensor& input,
  const AdaptiveAvgPool3dOptions& options) {
   return torch::adaptive_avg_pool3d(input, options.output_size());
}

} // namespace functional
} // namespace nn
} // namespace torch
