#pragma once

#include <torch/nn/functional/activation.h>
#include <torch/nn/options/pooling.h>

namespace torch {
namespace nn{
namespace functional {

inline Tensor avg_pool1d(const Tensor& input, const AvgPool1dFuncOptions& options) {
  return torch::avg_pool1d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad());
}

inline Tensor avg_pool2d(const Tensor& input, const AvgPool2dFuncOptions& options) {
  return torch::avg_pool2d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad(),
      options.divisor_override());
}

inline Tensor avg_pool3d(const Tensor& input, const AvgPool3dFuncOptions& options) {
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

inline Tensor max_pool1d(const Tensor& input, const MaxPool1dFuncOptions& options) {
   return torch::max_pool1d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

inline std::tuple<Tensor, Tensor> max_pool1d_with_indices(const Tensor& input, const MaxPool1dFuncOptions& options) {
  return torch::max_pool1d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

inline Tensor max_pool2d(const Tensor& input, const MaxPool2dFuncOptions& options) {
  return torch::max_pool2d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

inline std::tuple<Tensor, Tensor> max_pool2d_with_indices(const Tensor& input, const MaxPool2dFuncOptions& options) {
  return torch::max_pool2d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

inline Tensor max_pool3d(const Tensor& input, const MaxPool3dFuncOptions& options) {
  return torch::max_pool3d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

inline std::tuple<Tensor, Tensor> max_pool3d_with_indices(const Tensor& input, const MaxPool3dFuncOptions& options) {
  return torch::max_pool3d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

// ============================================================================

inline Tensor adaptive_max_pool1d(const Tensor& input,
  const AdaptiveMaxPool1dFuncOptions& options) {
   return std::get<0>(torch::adaptive_max_pool1d(input, options.output_size()));
}

inline std::tuple<Tensor, Tensor> adaptive_max_pool1d_with_indices(
  const Tensor& input, const AdaptiveMaxPool1dFuncOptions& options) {
   return torch::adaptive_max_pool1d(input, options.output_size());
}

inline Tensor adaptive_max_pool2d(const Tensor& input,
  const AdaptiveMaxPool2dFuncOptions& options) {
   return std::get<0>(torch::adaptive_max_pool2d(input, options.output_size()));
}

inline std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices(
  const Tensor& input, const AdaptiveMaxPool2dFuncOptions& options) {
   return torch::adaptive_max_pool2d(input, options.output_size());
}

inline Tensor adaptive_max_pool3d(const Tensor& input,
  const AdaptiveMaxPool3dFuncOptions& options) {
   return std::get<0>(torch::adaptive_max_pool3d(input, options.output_size()));
}

inline std::tuple<Tensor, Tensor> adaptive_max_pool3d_with_indices(
  const Tensor& input, const AdaptiveMaxPool3dFuncOptions& options) {
   return torch::adaptive_max_pool3d(input, options.output_size());
}

// ============================================================================

inline Tensor adaptive_avg_pool1d(const Tensor& input,
  const AdaptiveAvgPool1dFuncOptions& options) {
   return torch::adaptive_avg_pool1d(input, options.output_size());
}

inline Tensor adaptive_avg_pool2d(const Tensor& input,
  const AdaptiveAvgPool2dFuncOptions& options) {
   return torch::adaptive_avg_pool2d(input, options.output_size());
}

inline Tensor adaptive_avg_pool3d(const Tensor& input,
  const AdaptiveAvgPool3dFuncOptions& options) {
   return torch::adaptive_avg_pool3d(input, options.output_size());
}

// ============================================================================

inline std::vector<int64_t> _unpool_output_size(const Tensor& input,
  const IntArrayRef& kernel_size, const IntArrayRef& stride,
  const IntArrayRef& padding, const c10::optional<IntArrayRef>& output_size) {
  auto input_size = input.sizes();
  std::vector<int64_t> default_size;
  for (size_t d = 0; d < kernel_size.size(); d++) {
    default_size.push_back((input_size[d + 2] - 1) * stride[d] +
                            kernel_size[d] - 2 * padding[d]);
  }
  if (!output_size) {
    return default_size;
  } else {
    std::vector<int64_t> output_size_;
    if (output_size->size() == kernel_size.size() + 2) {
      output_size_ = output_size->slice(2).vec();
    }
    if (output_size_.size() != kernel_size.size()) {
      TORCH_CHECK(false, "output_size should be a sequence containing ",
                  kernel_size.size(), " or ", kernel_size.size() + 2,
                  " elements, but it has a length of '",
                  output_size_.size(), "'");
    }
    for (size_t d = 0; d < kernel_size.size(); d++) {
      const auto min_size = default_size[d] - stride[d];
      const auto max_size = default_size[d] + stride[d];
      if (!(min_size <= output_size_[d] && output_size_[d] <= max_size)) {
        TORCH_CHECK(false, "invalid output_size ", output_size_, " (dim ", d,
                    " must be between ", min_size, " and ", max_size, ")");
      }
    }
    return output_size_;
  }
}

inline Tensor max_unpool1d(const Tensor& input, const Tensor& indices,
    const MaxUnpool1dFuncOptions& options,
    const c10::optional<IntArrayRef>& output_size = c10::nullopt) {
  auto output_size_ = _unpool_output_size(input, options.kernel_size(),
                                          options.stride(), options.padding(),
                                          output_size);
  output_size_.push_back(1);
  return torch::max_unpool2d(input.unsqueeze(3), indices.unsqueeze(3),
                             output_size_).squeeze(3);
}

inline Tensor max_unpool2d(const Tensor& input, const Tensor& indices,
  const MaxUnpool2dFuncOptions& options,
  const c10::optional<IntArrayRef>& output_size = c10::nullopt) {
  auto output_size_ = _unpool_output_size(input, options.kernel_size(),
                                          options.stride(), options.padding(),
                                          output_size);

  return torch::max_unpool2d(input, indices, output_size_);
}

inline Tensor max_unpool3d(const Tensor& input, const Tensor& indices,
  const MaxUnpool3dFuncOptions& options,
  const c10::optional<IntArrayRef>& output_size = c10::nullopt) {
  auto output_size_ = _unpool_output_size(input, options.kernel_size(),
                                          options.stride(), options.padding(),
                                          output_size);

  return torch::max_unpool3d(input, indices, output_size_,
                             options.stride(), options.padding());
}

inline Tensor lp_pool1d(const Tensor& input, const LPPool1dFuncOptions& options) {
  Tensor out = avg_pool1d(
    input.pow(options.norm_type()),
    AvgPool1dFuncOptions(options.kernel_size()).stride(options.stride()).padding(0).ceil_mode(options.ceil_mode()));

  return (torch::sign(out) * relu(torch::abs(out))).mul((*options.kernel_size())[0]).pow(1. / options.norm_type());
}

inline Tensor lp_pool2d(const Tensor& input, const LPPool2dFuncOptions& options) {
  int kw = (*options.kernel_size())[0];
  int kh = (*options.kernel_size())[1];
  Tensor out = avg_pool2d(
    input.pow(options.norm_type()),
    AvgPool2dFuncOptions(options.kernel_size()).stride(options.stride()).padding(0).ceil_mode(options.ceil_mode()));

  return (torch::sign(out) * relu(torch::abs(out))).mul(kw * kh).pow(1. / options.norm_type());
}

} // namespace functional
} // namespace nn
} // namespace torch
