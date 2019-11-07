#pragma once

#include <torch/nn/functional/activation.h>
#include <torch/nn/options/pooling.h>

namespace torch {
namespace nn {
namespace functional {

namespace detail {
inline Tensor avg_pool1d(const Tensor& input,
                         ExpandingArray<1> kernel_size,
                         ExpandingArray<1> stride,
                         ExpandingArray<1> padding = 0,
                         bool ceil_mode = false,
                         bool count_include_pad = true) {
  return torch::avg_pool1d(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad);
}
} // namespace detail

inline Tensor avg_pool1d(const Tensor& input, AvgPool1dFuncOptions options) {
  return avg_pool1d(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.ceil_mode(),
    options.count_include_pad());
}

namespace detail {
inline Tensor avg_pool2d(const Tensor& input,
                         ExpandingArray<2> kernel_size,
                         ExpandingArray<2> stride,
                         ExpandingArray<2> padding = 0,
                         bool ceil_mode = false,
                         bool count_include_pad = true,
                         c10::optional<int64_t> divisor_override = c10::nullopt) {
  return torch::avg_pool2d(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
}
} // namespace detail

inline Tensor avg_pool2d(const Tensor& input, AvgPool2dFuncOptions options) {
  return detail::avg_pool2d(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.ceil_mode(),
    options.count_include_pad(),
    options.divisor_override());
}

namespace detail {
inline Tensor avg_pool3d(const Tensor& input,
                         ExpandingArray<3> kernel_size,
                         ExpandingArray<3> stride,
                         ExpandingArray<3> padding = 0,
                         bool ceil_mode = false,
                         bool count_include_pad = true,
                         c10::optional<int64_t> divisor_override = c10::nullopt) {
  return torch::avg_pool3d(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
}
} // namespace detail

inline Tensor avg_pool3d(const Tensor& input, AvgPool3dFuncOptions options) {
  return detail::avg_pool3d(
    input,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.ceil_mode(),
    options.count_include_pad(),
    options.divisor_override());
}

// ============================================================================

namespace detail {
inline Tensor max_pool1d(const Tensor& input,
                         ExpandingArray<1> kernel_size,
                         ExpandingArray<1> stride,
                         ExpandingArray<1> padding = 0,
                         ExpandingArray<1> dilation = 1,
                         bool ceil_mode = false) {
   return torch::max_pool1d(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
}
} // namespace detail

inline Tensor max_pool1d(const Tensor& input, MaxPool1dFuncOptions options) {
   return detail::max_pool1d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

namespace detail {
inline std::tuple<Tensor, Tensor> max_pool1d_with_indices(
  const Tensor& input,
  ExpandingArray<1> kernel_size,
  ExpandingArray<1> stride,
  ExpandingArray<1> padding = 0,
  ExpandingArray<1> dilation = 1,
  bool ceil_mode = false) {
  return torch::max_pool1d_with_indices(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
}
} // namespace detail

inline std::tuple<Tensor, Tensor> max_pool1d_with_indices(const Tensor& input, MaxPool1dFuncOptions options) {
  return detail::max_pool1d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

namespace detail {
inline Tensor max_pool2d(const Tensor& input,
                         ExpandingArray<2> kernel_size,
                         ExpandingArray<2> stride,
                         ExpandingArray<2> padding = 0,
                         ExpandingArray<2> dilation = 1,
                         bool ceil_mode = false) {
  return torch::max_pool2d(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
}
} // namespace detail

inline Tensor max_pool2d(const Tensor& input, MaxPool2dFuncOptions options) {
  return detail::max_pool2d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

namespace detail {
inline std::tuple<Tensor, Tensor> max_pool2d_with_indices(
  const Tensor& input,
  ExpandingArray<2> kernel_size,
  ExpandingArray<2> stride,
  ExpandingArray<2> padding = 0,
  ExpandingArray<2> dilation = 1,
  bool ceil_mode = false) {
  return torch::max_pool2d_with_indices(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
}
} // namespace detail

inline std::tuple<Tensor, Tensor> max_pool2d_with_indices(const Tensor& input, MaxPool2dFuncOptions options) {
  return detail::max_pool2d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

namespace detail {
inline Tensor max_pool3d(const Tensor& input,
                         ExpandingArray<3> kernel_size,
                         ExpandingArray<3> stride,
                         ExpandingArray<3> padding = 0,
                         ExpandingArray<3> dilation = 1,
                         bool ceil_mode = false) {
  return torch::max_pool3d(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
}
} // namespace detail

inline Tensor max_pool3d(const Tensor& input, MaxPool3dFuncOptions options) {
  return detail::max_pool3d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

namespace detail {
inline std::tuple<Tensor, Tensor> max_pool3d_with_indices(
  const Tensor& input,
  ExpandingArray<3> kernel_size,
  ExpandingArray<3> stride,
  ExpandingArray<3> padding = 0,
  ExpandingArray<3> dilation = 1,
  bool ceil_mode = false) {
  return torch::max_pool3d_with_indices(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
}
} // namespace detail

inline std::tuple<Tensor, Tensor> max_pool3d_with_indices(const Tensor& input, MaxPool3dFuncOptions options) {
  return detail::max_pool3d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

// ============================================================================

namespace detail {
inline Tensor adaptive_max_pool1d(const Tensor& input,
  ExpandingArray<1> output_size) {
   return std::get<0>(torch::adaptive_max_pool1d(input, output_size));
}
} // namespace detail

inline Tensor adaptive_max_pool1d(const Tensor& input,
  AdaptiveMaxPool1dFuncOptions options) {
   return detail::adaptive_max_pool1d(input, options.output_size());
}

namespace detail {
inline std::tuple<Tensor, Tensor> adaptive_max_pool1d_with_indices(
  const Tensor& input, ExpandingArray<1> output_size) {
   return torch::adaptive_max_pool1d(input, output_size);
}
} // namespace detail

inline std::tuple<Tensor, Tensor> adaptive_max_pool1d_with_indices(
  const Tensor& input, AdaptiveMaxPool1dFuncOptions options) {
   return detail::adaptive_max_pool1d_with_indices(input, options.output_size());
}

namespace detail {
inline Tensor adaptive_max_pool2d(const Tensor& input,
  ExpandingArray<2> output_size) {
   return std::get<0>(torch::adaptive_max_pool2d(input, output_size));
}
} // namespace detail

inline Tensor adaptive_max_pool2d(const Tensor& input,
  AdaptiveMaxPool2dFuncOptions options) {
   return detail::adaptive_max_pool2d(input, options.output_size());
}

namespace detail {
inline std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices(
  const Tensor& input, ExpandingArray<2> output_size) {
   return torch::adaptive_max_pool2d(input, output_size);
}
} // namespace detail

inline std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices(
  const Tensor& input, AdaptiveMaxPool2dFuncOptions options) {
   return detail::adaptive_max_pool2d_with_indices(input, options.output_size());
}

namespace detail {
inline Tensor adaptive_max_pool3d(const Tensor& input,
  ExpandingArray<3> output_size) {
   return std::get<0>(torch::adaptive_max_pool3d(input, output_size));
}
} // namespace detail

inline Tensor adaptive_max_pool3d(const Tensor& input,
  AdaptiveMaxPool3dFuncOptions options) {
   return detail::adaptive_max_pool3d(input, options.output_size());
}

namespace detail {
inline std::tuple<Tensor, Tensor> adaptive_max_pool3d_with_indices(
  const Tensor& input, ExpandingArray<3> output_size) {
   return torch::adaptive_max_pool3d(input, output_size);
}
} // namespace detail

inline std::tuple<Tensor, Tensor> adaptive_max_pool3d_with_indices(
  const Tensor& input, AdaptiveMaxPool3dFuncOptions options) {
   return detail::adaptive_max_pool3d_with_indices(input, options.output_size());
}

// ============================================================================

namespace detail {
inline Tensor adaptive_avg_pool1d(const Tensor& input,
  ExpandingArray<1> output_size) {
   return torch::adaptive_avg_pool1d(input, output_size);
}
} // namespace detail

inline Tensor adaptive_avg_pool1d(const Tensor& input,
  AdaptiveAvgPool1dFuncOptions options) {
   return detail::adaptive_avg_pool1d(input, options.output_size());
}

namespace detail {
inline Tensor adaptive_avg_pool2d(const Tensor& input,
  ExpandingArray<2> output_size) {
   return torch::adaptive_avg_pool2d(input, output_size);
}
} // namespace detail

inline Tensor adaptive_avg_pool2d(const Tensor& input,
  AdaptiveAvgPool2dFuncOptions options) {
   return detail::adaptive_avg_pool2d(input, options.output_size());
}

namespace detail {
inline Tensor adaptive_avg_pool3d(const Tensor& input,
  ExpandingArray<3> output_size) {
   return torch::adaptive_avg_pool3d(input, output_size);
}
} // namespace detail

inline Tensor adaptive_avg_pool3d(const Tensor& input,
  AdaptiveAvgPool3dFuncOptions options) {
   return detail::adaptive_avg_pool3d(input, options.output_size());
}

// ============================================================================

inline std::vector<int64_t> _unpool_output_size(const Tensor& input,
  const IntArrayRef& kernel_size, const IntArrayRef& stride,
  const IntArrayRef& padding, const c10::optional<std::vector<int64_t>>& output_size) {
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
      output_size_ = IntArrayRef(*output_size).slice(2).vec();
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

namespace detail {
inline Tensor max_unpool1d(
    const Tensor& input,
    const Tensor& indices,
    ExpandingArray<1> kernel_size,
    ExpandingArray<1> stride,
    ExpandingArray<1> padding = 0,
    const c10::optional<std::vector<int64_t>>& output_size = c10::nullopt) {
  auto output_size_ = _unpool_output_size(input, kernel_size,
                                          stride, padding,
                                          output_size);
  output_size_.push_back(1);
  return torch::max_unpool2d(input.unsqueeze(3), indices.unsqueeze(3),
                             output_size_).squeeze(3);
}
} // namespace detail

inline Tensor max_unpool1d(const Tensor& input, const Tensor& indices,
    MaxUnpool1dFuncOptions options) {
  return detail::max_unpool1d(
    input,
    indices,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.output_size());
}

namespace detail {
inline Tensor max_unpool2d(
  const Tensor& input,
  const Tensor& indices,
  ExpandingArray<2> kernel_size,
  ExpandingArray<2> stride,
  ExpandingArray<2> padding = 0,
  const c10::optional<std::vector<int64_t>>& output_size = c10::nullopt) {
  auto output_size_ = _unpool_output_size(input, kernel_size,
                                          stride, padding,
                                          output_size);

  return torch::max_unpool2d(input, indices, output_size_);
}
} // namespace detail

inline Tensor max_unpool2d(const Tensor& input, const Tensor& indices,
  MaxUnpool2dFuncOptions options) {
  return detail::max_unpool2d(
    input,
    indices,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.output_size());
}

namespace detail {
inline Tensor max_unpool3d(
  const Tensor& input,
  const Tensor& indices,
  ExpandingArray<3> kernel_size,
  ExpandingArray<3> stride,
  ExpandingArray<3> padding = 0,
  const c10::optional<std::vector<int64_t>>& output_size = c10::nullopt) {
  auto output_size_ = _unpool_output_size(input, kernel_size,
                                          stride, padding,
                                          output_size);

  return torch::max_unpool3d(input, indices, output_size_,
                             stride, padding);
}
} // namespace detail

inline Tensor max_unpool3d(const Tensor& input, const Tensor& indices,
  MaxUnpool3dFuncOptions options) {
  return detail::max_unpool3d(
    input,
    indices,
    options.kernel_size(),
    options.stride(),
    options.padding(),
    options.output_size());
}

// ============================================================================

namespace detail {
inline Tensor lp_pool1d(
  const Tensor& input,
  float norm_type,
  ExpandingArray<1> kernel_size,
  ExpandingArray<1> stride,
  bool ceil_mode = false) {
  Tensor out = detail::avg_pool1d(
    input.pow(norm_type),
    kernel_size,
    stride,
    /*padding=*/0,
    ceil_mode);

  return (torch::sign(out) * relu(torch::abs(out))).mul((*kernel_size)[0]).pow(1. / norm_type);
}
} // namespace detail

inline Tensor lp_pool1d(const Tensor& input, LPPool1dFuncOptions options) {
  return detail::lp_pool1d(
    input,
    options.norm_type(),
    options.kernel_size(),
    options.stride(),
    options.ceil_mode());
}

namespace detail {
inline Tensor lp_pool2d(
  const Tensor& input,
  float norm_type,
  ExpandingArray<2> kernel_size,
  ExpandingArray<2> stride,
  bool ceil_mode = false) {
  int kw = (*kernel_size)[0];
  int kh = (*kernel_size)[1];
  Tensor out = detail::avg_pool2d(
    input.pow(norm_type),
    kernel_size,
    stride,
    /*padding=*/0,
    ceil_mode);

  return (torch::sign(out) * relu(torch::abs(out))).mul(kw * kh).pow(1. / norm_type);
}
} // namespace detail

inline Tensor lp_pool2d(const Tensor& input, LPPool2dFuncOptions options) {
  return detail::lp_pool2d(
    input,
    options.norm_type(),
    options.kernel_size(),
    options.stride(),
    options.ceil_mode());
}

} // namespace functional
} // namespace nn
} // namespace torch
