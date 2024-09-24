#pragma once

#include <c10/util/irange.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/modules/utils.h>
#include <torch/nn/options/pooling.h>

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor avg_pool1d(
    const Tensor& input,
    ExpandingArray<1> kernel_size,
    ExpandingArray<1> stride,
    ExpandingArray<1> padding,
    bool ceil_mode,
    bool count_include_pad) {
  return torch::avg_pool1d(
      input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.avg_pool1d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::AvgPool1dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::avg_pool1d(x, F::AvgPool1dFuncOptions(3).stride(2));
/// ```
inline Tensor avg_pool1d(
    const Tensor& input,
    const AvgPool1dFuncOptions& options) {
  return avg_pool1d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor avg_pool2d(
    const Tensor& input,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> stride,
    ExpandingArray<2> padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
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
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.avg_pool2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::AvgPool2dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::avg_pool2d(x, F::AvgPool2dFuncOptions(3).stride(2));
/// ```
inline Tensor avg_pool2d(
    const Tensor& input,
    const AvgPool2dFuncOptions& options) {
  return detail::avg_pool2d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.ceil_mode(),
      options.count_include_pad(),
      options.divisor_override());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor avg_pool3d(
    const Tensor& input,
    ExpandingArray<3> kernel_size,
    ExpandingArray<3> stride,
    ExpandingArray<3> padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
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
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.avg_pool3d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::AvgPool3dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::avg_pool3d(x, F::AvgPool3dFuncOptions(3).stride(2));
/// ```
inline Tensor avg_pool3d(
    const Tensor& input,
    const AvgPool3dFuncOptions& options) {
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

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor max_pool1d(
    const Tensor& input,
    ExpandingArray<1> kernel_size,
    ExpandingArray<1> stride,
    ExpandingArray<1> padding,
    ExpandingArray<1> dilation,
    bool ceil_mode) {
  return torch::max_pool1d(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_pool1d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::MaxPool1dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool1d(x, F::MaxPool1dFuncOptions(3).stride(2));
/// ```
inline Tensor max_pool1d(
    const Tensor& input,
    const MaxPool1dFuncOptions& options) {
  return detail::max_pool1d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline std::tuple<Tensor, Tensor> max_pool1d_with_indices(
    const Tensor& input,
    ExpandingArray<1> kernel_size,
    ExpandingArray<1> stride,
    ExpandingArray<1> padding,
    ExpandingArray<1> dilation,
    bool ceil_mode) {
  return torch::max_pool1d_with_indices(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation for `torch::nn::functional::MaxPool1dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool1d_with_indices(x, F::MaxPool1dFuncOptions(3).stride(2));
/// ```
inline std::tuple<Tensor, Tensor> max_pool1d_with_indices(
    const Tensor& input,
    const MaxPool1dFuncOptions& options) {
  return detail::max_pool1d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor max_pool2d(
    const Tensor& input,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> stride,
    ExpandingArray<2> padding,
    ExpandingArray<2> dilation,
    bool ceil_mode) {
  return torch::max_pool2d(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_pool2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::MaxPool2dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool2d(x, F::MaxPool2dFuncOptions(3).stride(2));
/// ```
inline Tensor max_pool2d(
    const Tensor& input,
    const MaxPool2dFuncOptions& options) {
  return detail::max_pool2d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline std::tuple<Tensor, Tensor> max_pool2d_with_indices(
    const Tensor& input,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> stride,
    ExpandingArray<2> padding,
    ExpandingArray<2> dilation,
    bool ceil_mode) {
  return torch::max_pool2d_with_indices(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation for `torch::nn::functional::MaxPool2dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool2d_with_indices(x, F::MaxPool2dFuncOptions(3).stride(2));
/// ```
inline std::tuple<Tensor, Tensor> max_pool2d_with_indices(
    const Tensor& input,
    const MaxPool2dFuncOptions& options) {
  return detail::max_pool2d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor max_pool3d(
    const Tensor& input,
    ExpandingArray<3> kernel_size,
    ExpandingArray<3> stride,
    ExpandingArray<3> padding,
    ExpandingArray<3> dilation,
    bool ceil_mode) {
  return torch::max_pool3d(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_pool3d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::MaxPool3dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool3d(x, F::MaxPool3dFuncOptions(3).stride(2));
/// ```
inline Tensor max_pool3d(
    const Tensor& input,
    const MaxPool3dFuncOptions& options) {
  return detail::max_pool3d(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline std::tuple<Tensor, Tensor> max_pool3d_with_indices(
    const Tensor& input,
    ExpandingArray<3> kernel_size,
    ExpandingArray<3> stride,
    ExpandingArray<3> padding,
    ExpandingArray<3> dilation,
    bool ceil_mode) {
  return torch::max_pool3d_with_indices(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation for `torch::nn::functional::MaxPool3dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_pool3d_with_indices(x, F::MaxPool3dFuncOptions(3).stride(2));
/// ```
inline std::tuple<Tensor, Tensor> max_pool3d_with_indices(
    const Tensor& input,
    const MaxPool3dFuncOptions& options) {
  return detail::max_pool3d_with_indices(
      input,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.dilation(),
      options.ceil_mode());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline std::tuple<Tensor, Tensor> adaptive_max_pool1d_with_indices(
    const Tensor& input,
    ExpandingArray<1> output_size) {
  return torch::adaptive_max_pool1d(input, output_size);
}
} // namespace detail

/// See the documentation for
/// `torch::nn::functional::AdaptiveMaxPool1dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool1d_with_indices(x, F::AdaptiveMaxPool1dFuncOptions(3));
/// ```
inline std::tuple<Tensor, Tensor> adaptive_max_pool1d_with_indices(
    const Tensor& input,
    const AdaptiveMaxPool1dFuncOptions& options) {
  return detail::adaptive_max_pool1d_with_indices(input, options.output_size());
}

namespace detail {
inline Tensor adaptive_max_pool1d(
    const Tensor& input,
    ExpandingArray<1> output_size) {
  return std::get<0>(adaptive_max_pool1d_with_indices(input, output_size));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_max_pool1d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::AdaptiveMaxPool1dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool1d(x, F::AdaptiveMaxPool1dFuncOptions(3));
/// ```
inline Tensor adaptive_max_pool1d(
    const Tensor& input,
    const AdaptiveMaxPool1dFuncOptions& options) {
  return detail::adaptive_max_pool1d(input, options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<2> output_size) {
  auto output_size_ =
      torch::nn::modules::utils::_list_with_default(output_size, input.sizes());
  return torch::adaptive_max_pool2d(input, output_size_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation for
/// `torch::nn::functional::AdaptiveMaxPool2dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool2d_with_indices(x, F::AdaptiveMaxPool2dFuncOptions(3));
/// ```
inline std::tuple<Tensor, Tensor> adaptive_max_pool2d_with_indices(
    const Tensor& input,
    const AdaptiveMaxPool2dFuncOptions& options) {
  return detail::adaptive_max_pool2d_with_indices(input, options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor adaptive_max_pool2d(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<2> output_size) {
  return std::get<0>(adaptive_max_pool2d_with_indices(input, output_size));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_max_pool2d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::AdaptiveMaxPool2dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool2d(x, F::AdaptiveMaxPool2dFuncOptions(3));
/// ```
inline Tensor adaptive_max_pool2d(
    const Tensor& input,
    const AdaptiveMaxPool2dFuncOptions& options) {
  return detail::adaptive_max_pool2d(input, options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline std::tuple<Tensor, Tensor> adaptive_max_pool3d_with_indices(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<3> output_size) {
  auto output_size_ =
      torch::nn::modules::utils::_list_with_default(output_size, input.sizes());
  return torch::adaptive_max_pool3d(input, output_size_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation for
/// `torch::nn::functional::AdaptiveMaxPool3dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool3d_with_indices(x, F::AdaptiveMaxPool3dFuncOptions(3));
/// ```
inline std::tuple<Tensor, Tensor> adaptive_max_pool3d_with_indices(
    const Tensor& input,
    const AdaptiveMaxPool3dFuncOptions& options) {
  return detail::adaptive_max_pool3d_with_indices(input, options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor adaptive_max_pool3d(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<3> output_size) {
  return std::get<0>(adaptive_max_pool3d_with_indices(input, output_size));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_max_pool3d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::AdaptiveMaxPool3dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_max_pool3d(x, F::AdaptiveMaxPool3dFuncOptions(3));
/// ```
inline Tensor adaptive_max_pool3d(
    const Tensor& input,
    const AdaptiveMaxPool3dFuncOptions& options) {
  return detail::adaptive_max_pool3d(input, options.output_size());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor adaptive_avg_pool1d(
    const Tensor& input,
    ExpandingArray<1> output_size) {
  return torch::adaptive_avg_pool1d(input, output_size);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_avg_pool1d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::AdaptiveAvgPool1dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_avg_pool1d(x, F::AdaptiveAvgPool1dFuncOptions(3));
/// ```
inline Tensor adaptive_avg_pool1d(
    const Tensor& input,
    const AdaptiveAvgPool1dFuncOptions& options) {
  return detail::adaptive_avg_pool1d(input, options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor adaptive_avg_pool2d(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<2> output_size) {
  auto output_size_ =
      torch::nn::modules::utils::_list_with_default(output_size, input.sizes());
  return torch::adaptive_avg_pool2d(input, output_size_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_avg_pool2d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::AdaptiveAvgPool2dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_avg_pool2d(x, F::AdaptiveAvgPool2dFuncOptions(3));
/// ```
inline Tensor adaptive_avg_pool2d(
    const Tensor& input,
    const AdaptiveAvgPool2dFuncOptions& options) {
  return detail::adaptive_avg_pool2d(input, options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor adaptive_avg_pool3d(
    const Tensor& input,
    ExpandingArrayWithOptionalElem<3> output_size) {
  auto output_size_ =
      torch::nn::modules::utils::_list_with_default(output_size, input.sizes());
  return torch::adaptive_avg_pool3d(input, output_size_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.adaptive_avg_pool3d
/// about the exact behavior of this functional.
///
/// See the documentation for
/// `torch::nn::functional::AdaptiveAvgPool3dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::adaptive_avg_pool3d(x, F::AdaptiveAvgPool3dFuncOptions(3));
/// ```
inline Tensor adaptive_avg_pool3d(
    const Tensor& input,
    const AdaptiveAvgPool3dFuncOptions& options) {
  return detail::adaptive_avg_pool3d(input, options.output_size());
}

// ============================================================================

inline std::vector<int64_t> _unpool_output_size(
    const Tensor& input,
    const IntArrayRef& kernel_size,
    const IntArrayRef& stride,
    const IntArrayRef& padding,
    const std::optional<std::vector<int64_t>>& output_size) {
  auto input_size = input.sizes();
  std::vector<int64_t> default_size;
  for (const auto d : c10::irange(kernel_size.size())) {
    default_size.push_back(
        (input_size[input_size.size() - kernel_size.size() + d] - 1) *
            stride[d] +
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
      TORCH_CHECK(
          false,
          "output_size should be a sequence containing ",
          kernel_size.size(),
          " or ",
          kernel_size.size() + 2,
          " elements, but it has a length of '",
          output_size_.size(),
          "'");
    }
    for (const auto d : c10::irange(kernel_size.size())) {
      const auto min_size = default_size[d] - stride[d];
      const auto max_size = default_size[d] + stride[d];
      if (!(min_size <= output_size_[d] && output_size_[d] <= max_size)) {
        TORCH_CHECK(
            false,
            "invalid output_size ",
            output_size_,
            " (dim ",
            d,
            " must be between ",
            min_size,
            " and ",
            max_size,
            ")");
      }
    }
    return output_size_;
  }
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor max_unpool1d(
    const Tensor& input,
    const Tensor& indices,
    ExpandingArray<1> kernel_size,
    ExpandingArray<1> stride,
    ExpandingArray<1> padding,
    const std::optional<std::vector<int64_t>>& output_size) {
  auto output_size_ =
      _unpool_output_size(input, kernel_size, stride, padding, output_size);
  output_size_.push_back(1);
  return torch::max_unpool2d(
             input.unsqueeze(-1), indices.unsqueeze(-1), output_size_)
      .squeeze(-1);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_unpool1d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::MaxUnpool1dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool1d(x, indices,
/// F::MaxUnpool1dFuncOptions(3).stride(2).padding(1));
/// ```
inline Tensor max_unpool1d(
    const Tensor& input,
    const Tensor& indices,
    const MaxUnpool1dFuncOptions& options) {
  return detail::max_unpool1d(
      input,
      indices,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor max_unpool2d(
    const Tensor& input,
    const Tensor& indices,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> stride,
    ExpandingArray<2> padding,
    const std::optional<std::vector<int64_t>>& output_size) {
  auto output_size_ =
      _unpool_output_size(input, kernel_size, stride, padding, output_size);

  return torch::max_unpool2d(input, indices, output_size_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_unpool2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::MaxUnpool2dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool2d(x, indices,
/// F::MaxUnpool2dFuncOptions(3).stride(2).padding(1));
/// ```
inline Tensor max_unpool2d(
    const Tensor& input,
    const Tensor& indices,
    const MaxUnpool2dFuncOptions& options) {
  return detail::max_unpool2d(
      input,
      indices,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.output_size());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor max_unpool3d(
    const Tensor& input,
    const Tensor& indices,
    ExpandingArray<3> kernel_size,
    ExpandingArray<3> stride,
    ExpandingArray<3> padding,
    const std::optional<std::vector<int64_t>>& output_size) {
  auto output_size_ =
      _unpool_output_size(input, kernel_size, stride, padding, output_size);

  return torch::max_unpool3d(input, indices, output_size_, stride, padding);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.max_unpool3d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::MaxUnpool3dFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::max_unpool3d(x, indices, F::MaxUnpool3dFuncOptions(3));
/// ```
inline Tensor max_unpool3d(
    const Tensor& input,
    const Tensor& indices,
    const MaxUnpool3dFuncOptions& options) {
  return detail::max_unpool3d(
      input,
      indices,
      options.kernel_size(),
      options.stride(),
      options.padding(),
      options.output_size());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline std::tuple<Tensor, Tensor> fractional_max_pool2d_with_indices(
    const Tensor& input,
    const ExpandingArray<2>& kernel_size,
    const std::optional<ExpandingArray<2>>& output_size,
    const std::optional<ExpandingArray<2, double>>& output_ratio,
    const Tensor& _random_samples) {
  if (output_size == std::nullopt && output_ratio == std::nullopt) {
    TORCH_CHECK(
        false,
        "fractional_max_pool2d requires specifying either ",
        "an output_size or an output_ratio");
  }
  std::optional<ExpandingArray<2>> output_size_ = output_size;
  if (output_size_ == std::nullopt) {
    TORCH_INTERNAL_ASSERT(output_ratio != std::nullopt);
    output_size_ = {
        (int64_t)(static_cast<double>(input.size(-2)) *
                  (*output_ratio.value())[0]),
        (int64_t)(static_cast<double>(input.size(-1)) *
                  (*output_ratio.value())[1])};
  }

  Tensor _random_samples_ = _random_samples;
  if (!_random_samples_.defined()) {
    auto n_batch = input.dim() == 3 ? 1 : input.size(0);
    _random_samples_ = torch::rand(
        {n_batch, input.size(-3), 2},
        torch::TensorOptions().dtype(input.dtype()).device(input.device()));
  }
  return torch::fractional_max_pool2d(
      input, kernel_size, *output_size_, _random_samples_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation for
/// `torch::nn::functional::FractionalMaxPool2dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool2d_with_indices(x,
/// F::FractionalMaxPool2dFuncOptions(3).output_size(2));
/// ```
inline std::tuple<Tensor, Tensor> fractional_max_pool2d_with_indices(
    const Tensor& input,
    const FractionalMaxPool2dFuncOptions& options) {
  return detail::fractional_max_pool2d_with_indices(
      input,
      options.kernel_size(),
      options.output_size(),
      options.output_ratio(),
      options._random_samples());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor fractional_max_pool2d(
    const Tensor& input,
    ExpandingArray<2> kernel_size,
    std::optional<ExpandingArray<2>> output_size,
    std::optional<ExpandingArray<2, double>> output_ratio,
    const Tensor& _random_samples) {
  return std::get<0>(fractional_max_pool2d_with_indices(
      input, kernel_size, output_size, output_ratio, _random_samples));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation for
/// `torch::nn::functional::FractionalMaxPool2dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool2d(x,
/// F::FractionalMaxPool2dFuncOptions(3).output_size(2));
/// ```
inline Tensor fractional_max_pool2d(
    const Tensor& input,
    const FractionalMaxPool2dFuncOptions& options) {
  return detail::fractional_max_pool2d(
      input,
      options.kernel_size(),
      options.output_size(),
      options.output_ratio(),
      options._random_samples());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline std::tuple<Tensor, Tensor> fractional_max_pool3d_with_indices(
    const Tensor& input,
    const ExpandingArray<3>& kernel_size,
    const std::optional<ExpandingArray<3>>& output_size,
    const std::optional<ExpandingArray<3, double>>& output_ratio,
    const Tensor& _random_samples) {
  if (output_size == std::nullopt && output_ratio == std::nullopt) {
    TORCH_CHECK(
        false,
        "fractional_max_pool3d requires specifying either ",
        "an output_size or an output_ratio");
  }

  std::optional<ExpandingArray<3>> output_size_ = output_size;
  if (output_size_ == std::nullopt) {
    TORCH_INTERNAL_ASSERT(output_ratio != std::nullopt);
    output_size_ = {
        (int64_t)(static_cast<double>(input.size(-3)) *
                  (*output_ratio.value())[0]),
        (int64_t)(static_cast<double>(input.size(-2)) *
                  (*output_ratio.value())[1]),
        (int64_t)(static_cast<double>(input.size(-1)) *
                  (*output_ratio.value())[2])};
  }

  Tensor _random_samples_ = _random_samples;
  if (!_random_samples_.defined()) {
    auto n_batch = input.dim() == 4 ? 1 : input.size(0);
    _random_samples_ = torch::rand(
        {n_batch, input.size(-4), 3},
        torch::TensorOptions().dtype(input.dtype()).device(input.device()));
  }
  return torch::fractional_max_pool3d(
      input, kernel_size, *output_size_, _random_samples_);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation for
/// `torch::nn::functional::FractionalMaxPool3dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool3d_with_indices(x,
/// F::FractionalMaxPool3dFuncOptions(3).output_size(2));
/// ```
inline std::tuple<Tensor, Tensor> fractional_max_pool3d_with_indices(
    const Tensor& input,
    const FractionalMaxPool3dFuncOptions& options) {
  return detail::fractional_max_pool3d_with_indices(
      input,
      options.kernel_size(),
      options.output_size(),
      options.output_ratio(),
      options._random_samples());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor fractional_max_pool3d(
    const Tensor& input,
    ExpandingArray<3> kernel_size,
    std::optional<ExpandingArray<3>> output_size,
    std::optional<ExpandingArray<3, double>> output_ratio,
    const Tensor& _random_samples) {
  return std::get<0>(fractional_max_pool3d_with_indices(
      input, kernel_size, output_size, output_ratio, _random_samples));
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation for
/// `torch::nn::functional::FractionalMaxPool3dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::fractional_max_pool3d(x,
/// F::FractionalMaxPool3dFuncOptions(3).output_size(2));
/// ```
inline Tensor fractional_max_pool3d(
    const Tensor& input,
    const FractionalMaxPool3dFuncOptions& options) {
  return detail::fractional_max_pool3d(
      input,
      options.kernel_size(),
      options.output_size(),
      options.output_ratio(),
      options._random_samples());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor lp_pool1d(
    const Tensor& input,
    double norm_type,
    ExpandingArray<1> kernel_size,
    ExpandingArray<1> stride,
    bool ceil_mode) {
  Tensor out = detail::avg_pool1d(
      input.pow(norm_type),
      kernel_size,
      stride,
      /*padding=*/0,
      ceil_mode,
      /*count_include_pad=*/true);

  return (torch::sign(out) * relu(torch::abs(out)))
      .mul((*kernel_size)[0])
      .pow(1. / norm_type);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.lp_pool1d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::LPPool1dFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::lp_pool1d(x, F::LPPool1dFuncOptions(2, 3).stride(2));
/// ```
inline Tensor lp_pool1d(
    const Tensor& input,
    const LPPool1dFuncOptions& options) {
  return detail::lp_pool1d(
      input,
      options.norm_type(),
      options.kernel_size(),
      options.stride(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor lp_pool2d(
    const Tensor& input,
    double norm_type,
    ExpandingArray<2> kernel_size,
    ExpandingArray<2> stride,
    bool ceil_mode) {
  int kw = (*kernel_size)[0];
  int kh = (*kernel_size)[1];
  Tensor out = detail::avg_pool2d(
      input.pow(norm_type),
      kernel_size,
      stride,
      /*padding=*/0,
      ceil_mode,
      /*count_include_pad=*/true,
      /*divisor_override=*/std::nullopt);

  return (torch::sign(out) * relu(torch::abs(out)))
      .mul(kw * kh)
      .pow(1. / norm_type);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.lp_pool2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::LPPool2dFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::lp_pool2d(x, F::LPPool2dFuncOptions(2, {2, 3}).stride(2));
/// ```
inline Tensor lp_pool2d(
    const Tensor& input,
    const LPPool2dFuncOptions& options) {
  return detail::lp_pool2d(
      input,
      options.norm_type(),
      options.kernel_size(),
      options.stride(),
      options.ceil_mode());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor lp_pool3d(
    const Tensor& input,
    double norm_type,
    ExpandingArray<3> kernel_size,
    ExpandingArray<3> stride,
    bool ceil_mode) {
  int kd = (*kernel_size)[0];
  int kw = (*kernel_size)[1];
  int kh = (*kernel_size)[2];
  Tensor out = detail::avg_pool3d(
      input.pow(norm_type),
      kernel_size,
      stride,
      /*padding=*/0,
      ceil_mode,
      /*count_include_pad=*/true,
      /*divisor_override=*/std::nullopt);

  return (torch::sign(out) * relu(torch::abs(out)))
      .mul(kd * kw * kh)
      .pow(1. / norm_type);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.lp_pool3d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::LPPool3dFuncOptions` class
/// to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::lp_pool3d(x, F::LPPool3dFuncOptions(3, {3, 3, 5}).stride(3));
/// ```
inline Tensor lp_pool3d(
    const Tensor& input,
    const LPPool3dFuncOptions& options) {
  return detail::lp_pool3d(
      input,
      options.norm_type(),
      options.kernel_size(),
      options.stride(),
      options.ceil_mode());
}

} // namespace functional
} // namespace nn
} // namespace torch
