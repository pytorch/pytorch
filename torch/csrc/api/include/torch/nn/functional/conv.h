#pragma once

#include <torch/nn/options/conv.h>
#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    ExpandingArray<1> stride,
    ExpandingArray<1> padding,
    ExpandingArray<1> dilation,
    int64_t groups) {
  return torch::conv1d(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.conv1d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::Conv1dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv1d(x, weight, F::Conv1dFuncOptions().stride(1));
/// ```
inline Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const Conv1dFuncOptions& options = {}) {
  return detail::conv1d(
    input,
    weight,
    options.bias(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    ExpandingArray<2> stride,
    ExpandingArray<2> padding,
    ExpandingArray<2> dilation,
    int64_t groups) {
  return torch::conv2d(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.conv2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::Conv2dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv2d(x, weight, F::Conv2dFuncOptions().stride(1));
/// ```
inline Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Conv2dFuncOptions& options = {}) {
  return detail::conv2d(
    input,
    weight,
    options.bias(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor conv3d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    ExpandingArray<3> stride,
    ExpandingArray<3> padding,
    ExpandingArray<3> dilation,
    int64_t groups) {
  return torch::conv3d(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.conv3d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::Conv3dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv3d(x, weight, F::Conv3dFuncOptions().stride(1));
/// ```
inline Tensor conv3d(
    const Tensor& input,
    const Tensor& weight,
    const Conv3dFuncOptions& options = {}) {
  return detail::conv3d(
    input,
    weight,
    options.bias(),
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor conv_transpose1d(const Tensor& input, const Tensor& weight,
                               const Tensor& bias, IntArrayRef stride,
                               IntArrayRef padding, IntArrayRef output_padding,
                               int64_t groups, IntArrayRef dilation) {
  return torch::conv_transpose1d(
    input, weight, bias, stride, padding, output_padding, groups, dilation);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.conv_transpose1d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::ConvTranspose1dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose1d(x, weight, F::ConvTranspose1dFuncOptions().stride(1));
/// ```
inline Tensor conv_transpose1d(const Tensor& input, const Tensor& weight,
                               const ConvTranspose1dFuncOptions& options = {}) {
  return detail::conv_transpose1d(
    input, weight,
    options.bias(), options.stride(),
    options.padding(), options.output_padding(),
    options.groups(), options.dilation());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor conv_transpose2d(const Tensor& input, const Tensor& weight,
                               const Tensor& bias, IntArrayRef stride,
                               IntArrayRef padding, IntArrayRef output_padding,
                               int64_t groups, IntArrayRef dilation) {
  return torch::conv_transpose2d(
    input, weight, bias, stride, padding, output_padding, groups, dilation);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.conv_transpose2d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::ConvTranspose2dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose2d(x, weight, F::ConvTranspose2dFuncOptions().stride(1));
/// ```
inline Tensor conv_transpose2d(const Tensor& input, const Tensor& weight,
                               const ConvTranspose2dFuncOptions& options = {}) {
  return detail::conv_transpose2d(
    input, weight,
    options.bias(), options.stride(),
    options.padding(), options.output_padding(),
    options.groups(), options.dilation());
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor conv_transpose3d(const Tensor& input, const Tensor& weight,
                               const Tensor& bias, IntArrayRef stride,
                               IntArrayRef padding, IntArrayRef output_padding,
                               int64_t groups, IntArrayRef dilation) {
  return torch::conv_transpose3d(
    input, weight, bias, stride, padding, output_padding, groups, dilation);
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.conv_transpose3d
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::ConvTranspose3dFuncOptions` class to learn what
/// optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::conv_transpose3d(x, weight, F::ConvTranspose3dFuncOptions().stride(1));
/// ```
inline Tensor conv_transpose3d(const Tensor& input, const Tensor& weight,
                               const ConvTranspose3dFuncOptions& options = {}) {
  return detail::conv_transpose3d(
    input, weight,
    options.bias(), options.stride(),
    options.padding(), options.output_padding(),
    options.groups(), options.dilation());
}

} // namespace functional
} // namespace nn
} // namespace torch
