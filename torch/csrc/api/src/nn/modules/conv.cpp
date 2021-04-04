#include <torch/nn/functional/conv.h>
#include <torch/nn/functional/padding.h>
#include <torch/nn/modules/conv.h>

#include <torch/enum.h>
#include <torch/expanding_array.h>
#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace F = torch::nn::functional;

F::PadFuncOptions::mode_t _get_pad_mode_from_conv_padding_mode(torch::nn::detail::conv_padding_mode_t conv_padding_mode) {
  F::PadFuncOptions::mode_t pad_mode;
  if (c10::get_if<torch::enumtype::kReflect>(&conv_padding_mode)) {
    pad_mode = torch::kReflect;
  } else if (c10::get_if<torch::enumtype::kReplicate>(&conv_padding_mode)) {
    pad_mode = torch::kReplicate;
  } else if (c10::get_if<torch::enumtype::kCircular>(&conv_padding_mode)) {
    pad_mode = torch::kCircular;
  } else {
    TORCH_CHECK(false, "Unsupported conv padding mode: ", torch::enumtype::get_enum_name(conv_padding_mode));
  }
  return pad_mode;
}

namespace torch {
namespace nn {
Conv1dImpl::Conv1dImpl(
    Conv1dOptions options_)
    : ConvNdImpl(
        detail::ConvNdOptions<1>(
          /*in_channels=*/options_.in_channels(),
          /*out_channels=*/options_.out_channels(),
          /*kernel_size=*/options_.kernel_size())
          .stride(options_.stride())
          .padding(options_.padding())
          .dilation(options_.dilation())
          .transposed(false)
          .output_padding(0)
          .groups(options_.groups())
          .bias(options_.bias())
          .padding_mode(options_.padding_mode())) {}

Tensor Conv1dImpl::forward(const Tensor& input) {
  if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
    return F::detail::conv1d(
      F::pad(input, F::PadFuncOptions(_reversed_padding_repeated_twice).mode(_get_pad_mode_from_conv_padding_mode(options.padding_mode()))),
      weight, bias,
      options.stride(),
      /*padding=*/0,
      options.dilation(),
      options.groups());
  }
  return F::detail::conv1d(
    input,
    weight,
    bias,
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

Conv2dImpl::Conv2dImpl(
    Conv2dOptions options_)
    : ConvNdImpl(
        detail::ConvNdOptions<2>(
          /*in_channels=*/options_.in_channels(),
          /*out_channels=*/options_.out_channels(),
          /*kernel_size=*/options_.kernel_size())
          .stride(options_.stride())
          .padding(options_.padding())
          .dilation(options_.dilation())
          .transposed(false)
          .output_padding(0)
          .groups(options_.groups())
          .bias(options_.bias())
          .padding_mode(options_.padding_mode())) {}

Tensor Conv2dImpl::_conv_forward(const Tensor& input, const Tensor& weight) {
  if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
    return F::detail::conv2d(
      F::pad(input, F::PadFuncOptions(_reversed_padding_repeated_twice).mode(_get_pad_mode_from_conv_padding_mode(options.padding_mode()))),
      weight, bias,
      options.stride(),
      /*padding=*/0,
      options.dilation(),
      options.groups());
  }
  return F::detail::conv2d(
    input,
    weight,
    bias,
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

Tensor Conv2dImpl::forward(const Tensor& input) {
  return _conv_forward(input, weight);
}

Conv3dImpl::Conv3dImpl(
    Conv3dOptions options_)
    : ConvNdImpl(
        detail::ConvNdOptions<3>(
          /*in_channels=*/options_.in_channels(),
          /*out_channels=*/options_.out_channels(),
          /*kernel_size=*/options_.kernel_size())
          .stride(options_.stride())
          .padding(options_.padding())
          .dilation(options_.dilation())
          .transposed(false)
          .output_padding(0)
          .groups(options_.groups())
          .bias(options_.bias())
          .padding_mode(options_.padding_mode())) {}

Tensor Conv3dImpl::forward(const Tensor& input) {
  if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
    return F::detail::conv3d(
      F::pad(input, F::PadFuncOptions(_reversed_padding_repeated_twice).mode(_get_pad_mode_from_conv_padding_mode(options.padding_mode()))),
      weight, bias,
      options.stride(),
      /*padding=*/0,
      options.dilation(),
      options.groups());
  }
  return F::detail::conv3d(
    input,
    weight,
    bias,
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

template class ConvNdImpl<1, Conv1dImpl>;
template class ConvNdImpl<2, Conv2dImpl>;
template class ConvNdImpl<3, Conv3dImpl>;

// ============================================================================

template <size_t D, typename Derived>
std::vector<int64_t> ConvTransposeNdImpl<D, Derived>::_output_padding(
    const Tensor& input, const c10::optional<at::IntArrayRef>& output_size,
    const ExpandingArray<D>& stride, const ExpandingArray<D>& padding,
    const ExpandingArray<D>& kernel_size) {
  std::vector<int64_t> ret;
  c10::optional<at::IntArrayRef> output_size_ = output_size;

  if (output_size_ == c10::nullopt) {
    ret = at::IntArrayRef(this->options.output_padding()).vec();
  } else {
    auto k = input.dim() - 2;
    if (output_size_.value().size() == k + 2) {
      output_size_ = output_size_.value().slice(2);
    }
    if (output_size_.value().size() != k) {
      TORCH_CHECK(false,
        "output_size must have ", k, " or ", k + 2, " elements (got ", output_size_.value().size(), ")");
    }

    std::vector<int64_t> min_sizes;
    std::vector<int64_t> max_sizes;
    for (int64_t d = 0; d < k; d++) {
      int64_t dim_size = ((input.sizes()[d + 2] - 1) * (*stride)[d] - 2 * (*padding)[d] + (*kernel_size)[d]);
      min_sizes.push_back(dim_size);
      max_sizes.push_back(min_sizes[d] + (*stride)[d] - 1);
    }

    for (size_t i = 0; i < output_size_.value().size(); i++) {
      int64_t size = output_size_.value()[i];
      int64_t min_size = min_sizes[i];
      int64_t max_size = max_sizes[i];
      if (size < min_size || size > max_size) {
        TORCH_CHECK(false,
          "requested an output size of ", output_size_.value(), ", but valid sizes range "
          "from ", min_sizes, " to ", max_sizes, " (for an input of ", input.sizes().slice(2), ")");
      }
    }

    for (int64_t d = 0; d < k; d++) {
      ret.push_back(output_size_.value()[d] - min_sizes[d]);
    }
  }
  return ret;
}

ConvTranspose1dImpl::ConvTranspose1dImpl(
    ConvTranspose1dOptions options_)
    : ConvTransposeNdImpl(
        detail::ConvNdOptions<1>(
          /*in_channels=*/options_.in_channels(),
          /*out_channels=*/options_.out_channels(),
          /*kernel_size=*/options_.kernel_size())
          .stride(options_.stride())
          .padding(options_.padding())
          .dilation(options_.dilation())
          .transposed(true)
          .output_padding(options_.output_padding())
          .groups(options_.groups())
          .bias(options_.bias())
          .padding_mode(options_.padding_mode())) {}

Tensor ConvTranspose1dImpl::forward(
    const Tensor& input, const c10::optional<at::IntArrayRef>& output_size) {
  if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
    TORCH_CHECK(false, "Only `zeros` padding mode is supported for ConvTranspose1d");
  }

  const auto & pad = padding();
  std::vector<int64_t> output_padding = _output_padding(
    input, output_size, options.stride(), pad, options.kernel_size());

  return F::detail::conv_transpose1d(
    input, weight, bias, options.stride(), pad,
    output_padding, options.groups(), options.dilation());
}

ConvTranspose2dImpl::ConvTranspose2dImpl(
    ConvTranspose2dOptions options_)
    : ConvTransposeNdImpl(detail::ConvNdOptions<2>(
          /*in_channels=*/options_.in_channels(),
          /*out_channels=*/options_.out_channels(),
          /*kernel_size=*/options_.kernel_size())
          .stride(options_.stride())
          .padding(options_.padding())
          .dilation(options_.dilation())
          .transposed(true)
          .output_padding(options_.output_padding())
          .groups(options_.groups())
          .bias(options_.bias())
          .padding_mode(options_.padding_mode())) {}

Tensor ConvTranspose2dImpl::forward(
    const Tensor& input, const c10::optional<at::IntArrayRef>& output_size) {
  if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
    TORCH_CHECK(false, "Only `zeros` padding mode is supported for ConvTranspose2d");
  }

  const auto & pad = padding();
  std::vector<int64_t> output_padding = _output_padding(
    input, output_size, options.stride(), pad, options.kernel_size());

  return F::detail::conv_transpose2d(
    input, weight, bias, options.stride(), pad,
    output_padding, options.groups(), options.dilation());
}

ConvTranspose3dImpl::ConvTranspose3dImpl(
    ConvTranspose3dOptions options_)
    : ConvTransposeNdImpl(detail::ConvNdOptions<3>(
          /*in_channels=*/options_.in_channels(),
          /*out_channels=*/options_.out_channels(),
          /*kernel_size=*/options_.kernel_size())
          .stride(options_.stride())
          .padding(options_.padding())
          .dilation(options_.dilation())
          .transposed(true)
          .output_padding(options_.output_padding())
          .groups(options_.groups())
          .bias(options_.bias())
          .padding_mode(options_.padding_mode())) {}

Tensor ConvTranspose3dImpl::forward(
    const Tensor& input, const c10::optional<at::IntArrayRef>& output_size) {
  if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
    TORCH_CHECK(false, "Only `zeros` padding mode is supported for ConvTranspose3d");
  }

  const auto & pad = padding();
  std::vector<int64_t> output_padding = _output_padding(
    input, output_size, options.stride(), pad, options.kernel_size());

  return F::detail::conv_transpose3d(
    input, weight, bias, options.stride(), pad,
    output_padding, options.groups(), options.dilation());
}

template class ConvTransposeNdImpl<1, ConvTranspose1dImpl>;
template class ConvTransposeNdImpl<2, ConvTranspose2dImpl>;
template class ConvTransposeNdImpl<3, ConvTranspose3dImpl>;

} // namespace nn
} // namespace torch
