#include <torch/nn/functional/conv.h>
#include <torch/nn/functional/padding.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/functional/conv.h>
#include <torch/nn/init.h>

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

namespace torch {
namespace nn {
template <size_t D, typename Derived>
ConvImpl<D, Derived>::ConvImpl(ConvOptions<D> options_)
    : options(std::move(options_)) {
  reset();
}

template <size_t D, typename Derived>
void ConvImpl<D, Derived>::reset() {
  TORCH_CHECK(
    options.in_channels() % options.groups() == 0,
    "in_channels must be divisible by groups");
  TORCH_CHECK(
    options.out_channels() % options.groups() == 0,
    "out_channels must be divisible by groups");

  if (options.transposed()) {
    std::vector<int64_t> weight_sizes = {
      options.in_channels(),
      options.out_channels() / options.groups()};
    weight_sizes.insert(weight_sizes.end(), (*options.kernel_size()).begin(), (*options.kernel_size()).end());
    weight = this->register_parameter(
      "weight",
      torch::empty(weight_sizes));
  } else {
    std::vector<int64_t> weight_sizes = {
      options.out_channels(),
      options.in_channels() / options.groups()};
    weight_sizes.insert(weight_sizes.end(), (*options.kernel_size()).begin(), (*options.kernel_size()).end());
    weight = this->register_parameter(
      "weight",
      torch::empty(weight_sizes));
  }

  if (options.bias()) {
    bias = this->register_parameter("bias", torch::empty({options.out_channels()}));
  } else {
    this->register_parameter("bias", Tensor(), /*requires_grad=*/false);
  }

  reset_parameters();
}

template <size_t D, typename Derived>
void ConvImpl<D, Derived>::reset_parameters() {
  init::kaiming_uniform_(weight, /*a=*/std::sqrt(5));  // NOLINT(cppcoreguidelines-avoid-magic-numbers)

  if (bias.defined()) {
    int64_t fan_in, fan_out;
    std::tie(fan_in, fan_out) = init::_calculate_fan_in_and_fan_out(weight);
    auto bound = 1 / std::sqrt(fan_in);
    init::uniform_(bias, -bound, bound);
  }
}

template <size_t D, typename Derived>
void ConvImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Conv" << D << "d"
         << "(" << options.in_channels()
         << ", " << options.out_channels()
         << ", kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride();
  if (*options.padding() != *ExpandingArray<D>(0)) {
    stream << ", padding=" << options.padding();
  }
  if (*options.dilation() != *ExpandingArray<D>(1)) {
    stream << ", dilation=" << options.dilation();
  }
  if (*options.output_padding() != *ExpandingArray<D>(0)) {
    stream << ", output_padding=" << options.output_padding();
  }
  if (options.groups() != 1) {
    stream << ", groups=" << options.groups();
  }
  if (!options.bias()) {
    stream << ", bias=" << std::boolalpha << false;
  }
  if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
    stream << ", padding_mode=" << enumtype::get_enum_name(options.padding_mode());
  }
  stream << ")";
}

Conv1dImpl::Conv1dImpl(
    ConvOptions<1> options_)
    : ConvImpl(options_.transposed(false).output_padding(0)) {}

Tensor Conv1dImpl::forward(const Tensor& input) {
  if (c10::get_if<enumtype::kCircular>(&options.padding_mode())) {
    std::vector<int64_t> expanded_padding = {((*options.padding())[0] + 1) / 2, (*options.padding())[0] / 2};
    return F::detail::conv1d(
      F::detail::pad(input, expanded_padding, torch::kCircular, 0),
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
    ConvOptions<2> options_)
    : ConvImpl(options_.transposed(false).output_padding(0)) {}

Tensor Conv2dImpl::forward(const Tensor& input) {
  if (c10::get_if<enumtype::kCircular>(&options.padding_mode())) {
    std::vector<int64_t> expanded_padding = {
      ((*options.padding())[1] + 1) / 2, (*options.padding())[1] / 2,
      ((*options.padding())[0] + 1) / 2, (*options.padding())[0] / 2};
    return F::detail::conv2d(
      F::detail::pad(input, expanded_padding, torch::kCircular, 0),
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

Conv3dImpl::Conv3dImpl(
    ConvOptions<3> options_)
    : ConvImpl(options_.transposed(false).output_padding(0)) {}

Tensor Conv3dImpl::forward(const Tensor& input) {
  if (c10::get_if<enumtype::kCircular>(&options.padding_mode())) {
    std::vector<int64_t> expanded_padding = {
      ((*options.padding())[2] + 1) / 2, (*options.padding())[2] / 2,
      ((*options.padding())[1] + 1) / 2, (*options.padding())[1] / 2,
      ((*options.padding())[0] + 1) / 2, (*options.padding())[0] / 2};
    return F::detail::conv3d(
      F::detail::pad(input, expanded_padding, torch::kCircular, 0),
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

template class ConvImpl<1, Conv1dImpl>;
template class ConvImpl<2, Conv2dImpl>;
template class ConvImpl<3, Conv3dImpl>;

template <size_t D, typename Derived>
ConvTransposeImplBase<D, Derived>::ConvTransposeImplBase(
    const ConvTransposeOptionsBase<D>& options_)
    : options(options_) {
  TORCH_CHECK((options.in_channels() % options.groups()) == 0,
              "in_channels must be divisible by groups");
  TORCH_CHECK((options.out_channels() % options.groups()) == 0,
              "out_channels must be divisible by groups");

  std::vector<int64_t> dims = {
    options.in_channels(), options.out_channels() / options.groups()
  };
  for (auto& d : *(options.kernel_size())) {
    dims.push_back(d);
  }
  weight = this->register_parameter("weight", torch::empty(dims));
  if (options.bias()) {
    bias = this->register_parameter("bias", torch::empty({options.out_channels()}));
  } else {
    bias = this->register_parameter("bias", Tensor());
  }
  this->reset_parameters();
}

template <size_t D, typename Derived>
void ConvTransposeImplBase<D, Derived>::reset_parameters() {
  torch::nn::init::kaiming_uniform_(weight, std::sqrt(5));
  if (bias.defined()) {
    auto fan_in = std::get<0>(torch::nn::init::_calculate_fan_in_and_fan_out(weight));
    double bound = 1 / std::sqrt(fan_in);
    torch::nn::init::uniform_(bias, -bound, bound);
  }
}

template <size_t D, typename Derived>
void ConvTransposeImplBase<D, Derived>::reset() {
  this->reset_parameters();
}

template <size_t D, typename Derived>
void ConvTransposeImplBase<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha
         << "torch::nn::ConvTranspose" << D << "d"
         << "(input_channels=" << options.in_channels()
         << ", output_channels=" << options.out_channels()
         << ", kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride()
         << ", padding=" << options.padding()
         << ", output_padding=" << options.output_padding()
         << ", groups=" << options.groups()
         << ", bias=" << options.bias()
         << ", dilation=" << options.dilation()
         << ", padding_mode=" << options.padding_mode() << ")";
}

template <size_t D, typename Derived>
ExpandingArray<D> ConvTransposeImplBase<D, Derived>::_output_padding(
    const Tensor& input, const std::vector<int64_t>& output_size,
    const ExpandingArray<D>& stride, const ExpandingArray<D>& padding,
    const ExpandingArray<D>& kernel_size) {
  std::vector<int64_t> ret;
  if (output_size.empty()) {
    ret.push_back(0);
  } else {
    auto k = input.dim() - 2;
    std::vector<int64_t> output_size_resized = output_size;
    if (output_size.size() == k + 2) {
      output_size_resized = std::vector<int64_t>(output_size.begin() + 2,
                                                 output_size.end());
    }
    TORCH_CHECK(output_size_resized.size() != k,
                "ouput_size must have %d or %d elements (got %d)",
                k, k + 2, output_size_resized.size());

    std::vector<int64_t> min_sizes;
    std::vector<int64_t> max_sizes;
    for (int d = 0; d < k; d++) {
      int64_t dim_size = ((input.size(d + 2) - 1) * (*stride)[d] - 2 * (*padding)[d] + (*kernel_size)[d]);
      min_sizes.push_back(dim_size);
      max_sizes.push_back(min_sizes[d] + (*stride)[d] - 1);
    }

    for (int i = 0; i < output_size_resized.size(); i++) {
      int64_t size = output_size_resized[i];
      int64_t min_size = min_sizes[i];
      int64_t max_size = max_sizes[i];
      TORCH_CHECK((size < min_size) || (size > max_size),
                  "requested an output size of [%s], but valid sizes range "
                  "from [%s] to [%s] (for an input of [%s])",
                  c10::Join(",", output_size_resized), c10::Join(",", min_sizes),
                  c10::Join(",", max_sizes),
                  c10::Join(",", std::vector<int64_t>(input.sizes().begin() + 2, input.sizes().end())));
    }

    for (int d = 0; d < k; d++) {
      ret.push_back(output_size_resized[d] - min_sizes[d]);
    }
  }

  return ExpandingArray<D>(ret);
}

Tensor ConvTranspose1dImpl::forward(
    const Tensor& input, const std::vector<int64_t>& output_size) {
  TORCH_CHECK(options.padding_mode() == std::string("zeros"),
              "Only `zeros` padding mode is supported for ConvTransposed1d");

  ExpandingArray<1> output_padding = this->_output_padding(
      input, output_size, options.stride(), options.padding(),
      options.kernel_size());
  return F::conv_transpose1d(input, weight, bias, options.stride(),
                             options.padding(), output_padding,
                             options.groups(), options.dilation());
}

template class ConvTransposeImplBase<1, ConvTranspose1dImpl>;

} // namespace nn
} // namespace torch
