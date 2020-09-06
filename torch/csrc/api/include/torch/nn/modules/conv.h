#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/init.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/utils.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) convolution modules.
template <size_t D, typename Derived>
class ConvNdImpl : public torch::nn::Cloneable<Derived> {
 public:
  explicit ConvNdImpl(detail::ConvNdOptions<D> options_) : options(std::move(options_)) {
    reset();
  }

  void reset() override {
    TORCH_CHECK(
      options.in_channels() % options.groups() == 0,
      "in_channels must be divisible by groups");
    TORCH_CHECK(
      options.out_channels() % options.groups() == 0,
      "out_channels must be divisible by groups");

    _reversed_padding_repeated_twice = torch::nn::modules::utils::_reverse_repeat_vector(options.padding(), 2);

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

  void reset_parameters() {
    init::kaiming_uniform_(weight, /*a=*/std::sqrt(5));  // NOLINT(cppcoreguidelines-avoid-magic-numbers)

    if (bias.defined()) {
      int64_t fan_in, fan_out;
      std::tie(fan_in, fan_out) = init::_calculate_fan_in_and_fan_out(weight);
      auto bound = 1 / std::sqrt(fan_in);
      init::uniform_(bias, -bound, bound);
    }
  }

  /// Pretty prints the `Conv{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
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

  /// The options with which this `Module` was constructed.
  detail::ConvNdOptions<D> options;

  /// The learned kernel (or "weight").
  Tensor weight;

  /// The learned bias. Only defined if the `bias` option was true.
  Tensor bias;

 protected:
  std::vector<int64_t> _reversed_padding_repeated_twice;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies convolution over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Conv1d to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::Conv1dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Conv1d model(Conv1dOptions(3, 2, 3).stride(1).bias(false));
/// ```
class TORCH_API Conv1dImpl : public ConvNdImpl<1, Conv1dImpl> {
 public:
  Conv1dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<1> kernel_size)
      : Conv1dImpl(Conv1dOptions(input_channels, output_channels, kernel_size)) {
  }
  explicit Conv1dImpl(Conv1dOptions options_);
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `Conv1dImpl`.
/// See the documentation for `Conv1dImpl` class to learn what methods it
/// provides, and examples of how to use `Conv1d` with `torch::nn::Conv1dOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Conv1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies convolution over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::Conv2dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Conv2d model(Conv2dOptions(3, 2, 3).stride(1).bias(false));
/// ```
class TORCH_API Conv2dImpl : public ConvNdImpl<2, Conv2dImpl> {
 public:
  Conv2dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<2> kernel_size)
      : Conv2dImpl(Conv2dOptions(input_channels, output_channels, kernel_size)) {
  }
  explicit Conv2dImpl(Conv2dOptions options_);
  Tensor forward(const Tensor& input);

 protected:
  Tensor _conv_forward(const Tensor& input, const Tensor& weight);
};

/// A `ModuleHolder` subclass for `Conv2dImpl`.
/// See the documentation for `Conv2dImpl` class to learn what methods it
/// provides, and examples of how to use `Conv2d` with `torch::nn::Conv2dOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Conv2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conv3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies convolution over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.Conv3d to learn about
/// the exact behavior of this module.
///
/// See the documentation for `torch::nn::Conv3dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Conv3d model(Conv3dOptions(3, 2, 3).stride(1).bias(false));
/// ```
class TORCH_API Conv3dImpl : public ConvNdImpl<3, Conv3dImpl> {
 public:
  Conv3dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<3> kernel_size)
      : Conv3dImpl(Conv3dOptions(input_channels, output_channels, kernel_size)) {
  }
  explicit Conv3dImpl(Conv3dOptions options_);
  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `Conv3dImpl`.
/// See the documentation for `Conv3dImpl` class to learn what methods it
/// provides, and examples of how to use `Conv3d` with `torch::nn::Conv3dOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Conv3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~ ConvTranspose ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Base class for all (dimension-specialized) convolution transpose modules.
template <size_t D, typename Derived>
class ConvTransposeNdImpl : public ConvNdImpl<D, Derived> {
 public:
  using torch::nn::ConvNdImpl<D, Derived>::ConvNdImpl;

  /// Pretty prints the `ConvTranspose{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ConvTranspose" << D << "d"
           << "(" << this->options.in_channels()
           << ", " << this->options.out_channels()
           << ", kernel_size=" << this->options.kernel_size()
           << ", stride=" << this->options.stride();
    if (*this->options.padding() != *ExpandingArray<D>(0)) {
      stream << ", padding=" << this->options.padding();
    }
    if (*this->options.dilation() != *ExpandingArray<D>(1)) {
      stream << ", dilation=" << this->options.dilation();
    }
    if (*this->options.output_padding() != *ExpandingArray<D>(0)) {
      stream << ", output_padding=" << this->options.output_padding();
    }
    if (this->options.groups() != 1) {
      stream << ", groups=" << this->options.groups();
    }
    if (!this->options.bias()) {
      stream << ", bias=" << std::boolalpha << false;
    }
    if (!c10::get_if<enumtype::kZeros>(&this->options.padding_mode())) {
      stream << ", padding_mode=" << enumtype::get_enum_name(this->options.padding_mode());
    }
    stream << ")";
  }

 protected:
  std::vector<int64_t> _output_padding(
      const Tensor& input, const c10::optional<at::IntArrayRef>& output_size,
      const ExpandingArray<D>& stride, const ExpandingArray<D>& padding,
      const ExpandingArray<D>& kernel_size);
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConvTranspose1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the ConvTranspose1d function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose1d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ConvTranspose1dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ConvTranspose1d model(ConvTranspose1dOptions(3, 2, 3).stride(1).bias(false));
/// ```
class TORCH_API ConvTranspose1dImpl : public ConvTransposeNdImpl<1, ConvTranspose1dImpl> {
 public:
  ConvTranspose1dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<1> kernel_size)
      : ConvTranspose1dImpl(ConvTranspose1dOptions(input_channels, output_channels, kernel_size)) {
  }
  explicit ConvTranspose1dImpl(ConvTranspose1dOptions options_);
  Tensor forward(const Tensor& input,
                 const c10::optional<at::IntArrayRef>& output_size = c10::nullopt);
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(c10::optional<at::IntArrayRef>())})
};

/// A `ModuleHolder` subclass for `ConvTranspose1dImpl`.
/// See the documentation for `ConvTranspose1dImpl` class to learn what methods it
/// provides, and examples of how to use `ConvTranspose1d` with `torch::nn::ConvTranspose1dOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(ConvTranspose1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConvTranspose2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the ConvTranspose2d function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose2d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ConvTranspose2dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ConvTranspose2d model(ConvTranspose2dOptions(3, 2, 3).stride(1).bias(false));
/// ```
class TORCH_API ConvTranspose2dImpl : public ConvTransposeNdImpl<2, ConvTranspose2dImpl> {
 public:
  ConvTranspose2dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<2> kernel_size)
      : ConvTranspose2dImpl(ConvTranspose2dOptions(input_channels, output_channels, kernel_size)) {
  }
  explicit ConvTranspose2dImpl(ConvTranspose2dOptions options_);
  Tensor forward(const Tensor& input,
                 const c10::optional<at::IntArrayRef>& output_size = c10::nullopt);
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(c10::optional<at::IntArrayRef>())})
};

/// A `ModuleHolder` subclass for `ConvTranspose2dImpl`.
/// See the documentation for `ConvTranspose2dImpl` class to learn what methods it
/// provides, and examples of how to use `ConvTranspose2d` with `torch::nn::ConvTranspose2dOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(ConvTranspose2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConvTranspose3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the ConvTranspose3d function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose3d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ConvTranspose3dOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ConvTranspose3d model(ConvTranspose3dOptions(2, 2, 2).stride(1).bias(false));
/// ```
class TORCH_API ConvTranspose3dImpl : public ConvTransposeNdImpl<3, ConvTranspose3dImpl> {
 public:
  ConvTranspose3dImpl(
      int64_t input_channels,
      int64_t output_channels,
      ExpandingArray<3> kernel_size)
      : ConvTranspose3dImpl(ConvTranspose3dOptions(input_channels, output_channels, kernel_size)) {
  }
  explicit ConvTranspose3dImpl(ConvTranspose3dOptions options_);
  Tensor forward(const Tensor& input,
                 const c10::optional<at::IntArrayRef>& output_size = c10::nullopt);
 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(c10::optional<at::IntArrayRef>())})
};

/// A `ModuleHolder` subclass for `ConvTranspose3dImpl`.
/// See the documentation for `ConvTranspose3dImpl` class to learn what methods it
/// provides, and examples of how to use `ConvTranspose3d` with `torch::nn::ConvTranspose3dOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(ConvTranspose3d);

} // namespace nn
} // namespace torch
