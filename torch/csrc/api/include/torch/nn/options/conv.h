#pragma once

#include <torch/arg.h>
#include <torch/enum.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

namespace torch {
namespace nn {

/// Options for a `D`-dimensional convolution module.
template <size_t D>
struct ConvOptions {
  typedef c10::variant<enumtype::kZeros, enumtype::kCircular> padding_mode_t;

  ConvOptions(
      int64_t in_channels,
      int64_t out_channels,
      ExpandingArray<D> kernel_size) :
                in_channels_(in_channels),
                out_channels_(out_channels),
                kernel_size_(std::move(kernel_size)) {}

  /// The number of channels the input volumes will have.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, in_channels);

  /// The number of output channels the convolution should produce.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, out_channels);

  /// The kernel size to use.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// The stride of the convolution.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, stride) = 1;

  /// The padding to add to the input volumes.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// The kernel dilation.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;

  /// If true, convolutions will be transpose convolutions (a.k.a.
  /// deconvolutions).
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, transposed) = false;

  /// For transpose convolutions, the padding to add to output volumes.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(ExpandingArray<D>, output_padding) = 0;

  /// The number of convolution groups.
  /// This parameter __can__ be changed after construction.
  TORCH_ARG(int64_t, groups) = 1;

  /// Whether to add a bias after individual applications of the kernel.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, bias) = true;

  /// Accepted values `zeros` and `circular` Default: `zeros`
  TORCH_ARG(padding_mode_t, padding_mode) = torch::kZeros;

  // FIXME: The following methods are added so that we can merge PR #28917
  // without breaking torchvision builds in CI. After PR #28917 is merged
  // and a new PyTorch nightly is built, @yf225 will open a PR to torchvision
  // to change the following:
  //
  // 1. `with_bias` -> `bias`
  // 2. `input_channels` -> `in_channels`
  // 3. `output_channels` -> `out_channels`
  //
  // and then he will open a PR to PyTorch to remove the methods below.
 public:
  inline auto input_channels(const int64_t& new_input_channels)->decltype(*this) {
    this->in_channels_ = new_input_channels;
    return *this;
  }

  inline auto input_channels(int64_t&& new_input_channels)->decltype(*this) {
    this->in_channels_ = std::move(new_input_channels);
    return *this;
  }

  inline const int64_t& input_channels() const noexcept {
    return this->in_channels_;
  }

  inline auto output_channels(const int64_t& new_output_channels)->decltype(*this) {
    this->out_channels_ = new_output_channels;
    return *this;
  }

  inline auto output_channels(int64_t&& new_output_channels)->decltype(*this) {
    this->out_channels_ = std::move(new_output_channels);
    return *this;
  }

  inline const int64_t& output_channels() const noexcept {
    return this->out_channels_;
  }

  inline auto with_bias(const bool& new_with_bias)->decltype(*this) {
    this->bias_ = new_with_bias;
    return *this;
  }

  inline auto with_bias(bool&& new_with_bias)->decltype(*this) {
    this->bias_ = std::move(new_with_bias);
    return *this;
  }

  inline const bool& with_bias() const noexcept {
    return this->bias_;
  }
};

/// `ConvOptions` specialized for 1-D convolution.
using Conv1dOptions = ConvOptions<1>;

/// `ConvOptions` specialized for 2-D convolution.
using Conv2dOptions = ConvOptions<2>;

/// `ConvOptions` specialized for 3-D convolution.
using Conv3dOptions = ConvOptions<3>;

// ============================================================================

namespace functional {

/// Options for a `D`-dimensional convolution functional.
template <size_t D>
struct ConvFuncOptions {
  /// optional bias of shape `(out_channels)`. Default: ``None``
  TORCH_ARG(torch::Tensor, bias) = Tensor();

  /// The stride of the convolving kernel.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, stride) = 1;

  /// Implicit paddings on both sides of the input.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// The spacing between kernel elements.
  /// For a `D`-dim convolution, must be a single number or a list of `D`
  /// numbers.
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;

  /// Split input into groups, `in_channels` should be divisible by
  /// the number of groups.
  TORCH_ARG(int64_t, groups) = 1;
};

/// `ConvFuncOptions` specialized for 1-D convolution.
using Conv1dFuncOptions = ConvFuncOptions<1>;

/// `ConvFuncOptions` specialized for 2-D convolution.
using Conv2dFuncOptions = ConvFuncOptions<2>;

/// `ConvFuncOptions` specialized for 3-D convolution.
using Conv3dFuncOptions = ConvFuncOptions<3>;

} // namespace functional

} // namespace nn
} // namespace torch
