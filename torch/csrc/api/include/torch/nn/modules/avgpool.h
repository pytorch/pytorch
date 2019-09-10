#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {

/// Options for a `D`-dimensional avgpool module.
template <size_t D>
struct AvgPoolOptions {
  AvgPoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}

  /// the size of the window to take an average over
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// the stride of the window. Default value is `kernel_size`
  TORCH_ARG(ExpandingArray<D>, stride);

  /// implicit zero padding to be added on both sides
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// when True, will use `ceil` instead of `floor` to compute the output shape
  TORCH_ARG(bool, ceil_mode) = false;

  /// when True, will include the zero-padding in the averaging calculation
  TORCH_ARG(bool, count_include_pad) = true;

  /// if specified, it will be used as divisor, otherwise `kernel_size` will be used
  TORCH_ARG(c10::optional<int64_t>, divisor_override) = c10::nullopt;
};

/// Base class for all (dimension-specialized) avgpool modules.
template <size_t D, typename Derived>
class TORCH_API AvgPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  AvgPoolImpl(ExpandingArray<D> kernel_size)
      : AvgPoolImpl(AvgPoolOptions<D>(kernel_size)) {}
  explicit AvgPoolImpl(AvgPoolOptions<D> options);

  void reset() override;

  /// Pretty prints the `AvgPool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  AvgPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AvgPool1d to learn
/// about the exact behavior of this module.
class TORCH_API AvgPool1dImpl : public AvgPoolImpl<1, AvgPool1dImpl> {
 public:
  using AvgPoolImpl<1, AvgPool1dImpl>::AvgPoolImpl;
  Tensor forward(const Tensor& input);
};

/// `AvgPoolOptions` specialized for 1-D avgpool.
using AvgPool1dOptions = AvgPoolOptions<1>;

/// A `ModuleHolder` subclass for `AvgPool1dImpl`.
/// See the documentation for `AvgPool1dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AvgPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AvgPool2d to learn
/// about the exact behavior of this module.
class TORCH_API AvgPool2dImpl : public AvgPoolImpl<2, AvgPool2dImpl> {
 public:
  using AvgPoolImpl<2, AvgPool2dImpl>::AvgPoolImpl;
  Tensor forward(const Tensor& input);
};

/// `AvgPoolOptions` specialized for 2-D avgpool.
using AvgPool2dOptions = AvgPoolOptions<2>;

/// A `ModuleHolder` subclass for `AvgPool2dImpl`.
/// See the documentation for `AvgPool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AvgPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies avgpool over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AvgPool3d to learn
/// about the exact behavior of this module.
class TORCH_API AvgPool3dImpl : public AvgPoolImpl<3, AvgPool3dImpl> {
 public:
  using AvgPoolImpl<3, AvgPool3dImpl>::AvgPoolImpl;
  Tensor forward(const Tensor& input);
};

/// `AvgPoolOptions` specialized for 3-D avgpool.
using AvgPool3dOptions = AvgPoolOptions<3>;

/// A `ModuleHolder` subclass for `AvgPool2dImpl`.
/// See the documentation for `AvgPool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AvgPool3d);

} // namespace nn
} // namespace torch
