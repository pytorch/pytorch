#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {

/// Options for a `D`-dimensional maxpool module.
template <size_t D>
struct MaxPoolOptions {
  MaxPoolOptions(ExpandingArray<D> kernel_size)
      : kernel_size_(kernel_size), stride_(kernel_size) {}

  /// the size of the window to take a max over
  TORCH_ARG(ExpandingArray<D>, kernel_size);

  /// the stride of the window. Default value is `kernel_size
  TORCH_ARG(ExpandingArray<D>, stride);

  /// implicit zero padding to be added on both sides
  TORCH_ARG(ExpandingArray<D>, padding) = 0;

  /// a parameter that controls the stride of elements in the window
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;

  /// if true, will return the max indices along with the outputs. Useful
  /// for `MaxUnpool1d` later
  TORCH_ARG(bool, return_indices) = false;

  /// when True, will use `ceil` instead of `floor` to compute the output shape
  TORCH_ARG(bool, ceil_mode) = false;
};

/// Base class for all (dimension-specialized) maxpool modules.
template <size_t D, typename Derived>
class TORCH_API MaxPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  MaxPoolImpl(ExpandingArray<D> kernel_size)
      : MaxPoolImpl(MaxPoolOptions<D>(kernel_size)) {}
  explicit MaxPoolImpl(MaxPoolOptions<D> options);

  void reset() override;

  /// Pretty prints the `MaxPool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  MaxPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxpool over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MaxPool1d to learn
/// about the exact behavior of this module.
class TORCH_API MaxPool1dImpl : public MaxPoolImpl<1, MaxPool1dImpl> {
 public:
  using MaxPoolImpl<1, MaxPool1dImpl>::MaxPoolImpl;
  Tensor forward(const Tensor& input);
};

/// `MaxPoolOptions` specialized for 1-D maxpool.
using MaxPool1dOptions = MaxPoolOptions<1>;

/// A `ModuleHolder` subclass for `MaxPool1dImpl`.
/// See the documentation for `MaxPool1dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MaxPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxpool over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MaxPool2d to learn
/// about the exact behavior of this module.
class TORCH_API MaxPool2dImpl : public MaxPoolImpl<2, MaxPool2dImpl> {
 public:
  using MaxPoolImpl<2, MaxPool2dImpl>::MaxPoolImpl;
  Tensor forward(const Tensor& input);
};

/// `MaxPoolOptions` specialized for 2-D maxpool.
using MaxPool2dOptions = MaxPoolOptions<2>;

/// A `ModuleHolder` subclass for `MaxPool2dImpl`.
/// See the documentation for `MaxPool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MaxPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxpool over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MaxPool3d to learn
/// about the exact behavior of this module.
class TORCH_API MaxPool3dImpl : public MaxPoolImpl<3, MaxPool3dImpl> {
 public:
  using MaxPoolImpl<3, MaxPool3dImpl>::MaxPoolImpl;
  Tensor forward(const Tensor& input);
};

/// `MaxPoolOptions` specialized for 3-D maxpool.
using MaxPool3dOptions = MaxPoolOptions<3>;

/// A `ModuleHolder` subclass for `MaxPool3dImpl`.
/// See the documentation for `MaxPool3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MaxPool3d);

} // namespace nn
} // namespace torch
