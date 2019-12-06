#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/options/pooling.h>
#include <torch/nn/functional/pooling.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) avgpool modules.
template <size_t D, typename Derived>
class TORCH_API AvgPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  AvgPoolImpl(ExpandingArray<D> kernel_size)
      : AvgPoolImpl(AvgPoolOptions<D>(kernel_size)) {}
  explicit AvgPoolImpl(const AvgPoolOptions<D>& options_);

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

/// A `ModuleHolder` subclass for `AvgPool2dImpl`.
/// See the documentation for `AvgPool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AvgPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) maxpool modules.
template <size_t D, typename Derived>
class TORCH_API MaxPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  MaxPoolImpl(ExpandingArray<D> kernel_size)
      : MaxPoolImpl(MaxPoolOptions<D>(kernel_size)) {}
  explicit MaxPoolImpl(const MaxPoolOptions<D>& options_);

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

  /// Returns the outputs and the indices of the max values.
  /// Useful for `torch::nn::MaxUnpool1d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

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

  /// Returns the outputs and the indices of the max values.
  /// Useful for `torch::nn::MaxUnpool2d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

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

  /// Returns the outputs and the indices of the max values.
  /// Useful for `torch::nn::MaxUnpool3d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `MaxPool3dImpl`.
/// See the documentation for `MaxPool3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MaxPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) adaptive maxpool modules.
template <size_t D, typename Derived>
class TORCH_API AdaptiveMaxPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  AdaptiveMaxPoolImpl(ExpandingArray<D> output_size)
      : AdaptiveMaxPoolImpl(AdaptiveMaxPoolOptions<D>(output_size)) {}
  explicit AdaptiveMaxPoolImpl(const AdaptiveMaxPoolOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `AdaptiveMaxPool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  AdaptiveMaxPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive maxpool over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveMaxPool1d
/// to learn about the exact behavior of this module.
class TORCH_API AdaptiveMaxPool1dImpl :
  public AdaptiveMaxPoolImpl<1, AdaptiveMaxPool1dImpl> {
 public:
  using AdaptiveMaxPoolImpl<1, AdaptiveMaxPool1dImpl>::AdaptiveMaxPoolImpl;

  Tensor forward(const Tensor& input);

  /// Returns the indices along with the outputs.
  /// Useful to pass to nn.MaxUnpool1d.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveMaxPool1dImpl`.
/// See the documentation for `AdaptiveMaxPool1dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AdaptiveMaxPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive maxpool over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveMaxPool2d
/// to learn about the exact behavior of this module.
class TORCH_API AdaptiveMaxPool2dImpl :
  public AdaptiveMaxPoolImpl<2, AdaptiveMaxPool2dImpl> {
 public:
  using AdaptiveMaxPoolImpl<2, AdaptiveMaxPool2dImpl>::AdaptiveMaxPoolImpl;

  Tensor forward(const Tensor& input);

  /// Returns the indices along with the outputs.
  /// Useful to pass to nn.MaxUnpool2d.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveMaxPool2dImpl`.
/// See the documentation for `AdaptiveMaxPool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AdaptiveMaxPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveMaxPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive maxpool over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveMaxPool3d
/// to learn about the exact behavior of this module.
class TORCH_API AdaptiveMaxPool3dImpl :
  public AdaptiveMaxPoolImpl<3, AdaptiveMaxPool3dImpl> {
 public:
  using AdaptiveMaxPoolImpl<3, AdaptiveMaxPool3dImpl>::AdaptiveMaxPoolImpl;

  Tensor forward(const Tensor& input);

  /// Returns the indices along with the outputs.
  /// Useful to pass to nn.MaxUnpool3d.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveMaxPool3dImpl`.
/// See the documentation for `AdaptiveMaxPool3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AdaptiveMaxPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) adaptive avgpool modules.
template <size_t D, typename Derived>
class TORCH_API AdaptiveAvgPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  AdaptiveAvgPoolImpl(ExpandingArray<D> output_size)
      : AdaptiveAvgPoolImpl(AdaptiveAvgPoolOptions<D>(output_size)) {}
  explicit AdaptiveAvgPoolImpl(const AdaptiveAvgPoolOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `AdaptiveAvgPool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  AdaptiveAvgPoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveAvgPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive avgpool over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveAvgPool1d
/// to learn about the exact behavior of this module.
class TORCH_API AdaptiveAvgPool1dImpl :
  public AdaptiveAvgPoolImpl<1, AdaptiveAvgPool1dImpl> {
 public:
  using AdaptiveAvgPoolImpl<1, AdaptiveAvgPool1dImpl>::AdaptiveAvgPoolImpl;

  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveAvgPool1dImpl`.
/// See the documentation for `AdaptiveAvgPool1dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AdaptiveAvgPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveAvgPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive avgpool over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveAvgPool2d
/// to learn about the exact behavior of this module.
class TORCH_API AdaptiveAvgPool2dImpl :
  public AdaptiveAvgPoolImpl<2, AdaptiveAvgPool2dImpl> {
 public:
  using AdaptiveAvgPoolImpl<2, AdaptiveAvgPool2dImpl>::AdaptiveAvgPoolImpl;

  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveAvgPool2dImpl`.
/// See the documentation for `AdaptiveAvgPool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AdaptiveAvgPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdaptiveAvgPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies adaptive avgpool over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.AdaptiveAvgPool3d
/// to learn about the exact behavior of this module.
class TORCH_API AdaptiveAvgPool3dImpl :
  public AdaptiveAvgPoolImpl<3, AdaptiveAvgPool3dImpl> {
 public:
  using AdaptiveAvgPoolImpl<3, AdaptiveAvgPool3dImpl>::AdaptiveAvgPoolImpl;

  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `AdaptiveAvgPool3dImpl`.
/// See the documentation for `AdaptiveAvgPool3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(AdaptiveAvgPool3d);

// ============================================================================

/// Base class for all (dimension-specialized) maxunpool modules.
template <size_t D, typename Derived>
class TORCH_API MaxUnpoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  MaxUnpoolImpl(ExpandingArray<D> kernel_size)
      : MaxUnpoolImpl(MaxUnpoolOptions<D>(kernel_size)) {}
  explicit MaxUnpoolImpl(const MaxUnpoolOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `MaxUnpool{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  MaxUnpoolOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxUnpool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxunpool over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MaxUnpool1d to learn
/// about the exact behavior of this module.
class TORCH_API MaxUnpool1dImpl : public MaxUnpoolImpl<1, MaxUnpool1dImpl> {
 public:
  using MaxUnpoolImpl<1, MaxUnpool1dImpl>::MaxUnpoolImpl;
  Tensor forward(const Tensor& input, const Tensor& indices,
                 const c10::optional<std::vector<int64_t>>& output_size = c10::nullopt);
};

/// A `ModuleHolder` subclass for `MaxUnpool1dImpl`.
/// See the documentation for `MaxUnpool1dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MaxUnpool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxUnpool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxunpool over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MaxUnpool2d to learn
/// about the exact behavior of this module.
class TORCH_API MaxUnpool2dImpl : public MaxUnpoolImpl<2, MaxUnpool2dImpl> {
 public:
  using MaxUnpoolImpl<2, MaxUnpool2dImpl>::MaxUnpoolImpl;
  Tensor forward(const Tensor& input, const Tensor& indices,
                 const c10::optional<std::vector<int64_t>>& output_size = c10::nullopt);
};

/// A `ModuleHolder` subclass for `MaxUnpool2dImpl`.
/// See the documentation for `MaxUnpool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MaxUnpool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxUnpool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies maxunpool over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.MaxUnpool3d to learn
/// about the exact behavior of this module.
class TORCH_API MaxUnpool3dImpl : public MaxUnpoolImpl<3, MaxUnpool3dImpl> {
 public:
  using MaxUnpoolImpl<3, MaxUnpool3dImpl>::MaxUnpoolImpl;
  Tensor forward(const Tensor& input, const Tensor& indices,
                 const c10::optional<std::vector<int64_t>>& output_size = c10::nullopt);
};

/// A `ModuleHolder` subclass for `MaxUnpool3dImpl`.
/// See the documentation for `MaxUnpool3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(MaxUnpool3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FractionalMaxPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies fractional maxpool over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.FractionalMaxPool2d to learn
/// about the exact behavior of this module.
class TORCH_API FractionalMaxPool2dImpl : public torch::nn::Cloneable<FractionalMaxPool2dImpl> {
 public:
  FractionalMaxPool2dImpl(ExpandingArray<2> kernel_size)
      : FractionalMaxPool2dImpl(FractionalMaxPool2dOptions(kernel_size)) {}
  explicit FractionalMaxPool2dImpl(const FractionalMaxPool2dOptions& options_);

  void reset() override;

  /// Pretty prints the `FractionalMaxPool2d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// Returns the outputs and the indices of the max values.
  /// Useful for `torch::nn::MaxUnpool2d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);

  /// The options with which this `Module` was constructed.
  FractionalMaxPool2dOptions options;

  Tensor _random_samples;
};

/// A `ModuleHolder` subclass for `FractionalMaxPool2dImpl`.
/// See the documentation for `FractionalMaxPool2dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(FractionalMaxPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FractionalMaxPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies fractional maxpool over a 3-D input.
class TORCH_API FractionalMaxPool3dImpl : public torch::nn::Cloneable<FractionalMaxPool3dImpl> {
 public:
  FractionalMaxPool3dImpl(ExpandingArray<3> kernel_size)
      : FractionalMaxPool3dImpl(FractionalMaxPool3dOptions(kernel_size)) {}
  explicit FractionalMaxPool3dImpl(const FractionalMaxPool3dOptions& options_);

  void reset() override;

  /// Pretty prints the `FractionalMaxPool3d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// Returns the outputs and the indices of the max values.
  /// Useful for `torch::nn::MaxUnpool3d` later.
  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);

  /// The options with which this `Module` was constructed.
  FractionalMaxPool3dOptions options;

  Tensor _random_samples;
};

/// A `ModuleHolder` subclass for `FractionalMaxPool3dImpl`.
/// See the documentation for `FractionalMaxPool3dImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(FractionalMaxPool3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LPPool ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Base class for all (dimension-specialized) lppool modules.
template <size_t D, typename Derived>
class TORCH_API LPPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  LPPoolImpl(double norm_type, ExpandingArray<D> kernel_size)
      : LPPoolImpl(LPPoolOptions<D>(norm_type, kernel_size)) {}
  explicit LPPoolImpl(const LPPoolOptions<D>& options_);

  void reset() override;

  /// Pretty prints the `LPPool{1,2}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  LPPoolOptions<D> options;
};

/// Applies the LPPool1d function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LPPool1d to learn
/// about the exact behavior of this module.
class TORCH_API LPPool1dImpl : public LPPoolImpl<1, LPPool1dImpl> {
 public:
  using LPPoolImpl<1, LPPool1dImpl>::LPPoolImpl;

  Tensor forward(const Tensor& input);

};

TORCH_MODULE(LPPool1d);

/// Applies the LPPool2d function element-wise.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LPPool2d to learn
/// about the exact behavior of this module.
class TORCH_API LPPool2dImpl : public LPPoolImpl<2, LPPool2dImpl> {
 public:
  using LPPoolImpl<2, LPPool2dImpl>::LPPoolImpl;

  Tensor forward(const Tensor& input);

};

TORCH_MODULE(LPPool2d);

} // namespace nn
} // namespace torch
