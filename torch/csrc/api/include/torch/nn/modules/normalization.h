#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/normalization.h>
#include <torch/nn/modules/_functions.h>
#include <torch/nn/options/normalization.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LayerNorm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Layer Normalization over a mini-batch of inputs as described in
/// the paper `Layer Normalization`_ .
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LayerNorm to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::LayerNormOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LayerNorm model(LayerNormOptions({2,
/// 2}).elementwise_affine(false).eps(2e-5));
/// ```
class TORCH_API LayerNormImpl : public torch::nn::Cloneable<LayerNormImpl> {
 public:
  LayerNormImpl(std::vector<int64_t> normalized_shape)
      : LayerNormImpl(LayerNormOptions(normalized_shape)) {}
  explicit LayerNormImpl(LayerNormOptions options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `LayerNorm` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Applies layer normalization over a mini-batch of inputs as described in
  /// the paper `Layer Normalization`_ .
  ///
  /// The mean and standard-deviation are calculated separately over the last
  /// certain number dimensions which have to be of the shape specified by
  /// input `normalized_shape`.
  ///
  /// `Layer Normalization`: https://arxiv.org/abs/1607.06450
  Tensor forward(const Tensor& input);

  /// The options with which this module was constructed.
  LayerNormOptions options;

  /// The learned weight.
  /// Initialized to ones if the `elementwise_affine` option is set to `true`
  /// upon construction.
  Tensor weight;

  /// The learned bias.
  /// Initialized to zeros `elementwise_affine` option is set to `true` upon
  /// construction.
  Tensor bias;
};

/// A `ModuleHolder` subclass for `LayerNormImpl`.
/// See the documentation for `LayerNormImpl` class to learn what methods it
/// provides, and examples of how to use `LayerNorm` with
/// `torch::nn::LayerNormOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(LayerNorm);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LocalResponseNorm
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies local response normalization over an input signal composed
/// of several input planes, where channels occupy the second dimension.
/// Applies normalization across channels.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.LocalResponseNorm to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::LocalResponseNormOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// LocalResponseNorm
/// model(LocalResponseNormOptions(2).alpha(0.0002).beta(0.85).k(2.));
/// ```
class TORCH_API LocalResponseNormImpl
    : public Cloneable<LocalResponseNormImpl> {
 public:
  LocalResponseNormImpl(int64_t size)
      : LocalResponseNormImpl(LocalResponseNormOptions(size)) {}
  explicit LocalResponseNormImpl(const LocalResponseNormOptions& options_);

  Tensor forward(const Tensor& input);

  void reset() override;

  /// Pretty prints the `LocalResponseNormImpl` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  LocalResponseNormOptions options;
};

/// A `ModuleHolder` subclass for `LocalResponseNormImpl`.
/// See the documentation for `LocalResponseNormImpl` class to learn what
/// methods it provides, and examples of how to use `LocalResponseNorm` with
/// `torch::nn::LocalResponseNormOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(LocalResponseNorm);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CrossMapLRN2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// See the documentation for `torch::nn::CrossMapLRN2dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// CrossMapLRN2d model(CrossMapLRN2dOptions(3).alpha(1e-5).beta(0.1).k(10));
/// ```
class TORCH_API CrossMapLRN2dImpl
    : public torch::nn::Cloneable<CrossMapLRN2dImpl> {
 public:
  CrossMapLRN2dImpl(int64_t size)
      : CrossMapLRN2dImpl(CrossMapLRN2dOptions(size)) {}
  explicit CrossMapLRN2dImpl(const CrossMapLRN2dOptions& options_)
      : options(options_) {}

  void reset() override;

  /// Pretty prints the `CrossMapLRN2d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  torch::Tensor forward(const torch::Tensor& input);

  CrossMapLRN2dOptions options;
};

/// A `ModuleHolder` subclass for `CrossMapLRN2dImpl`.
/// See the documentation for `CrossMapLRN2dImpl` class to learn what methods it
/// provides, and examples of how to use `CrossMapLRN2d` with
/// `torch::nn::CrossMapLRN2dOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(CrossMapLRN2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GroupNorm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies Group Normalization over a mini-batch of inputs as described in
/// the paper `Group Normalization`_ .
/// See https://pytorch.org/docs/master/nn.html#torch.nn.GroupNorm to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::GroupNormOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// GroupNorm model(GroupNormOptions(2, 2).eps(2e-5).affine(false));
/// ```
class TORCH_API GroupNormImpl : public torch::nn::Cloneable<GroupNormImpl> {
 public:
  GroupNormImpl(int64_t num_groups, int64_t num_channels)
      : GroupNormImpl(GroupNormOptions(num_groups, num_channels)) {}
  explicit GroupNormImpl(const GroupNormOptions& options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `GroupNorm` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);

  /// The options with which this module was constructed.
  GroupNormOptions options;

  /// The learned weight.
  Tensor weight;

  /// The learned bias.
  Tensor bias;
};

/// A `ModuleHolder` subclass for `GroupNormImpl`.
/// See the documentation for `GroupNormImpl` class to learn what methods it
/// provides, and examples of how to use `GroupNorm` with
/// `torch::nn::GroupNormOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(GroupNorm);

} // namespace nn
} // namespace torch
