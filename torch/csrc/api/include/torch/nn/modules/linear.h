#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/linear.h>
#include <torch/nn/module.h>
#include <torch/nn/options/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Identity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A placeholder identity operator that is argument-insensitive.
/// See https://pytorch.org/docs/master/generated/torch.nn.Identity.html to
/// learn about the exact behavior of this module.
class TORCH_API IdentityImpl : public Cloneable<IdentityImpl> {
 public:
  void reset() override;

  /// Pretty prints the `Identity` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  Tensor forward(const Tensor& input);
};

/// A `ModuleHolder` subclass for `IdentityImpl`.
/// See the documentation for `IdentityImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Identity);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Linear ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies a linear transformation with optional bias.
/// See https://pytorch.org/docs/master/generated/torch.nn.Linear.html to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::LinearOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Linear model(LinearOptions(5, 2).bias(false));
/// ```
class TORCH_API LinearImpl : public Cloneable<LinearImpl> {
 public:
  LinearImpl(int64_t in_features, int64_t out_features)
      : LinearImpl(LinearOptions(in_features, out_features)) {}
  explicit LinearImpl(const LinearOptions& options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `Linear` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Transforms the `input` tensor by multiplying with the `weight` and
  /// optionally adding the `bias`, if `with_bias` is true in the options.
  Tensor forward(const Tensor& input);

  /// The options used to configure this module.
  LinearOptions options;

  /// The learned weight.
  Tensor weight;

  /// The learned bias. If `bias` is false in the `options`, this tensor is
  /// undefined.
  Tensor bias;
};

/// A `ModuleHolder` subclass for `LinearImpl`.
/// See the documentation for `LinearImpl` class to learn what methods it
/// provides, and examples of how to use `Linear` with
/// `torch::nn::LinearOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Linear);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Flatten ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A placeholder for Flatten operator
/// See https://pytorch.org/docs/master/generated/torch.nn.Flatten.html to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::FlattenOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Flatten model(FlattenOptions().start_dim(2).end_dim(4));
/// ```
class TORCH_API FlattenImpl : public Cloneable<FlattenImpl> {
 public:
  explicit FlattenImpl(const FlattenOptions& options_ = {});

  void reset() override;

  /// Pretty prints the `Flatten` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Applies a flatten transform on the `input`.
  Tensor forward(const Tensor& input);

  /// The options used to configure this module.
  FlattenOptions options;
};

/// A `ModuleHolder` subclass for `FlattenImpl`.
/// See the documentation for `FlattenImpl` class to learn what methods it
/// provides, and examples of how to use `Flatten` with
/// `torch::nn::FlattenOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Flatten);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Unflatten
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A placeholder for unflatten operator
/// See https://pytorch.org/docs/master/generated/torch.nn.Unflatten.html to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::UnflattenOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Unflatten model(UnflattenOptions(0, {2, 2}));
/// Unflatten model(UnflattenOptions("B", {{"B1", 2}, {"B2", 2}}));
/// ```
class TORCH_API UnflattenImpl : public Cloneable<UnflattenImpl> {
 public:
  UnflattenImpl(int64_t dim, std::vector<int64_t> sizes)
      : UnflattenImpl(UnflattenOptions(dim, sizes)) {}
  UnflattenImpl(std::string dimname, UnflattenOptions::namedshape_t namedshape)
      : UnflattenImpl(UnflattenOptions(dimname, namedshape)) {}
  explicit UnflattenImpl(UnflattenOptions options_);

  void reset() override;

  /// Pretty prints the `Unflatten` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Applies an unflatten transform on the `input`.
  Tensor forward(const Tensor& input);

  /// The options used to configure this module.
  UnflattenOptions options;
};

/// A `ModuleHolder` subclass for `UnflattenImpl`.
/// See the documentation for `UnflattenImpl` class to learn what methods it
/// provides, and examples of how to use `Unflatten` with
/// `torch::nn::UnflattenOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Unflatten);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Bilinear ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies a billinear transformation with optional bias.
/// See https://pytorch.org/docs/master/generated/torch.nn.Bilinear.html to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BilinearOptions` class to learn what
/// constructor arguments are supported for this module.
///
/// Example:
/// ```
/// Bilinear model(BilinearOptions(3, 2, 4).bias(false));
/// ```
class TORCH_API BilinearImpl : public Cloneable<BilinearImpl> {
 public:
  BilinearImpl(int64_t in1_features, int64_t in2_features, int64_t out_features)
      : BilinearImpl(
            BilinearOptions(in1_features, in2_features, out_features)) {}
  explicit BilinearImpl(const BilinearOptions& options_);

  void reset() override;

  void reset_parameters();

  /// Pretty prints the `Bilinear` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Applies a bilinear transform on the `input1` and `input2` tensor by
  /// multiplying with the `weight` and optionally adding the `bias`, if
  /// `with_bias` is true in the options.
  Tensor forward(const Tensor& input1, const Tensor& input2);

  /// The options used to configure this module.
  BilinearOptions options;

  /// The learned weight.
  Tensor weight;

  /// The learned bias. If `with_bias` is false in the `options`, this tensor is
  /// undefined.
  Tensor bias;
};

/// A `ModuleHolder` subclass for `BilinearImpl`.
/// See the documentation for `BilinearImpl` class to learn what methods it
/// provides, and examples of how to use `Bilinear` with
/// `torch::nn::BilinearOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(Bilinear);

} // namespace nn
} // namespace torch
