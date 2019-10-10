#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/options/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {

/// A placeholder identity operator that is argument-insensitive.
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

// ============================================================================

/// Applies a linear transformation with optional bias.
class TORCH_API LinearImpl : public Cloneable<LinearImpl> {
 public:
  LinearImpl(int64_t in, int64_t out) : LinearImpl(LinearOptions(in, out)) {}
  explicit LinearImpl(const LinearOptions& options_);

  void reset() override;

  /// Pretty prints the `Linear` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Transforms the `input` tensor by multiplying with the `weight` and
  /// optionally adding the `bias`, if `with_bias` is true in the options.
  Tensor forward(const Tensor& input);

  /// The options used to configure this module.
  LinearOptions options;

  /// The learned weight.
  Tensor weight;

  /// The learned bias. If `with_bias` is false in the `options`, this tensor is
  /// undefined.
  Tensor bias;
};

/// A `ModuleHolder` subclass for `LinearImpl`.
/// See the documentation for `LinearImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Linear);

/// Options for the `Bilinear` module.
struct TORCH_API BilinearOptions {
  BilinearOptions(int64_t in1, int64_t in2, int64_t out);
  /// The number of features in input 1 (columns of the input1 matrix).
  TORCH_ARG(int64_t, in1);
  /// The number of features in input 2 (columns of the input2 matrix).
  TORCH_ARG(int64_t, in2);
  /// The number of output features to produce (columns of the output matrix).
  TORCH_ARG(int64_t, out);
  /// Whether to learn and add a bias after the bilinear transformation.
  TORCH_ARG(bool, with_bias) = true;
};

/// Applies a billinear transformation with optional bias.
class TORCH_API BilinearImpl : public Cloneable<BilinearImpl> {
 public:
  BilinearImpl(int64_t in1, int64_t in2, int64_t out) : BilinearImpl(BilinearOptions(in1, in2, out)) {}
  explicit BilinearImpl(BilinearOptions options);

  void reset() override;

  /// Pretty prints the `Bilinear` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Applies a bilinear transform on the `input1` and `input2` tensor by multiplying 
  /// with the `weight` and optionally adding the `bias`, if `with_bias` 
  /// is true in the options.
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
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Bilinear);

} // namespace nn
} // namespace torch
