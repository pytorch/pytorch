#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {
/// Options for the `Linear` module.
struct TORCH_API LinearOptions {
  LinearOptions(int64_t in, int64_t out);
  /// The number of input features (columns of the input matrix).
  TORCH_ARG(int64_t, in);
  /// The number of output features to produce (columns of the output matrix).
  TORCH_ARG(int64_t, out);
  /// Whether to learn and add a bias after the linear transformation.
  TORCH_ARG(bool, with_bias) = true;
};

/// Applies a linear transformation with optional bias.
class TORCH_API LinearImpl : public Cloneable<LinearImpl> {
 public:
  LinearImpl(int64_t in, int64_t out) : LinearImpl(LinearOptions(in, out)) {}
  explicit LinearImpl(LinearOptions options);

  void reset() override;

  /// Transforms the `input` tensor by multiplying with the `weight` and
  /// optionally adding the `bias`, if `with_bias` is true in the options.
  Tensor forward(Tensor input);

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

} // namespace nn
} // namespace torch
