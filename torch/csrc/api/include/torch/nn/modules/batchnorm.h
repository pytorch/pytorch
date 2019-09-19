#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>

namespace torch {
namespace nn {

/// Applies [Batch Normalization](https://arxiv.org/abs/1502.03167) to an input.
///
/// Refer to the documentation for
/// [`BatchNorm1d`](https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm1d)
/// in PyTorch to learn more about the exact semantics of this module, __but see
/// the note below regarding differences between the Python and C++ API__.
///
/// \rst
/// .. attention::
///   In the Python API, there are separate implementations for 1-D, 2-D and 3-D
///   BatchNorm. In C++, there is only one `BatchNorm` module, which works for
///   any of these dimensions.
/// \endrst
class TORCH_API BatchNormImpl : public torch::nn::Cloneable<BatchNormImpl> {
 public:
  explicit BatchNormImpl(int64_t features)
      : BatchNormImpl(BatchNormOptions(features)) {}
  explicit BatchNormImpl(BatchNormOptions options);

  void reset() override;

  /// Pretty prints the `BatchNorm` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Applies batch normalization on the `input` using the stored mean and
  /// variance.
  ///
  /// The module must be constructed with `stateful = true` when calling this
  /// method, as the module will otherwise not store running statistics. If you
  /// want to supply the mean and variance yourself, use `pure_forward`.
  Tensor forward(const Tensor& input);

  /// Applies batch normalization on the `input` using the given `mean` and
  /// `variance` statistics.
  Tensor pure_forward(
      const Tensor& input,
      const Tensor& mean,
      const Tensor& variance);

  /// The options with which this module was constructed.
  BatchNormOptions options;

  /// The learned weight.
  /// Only defined if the `affine` option was `true` upon construction.
  Tensor weight;

  /// The learned bias.
  /// Only defined if the `affine` option was `true` upon construction.
  Tensor bias;

  /// The running mean.
  /// Only defined if the `stateful` option was `true` upon construction.
  Tensor running_mean;

  /// The running variance.
  /// Only defined if the `stateful` option was `true` upon construction.
  Tensor running_var;
};

/// A `ModuleHolder` subclass for `BatchNormImpl`.
/// See the documentation for `BatchNormImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(BatchNorm);

} // namespace nn
} // namespace torch
