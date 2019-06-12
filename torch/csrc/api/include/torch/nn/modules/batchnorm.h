#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>

namespace torch {
namespace nn {

/// Options for the `BatchNorm` module.
struct TORCH_API BatchNormOptions {
  /* implicit */ BatchNormOptions(int64_t features);
  /// The number of features of the input tensor.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(int64_t, features);
  /// Whether to learn a scale and bias that are applied in an affine
  /// transformation on the input.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, affine) = true;
  /// Whether to store and update batch statistics (mean and variance) in the
  /// module. If `false`, you should call `pure_forward` and supply those batch
  /// statistics yourself.
  /// Changing this parameter after construction __has no effect__.
  TORCH_ARG(bool, stateful) = true;
  /// The epsilon value added for numerical stability.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, eps) = 1e-5;
  /// A momentum multiplier for the mean and variance.
  /// Changing this parameter after construction __is effective__.
  TORCH_ARG(double, momentum) = 0.1;
};

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

  /// Applies batch normalization on the `input` using the stored mean and
  /// variance.
  ///
  /// The module must be constructed with `stateful = true` when calling this
  /// method, as the module will otherwise not store running statistics. If you
  /// want to supply the mean and variance yourself, use `pure_forward`.
  Tensor forward(Tensor input);

  /// Applies batch normalization on the `input` using the given `mean` and
  /// `variance` statistics.
  Tensor pure_forward(Tensor input, Tensor mean, Tensor variance);

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
  Tensor running_variance;
};

/// A `ModuleHolder` subclass for `BatchNormImpl`.
/// See the documentation for `BatchNormImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(BatchNorm);

} // namespace nn
} // namespace torch
