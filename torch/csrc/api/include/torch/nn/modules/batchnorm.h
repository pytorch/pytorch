#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/batchnorm.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/init.h>
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
  explicit BatchNormImpl(int64_t num_features)
      : BatchNormImpl(BatchNormOptions(num_features)) {}
  explicit BatchNormImpl(const BatchNormOptions& options_);

  void reset() override;

  /// Pretty prints the `BatchNorm` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Applies batch normalization on the `input` using the stored mean and
  /// variance.
  ///
  /// The module must be constructed with `track_running_stats = true` when calling this
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
  /// Only defined if the `track_running_stats` option was `true` upon construction.
  Tensor running_mean;

  /// The running variance.
  /// Only defined if the `track_running_stats` option was `true` upon construction.
  Tensor running_var;
};

/// A `ModuleHolder` subclass for `BatchNormImpl`.
/// See the documentation for `BatchNormImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(BatchNorm);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BatchNorm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Base class for all (dimension-specialized) batchnorm and instancenorm modules.
template <size_t D, typename Derived, typename DerivedOptions>
class NormImplBase : public torch::nn::Cloneable<Derived> {
 protected:
  virtual void _check_input_dim(const Tensor& input) = 0;

 public:
  NormImplBase(const DerivedOptions& options_) : options(options_) {
    reset();
  }

  void reset() override {
    if (options.affine()) {
      weight = this->register_parameter("weight", torch::empty({options.num_features()}));
      bias = this->register_parameter("bias", torch::empty({options.num_features()}));
    } else {
      weight = this->register_parameter("weight", Tensor());
      bias = this->register_parameter("bias", Tensor());
    }
    if (options.track_running_stats()) {
      running_mean = this->register_buffer("running_mean", torch::zeros({options.num_features()}));
      running_var = this->register_buffer("running_var", torch::ones({options.num_features()}));
      num_batches_tracked = this->register_buffer("num_batches_tracked", torch::tensor(0, torch::dtype(torch::kLong)));
    } else {
      running_mean = this->register_buffer("running_mean", Tensor());
      running_var = this->register_buffer("running_var", Tensor());
      num_batches_tracked = this->register_buffer("num_batches_tracked", Tensor());
    }
    reset_parameters();
  }

  void reset_running_stats() {
    if (options.track_running_stats()) {
      running_mean.zero_();
      running_var.fill_(1);
      num_batches_tracked.zero_();
    }
  }

  void reset_parameters() {
    reset_running_stats();
    if (options.affine()) {
      torch::nn::init::ones_(weight);
      torch::nn::init::zeros_(bias);
    }
  }

  /// The options with which this module was constructed.
  DerivedOptions options;

  /// The learned weight.
  /// Only defined if the `affine` option was `true` upon construction.
  Tensor weight;

  /// The learned bias.
  /// Only defined if the `affine` option was `true` upon construction.
  Tensor bias;

  /// The running mean.
  /// Only defined if the `track_running_stats` option was `true` upon construction.
  Tensor running_mean;

  /// The running variance.
  /// Only defined if the `track_running_stats` option was `true` upon construction.
  Tensor running_var;

  /// The number of the forward call.
  /// Only defined if the `track_running_stats` option was `true` upon construction.
  Tensor num_batches_tracked;
};

/// Base class for all (dimension-specialized) batchnorm modules.
template <size_t D, typename Derived>
class BatchNormImplBase : public NormImplBase<D, Derived, BatchNormOptions> {
 public:
  using NormImplBase<D, Derived, BatchNormOptions>::NormImplBase;

  Tensor forward(const Tensor& input) {
    this->_check_input_dim(input);
    double exponential_average_factor;
    if (this->options.momentum() == c10::nullopt) {
      exponential_average_factor = 0.0;
    } else {
      exponential_average_factor = this->options.momentum().value();
    }

    if (this->is_training() && this->options.track_running_stats()) {
      if (this->num_batches_tracked.defined()) {
        this->num_batches_tracked += 1;
        if (this->options.momentum() == c10::nullopt) {  // use cumulative moving average
          exponential_average_factor = 1.0 / this->num_batches_tracked.template item<double>();
        } else {  // use exponential moving average
          exponential_average_factor = this->options.momentum().value();
        }
      }
    }

    return torch::nn::functional::detail::batch_norm(
        input,
        this->running_mean,
        this->running_var,
        this->weight,
        this->bias,
        this->is_training() || !this->options.track_running_stats(),
        /*momentum=*/exponential_average_factor,
        this->options.eps());
  }

  /// Pretty prints the `BatchNorm{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;
};

/// Applies the BatchNorm1d function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.BatchNorm1d to learn
/// about the exact behavior of this module.
class TORCH_API BatchNorm1dImpl : public BatchNormImplBase<1, BatchNorm1dImpl> {
 protected:
  virtual void _check_input_dim(const Tensor& input) override;

 public:
  using BatchNormImplBase<1, BatchNorm1dImpl>::BatchNormImplBase;
};

TORCH_MODULE(BatchNorm1d);

/// Applies the BatchNorm2d function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.BatchNorm2d to learn
/// about the exact behavior of this module.
class TORCH_API BatchNorm2dImpl : public BatchNormImplBase<2, BatchNorm2dImpl> {
 protected:
  virtual void _check_input_dim(const Tensor& input) override;

 public:
  using BatchNormImplBase<2, BatchNorm2dImpl>::BatchNormImplBase;
};

TORCH_MODULE(BatchNorm2d);

/// Applies the BatchNorm3d function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.BatchNorm3d to learn
/// about the exact behavior of this module.
class TORCH_API BatchNorm3dImpl : public BatchNormImplBase<3, BatchNorm3dImpl> {
 protected:
  virtual void _check_input_dim(const Tensor& input) override;

 public:
  using BatchNormImplBase<3, BatchNorm3dImpl>::BatchNormImplBase;
};

TORCH_MODULE(BatchNorm3d);

} // namespace nn
} // namespace torch
