#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/batchnorm.h>
#include <torch/nn/init.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <cstdint>

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) batchnorm and instancenorm
/// modules.
template <size_t D, typename Derived, typename DerivedOptions>
class NormImplBase : public torch::nn::Cloneable<Derived> {
 protected:
  virtual void _check_input_dim(const Tensor& input) = 0;

 public:
  NormImplBase(const DerivedOptions& options_) : options(options_) {
    // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
    reset();
  }

  void reset() override {
    if (options.affine()) {
      weight = this->register_parameter(
          "weight", torch::empty({options.num_features()}));
      bias = this->register_parameter(
          "bias", torch::empty({options.num_features()}));
    } else {
      weight =
          this->register_parameter("weight", Tensor(), /*requires_grad=*/false);
      bias =
          this->register_parameter("bias", Tensor(), /*requires_grad=*/false);
    }
    if (options.track_running_stats()) {
      running_mean = this->register_buffer(
          "running_mean", torch::zeros({options.num_features()}));
      running_var = this->register_buffer(
          "running_var", torch::ones({options.num_features()}));
      num_batches_tracked = this->register_buffer(
          "num_batches_tracked", torch::tensor(0, torch::dtype(torch::kLong)));
    } else {
      running_mean = this->register_buffer("running_mean", Tensor());
      running_var = this->register_buffer("running_var", Tensor());
      num_batches_tracked =
          this->register_buffer("num_batches_tracked", Tensor());
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
  /// Only defined if the `track_running_stats` option was `true` upon
  /// construction.
  Tensor running_mean;

  /// The running variance.
  /// Only defined if the `track_running_stats` option was `true` upon
  /// construction.
  Tensor running_var;

  /// The number of the forward call.
  /// Only defined if the `track_running_stats` option was `true` upon
  /// construction.
  Tensor num_batches_tracked;
};

/// Base class for all (dimension-specialized) batchnorm modules.
template <size_t D, typename Derived>
class BatchNormImplBase : public NormImplBase<D, Derived, BatchNormOptions> {
 public:
  using NormImplBase<D, Derived, BatchNormOptions>::NormImplBase;

  Tensor forward(const Tensor& input) {
    this->_check_input_dim(input);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    double exponential_average_factor;
    if (this->options.momentum() == c10::nullopt) {
      exponential_average_factor = 0.0;
    } else {
      exponential_average_factor = this->options.momentum().value();
    }

    if (this->is_training() && this->options.track_running_stats()) {
      if (this->num_batches_tracked.defined()) {
        this->num_batches_tracked += 1;
        if (this->options.momentum() ==
            c10::nullopt) { // use cumulative moving average
          exponential_average_factor =
              1.0 / this->num_batches_tracked.template item<double>();
        } else { // use exponential moving average
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BatchNorm1d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the BatchNorm1d function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.BatchNorm1d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BatchNorm1dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// BatchNorm1d
/// model(BatchNorm1dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
class TORCH_API BatchNorm1dImpl : public BatchNormImplBase<1, BatchNorm1dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using BatchNormImplBase<1, BatchNorm1dImpl>::BatchNormImplBase;
};

/// A `ModuleHolder` subclass for `BatchNorm1dImpl`.
/// See the documentation for `BatchNorm1dImpl` class to learn what methods it
/// provides, and examples of how to use `BatchNorm1d` with
/// `torch::nn::BatchNorm1dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(BatchNorm1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BatchNorm2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the BatchNorm2d function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.BatchNorm2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BatchNorm2dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// BatchNorm2d
/// model(BatchNorm2dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
class TORCH_API BatchNorm2dImpl : public BatchNormImplBase<2, BatchNorm2dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using BatchNormImplBase<2, BatchNorm2dImpl>::BatchNormImplBase;
};

/// A `ModuleHolder` subclass for `BatchNorm2dImpl`.
/// See the documentation for `BatchNorm2dImpl` class to learn what methods it
/// provides, and examples of how to use `BatchNorm2d` with
/// `torch::nn::BatchNorm2dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(BatchNorm2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BatchNorm3d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the BatchNorm3d function.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.BatchNorm3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BatchNorm3dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// BatchNorm3d
/// model(BatchNorm3dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
class TORCH_API BatchNorm3dImpl : public BatchNormImplBase<3, BatchNorm3dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using BatchNormImplBase<3, BatchNorm3dImpl>::BatchNormImplBase;
};

/// A `ModuleHolder` subclass for `BatchNorm3dImpl`.
/// See the documentation for `BatchNorm3dImpl` class to learn what methods it
/// provides, and examples of how to use `BatchNorm3d` with
/// `torch::nn::BatchNorm3dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(BatchNorm3d);

} // namespace nn
} // namespace torch
