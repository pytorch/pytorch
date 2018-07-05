#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <cstdint>

namespace torch {
namespace nn {
struct BatchNormOptions {
  /* implicit */ BatchNormOptions(int64_t features);
  TORCH_ARG(int64_t, features);
  TORCH_ARG(bool, affine) = true;
  TORCH_ARG(bool, stateful) = false;
  TORCH_ARG(double, eps) = 1e-5;
  TORCH_ARG(double, momentum) = 0.1;
};

namespace detail {
template <typename Derived>
class BatchNormImplBase : public torch::nn::Cloneable<Derived> {
 public:
  explicit BatchNormImplBase(BatchNormOptions options);

  void reset() override;

  Tensor forward(Tensor input);
  Tensor pure_forward(Tensor input, Tensor mean, Tensor variance);

  BatchNormOptions options;
  Tensor weight;
  Tensor bias;
  Tensor running_mean;
  Tensor running_variance;

 protected:
  virtual void check_input_dimensions(Tensor input) const = 0;
};
} // namespace detail

// Our usual convention is that the name of the option is <class-name> +
// "Options", so we define these aliases.

using BatchNorm1dOptions = BatchNormOptions;
class BatchNorm1dImpl : public detail::BatchNormImplBase<BatchNorm1dImpl> {
 public:
  using detail::BatchNormImplBase<BatchNorm1dImpl>::BatchNormImplBase;

 private:
  virtual void check_input_dimensions(Tensor input) const override;
};
TORCH_MODULE(BatchNorm1d);

using BatchNorm2dOptions = BatchNormOptions;
class BatchNorm2dImpl : public detail::BatchNormImplBase<BatchNorm2dImpl> {
 public:
  using detail::BatchNormImplBase<BatchNorm2dImpl>::BatchNormImplBase;

 private:
  virtual void check_input_dimensions(Tensor input) const override;
};
TORCH_MODULE(BatchNorm2d);

using BatchNorm3dOptions = BatchNormOptions;
class BatchNorm3dImpl : public detail::BatchNormImplBase<BatchNorm3dImpl> {
 public:
  using detail::BatchNormImplBase<BatchNorm3dImpl>::BatchNormImplBase;

 private:
  virtual void check_input_dimensions(Tensor input) const override;
};
TORCH_MODULE(BatchNorm3d);

} // namespace nn
} // namespace torch
