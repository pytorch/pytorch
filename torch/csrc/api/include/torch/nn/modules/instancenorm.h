#pragma once

#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/options/instancenorm.h>
#include <torch/nn/functional/instancenorm.h>

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) instanceNorm modules
template <typename Derived>
class TORCH_API InstanceNormImpl : public torch::nn::BatchNormImpl {
 protected:
  virtual void _check_input_dim(const Tensor& input) = 0;

 public:
  InstanceNormImpl(int64_t num_features)
    : InstanceNormImpl(InstanceNormOptions(num_features)) {} 
  explicit InstanceNormImpl(const InstanceNormOptions& options_);

  Tensor forward(const Tensor& input);
  
  /// The options with which this `Module` was constructed.
  InstanceNormOptions options;
};

class TORCH_API InstanceNorm1dImpl : public InstanceNormImpl<InstanceNorm1dImpl> {
 public:
  using InstanceNormImpl<InstanceNorm1dImpl>::InstanceNormImpl;
 private:
  void _check_input_dim(const Tensor& input) override;
};

TORCH_MODULE(InstanceNorm1d);

class TORCH_API InstanceNorm2dImpl : public InstanceNormImpl<InstanceNorm2dImpl> {
 public:
  using InstanceNormImpl<InstanceNorm2dImpl>::InstanceNormImpl;
 private:
  void _check_input_dim(const Tensor& input) override;
};

TORCH_MODULE(InstanceNorm2d);

class TORCH_API InstanceNorm3dImpl : public InstanceNormImpl<InstanceNorm3dImpl> {
 public:
  using InstanceNormImpl<InstanceNorm3dImpl>::InstanceNormImpl;
 private:
  void _check_input_dim(const Tensor& input) override;
};

TORCH_MODULE(InstanceNorm3d);

} // namespace nn
} // namespace torch
