#pragma once

#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/options/instancenorm.h>
#include <torch/nn/functional/instancenorm.h>

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) instanceNorm modules
template <size_t D, typename Derived, typename BatchNormDerived>
class TORCH_API InstanceNormImpl : public torch::nn::BatchNormImplBase<D, BatchNormDerived> {
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

class TORCH_API InstanceNorm1dImpl : public InstanceNormImpl<1, InstanceNorm1dImpl, BatchNorm1dImpl> {
 public:
  using InstanceNormImpl<1, InstanceNorm1dImpl, BatchNorm1dImpl>::InstanceNormImpl;
 private:
  void _check_input_dim(const Tensor& input) override;
};

TORCH_MODULE(InstanceNorm1d);

/*
class TORCH_API InstanceNorm2dImpl : public InstanceNormImpl<2, InstanceNorm2dImpl, BatchNorm2dImpl> {
 public:
  using InstanceNormImpl<2, InstanceNorm2dImpl, BatchNorm2dImpl>::InstanceNormImpl;
 private:
  void _check_input_dim(const Tensor& input) override;
};

TORCH_MODULE(InstanceNorm2d);

class TORCH_API InstanceNorm3dImpl : public InstanceNormImpl<3, InstanceNorm3dImpl, BatchNorm3dImpl> {
 public:
  using InstanceNormImpl<3, InstanceNorm3dImpl, BatchNorm3dImpl>::InstanceNormImpl;
 private:
  void _check_input_dim(const Tensor& input) override;
};

TORCH_MODULE(InstanceNorm3d);
*/
} // namespace nn
} // namespace torch
