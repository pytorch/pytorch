#pragma once

#include <torch/nn/functional/instancenorm.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/options/instancenorm.h>

namespace torch::nn {

/// Base class for all (dimension-specialized) instance norm modules
template <size_t D, typename Derived>
class InstanceNormImpl
    : public torch::nn::NormImplBase<D, Derived, InstanceNormOptions> {
 private:
  inline Tensor apply_instance_norm(const Tensor& input) {
    // Convert input tensor to match the dtype of running stats and parameters
    auto input_dtype = input.dtype();
    
    // Ensure the running mean, variance, weight, and bias have the same dtype as input
    auto mean = this->running_mean.to(input_dtype);
    auto var = this->running_var.to(input_dtype);
    auto weight = this->weight.to(input_dtype);
    auto bias = this->bias.to(input_dtype);

    return torch::nn::functional::detail::instance_norm(
        input,
        mean,
        var,
        weight,
        bias,
        this->is_training() || !this->options.track_running_stats(),
        this->options.momentum(),
        this->options.eps());
  }

  inline Tensor handle_no_batch_input(const Tensor& input) {
    return this->apply_instance_norm(input.unsqueeze(0)).squeeze(0);
  }

 public:
  using torch::nn::NormImplBase<D, Derived, InstanceNormOptions>::NormImplBase;

  Tensor forward(const Tensor& input) {
    this->_check_input_dim(input);

    // Check if input does not have a batch-dim
    if (input.dim() == D + 1) {
      return this->handle_no_batch_input(input);
    }

    return this->apply_instance_norm(input);
  }

  void pretty_print(std::ostream& stream) const override {
    stream << std::boolalpha << "torch::nn::InstanceNorm" << D << "d("
           << this->options.num_features() << ", "
           << "eps=" << this->options.eps() << ", "
           << "momentum=" << this->options.momentum() << ", "
           << "affine=" << this->options.affine() << ", "
           << "track_running_stats=" << this->options.track_running_stats()
           << ")";
  }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ InstanceNorm1d
class TORCH_API InstanceNorm1dImpl
    : public InstanceNormImpl<1, InstanceNorm1dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using InstanceNormImpl<1, InstanceNorm1dImpl>::InstanceNormImpl;
};

TORCH_MODULE(InstanceNorm1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ InstanceNorm2d
class TORCH_API InstanceNorm2dImpl
    : public InstanceNormImpl<2, InstanceNorm2dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using InstanceNormImpl<2, InstanceNorm2dImpl>::InstanceNormImpl;
};

TORCH_MODULE(InstanceNorm2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ InstanceNorm3d
class TORCH_API InstanceNorm3dImpl
    : public InstanceNormImpl<3, InstanceNorm3dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using InstanceNormImpl<3, InstanceNorm3dImpl>::InstanceNormImpl;
};

TORCH_MODULE(InstanceNorm3d);

} // namespace torch::nn

