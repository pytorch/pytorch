#pragma once

namespace torch {
namespace nn {
namespace functional {

inline Tensor instance_norm(const Tensor& input, const Tensor& running_mean,
    const Tensor& running_var, const Tensor& weight, const Tensor& bias, 
    bool use_input_stats, double momentum, double eps) {
  
  return torch::instance_norm(input, weight, bias, running_mean, 
      running_var, use_input_stats, momentum, eps, 
      at::globalContext().userEnabledCuDNN()); 
}
} // namespace functional
} // namespace nn
} // namespace torch
