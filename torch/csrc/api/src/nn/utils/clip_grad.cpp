#include <torch/nn/utils/clip_grad.h>
#include <cmath>
#include <limits>
#include <memory>

namespace torch {
namespace nn {

float clip_grad_norm_(
    std::vector<Tensor>& parameters,
    float max_norm,
    float norm_type) {
  std::vector<std::unique_ptr<Tensor>> params_with_grad;

  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      params_with_grad.push_back(torch::make_unique<Tensor>(param));
    }
  }
  double total_norm = 0.0;
  double inf = std::numeric_limits<double>::infinity();
  if (norm_type == inf) {
    for (const auto& param : params_with_grad) {
      auto param_max = param->grad().abs().max().item().toDouble();
      if (param_max > total_norm) {
        total_norm = param_max;
      }
    }
  } else {
    for (const auto& param : params_with_grad) {
      auto param_norm = torch::norm(param->grad(), norm_type);
      total_norm += torch::pow(param_norm, norm_type).item().toDouble();
    }
    total_norm = std::pow(total_norm, 1.0 / norm_type);
  }
  auto clip_coef = max_norm / (total_norm + 1e-6);
  if (clip_coef < 1) {
    for (auto& param : params_with_grad) {
      param->grad().mul_(clip_coef);
    }
  }
  return total_norm;
}

} // namespace nn
} // namespace torch
