#include <torch/nn/modules/linear.h>
#include <torch/nn/functional/linear.h>
#include <torch/nn/init.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

void IdentityImpl::reset() {}

void IdentityImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Identity()";
}

Tensor IdentityImpl::forward(const Tensor& input) {
  return input;
}

// ============================================================================

LinearImpl::LinearImpl(const LinearOptions& options_) : options(options_) {
  reset();
}

void LinearImpl::reset() {
  weight = register_parameter("weight", 
    torch::empty({options.out_features(), options.in_features()}));
  if (options.bias()) {
    bias = register_parameter("bias", torch::empty(options.out_features()));
  } else {
    bias = register_parameter("bias", {});
  }

  torch::nn::init::kaiming_uniform_(weight, sqrt(5));
  if (bias.defined()) {
    int64_t fan_in, fan_out;
    std::tie(fan_in, fan_out) =
      torch::nn::init::calculate_fan_in_and_fan_out(weight);
    const auto bound = 1 / sqrt(fan_in);
    torch::nn::init::uniform_(bias, -bound, bound);
  }
}

void LinearImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha 
         << "torch::nn::Linear(in_features=" << options.in_features()
         << ", out_features=" << options.out_features() 
         << ", bias=" << options.bias() << ")";
}

Tensor LinearImpl::forward(const Tensor& input) {
  return F::linear(input, weight, bias);
}
} // namespace nn
} // namespace torch
