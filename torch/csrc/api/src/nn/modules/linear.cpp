#include <torch/nn/modules/linear.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

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
  weight =
      register_parameter("weight", torch::empty({options.out(), options.in()}));
  if (options.with_bias()) {
    bias = register_parameter("bias", torch::empty(options.out()));
  }

  const auto stdv = 1.0 / std::sqrt(weight.size(1));
  NoGradGuard no_grad;
  for (auto& p : this->parameters()) {
    p.uniform_(-stdv, stdv);
  }
}

void LinearImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::Linear(in=" << options.in()
         << ", out=" << options.out() << ", with_bias=" << options.with_bias()
         << ")";
}

Tensor LinearImpl::forward(const Tensor& input) {
  AT_ASSERT(!options.with_bias() || bias.defined());
  return torch::linear(input, weight, bias);
}
} // namespace nn
} // namespace torch
